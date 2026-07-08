"""Transport recovery utilities for the LLM gateway.

When the upstream LLM endpoint is briefly unreachable (corporate firewall
blip, NAT timeout, provider 5xx storm), httpx's internal connection pool
can be left with TCP sockets the remote has RST'd. httpx doesn't always
detect the dead connections — new requests get assigned to zombie slots
and hang forever. Even after the network recovers, the orchestrator
can't make progress because every retry inherits the same stuck pool.

This module provides two reusable pieces:

  * ``SwappableAsyncClient`` — a transparent proxy around
    ``httpx.AsyncClient`` whose underlying client can be swapped at
    runtime. Existing code paths that hold a reference to the proxy
    transparently route through whichever client is current.

  * ``transport_recovery_loop`` — a background coroutine that watches an
    ``AdaptiveLLMRuntime`` for prolonged ``'open'`` state and, when the
    threshold is exceeded, atomically swaps the underlying client AND
    cancels in-flight tasks so the orchestrator's dispatch loop can
    relaunch them with the fresh client.

Order matters in the swap: build + install the new client BEFORE
cancelling old tasks. Cancellation forces the orchestrator's
``asyncio.wait`` to return, which lets new files dispatch immediately;
if those new dispatches happen before the swap is visible they would
inherit the zombie client.
"""

from __future__ import annotations

import asyncio
import time
from typing import Any, Callable, Iterable

import httpx

__all__ = ["SwappableAsyncClient", "transport_recovery_loop"]


class SwappableAsyncClient:
    """Transparent proxy over ``httpx.AsyncClient`` that supports hot-swap.

    Use it like a normal AsyncClient — ``async with SwappableAsyncClient(factory) as client:``.
    All attribute access (``client.post``, ``client._transport``, etc.) is
    forwarded to the current underlying instance. Call ``await client.swap()``
    to replace the underlying instance with a fresh one. In-flight requests
    on the old client are left to either complete or fail on their own; new
    requests immediately use the fresh client.
    """

    def __init__(self, factory: Callable[[], httpx.AsyncClient]):
        self._factory = factory
        self._client: httpx.AsyncClient | None = None
        # ``_swap_lock`` is created lazily inside __aenter__ so the proxy
        # can be constructed before an event loop exists.
        self._swap_lock: asyncio.Lock | None = None

    async def __aenter__(self) -> "SwappableAsyncClient":
        self._swap_lock = asyncio.Lock()
        self._client = self._factory()
        await self._client.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        if self._client is not None:
            try:
                await self._client.__aexit__(exc_type, exc, tb)
            except Exception:
                pass
            self._client = None

    async def swap(self) -> tuple[int, int]:
        """Replace the underlying client with a fresh instance.

        Returns ``(old_id, new_id)`` so callers can verify the swap
        actually changed objects (useful for tests).
        """
        if self._swap_lock is None:
            self._swap_lock = asyncio.Lock()
        async with self._swap_lock:
            old = self._client
            new = self._factory()
            await new.__aenter__()
            self._client = new
            if old is not None:
                try:
                    await old.__aexit__(None, None, None)
                except Exception:
                    pass
            return (id(old) if old is not None else 0, id(new))

    @property
    def underlying(self) -> httpx.AsyncClient | None:
        """Direct access to the current underlying client (for tests)."""
        return self._client

    def __getattr__(self, name: str) -> Any:
        # Triggered only when normal attribute lookup misses on self.
        # Forward to the underlying client.
        client = object.__getattribute__(self, "_client")
        if client is None:
            raise AttributeError(
                f"SwappableAsyncClient not entered yet; cannot access {name!r}"
            )
        return getattr(client, name)


async def transport_recovery_loop(
    *,
    runtime: Any,
    swappable: SwappableAsyncClient,
    threshold_seconds: int,
    cancel_in_flight: Callable[[], Iterable[asyncio.Task]] | None = None,
    on_after_swap: Callable[[SwappableAsyncClient], None] | None = None,
    pprint: Callable[[str], None] = print,
    label: str = "Pass 2",
) -> None:
    """Watch ``runtime.state`` and reset transport when stuck in ``'open'``.

    Polls every ``max(min(threshold_seconds // 4, 30), 1)`` seconds. When
    the runtime has been in ``'open'`` state for at least
    ``threshold_seconds`` continuously, this coroutine:

      1. Calls ``swappable.swap()`` to install a fresh httpx client.
      2. Calls ``cancel_in_flight()`` (if provided) and cancels every
         returned task so they relinquish their grip on the old pool.
      3. Pushes ``runtime`` out of ``'open'`` via ``_enter_probing`` so
         the gate doesn't stay at the degraded floor with fresh sockets.
      4. Calls ``on_after_swap(swappable)`` so callers can update any
         registered forensics state.

    The order is critical: swap MUST happen before cancellation, because
    cancellation triggers the orchestrator's dispatch loop to launch new
    work — and if the swap hasn't landed yet those new tasks inherit the
    zombie client and the recovery accomplishes nothing.

    The coroutine returns when cancelled (orchestrator shutdown). It is
    safe to leave running for the full lifetime of a Pass.
    """
    if threshold_seconds <= 0:
        return
    open_since: float | None = None
    poll = max(min(threshold_seconds // 4, 30), 1)
    while True:
        try:
            await asyncio.sleep(poll)
        except asyncio.CancelledError:
            return
        state = getattr(runtime, "state", None) if runtime is not None else None
        if state != "open":
            open_since = None
            continue
        now = time.monotonic()
        if open_since is None:
            open_since = now
            continue
        if now - open_since < threshold_seconds:
            continue
        # Threshold exceeded — do the swap.
        try:
            old_id, new_id = await swappable.swap()
        except Exception as exc:
            try:
                pprint(f"[red]{label}: transport reset failed to build new client: {exc!r}[/red]")
            except Exception:
                pass
            open_since = None
            continue
        # After swap is visible, ask the orchestrator to cancel its
        # in-flight tasks so they release the old pool and the dispatch
        # loop can relaunch them on the new client.
        cancelled_count = 0
        if cancel_in_flight is not None:
            try:
                tasks_iterable = cancel_in_flight() or []
                tasks = list(tasks_iterable)
                cancelled_count = len(tasks)
                for t in tasks:
                    t.cancel()
                if tasks:
                    try:
                        await asyncio.wait(tasks, timeout=10.0)
                    except Exception:
                        pass
            except Exception as exc:
                try:
                    pprint(f"[red]{label}: cancel-in-flight callback failed: {exc!r}[/red]")
                except Exception:
                    pass
        # Push runtime out of open state.
        try:
            if runtime is not None and getattr(runtime, "state", None) == "open":
                if hasattr(runtime, "_enter_probing"):
                    runtime._enter_probing(reason=f"{label}_transport_reset")
        except Exception:
            pass
        if on_after_swap is not None:
            try:
                on_after_swap(swappable)
            except Exception:
                pass
        try:
            pprint(
                f"[yellow]{label}: transport reset after {int(now - open_since)}s runtime open — "
                f"rebuilt httpx client, runtime → probing. Cancelled {cancelled_count} stuck task(s).[/yellow]"
            )
        except Exception:
            pass
        open_since = None
