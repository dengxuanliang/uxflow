// Trajectory Inspector frontend (spec §9, decisions 10-12).
const $ = (id) => document.getElementById(id);

const STAGES = ["module0", "module1", "module2", "module3", "done"];

let state = {
  runId: null,
  view: null,
  activeProblem: null,   // problem id
  activeCap: null,       // focused capability label (click a capability to focus)
  activeTraj: null,      // {trajectory_id, slice_index}
  trajCache: {},
  // progress
  es: null,                  // active EventSource, so we can stop it
  startedAt: null,
  timer: null,
  compileFirstDoneAt: null,  // wall time when first complaint finished compiling
  compileTotal: 0,
  stopping: false,
};

$("run").addEventListener("click", startRun);
$("stop").addEventListener("click", stopRun);

// echo selected filenames into the slot chips
for (const [id, slot, empty] of [
  ["manifest", "slot-manifest", "点击选择文件…"],
  ["trajectories", "slot-trajectories", "点击选择文件…"],
]) {
  $(id).addEventListener("change", (e) => {
    const f = e.target.files[0];
    $("name-" + id).textContent = f ? f.name : empty;
    $(slot).classList.toggle("filled", !!f);
  });
}

// ── Run + progress ────────────────────────────────────────────────
async function startRun() {
  const mf = $("manifest").files[0];
  const tj = $("trajectories").files[0];
  if (!mf || !tj) { alert("请先上传清单和轨迹两个文件"); return; }
  const fd = new FormData();
  fd.append("manifest", mf);
  fd.append("trajectories", tj);

  state.stopping = false;
  state.runId = null;
  $("run").disabled = true;
  $("progress").classList.remove("hidden");
  $("workspace").classList.add("hidden");
  resetProgress();
  resetRunState();
  setStage("module0");
  setMsg("上传中…");
  $("stop").style.display = "";       // show stop while running
  startTimer();

  let run_id;
  try {
    const resp = await fetch("/runs", { method: "POST", body: fd });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    ({ run_id } = await resp.json());
  } catch (err) {
    setMsg("上传失败: " + (err.message || err));
    stopTimer();
    $("stop").style.display = "none";
    $("run").disabled = false;
    return;
  }

  // User may have clicked 停止 during the upload window (before runId existed).
  if (state.stopping) {
    // best-effort cancel the run we just created, then bail out of the UI flow
    try { await fetch(`/runs/${run_id}/cancel`, { method: "POST" }); } catch (_) {}
    return;
  }

  state.runId = run_id;
  subscribeEvents(run_id);
}

function resetRunState() {
  state.view = null;
  state.activeProblem = null;
  state.activeCap = null;
  state.activeTraj = null;
  state.trajCache = {};
}

function resetProgress() {
  state.compileFirstDoneAt = null;
  state.compileTotal = 0;
  $("prog-count").textContent = "";
  $("prog-eta").textContent = "";
  $("prog-elapsed").textContent = "⏱ 0:00";
  setBarSweep();
  for (const s of STAGES) stepNode(s).classList.remove("active", "done");
}

function stepNode(stage) {
  return document.querySelector(`.step-node[data-stage="${stage}"]`);
}

// mark given stage active, all prior stages done
function setStage(stage) {
  const idx = STAGES.indexOf(stage);
  STAGES.forEach((s, i) => {
    const n = stepNode(s);
    n.classList.toggle("done", i < idx);
    n.classList.toggle("active", i === idx);
  });
}

function setMsg(t) { $("progress-msg").textContent = t; }

// determinate fill (0..1); pass null for indeterminate sweep
function setBarFraction(f) {
  $("prog-bar").classList.remove("sweep");
  $("prog-fill").style.width = Math.round(f * 100) + "%";
}
function setBarSweep() {
  $("prog-bar").classList.add("sweep");
  $("prog-fill").style.width = "";
}

function startTimer() {
  state.startedAt = Date.now();
  clearInterval(state.timer);
  state.timer = setInterval(() => {
    $("prog-elapsed").textContent = "⏱ " + fmtDur((Date.now() - state.startedAt) / 1000);
  }, 500);
}
function stopTimer() { clearInterval(state.timer); state.timer = null; }

function fmtDur(sec) {
  sec = Math.max(0, Math.floor(sec));
  const m = Math.floor(sec / 60), s = sec % 60;
  return `${m}:${String(s).padStart(2, "0")}`;
}

function subscribeEvents(runId) {
  if (state.es) { state.es.close(); state.es = null; }   // never leak a prior stream
  const es = new EventSource(`/runs/${runId}/events`);
  state.es = es;
  es.onmessage = async (e) => {
    const ev = JSON.parse(e.data);
    handleEvent(ev);
    if (ev.stage === "done") {
      es.close();
      stopTimer();
      $("stop").style.display = "none";
      $("run").disabled = false;
      if (ev.status === "error" || ev.status === "cancelled") {
        setMsg((ev.status === "cancelled" ? "已停止" : "运行出错: ") + (ev.msg || ""));
        return;
      }
      await loadView(runId);
    }
  };
  es.onerror = () => { es.close(); stopTimer(); $("run").disabled = false; };
}

// Stop the current run: tell the backend to cancel, close the stream locally.
async function stopRun() {
  state.stopping = true;
  setMsg("正在停止…");
  if (state.es) { state.es.close(); state.es = null; }
  stopTimer();
  $("stop").style.display = "none";
  $("run").disabled = false;
  if (state.runId) {
    try { await fetch(`/runs/${state.runId}/cancel`, { method: "POST" }); } catch (_) {}
  }
  setMsg("已停止");
  setBarSweep();
}

function handleEvent(ev) {
  if (ev.stage && ev.stage !== "done") setStage(ev.stage);
  setMsg(ev.msg || ev.status || ev.stage);

  const total = typeof ev.total === "number" ? ev.total : 0;
  const done = ev.index || 0;

  if (ev.stage === "module0" && total > 0) {
    // Compile: real i/N determinate progress + honest per-item ETA.
    state.compileTotal = total;
    $("prog-count").textContent = `编译 ${done}/${total}`;
    setBarFraction(done / total);

    // ETA: after the 1st complaint finishes, extrapolate the rest (honest —
    // compile is the only stage with a defensible per-item cost estimate).
    if (done >= 1 && state.compileFirstDoneAt === null) {
      state.compileFirstDoneAt = Date.now();   // mark first completion; no ETA yet
    } else if (done >= 2 && state.compileFirstDoneAt !== null && done < total) {
      const perItem = (Date.now() - state.compileFirstDoneAt) / (done - 1) / 1000;
      const remain = perItem * (total - done);
      $("prog-eta").textContent = `编译约剩 ~${fmtDur(remain)}`;
    }
    if (done >= total) $("prog-eta").textContent = "";
  } else if (ev.stage === "module1" && total > 0 && done >= 1) {
    // Judge phase: determinate "精判 i/N" bar (no ETA — per-sub_problem judge
    // cost varies too much to extrapolate honestly).
    $("prog-count").textContent = `精判 ${done}/${total}`;
    $("prog-eta").textContent = "";
    setBarFraction(done / total);
  } else {
    // Other stages / module1 before the first sub_problem: duration dominated
    // by LLM variance → no fake ETA, sweep + live elapsed convey "still working".
    $("prog-count").textContent = "";
    $("prog-eta").textContent = "";
    setBarSweep();
  }
}

async function loadView(runId) {
  setStage("done");
  setBarFraction(1);
  const resp = await fetch(`/runs/${runId}/view`);
  state.view = await resp.json();
  $("progress").classList.add("hidden");
  $("workspace").classList.remove("hidden");
  renderManifest();
  renderProblems();
  $("hits").innerHTML = '<div class="empty-hint">← 选一个问题</div>';
  $("detail").innerHTML = "";
}

function renderManifest() {
  const m = state.view.manifest || {};
  $("manifest-summary").innerHTML =
    `最终入选 SFT 样本 <b>${m.targeted_count}</b> 条`;
}

// ── Column 1: problems + capabilities ─────────────────────────────
function renderProblems() {
  const el = $("problems");
  el.innerHTML = "";
  if (!state.view.problems.length) {
    el.innerHTML = '<div class="empty-hint">无通过的问题（模块0 全部 drop）</div>';
    return;
  }
  state.view.problems.forEach((p, i) => {
    const pd = document.createElement("div");
    pd.className = "problem" + (p.id === state.activeProblem ? " active" : "");
    pd.innerHTML =
      `<button class="p-detail" title="查看 module0 编译详情">详情</button>` +
      `<span class="p-conf">${(p.confidence ?? 0).toFixed(2)}</span>` +
      `<span class="p-num">${i + 1}.</span> ${escapeHtml(p.failure_summary)}`;
    pd.onclick = () => selectProblem(p.id);
    pd.querySelector(".p-detail").onclick = (e) => {
      e.stopPropagation();   // don't also select the problem
      openCompileModal(p);
    };
    el.appendChild(pd);

    if (p.id === state.activeProblem) {
      for (const cap of p.capabilities) {
        const cd = document.createElement("div");
        const focused = cap.label === state.activeCap;
        const dimmed = state.activeCap && !focused;
        cd.className = "capability" + (focused ? " focused" : "") + (dimmed ? " dimmed" : "");
        cd.style.color = cap.color;
        cd.style.background = hexA(cap.color, 0.13);
        cd.innerHTML = `<span>● ${escapeHtml(cap.label)}</span><span class="cap-n">${cap.hit_count}</span>`;
        cd.onclick = () => focusCapability(p.id, cap.label);
        el.appendChild(cd);
      }
    }
  });
}

function selectProblem(pid) {
  state.activeProblem = pid;
  state.activeCap = null;
  state.activeTraj = null;
  renderProblems();
  renderHits();
  $("detail").innerHTML = "";
}

function focusCapability(pid, label) {
  state.activeCap = (state.activeCap === label) ? null : label;
  renderProblems();
  renderHits();
  renderDetail();
}

function currentProblem() {
  return state.view.problems.find((p) => p.id === state.activeProblem);
}

// Union of hit trajectories across capabilities (or focused capability only)
function currentHits() {
  const p = currentProblem();
  if (!p) return [];
  const caps = state.activeCap
    ? p.capabilities.filter((c) => c.label === state.activeCap)
    : p.capabilities;
  const byTraj = {};
  for (const cap of caps) {
    for (const h of cap.hit_trajectories) {
      const key = `${h.trajectory_id}#${h.slice_index}`;
      if (!byTraj[key]) byTraj[key] = { ...h, caps: [] };
      byTraj[key].caps.push({ label: cap.label, color: cap.color, spans: h.loss_mask_spans });
    }
  }
  return Object.values(byTraj);
}

// ── Column 2: hit trajectories ────────────────────────────────────
function renderHits() {
  const el = $("hits");
  el.innerHTML = "";
  const hits = currentHits();
  if (!hits.length) {
    el.innerHTML = '<div class="empty-hint">该问题无命中轨迹</div>';
    return;
  }
  for (const h of hits) {
    const hd = document.createElement("div");
    const isActive = state.activeTraj && state.activeTraj.trajectory_id === h.trajectory_id
      && state.activeTraj.slice_index === h.slice_index;
    hd.className = "hit" + (isActive ? " active" : "");
    const badge = h.selected ? '<span class="selected-badge">已入选</span>' : "";
    hd.innerHTML =
      `<span><span class="hit-id">${escapeHtml(h.trajectory_id)}</span>` +
      `<span class="hit-seg"> · slice${h.slice_index} · ${h.caps.length}片段</span></span>${badge}`;
    hd.onclick = () => selectTrajectory(h);
    el.appendChild(hd);
  }
}

async function selectTrajectory(hit) {
  state.activeTraj = { trajectory_id: hit.trajectory_id, slice_index: hit.slice_index };
  await ensureTrajectory(hit.trajectory_id);
  renderHits();
  renderDetail();
}

function cacheKey(tid) { return `${state.runId}#${tid}`; }

async function ensureTrajectory(tid) {
  const k = cacheKey(tid);
  if (state.trajCache[k]) return;
  const resp = await fetch(`/runs/${state.runId}/trajectory/${tid}`);
  state.trajCache[k] = await resp.json();
}

// Which capabilities' spans should highlight this step. Highlight color is a
// single unified yellow (see renderDetail); focus just controls dim vs solid.
function spansForStep(stepIndex, hit) {
  const out = [];
  for (const cap of hit.caps) {
    const inSpan = (cap.spans || []).some(
      (s) => stepIndex >= s.start_step && stepIndex <= s.end_step);
    if (!inSpan) continue;
    const focused = !state.activeCap || cap.label === state.activeCap;
    out.push({ ...cap, focused });
  }
  return out;
}

// ── Column 3: trajectory detail with step-level highlight ─────────
function renderDetail() {
  const el = $("detail");
  el.innerHTML = "";
  if (!state.activeTraj) return;
  const traj = state.trajCache[cacheKey(state.activeTraj.trajectory_id)];
  if (!traj) return;
  const hit = currentHits().find(
    (h) => h.trajectory_id === state.activeTraj.trajectory_id
      && h.slice_index === state.activeTraj.slice_index);

  for (const step of traj.steps) {
    const sd = document.createElement("div");
    let cls = "step " + step.role;
    const caps = hit ? spansForStep(step.index, hit) : [];
    const focusedCaps = caps.filter((c) => c.focused);
    const dimCaps = caps.filter((c) => !c.focused);

    if (focusedCaps.length) {
      cls += " hit-span";   // unified yellow highlight (CSS), color-independent
    } else if (dimCaps.length) {
      cls += " hit-dim";  // 非聚焦能力命中：淡显（decision 11）
    }
    sd.className = cls;

    // head line: icon + role + optional tool name
    const roleIcon = { user: "👤", assistant: "🤖", tool: "⚙", system: "◆" }[step.role] || "·";
    const head = document.createElement("div");
    head.className = "step-head";
    head.innerHTML = `${roleIcon} ${escapeHtml(step.role)}` +
      (step.tool_call_name ? ` · <span class="tc">🔧 ${escapeHtml(step.tool_call_name)}</span>` : "");
    sd.appendChild(head);

    // body
    const body = (step.content || step.tool_result || "").slice(0, 600);
    const bodyEl = document.createElement("div");
    bodyEl.textContent = body;
    sd.appendChild(bodyEl);

    // 小色标签：命中该 step 的所有能力（含非聚焦，decision 11）
    for (const c of caps) {
      const tag = document.createElement("span");
      tag.className = "cap-tag";
      tag.style.background = c.color;
      tag.textContent = c.label;
      if (!c.focused) tag.style.opacity = ".5";
      sd.appendChild(tag);
    }
    el.appendChild(sd);
  }
}

// ── utils ─────────────────────────────────────────────────────────
function hexA(hex, a) {
  // "#rrggbb" + alpha(0..1) → rgba(); fallback passthrough for non-hex
  const m = /^#([0-9a-f]{6})$/i.exec(hex || "");
  if (!m) return hex;
  const n = parseInt(m[1], 16);
  return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
}

function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

// ── module0 compile-detail modal ──────────────────────────────────
function openCompileModal(p) {
  const c = p.compile || {};
  const rows = [];
  const row = (label, html) => rows.push(
    `<div class="mf-row"><div class="mf-k">${label}</div><div class="mf-v">${html}</div></div>`);

  row("问题", `<span class="mf-num">${escapeHtml(p.failure_summary)}</span>`);
  if (c.raw_text) row("原始片段", escapeHtml(c.raw_text));
  row("置信度 / 来源",
    `<code>${(p.confidence ?? 0).toFixed(2)}</code>` +
    (c.origin ? ` · ${escapeHtml(c.origin)}` : ""));

  const caps = (c.target_capability || []).map((l) => `<code>${escapeHtml(l)}</code>`).join(" ");
  row("能力标签", caps || "<span class='mf-empty'>无</span>");

  if (c.trajectory_signal) row("轨迹信号", escapeHtml(c.trajectory_signal));

  const kws = (c.keywords || []).map((k) => `<span class="mf-chip">${escapeHtml(k)}</span>`).join("");
  row("BM25 关键词", kws || "<span class='mf-empty'>无</span>");

  const sf = c.structured_filters || {};
  const sfParts = [];
  if (sf.languages) sfParts.push(`languages: ${sf.languages.map(escapeHtml).join(", ")}`);
  if (sf.tools_used) sfParts.push(`tools_used: ${sf.tools_used.map(escapeHtml).join(", ")}`);
  if (sf.has_verification_step != null) sfParts.push(`has_verification_step: ${sf.has_verification_step}`);
  row("结构化过滤", sfParts.length
    ? `<code>${sfParts.join("</code> · <code>")}</code>` : "<span class='mf-empty'>无</span>");

  const hyde = (c.hyde_positive || []);
  const hydeHtml = hyde.length
    ? hyde.map((h, i) => `<div class="mf-hyde"><span class="mf-hyde-n">HyDE #${i + 1}</span>${escapeHtml(h)}</div>`).join("")
    : "<span class='mf-empty'>无</span>";
  row(`HyDE 正例 (${hyde.length})`, hydeHtml);

  $("modal-title").textContent = `module0 编译详情 · ${p.id}`;
  $("modal-body").innerHTML = rows.join("");
  $("compile-modal").classList.remove("hidden");
}

function closeCompileModal() {
  $("compile-modal").classList.add("hidden");
}

$("modal-close").addEventListener("click", closeCompileModal);
$("compile-modal").addEventListener("click", (e) => {
  if (e.target.id === "compile-modal") closeCompileModal();  // click backdrop
});
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") closeCompileModal();
});
