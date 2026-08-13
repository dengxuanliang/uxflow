# SPDX-License-Identifier: Apache-2.0
"""P0: ingest 写路径批量 embedding —— 模型调用数从每切片一次降到每批一次。

正确性红线：批量产出的签名必须与逐切片路径**逐字段相同**，且每个切片拿到的
必须是**自己**那段文本的向量（zip 错位是这类改动最隐蔽的 bug —— 不报错，只让
召回结果莫名其妙地差）。
"""
import pytest

from module1.pipeline import TrajectoryPipeline, PipelineConfig
from module1.signature import build_embedding_text, extract_signature
from module1.slicer import slice_trajectory
from module1.loader import load_trajectories


class CountingEmbedder:
    """记录 embed / embed_batch 各被调用多少次、每批多大。"""

    dimension = 8

    def __init__(self):
        self.embed_calls = 0
        self.batch_calls = 0
        self.batch_sizes = []

    def embed(self, text):
        self.embed_calls += 1
        return self._vec(text)

    def embed_batch(self, texts):
        self.batch_calls += 1
        self.batch_sizes.append(len(texts))
        return [self._vec(t) for t in texts]

    @staticmethod
    def _vec(text):
        # 编码输入文本身份 → 任何错位都会被下面的断言抓到
        return [float(len(text)), float(sum(map(ord, text[:32])))] + [0.0] * 6


def _cfg(**kw):
    kw.setdefault("judge_model", "t")
    kw.setdefault("judge_batch_size", 1)
    return PipelineConfig(**kw)


def test_ingest_uses_embed_batch_not_per_slice(trajectories_path):
    """写路径必须走 embed_batch；逐切片 embed 一次都不许调。"""
    emb = CountingEmbedder()
    p = TrajectoryPipeline(config=_cfg(embedding_model=emb), gateway=None)

    p.ingest_trajectories([trajectories_path])

    n_slices = sum(len(slice_trajectory(t)) for t in load_trajectories(trajectories_path))
    assert n_slices > 1, "fixture 需要多个切片才有意义"
    assert emb.embed_calls == 0, "写路径仍在逐切片 embed"
    assert emb.batch_calls == 1, f"期望 1 次 embed_batch，实际 {emb.batch_calls}"
    assert emb.batch_sizes == [n_slices]


def test_batched_signatures_identical_to_sequential(trajectories_path):
    """批量 vs 逐切片：所有结构化字段 + embedding 必须逐字段相同。"""
    emb = CountingEmbedder()
    p = TrajectoryPipeline(config=_cfg(embedding_model=emb), gateway=None)
    p.ingest_trajectories([trajectories_path])
    batched = sorted(p._store._signatures, key=lambda s: (s.trajectory_id, s.slice_index))

    # 逐切片参照实现（改动前的行为）
    reference = []
    for traj in load_trajectories(trajectories_path):
        for sl in slice_trajectory(traj):
            reference.append(extract_signature(sl, embedding_model=CountingEmbedder()))
    reference.sort(key=lambda s: (s.trajectory_id, s.slice_index))

    assert len(batched) == len(reference)
    fields = ("trajectory_id", "slice_index", "step_range", "step_count", "turn_count",
              "languages", "tools_used", "has_error_pattern", "has_success_pattern",
              "has_verification_step", "bm25_tokens", "embedding")
    for got, want in zip(batched, reference):
        for f in fields:
            assert getattr(got, f) == getattr(want, f), (
                f"{got.trajectory_id}#{got.slice_index} 字段 {f} 不一致")


def test_each_slice_gets_its_own_vector(trajectories_path):
    """防 zip 错位：每个切片的向量必须由它自己的 summary 文本算出。"""
    emb = CountingEmbedder()
    p = TrajectoryPipeline(config=_cfg(embedding_model=emb), gateway=None)
    p.ingest_trajectories([trajectories_path])

    by_key = {(s.trajectory_id, s.slice_index): s for s in p._store._signatures}
    for traj in load_trajectories(trajectories_path):
        for sl in slice_trajectory(traj):
            want = CountingEmbedder._vec(build_embedding_text(sl))
            got = by_key[(sl.trajectory_id, sl.slice_index)].embedding
            assert got == want, f"{sl.trajectory_id}#{sl.slice_index} 拿到了别人的向量"


def test_no_embedding_model_still_works(trajectories_path):
    """embedding_model=None（单测/无 ML 依赖路径）不得回归。"""
    p = TrajectoryPipeline(config=_cfg(), gateway=None)
    p.ingest_trajectories([trajectories_path])
    assert p._store.size > 0
    assert all(s.embedding == [] for s in p._store._signatures)


def test_on_trajectory_callback_still_fires(trajectories_path):
    """批量化不得影响 on_trajectory 回调（run_ingest 的轨迹计数依赖它）。"""
    emb = CountingEmbedder()
    p = TrajectoryPipeline(config=_cfg(embedding_model=emb), gateway=None)
    seen = []
    p.ingest_trajectories([trajectories_path],
                          on_trajectory=lambda t, src: seen.append((t.id, src)))
    assert len(seen) == len(load_trajectories(trajectories_path))


def test_chunking_splits_large_batches(tmp_path):
    """超过 _EMBED_CHUNK 的切片必须分多批，且跨批不得错位。

    ApiEmbedder 把整批塞进单个 HTTP 请求，不分块会撞 provider 的输入上限。
    """
    import json
    from module1.pipeline import _EMBED_CHUNK

    n_traj = _EMBED_CHUNK + 6          # 每条短轨迹恰好 1 个切片
    path = tmp_path / "many.jsonl"
    path.write_text("\n".join(json.dumps({
        "id": f"t{i}",
        "messages": [{"role": "user", "content": f"q{i}"},
                     {"role": "assistant", "content": f"a{i}"}],
    }) for i in range(n_traj)), encoding="utf-8")

    emb = CountingEmbedder()
    p = TrajectoryPipeline(config=_cfg(embedding_model=emb), gateway=None)
    p.ingest_trajectories([path])

    assert emb.batch_sizes == [_EMBED_CHUNK, 6]

    # 跨批错位检查：每个切片仍须持有自己那段文本的向量
    by_key = {(s.trajectory_id, s.slice_index): s for s in p._store._signatures}
    for traj in load_trajectories(path):
        for sl in slice_trajectory(traj):
            want = CountingEmbedder._vec(build_embedding_text(sl))
            assert by_key[(sl.trajectory_id, sl.slice_index)].embedding == want


def test_chunking_honours_embedder_preferred_batch_size(tmp_path):
    """分块大小必须取自后端自报的 preferred_batch_size，而非模块常量。

    批次上限是后端属性而非调用点属性：ApiEmbedder 受网关响应体上限约束
    （32 条 3072 维 base64 ≈0.5MB，上限约 1MB）。取 8 —— 与 _EMBED_CHUNK(32)
    和 FakeEmbedder(128) 都不同，断言才不会碰巧撞上 fallback 而假绿。
    """
    import json
    from module1.pipeline import _EMBED_CHUNK

    batch = 8
    assert batch != _EMBED_CHUNK, "取值必须区别于 fallback，否则测不出差异"

    class PreferredBatchEmbedder(CountingEmbedder):
        preferred_batch_size = batch

    n_traj = batch * 3 + 5             # 29 条 → 4 批（8/8/8/5），故意非整除
    path = tmp_path / "many.jsonl"
    path.write_text("\n".join(json.dumps({
        "id": f"t{i}",
        "messages": [{"role": "user", "content": f"q{i}"},
                     {"role": "assistant", "content": f"a{i}"}],
    }) for i in range(n_traj)), encoding="utf-8")

    emb = PreferredBatchEmbedder()
    p = TrajectoryPipeline(config=_cfg(embedding_model=emb), gateway=None)
    p.ingest_trajectories([path])

    n_texts = sum(len(slice_trajectory(t)) for t in load_trajectories(path))
    assert n_texts > batch, "fixture 必须跨多个 chunk 才有意义"

    expected_calls = -(-n_texts // batch)          # ceil(n_texts / batch)
    assert emb.batch_calls == expected_calls, (
        f"期望 ceil({n_texts}/{batch})={expected_calls} 次 embed_batch，"
        f"实际 {emb.batch_calls}（batch_sizes={emb.batch_sizes}）")
    assert all(size <= batch for size in emb.batch_sizes), (
        f"有批次超过后端自报上限 {batch}：{emb.batch_sizes}")
    assert sum(emb.batch_sizes) == n_texts, "分块丢了或重复了文本"

    # 跨批错位检查：每个切片仍须持有自己那段文本的向量
    by_key = {(s.trajectory_id, s.slice_index): s for s in p._store._signatures}
    for traj in load_trajectories(path):
        for sl in slice_trajectory(traj):
            want = CountingEmbedder._vec(build_embedding_text(sl))
            assert by_key[(sl.trajectory_id, sl.slice_index)].embedding == want


# ── on_progress: 入库三阶段可观测性 ──────────────────────────────────
def test_on_progress_reports_three_phases(trajectories_path):
    """三个阶段都要报：切片 / 向量化 / 写库。"""
    emb = CountingEmbedder()
    p = TrajectoryPipeline(config=_cfg(embedding_model=emb), gateway=None)
    seen = []
    p.ingest_trajectories([trajectories_path],
                          on_progress=lambda ph, d, t: seen.append((ph, d, t)))

    phases = [ph for ph, _, _ in seen]
    assert "slicing" in phases
    assert "embedding" in phases
    assert "writing" in phases
    # 阶段顺序不得乱：切片全部先于向量化，向量化全部先于写库。
    assert phases.index("embedding") > len([x for x in phases if x == "slicing"]) - 1
    assert phases.index("writing") > phases.index("embedding")


def test_progress_done_never_exceeds_total(trajectories_path):
    """任何阶段都不许出现 done > total（off-by-one 会让进度条冲过 100%）。"""
    emb = CountingEmbedder()
    p = TrajectoryPipeline(config=_cfg(embedding_model=emb), gateway=None)
    seen = []
    p.ingest_trajectories([trajectories_path],
                          on_progress=lambda ph, d, t: seen.append((ph, d, t)))
    for phase, done, total in seen:
        assert 0 <= done <= total, f"{phase}: done={done} total={total}"


def test_slicing_progress_is_monotonic_across_multiple_files(tmp_path):
    """多文件输入时切片进度必须全局单调递增，不得按文件重置。

    按文件各自计数会让前端收到 1/3 2/3 3/3 然后 1/2 2/2 —— 进度条冲满
    再退回，比不显示更糟。
    """
    import json

    def write(name, n):
        path = tmp_path / name
        path.write_text("\n".join(json.dumps({
            "id": f"{name}_{i}",
            "messages": [{"role": "user", "content": f"q{i}"},
                         {"role": "assistant", "content": f"a{i}"}],
        }) for i in range(n)), encoding="utf-8")
        return path

    a, b = write("a.jsonl", 3), write("b.jsonl", 2)
    emb = CountingEmbedder()
    p = TrajectoryPipeline(config=_cfg(embedding_model=emb), gateway=None)
    seen = []
    p.ingest_trajectories([a, b], on_progress=lambda ph, d, t: seen.append((ph, d, t)))

    slicing = [(d, t) for ph, d, t in seen if ph == "slicing"]
    assert slicing == [(1, 5), (2, 5), (3, 5), (4, 5), (5, 5)], slicing


def test_embedding_progress_reports_每个_chunk(tmp_path):
    """向量化先报 0/N 再按 chunk 报进度 —— 这是最慢的一段，静默等于假死。

    0/N 那条是关键：进度只在 chunk 边界更新，切片数 <= _EMBED_CHUNK 时只有一个
    chunk，不先发这条，界面会一直停在"切片 N/N"直到整批算完，用户以为卡在切片。
    """
    import json
    from module1.pipeline import _EMBED_CHUNK

    n = _EMBED_CHUNK + 6
    path = tmp_path / "many.jsonl"
    path.write_text("\n".join(json.dumps({
        "id": f"t{i}",
        "messages": [{"role": "user", "content": f"q{i}"},
                     {"role": "assistant", "content": f"a{i}"}],
    }) for i in range(n)), encoding="utf-8")

    emb = CountingEmbedder()
    p = TrajectoryPipeline(config=_cfg(embedding_model=emb), gateway=None)
    seen = []
    p.ingest_trajectories([path], on_progress=lambda ph, d, t: seen.append((ph, d, t)))

    embedding = [(d, t) for ph, d, t in seen if ph == "embedding"]
    assert embedding == [(0, n), (_EMBED_CHUNK, n), (n, n)], embedding


def test_embedding_reports_start_even_for_single_chunk(trajectories_path):
    """单 chunk（切片数 <= _EMBED_CHUNK）也必须先报 0/N。

    这正是用户实测撞到的场景：5 条轨迹只有一个 chunk，修复前整段静默，
    界面停在"切片 5/5"，真正在跑的是向量化。
    """
    from module1.pipeline import _EMBED_CHUNK

    emb = CountingEmbedder()
    p = TrajectoryPipeline(config=_cfg(embedding_model=emb), gateway=None)
    seen = []
    p.ingest_trajectories([trajectories_path],
                          on_progress=lambda ph, d, t: seen.append((ph, d, t)))

    embedding = [(d, t) for ph, d, t in seen if ph == "embedding"]
    assert len(embedding) >= 2, f"单 chunk 也要有起止两条，实际 {embedding}"
    assert embedding[0][0] == 0, f"首条必须是 0/N，实际 {embedding[0]}"
    assert embedding[0][1] <= _EMBED_CHUNK, "本例应只有一个 chunk"
    assert embedding[-1][0] == embedding[-1][1], "末条必须是 N/N"


def test_on_progress_is_optional(trajectories_path):
    """不传 on_progress 不得报错（向后兼容既有调用方）。"""
    emb = CountingEmbedder()
    p = TrajectoryPipeline(config=_cfg(embedding_model=emb), gateway=None)
    p.ingest_trajectories([trajectories_path])      # 不传
    assert p._store.size > 0


# ── 回归：进度回调与批量返回的健壮性 ─────────────────────────────────
def test_progress_callback_failure_does_not_abort_ingest(trajectories_path):
    """on_progress 抛异常不得中止入库。

    回调那头是 SSE/前端，断连或序列化失败都可能抛。而 add_batch 在最末尾 —— 让
    异常冒泡等于整批白干（回归前实测：20 条轨迹一条都没入库）。
    """
    emb = CountingEmbedder()
    p = TrajectoryPipeline(config=_cfg(embedding_model=emb), gateway=None)

    def boom(phase, done, total):
        raise RuntimeError("前端断开")

    p.ingest_trajectories([trajectories_path], on_progress=boom)
    assert p._store.size > 0, "进度回调把整个入库炸掉了"


def test_short_embed_batch_raises_instead_of_silently_truncating(trajectories_path):
    """embed_batch 少返回时必须报错，不能让切片带空向量入库。

    zip 遇到短列表会静默截断 —— 尾部切片 embedding 为 []，不报错，只是在向量
    召回里永远命不中。这类静默腐化比直接失败危险得多。
    """
    class ShortEmbedder(CountingEmbedder):
        def embed_batch(self, texts):
            return super().embed_batch(texts)[:-1]      # 少一条

    p = TrajectoryPipeline(config=_cfg(embedding_model=ShortEmbedder()), gateway=None)
    with pytest.raises(ValueError, match="与输入"):
        p.ingest_trajectories([trajectories_path])


def test_long_embed_batch_also_raises(trajectories_path):
    """多返回同样要报错 —— 说明 provider 契约已破，不该继续写库。"""
    class LongEmbedder(CountingEmbedder):
        def embed_batch(self, texts):
            out = super().embed_batch(texts)
            return out + [out[0]]                       # 多一条

    p = TrajectoryPipeline(config=_cfg(embedding_model=LongEmbedder()), gateway=None)
    with pytest.raises(ValueError, match="与输入"):
        p.ingest_trajectories([trajectories_path])
