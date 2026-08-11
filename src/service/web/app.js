// Trajectory Inspector frontend (spec §9, decisions 10-12).
const $ = (id) => document.getElementById(id);

// Per-mode stage sets. `runs` = 单次检分老路径; `search`/`ingest` = 双模式新路径.
// ingest_traj/ingest_manifest 两个后端事件都映射到单一 "ingest" 节点(见 handleEvent).
const STAGE_SETS = {
  runs: ["module0", "module1", "module2", "module3", "done"],
  search: ["search0", "search1", "done"],
  ingest: ["ingest", "done"],
};
const STAGE_LABELS = {
  module0: "编译", module1: "召回·精判", module2: "软加分", module3: "优选",
  done: "完成", search0: "分析", search1: "检索", ingest: "写库",
};
let STAGES = STAGE_SETS.runs;   // current mode's stages; setStage/resetProgress read this

// Rebuild the #stepper DOM for the given mode's stage set.
function setStepper(mode) {
  STAGES = STAGE_SETS[mode] || STAGE_SETS.runs;
  const el = $("stepper");
  el.innerHTML = "";
  STAGES.forEach((stage, i) => {
    if (i > 0) {
      const sep = document.createElement("div");
      sep.className = "step-sep";
      el.appendChild(sep);
    }
    const node = document.createElement("div");
    node.className = "step-node";
    node.dataset.stage = stage;
    node.innerHTML = `<i></i><span>${STAGE_LABELS[stage] || stage}</span>`;
    el.appendChild(node);
  });
}

let state = {
  runId: null,
  view: null,
  activeProblem: null,   // problem id
  activeCap: null,       // focused capability label (click a capability to focus)
  activeTraj: null,      // {trajectory_id, slice_index}
  activeHighlight: -1,   // index into current .hit-span elements
  trajCache: {},
  // progress
  es: null,                  // active EventSource, so we can stop it
  startedAt: null,
  timer: null,
  compileFirstDoneAt: null,  // wall time when first complaint finished compiling
  compileFirstDoneN: null,   // `done` count observed when the anchor was set
  compileTotal: 0,
  stopping: false,
  runGen: 0,                 // per-run generation token; isolates stop/rerun races
  currentDbName: null,       // basename of the current DB, for the stats line
};

$("run").addEventListener("click", startRun);
$("stop").addEventListener("click", stopRun);
$("next-highlight").addEventListener("click", jumpToNextHighlight);
$("btn-search").addEventListener("click", startSearch);
$("btn-ingest").addEventListener("click", startIngest);
$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); startSearch(); }
});

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

// ✕ 取消误选：清空该槽，阻止冒泡到 <label>（否则会重新打开文件选择框）。
for (const [clearId, inputId] of [
  ["clear-manifest", "manifest"],
  ["clear-trajectories", "trajectories"],
]) {
  $(clearId).addEventListener("click", (e) => {
    e.preventDefault();
    e.stopPropagation();
    clearFileSlot(inputId);
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
  const myGen = ++state.runGen;   // this run's token; a later startRun/stopRun bumps runGen and invalidates us
  $("run").disabled = true;
  $("progress").classList.remove("hidden");
  $("workspace").classList.add("hidden");
  setStepper("runs");
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

  // 若在上传窗口内用户点了停止、或又发起了新的运行，本次 startRun 作废
  if (myGen !== state.runGen || state.stopping) {
    try { await fetch(`/runs/${run_id}/cancel`, { method: "POST" }); } catch (_) {}
    return;
  }

  state.runId = run_id;
  subscribeEvents(run_id);
}

// ── Search mode (POST /search) ────────────────────────────────────
async function startSearch() {
  const q = $("question").value.trim();
  if (!q) { alert("请先输入用户问题"); return; }

  state.stopping = false;
  state.runId = null;
  const myGen = ++state.runGen;
  $("run").disabled = true;
  $("progress").classList.remove("hidden");
  $("workspace").classList.add("hidden");
  setStepper("search");
  resetProgress();
  resetRunState();
  hideDedupBanner();
  setStage("search0");
  setMsg("分析中…");
  $("stop").style.display = "";
  startTimer();

  let run_id;
  try {
    const resp = await fetch("/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: q }),
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    ({ run_id } = await resp.json());
  } catch (err) {
    setMsg("搜索失败: " + (err.message || err));
    stopTimer();
    $("stop").style.display = "none";
    $("run").disabled = false;
    return;
  }

  if (myGen !== state.runGen || state.stopping) {
    try { await fetch(`/runs/${run_id}/cancel`, { method: "POST" }); } catch (_) {}
    return;
  }
  state.runId = run_id;
  subscribeEvents(run_id);
}

// ── Ingest mode (POST /ingest multipart) ──────────────────────────
async function startIngest() {
  const mf = $("manifest").files[0];
  const tj = $("trajectories").files[0];
  if (!mf && !tj) { alert("请至少选择清单或轨迹文件之一"); return; }
  const fd = new FormData();
  if (mf) fd.append("manifest", mf);
  if (tj) fd.append("trajectories", tj);

  state.stopping = false;
  state.runId = null;
  const myGen = ++state.runGen;
  $("run").disabled = true;
  $("progress").classList.remove("hidden");
  $("workspace").classList.add("hidden");
  setStepper("ingest");
  resetProgress();
  resetRunState();
  hideDedupBanner();
  setStage("ingest");
  setMsg("上传中…");
  $("stop").style.display = "";
  startTimer();

  let run_id;
  try {
    const resp = await fetch("/ingest", { method: "POST", body: fd });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    ({ run_id } = await resp.json());
  } catch (err) {
    setMsg("入库失败: " + (err.message || err));
    stopTimer();
    $("stop").style.display = "none";
    $("run").disabled = false;
    return;
  }

  if (myGen !== state.runGen || state.stopping) {
    try { await fetch(`/runs/${run_id}/cancel`, { method: "POST" }); } catch (_) {}
    return;
  }
  state.runId = run_id;
  // 清空文件输入，便于下次选择（FormData 已构造，此处清空不影响本次上传）
  clearIngestInputs();
  subscribeEvents(run_id);
}

// 清空单个文件槽：input 值、名字占位、filled 状态。
function clearFileSlot(id) {
  $(id).value = "";
  $("name-" + id).textContent = "点击选择文件…";
  $("slot-" + id).classList.remove("filled");
}

// Reset both ingest file inputs + their slot chips after a submit.
function clearIngestInputs() {
  clearFileSlot("manifest");
  clearFileSlot("trajectories");
}

// ── Stats bar (GET /stats) ────────────────────────────────────────
async function loadStats() {
  try {
    const resp = await fetch("/stats");
    if (!resp.ok) return;
    const s = await resp.json();
    const dbName = state.currentDbName || "当前库";
    $("stats-bar").textContent =
      `【${dbName}】中 ${s.problems} 问题 · ${s.trajectories} 轨迹 · ${s.signatures} 切片`;
    renderFakeBanner(s);
  } catch (_) { /* stats are best-effort */ }
}

// 后端披露：任一后端为 fake 时常驻告警。两个后端都回显 —— 最容易踩的是
// 切了 LLM 忘了切 embedding 这种半真半假状态，只报一个会掩盖它。
function renderFakeBanner(s) {
  const el = $("fake-banner");
  if (!el) return;
  const llmFake = s.llm_backend === "fake";
  const embedFake = s.embed_backend === "fake";
  if (!llmFake && !embedFake) { el.classList.add("hidden"); return; }

  const parts = [];
  if (llmFake) parts.push("LLM 为固定回放，不是真实模型输出");
  if (embedFake) parts.push("Embedding 向量为确定性哈希，语义无意义");
  el.textContent =
    `⚠️ FAKE 后端（UXFLOW_LLM_BACKEND=${s.llm_backend} · ` +
    `UXFLOW_EMBED_BACKEND=${s.embed_backend}）—— ${parts.join("；")}。` +
    `本次结果仅验证流水线跑通，不可用于真实数据筛选。`;
  el.classList.remove("hidden");
}

function showDedupBanner(dedup) {
  const el = $("dedup-banner");
  const sim = (dedup.similarity ?? 0).toFixed(2);
  el.innerHTML =
    `<span class="dedup-text">≈ 与已有问题高度重复：${escapeHtml(dedup.matched_question)}` +
    `（相似度 ${sim}）</span>` +
    `<button class="dedup-close" type="button" aria-label="关闭">✕</button>`;
  el.querySelector(".dedup-close").onclick = hideDedupBanner;
  el.classList.remove("hidden");
}
function hideDedupBanner() { $("dedup-banner").classList.add("hidden"); }

// ── 库管理（spec §7）───────────────────────────────────
async function loadDatabases() {
  try {
    const resp = await fetch("/databases");
    if (!resp.ok) return;
    const { current, databases } = await resp.json();
    state.currentDbName = (current || "").split("/").pop() || "当前库";
    const sel = $("switch-select");
    sel.innerHTML = "";
    for (const d of databases) {
      const opt = document.createElement("option");
      opt.value = d.name;   // 文件名：后端 switch 按 current().parent 解析
      const cnt = d.problems == null ? "?" : d.problems;
      opt.textContent = `${d.name}（${cnt} 问题）` + (d.is_current ? " · 当前" : "");
      if (d.is_current) opt.selected = true;
      sel.appendChild(opt);
    }
    loadStats();   // 库名可能已更新，刷新信息行
  } catch (_) { /* best-effort */ }
}

async function switchDatabase(path) {
  if (!path) return false;
  let resp;
  try {
    resp = await fetch("/databases/switch", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ path }),
    });
  } catch (err) {
    alert("切换失败: " + (err.message || err));
    loadDatabases();   // resync dropdown to actual backend state
    return false;
  }
  if (resp.status === 409) { alert("有任务运行中，无法切换库"); loadDatabases(); return false; }
  if (!resp.ok) { alert("切换失败: HTTP " + resp.status); loadDatabases(); return false; }
  afterDbChange();
  return true;
}

// 切换库弹窗：复用 .modal 样式，从下拉选库后走 switch。
function openSwitchModal() { $("switch-modal").classList.remove("hidden"); }
function closeSwitchModal() { $("switch-modal").classList.add("hidden"); }

async function confirmSwitch() {
  const p = $("switch-select").value;
  if (!p) { return; }
  const ok = await switchDatabase(p);
  if (ok) closeSwitchModal();   // 成功关弹窗；失败 switchDatabase 已 alert，弹窗留着
}

async function clearDatabase() {
  if (!confirm("将永久删除当前库文件，不可恢复，确认？")) return;
  let resp;
  try {
    resp = await fetch("/databases/clear", { method: "POST" });
  } catch (err) {
    alert("清除失败: " + (err.message || err));
    return;
  }
  if (resp.status === 409) { alert("有任务运行中，无法清除库"); return; }
  if (!resp.ok) { alert("清除失败: HTTP " + resp.status); return; }
  afterDbChange();
}

// 新建库弹窗：复用 .modal 样式，输入路径后走 switch（后端切到不存在路径=新建空库）。
function openNewDbModal() {
  $("newdb-path").value = "";
  $("newdb-modal").classList.remove("hidden");
  $("newdb-path").focus();
}
function closeNewDbModal() { $("newdb-modal").classList.add("hidden"); }

async function confirmNewDb() {
  const p = $("newdb-path").value.trim();
  if (!p) { $("newdb-path").focus(); return; }
  // 复用切换；不存在路径→后端建空库并切换。仅成功时关弹窗，失败保留输入供重试。
  const ok = await switchDatabase(p);
  if (ok) closeNewDbModal();
}

function afterDbChange() {
  $("workspace").classList.add("hidden");
  hideDedupBanner();
  state.view = null;
  state.trajCache = {};
  loadDatabases();   // 内部会刷新 currentDbName 后再 loadStats()，避免旧库名配新计数的闪烁
}

function resetRunState() {
  state.view = null;
  state.activeProblem = null;
  state.activeCap = null;
  state.activeTraj = null;
  state.activeHighlight = -1;
  state.trajCache = {};
  updateHighlightTools();
}

function resetProgress() {
  state.compileFirstDoneAt = null;
  state.compileFirstDoneN = null;
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
  const myGen = state.runGen;   // this run's generation; a newer run bumps it (I5)
  const es = new EventSource(`/runs/${runId}/events`);
  state.es = es;
  es.onmessage = async (e) => {
    if (myGen !== state.runGen) { es.close(); return; }   // superseded by a newer run — stop
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
      await loadView(runId, myGen);
    }
  };
  es.onerror = () => {
    es.close(); state.es = null; stopTimer();
    $("run").disabled = false; $("stop").style.display = "none";
    setMsg("连接中断");
  };
}

// Stop the current run: tell the backend to cancel, close the stream locally.
async function stopRun() {
  state.stopping = true;
  state.runGen++;   // invalidate any startRun still in flight (upload window of an older/newer run)
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
  // ingest 后端发 ingest_traj/ingest_manifest 两种阶段 → 统一映射到 "ingest" 节点
  let stage = ev.stage;
  if (stage === "ingest_traj" || stage === "ingest_manifest") stage = "ingest";
  if (stage && stage !== "done") setStage(stage);
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
      state.compileFirstDoneN = done;          // anchor the count too (may skip past done=1)
    } else if (state.compileFirstDoneAt !== null && done > state.compileFirstDoneN && done < total) {
      const perItem = (Date.now() - state.compileFirstDoneAt) / (done - state.compileFirstDoneN) / 1000;
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

async function loadView(runId, myGen) {
  setStage("done");
  setBarFraction(1);
  const resp = await fetch(`/runs/${runId}/view`);
  if (myGen !== state.runGen) return;   // a newer run started during the fetch — drop stale view (I5)
  state.view = await resp.json();

  // ingest 模式：不渲染三栏，仅提示汇总 + 刷新状态条
  if (state.view.mode === "ingest") {
    const s = state.view.summary || {};
    $("workspace").classList.add("hidden");
    hideDedupBanner();
    // traj_ingested 为 null/缺失 = 本次没传轨迹（含老 runs 的 view）→ 不显示该段
    const parts = [];
    if (s.traj_ingested != null) parts.push(`处理轨迹 ${s.traj_ingested} 条`);
    parts.push(`新增 ${s.added ?? 0} 问题`, `跳过重复 ${s.skipped_dup ?? 0}`, `失败 ${s.failed ?? 0}`);
    setMsg(`入库完成：${parts.join(" · ")}`);
    loadStats();
    return;
  }

  // search / 老 runs 模式：三栏渲染
  $("progress").classList.add("hidden");
  $("workspace").classList.remove("hidden");

  // search 模式的问题去重提示
  if (state.view.dedup) showDedupBanner(state.view.dedup);
  else hideDedupBanner();

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
      `<span class="p-conf" title="Module0 子问题编译置信度，不代表轨迹命中质量">置信度 ${(p.confidence ?? 0).toFixed(2)}</span>` +
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
  state.activeHighlight = -1;
  renderProblems();
  renderHits();
  $("detail").innerHTML = "";
  updateHighlightTools();
}

function focusCapability(pid, label) {
  state.activeCap = (state.activeCap === label) ? null : label;
  state.activeHighlight = -1;
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
      `<span class="hit-main"><span class="hit-id">${escapeHtml(h.trajectory_id)}</span>` +
      `<span class="hit-seg"> · slice${h.slice_index} · ${h.caps.length}片段</span></span>${badge}`;
    // 证据：点击弹小窗（复用详情 modal），不再就地展开。
    const hasEvidence = h.evidence_step != null
      || (h.criteria_hit && h.criteria_hit.length);
    if (hasEvidence) {
      const btn = document.createElement("button");
      btn.className = "evidence-toggle";
      btn.type = "button";
      btn.textContent = "证据";
      btn.onclick = (e) => {
        e.stopPropagation();   // 不触发 selectTrajectory
        openEvidenceModal(h);
      };
      hd.appendChild(btn);
    }
    hd.onclick = () => selectTrajectory(h);
    el.appendChild(hd);
  }
}

async function selectTrajectory(hit) {
  state.activeTraj = { trajectory_id: hit.trajectory_id, slice_index: hit.slice_index };
  state.activeHighlight = -1;
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

// The focused loss_mask spans for the active trajectory, deduped by
// (start_step,end_step) and sorted — these are the jump units (one span,
// possibly spanning several steps, counts as ONE highlight; decision: span-level).
function currentFocusedSpans() {
  if (!state.activeTraj) return [];
  const hit = currentHits().find(
    (h) => h.trajectory_id === state.activeTraj.trajectory_id
      && h.slice_index === state.activeTraj.slice_index);
  if (!hit) return [];
  const seen = new Map();
  for (const cap of hit.caps) {
    const focused = !state.activeCap || cap.label === state.activeCap;
    if (!focused) continue;   // only focused-capability spans are jump targets
    for (const s of (cap.spans || [])) {
      const key = `${s.start_step}#${s.end_step}`;
      if (!seen.has(key)) seen.set(key, { start: s.start_step, end: s.end_step });
    }
  }
  return Array.from(seen.values()).sort(
    (a, b) => a.start - b.start || a.end - b.end);
}

// ── Column 3: trajectory detail with step-level highlight ─────────
function renderDetail() {
  const el = $("detail");
  el.innerHTML = "";
  if (!state.activeTraj) {
    updateHighlightTools();
    return;
  }
  const traj = state.trajCache[cacheKey(state.activeTraj.trajectory_id)];
  if (!traj) {
    updateHighlightTools();
    return;
  }
  const hit = currentHits().find(
    (h) => h.trajectory_id === state.activeTraj.trajectory_id
      && h.slice_index === state.activeTraj.slice_index);

  for (const step of traj.steps) {
    const sd = document.createElement("div");
    sd.dataset.stepIndex = String(step.index);   // for span-based jump targeting
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
    // 决定性证据步(PR-3):judge 指认的 evidence_step 上加星标,一眼定位证据落点。
    const isEvidence = hit && hit.evidence_step != null && step.index === hit.evidence_step;
    head.innerHTML = `${roleIcon} ${escapeHtml(step.role)}` +
      (step.tool_call_name ? ` · <span class="tc">🔧 ${escapeHtml(step.tool_call_name)}</span>` : "") +
      (isEvidence ? ` <span class="evidence-star" title="judge 指认的决定性证据步">⭐ 决定性证据</span>` : "");
    if (isEvidence) sd.classList.add("evidence-step");
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
  updateHighlightTools();
}

// Mark every step DOM element covered by the given span as the "current" highlight.
function markCurrentSpan(span) {
  for (const el of document.querySelectorAll("#detail .step")) {
    const idx = Number(el.dataset.stepIndex);
    const inSpan = span && idx >= span.start && idx <= span.end;
    el.classList.toggle("hit-current", !!inSpan);
  }
}

function updateHighlightTools() {
  const tools = $("detail-tools");
  const btn = $("next-highlight");
  const count = $("highlight-count");
  if (!tools || !btn || !count) return;

  const spans = currentFocusedSpans();   // jump units are spans, not steps
  const total = spans.length;
  tools.classList.toggle("hidden", !state.activeTraj);
  btn.disabled = total === 0;

  if (total === 0) {
    state.activeHighlight = -1;
    count.textContent = "高亮 0/0";
    markCurrentSpan(null);
    return;
  }
  if (state.activeHighlight >= total) state.activeHighlight = total - 1;
  const current = state.activeHighlight >= 0 ? state.activeHighlight + 1 : 0;
  count.textContent = `高亮 ${current}/${total}`;

  markCurrentSpan(state.activeHighlight >= 0 ? spans[state.activeHighlight] : null);
}

function jumpToNextHighlight() {
  const spans = currentFocusedSpans();
  if (!spans.length) {
    updateHighlightTools();
    return;
  }
  state.activeHighlight = (state.activeHighlight + 1) % spans.length;
  updateHighlightTools();
  // Scroll to the first step of the target span.
  const span = spans[state.activeHighlight];
  const target = document.querySelector(
    `#detail .step[data-step-index="${span.start}"]`);
  if (target) target.scrollIntoView({ behavior: "smooth", block: "center" });
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

// 证据弹窗：复用 #compile-modal 容器（同一时刻只开一个弹窗）。
function openEvidenceModal(h) {
  const rows = [];
  const row = (label, html) => rows.push(
    `<div class="mf-row"><div class="mf-k">${label}</div><div class="mf-v">${html}</div></div>`);

  if (h.evidence_step != null) {
    row("决定性证据", `<span class="mf-num">⭐ step ${h.evidence_step}</span>`);
  }
  if (h.criteria_hit && h.criteria_hit.length) {
    row("命中判据", h.criteria_hit.map(escapeHtml).join(" / "));
  }
  const caps = (h.caps || []).map((c) =>
    // c.color 来自后端 assign_colors 固定调色板（非用户输入），直接插入 style 安全。
    `<span class="ev-cap" style="background:${c.color}">${escapeHtml(c.label)}</span>`
  ).join("");
  if (caps) row("覆盖能力", caps);

  $("modal-title").textContent = `决定性证据 · ${h.trajectory_id}`;
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
  if (e.key === "Escape") { closeCompileModal(); closeNewDbModal(); closeSwitchModal(); }
});

// 库控件事件绑定
$("db-switch-btn").addEventListener("click", openSwitchModal);
$("db-new").addEventListener("click", openNewDbModal);
$("db-clear").addEventListener("click", clearDatabase);
// 切换弹窗
$("switch-close").addEventListener("click", closeSwitchModal);
$("switch-cancel").addEventListener("click", closeSwitchModal);
$("switch-confirm").addEventListener("click", confirmSwitch);
$("switch-modal").addEventListener("click", (e) => {
  if (e.target.id === "switch-modal") closeSwitchModal();
});
// 新建弹窗（保留）
$("newdb-close").addEventListener("click", closeNewDbModal);
$("newdb-cancel").addEventListener("click", closeNewDbModal);
$("newdb-confirm").addEventListener("click", confirmNewDb);
$("newdb-modal").addEventListener("click", (e) => {
  if (e.target.id === "newdb-modal") closeNewDbModal();   // 点背景关闭
});
$("newdb-path").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); confirmNewDb(); }
});

// initial stats-bar + 库列表 populate
loadStats();
loadDatabases();
