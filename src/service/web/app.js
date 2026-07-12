// Trajectory Inspector frontend (spec §9, decisions 10-12).
const $ = (id) => document.getElementById(id);

let state = {
  runId: null,
  view: null,
  activeProblem: null,   // problem id
  activeCap: null,       // focused capability label
  showAll: false,        // "显示全部能力" toggle
  activeTraj: null,      // {trajectory_id, slice_index}
  trajCache: {},
};

$("run").addEventListener("click", startRun);
$("show-all").addEventListener("change", (e) => {
  state.showAll = e.target.checked;
  renderDetail();
  renderProblems();
});

async function startRun() {
  const mf = $("manifest").files[0];
  const tj = $("trajectories").files[0];
  if (!mf || !tj) { alert("请先上传清单和轨迹两个文件"); return; }
  const fd = new FormData();
  fd.append("manifest", mf);
  fd.append("trajectories", tj);

  $("progress").classList.remove("hidden");
  $("workspace").classList.add("hidden");
  $("progress-msg").textContent = "上传中…";

  const resp = await fetch("/runs", { method: "POST", body: fd });
  const { run_id } = await resp.json();
  state.runId = run_id;
  subscribeEvents(run_id);
}

function subscribeEvents(runId) {
  const es = new EventSource(`/runs/${runId}/events`);
  es.onmessage = async (e) => {
    const ev = JSON.parse(e.data);
    $("progress-msg").textContent = `[${ev.stage}] ${ev.msg || ev.status}`;
    if (ev.stage === "done") {
      es.close();
      if (ev.status === "error") {
        $("progress-msg").textContent = "运行出错: " + (ev.msg || "");
        return;
      }
      await loadView(runId);
    }
  };
  es.onerror = () => { es.close(); };
}

async function loadView(runId) {
  const resp = await fetch(`/runs/${runId}/view`);
  state.view = await resp.json();
  $("progress").classList.add("hidden");
  $("workspace").classList.remove("hidden");
  renderManifest();
  renderProblems();
  $("hits").innerHTML = "";
  $("detail").innerHTML = "";
}

function renderManifest() {
  const m = state.view.manifest || {};
  $("manifest-summary").textContent =
    `最终选集: targeted=${m.targeted_count} · general=${m.general_count} · ratio=${m.general_ratio}`;
}

function renderProblems() {
  const el = $("problems");
  el.innerHTML = "";
  for (const p of state.view.problems) {
    const pd = document.createElement("div");
    pd.className = "problem" + (p.id === state.activeProblem ? " active" : "");
    pd.textContent = `❗ ${p.failure_summary} (${p.confidence.toFixed(2)})`;
    pd.onclick = () => selectProblem(p.id);
    el.appendChild(pd);

    if (p.id === state.activeProblem) {
      for (const cap of p.capabilities) {
        const cd = document.createElement("div");
        const focused = cap.label === state.activeCap;
        const dimmed = state.activeCap && !focused && !state.showAll;
        cd.className = "capability" + (focused ? " focused" : "") + (dimmed ? " dimmed" : "");
        cd.style.color = cap.color;
        cd.style.background = cap.color + "22";
        cd.textContent = `${cap.label} (${cap.hit_count})`;
        cd.onclick = () => focusCapability(p.id, cap.label);
        el.appendChild(cd);
      }
    }
  }
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
  const caps = state.activeCap && !state.showAll
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

function renderHits() {
  const el = $("hits");
  el.innerHTML = "";
  const hits = currentHits();
  for (const h of hits) {
    const hd = document.createElement("div");
    const isActive = state.activeTraj && state.activeTraj.trajectory_id === h.trajectory_id
      && state.activeTraj.slice_index === h.slice_index;
    hd.className = "hit" + (isActive ? " active" : "");
    const badge = h.selected ? '<span class="selected-badge">已入选</span>' : "";
    hd.innerHTML = `${h.trajectory_id} · slice${h.slice_index} · ${h.caps.length}片段 ${badge}`;
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

async function ensureTrajectory(tid) {
  if (state.trajCache[tid]) return;
  const resp = await fetch(`/runs/${state.runId}/trajectory/${tid}`);
  state.trajCache[tid] = await resp.json();
}

// Which capabilities' spans should highlight this step (focus vs show-all)
function spansForStep(stepIndex, hit) {
  const out = [];
  for (const cap of hit.caps) {
    const inSpan = (cap.spans || []).some(
      (s) => stepIndex >= s.start_step && stepIndex <= s.end_step);
    if (!inSpan) continue;
    const focused = !state.activeCap || state.showAll || cap.label === state.activeCap;
    out.push({ ...cap, focused });
  }
  return out;
}

function renderDetail() {
  const el = $("detail");
  el.innerHTML = "";
  if (!state.activeTraj) return;
  const traj = state.trajCache[state.activeTraj.trajectory_id];
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
      cls += " hit-span";
      sd.style.borderLeftColor = focusedCaps[0].color;
      sd.style.background = focusedCaps[0].color + "22";
    } else if (dimCaps.length) {
      cls += " hit-dim";  // 非聚焦能力命中：淡显（decision 11）
    }
    sd.className = cls;

    const roleIcon = { user: "👤", assistant: "🤖", tool: "⚙️", system: "⚙" }[step.role] || "";
    let text = `${roleIcon} ${step.role}`;
    if (step.tool_call_name) text += ` 🔧 ${step.tool_call_name}`;
    const body = (step.content || step.tool_result || "").slice(0, 300);
    sd.textContent = `${text}: ${body}`;

    // 小色标签：命中该 step 的所有能力（含非聚焦，decision 11）
    for (const c of caps) {
      const tag = document.createElement("span");
      tag.className = "cap-tag";
      tag.style.background = c.color;
      tag.textContent = c.label;
      if (!c.focused) tag.style.opacity = ".6";
      sd.appendChild(tag);
    }
    el.appendChild(sd);
  }
}
