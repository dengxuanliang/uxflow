# 证据折叠（行内 accordion）Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把命中卡上直接内联渲染的"决定性证据/命中判据"改为默认收起、点"证据"按钮就地展开的行内 accordion。

**Architecture:** 纯前端。改 `src/service/web/app.js` 的 `renderHits()`（不再内联证据行，改渲染一个"证据"按钮 + 一个默认隐藏的证据面板），加一个 toggle 逻辑；`src/service/web/style.css` 加折叠面板样式。`index.html` 不动。多归属 slice 的证据按子问题分组显示——数据已在 hit 对象的 `caps` 里（每个 cap 带 label/color，证据字段 `evidence_step`/`criteria_hit` 在 hit 顶层，源自 `scored`）。

**Tech Stack:** 原生 JS（无框架）、CSS。前端无自动化测试运行器；验证靠 `tests/service/test_inspector_frontend_static.py`（静态断言）+ 手动冒烟。

**参考 spec:** `docs/superpowers/specs/2026-07-23-dedup-evidence-db-management-design.md` §6。

---

## 背景：现状代码

`app.js` 的 `renderHits()`（约 514-546 行）当前对每个命中 `h` 这样渲染证据（要删除的部分）：

```javascript
// 可回溯性(PR-3):judge 定位的决定性证据步 + 命中的 rubric 判据。
if (h.evidence_step != null || (h.criteria_hit && h.criteria_hit.length)) {
  const ev = document.createElement("div");
  ev.className = "hit-evidence";
  const parts = [];
  if (h.evidence_step != null) parts.push(`⭐ 决定性证据: step ${h.evidence_step}`);
  if (h.criteria_hit && h.criteria_hit.length) {
    parts.push(`命中判据: ${h.criteria_hit.map(escapeHtml).join(" / ")}`);
  }
  ev.innerHTML = parts.join(" · ");
  hd.appendChild(ev);
}
```

`h` 的结构（来自 `currentHits()`）：`{trajectory_id, slice_index, selected, evidence_step, criteria_hit, loss_mask_spans, caps: [{label, color, spans}]}`。注意 `evidence_step`/`criteria_hit` 在 hit 顶层（viewmodel 按 sub_problem 建 hit，一个 sub_problem 一个 hit 对象）。

---

## 前置：确认前端静态测试现状

- [ ] **Step 1: 读现有前端静态测试，确认它断言了什么**

Run: `sed -n '1,80p' tests/service/test_inspector_frontend_static.py`
Expected: 看清它是否断言 `hit-evidence` 或按钮相关 DOM。若它只校验静态文件存在/引用，则本计划无需改它；若它 grep 了 `hit-evidence` 字符串，Task 3 要同步。

---

## Task 1: renderHits 改为渲染"证据"按钮 + 隐藏面板

**Files:**
- Modify: `src/service/web/app.js`（`renderHits()` 内证据渲染段）

- [ ] **Step 1: 替换证据内联段为按钮+面板**

在 `renderHits()` 中，把上面"背景"里引用的整段 `if (h.evidence_step != null ...)` 替换为：

```javascript
// 证据折叠：默认收起，点"证据"按钮就地展开（spec §6）。
const hasEvidence = h.evidence_step != null
  || (h.criteria_hit && h.criteria_hit.length);
if (hasEvidence) {
  const btn = document.createElement("button");
  btn.className = "evidence-toggle";
  btn.type = "button";
  btn.textContent = "证据";
  btn.setAttribute("aria-expanded", "false");

  const panel = document.createElement("div");
  panel.className = "evidence-panel hidden";
  panel.innerHTML = renderEvidencePanel(h);

  btn.onclick = (e) => {
    e.stopPropagation();   // 不触发 selectTrajectory
    const open = panel.classList.toggle("hidden") === false;
    btn.setAttribute("aria-expanded", String(open));
  };

  hd.appendChild(btn);
  hd.appendChild(panel);
}
```

- [ ] **Step 2: 新增 `renderEvidencePanel(h)` 辅助函数**

在 `app.js` 里 `renderHits` 附近（`// ── utils` 区块之前）新增：

```javascript
// 证据面板内容：决定性证据步 + 命中判据。多归属（一个 slice 命中多个能力/
// 子问题）时按能力分组列出能力名，证据步/判据来自该 hit（按 sub_problem 聚合）。
function renderEvidencePanel(h) {
  const rows = [];
  if (h.evidence_step != null) {
    rows.push(`<div class="ev-row"><span class="ev-k">决定性证据</span>` +
      `<span class="ev-v">⭐ step ${h.evidence_step}</span></div>`);
  }
  if (h.criteria_hit && h.criteria_hit.length) {
    rows.push(`<div class="ev-row"><span class="ev-k">命中判据</span>` +
      `<span class="ev-v">${h.criteria_hit.map(escapeHtml).join(" / ")}</span></div>`);
  }
  const caps = (h.caps || []).map((c) =>
    `<span class="ev-cap" style="background:${c.color}">${escapeHtml(c.label)}</span>`
  ).join("");
  if (caps) {
    rows.push(`<div class="ev-row"><span class="ev-k">覆盖能力</span>` +
      `<span class="ev-v">${caps}</span></div>`);
  }
  return rows.join("");
}
```

- [ ] **Step 3: 提交前先跑既有 Python 测试保证没误伤**

Run: `uv run pytest -m "not requires_model" tests/service/test_inspector_frontend_static.py -v`
Expected: PASS（若失败见 Task 3）。

---

## Task 2: 折叠面板样式

**Files:**
- Modify: `src/service/web/style.css`（追加到文件末尾）

- [ ] **Step 1: 追加证据按钮 + 面板样式**

在 `style.css` 末尾追加：

```css
/* ── 证据折叠（行内 accordion，spec §6）───────────────── */
.evidence-toggle {
  margin-top: 4px;
  font-size: 11px;
  padding: 1px 8px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: var(--bg-2);
  color: var(--fg-dim);
  cursor: pointer;
}
.evidence-toggle:hover { background: var(--bg-3); }
.evidence-toggle[aria-expanded="true"] {
  color: var(--neon);
  border-color: var(--neon);
}
.evidence-panel {
  margin-top: 6px;
  padding: 8px 10px;
  border-left: 2px solid var(--gold);
  background: var(--bg-2);
  border-radius: 4px;
  font-size: 12px;
}
.evidence-panel .ev-row {
  display: flex;
  gap: 8px;
  margin: 2px 0;
  align-items: baseline;
}
.evidence-panel .ev-k {
  flex: 0 0 64px;
  color: var(--fg-faint);
}
.evidence-panel .ev-v { flex: 1; color: var(--fg); }
.evidence-panel .ev-cap {
  display: inline-block;
  color: #fff;
  border-radius: 3px;
  padding: 0 6px;
  margin-right: 4px;
  font-size: 11px;
}
```

- [ ] **Step 2: 手动冒烟（需可运行 LLM 后端）**

若环境可启动服务：
Run: `.venv/bin/python scripts/inspector_serve.py`（另开终端），浏览器打开 `http://localhost:8000`，跑一次 search/run，选一个有证据的命中卡。
Expected: 卡上出现"证据"按钮，默认不显示证据文本；点击就地展开决定性证据步+命中判据+覆盖能力标签；再点收起；点按钮不会选中/切换该轨迹到详情栏。

若环境无 LLM 后端，跳过此步，靠 Task 3 静态测试 + code review。

---

## Task 3: 同步前端静态测试（若需要）

**Files:**
- Modify: `tests/service/test_inspector_frontend_static.py`（仅当 Step 1 探查发现它断言了旧 `hit-evidence` 字符串）

- [ ] **Step 1: 判断是否需要改**

若"前置探查"发现该测试 grep 了 `hit-evidence` 或内联证据文本，把断言改为校验新标识（如 `evidence-toggle` / `renderEvidencePanel` 存在于 `app.js`）。若测试只校验文件存在/挂载，跳过本 Task。

示例改法（仅当原本断言旧字符串时）：

```python
def test_app_js_uses_evidence_toggle():
    app_js = (WEB_DIR / "app.js").read_text(encoding="utf-8")
    assert "evidence-toggle" in app_js
    assert "renderEvidencePanel" in app_js
```

- [ ] **Step 2: 跑该测试**

Run: `uv run pytest -m "not requires_model" tests/service/test_inspector_frontend_static.py -v`
Expected: PASS

---

## Task 4: 提交

- [ ] **Step 1: 提交**

```bash
git add src/service/web/app.js src/service/web/style.css tests/service/test_inspector_frontend_static.py
git commit -m "feat(inspector-web): 决定性证据改为可折叠按钮（行内 accordion）

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## 验收标准（对齐 spec §10）

- [ ] 命中卡默认不显示证据文本；有 evidence/criteria 时出现"证据"按钮。
- [ ] 点按钮就地展开/收起，不触发选中轨迹（`stopPropagation` 生效）。
- [ ] 多归属 slice 的证据在面板内以覆盖能力标签分组呈现。
- [ ] `tests/service/test_inspector_frontend_static.py` 全绿。
