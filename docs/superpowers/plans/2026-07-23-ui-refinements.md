# UI 改造：证据弹窗 · 文件取消选择 · header 重排 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 三处前端 UI 改造：(1) "命中轨迹"栏的"证据"按钮改为弹小窗（复用"详情"的 modal 组件）；(2) 入库栏两个文件槽加浅灰 ✕ 取消误选；(3) header 重排——切换控件在最上一行，库信息行居中（左"库中…"、右"最终入选…"）。

**Architecture:** 纯前端，改 `src/service/web/{index.html,app.js,style.css}`，无后端改动。证据弹窗复用现有 `#compile-modal` 容器（背景遮罩/关闭/Esc 全现成）。文件槽 ✕ 复用 `clearIngestInputs` 的清空逻辑抽成 `clearFileSlot(id)`。header 把 `#manifest-summary` 从 `.modes` 外移入一个新的库信息行，与 `#stats-bar` 同排。

**Tech Stack:** 原生 JS/CSS/HTML，无框架。无 JS 测试运行器；验证靠 `tests/service/test_inspector_frontend_static.py`（静态断言）+ 手动浏览器冒烟。

**参考:** 前置对话已定三处决策——证据弹窗复用同一组件；✕ 仅清空该文件；header 方案A（切换行横铺 → 库信息行左右分布）。

---

## 背景：现状代码定位

- `app.js` `renderHits()`（约 582-617）：命中卡内建 `evidence-toggle` 按钮 + 就地展开的 `evidence-panel`（accordion）。
- `app.js` `renderEvidencePanel(h)`（约 786-805）：产出证据 HTML（决定性证据步 / 命中判据 / 覆盖能力），复用。
- `app.js` `openCompileModal(p)`（约 822）/ `closeCompileModal()`（约 861）：详情弹窗，写 `#modal-title` + `#modal-body`，用 `.mf-row/.mf-k/.mf-v` 结构。`#compile-modal` 背景点击 + Esc 关闭已绑定（约 866-877）。
- `app.js` `clearIngestInputs()`（214-224）：遍历两个槽清空 `input.value` + 复位 `.slot-name` + 移除 `.filled`。
- `app.js` `renderManifest()`（488-492）：写 `#manifest-summary` = "最终入选 SFT 样本 N 条"。
- `index.html`：`.modes` 内含 `.mode-bar`（搜索+入库）、`#stats-bar`、`#db-bar`；`#manifest-summary` 在 `.modes` **外**（header 内，第 48 行）。
- `index.html` 文件槽（24-33）：`<label class="file-slot">` 包 `.slot-tag` + `.slot-name` + 隐藏 `<input type=file>`。
- `style.css`：`.file-slot`（53-60，`position:relative`）、`.file-slot.filled`（62）、`.slot-name`（66）、`.stats-bar`（126-128）、`.db-bar`（409+）、`.modal*`（314+）、`.mf-row/.mf-k/.mf-v`（337-343）。

---

## 前置：确认前端静态测试断言了什么

- [ ] **Step 0: 读 tests/service/test_inspector_frontend_static.py**

Run: `sed -n '1,80p' tests/service/test_inspector_frontend_static.py`
记住它断言的元素/字符串。若它 grep 了 `evidence-toggle` / `db-bar` / `manifest-summary` 等本计划会动的标识，相应 Task 末尾要同步；若只校验文件存在/挂载，则无需改。

---

## Task 1: 证据按钮改弹窗（复用 #compile-modal）

**Files:**
- Modify: `src/service/web/app.js`（renderHits 证据段、新增 openEvidenceModal）
- Modify: `src/service/web/style.css`（证据在 modal-body 内的样式，若需要）

- [ ] **Step 1: renderHits 里证据按钮改为开弹窗**

在 `renderHits()` 中，把现有 accordion 段（`const hasEvidence …` 到 `hd.appendChild(panel);` 整块，约 591-613）替换为：

```javascript
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
```

- [ ] **Step 2: 新增 openEvidenceModal(h)，复用 #compile-modal 容器**

在 `openCompileModal` 附近（同一 modal 区块）新增。用与详情一致的 `.mf-row` 结构，视觉统一：

```javascript
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
```

- [ ] **Step 3: 删除 renderEvidencePanel（已无引用）**

`renderEvidencePanel(h)`（约 786-805）现在没有调用方（accordion 已删）。删除该函数，避免死代码。（若 Step 0 的静态测试断言了它的名字，改测试而非留死码。）

- [ ] **Step 4: CSS — 确保 .ev-cap 在 modal-body 内正常显示**

`renderEvidencePanel` 删了，但 `.ev-cap`（能力色标签）仍被 openEvidenceModal 用。检查 style.css 现有 `.evidence-panel .ev-cap` 规则（在证据折叠那次提交追加的）——它是后代选择器，脱离 `.evidence-panel` 就失效。把 `.ev-cap` 提为独立规则：

在 style.css 末尾（或替换旧的 `.evidence-panel .ev-cap` 块）加：
```css
/* 证据弹窗内的能力色标签（脱离原 accordion，独立选择器）*/
.mf-v .ev-cap {
  display: inline-block;
  color: #fff;
  border-radius: 3px;
  padding: 0 6px;
  margin-right: 4px;
  font-size: 11px;
}
```
同时删除已无用的 accordion 面板样式：`.evidence-panel`、`.evidence-panel .ev-row`、`.evidence-panel .ev-k`、`.evidence-panel .ev-v`、`.evidence-panel .ev-cap`、以及 `.evidence-toggle[aria-expanded=...]`（不再有 aria-expanded 状态）。保留 `.evidence-toggle` 基础按钮样式（去掉 aria-expanded 相关那条）。读 style.css 找到证据折叠那段（`/* ── 证据折叠…`）精确删改。

- [ ] **Step 5: 冒烟 + 静态测试**

Run: `uv run pytest -m "not requires_model" tests/service/test_inspector_frontend_static.py -v`
Expected: PASS（若断言旧 `renderEvidencePanel`/`evidence-panel` 则同步）。

手动（若服务在跑）：搜索一个复合问题 → 命中卡点"证据" → 弹出小窗显示决定性证据/命中判据/覆盖能力，点背景或 ✕ 或 Esc 关闭，点"证据"不选中该轨迹到详情栏。

- [ ] **Step 6: 提交**
```bash
git add src/service/web/app.js src/service/web/style.css tests/service/test_inspector_frontend_static.py
git commit -m "feat(inspector-web): 证据按钮改弹窗（复用详情 modal 组件）

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 2: 入库文件槽加浅灰 ✕ 取消选择

**Files:**
- Modify: `src/service/web/index.html`（两个 file-slot 加 ✕ 按钮）
- Modify: `src/service/web/app.js`（clearFileSlot + 绑定，clearIngestInputs 复用）
- Modify: `src/service/web/style.css`（.slot-clear 样式）

- [ ] **Step 1: index.html 两个文件槽加 ✕ 按钮**

在 `#slot-manifest`（24-28）和 `#slot-trajectories`（29-33）的 `<input type="file">` 之后、`</label>` 之前，各加一个 ✕。manifest 槽：
```html
      <label class="file-slot" id="slot-manifest">
        <span class="slot-tag">用户清单 <em>.txt / .jsonl</em></span>
        <span class="slot-name" id="name-manifest">点击选择文件…</span>
        <input type="file" id="manifest" accept=".txt,.jsonl">
        <button type="button" class="slot-clear" id="clear-manifest" aria-label="取消选择">✕</button>
      </label>
```
trajectories 槽同理，`id="clear-trajectories"`。

- [ ] **Step 2: app.js — 抽 clearFileSlot(id) + 绑定 ✕**

把现有 `clearIngestInputs()`（214-224）改为复用一个单槽清空函数：
```javascript
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
```

在现有文件槽 change 监听（64-74）之后，绑定两个 ✕ 按钮：
```javascript
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
```

- [ ] **Step 3: style.css — .slot-clear（仅 .filled 时显示）**

`.file-slot` 已是 `position:relative`。追加：
```css
/* 文件槽取消选择 ✕：仅在已选文件（.filled）时显示，浅灰、悬停加深 */
.slot-clear {
  position: absolute; top: 50%; right: 8px; transform: translateY(-50%);
  display: none; z-index: 1;
  border: none; background: transparent; cursor: pointer;
  color: var(--fg-faint); font-size: 13px; line-height: 1;
  padding: 2px 4px; border-radius: 4px;
}
.file-slot.filled .slot-clear { display: block; }
.slot-clear:hover { color: var(--hot); background: #cf222e14; }
```
注意：`.slot-name` 需给右侧留出空间避免和 ✕ 重叠——若文件名过长，加 `padding-right`。检查 `.slot-name`（66 行）现状后，给 `.file-slot.filled .slot-name` 补 `padding-right: 20px;`（仅 filled 时，因为 ✕ 只在那时出现）。

- [ ] **Step 4: 冒烟 + 静态测试**

Run: `uv run pytest -m "not requires_model" tests/service/test_inspector_frontend_static.py -v`
Expected: PASS。

手动（若服务在跑）：入库栏选一个文件 → 出现浅灰 ✕ → 点 ✕ → 文件名回到"点击选择文件…"、绿框消失、✕ 消失；此时点"入库"不会带上该文件。点 ✕ 不会弹出文件选择框（stopPropagation 生效）。

- [ ] **Step 5: 提交**
```bash
git add src/service/web/index.html src/service/web/app.js src/service/web/style.css
git commit -m "feat(inspector-web): 文件槽加 ✕ 取消误选（阻止冒泡不重开选择框）

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## Task 3: header 三行重排（方案A）

**Files:**
- Modify: `src/service/web/index.html`（.modes 内结构重排，manifest-summary 移入）
- Modify: `src/service/web/style.css`（.db-info-row 布局）

目标布局（方案A）：
```
[.mode-bar 搜索/入库]                         ← 不变
[#db-bar 切换控件：下拉 + 路径 + 切换 + 清空]   ← 上移到 stats 之上
[库中 6问题·64轨迹·1129切片   ⟷   最终入选 SFT 样本 10 条]  ← 新库信息行，左右分布
```

- [ ] **Step 1: index.html — 重排 .modes，manifest-summary 移入库信息行**

当前 `.modes`（16-47）内顺序：`.mode-bar` → `#stats-bar` → `#db-bar`；`#manifest-summary`（48）在 `.modes` 外。改为 `.mode-bar` → `#db-bar` → 新增 `.db-info-row`（含 `#stats-bar` + `#manifest-summary`），并删除 `.modes` 外那行 manifest-summary。

把第 39-48 行区域改为：
```html
      <div id="db-bar" class="db-bar">
        <span class="db-label">库</span>
        <select id="db-select" class="db-select"></select>
        <input type="text" id="db-new-path" class="db-new-path" placeholder="新建/切换到路径…" />
        <button id="db-switch" class="db-btn" type="button">切换</button>
        <button id="db-clear" class="db-btn db-clear" type="button">🗑 清空当前库</button>
      </div>
      <div class="db-info-row">
        <div id="stats-bar" class="stats-bar"></div>
        <div id="manifest-summary" class="manifest"></div>
      </div>
    </div>
```
即：`#db-bar` 移到 `#stats-bar` 之前；`#stats-bar` 与 `#manifest-summary` 一起包进 `.db-info-row`；原第 48 行 `.modes` 外的 `<div id="manifest-summary" class="manifest"></div>` 删除（已移入）。确认 `.modes` 的闭合 `</div>` 位置正确（`.db-info-row` 在 `.modes` 内）。

- [ ] **Step 2: style.css — .db-info-row 左右分布**

追加：
```css
/* 库信息行：左"库中…"，右"最终入选…"，同排左右分布 */
.db-info-row {
  display: flex; align-items: baseline; justify-content: space-between;
  gap: 16px; flex-wrap: wrap;
}
```
`.manifest`（89-93）现状是独立成行的样式（可能带 margin/padding 撑开）。读它当前值后，在库信息行内它应与 stats 基线对齐——若 `.manifest` 有 `margin` 撑开换行，在 `.db-info-row .manifest` 覆盖为 `margin: 0`。保留 `.manifest b { color:#fff }`（入选数加粗高亮）。

- [ ] **Step 3: 确认 renderManifest 仍写 #manifest-summary（无需改 JS）**

`renderManifest()`（488-492）写 `$("manifest-summary")`，元素 id 不变，只是位置变了 → JS 无需改。确认 `#manifest-summary` 在搜索/运行前为空（不显示"最终入选"），运行完 `renderManifest()` 填入。

- [ ] **Step 4: 冒烟 + 静态测试**

Run: `uv run pytest -m "not requires_model" tests/service/test_inspector_frontend_static.py -v`
Expected: PASS（若断言了 stats-bar/manifest-summary 的位置或 db-bar 顺序，同步）。

手动（若服务在跑）：header 从上到下为 搜索/入库行 → 切换控件行（下拉+路径+切换+清空）→ 库信息行（左"库中…"、右"最终入选…"）。搜索完"最终入选 N 条"出现在库信息行右侧。切库后库信息行左侧刷新。

- [ ] **Step 5: 提交**
```bash
git add src/service/web/index.html src/service/web/style.css
git commit -m "feat(inspector-web): header 重排——切换行置顶，库信息与入选数同排左右分布

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## 验收标准

**Task 1 证据弹窗:**
- [ ] 命中卡"证据"按钮点击弹出小窗（同"详情"的 modal 视觉），显示决定性证据步/命中判据/覆盖能力。
- [ ] 弹窗可用背景点击 / ✕ / Esc 关闭。
- [ ] 点"证据"不触发选中该轨迹到详情栏（stopPropagation）。
- [ ] 无死代码（renderEvidencePanel 及 accordion 样式已清）。

**Task 2 文件取消选择:**
- [ ] 选文件后文件槽出现浅灰 ✕。
- [ ] 点 ✕ 清空该槽（名字复位、绿框消失），另一槽不受影响。
- [ ] 点 ✕ 不重新打开文件选择框（阻止冒泡）。
- [ ] 清空后点"入库"不会带上该文件。

**Task 3 header 重排:**
- [ ] 切换控件在库信息行**上方**独占一行。
- [ ] "库中 x问题·x轨迹·x切片"在库信息行**左**，"最终入选 SFT 样本 x条"在**右**，同一行。
- [ ] 下拉菜单保留（在切换行内），非在库信息行。

**整体:**
- [ ] `uv run pytest -m "not requires_model" tests/service` 全绿。
