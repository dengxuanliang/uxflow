# UI 修订 v2：✕ 精致化 · 库信息纯文本 · 切换/新建解耦 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修订上一轮 UI：(1) 文件槽 ✕ 做得更小更克制，并给 /style.css /app.js 加版本号防浏览器缓存；(2) 库信息行改为纯文本「【当前库名】中 x 问题 · x 轨迹 · x 切片」，当前库名只显文件名、非下拉；(3) 库下拉只负责「切换到已有库」，「新建库」解耦成独立按钮 → 弹出复用 modal 组件的输入弹窗。

**Architecture:** 纯前端，改 `src/service/web/{index.html,app.js,style.css}`，无后端改动（`/databases/switch` 已能处理「切到不存在路径=新建」）。新建弹窗复用 modal 视觉规范，但因需要一个输入框 + 确认按钮，用一个**独立的 `#newdb-modal`**（`#compile-modal` 结构是只读展示，没有输入控件；硬塞输入不如另建一个轻量弹窗，仍套用 `.modal-overlay/.modal/.modal-head/.modal-body` 样式类保持视觉一致）。

**Tech Stack:** 原生 JS/CSS/HTML。无 JS 测试运行器；验证靠 `tests/service/test_inspector_frontend_static.py` + 手动浏览器冒烟。

**关键背景（已核验）:**
- 上一轮 `.slot-clear` CSS 已是「框内右侧绝对定位小按钮」（style.css:426-436）；用户看到「下方大横条」实为浏览器缓存旧 CSS。故本轮既要视觉再收紧，也要加版本号根治缓存。
- `/databases` 返回 `{current: "<full path>", databases: [{name, path, problems, trajectories, is_current}]}`。`/stats` 返回 `{problems, trajectories, signatures}`。当前库文件名可从 `/databases` 的 `current`（取 basename）或 `databases[].is_current` 项的 `name` 拿到。
- 现有 db 事件绑定在 app.js:878-883（db-select change→switchDatabase；db-switch click→用 db-new-path 值切换；db-clear click→clearDatabase）。这些要改。
- `loadStats()`（app.js:240-248）当前写 `#stats-bar` = "库中 x 问题 · x 轨迹 · x 切片"。要改文案并前置当前库名。
- modal 样式类：`.modal-overlay`（fixed 全屏遮罩）、`.modal`、`.modal-head`、`.modal-close`、`.modal-body`（style.css:321+）。`#compile-modal` 的关闭绑定（modal-close / 背景 / Esc）见 app.js:866-877，只对 `#compile-modal` 生效。

---

## Task 1: ✕ 精致化 + 静态资源版本号防缓存

**Files:**
- Modify: `src/service/web/style.css`（.slot-clear 收紧）
- Modify: `src/service/web/index.html`（/style.css /app.js 加 ?v=2）

- [ ] **Step 1: index.html 给资源加版本号**

`index.html` 第 7 行 `<link rel="stylesheet" href="/style.css">` 改为：
```html
  <link rel="stylesheet" href="/style.css?v=2">
```
第 112 行 `<script src="/app.js"></script>` 改为：
```html
  <script src="/app.js?v=2"></script>
```
（版本号变更会让浏览器重新拉取，根治「改了 CSS/JS 但页面没变」的缓存问题。以后每次前端改动递增此号。）

- [ ] **Step 2: style.css 收紧 .slot-clear**

把现有 `.slot-clear` 规则（style.css:426-434，从 `/* 文件槽取消选择 ✕…` 到 `.slot-clear:hover…`）替换为更小更克制的版本：
```css
/* 文件槽取消选择 ✕：框内右侧小图标，默认淡、悬停才明显 */
.slot-clear {
  position: absolute; top: 50%; right: 6px; transform: translateY(-50%);
  display: none; z-index: 2;
  width: 16px; height: 16px; padding: 0;
  border: none; border-radius: 50%; background: transparent; cursor: pointer;
  color: var(--fg-faint); font-size: 10px; line-height: 16px; text-align: center;
  opacity: .6; transition: opacity .12s, background .12s, color .12s;
}
.file-slot.filled .slot-clear { display: block; }
.slot-clear:hover { opacity: 1; color: var(--hot); background: #cf222e1f; }
```
保留其后的 `.file-slot.filled .slot-name { padding-right: 20px; }`（style.css:436）不变——已选文件时给 ✕ 留空间。

- [ ] **Step 3: 静态测试 + 冒烟**

Run: `uv run pytest -m "not requires_model" tests/service/test_inspector_frontend_static.py -v`
Expected: PASS。

手动（服务在跑）：**Cmd+Shift+R 强刷**，入库栏选文件 → 框内右侧出现一个小小的淡灰 ✕（不是下方横条），悬停变红；点它清空该槽。

- [ ] **Step 4: 提交**
```bash
git add src/service/web/index.html src/service/web/style.css
git commit -m "fix(inspector-web): ✕ 取消按钮收紧为框内小图标 + 静态资源加版本号防缓存

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---
## Task 2: 库信息纯文本 +（下拉切换 / 新建解耦弹窗）

**Files:**
- Modify: `src/service/web/index.html`（db-bar 重构 + 新增 #newdb-modal）
- Modify: `src/service/web/app.js`（loadStats 文案、loadDatabases、新建弹窗逻辑、事件绑定）
- Modify: `src/service/web/style.css`（新建弹窗输入样式，复用 modal 类）

### 目标形态
- **切换控件行**（header 中，db-bar）：`库` 标签 + `#db-select`（**只切换已有库**，选中即切）+ `＋ 新建库` 按钮 + `🗑 清空当前库`。删掉原来的 `#db-new-path` 路径输入框和 `#db-switch` 按钮。
- **库信息行**（下方）：纯文本 `【uxflow.db】中 6 问题 · 64 轨迹 · 1129 切片`，当前库名（只文件名）在方括号里；右侧仍是「最终入选 SFT 样本 N 条」。
- **新建库弹窗**：点「＋ 新建库」弹出 `#newdb-modal`（复用 .modal 样式），一个输入框填库文件名/路径 + 确定/取消，确定后走 `/databases/switch`（后端切到不存在路径=新建空库）。

- [ ] **Step 1: index.html — db-bar 重构 + 新建弹窗 DOM**

把 `#db-bar`（index.html:41-47）改为：
```html
      <div id="db-bar" class="db-bar">
        <span class="db-label">切换库</span>
        <select id="db-select" class="db-select"></select>
        <button id="db-new" class="db-btn" type="button">＋ 新建库</button>
        <button id="db-clear" class="db-btn db-clear" type="button">🗑 清空当前库</button>
      </div>
```
（移除 `#db-new-path` 与 `#db-switch`。）

在文件末尾 `#compile-modal` 那个 modal（index.html:102-110）之后，新增一个新建库弹窗：
```html
  <div id="newdb-modal" class="modal-overlay hidden">
    <div class="modal newdb-modal" role="dialog" aria-modal="true">
      <div class="modal-head">
        <span>新建库</span>
        <button id="newdb-close" class="modal-close" aria-label="关闭">✕</button>
      </div>
      <div class="modal-body">
        <p class="newdb-hint">输入新库文件名（如 <code>demo.db</code>）或完整路径。相对名将建在当前库同目录。</p>
        <input type="text" id="newdb-path" class="newdb-input" placeholder="demo.db" />
        <div class="newdb-actions">
          <button id="newdb-cancel" class="db-btn" type="button">取消</button>
          <button id="newdb-confirm" class="db-btn db-primary" type="button">新建并切换</button>
        </div>
      </div>
    </div>
  </div>
```

- [ ] **Step 2: app.js — loadStats 文案改「【当前库名】中 …」**

现状 `loadStats()`（app.js:240-248）写死「库中 …」。当前库名要从 `/databases` 拿。改为让 `loadStats` 接受可选库名，或在 `afterDbChange`/初始化时用 `/databases` 的 `current` 派生。最简单：`loadStats` 内并行取当前库名。替换 `loadStats`：
```javascript
async function loadStats() {
  try {
    const resp = await fetch("/stats");
    if (!resp.ok) return;
    const s = await resp.json();
    const dbName = state.currentDbName || "当前库";
    $("stats-bar").textContent =
      `【${dbName}】中 ${s.problems} 问题 · ${s.trajectories} 轨迹 · ${s.signatures} 切片`;
  } catch (_) { /* stats are best-effort */ }
}
```
在 `state` 对象初始化处加一个字段 `currentDbName: null`（读 app.js 顶部 `state = {...}` 定义，加入该键）。`loadDatabases` 负责填它（见下）。

- [ ] **Step 3: app.js — loadDatabases 记录当前库名 + 下拉只列切换项**

替换 `loadDatabases()`（app.js:263-279）：
```javascript
async function loadDatabases() {
  try {
    const resp = await fetch("/databases");
    if (!resp.ok) { $("db-bar").classList.add("hidden"); return; }
    const { current, databases } = await resp.json();
    // 当前库文件名（纯文本，供库信息行显示）
    state.currentDbName = (current || "").split("/").pop() || "当前库";
    const sel = $("db-select");
    sel.innerHTML = "";
    for (const d of databases) {
      const opt = document.createElement("option");
      opt.value = d.path;
      const cnt = d.problems == null ? "?" : d.problems;
      opt.textContent = `${d.name}（${cnt} 问题）` + (d.is_current ? " · 当前" : "");
      if (d.is_current) opt.selected = true;
      sel.appendChild(opt);
    }
    loadStats();   // 库名可能已更新，刷新信息行
  } catch (_) { /* best-effort */ }
}
```

- [ ] **Step 4: app.js — 新建弹窗逻辑**

在库管理区（clearDatabase 附近）新增：
```javascript
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
  closeNewDbModal();
  await switchDatabase(p);   // 复用切换；不存在路径→后端建空库并切换
}
```

- [ ] **Step 5: app.js — 更新事件绑定**

现有绑定（app.js:878-883）：
```javascript
$("db-select").addEventListener("change", (e) => switchDatabase(e.target.value));
$("db-switch").addEventListener("click", () => {
  const p = $("db-new-path").value.trim();
  if (p) switchDatabase(p);
});
$("db-clear").addEventListener("click", clearDatabase);
```
替换为（移除 db-switch/db-new-path 引用，加新建弹窗绑定）：
```javascript
$("db-select").addEventListener("change", (e) => switchDatabase(e.target.value));
$("db-new").addEventListener("click", openNewDbModal);
$("db-clear").addEventListener("click", clearDatabase);
$("newdb-close").addEventListener("click", closeNewDbModal);
$("newdb-cancel").addEventListener("click", closeNewDbModal);
$("newdb-confirm").addEventListener("click", confirmNewDb);
$("newdb-modal").addEventListener("click", (e) => {
  if (e.target.id === "newdb-modal") closeNewDbModal();   // 点背景关闭
});
$("newdb-path").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); confirmNewDb(); }
});
```
另外把全局 Esc 关闭也覆盖新建弹窗：读 app.js 里现有的 `document.addEventListener("keydown", (e) => { if (e.key === "Escape") closeCompileModal(); })`（约 875-877），改为也关新建弹窗：
```javascript
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { closeCompileModal(); closeNewDbModal(); }
});
```

- [ ] **Step 6: style.css — 新建弹窗内容样式**

追加：
```css
/* 新建库弹窗：套用 .modal 规范，补输入区样式 */
.newdb-modal { width: min(440px, 92vw); }
.newdb-hint { margin: 0 0 10px; font-size: 12px; color: var(--fg-dim); line-height: 1.5; }
.newdb-hint code { font-family: var(--mono); background: var(--bg-2); padding: 0 4px; border-radius: 3px; }
.newdb-input {
  width: 100%; box-sizing: border-box; font-family: var(--mono); font-size: 13px;
  padding: 7px 10px; border: 1px solid var(--line); border-radius: 6px; color: var(--fg);
}
.newdb-input:focus { outline: none; border-color: var(--neon); box-shadow: 0 0 0 2px #0969da33; }
.newdb-actions { display: flex; justify-content: flex-end; gap: 8px; margin-top: 14px; }
.db-primary { background: var(--neon); color: #fff; border-color: var(--neon); }
.db-primary:hover { background: var(--neon-2); }
```

- [ ] **Step 7: 静态测试 + 冒烟**

Run: `uv run pytest -m "not requires_model" tests/service/test_inspector_frontend_static.py -v`
Then: `uv run pytest -m "not requires_model" tests/service -q`
Expected: PASS。（若静态测试断言了 `db-new-path`/`db-switch` 这些被删的 id，同步；若断言 `db-select`/`stats-bar` 存在，仍在。）

手动（服务在跑，强刷）：
- 库信息行显示「【uxflow.db】中 6 问题 · 64 轨迹 · 1129 切片」，右侧「最终入选…」。
- 库下拉列出所有 .db，选另一个 → 直接切换（信息行库名随之变）。
- 点「＋ 新建库」→ 弹窗出现，输 `demo.db` → 「新建并切换」→ 切到空库（信息行变 demo.db，问题数 0）；背景点击 / Esc / 取消都能关弹窗。
- 「🗑 清空当前库」仍工作（在 demo.db 上测，别在主库）。

- [ ] **Step 8: 提交**
```bash
git add src/service/web/index.html src/service/web/app.js src/service/web/style.css tests/service/test_inspector_frontend_static.py
git commit -m "feat(inspector-web): 库信息改纯文本【当前库】+ 下拉专切换、新建库解耦为 modal 弹窗

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## 验收标准

**Task 1（✕ + 缓存）:**
- [ ] ✕ 是文件框内右侧的小圆图标（16px），默认淡灰、悬停变红，非下方横条。
- [ ] /style.css /app.js 带 ?v=2，强刷后新样式生效。

**Task 2（库信息 + 解耦）:**
- [ ] 库信息行是纯文本「【<当前库文件名>】中 x 问题 · x 轨迹 · x 切片」，库名非下拉。
- [ ] 下拉只切换已有库；新建库是独立「＋ 新建库」按钮 → modal 弹窗输入路径。
- [ ] 新建弹窗可 确定/取消/背景/Esc 关闭；确定后新建并切换。
- [ ] 删掉的 `#db-new-path`/`#db-switch` 无残留引用（grep app.js 无 `db-new-path`/`db-switch`）。

**整体:**
- [ ] `uv run pytest -m "not requires_model" tests/service` 全绿。

