# UI 修订 v4：header 左右分块 + 切换改弹窗 + 三按钮统一尺寸 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 重排 header 主区为左右两块：左块=搜索（输入框+按钮，垂直拉高与右块等高）；右块两行=右上入库组、右下「切换/新建/清空」三个等尺寸按钮。切换从内联输入框改为纯按钮+弹窗选库。

**Architecture:** 纯前端，改 `src/service/web/{index.html,app.js,style.css}`，无后端改动。切换新增 `#switch-modal`（仿 `#newdb-modal`），内含 `<select id="switch-select">` 列所有 .db + 确定按钮，走已有 `switchDatabase()`。移除上一版的内联切换组（`#db-switch-path`/`#db-list`/`#db-switch-clear`/`#db-switch-btn` + `switchFromInput`/`clearSwitchInput`）。三个库操作按钮统一 `.menu-btn` 尺寸（仿 `#btn-search`/`#btn-ingest` 的 40px 高）。`?v=3`→`?v=4`。库信息行不变。

**Tech Stack:** 原生 JS/CSS/HTML。无 JS 测试运行器；验证靠 `tests/service/test_inspector_frontend_static.py` + 手动浏览器冒烟。

**关键背景（已核验）:**
- 现状 `.mode-bar`（index.html:17-55）：`.mode-search` | divider | `.mode-ingest` | divider | `.mode-switch`(内联输入框组) | divider | `.mode-dbops`(新建+清空)。要改成左右两大块。
- `.mode-search`（style.css:100 `flex:1 1 320px`）、`#question`（103-108，40px 高）、`#btn-search`/`#btn-ingest`（113-124，40px 高、圆角、不同色）。
- 文件槽 `.file-slot`（40px 高）+ `.slot-clear` ✕（文件槽内，`.filled` 时显示）——入库组不动。
- db 事件绑定 app.js:898-910（v3）：`db-switch-btn`/`db-switch-path`(Enter)/`db-switch-clear`/`db-new`/`db-clear` + newdb-* 组。要改：删 switch-* 内联三项，加 switch-modal 相关。
- `switchDatabase(path)`（app.js:286-303）成功 true / 失败 alert+false，保留复用。`openNewDbModal/closeNewDbModal/confirmNewDb`（331-345）+ `#newdb-modal`（index.html:120-135）保留不动。
- `clearDatabase`（317-329）二次确认，保留。`loadDatabases`（265-284）现填 `#db-list` datalist——要改填 `#switch-select`。
- `#run` 按钮（隐藏，app.js 有绑定依赖）保留。

---

## 前置：确认静态测试断言

- [ ] **Step 0: 读 tests/service/test_inspector_frontend_static.py**

Run: `sed -n '1,80p' tests/service/test_inspector_frontend_static.py`
若断言了 `db-switch-path`/`db-list`/`mode-switch`（本次删除），Task 末尾同步；若只校验文件存在/无关 id，确认改动后仍满足。

---

## Task 1: 切换改按钮 + #switch-modal 弹窗

**Files:**
- Modify: `src/service/web/index.html`（新增 #switch-modal）
- Modify: `src/service/web/app.js`（loadDatabases 填 select、切换弹窗逻辑、删内联切换逻辑、重绑）
- Modify: `src/service/web/style.css`（switch-modal 内 select 样式）

- [ ] **Step 1: index.html 新增 #switch-modal（放在 #newdb-modal 之后）**

在 `#newdb-modal` 结束的 `</div>`（index.html:135）之后、`<script>` 之前，新增：
```html
  <div id="switch-modal" class="modal-overlay hidden">
    <div class="modal newdb-modal" role="dialog" aria-modal="true">
      <div class="modal-head">
        <span>切换库</span>
        <button id="switch-close" class="modal-close" aria-label="关闭">✕</button>
      </div>
      <div class="modal-body">
        <p class="newdb-hint">选择要切换到的库（当前库已标记）。</p>
        <select id="switch-select" class="switch-select"></select>
        <div class="newdb-actions">
          <button id="switch-cancel" class="db-btn" type="button">取消</button>
          <button id="switch-confirm" class="db-btn db-primary" type="button">确定切换</button>
        </div>
      </div>
    </div>
  </div>
```

- [ ] **Step 2: app.js — loadDatabases 改填 #switch-select（不再填 datalist）**

替换 `loadDatabases()`（app.js:265-284）：
```javascript
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
```

- [ ] **Step 3: app.js — 切换弹窗逻辑，删内联切换逻辑**

删除 `switchFromInput()` 和 `clearSwitchInput()`（app.js:305-315，含上面的注释行 `// 主菜单切换组…`）。在其位置新增：
```javascript
// 切换库弹窗：复用 .modal 样式，从下拉选库后走 switch。
function openSwitchModal() { $("switch-modal").classList.remove("hidden"); }
function closeSwitchModal() { $("switch-modal").classList.add("hidden"); }

async function confirmSwitch() {
  const p = $("switch-select").value;
  if (!p) { return; }
  const ok = await switchDatabase(p);
  if (ok) closeSwitchModal();   // 成功关弹窗；失败 switchDatabase 已 alert，弹窗留着
}
```

- [ ] **Step 4: app.js — 重绑事件**

把 db 事件绑定块（app.js:898-910，v3 的 `$("db-switch-btn")…` 到 newdb-* 那一整组）替换为：
```javascript
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
  if (e.target.id === "newdb-modal") closeNewDbModal();
});
$("newdb-path").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); confirmNewDb(); }
});
```
（注意：`db-switch-btn` 现在指向 Task 2 里新加的那个"🔀 切换"按钮；`db-switch-path`/`db-switch-clear` 的绑定删除——元素将在 Task 2 移除。）

- [ ] **Step 5: app.js — Esc 关闭覆盖 switch-modal**

现有 Esc 处理（app.js 约 894-896）：`if (e.key === "Escape") { closeCompileModal(); closeNewDbModal(); }`。改为也关切换弹窗：
```javascript
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") { closeCompileModal(); closeNewDbModal(); closeSwitchModal(); }
});
```

- [ ] **Step 6: style.css — switch-select 样式**

追加（复用 newdb 弹窗视觉）：
```css
/* 切换库弹窗内的下拉 */
.switch-select {
  width: 100%; box-sizing: border-box; font-family: var(--mono); font-size: 13px;
  padding: 7px 10px; border: 1px solid var(--line); border-radius: 6px;
  color: var(--fg); background: #fff;
}
.switch-select:focus { outline: none; border-color: var(--neon); box-shadow: 0 0 0 2px #0969da33; }
```

- [ ] **Step 7: 静态测试**

Run: `uv run pytest -m "not requires_model" tests/service/test_inspector_frontend_static.py -v`
Expected: PASS。（Task 1 单独跑时，index.html 里 `#db-switch-btn` 还没建/还是旧内联组——不要在此步跑浏览器；Task 2 完成后一起冒烟。若静态测试因 app.js 引用未建元素而失败，注意 app.js 顶层 `$("db-switch-btn")` 在页面加载时执行，需 Task 2 的 DOM 就位。**因此 Task 1 与 Task 2 必须一起提交验证**——见下方说明。）

**重要顺序说明:** Task 1 改了 app.js 的事件绑定引用 `#db-switch-btn`（Task 2 才建该按钮、才删旧的 `#db-switch-path`）。单独完成 Task 1 会导致页面加载时 `$("db-switch-path")` 绑定失败（元素还在但 switchFromInput 已删）或 `$("db-switch-btn")` 不存在。**故本计划 Task 1 和 Task 2 作为一个原子改动实现，最后一次性提交**（见 Task 2 Step 6 的提交）。Task 1 不单独提交。

---
## Task 2: header 左右分块 + 三按钮统一尺寸

**Files:**
- Modify: `src/service/web/index.html`（重构 .mode-bar 为左右两块；?v=4）
- Modify: `src/service/web/style.css`（左右布局、等高、三按钮 .menu-btn 统一尺寸；删死代码）

### 目标结构
```
.mode-bar (flex, align-items: stretch)
├─ .mode-search  (左块：搜索输入框 + 搜索按钮，竖直，拉高与右块等高)
└─ .mode-right   (右块：竖直两行)
   ├─ .mode-ingest  (右上：清单槽 + 回流槽 + 📥入库)
   └─ .mode-dbops   (右下：🔀切换 / ＋新建 / 🗑清空 三个等尺寸按钮)
```

- [ ] **Step 1: index.html — 重构 .mode-bar 左右两块**

把 `.mode-bar`（index.html:17-55，从 `<div class="mode-bar">` 到它的闭合 `</div>`，即 `.db-info-row` 之前）整体替换为：
```html
      <div class="mode-bar">
        <div class="mode-row mode-search">
          <input type="text" id="question" placeholder="输入新用户问题，回车或点搜索…" />
          <button id="btn-search">🔍 搜索</button>
        </div>
        <div class="mode-right">
          <div class="mode-row mode-ingest">
            <label class="file-slot" id="slot-manifest">
              <span class="slot-tag">用户清单 <em>.txt / .jsonl</em></span>
              <span class="slot-name" id="name-manifest">点击选择文件…</span>
              <input type="file" id="manifest" accept=".txt,.jsonl">
              <button type="button" class="slot-clear" id="clear-manifest" aria-label="取消选择">✕</button>
            </label>
            <label class="file-slot" id="slot-trajectories">
              <span class="slot-tag">候选回流 <em>.jsonl</em></span>
              <span class="slot-name" id="name-trajectories">点击选择文件…</span>
              <input type="file" id="trajectories" accept=".jsonl">
              <button type="button" class="slot-clear" id="clear-trajectories" aria-label="取消选择">✕</button>
            </label>
            <button id="btn-ingest">📥 入库</button>
            <!-- 旧单次检分入口：保留 DOM（现有 app.js 绑定依赖），双模式下隐藏 -->
            <button id="run" style="display:none">▶ 运行</button>
          </div>
          <div class="mode-row mode-dbops">
            <button id="db-switch-btn" class="menu-btn menu-switch" type="button">🔀 切换</button>
            <button id="db-new" class="menu-btn menu-new" type="button">＋ 新建库</button>
            <button id="db-clear" class="menu-btn menu-clear" type="button">🗑 清空当前库</button>
          </div>
        </div>
      </div>
```
（删除了 v3 的所有 `.mode-divider`、`.mode-switch` 内联输入框组、`#db-switch-path`/`#db-list`/`#db-switch-clear`。`#db-switch-btn` 现在是纯按钮。）

- [ ] **Step 2: index.html — ?v=3 → ?v=4**

第 7 行 `/style.css?v=3` → `/style.css?v=4`；末尾 `/app.js?v=3` → `/app.js?v=4`。

- [ ] **Step 3: style.css — 左右布局 + 等高 + 三按钮统一尺寸**

替换 `.mode-bar` 及相关（style.css:97-101 那几条 `.mode-bar/.mode-divider/.mode-row/.mode-search/.mode-ingest`）为：
```css
.mode-bar { display: flex; gap: 18px; align-items: stretch; flex-wrap: wrap; }
.mode-row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
/* 左块：搜索，竖直排列，拉高与右块等高 */
.mode-search {
  flex: 1 1 300px; flex-direction: column; align-items: stretch; gap: 10px;
  justify-content: center;
}
.mode-search #question { flex: 1 1 auto; }   /* 输入框吃满左块高度 */
/* 右块：竖直两行 */
.mode-right { display: flex; flex-direction: column; gap: 12px; flex: 2 1 460px; }
.mode-ingest { flex: 0 0 auto; }
.mode-dbops { flex: 0 0 auto; }
```
注意：`.mode-search` 之前是横排（输入框+按钮并排）；现在改竖排（`flex-direction:column`），输入框在上、搜索按钮在下，整块 `align-items:stretch` 让两者等宽，`#question` 用 `flex:1 1 auto` 吃满多余高度（因为右块两行更高，左块被 stretch 拉高，输入框撑开）。

`#question` 现有规则（style.css:103-108）有 `flex:1; min-width:180px; height:40px`。`height:40px` 会阻止它拉高。改：把 `.mode-search #question` 的 `height:40px` 去掉（或改 `min-height:40px`），让它能被 `flex:1 1 auto` 拉高。读 103-108 行，把 `height: 40px;` 改为 `min-height: 40px;`。

- [ ] **Step 4: style.css — 三按钮 .menu-btn 统一尺寸（仿 btn-search）**

`#btn-search`/`#btn-ingest`（style.css:113-124）已是 40px 高统一按钮。给三个库操作按钮同款尺寸，追加：
```css
/* 库操作三按钮：与搜索/入库同尺寸（40px 高），三者等宽 */
.menu-btn {
  flex: 1 1 0; height: 40px; box-sizing: border-box;
  padding: 0 16px; cursor: pointer; font-size: 13px; font-weight: 600;
  color: #fff; border-radius: 6px; letter-spacing: .5px; white-space: nowrap;
  transition: background .15s, transform .05s;
}
.menu-btn:active { transform: translateY(1px); }
.menu-switch { background: var(--neon); border: 1px solid #0a5cc4; }
.menu-switch:hover { background: #0a5cc4; }
.menu-new { background: #1f883d; border: 1px solid #1a7f37; }
.menu-new:hover { background: #1a7f37; }
.menu-clear { background: var(--hot); border: 1px solid #a40e26; }
.menu-clear:hover { background: #a40e26; }
```
（`flex:1 1 0` 让三按钮在 `.mode-dbops` 行里均分等宽；配色：切换蓝、新建绿、清空红。）

- [ ] **Step 5: style.css — 删死代码**

删除 v3 遗留、现已无 DOM 对应的规则：`.db-bar` 若还在（应已在 v3 删）、`.mode-switch`/`.switch-field`/`#db-switch-path` 相关、`.switch-clear` 两条（`.switch-field .switch-clear` + `:placeholder-shown ~`）。`grep -n "mode-switch\|switch-field\|db-switch-path\|switch-clear\|db-list" src/service/web/style.css` 找到并删除。保留 `.db-btn`/`.db-primary`（newdb/switch 弹窗按钮用）、`.slot-clear`（文件槽 ✕ 用）、`.switch-select`（Task 1 加）。

- [ ] **Step 6: 静态测试 + 全套 + grep + 提交（Task 1+2 一起）**

Run: `uv run pytest -m "not requires_model" tests/service/test_inspector_frontend_static.py -v`
Then: `uv run pytest -m "not requires_model" tests/service -q` → 135 pass。
Grep 检查（旧内联切换组彻底移除）：
`grep -rn "db-switch-path\|db-list\|switchFromInput\|clearSwitchInput\|mode-switch\|switch-field\|db-switch-clear" src/service/web/` → 应为空。
确认新元素都有 DOM + 绑定：`db-switch-btn`/`db-new`/`db-clear`/`switch-modal`/`switch-select`/`switch-close`/`switch-cancel`/`switch-confirm` 各出现在 index.html 且在 app.js 有绑定。

手动（服务在跑，强刷 Cmd+Shift+R）：
- header 左右两块：左=搜索（输入框+按钮竖排，整块拉高与右侧等高）；右上=入库组，右下=切换/新建/清空三个等尺寸等宽按钮（蓝/绿/红），与搜索/入库按钮同高。
- 点「🔀切换」弹窗 → 下拉选库（当前库标记）→「确定切换」切库；取消/背景/Esc 关。
- 「＋新建库」弹 newdb 弹窗；「🗑清空」二次确认。
- 库信息行不变。

提交：
```bash
git add src/service/web/index.html src/service/web/app.js src/service/web/style.css tests/service/test_inspector_frontend_static.py
git commit -m "feat(inspector-web): header 左右分块（左搜索/右入库+库操作），切换改弹窗，三按钮统一尺寸

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## 验收标准

- [ ] header 左块=搜索（输入框+按钮），垂直拉高与右块两行等高。
- [ ] 右上=入库组（两文件槽+入库按钮，不变）；右下=切换/新建/清空三个等尺寸等宽按钮，与搜索/入库按钮同 40px 高、同风格（配色区分）。
- [ ] 切换是纯按钮 → 点击弹 `#switch-modal`，下拉列所有 .db（当前库标记）+ 确定切换；取消/背景/Esc 关闭。
- [ ] 新建走 `#newdb-modal`（沿用）；清空二次确认（沿用）。
- [ ] v3 内联切换组（db-switch-path/db-list/switch-field/mode-switch/switch-clear）无残留（grep 空）。
- [ ] `?v=4` 生效；`uv run pytest -m "not requires_model" tests/service` 全绿。

