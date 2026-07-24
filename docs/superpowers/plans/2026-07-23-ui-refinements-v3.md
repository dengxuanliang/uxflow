# UI 修订 v3：证据按钮固化 · 五功能菜单行 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** (1) 修复"命中轨迹"栏"证据"按钮在轨迹名过长时竖排的问题——固化为水平、固定尺寸；(2) 把 header 重构成一条水平"菜单"：搜索 | 入库 | 切换 | 新建 | 清空 五组功能同行、以竖分隔线分区，切换用「输入框+datalist 候选+✕+切换按钮」，新建/清空为按钮。

**Architecture:** 纯前端，改 `src/service/web/{index.html,app.js,style.css}`，无后端改动。切换沿用已有 `switchDatabase()`；datalist 的 option value 用**库文件名**（后端 `/databases/switch` 对非绝对路径按 `current().parent` 解析，而 `list_databases` 只 glob 该目录，故文件名可正确回指到目标库，输入框也保持简洁）。新建沿用 `#newdb-modal` 弹窗，清空沿用 `clearDatabase()` 二次确认。version 号从 `?v=2` 升到 `?v=3` 防缓存。

**Tech Stack:** 原生 JS/CSS/HTML。无 JS 测试运行器；验证靠 `tests/service/test_inspector_frontend_static.py` + 手动浏览器冒烟。

**关键背景（已核验）:**
- `.hit`（style.css:266-270）是 `display:flex; justify-content:space-between; align-items:center` 的行。子元素：`<span>(hit-id + hit-seg)</span>` + 可选 `.selected-badge` + 可选 `.evidence-toggle`。`.selected-badge` 有 `flex:0 0 auto; white-space:nowrap`（不挤），但 `.evidence-toggle`（style.css:372-382）**没有** `flex-shrink:0`/`white-space:nowrap`，且带遗留 `margin-top:4px`（accordion 时代残留）。轨迹名长时，左侧 span 把按钮挤到没宽度 →"证据"竖排。修复要点：左 span 可收缩+省略号，证据按钮不收缩+不换行。
- header 现状（index.html:16-51）：`.modes` 内含 `.mode-bar`（`.mode-search` + `.mode-divider` + `.mode-ingest`）、`#db-bar`（切换库 label + `#db-select` + `#db-new` + `#db-clear`）、`.db-info-row`（stats + manifest）。
- `.mode-bar`（style.css:97）已是 `flex; gap:18px; align-items:center; flex-wrap:wrap`。`.mode-divider`（:98）竖线。`#question`/按钮/文件槽都是 40px 高。
- db 事件绑定 app.js:898-910：`db-select` change→switchDatabase、`db-new`→openNewDbModal、`db-clear`→clearDatabase、newdb-* 一组。底部初始化 `loadStats(); loadDatabases();`（:913-914）。
- `loadDatabases()`（app.js:263-283）现填 `#db-select`；要改填 `#db-list`（datalist）。`switchDatabase(path)`（:286-302）成功返回 true、失败 alert 并返回 false（v2 已改）。`#newdb-modal` 弹窗与 `openNewDbModal/closeNewDbModal/confirmNewDb`（:322-331）保留不动。
- `/databases` 返回 `{current, databases:[{name,path,problems,trajectories,is_current}]}`。
- `.db-btn`/`.db-primary` 仍被 `#newdb-modal` 内的按钮使用（保留）；`.db-bar`/`.db-bar .db-label`/`.db-select`（style.css:394-411）在 `#db-bar` 拆除后变死代码（删除）。

---

## 前置：确认静态测试断言

- [ ] **Step 0: 读 tests/service/test_inspector_frontend_static.py**

Run: `sed -n '1,80p' tests/service/test_inspector_frontend_static.py`
若它断言了 `db-select`/`db-bar`（本次删除）或依赖 header 结构，Task 2 末尾同步；若只校验文件存在/关键 id，确认改动后仍满足。

---

## Task 1: "证据"按钮固化为水平固定尺寸

**Files:**
- Modify: `src/service/web/app.js`（renderHits 左侧 span 包一层可省略）
- Modify: `src/service/web/style.css`（.evidence-toggle 不收缩 + 左侧信息可省略）

- [ ] **Step 1: renderHits 给左侧信息包一个可省略容器**

`renderHits()`（app.js:622-624）现在把 id/seg 放在一个裸 `<span>` 里。给它一个类名以便 CSS 控制收缩+省略。把这段：
```javascript
    hd.innerHTML =
      `<span><span class="hit-id">${escapeHtml(h.trajectory_id)}</span>` +
      `<span class="hit-seg"> · slice${h.slice_index} · ${h.caps.length}片段</span></span>${badge}`;
```
改为（外层 span 加 `class="hit-main"`）：
```javascript
    hd.innerHTML =
      `<span class="hit-main"><span class="hit-id">${escapeHtml(h.trajectory_id)}</span>` +
      `<span class="hit-seg"> · slice${h.slice_index} · ${h.caps.length}片段</span></span>${badge}`;
```
其余（evidence-toggle 的创建/绑定、badge）不变。

- [ ] **Step 2: style.css — 左信息可收缩省略 + 证据按钮固化**

在 `.hit` 相关规则区（style.css:266-280 附近）追加/修改。新增 `.hit-main`：
```css
/* 左侧轨迹信息：可收缩，过长时省略号，避免挤压右侧徽标/证据按钮 */
.hit .hit-main {
  flex: 1 1 auto; min-width: 0;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}
```
把 `.evidence-toggle`（style.css:372-382）改为不收缩、不换行、去掉 margin-top：
```css
.evidence-toggle {
  flex: 0 0 auto; white-space: nowrap;
  font-size: 11px;
  padding: 2px 10px;
  border: 1px solid var(--line);
  border-radius: 10px;
  background: var(--bg-2);
  color: var(--fg-dim);
  cursor: pointer;
}
.evidence-toggle:hover { background: var(--bg-3); }
```
（关键：`flex:0 0 auto` + `white-space:nowrap` 让"证据"永远水平单行、固定尺寸；`.hit-main` 的 `min-width:0` 是让 flex 子元素能真正收缩+省略的必要条件。）

- [ ] **Step 3: 静态测试**

Run: `uv run pytest -m "not requires_model" tests/service/test_inspector_frontend_static.py -v`
Expected: PASS。

手动（服务在跑，强刷）：命中一个 trajectory_id 很长的轨迹 → 名字被省略号截断，"证据"按钮仍是水平单行固定大小，不竖排。

- [ ] **Step 4: 提交**
```bash
git add src/service/web/app.js src/service/web/style.css
git commit -m "fix(inspector-web): 证据按钮固化为水平固定尺寸（长轨迹名截断不挤压）

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---
## Task 2: header 五功能菜单行（搜索|入库|切换|新建|清空）

**Files:**
- Modify: `src/service/web/index.html`（把切换/新建/清空并入 .mode-bar，删除 #db-bar）
- Modify: `src/service/web/app.js`（loadDatabases 填 datalist；切换输入框 ✕；重绑事件；升 ?v=3）
- Modify: `src/service/web/style.css`（切换组样式；删死代码 .db-bar/.db-select）

### 目标形态（.mode-bar 一行，竖线分区，屏窄 flex-wrap）
```
🔍搜索[输入框] | 📥入库[清单槽][回流槽] | 🔀[切换输入框+datalist+✕][切换] | ＋新建库 | 🗑清空当前库
```
库信息行（`.db-info-row`：`【当前库】中…` + 右侧「最终入选…」）保持不变。

- [ ] **Step 1: index.html — 重构 .mode-bar，删除 #db-bar**

把 `.mode-bar`（index.html:17-40）与其后的 `#db-bar`（41-46）整体替换为：把切换/新建/清空并进同一条 `.mode-bar`，各组之间用 `.mode-divider`。`.db-info-row`（47-50）保持不动。

将 index.html 第 17-46 行（`<div class="mode-bar">` 到 `#db-bar` 的 `</div>`）替换为：
```html
      <div class="mode-bar">
        <div class="mode-row mode-search">
          <input type="text" id="question" placeholder="输入新用户问题，回车或点搜索…" />
          <button id="btn-search">🔍 搜索</button>
        </div>
        <div class="mode-divider"></div>
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
        <div class="mode-divider"></div>
        <div class="mode-row mode-switch">
          <div class="switch-field">
            <input type="text" id="db-switch-path" list="db-list"
                   placeholder="选择或输入库…" autocomplete="off" />
            <datalist id="db-list"></datalist>
            <button type="button" class="slot-clear switch-clear" id="db-switch-clear" aria-label="清空输入">✕</button>
          </div>
          <button id="db-switch-btn" class="db-btn">🔀 切换</button>
        </div>
        <div class="mode-divider"></div>
        <div class="mode-row mode-dbops">
          <button id="db-new" class="db-btn" type="button">＋ 新建库</button>
          <button id="db-clear" class="db-btn db-clear" type="button">🗑 清空当前库</button>
        </div>
      </div>
```
（注意：删除了整个 `#db-bar` 块——`#db-select` 下拉、`切换库` label 都没了。`#newdb-modal`（在 `#compile-modal` 之后）保持不动。）

- [ ] **Step 2: index.html — 资源版本号升 ?v=3**

第 7 行 `href="/style.css?v=2"` → `href="/style.css?v=3"`；末尾 `src="/app.js?v=2"` → `src="/app.js?v=3"`。

- [ ] **Step 3: app.js — loadDatabases 填 datalist（不再填 select）**

`loadDatabases()`（app.js:263-283）当前操作 `#db-select`（已删）。改为填 `#db-list` datalist（option 的 value 用文件名，label 附带问题数）：
```javascript
async function loadDatabases() {
  try {
    const resp = await fetch("/databases");
    if (!resp.ok) return;
    const { current, databases } = await resp.json();
    // 当前库文件名（纯文本，供库信息行显示）
    state.currentDbName = (current || "").split("/").pop() || "当前库";
    const list = $("db-list");
    list.innerHTML = "";
    for (const d of databases) {
      const opt = document.createElement("option");
      // value 用文件名：后端 switch 对非绝对路径按 current().parent 解析，回指正确库
      opt.value = d.name;
      const cnt = d.problems == null ? "?" : d.problems;
      opt.label = `${cnt} 问题` + (d.is_current ? " · 当前" : "");
      list.appendChild(opt);
    }
    loadStats();   // 库名可能已更新，刷新信息行
  } catch (_) { /* best-effort */ }
}
```

- [ ] **Step 4: app.js — 切换组逻辑（切换按钮 + 输入框 ✕ + Enter）**

新增切换相关函数（放在 switchDatabase 附近）：
```javascript
// 主菜单切换组：从输入框（可手输或选 datalist）取值切库。
async function switchFromInput() {
  const p = $("db-switch-path").value.trim();
  if (!p) { $("db-switch-path").focus(); return; }
  const ok = await switchDatabase(p);
  if (ok) $("db-switch-path").value = "";   // 成功后清空输入，失败保留供重试
}
function clearSwitchInput() {
  $("db-switch-path").value = "";
  $("db-switch-path").focus();
}
```

- [ ] **Step 5: app.js — 重绑事件（移除 db-select，加切换组）**

把 db 事件绑定块（app.js:898-901，从 `$("db-select")…` 到 `$("db-clear")…`）替换为：
```javascript
// 库控件事件绑定
$("db-switch-btn").addEventListener("click", switchFromInput);
$("db-switch-path").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); switchFromInput(); }
});
$("db-switch-clear").addEventListener("click", clearSwitchInput);
$("db-new").addEventListener("click", openNewDbModal);
$("db-clear").addEventListener("click", clearDatabase);
```
（`db-select` 的 change 绑定删除——元素已不存在。newdb-* 绑定 app.js:902-910 保持不变。）

- [ ] **Step 6: style.css — 切换组样式 + 删死代码**

删除死代码：`.db-bar`、`.db-bar .db-label`、`.db-select`（style.css:394-411 那几条，`.db-btn`/`.db-btn:hover` 保留——新建/切换/清空按钮都用它）。

追加切换组样式（切换输入框仿 #question 外观、40px 高对齐；✕ 复用 `.slot-clear` 小图标但在输入框内右侧定位）：
```css
/* 主菜单切换组：输入框（含 datalist 候选）+ 框内 ✕ + 切换按钮 */
.mode-switch { flex: 1 1 260px; }
.switch-field { position: relative; display: flex; align-items: center; flex: 1; min-width: 160px; }
.switch-field #db-switch-path {
  flex: 1; min-width: 0; height: 40px; box-sizing: border-box;
  padding: 0 28px 0 12px; font-size: 13px; font-family: var(--mono);
  color: var(--fg); background: #fff;
  border: 1px solid transparent; border-radius: 6px; outline: none;
  transition: box-shadow .15s;
}
.switch-field #db-switch-path:focus { box-shadow: 0 0 0 2px var(--neon-2); }
.switch-field #db-switch-path::placeholder { color: var(--fg-faint); font-family: var(--sans); }
/* 切换输入框内的 ✕：仅有内容时显示（JS 无需切换，用 :placeholder-shown 兜底） */
.switch-clear { right: 8px; }
.switch-field #db-switch-path:placeholder-shown ~ .switch-clear { display: none; }
```
注意：`.slot-clear` 基础样式（16px 圆、绝对定位、`display:none` 默认、`.file-slot.filled .slot-clear{display:block}`）是针对文件槽的。切换 ✕ 复用 `.slot-clear` 的视觉，但显隐逻辑不同——文件槽靠 `.filled` 类，切换靠输入框有无内容。上面用 `:placeholder-shown ~ .switch-clear { display:none }` 实现"空输入时隐藏 ✕"；但 `.slot-clear` 默认 `display:none`，需要一条让切换 ✕ 在**有内容时显示**的规则。补一条：
```css
.switch-clear { display: block; }   /* 切换 ✕ 默认可见；下面的 :placeholder-shown 规则在空输入时隐藏它 */
```
把这条放在 `.switch-field #db-switch-path:placeholder-shown ~ .switch-clear { display: none; }` **之前**（后者靠更高特异性/顺序覆盖，空输入时隐藏）。确认最终效果：输入框为空 → ✕ 隐藏；有内容 → ✕ 显示。

- [ ] **Step 7: 静态测试 + 全套**

Run: `uv run pytest -m "not requires_model" tests/service/test_inspector_frontend_static.py -v`
Then: `uv run pytest -m "not requires_model" tests/service -q`
Expected: PASS。若静态测试断言了 `db-select`/`db-bar`，同步为新结构（`db-switch-path`/`db-switch-btn` 等）。

Grep 检查：`grep -n "db-select\|db-bar\|#db-select" src/service/web/app.js src/service/web/index.html` → 应为空（除了 CSS 里可能残留，Step 6 已删）。

手动（服务在跑，强刷 Cmd+Shift+R）：
- header 一行五组：搜索[框+按钮] | 入库[两槽+按钮] | 切换[框+✕+按钮] | 新建 | 清空，竖线分区；屏窄时整齐换行。
- 切换输入框点击/聚焦 → 弹出 datalist 候选（所有 .db，附问题数/当前标记）；选一个或手输库名 → 输入框出现 ✕；点"🔀切换"或回车 → 切库；点 ✕ 清空输入。
- 「＋新建库」弹 modal 输入新建；「🗑清空当前库」二次确认。
- 库信息行仍显示「【当前库】中…」+ 右侧「最终入选…」。

- [ ] **Step 8: 提交**
```bash
git add src/service/web/index.html src/service/web/app.js src/service/web/style.css tests/service/test_inspector_frontend_static.py
git commit -m "feat(inspector-web): header 重构为五功能菜单行（搜索|入库|切换|新建|清空）

切换改为输入框+datalist候选+✕；删除 #db-bar 下拉。

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## 验收标准

**Task 1（证据按钮固化）:**
- [ ] 轨迹名很长时，名字省略号截断，"证据"按钮水平单行、固定尺寸，绝不竖排。
- [ ] `.evidence-toggle` 有 `flex:0 0 auto; white-space:nowrap`；`.hit-main` 有 `min-width:0` + 省略。

**Task 2（五功能菜单行）:**
- [ ] header 一行含五组：搜索/入库/切换/新建/清空，`.mode-divider` 竖线分区，屏窄 flex-wrap 换行。
- [ ] 切换：输入框 + datalist 候选（列所有 .db）+ 框内 ✕（空则隐藏）+ 「🔀切换」按钮；回车亦可切；成功清空输入、失败保留。
- [ ] 新建走 `#newdb-modal` 弹窗（沿用）；清空二次确认（沿用）。
- [ ] `#db-select`/`#db-bar` 完全移除，无残留引用（grep app.js/index.html 为空）；死 CSS 已删。
- [ ] 库信息行不变。

**整体:**
- [ ] `?v=3` 生效；`uv run pytest -m "not requires_model" tests/service` 全绿。

