# UI 修订 v5：全部压成一行 · 按钮统一自然尺寸 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 header 主区摊平成一条水平行：左=搜索输入框（伸缩充满）+ 🔍搜索按钮（框右侧），竖分隔线，右=两个文件槽 + 四个按钮（📥入库/🔀切换/＋新建/🗑清空）。所有五个按钮去掉等宽拉伸、统一为"内容自然宽 + 40px 高"（以入库按钮为基准）。按钮改名：`＋新建库`→`＋新建`、`🗑清空当前库`→`🗑清空`。

**Architecture:** 纯前端，改 `src/service/web/{index.html,style.css}`（**app.js 不改**——所有绑定 id、弹窗逻辑、行为都不变；只动 DOM 布局与按钮文案与 CSS）。撤掉 v4 的 `.mode-right` 上下两行嵌套，恢复横向单行结构。`.menu-btn` 去掉 `flex:1 1 0`（自然宽）。`?v=4`→`?v=5`。库信息行不变。

**Tech Stack:** 原生 HTML/CSS。无 JS 测试运行器；验证靠 `tests/service/test_inspector_frontend_static.py` + 手动浏览器冒烟。

**目标布局:**
```
[ 搜索输入框 ──伸缩充满── ][🔍搜索]  |  [用户清单▾][候选回流▾][📥入库][🔀切换][＋新建][🗑清空]
【uxflow.db】中 x问题·x轨迹·x切片                                      最终入选 x 条
```

**关键背景（已核验，v4 现状）:**
- index.html:17-46 是 `.mode-bar` → `.mode-search`(搜索输入框+按钮，竖排) + `.mode-right`(内含 `.mode-ingest` 行 + `.mode-dbops` 行)。要摊平：去掉 `.mode-right` 包裹，把入库组和四个按钮放同一横向操作组。
- 按钮 id 与文案：`#btn-search`(🔍 搜索)、`#btn-ingest`(📥 入库)、`#db-switch-btn`(🔀 切换)、`#db-new`(＋ 新建库)、`#db-clear`(🗑 清空当前库)。id 全部保留（app.js 绑定依赖），只改可见文案后两个。
- CSS(style.css:96-145)：`.mode-bar{align-items:stretch}`、`.mode-search{flex-direction:column}`、`.mode-right`、`#question{flex:1 1 auto; min-height:40px}`、`button#btn-search/#btn-ingest`(40px 自然宽)、`.menu-btn{flex:1 1 0; height:40px}` 三色。
- app.js 不动：`btn-search`/`btn-ingest`/`db-switch-btn`/`db-new`/`db-clear` 绑定、切换/新建/清空弹窗逻辑、文件槽 ✕、库信息 loadStats/loadDatabases 全部保持。

---

## Task 1: HTML 摊平为一行 + 按钮改名 + 版本号

**Files:**
- Modify: `src/service/web/index.html`

- [ ] **Step 1: 重构 .mode-bar（去掉 .mode-right，横向单行）**

把 index.html 第 17-46 行（`<div class="mode-bar">` 到其闭合 `</div>`，`.db-info-row` 之前）替换为：
```html
      <div class="mode-bar">
        <div class="mode-row mode-search">
          <input type="text" id="question" placeholder="输入新用户问题，回车或点搜索…" />
          <button id="btn-search">🔍 搜索</button>
        </div>
        <div class="mode-divider"></div>
        <div class="mode-row mode-ops">
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
          <button id="db-switch-btn" class="menu-btn menu-switch" type="button">🔀 切换</button>
          <button id="db-new" class="menu-btn menu-new" type="button">＋ 新建</button>
          <button id="db-clear" class="menu-btn menu-clear" type="button">🗑 清空</button>
          <!-- 旧单次检分入口：保留 DOM（现有 app.js 绑定依赖），双模式下隐藏 -->
          <button id="run" style="display:none">▶ 运行</button>
        </div>
      </div>
```
要点：`.mode-right` 包裹层删除；`.mode-ingest`/`.mode-dbops` 合并为一个横向 `.mode-ops`；两个文件槽 + 四个按钮同排；`#db-new` 文案 `＋ 新建库`→`＋ 新建`、`#db-clear` 文案 `🗑 清空当前库`→`🗑 清空`；`#btn-search` 回到 `.mode-search` 输入框右侧（本就在，保持）。所有 id 不变。

- [ ] **Step 2: 版本号 ?v=4 → ?v=5**

第 7 行 `/style.css?v=4` → `/style.css?v=5`；末尾 `/app.js?v=4` → `/app.js?v=5`。

- [ ] **Step 3: 静态测试**

Run: `uv run pytest -m "not requires_model" tests/service/test_inspector_frontend_static.py -v`
Expected: PASS。（此 Task 单独不跑浏览器——CSS 未改完，Task 2 一起验证。但静态测试是 Python 端断言，此步应已能过。）

---

## Task 2: CSS 摊平横向 + 按钮统一自然尺寸

**Files:**
- Modify: `src/service/web/style.css`

- [ ] **Step 1: 重写 .mode-bar / .mode-search / #question / 删 .mode-right/.mode-ingest/.mode-dbops**

把 style.css 第 96-118 行（`.modes` 之后到 `#question::placeholder`，即 `.mode-bar`/`.mode-row`/`.mode-search`/`.mode-right`/`.mode-ingest`/`.mode-dbops`/`.mode-search #question`/`:focus`/`::placeholder` 这一段）替换为：
```css
.modes { display: flex; flex-direction: column; gap: 12px; }
.mode-bar { display: flex; gap: 18px; align-items: center; flex-wrap: wrap; }
.mode-divider { align-self: stretch; width: 1px; background: var(--line-2); margin: 2px 0; }
.mode-row { display: flex; gap: 12px; align-items: center; flex-wrap: wrap; }
.mode-search { flex: 1 1 300px; }
.mode-ops { flex: 2 1 520px; }

/* 搜索输入框：横向伸缩充满剩余空间，固定 40px 高 */
.mode-search #question {
  flex: 1; min-width: 180px; height: 40px; box-sizing: border-box;
  padding: 0 14px; font-size: 13px;
  font-family: var(--sans); color: var(--fg); background: #fff;
  border: 1px solid transparent; border-radius: 6px; outline: none;
  transition: box-shadow .15s;
}
.mode-search #question:focus { box-shadow: 0 0 0 2px var(--neon-2); }
.mode-search #question::placeholder { color: var(--fg-faint); }
```
要点：`.mode-bar` 回 `align-items:center`（不再 stretch）；`.mode-divider` 恢复（v4 删过，这里加回，颜色用 `--line-2`）；`.mode-search` 回横排（去掉 `flex-direction:column`）；`.mode-ops` 新类（横向操作组）；`#question` 回固定 `height:40px` + `flex:1`（横向伸缩，不再竖向拉高）。`.mode-right`/`.mode-ingest`/`.mode-dbops` 三个规则删除（DOM 已无）。

- [ ] **Step 2: .menu-btn 去掉等宽拉伸 → 自然宽（对齐入库按钮）**

把 `.menu-btn` 规则（style.css:134-139）里的 `flex: 1 1 0;` 去掉，其余不变，使三个库操作按钮与 `#btn-ingest` 一样按内容自然宽：
```css
/* 库操作按钮：与入库按钮同尺寸（40px 高，内容自然宽） */
.menu-btn {
  height: 40px; box-sizing: border-box;
  padding: 0 16px; cursor: pointer; font-size: 13px; font-weight: 600;
  color: #fff; border-radius: 6px; letter-spacing: .5px; white-space: nowrap;
  transition: background .15s, transform .05s;
}
```
（`.menu-btn:active`、`.menu-switch/.menu-new/.menu-clear` 三色规则保持不变。）

- [ ] **Step 3: 静态测试 + 全套 + grep 死代码 + 提交**

Run: `uv run pytest -m "not requires_model" tests/service/test_inspector_frontend_static.py -v`
Then: `uv run pytest -m "not requires_model" tests/service -q` → 135 pass。
Grep：`grep -rn "mode-right\|mode-ingest\|mode-dbops" src/service/web/` → 应为空（HTML 用 `.mode-ops`、CSS 已删旧三类）。
确认 `.mode-divider` 在 index.html（搜索组与操作组之间）与 style.css 都在。

手动（服务在跑，强刷 Cmd+Shift+R）：
- 一整行：左搜索框伸缩充满 + 🔍搜索按钮在框右侧；竖线；右侧 用户清单槽 + 候选回流槽 + 📥入库 🔀切换 ＋新建 🗑清空。
- 五个按钮尺寸一致（都 40px 高、内容自然宽），不再有等宽拉伸的宽按钮。
- 屏窄时整齐 flex-wrap 换行。
- 切换点击弹窗、新建弹窗、清空二次确认、库信息行——全部行为不变。

提交（Task 1 + Task 2 一起，一次提交，因 HTML 改了类名而 CSS 配套）：
```bash
git add src/service/web/index.html src/service/web/style.css
git commit -m "feat(inspector-web): header 摊平为单行（搜索伸缩+六项操作横排），按钮统一自然尺寸并改名

Co-Authored-By: Claude Fable 5 <noreply@anthropic.com>"
```

---

## 验收标准

- [ ] header 主区一整行：左=搜索输入框（伸缩充满）+ 🔍搜索按钮（框右侧）；竖分隔线；右=用户清单槽 + 候选回流槽 + 📥入库/🔀切换/＋新建/🗑清空。
- [ ] 五个按钮（搜索/入库/切换/新建/清空）尺寸一致：40px 高、内容自然宽（无 flex 等宽拉伸），以入库按钮为基准。
- [ ] `#db-new` 文案 `＋ 新建`、`#db-clear` 文案 `🗑 清空`。
- [ ] 切换弹窗/新建弹窗/清空二次确认/文件槽 ✕/库信息行行为全部不变（app.js 未改）。
- [ ] `.mode-right`/`.mode-ingest`/`.mode-dbops` 无残留（grep 空）。
- [ ] `?v=5` 生效；`uv run pytest -m "not requires_model" tests/service` 全绿。
