# 移除 min_turns 过滤器 设计决策

> **状态**：决策定稿，待实施。
> **类型**：契约级变更（`StructuredFilters` 移除一个字段）。
> **触发**：真实全链路 e2e 验证时,module0 (gpt-5.5) 为 SFT 查询生成 `min_turns=2`,但所有 SWE agent 轨迹的 `turn_count` 恒为 1,导致结构化过滤阶段全部候选被滤空(空产出)。

---

## 1. 问题诊断（实测）

真实 e2e(`scripts/e2e_smoke.py` on `smoke_20.jsonl`)链路各阶段均正常执行(module0 编译✓、module1 切片+签名+索引✓、gateway 100% 成功、embedding✓),**唯一断点**在结构化过滤:

```
过滤后候选: 0 条 (共 20 条签名)
    单独 min_turns=2 → 0/20 通过     ← 全部被滤空
```

**根因链:**
1. `turn_count` 定义(`signature.py:87`)：`sum(1 for s in steps if s.role == "user")` —— 即"用户消息条数"。数据确定,无需 LLM。
2. SWE agent 轨迹是**单轮多步**结构：用户提一次需求(1 个 user 消息)→ assistant + tool 来回执行 N 步。实测 `smoke_20` 每条轨迹 `roles = [system, user, assistant, tool, ...]`,**turn_count 恒为 1**。
3. `min_turns` 是 **module0 的 LLM 在编译时凭直觉生成**的过滤条件(本次生成 `min_turns=2`)。

## 2. 决策依据：为何移除而非修正

`min_turns` 对 agent 轨迹数据是一个**无效且有害**的过滤器：

- **无区分度**：turn_count 在 agent 轨迹里恒为 1。`min_turns=1` 等于全放行(等于没过滤);`min_turns≥2` 等于全滤空(误杀)。二者都不提供有效筛选。
- **LLM 猜测,不精确**：它是 `structured_filters` 里**唯一**一个"由 LLM 直觉生成 + 过滤无区分度字段"的组合。其余字段(`languages`/`tools_used`/`has_verification_step`)都是数据确定、有区分度的精确筛选。
- **唯一实际效果是偶发误杀**（如本次空产出）。

结论：`min_turns` 是 `structured_filters` 的一个设计缺陷,应**移除**,而非改阈值或引导 LLM。规模筛选若未来需要,应基于 `step_count`（数据确定、有区分度）另行设计,不在本次范围。

## 3. 契约变更

`interface-contract.md` §1.3 `StructuredFilters` 移除 `min_turns` 字段;§2.4 `min_turns` 小节删除。变更后 `StructuredFilters` 为 3 字段：`languages`、`tools_used`、`has_verification_step`。

## 4. 爆炸半径（实施清单）

**源码：**
- `src/module0/schema.py`：`StructuredFilters` 删 `min_turns` 字段 + `__post_init__` 里的 `min_turns < 1` 校验;`_parse_structured_filters`(:135)删 `min_turns=d.get(...)`。
- `src/module0/compiler.py:316`：`_build_filters` 删 `min_turns=d.get("min_turns")`。
- `src/module0/prompts.py`：两处 `- min_turns: 整数 ≥1`(:73,:142)删除;两处 JSON schema 示例(:85,:154)里的 `"min_turns": 3` 删除。
- `src/module1/index.py`：`_apply_filters` 删 min_turns 过滤分支(:135,:147-149);docstring(:13,:65)更新。

**测试：**
- `tests/module1/test_index.py`：删 `test_filter_min_turns`(:73-79);其它 `structured_filters` 断言里的 `min_turns` 键移除(:214)。
- `tests/module0/test_compiler.py:97`：该 item 的 `structured_filters={"min_turns": 5}` 改为不含 min_turns 的等价过滤(或空 dict)。
- `tests/module1/test_pipeline.py:45`：删 `min_turns` 键。

**契约 + fixture：**
- `docs/superpowers/specs/interface-contract.md`：§1.3、§2.4 如上。
- `fixtures/problem_specs/test_01.json`：两处 `structured_filters` 删 `min_turns` 键。

## 5. 验证

- `.venv/bin/python -m ruff check` clean;`.venv/bin/python -m pytest -m "not requires_model"` 全绿(移除后无残留引用)。
- `grep -rn "min_turns" src/ tests/ docs/superpowers/specs/interface-contract.md fixtures/`（除本 spec 外）应为空。
- 重跑真实 e2e(`scripts/e2e_smoke.py ... smoke_20.jsonl`）：结构化过滤不再空滤,主链路应产出 SFT 候选（前提是 judge 命中）。

## 6. 不做（本次）

- 基于 `step_count` 的规模过滤（`min_steps`）——如需,另行立项。
- turn_count 定义变更——保持不变（它对多轮数据仍正确,只是 agent 轨迹里恒为 1）。
