# Golden Evaluation Set (v1)

> 用于 prompt 优化的评估基线。每条标注期望输出，作为准确率/漂移率的判定标准。
> 来源：7 条真实用户反馈 + 13 条扩展（覆盖 v0 taxonomy 全叶子 + 各 drop 类型）。

## 评估指标定义

- **标签准确率**：target_capability 命中期望标签的比例（交集/期望集）
- **route 准确率**：pass/drop 判对的比例
- **drop_reason 准确率**：drop 时 reason 判对的比例
- **漂移率**：同一输入跑 N=3 次，target_capability + route 完全一致的比例（越高越好）

---

## 评估数据（20 条）

### 真实用户问题（7 条）

| id | raw_input | 期望拆分 | 期望 target_capability | 期望 route | 期望 drop_reason | 备注 |
|---|---|---|---|---|---|---|
| e01 | "用户提供了三个文件，模型在处理过程中把文件内容搞混了" | 1 条 | `file_localization_and_edit` | pass | — | 文件定位能力 |
| e02 | "python代码写入时有语法错误，且缩进出现问题" | 2 条 | p1: `valid_syntax_in_toolcall`; p2: `correct_indentation` | pass | — | 两个独立失败模式 |
| e03 | "agent工具解析有问题,tool_call的标签出现在thinking里面" | 1 条 | `wellformed_tool_call` | pass | — | 直接命中 |
| e04 | "python实现五子棋，玩家胜利后无胜利提示，用户提醒后仍然没有实现" | 1 条 | `requirement_completeness` | pass | — | 功能点遗漏 |
| e05 | "用户提示使用python虚拟环境，但模型仍然未使用直接执行" | 1 条 | 新标签提议（如 `follow_user_instruction`） | pass | — | v0 无覆盖，允许 taxonomy_extension |
| e06 | "复杂问题理解不全面就开始实现，导致实现逻辑出错" | 1 条 | `requirement_analysis_before_coding` | pass | — | 直接命中 |
| e07 | "bash command中python代码闭合有问题" | 1 条 | `correct_shell_embedding` | pass | — | bash 嵌入他语言 |

### 扩展 case：覆盖剩余 v0 标签（5 条）

| id | raw_input | 期望拆分 | 期望 target_capability | 期望 route | 期望 drop_reason | 备注 |
|---|---|---|---|---|---|---|
| e08 | "修复bug前没有先复现问题就开始改代码" | 1 条 | `reproduce_before_fix` | pass | — | execution_control 子类 |
| e09 | "相同的错误处理方式重复尝试了五六次，没有切换策略" | 1 条 | `avoid_redundant_repetition` | pass | — | 循环卡死 |
| e10 | "实现完功能后没有验证，直接交给用户，结果还是有bug" | 1 条 | `self_verification` | pass | — | 缺自检 |
| e11 | "用户指出代码有bug后，模型说修复了但实际运行还是有同样的错误" | 1 条 | `effective_error_fix` | pass | — | 修复无效 |
| e12 | "编辑文件时找错了位置，改了别的函数" | 1 条 | `file_localization_and_edit` | pass | — | 定位失败 |

### 扩展 case：drop 场景（5 条）

| id | raw_input | 期望拆分 | 期望 target_capability | 期望 route | 期望 drop_reason | 备注 |
|---|---|---|---|---|---|---|
| e13 | "输出的UI样式不好看" | 1 条 | — | drop | `not_applicable` | 前端视觉，无硬信号 |
| e14 | "回答速度太慢" | 1 条 | — | drop | `not_applicable` | 基础设施/延迟，非模型能力 |
| e15 | "代码有问题" | 1 条 | — | drop | `ambiguous` | 太模糊，多种解释 |
| e16 | "整体感觉不太行，有很多小问题" | 1 条 | — | drop | `ambiguous` | 未具体化，多种可能 |
| e17 | "编辑文件出错了，工具调用也有问题，而且没有分析需求" | 3 条 | p1: file_; p2: wellformed_; p3: requirement_analysis_ | pass（全部）| — | 多问题拆分 + 全命中 |

### 扩展 case：边界/混合（3 条）

| id | raw_input | 期望拆分 | 期望 target_capability | 期望 route | 期望 drop_reason | 备注 |
|---|---|---|---|---|---|---|
| e18 | "生成的代码语法没错但逻辑完全不对" | 1 条 | — | drop | `label_diverged` | 语法对+逻辑错 → v0 无覆盖"逻辑正确性"标签，且不明确是哪种逻辑问题 |
| e19 | "模型一直在循环尝试同一种方法，最后超时了" | 1 条 | `avoid_redundant_repetition` | pass | — | 即使超时表述模糊，核心是重复循环 |
| e20 | "有时候行有时候不行" | 1 条 | — | drop | `ambiguous` | 完全无法确定指什么 |
