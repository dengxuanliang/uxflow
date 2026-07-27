## Summary / 概要
<!-- What & why. Reference any issue: Closes #N / 是什么与为什么。引用相关 issue: Closes #N -->

## Checklist / 自检清单
- [ ] `uv run ruff check` is clean / ruff 通过
- [ ] `uv run pytest -m "not requires_model"` is green / 纯逻辑测试通过
- [ ] New tests added for any new behavior; `@pytest.mark.requires_model` only when genuinely model-bound / 新行为已加测试; 仅在确需真实模型时才打 requires_model 标记
- [ ] No secrets, real keys, or `.env` content committed / 未提交密钥、真实 key 或 `.env` 内容
- [ ] SPDX header on new source files (`# SPDX-License-Identifier: Apache-2.0`) / 新源文件带 SPDX 头

## Notes / 备注
<!-- Anything reviewers should know. / 评审者需要知道的事项。 -->
