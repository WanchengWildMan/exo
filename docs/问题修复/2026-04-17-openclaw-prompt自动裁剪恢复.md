# 2026-04-17 OpenClaw prompt 自动裁剪恢复

## 现象

此前 exo 对 OpenClaw 注入的超长 system prompt 有专门裁剪保护，
可将总长度压到约 `3000` 字符；近期如果直接使用 `uv run exo`、
`scripts/hot-reload.sh` 或其他未注入环境变量的启动方式，
同样的 OpenClaw prompt 会以接近原始长度进入推理链路，
表现上像是“从 3k 退回到 16k+”。

## 根因

`src/exo/api/adapters/chat_completions.py` 中的裁剪代码并未删除，
但它完全依赖显式环境变量：

- `EXO_STRIP_SKILLS_BLOCK`
- `EXO_SKILLS_DESC_MAX_CHARS`
- `EXO_MAX_SYSTEM_PROMPT_CHARS`

只要启动入口没有注入这些变量，就不会发生任何裁剪。

与此同时，仓库里的保守启动脚本 `scripts/start-my.sh`
仍然显式设置了 `EXO_MAX_SYSTEM_PROMPT_CHARS=2000`，
这是 2026-04-05 为 16GB 机器“保活优先”做的降档策略，
并不等同于 OpenClaw 默认裁剪能力本身。

## 修复

在 `chat_completions` 适配层新增 OpenClaw prompt 特征识别。

当且仅当满足以下条件时，自动启用 OpenClaw 默认裁剪档：

1. system prompt 呈现 OpenClaw 特征，例如：
   - `You are a personal assistant running inside OpenClaw.`
   - `## OpenClaw CLI Quick Reference`
   - `<available_skills>`
2. 未显式设置全局裁剪开关。

自动默认档如下：

- `EXO_OPENCLAW_SKILLS_DESC_MAX_CHARS=30`
- `EXO_OPENCLAW_MAX_SYSTEM_PROMPT_CHARS=3000`

显式全局配置仍然优先：

- 若设置了 `EXO_STRIP_SKILLS_BLOCK` / `EXO_SKILLS_DESC_MAX_CHARS` /
  `EXO_MAX_SYSTEM_PROMPT_CHARS`，则继续按显式配置执行。
- `scripts/start-my.sh` 的 `2000` 保守档保持不变，用于低内存机器。

## 结果

- 裸启动 exo 时，OpenClaw 的超长 system prompt 会自动恢复到默认 3k 档。
- 低内存保守启动脚本仍可继续显式压到 2k，不与本次恢复冲突。
- 日志会额外标记裁剪策略来源，例如 `openclaw-default` 或 `explicit-env`，
  便于区分是自动默认档还是手工环境变量在生效。
