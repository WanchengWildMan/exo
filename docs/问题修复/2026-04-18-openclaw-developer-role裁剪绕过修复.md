# OpenClaw "developer" role 导致 system prompt 裁剪完全失效

日期：2026-04-18

## 问题现象

OpenClaw 通过 exo Chat Completions API 发送请求时，system prompt 约 57k 字符（~19k tokens），
`EXO_MAX_SYSTEM_PROMPT_CHARS=2000` 和 `EXO_SKILLS_DESC_MAX_CHARS=30` 均已设置，
但裁剪从未生效——exo 日志中没有任何 `[System prompt trim:...]` 记录。

## 根因

OpenClaw 使用 OpenAI 新规范中的 `role: "developer"` 而非传统的 `role: "system"` 发送系统指令。
exo 的 `chat_request_to_text_generation()` 只在 `msg.role == "system"` 时触发 `trim_system_prompt()`，
`"developer"` 消息走的是 `else` 分支，被当作普通消息处理，完全绕过了裁剪逻辑。

## 修复

### `src/exo/api/adapters/chat_completions.py`

1. **扩展 role 匹配**：`msg.role == "system"` → `msg.role in ("system", "developer")`
2. **保留原始 role**：`chat_template_messages` 中保留 `msg.role`（而非硬编码 `"system"`），
   确保 chat template 正确使用 `developer` role（Qwen3 等模型支持）
3. **移除冗余分支**：`else` 分支中 `("user", "assistant", "developer")` 改为 `("user", "assistant")`，
   因为 `developer` 已被 `if` 捕获

### 影响

- 修复后，OpenClaw 的 ~57k 字符 system prompt 将被裁剪到 ~2000 字符
- 预计从 ~19k tokens 降至 ~700 tokens（节省 ~18k tokens）
- 大幅减少 prefill 时间和内存占用

## 验证

```bash
uv run pytest src/exo/api/tests/test_chat_completions_adapter.py -v
```

检查 exo 日志中出现 `[System prompt trim:openclaw-default]` 或 `[System prompt trim:explicit-env]`。
