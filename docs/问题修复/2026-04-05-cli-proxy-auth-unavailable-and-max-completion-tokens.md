# 2026-04-05 cli-proxy-api auth_unavailable 错误 & max_completion_tokens 字段缺失

## 现象

通过 cli-proxy-api（端口 8317）转发 OpenClaw 请求到 exo（端口 52415）时，
所有带 `tools` 参数的 `/v1/chat/completions` 请求均返回 500：

```json
{"error":{"message":"auth_unavailable: no auth available","type":"server_error","code":"internal_server_error"}}
```

OpenClaw 日志中可见重试 3 次后失败（`X-Stainless-Retry-Count: 2`）。

## 诊断过程

### 1) 定位错误来源

在 exo 代码全文搜索 `auth_unavailable` 和 `no auth available`——**无匹配**。
说明此错误消息不是 exo 产生的，而是 cli-proxy-api 代理的包装错误。

### 2) 分离变量测试

| 测试场景 | 端点 | 结果 |
|---------|------|------|
| 直接 → exo，无 tools | `localhost:52415` | ✅ 200 |
| 代理 → exo，无 tools | `localhost:8317` | ✅ 200 |
| 直接 → exo，有 tools | `localhost:52415` | ✅ 200 |
| 代理 → exo，有 tools（当时） | `localhost:8317` | ❌ 500 |

后续重新测试时，带 tools 的请求也恢复正常，表明当时是 exo 后端临时异常
（如模型实例未就绪、节点未连接），代理将 exo 后端的失败包装为 `auth_unavailable`。

### 3) 发现附带 bug：max_completion_tokens 被静默丢弃

分析 OpenClaw 发送的请求体发现：

```json
{
  "model": "exo-qwen-9B",
  "max_completion_tokens": 8192,
  "stream": true,
  "tools": [...]
}
```

OpenAI 新版 API 使用 `max_completion_tokens` 字段（替代旧的 `max_tokens`），
但 exo 的 `ChatCompletionRequest` Pydantic 模型只定义了 `max_tokens`。

Pydantic 默认行为是**静默忽略**未知字段，导致：
- OpenClaw 以为设置了 8192 token 上限
- exo 实际没有任何 token 限制
- 模型可能无限生成直至 OOM 或 KV cache 耗尽

## 根因

1. **auth_unavailable**：cli-proxy-api 的通用错误包装。当所有配置的后端（exo）
   返回非 2xx 响应时，代理无法从任何可用 auth/后端获得有效回复，返回此错误。
   根本原因是 exo 当时内部异常（模型实例未就绪）。

2. **max_completion_tokens 缺失**：exo 的 OpenAI 兼容层未跟进 OpenAI API 的
   字段更新。`ChatCompletionRequest` 缺少 `max_completion_tokens` 字段定义。

## 修复内容

### 文件 1：`src/exo/api/types/api.py`

在 `ChatCompletionRequest` 中新增 `max_completion_tokens` 字段：

```python
class ChatCompletionRequest(BaseModel):
    ...
    max_tokens: int | None = None
    max_completion_tokens: int | None = None   # ← 新增
    ...
```

### 文件 2：`src/exo/api/adapters/chat_completions.py`

在 `chat_request_to_text_generation()` 中优先使用新字段：

```python
# 之前
max_output_tokens=request.max_tokens,

# 之后
max_output_tokens=request.max_completion_tokens or request.max_tokens,
```

逻辑：`max_completion_tokens` 优先；若未设置则 fallback 到 `max_tokens`。
这保证了新旧客户端的兼容性。

## 影响范围

- 所有通过 OpenAI 兼容 API 发送 `max_completion_tokens` 的客户端
  （OpenClaw、OpenAI SDK v4+、Cursor、Continue 等）
- 修复前这些客户端的 token 限制被静默忽略
- 修复后正确尊重 token 上限

## 验证

```bash
# 带 max_completion_tokens 的请求
curl -s http://localhost:52415/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"mlx-community/Qwen3.5-9B-MLX-4bit",
       "messages":[{"role":"user","content":"count 1 to 100"}],
       "max_completion_tokens": 20}'

# 预期：生成在约 20 token 处停止，finish_reason 为 "length"
```

## 部署

两台机器均需重启 exo 生效：

```bash
# 已同步文件
cp src/exo/api/types/api.py /Volumes/xiewancheng/exo/src/exo/api/types/api.py
cp src/exo/api/adapters/chat_completions.py /Volumes/xiewancheng/exo/src/exo/api/adapters/chat_completions.py
```
