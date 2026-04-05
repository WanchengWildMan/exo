# exo 环境变量速查表

所有 `EXO_*` 环境变量按功能分组，标注默认值和源码位置。

---

## 一、内存 & 放置

| 变量 | 默认值 | 说明 | 源码 |
|------|--------|------|------|
| `EXO_USE_TOTAL_MEMORY_FOR_PLACEMENT` | `1`（开） | 放置决策使用 **总内存** 而非可用内存。设为 `0` 恢复旧行为（用可用内存）。同时影响 `create_instance` 接口的前置检查。 | `placement_utils.py`, `placement.py`, `api/main.py` |
| `EXO_SKIP_PLACEMENT_MEMORY_CHECK` | 空（关） | 完全跳过放置阶段的内存检查。调试用，可能导致运行时 OOM。 | `placement_utils.py` |

---

## 二、KV Cache & 推理性能

| 变量 | 默认值 | 说明 | 源码 |
|------|--------|------|------|
| `EXO_KV_CACHE_BITS` | 空（不量化） | KV cache 量化位数，例如 `2`、`4`。降低显存占用，轻微损失精度。 | `constants.py` |
| `EXO_KV_GROUP_SIZE` | `32` | KV cache 量化分组大小。 | `constants.py` |
| `EXO_KV_BITS` | 空 | 关联量化 KV bits 参数（注意与 `EXO_KV_CACHE_BITS` 的区别：此参数走 mlx_lm 底层路径）。 | `constants.py` |
| `EXO_MAX_KV_SIZE` | `3200` | KV cache 最大容量（tokens）。超出后按 LRU 滚动。设为空/`none` 禁用限制。 | `constants.py` |
| `EXO_KEEP_KV_SIZE` | `1600` | KV cache 滚动时保留的 tokens 数。 | `constants.py` |
| `EXO_CACHE_GROUP_SIZE` | `64` | KV cache 分组大小（量化相关）。 | `constants.py` |
| `EXO_MEMORY_THRESHOLD` | 按机器大小：<16GB→0.70，<32GB→0.75，<64GB→0.80，≥128GB→0.85 | RAM 使用率超过此阈值时触发 LRU 驱逐旧 KV cache。范围 0.0–1.0。 | `cache.py` |
| `EXO_MAX_TOKENS` | `32168` | 单次请求最大输出 token 数（decode 阶段预算）。不影响 prefill 长度。 | `constants.py` |

---

## 三、Prefill 控制 & OOM 保护

| 变量 | 默认值 | 说明 | 源码 |
|------|--------|------|------|
| `EXO_PREFILL_STEP_SIZE` | `64` | Prefill 每步处理的 token 数。调大（如 `512`）可减少循环次数。Pipeline 模式下会被 `÷ min(4, world_size)` 缩小。 | `generate.py` |
| `EXO_PREFILL_MEMORY_THRESHOLD` | `0.92` | Prefill 过程中内存使用率超过此值时，**中止 prefill** 并返回错误，防止 OOM。范围 0.0–1.0。 | `generate.py` |
| `EXO_PREFILL_MEMORY_CHECK_INTERVAL` | `8` | 每隔 N 个 prefill step 检查一次内存（并打印日志）。调小可更早发现 OOM 风险。 | `generate.py` |
| `EXO_PREFILL_MAX_METAL_GB` | `0`（自动=总内存-1.5GB） | Metal GPU 内存预填充上限（GB）。超过此值触发 OOM 中止。`0` 表示自动计算。 | `generate.py` |
| `EXO_PREFILL_SYNC_TIMEOUT` | `60` | Pipeline 并行 prefill 时，单次分布式 sync（`all_sum`/`all_gather`）的超时秒数。超时后抛出 `PrefillSyncTimeout` 终止 prefill，防止对端崩溃后本端无限阻塞。设为 `0` 禁用。 | `generate.py` |
| `EXO_SKIP_WARMUP` | `1`（跳过） | 设为 `0` 恢复启动时的推理预热（默认已跳过）。 | `generate.py` |
| `EXO_GENERATION_MEMORY_CHECK_INTERVAL` | `64` | 生成阶段每隔 N 步检查 Metal 内存并记录日志。 | `batch_generate.py` |

---

## 四、System Prompt 裁剪

> 用于截断来自 OpenClaw / 其他 AI 助手注入的过长 system prompt。

| 变量 | 默认值 | 说明 | 源码 |
|------|--------|------|------|
| `EXO_STRIP_SKILLS_BLOCK` | 空（关） | 设为 `1` 时，从 system prompt 中完整剥除 `<available_skills>…</available_skills>` 块。 | `adapters/chat_completions.py` |
| `EXO_SKILLS_DESC_MAX_CHARS` | 空（不裁剪） | 保留 skills 块结构，但每条技能描述超过 N 字符时截断。例如 `60`。与 `EXO_STRIP_SKILLS_BLOCK` 互斥（后者优先）。 | `adapters/chat_completions.py` |
| `EXO_MAX_SYSTEM_PROMPT_CHARS` | 空（不裁剪） | System prompt 总字符超过 N 时截断并加 `[trimmed]` 标记。兜底保险，在上面两个开关之后执行。 | `adapters/chat_completions.py` |

**优先级**：`STRIP_SKILLS_BLOCK` > `SKILLS_DESC_MAX_CHARS`，两者之后再执行 `MAX_SYSTEM_PROMPT_CHARS`。

---

## 4.5、SSD KV Cache 持久化

| 变量 | 默认值 | 说明 | 源码 |
|------|--------|------|------|
| `EXO_SSD_CACHE_ENABLED` | `1`（开） | 是否启用 SSD KV 缓存持久化。LRU 淘汰的 KV cache 条目会序列化到 SSD，再次需要时从 SSD 恢复而非重新 prefill。 | `ssd_kv_cache.py` |
| `EXO_SSD_CACHE_DIR` | `~/.cache/exo/kv_cache` | SSD 缓存存储目录。 | `ssd_kv_cache.py` |
| `EXO_SSD_CACHE_MAX_GB` | `10` | SSD 缓存最大总容量（GB）。超出时淘汰最旧条目。 | `ssd_kv_cache.py` |

---

## 五、并发 & 调度

| 变量 | 默认值 | 说明 | 源码 |
|------|--------|------|------|
| `EXO_MAX_CONCURRENT_REQUESTS` | `8` | 单节点最大并发推理请求数。 | `shared/constants.py` |
| `EXO_REQUIRE_READY_INSTANCE` | 空（关） | 设为 `1` 时，只向状态为 Ready/Running 的实例分配请求。 | `master/main.py` |
| `EXO_NO_BATCH` | 空（关） | 禁用 batch 推理模式（分布式场景下自动启用）。 | `runner/runner.py` |

---

## 六、网络 & 节点通信

| 变量 | 默认值 | 说明 | 源码 |
|------|--------|------|------|
| `EXO_DISABLE_JACCL` | 空（关） | 设为 `1` 时，即便请求 JACCL（RDMA）模式也强制回退到 MlxRing。 | `placement.py` |
| `EXO_FAST_SYNCH` | 由 `--fast-synch` 参数控制 | 控制节点同步模式。`on`/`off`。 | `main.py`, `bootstrap.py` |
| `EXO_BOOTSTRAP_PEERS` | 空 | 逗号分隔的初始 peer 地址列表，用于覆盖默认的节点发现。 | `main.py` |

---

## 七、其他

| 变量 | 默认值 | 说明 | 源码 |
|------|--------|------|------|
| `EXO_OFFLINE` | `false` | 离线模式，禁止下载模型。 | `main.py`, `shared/constants.py` |
| `EXO_ENABLE_IMAGE_MODELS` | `false` | 启用图像生成模型支持（FLUX 等）。 | `shared/constants.py` |
| `EXO_TRACING_ENABLED` | `false` | 启用追踪（tracing）。 | `shared/constants.py` |
| `EXO_MACMON_PATH` | 自动寻找 `macmon` | 指定 macmon 二进制路径，用于 Apple Silicon 内存监控。 | `info_gatherer.py` |

---

## 快速参考：长文本分布式推理推荐配置

```bash
# 所有节点都需要设置
export EXO_USE_TOTAL_MEMORY_FOR_PLACEMENT=1   # 用总内存做放置决策（默认已开）
export EXO_KV_CACHE_BITS=2                    # 大幅降低 KV cache 内存（约降 8x）
export EXO_MAX_KV_SIZE=3200                   # 限制 KV cache token 容量
export EXO_PREFILL_STEP_SIZE=512              # 加大 prefill 步长（加速）
export EXO_MAX_TOKENS=4096                    # 限制单次最大输出 token

# prompt 很短但 system prompt 较长时（如使用 OpenClaw）
export EXO_SKILLS_DESC_MAX_CHARS=60           # 每条技能描述截断至 60 字符
```
