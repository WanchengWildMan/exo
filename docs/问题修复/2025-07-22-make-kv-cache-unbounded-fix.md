# 2025-07-22 make_kv_cache 无限增长修复

## 诊断

### 核心问题
机器在推理过程中反复冻死/OOM 重启。

### 根因分析

**根因 1：`model.make_cache()` 返回无界 KVCache，`MAX_KV_SIZE` 完全失效**

`cache.py` 的 `make_kv_cache()` 函数中，当模型实现了 `make_cache()` 方法时（如 Qwen3.5），
直接返回 `model.make_cache()` 的结果，跳过了所有 `MAX_KV_SIZE` / `KV_CACHE_BITS` 逻辑。

Qwen3.5 是混合架构：
- ~7/8 层是 GatedDeltaNet（线性注意力）→ 返回 `ArraysCache`，O(1) 固定大小，无问题
- ~1/8 层是标准 Attention → 返回 **裸 `KVCache()`**，**无上限**，不量化

后果：环境变量 `EXO_MAX_KV_SIZE=2000` / `EXO_KV_CACHE_BITS=2` 对 Qwen3.5 都是死代码。
Attention 层的 KV 缓存以 FP16 无限增长。

**根因 2：请求完成后不释放 MLX 缓存**

`batch_generate.py` 的 `step()` 方法中，任务完成后 (`is_done`) 只做了
`del self._active_tasks[response.uid]`，从未调用 `mx.clear_cache()` 或 `gc.collect()`。
MLX Metal 内存池单调增长。

**根因 3：`start-my.sh` 参数未实际更新到保活值**

此前讨论过的"保活优先"参数（MAX_KV_SIZE=1500, MAX_TOKENS=2048, PREFILL_MEMORY_THRESHOLD=0.88）
实际并未保存到文件。当前文件仍为：MAX_KV_SIZE=2000, MAX_TOKENS=4096, PREFILL_MEMORY_THRESHOLD=0.90。

## 修复方案

### FIX 1：`cache.py` — 后处理 `model.make_cache()` 结果
调用 `model.make_cache()` 后，遍历返回的缓存列表：
- `ArraysCache` → 保持原样（GatedDeltaNet，O(1)）
- `KVCache`（裸）→ 替换为 `RotatingKVCache(max_size=MAX_KV_SIZE, keep=keep)`

这样 GatedDeltaNet 的缓存格式不受影响，而 Attention 层的 KV 缓存被限制大小。

### FIX 2：`batch_generate.py` — 请求完成后清理
任务完成时调用 `mx.clear_cache()` + `gc.collect()` 释放 Metal 内存。

### FIX 3：`start-my.sh` — 应用保活参数
- `EXO_MAX_KV_SIZE=1500`
- `EXO_MAX_TOKENS=2048`
- `EXO_PREFILL_MIN_AVAILABLE_GB=0.5`（替代旧的百分比阈值）

### FIX 4：`generate.py` — 改为绝对可用内存判断
macOS 的 `psutil.percent` 包含可回收文件缓存，导致百分比阈值过于保守。
改为 `psutil.virtual_memory().available` 绝对值判断（默认 < 0.5GB 才中止）。

### FIX 5：`batch_generate.py` — prefill 前主动清理
在 `submit()` 开头添加 `mx.clear_cache()` + `gc.collect()` + KV prefix cache 驱逐，
降低 prefill 前的内存基线。

## 内存估算（单节点 16 层，Qwen3.5-9B-MLX-4bit）

| 项目 | 估算内存 |
|------|---------|
| macOS + 后台 | ~4-5GB |
| 模型权重（16 层 4bit） | ~2.5GB |
| GatedDeltaNet 状态（~14 层） | ~128MB×14≈1.8GB |
| Attention KV cache（~2 层，1500 token，FP16） | 2×28KB×1500≈84MB |
| KV prefix cache 深拷贝（1 份） | ~84MB |
| MLX 内存池开销 | ~1-2GB |
| **总计** | **~11-12GB** |
| **剩余** | **~4-5GB** |

有 RotatingKVCache 限制后，Attention KV cache 不可能超过 84MB/份。
主要内存压力来自 GatedDeltaNet 状态矩阵（固定，不可压缩）和模型权重。
