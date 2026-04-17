# MLX 内存池泄漏修复

## 2026-04-05 P0 追加结论

当前两台 16GB Mac Mini 的首要目标不是继续冲更长上下文，而是先避免整机卡死、自动关机、
重复启动导致的混乱状态。

新增确认到的两个风险：

- `EXO_PREFILL_MEMORY_THRESHOLD=0.95` 对 16GB macOS 机器过高。此时系统只剩几百 MB 可用内存，
  已经来不及给内核和图形栈留缓冲，机器可能在 exo 抛出 PrefillOOM 前就先卡死。
- 启动脚本此前没有在启动前阻止 `52415` 端口已被占用，重复启动会造成 `OSError: [Errno 48] Address already in use`，
  进一步放大状态混乱与内存压力。

因此本轮策略调整为：

- 先把默认运行档位降为“保活优先”
- 先阻止重复启动
- token 存盘 / 外部 offload 仍可继续评估，但不作为当前 P0 的第一步

## 问题描述

exo 运行多次请求后，系统内存基线持续上涨：

- 第1次请求(3263 token)后：91.8%
- 第2-3次请求后：约 92-94%
- 第4次请求(70 token 短请求)：开始时 94.6%，16 token 后触发 95% OOM 阈值失败

## 根本原因

`mx.clear_cache()` 只在 `runner.py` 的 `shutdown()` 方法中调用，每次请求完成后从不调用清理。

MLX 使用内存池管理 GPU/Metal 内存：计算完成后的临时张量（激活值、中间计算结果）引用已丢弃，但内存仍留在池中不归还操作系统。多次请求后这些"死"张量累积，导致基线内存持续上涨。

## 修复位置

`src/exo/worker/runner/llm_inference/runner.py` - `handle_generation_tasks()` 方法

在 `while self.active_tasks:` 循环结束后（全部任务完成后），更新状态到 RunnerReady 之前，添加：

```python
mx.clear_cache()
import gc
gc.collect()
self.update_status(RunnerReady())
```

## 注意事项

- `mx.clear_cache()` 只释放无 Python 引用的"死"张量，不影响：
  - 模型权重（model 对象持有引用）
  - KV 前缀缓存（kv_prefix_cache 持有引用）
  - 活跃请求的 KV cache
- Qwen3.5-9B 混合架构：GatedDeltaNet 层缓存 O(1) 不随上下文增长，标准 Attention 层缓存由 RotatingKVCache(max_kv_size=2000) 限制上界
- 同步文件：M4 和 M2 均已同步（M2 通过 /Volumes/xiewancheng/exo/）

## 建议的安全档参数

对当前 16GB / macOS / 双机分片场景，推荐先使用以下更保守的默认值：

- `EXO_PREFILL_STEP_SIZE=16`
- `EXO_PREFILL_EVAL_INTERVAL=2`
- `EXO_PREFILL_MEMORY_THRESHOLD=0.90`
- `EXO_MAX_SYSTEM_PROMPT_CHARS=2000`

这组参数的目标不是最快，而是尽量在系统级冻结前提前中止或减小单步峰值。
