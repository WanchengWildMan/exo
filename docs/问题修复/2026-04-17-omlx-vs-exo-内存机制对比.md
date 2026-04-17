# oMLX vs exo 内存管理机制对比分析

日期: 2026-04-17
背景: Mac Mini M4 16GB 运行 Qwen3.5-9B-MLX-4bit，8个token即OOM死机

## 一、问题的本质

16GB 统一内存分配：
- macOS + 系统服务: ~3-4GB
- 模型权重 (9B × 4bit): ~5GB  
- **剩余可用: ~7-8GB** (其中 GPU 和 CPU 共享)
- KV cache + 前向传播中间激活 + MLX 计算图 需要在这 7-8GB 里运行

**核心矛盾**: exo 没有进程级内存预算机制，模型加载后不知道自己还剩多少内存可用于 KV cache 和推理。

## 二、oMLX 的内存管理架构

```
FastAPI Server
    │
    ├── ProcessMemoryEnforcer (进程级内存上限，默认 RAM - 8GB)
    │
    ├── Scheduler (FCFS, 并发控制)
    │   └── mlx-lm BatchGenerator
    │
    └── Cache Stack
        ├── PagedCacheManager (GPU, 块级, CoW, 前缀共享)
        ├── Hot Cache (内存热层, write-back)
        └── PagedSSDCacheManager (SSD冷层, safetensors格式)
```

### 关键模块:
| 文件 | 大小 | 功能 |
|------|------|------|
| `omlx/cache/paged_cache.py` | 59KB | vLLM式块级KV cache管理 |
| `omlx/cache/paged_ssd_cache.py` | 74KB | SSD冷层持久化 |
| `omlx/cache/prefix_cache.py` | 99KB | 块级前缀共享 |
| `omlx/cache/tiered_manager.py` | 12KB | 热层+冷层协调 |
| `omlx/memory_monitor.py` | 15KB | Metal内存实时监控 |
| `omlx/process_memory_enforcer.py` | 12KB | 进程内存强制限制 |
| `omlx/scheduler.py` | 186KB | 请求调度+内存预算 |

## 三、核心差异对比

### 差异1: 块级 vs 条目级 KV Cache

**oMLX (块级 Paged KV Cache)**:
- KV cache 被切分为固定大小的 block（类似虚拟内存页）
- 请求间可共享前缀 block (Copy-on-Write)
- 内存紧张时可逐 block 驱逐: 保留热块在 RAM，冷块存 SSD
- 同一请求的 KV cache 可以部分在 RAM、部分在 SSD
- **优势**: 精细粒度内存管理，不需要完整缓存就能继续推理

**exo (条目级 RotatingKVCache)**:
- 每个请求一个完整的 RotatingKVCache (max_size=1500)
- 整个缓存要么全在内存，要么全部驱逐
- 无法部分驱逐:不能只把某些层或某些位置的 KV 放到 SSD
- 前缀共享是条目级:两个请求即使前缀完全相同，也各自持有独立缓存
- **劣势**: 内存效率低，粒度太粗

### 差异2: 进程级内存预算 vs 无预算

**oMLX**:
```
--max-process-memory 80%    (默认: RAM - 8GB)
--max-model-memory 32GB     (模型层面限制)
```
- 16GB 机器: 进程上限 = 16 - 8 = 8GB
- 模型权重 5GB → 只允许 3GB 用于 KV cache + 推理
- **主动监控**: 超过预算就停止接受新请求/驱逐模型
- `ProcessMemoryEnforcer` 持续运行，检查实际进程内存

**exo**:
- 没有进程级内存预算
- 没有模型级内存限制
- 仅在 prefill 时检查 `mx.get_active_memory()`
- **被动反应**: 等到内存快爆了才 abort，有时来不及（已经 OOM）
- **generation (decode) 阶段几乎无内存检查**

### 差异3: 内存预留机制

**oMLX**:
- 知道系统总内存、模型占用、当前 KV cache 占用
- 计算出"还能分配多少 block"，超过就拒绝/排队
- 模型 unload 机制:内存不够时自动卸载 LRU 模型

**exo**:
- 不知道模型实际占了多少 Metal 内存
- 不知道 RotatingKVCache 实际占了多少内存
- 无法预测 forward pass 的峰值内存需求（虽然有 _estimate_prefill_peak_bytes 但仅用于 prefill 前检查）
- **decode 阶段无任何预算/检查**

### 差异4: SSD 热冷分层策略

**oMLX**:
- **热层**: RAM 中的 KV 块，用于当前活跃请求
- **冷层**: SSD 上的 KV 块，通过 safetensors 持久化
- `--hot-cache-max-size 20%` 明确限制热层大小
- 块级 write-back:只有修改过的块需要写回 SSD
- 根据访问频率自动迁移:热→冷(驱逐)、冷→热(命中)

**exo**:
- SSD 缓存仅用于驱逐的**完整**前缀缓存条目
- 没有热层大小限制（由 EXO_MEMORY_THRESHOLD 间接控制）
- 条目级:整个缓存序列化/反序列化，不能只存部分
- 没有自动迁移:驱逐就是删掉内存版本，需要时整个从 SSD 加载

### 差异5: 生成阶段内存安全

**oMLX**:
- `memory_monitor.py` 持续监控 Metal 内存
- 即使在 decode/generation 阶段也有内存保护
- 如果内存紧张:暂停请求 → 驱逐冷块 → 释放内存 → 继续

**exo**:
- prefill 阶段: ✅ 有 `_check_memory_during_prefill()` + adaptive chunking
- generation (decode) 阶段: ❌ **基本没有内存保护**
- `EXO_GENERATION_MEMORY_CHECK_INTERVAL=64` 存在但检查粒度太粗
- 一旦 decode 开始，没有办法中途停止释放内存

## 四、为什么 8 个 token 就崩溃

推测的内存时间线:
```
[启动]            系统占用 ~3.5GB，剩余 ~12.5GB
[加载模型]        权重 ~5GB → Metal 内存 ~8.5GB，剩余 ~7.5GB
[创建KV Cache]    RotatingKV(1500, 2bit) × 32层 ≈ 前几个状态较小
                  但 ArraysCache (GatedDeltaNet) 的 SSM 状态也初始化
[Prefill prompt]  输入 tokens → 逐 chunk 前向传播
                  每步 eval+sync+clear ✅
                  prefill 完成后清理 ✅
[Decode token 1-8] 每个 token:
                  - 前向传播分配中间激活(注意力矩阵等)
                  - KV cache append 新 token
                  - SSM state 更新 (GatedDeltaNet)
                  - eval 只在 has_work 循环中做(不频繁)
                  - ⚠️ Metal command buffer 可能累积
                  → 第8个 token 时累积的中间张量 + KV + 权重 > 16GB
                  → macOS kills the process / Metal crashes
```

关键: **decode 阶段缺少 mx.synchronize() + mx.clear_cache()**

## 五、立即可做的改进 (不需要大重构)

### P0: Generation (decode) 阶段添加内存检查+清理 ✅ 已完成
- ~~在 decode 循环中每 N 步做 `mx.synchronize() + mx.clear_cache()`~~
- ~~参照 oMLX 的做法:每步都检查，不只是每 64 步~~
- 实现: `EXO_DECODE_SYNC_INTERVAL=1`（默认每步 sync+clear）

### P1: 进程级内存预算
- 启动时计算: `budget = total_ram - 8GB`
- 模型加载后: `kv_budget = budget - model_size`
- 如果 `kv_budget < 安全阈值`: 拒绝加载/使用更小的 max_kv_size

### P2: Decode 阶段的自适应 max_tokens ✅ 已完成
- ~~如果检测到 Metal 内存接近上限:提前结束生成 (返回 finish_reason: "length")~~
- ~~比 crash 好得多~~
- 实现: `EXO_DECODE_MAX_METAL_GB`（默认 total_ram-1.5GB），超限时 finish_reason="length"
- 详见: [2026-04-17-decode内存安全阀.md](2026-04-17-decode内存安全阀.md)

### P3: 块级 KV cache (长期)
- 将 RotatingKVCache 替换为块级实现
- 这是 oMLX 的核心竞争力，需要大规模重构

## 六、关于"把缓存存到外置盘"

已将 `EXO_SSD_CACHE_DIR` 配置到 `/Volumes/External/exo_cache/kv_cache`。

但需要明确：
- ✅ SSD 缓存（驱逐的 KV 条目）可以存外置盘 → 不占内部 SSD 空间
- ❌ **这不能解决 OOM 问题** → OOM 是因为 Metal 统一内存不够
- MLX 张量在推理时必须在 Metal 统一内存中，无法使用外置盘
- 外置盘的用途是当做"冷存储"，让驱逐的 KV 有地方放
