# 2026-04-03 M2/M4 分布式卡在 TaskCreated 修复记录

## 现象

- 两端日志都能到 `runner loaded` / `runner ready`。
- Master 也能打印 `Master broadcasting event to cluster: TaskCreated`。
- 但推理不继续，表现为任务创建后无 token 输出。

## 根因

这个问题不是单一故障，而是两个状态同步边界条件叠加：

1. Runner 子进程在某些时序下先上报 `TaskStatus.Complete`，后上报下一阶段 `RunnerStatusUpdated`。
2. 当 `RunnerLoaded` / `RunnerReady` 状态事件延迟或丢失时，计划器会卡在“实例未就绪”，导致 TextGeneration 任务无法被转发执行。

另外，JACCL 报错文案存在误导性：

- 日志看起来像环境变量缺失。
- 实际也可能是 JACCL backend 本身不可用（RDMA/ibverbs 不可用）。

## 修复内容

### 1) 监督器状态兜底（防止 warmup/ready 死锁）

文件：`src/exo/worker/runner/runner_supervisor.py`

- 当 `LoadModel` 任务完成且当前状态仍为 `RunnerLoading` 时，监督器补发一次 `RunnerLoaded`。
- 当 `StartWarmup` 任务完成且当前状态仍为 `RunnerWarmingUp` 时，监督器补发一次 `RunnerReady`。
- 去除临时调试 `print`，避免干扰日志。

目的：即使状态事件有时序抖动，也不会卡死在 TaskCreated 前后。

### 2) JACCL 可用性与诊断增强

文件：`src/exo/worker/engines/mlx/utils_mlx.py`

- 保持 `MLX_IBV_DEVICES` 为 JSON `null` 对角线格式。
- 对 `MLX_JACCL_COORDINATOR` 做规范化与格式校验。
- 调用 JACCL 前先检查 `mx.distributed.is_available(backend="jaccl")`。
- 若 backend 不可用，直接报清晰错误，避免误导为“环境变量缺失”。

### 3) Master 任务选择诊断

文件：`src/exo/master/main.py`

- 增加 TextGeneration 选实例日志：
  - 跳过未就绪实例时打印 runner 状态。
  - 创建任务时打印选中的 instance_id 与 runner 状态。
- 增加可选开关 `EXO_REQUIRE_READY_INSTANCE=1`：只允许分配到全 `Ready/Running` 的实例。

### 4) 先可用策略（无 RDMA 时自动走 Ring）

文件：`src/exo/master/placement.py`

- 增加开关 `EXO_DISABLE_JACCL=1`：请求 JACCL 时自动回退到 Ring。

## 推荐启动参数（先可用）

两台机器统一：

```bash
EXO_DISABLE_JACCL=1 EXO_REQUIRE_READY_INSTANCE=1 EXO_FAST_SYNCH=on EXO_PREFILL_STEP_SIZE=1 uv run exo -v
```

## 验证标准

满足以下日志链路即视为修复成功：

1. `runner loaded`
2. `Worker plan: StartWarmup`
3. `runner ready`
4. `Master broadcasting event to cluster: TaskCreated`
5. worker 进入 `runner running` 且出现 token/chunk 输出

## 备注

- 该修复优先保证 M2/M4 混合集群稳定运行。
- 若后续恢复 RDMA，可关闭 `EXO_DISABLE_JACCL` 逐步回归 JACCL 路径。
