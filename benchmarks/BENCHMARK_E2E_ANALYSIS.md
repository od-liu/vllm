# benchmark_e2e_metrics.py 实现逻辑详细分析

## 概述

`benchmark_e2e_metrics.py` 是专门用于测量 vLLM 端到端（E2E）性能指标的脚本，主要测量：
- **TTFT (Time To First Token)**: 首 token 延迟
- **TPOT (Time Per Output Token)**: 每个输出 token 的平均时间
- **E2E Latency**: 端到端总延迟
- **Queued Time**: 请求在队列中的等待时间
- **Throughput**: 吞吐量（tokens/秒）

## 架构设计

### 核心设计原则

1. **单进程 vs 多进程模式**：根据 `data_parallel_size` 自动选择
2. **Trace 数据驱动**：必须使用真实的 trace 数据（不支持合成数据）
3. **无算子 Profiling**：Phase 1 不启用算子级性能分析，避免开销影响 E2E 指标
4. **结果聚合**：多进程模式下自动聚合所有 rank 的结果

## 详细实现流程

### 1. 入口函数：`main()`

```python
def main():
    # 1. 解析命令行参数
    #    --config: 配置文件路径（必需）
    #    --phase: 必须为 1（验证）
    #    --verbose: 可选，启用详细日志
    
    # 2. 验证配置
    #    - 检查 enable_e2e_metrics（如果为 False，自动设置为 True）
    #    - 检查 use_trace_data（必须为 True）
    
    # 3. 加载配置文件
    config = load_config_from_file(args.config)
    
    # 4. 运行 benchmark
    results = run_e2e_benchmark(config)
```

**关键验证点**：
- Phase 必须为 1（E2E metrics 专用）
- Trace 数据必须启用
- 配置文件必须包含 `config` 变量

---

### 2. 主执行函数：`run_e2e_benchmark()`

这是整个系统的调度器，根据 `data_parallel_size` 选择执行路径：

```python
def run_e2e_benchmark(config: BenchmarkConfig):
    config.validate()
    
    if config.data_parallel_size > 1:
        # 多进程模式
        return run_e2e_benchmark_multi_rank(config)
    else:
        # 单进程模式
        result = run_e2e_benchmark_single_rank(config, dp_rank=0, dp_size=1)
        # 格式化输出以保持一致性
        return format_single_rank_result(result)
```

**决策逻辑**：
- `DP = 1`: 单进程执行，直接返回结果
- `DP > 1`: 多进程执行，启动多个独立进程，然后聚合结果

---

### 3. 单进程执行：`run_e2e_benchmark_single_rank()`

这是核心执行函数，包含完整的 E2E benchmark 流程。

#### 3.1 初始化阶段（42-105行）

**A. 日志配置**
```python
# 为每个 DP rank 创建带前缀的日志格式
rank_prefix = f"[DP{dp_rank}]"
# 自定义 Formatter 在所有日志前添加 rank 前缀
```

**目的**：在多进程模式下，可以区分不同 rank 的日志输出

**B. 配置验证**
```python
config.validate()  # 验证所有必需参数
```

**C. Trace 数据验证**
```python
if not config.use_trace_data:
    raise ValueError("E2E benchmark requires trace data")
```

**关键约束**：E2E benchmark **必须**使用 trace 数据，不支持合成数据模式

#### 3.2 临时文件管理（107-121行）

```python
# 创建临时目录结构
config_subdir = f"tp{config.tensor_parallel_size}_dp{dp_size}"
temp_dir = Path(f"./tmp_benchmark/{config_subdir}")
temp_dir.mkdir(parents=True, exist_ok=True)

# 创建临时输出文件
temp_file_prefix = f'e2e_bench_dp{dp_rank}_' if dp_size > 1 else 'e2e_bench_'
temp_file = tempfile.NamedTemporaryFile(
    mode='w', suffix='.json', delete=False,
    dir=str(temp_dir), prefix=temp_file_prefix
)
```

**文件命名规则**：
- 单进程：`e2e_bench_{random}.json`
- 多进程：`e2e_bench_dp{rank}_{random}.json`
- 目录：`tmp_benchmark/tp{TP}_dp{DP}/`

**目的**：
- 避免多进程文件冲突
- 便于调试和追踪
- 支持结果聚合

#### 3.3 vLLM 引擎初始化（123-149行）

```python
llm_kwargs = {
    "model": config.model_path,
    "tensor_parallel_size": config.tensor_parallel_size,
    "pipeline_parallel_size": config.pipeline_parallel_size,
    "gpu_memory_utilization": config.gpu_memory_utilization,
    "dtype": config.dtype,
    "disable_log_stats": False,  # 关键：启用 metrics 收集
}

if not config.enable_cuda_graph:
    llm_kwargs["enforce_eager"] = True  # 禁用 CUDA graph 以获得准确测量

llm = LLM(**llm_kwargs)
```

**关键配置**：
- `disable_log_stats=False`: **必须**启用，否则 `RequestOutput.metrics` 为空
- `enforce_eager=True`: 禁用 CUDA graph，避免优化影响测量准确性

**注意**：此阶段专注于 E2E 指标收集，不进行算子级别的 profiling

#### 3.4 Trace 数据加载与分片（151-177行）

**A. 加载 Trace 数据**
```python
trace_loader = TraceLoader(
    config.trace_file_path,
    time_range_minutes=config.trace_time_range_minutes,
)
trace_requests = trace_loader.load_requests()
```

**TraceLoader 功能**：
- 从 JSONL 文件读取请求数据
- 按时间范围过滤（如果指定）
- 按 timestamp 排序
- 返回 `List[TraceRequest]`

**TraceRequest 结构**：
```python
@dataclass
class TraceRequest:
    chat_id: int              # 会话 ID
    parent_chat_id: int       # 父会话 ID
    timestamp: float          # 相对时间戳（秒）
    input_length: int         # 输入 token 数
    output_length: int        # 输出 token 数
    type: str                 # 请求类型
    turn: int                 # 对话轮次
    hash_ids: List[int]       # 哈希 ID 列表（用于 token 映射）
```

**B. Round-Robin 数据分片（多进程模式）**
```python
if dp_size > 1:
    rank_requests = [
        req for i, req in enumerate(trace_requests) 
        if i % dp_size == dp_rank
    ]
```

**分片策略**：
- Rank 0: 处理索引 [0, DP, 2*DP, 3*DP, ...]
- Rank 1: 处理索引 [1, DP+1, 2*DP+1, 3*DP+1, ...]
- Rank 2: 处理索引 [2, DP+2, 2*DP+2, 3*DP+2, ...]
- ...

**优势**：
- 每个 rank 处理的数据均匀分布在整个时间范围内
- 避免某个 rank 只处理早期或晚期请求
- 保证负载均衡

#### 3.5 HashID 映射器初始化（179-185行）

```python
vocab_size = llm.llm_engine.model_config.get_vocab_size()
mapper = HashIDMapper(
    vocab_size=vocab_size,
    seed=config.trace_hash_id_seed,
)
```

**HashIDMapper 作用**：
- Trace 数据使用 `hash_ids`（SipHash 块）而不是真实 token IDs
- 为了隐私保护，trace 不包含真实 token 序列
- HashIDMapper 将 `hash_ids` 映射回 token IDs（使用确定性映射）

**映射过程**：
1. 每个 `hash_id` 代表 16 个 token 的哈希块
2. 使用 seed 和 vocab_size 进行确定性映射
3. 生成对应的 token ID 序列

#### 3.6 Warmup 阶段（187-194行）

```python
if config.warmup_steps > 0:
    warmup_prompts = ["Warmup prompt: ..."] * 4
    warmup_params = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)
    for i in range(config.warmup_steps):
        _ = llm.generate(warmup_prompts, warmup_params)
```

**目的**：
- 预热 GPU 和 CUDA kernels
- 初始化 KV cache
- 稳定性能测量

**特点**：
- 使用简单的固定 prompt
- 不记录 metrics（warmup 结果被丢弃）
- 快速执行（max_tokens=16）

#### 3.7 E2E Metrics 收集阶段（196-265行）

这是核心测量阶段。

**A. 创建 TraceScheduler**
```python
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=128,  # 默认值，会被每个请求的 output_length 覆盖
    ignore_eos=True,
)
scheduler = TraceScheduler(llm, mapper, sampling_params)
```

**TraceScheduler 功能**：
- 异步调度 trace 请求
- 支持实时回放（realtime replay）或批量提交
- 将 trace 请求转换为 vLLM 可执行的格式

**B. 执行请求调度**
```python
e2e_start_time = time.perf_counter()

e2e_outputs = asyncio.run(
    scheduler.schedule_requests(
        trace_requests,
        realtime=config.trace_realtime_replay,
    )
)

e2e_end_time = time.perf_counter()
e2e_total_time = e2e_end_time - e2e_start_time
```

**Realtime Replay 模式**（`realtime=True`）：
- 按照 trace 中的时间戳间隔提交请求
- 模拟真实工作负载的时间分布
- 使用 `asyncio.sleep()` 等待到目标时间

**批量模式**（`realtime=False`）：
- 立即提交所有请求
- 最大化吞吐量测试
- 不考虑时间间隔

**C. 过滤有效输出**
```python
valid_outputs = [out for out in e2e_outputs if out is not None]
```

**为什么会有 None**：
- 请求可能超时
- 请求可能失败
- 某些边界情况下的错误

#### 3.8 Metrics 提取与统计（230-265行）

**A. 使用 E2EMetricsCollector 收集指标**
```python
e2e_collector = E2EMetricsCollector()
collected_metrics = e2e_collector.add_requests(valid_outputs)
```

**E2EMetricsCollector 工作流程**：

1. **遍历所有 RequestOutput**
   ```python
   for request_output in valid_outputs:
       metrics_obj = request_output.metrics  # 从 vLLM 获取 metrics
   ```

2. **提取指标**（根据 metrics 类型）：
   
   **RequestMetrics 格式**（旧版引擎）：
   ```python
   ttft_ms = metrics_obj.time_to_first_token * 1000
   tpot_ms = metrics_obj.time_per_output_token * 1000
   e2e_latency_ms = metrics_obj.finished_time - metrics_obj.arrival_time
   queued_time_ms = metrics_obj.time_in_queue * 1000
   ```
   
   **RequestStateStats 格式**（v1 引擎）：
   ```python
   # TTFT: first_token_latency（已经是 wall-clock 时间差）
   ttft_ms = metrics_obj.first_token_latency * 1000
   
   # Queued time: scheduled_ts - queued_ts（都是单调时间戳）
   queued_time_ms = (metrics_obj.scheduled_ts - metrics_obj.queued_ts) * 1000
   
   # TPOT: decode_time / (num_tokens - 1)
   # decode_time = last_token_ts - first_token_ts
   if num_generation_tokens > 1:
       decode_time = metrics_obj.last_token_ts - metrics_obj.first_token_ts
       tpot_ms = (decode_time / (num_generation_tokens - 1)) * 1000
   ```

3. **创建 E2EMetrics 对象**
   ```python
   @dataclass
   class E2EMetrics:
       request_id: str
       ttft_ms: float
       tpot_ms: float
       e2e_latency_ms: float
       queued_time_ms: float
       num_prompt_tokens: int
       num_generation_tokens: int
   ```

**B. 计算统计信息**
```python
e2e_stats = e2e_collector.calculate_statistics()
```

**统计计算**：
- 对每个指标（TTFT, TPOT, E2E Latency, Queued Time）计算：
  - `mean`: 平均值
  - `p50`: 50th 百分位数（中位数）
  - `p90`: 90th 百分位数
  - `p99`: 99th 百分位数
  - `std`: 标准差
  - `min`: 最小值
  - `max`: 最大值

**C. 计算吞吐量**
```python
e2e_throughput = e2e_collector.calculate_throughput(e2e_total_time)
```

**吞吐量计算**：
```python
total_tokens = sum([m.num_generation_tokens for m in collected_metrics])
throughput = total_tokens / e2e_total_time  # tokens per second
```

#### 3.9 结果编译与保存（267-294行）

```python
e2e_metrics_result = {
    "num_requests": len(e2e_outputs),
    "num_valid_requests": len(valid_outputs),
    "total_time_seconds": e2e_total_time,
    "num_ranks": 1,
    **e2e_stats,  # 包含所有统计指标
    "throughput": {"tokens_per_second": e2e_throughput}
}

final_results = {
    "config": config.to_dict(),
    "metadata": {
        "timestamp": ...,
        "cuda_available": ...,
        "cuda_device_count": ...,
        "dp_rank": dp_rank,
        "dp_size": dp_size,
    },
    "e2e_metrics": e2e_metrics_result,
}

# 保存到临时文件
with open(temp_output_file, 'w') as f:
    json.dump(final_results, f, indent=2)
```

**输出结构**：
- `config`: 完整的配置信息
- `metadata`: 运行环境元数据
- `e2e_metrics`: E2E 性能指标

---

### 4. 多进程执行：`run_e2e_benchmark_multi_rank()`

当 `data_parallel_size > 1` 时，启动多个独立进程并行执行。

#### 4.1 GPU 分配（425-465行）

```python
# 获取可用 GPU
cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
if cuda_visible:
    available_gpus = cuda_visible.split(',')
else:
    total_gpus = config.tensor_parallel_size * dp_size
    available_gpus = [str(i) for i in range(total_gpus)]

# 为每个 rank 分配 GPU
for rank in range(dp_size):
    start_idx = rank * tp_size
    end_idx = start_idx + tp_size
    rank_gpus = available_gpus[start_idx:end_idx]
    rank_gpu_str = ','.join(rank_gpus)
```

**分配策略**：
- Rank 0: GPUs [0, 1, ..., TP-1]
- Rank 1: GPUs [TP, TP+1, ..., 2*TP-1]
- Rank 2: GPUs [2*TP, 2*TP+1, ..., 3*TP-1]
- ...

**示例**（TP=2, DP=2）：
- Rank 0: GPUs "0,1"
- Rank 1: GPUs "2,3"

#### 4.2 进程启动（467-492行）

```python
procs = []
for rank in range(dp_size):
    proc = Process(
        target=_run_rank_process,
        args=(config, rank, dp_size, rank_output_files[rank], rank_gpu_assignments[rank])
    )
    proc.start()
    procs.append(proc)
```

**进程特性**：
- 每个进程完全独立
- 通过 `CUDA_VISIBLE_DEVICES` 隔离 GPU
- 无进程间通信（无 NCCL/TCPStore）
- 每个进程处理自己的 trace 数据分片

#### 4.3 进程同步与等待（478-502行）

```python
timeout = 3600  # 1 小时超时
for rank, proc in enumerate(procs):
    proc.join(timeout=timeout)
    if proc.exitcode is None:
        proc.kill()  # 超时则杀死进程
        all_success = False
    elif proc.exitcode != 0:
        all_success = False  # 失败
```

**错误处理**：
- 超时检测（1 小时）
- 退出码检查
- 失败时清理临时文件

#### 4.4 结果聚合（504-554行）

**A. 加载所有 rank 的结果**
```python
rank_results = []
for rank, file_path in enumerate(rank_output_files):
    with open(file_path, 'r') as f:
        result = json.load(f)
        rank_results.append(result)
```

**B. 调用聚合函数**
```python
aggregated_e2e_metrics = _aggregate_e2e_metrics(rank_results)
```

---

### 5. 结果聚合：`_aggregate_e2e_metrics()`

这是多进程模式下的关键函数，将多个 rank 的结果合并为统一指标。

#### 5.1 数据收集（347-358行）

```python
all_metrics = []
for result in rank_results:
    e2e = result.get('e2e_metrics', {})
    if e2e and e2e.get('num_requests', 0) > 0:
        all_metrics.append(e2e)
```

收集所有 rank 的 `e2e_metrics` 数据。

#### 5.2 基础统计聚合（360-383行）

```python
total_requests = sum([m['num_requests'] for m in all_metrics])
total_valid_requests = sum([m.get('num_valid_requests', 0) for m in all_metrics])
total_time = max([m.get('total_time_seconds', 0) for m in all_metrics])
```

**聚合策略**：
- `total_requests`: 所有 rank 的请求数**求和**
- `total_valid_requests`: 所有 rank 的有效请求数**求和**
- `total_time`: 所有 rank 中**最大值**（因为并行执行）

#### 5.3 指标值聚合（365-396行）

```python
def collect_metric(metric_name):
    values = []
    for m in all_metrics:
        metric_data = m.get(metric_name, {})
        if metric_data:
            # 收集每个 rank 的 mean, p50, p90, p99
            for stat in ['mean', 'p50', 'p90', 'p99']:
                val = metric_data.get(stat)
                if val is not None:
                    values.append(val)
    return values

# 对每个指标聚合
for metric_name in ['ttft_ms', 'tpot_ms', 'e2e_latency_ms', 'queued_time_ms']:
    values = collect_metric(metric_name)
    if values:
        aggregated[metric_name] = {
            'mean': float(np.mean(values)),
            'p50': float(np.percentile(values, 50)),
            'p90': float(np.percentile(values, 90)),
            'p99': float(np.percentile(values, 99)),
        }
```

**聚合逻辑**：
1. **收集阶段**：从每个 rank 收集所有统计值（mean, p50, p90, p99）
2. **聚合阶段**：对所有收集的值再次计算统计量

**示例**（2 个 rank）：
- Rank 0: TTFT mean = 45ms, p50 = 42ms, p90 = 52ms, p99 = 58ms
- Rank 1: TTFT mean = 48ms, p50 = 44ms, p90 = 55ms, p99 = 60ms
- 聚合后：收集 [45, 42, 52, 58, 48, 44, 55, 60]，然后计算新的统计量

**注意**：这不是简单的平均，而是将所有 rank 的统计值合并后重新计算。

#### 5.4 吞吐量聚合（398-405行）

```python
total_tokens = sum([m.get('total_generation_tokens', 0) for m in all_metrics])
if total_time > 0:
    aggregated['throughput'] = {
        'tokens_per_second': total_tokens / total_time
    }
```

**吞吐量计算**：
- `total_tokens`: 所有 rank 生成的 token 总数（求和）
- `total_time`: 最大执行时间（并行执行的总时间）
- `throughput = total_tokens / total_time`

**为什么用最大时间**：
- 多个进程并行执行
- 总执行时间 = 最慢进程的时间
- 吞吐量 = 总产出 / 总时间

---

### 6. 进程包装函数：`_run_rank_process()`

这是每个子进程的入口点。

```python
def _run_rank_process(config, dp_rank, dp_size, output_file, gpu_indices):
    # 1. 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_indices
    
    # 2. 运行 benchmark
    result = run_e2e_benchmark_single_rank(config, dp_rank, dp_size)
    
    # 3. 保存结果
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    # 4. 退出
    sys.exit(0)
```

**关键点**：
- 每个进程设置自己的 `CUDA_VISIBLE_DEVICES`
- 进程间完全独立，无通信
- 结果保存到临时文件，由主进程读取

---

## 数据流图

```
┌─────────────────────────────────────────────────────────────┐
│                    main()                                   │
│  - 解析参数                                                  │
│  - 加载配置                                                  │
│  - 调用 run_e2e_benchmark()                                 │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │ run_e2e_benchmark()  │
         │                      │
         │ if DP > 1:           │
         │   ┌──────────────┐   │
         │   │ Multi-Rank   │   │
         │   └──────────────┘   │
         │ else:                │
         │   ┌──────────────┐   │
         │   │ Single-Rank  │   │
         │   └──────────────┘   │
         └───────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌──────────────────┐    ┌──────────────────────┐
│ Single-Rank Path │    │ Multi-Rank Path      │
│                  │    │                      │
│ 1. 初始化 vLLM   │    │ 1. 分配 GPU          │
│ 2. 加载 Trace    │    │ 2. 启动 N 个进程     │
│ 3. 初始化 Mapper │    │ 3. 每个进程执行      │
│ 4. Warmup       │    │    run_e2e_benchmark_ │
│ 5. 执行请求      │    │    single_rank()     │
│ 6. 收集 Metrics  │    │ 4. 等待所有进程完成  │
│ 7. 计算统计      │    │ 5. 聚合结果          │
│ 8. 保存结果      │    │ 6. 保存聚合结果      │
└──────────────────┘    └──────────────────────┘
```

---

## 关键设计决策

### 1. 为什么必须使用 Trace 数据？

- E2E metrics 需要真实的请求时间分布
- 需要真实的 prompt 长度和输出长度分布
- 需要模拟真实工作负载的并发模式

### 2. 为什么禁用算子 Profiling？

- 算子 profiling 会引入额外开销（CUDA event 同步、数据收集）
- 这些开销会影响 E2E 指标的准确性
- Phase 1 专注于 E2E 指标，Phase 2 专注于算子性能

### 3. 为什么使用 Round-Robin 分片？

- 保证每个 rank 处理的数据均匀分布
- 避免时间偏差（某个 rank 只处理早期请求）
- 负载均衡

### 4. 为什么多进程模式不使用真实 DP？

- 真实 DP 需要 NCCL 通信，增加复杂性
- Benchmark 场景下，独立进程已足够
- 每个进程处理独立的数据分片，结果聚合即可

### 5. 聚合策略的选择

- **请求数/Token 数**：求和（所有 rank 的总和）
- **执行时间**：最大值（并行执行的最长时间）
- **统计指标**：收集所有 rank 的统计值，重新计算（不是简单平均）

---

## 输出格式

### 单进程输出

```json
{
  "config": {...},
  "metadata": {
    "timestamp": "2024-01-01 12:00:00",
    "cuda_available": true,
    "cuda_device_count": 4,
    "dp_rank": 0,
    "dp_size": 1
  },
  "e2e_metrics": {
    "num_requests": 100,
    "num_valid_requests": 98,
    "total_time_seconds": 120.5,
    "num_ranks": 1,
    "ttft_ms": {
      "mean": 45.2,
      "p50": 42.1,
      "p90": 52.3,
      "p99": 58.7,
      "std": 8.5,
      "min": 30.1,
      "max": 65.2
    },
    "tpot_ms": {...},
    "e2e_latency_ms": {...},
    "queued_time_ms": {...},
    "throughput": {
      "tokens_per_second": 1234.5
    }
  }
}
```

### 多进程输出（聚合后）

```json
{
  "config": {...},
  "metadata": {
    "timestamp": "2024-01-01 12:00:00",
    "cuda_available": true,
    "cuda_device_count": 8,
    "data_parallel_size": 2
  },
  "e2e_metrics_aggregated": {
    "num_requests": 200,  // 所有 rank 的总和
    "num_valid_requests": 196,
    "total_time_seconds": 125.3,  // 最大值
    "num_ranks": 2,
    "ttft_ms": {
      "mean": 46.5,  // 从所有 rank 的统计值重新计算
      "p50": 43.2,
      "p90": 53.8,
      "p99": 59.5
    },
    "throughput": {
      "tokens_per_second": 2456.7  // total_tokens / max_time
    }
  },
  "e2e_metrics_per_rank": [
    {
      "dp_rank": 0,
      "e2e_metrics": {...}  // Rank 0 的原始结果
    },
    {
      "dp_rank": 1,
      "e2e_metrics": {...}  // Rank 1 的原始结果
    }
  ]
}
```

---

## 性能考虑

### 1. 内存使用

- 每个进程独立加载模型（DP>1 时）
- 临时文件存储在 `tmp_benchmark/`，可定期清理
- Trace 数据按需加载，不全部加载到内存

### 2. 执行时间

- Warmup 阶段：~10-30 秒（取决于模型大小）
- Trace 执行：取决于 trace 数据量和 realtime 模式
- 结果聚合：<1 秒

### 3. 并发控制

- 多进程模式：使用 Python `multiprocessing.Process`
- 异步请求调度：使用 `asyncio` 和 `TraceScheduler`
- GPU 隔离：通过 `CUDA_VISIBLE_DEVICES`

---

## 错误处理

### 1. 配置验证错误

- `use_trace_data=False`: 抛出 `ValueError`
- `enable_e2e_metrics=False`: 自动设置为 `True`（警告）

### 2. Trace 数据错误

- 文件不存在：`TraceLoader` 抛出异常
- 无有效请求：抛出 `ValueError("No trace requests loaded")`
- 分片后无请求：抛出 `ValueError(f"DP rank {rank} has no requests")`

### 3. 进程执行错误

- 超时：杀死进程，标记为失败
- 退出码非 0：标记为失败，清理临时文件
- 部分失败：所有进程失败才抛出异常

### 4. Metrics 收集错误

- `RequestOutput.metrics` 为 `None`：记录警告，跳过该请求
- 无效输出：过滤掉 `None` 值

---

## 总结

`benchmark_e2e_metrics.py` 是一个专门设计用于测量 E2E 性能指标的脚本，具有以下特点：

1. **专注性**：只测量 E2E 指标，不进行算子 profiling
2. **真实性**：必须使用真实 trace 数据
3. **可扩展性**：支持单进程和多进程模式
4. **准确性**：禁用 CUDA graph，启用 metrics 收集
5. **完整性**：提供详细的统计信息（mean, p50, p90, p99 等）

整个实现遵循了清晰的模块化设计，每个函数职责单一，便于维护和扩展。


