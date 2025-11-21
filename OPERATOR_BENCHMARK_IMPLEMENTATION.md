# vLLM Operator Benchmark 实现文档

## 目录
1. [系统架构概览](#系统架构概览)
2. [文件结构与职责](#文件结构与职责)
3. [核心工作流程](#核心工作流程)
4. [关键设计决策](#关键设计决策)
5. [数据流分析](#数据流分析)
6. [性能优化机制](#性能优化机制)
7. [多进程通信机制](#多进程通信机制)
8. [关键代码路径](#关键代码路径)

---

## 系统架构概览

vLLM Operator Benchmark 系统是一个**多进程、事件驱动的性能分析框架**，用于在模型推理过程中收集计算密集型算子（如 Attention、Linear、LayerNorm 等）的详细性能指标。

### 架构特点

1. **多进程架构**：主进程负责协调，worker 子进程负责实际的数据收集
2. **非侵入式注入**：通过装饰器模式动态包装模块的 `forward` 方法
3. **批量同步优化**：使用 CUDA Event 批量处理，避免频繁同步带来的性能损失
4. **环境变量驱动**：通过环境变量在 worker 进程启动时自动启用 benchmark

### 系统组件关系图

```
┌─────────────────────────────────────────────────────────────┐
│                    Main Process                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  benchmark_operators.py (主脚本)                      │   │
│  │  - 解析配置                                           │   │
│  │  - 设置环境变量                                        │   │
│  │  - 初始化 vLLM LLM                                    │   │
│  │  - 运行推理循环                                        │   │
│  │  - 读取 worker 结果                                   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ 环境变量传递
                            │ (VLLM_OPERATOR_BENCHMARK_*)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                  Worker Process (子进程)                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  gpu_model_runner.py                                 │   │
│  │  - _inject_operator_benchmark_if_enabled()           │   │
│  │    └─> 读取环境变量                                    │   │
│  │    └─> 调用 inject_benchmark_hooks()                  │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                │
│                            ▼                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  operator_injector.py                                │   │
│  │  - inject_benchmark_hooks()                          │   │
│  │    └─> 遍历模型模块                                    │   │
│  │    └─> 创建 benchmark_wrapper                         │   │
│  │    └─> 替换 forward 方法                              │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                │
│                            ▼                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  模型推理 (forward pass)                              │   │
│  │  - 每个被包装的 operator 调用                           │   │
│  │  - benchmark_wrapper 记录 CUDA Event                  │   │
│  │  - 添加到 pending_events 队列                          │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                │
│                            ▼                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  execute_model() 结束                                 │   │
│  │  - flush_pending_events() 被调用                      │   │
│  │    └─> 批量同步 CUDA                                   │   │
│  │    └─> 计算 elapsed_time                              │   │
│  │    └─> 调用 record_operator_call()                    │   │
│  └──────────────────────────────────────────────────────┘   │
│                            │                                │
│                            ▼                                │
│  ┌──────────────────────────────────────────────────────┐   │
│  │  operator_profiler.py                                │   │
│  │  - OperatorBenchmark (单例)                           │   │
│  │  - 收集 records                                       │   │
│  │  - 自动保存到临时文件                                   │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ 临时文件
                            │ (/tmp/vllm_operator_benchmark.json)
                            ▼
┌─────────────────────────────────────────────────────────────┐
│  Main Process 读取结果                                       │
│  - 从临时文件加载 JSON                                        │
│  - 生成最终报告                                               │
└─────────────────────────────────────────────────────────────┘
```

---

## 文件结构与职责

### 1. 主脚本层

#### `benchmarks/benchmark_operators.py`
**职责**：benchmark 系统的入口点和协调器

**核心功能**：
- 配置解析（支持 Python 配置文件或命令行参数）
- 环境变量设置（在 LLM 初始化前设置，确保 worker 进程能读取）
- vLLM 引擎初始化
- 推理循环控制（warmup + benchmark）
- 结果收集和汇总
- JSON 结果输出

**关键函数**：
```python
def run_benchmark(config: BenchmarkConfig) -> Dict[str, Any]:
    """主 benchmark 流程"""
    # 1. 创建临时文件用于 worker 输出
    # 2. 设置环境变量
    # 3. 初始化 LLM（触发 worker 进程启动）
    # 4. 运行 warmup 和 benchmark 迭代
    # 5. 从临时文件读取 worker 结果
    # 6. 生成最终报告
```

**环境变量设置**（关键！必须在 LLM 初始化前设置）：
```python
os.environ["VLLM_OPERATOR_BENCHMARK_ENABLE"] = "1"
os.environ["VLLM_OPERATOR_BENCHMARK_OPS"] = ",".join(operators)
os.environ["VLLM_OPERATOR_BENCHMARK_WARMUP"] = str(warmup_steps)
os.environ["VLLM_OPERATOR_BENCHMARK_AUTO_SAVE"] = "1"
os.environ["VLLM_OPERATOR_BENCHMARK_OUTPUT"] = worker_output_file
```

---

### 2. 配置层

#### `vllm/profiler/benchmark_config.py`
**职责**：定义 benchmark 配置的数据结构

**核心类**：
```python
@dataclass
class BenchmarkConfig:
    # 模型配置
    model_path: str
    tensor_parallel_size: int = 1
    pipeline_parallel_size: int = 1
    max_model_len: Optional[int] = None
    
    # Benchmark 参数
    batch_sizes: List[int] = [1, 4, 8]
    seq_lengths: List[int] = [512, 1024]
    warmup_steps: int = 5
    benchmark_steps: int = 20
    
    # 算子选择
    operators_to_benchmark: List[str] = ["attention", "linear", ...]
    
    # 输出配置
    output_file: str = "benchmark_results.json"
    include_per_layer_stats: bool = True
    include_summary_stats: bool = True
```

**设计特点**：
- 使用 `dataclass` 提供类型安全和默认值
- `validate()` 方法确保配置有效性
- `to_dict()` 方法便于序列化

---

### 3. 注入层

#### `vllm/profiler/operator_injector.py`
**职责**：动态注入 benchmark hooks 到模型模块

**核心组件**：

**1. 算子类型映射** (`OPERATOR_TYPE_MAPPING`)
```python
OPERATOR_TYPE_MAPPING = {
    "Attention": "attention",
    "Linear": "linear",
    "RMSNorm": "layernorm",
    "MLP": "mlp",
    # ... 更多映射
}
```

**2. 核心注入函数** (`inject_benchmark_hooks`)
```python
def inject_benchmark_hooks(
    model: nn.Module,
    operators_to_benchmark: Optional[List[str]] = None,
) -> int:
    """
    遍历模型的所有模块，为匹配的算子创建 wrapper
    返回：被注入的模块数量
    """
    for name, module in model.named_modules():
        operator_type = get_operator_type(module)
        if should_instrument(operator_type, operators_to_benchmark):
            # 包装 forward 方法
            module.forward = create_benchmark_wrapper(...)
```

**3. Benchmark Wrapper** (`create_benchmark_wrapper`)
```python
def benchmark_wrapper(*args, **kwargs):
    """包装后的 forward 方法"""
    # 1. 创建 CUDA Event (start, end)
    # 2. 调用原始 forward
    # 3. 记录 end event
    # 4. 提取 input/output shapes
    # 5. 添加到 pending_events 队列（不同步！）
    # 6. 返回结果
```

**关键设计**：
- **延迟同步**：不在 wrapper 中调用 `torch.cuda.synchronize()`，而是将事件加入队列
- **批量处理**：所有事件在 `flush_pending_events()` 中统一处理
- **非侵入式**：使用 `functools.wraps` 保持原始函数签名

---

### 4. 数据收集层

#### `vllm/profiler/operator_profiler.py`
**职责**：收集、存储和统计 benchmark 数据

**核心数据结构**：

**1. OperatorCallRecord**
```python
@dataclass
class OperatorCallRecord:
    layer_name: str
    operator_type: str
    elapsed_time_ms: float
    input_shapes: List[Tuple[int, ...]]
    output_shapes: List[Tuple[int, ...]]
    device: str
```

**2. PendingEvent**
```python
@dataclass
class PendingEvent:
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    layer_name: str
    operator_type: str
    input_shapes: List[Tuple[int, ...]]
    output_shapes: List[Tuple[int, ...]]
    device: str
```

**3. OperatorBenchmark（单例类）**

**核心方法**：

**`add_pending_event()`** - 添加待处理事件
```python
def add_pending_event(self, event: PendingEvent):
    """将事件加入队列，不进行同步"""
    with self._pending_lock:
        self._pending_events.append(event)
```

**`flush_pending_events()`** - 批量处理事件（关键优化点）
```python
def flush_pending_events(self):
    """
    1. 单次 CUDA 同步（所有事件）
    2. 批量计算 elapsed_time
    3. 批量创建 records
    4. 批量添加到 _records
    5. 自动保存（如果启用）
    """
    # 单次同步
    torch.cuda.synchronize()
    
    # 批量处理
    for event in self._pending_events:
        elapsed_time = event.start_event.elapsed_time(event.end_event)
        record = OperatorCallRecord(...)
        records_to_add.append(record)
    
    # 批量添加
    self._records.extend(records_to_add)
```

**`get_statistics()`** - 计算统计信息
```python
def get_statistics(self) -> BenchmarkResults:
    """
    1. 按 layer_name 分组
    2. 计算每层的统计（avg, std, min, max, total）
    3. 按 operator_type 分组
    4. 计算汇总统计
    """
```

**Warmup 机制**：
- 使用 `_flush_count` 跟踪 forward pass 次数
- `in_warmup = _flush_count <= _warmup_steps`
- Warmup 期间的事件被丢弃，不记录

**自动保存机制**：
- 检查环境变量 `VLLM_OPERATOR_BENCHMARK_AUTO_SAVE`
- 每次 `flush_pending_events()` 后自动保存到临时文件
- 主进程从临时文件读取结果

---

### 5. Worker 集成层

#### `vllm/v1/worker/gpu_model_runner.py`
**职责**：在 worker 进程中集成 benchmark 功能

**关键方法**：

**`_inject_operator_benchmark_if_enabled()`**
```python
def _inject_operator_benchmark_if_enabled(self):
    """在模型加载后调用，检查环境变量并注入 hooks"""
    enable_benchmark = os.environ.get("VLLM_OPERATOR_BENCHMARK_ENABLE", "0")
    if enable_benchmark not in ("1", "true", "True", "TRUE"):
        return
    
    operators_str = os.environ.get("VLLM_OPERATOR_BENCHMARK_OPS", ...)
    warmup_steps = int(os.environ.get("VLLM_OPERATOR_BENCHMARK_WARMUP", "5"))
    
    enable_operator_benchmark(warmup_steps=warmup_steps)
    inject_benchmark_hooks(self.model, operators_to_benchmark)
```

**`execute_model()` 中的 flush 调用**
```python
def execute_model(...):
    # ... 模型 forward ...
    
    # Flush pending benchmark events after forward pass
    try:
        from vllm.profiler import get_operator_benchmark
        benchmark = get_operator_benchmark()
        if benchmark.is_enabled():
            benchmark.flush_pending_events()  # 关键！
    except Exception:
        pass
```

**调用时机**：
- `_inject_operator_benchmark_if_enabled()` 在 `load_model()` 后调用
- `flush_pending_events()` 在每次 `execute_model()` 结束后调用

---

### 6. 聚合层（可选）

#### `vllm/profiler/multiworker_aggregator.py`
**职责**：多 worker 结果聚合和比较

**核心功能**：
- `aggregate_worker_results()` - 聚合多个 worker 的结果
- `compare_results()` - 比较两次 benchmark 的结果
- `merge_results_by_config()` - 合并不同配置的结果

**使用场景**：
- Tensor Parallel (TP) 模式：多个 worker 需要聚合
- Pipeline Parallel (PP) 模式：不同 stage 的结果需要合并
- 结果对比：比较不同配置或优化前后的性能

---

## 核心工作流程

### 完整执行流程

```
1. 主进程启动
   └─> benchmark_operators.py::main()
       └─> 解析配置（文件或命令行）
       └─> run_benchmark(config)

2. 环境准备
   └─> 创建临时文件路径
   └─> 设置环境变量（关键！）
       - VLLM_OPERATOR_BENCHMARK_ENABLE=1
       - VLLM_OPERATOR_BENCHMARK_OPS=...
       - VLLM_OPERATOR_BENCHMARK_WARMUP=...
       - VLLM_OPERATOR_BENCHMARK_AUTO_SAVE=1
       - VLLM_OPERATOR_BENCHMARK_OUTPUT=/tmp/xxx.json

3. vLLM 初始化
   └─> LLM(**kwargs)  # 触发 worker 进程启动
       └─> Worker 进程读取环境变量
       └─> gpu_model_runner.py::_inject_operator_benchmark_if_enabled()
           └─> inject_benchmark_hooks(model)
               └─> 遍历模型模块，包装 forward 方法

4. Warmup 阶段
   └─> for i in range(warmup_steps):
       └─> llm.generate(prompts)
           └─> execute_model()
               └─> 模型 forward（被包装的 operators）
                   └─> benchmark_wrapper()
                       └─> 记录 CUDA Event，加入 pending_events
               └─> flush_pending_events()
                   └─> 检查 warmup（_flush_count <= warmup_steps）
                   └─> 如果是 warmup，清空队列，不记录

5. Benchmark 阶段
   └─> for i in range(benchmark_steps):
       └─> llm.generate(prompts)
           └─> execute_model()
               └─> 模型 forward
               └─> flush_pending_events()
                   └─> 同步 CUDA
                   └─> 计算 elapsed_time
                   └─> 创建 records
                   └─> 添加到 _records
                   └─> 自动保存到临时文件

6. 结果收集
   └─> 主进程等待 0.5 秒
   └─> 读取临时文件（JSON）
   └─> 解析 worker 结果
   └─> 生成最终报告

7. 清理
   └─> 删除临时文件
```

---

## 关键设计决策

### 1. 单例模式 (Singleton)

**为什么使用单例**：
- Worker 进程是独立的 Python 进程，每个进程有自己的内存空间
- 单例确保在同一个进程内只有一个 `OperatorBenchmark` 实例
- 避免多个模块创建多个实例导致数据不一致

**实现方式**：
```python
class OperatorBenchmark:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
```

### 2. 批量同步优化

**问题**：
- 每个 operator 调用 `torch.cuda.synchronize()` 会导致严重的性能下降
- 327 个 operators × 每次同步 ≈ 程序"卡死"

**解决方案**：
- **延迟同步**：在 wrapper 中只记录 CUDA Event，不同步
- **批量处理**：在 forward pass 结束后统一同步一次
- **性能提升**：从 O(n) 次同步降到 O(1) 次同步

**实现**：
```python
# wrapper 中（不同步）
start_event.record()
result = original_forward(*args, **kwargs)
end_event.record()
benchmark.add_pending_event(PendingEvent(...))  # 只加入队列

# flush_pending_events 中（批量同步）
torch.cuda.synchronize()  # 只同步一次
for event in self._pending_events:
    elapsed_time = event.start_event.elapsed_time(event.end_event)
    # ... 处理
```

### 3. 环境变量通信

**为什么使用环境变量**：
- vLLM 使用多进程架构，主进程和 worker 进程内存隔离
- 环境变量在进程启动时自动继承
- 简单、可靠，无需额外的 IPC 机制

**关键点**：
- **必须在 LLM 初始化前设置**，否则 worker 进程读取不到
- Worker 进程在 `load_model()` 时读取环境变量

### 4. 临时文件通信

**为什么使用临时文件**：
- Worker 进程和主进程内存隔离，无法直接共享数据
- 文件系统是跨进程通信的可靠方式
- JSON 格式便于序列化和调试

**流程**：
1. 主进程创建临时文件路径
2. 通过环境变量传递给 worker
3. Worker 自动保存结果到文件
4. 主进程读取文件获取结果

### 5. Warmup 机制

**为什么需要 warmup**：
- 第一次执行可能触发 CUDA kernel 编译、内存分配等
- 需要多次迭代让性能稳定

**实现方式**：
- 使用 `_flush_count` 跟踪 forward pass 次数
- `in_warmup = _flush_count <= _warmup_steps`
- Warmup 期间的事件被丢弃

**注意**：
- 之前使用 operator 调用次数判断 warmup，导致逻辑错误
- 现在改为 forward pass 次数，更准确

---

## 数据流分析

### 数据收集流程

```
┌─────────────────────────────────────────────────────────┐
│  模型推理 (Forward Pass)                                 │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  benchmark_wrapper()                                    │
│  - 创建 CUDA Event (start, end)                          │
│  - 调用 original_forward()                               │
│  - 记录 end event                                        │
│  - 提取 shapes                                           │
│  - 创建 PendingEvent                                     │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  add_pending_event()                                    │
│  - 加入 _pending_events 队列                           │
│  - 不进行同步                                           │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  execute_model() 结束                                   │
│  - 调用 flush_pending_events()                         │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  flush_pending_events()                                │
│  1. 单次 CUDA 同步                                      │
│  2. 检查 warmup 状态                                    │
│  3. 批量计算 elapsed_time                              │
│  4. 批量创建 OperatorCallRecord                        │
│  5. 批量添加到 _records                                │
│  6. 自动保存到临时文件（如果启用）                      │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  _records 列表                                          │
│  - 存储所有 OperatorCallRecord                          │
│  - 线程安全（使用 _records_lock）                       │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  get_statistics()                                       │
│  - 按 layer_name 分组                                   │
│  - 计算每层统计（avg, std, min, max, total）            │
│  - 按 operator_type 分组                                │
│  - 计算汇总统计                                          │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  BenchmarkResults                                       │
│  - per_layer_stats: List[Dict]                         │
│  - summary_stats: Dict[str, Dict]                      │
│  - total_forward_time_ms: float                        │
└─────────────────────────────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────────┐
│  保存到临时文件 (JSON)                                  │
│  - 主进程读取                                           │
│  - 生成最终报告                                         │
└─────────────────────────────────────────────────────────┘
```

### 数据结构转换

```
PendingEvent (队列中)
    │
    ├─> flush_pending_events()
    │   └─> 计算 elapsed_time
    │
    └─> OperatorCallRecord (存储)
            │
            ├─> get_statistics()
            │   ├─> 按 layer_name 分组
            │   │   └─> OperatorStats (per_layer_stats)
            │   │
            │   └─> 按 operator_type 分组
            │       └─> SummaryStats (summary_stats)
            │
            └─> BenchmarkResults
                    │
                    └─> JSON 序列化
                        └─> 临时文件 / 最终输出
```

---

## 性能优化机制

### 1. 批量 CUDA 同步

**优化前**：
```python
# 每个 operator 都同步（327 次/forward pass）
for operator in operators:
    start_event.record()
    result = operator.forward(...)
    end_event.record()
    torch.cuda.synchronize()  # ❌ 327 次同步！
    elapsed_time = start_event.elapsed_time(end_event)
```

**优化后**：
```python
# 只同步一次/forward pass
for operator in operators:
    start_event.record()
    result = operator.forward(...)
    end_event.record()
    # 只加入队列，不同步

# Forward pass 结束后
torch.cuda.synchronize()  # ✅ 只同步一次
for event in pending_events:
    elapsed_time = event.start_event.elapsed_time(event.end_event)
```

**性能提升**：
- 从 O(n) 次同步降到 O(1) 次同步
- 实际测试：从"卡死"到正常速度（3-25 秒/benchmark）

### 2. 批量记录添加

**优化**：
```python
# 批量创建 records
records_to_add = []
for event in self._pending_events:
    record = OperatorCallRecord(...)
    records_to_add.append(record)

# 批量添加（一次 extend，而不是多次 append）
self._records.extend(records_to_add)
```

### 3. 批量 Warmup 检查

**优化**：
```python
# 批量更新 call counts
with self._records_lock:
    for event in self._pending_events:
        self._call_counts[event.layer_name] += 1

# 只检查一次
in_warmup = self._flush_count <= self._warmup_steps
if in_warmup:
    self._pending_events.clear()
    return
```

### 4. 延迟文件 I/O

**优化**：
- 不在每次 operator 调用时保存
- 在每次 `flush_pending_events()` 后保存
- 减少 I/O 频率，提高性能

---

## 多进程通信机制

### 主进程 → Worker 进程

**方式**：环境变量

```python
# 主进程（benchmark_operators.py）
os.environ["VLLM_OPERATOR_BENCHMARK_ENABLE"] = "1"
os.environ["VLLM_OPERATOR_BENCHMARK_OPS"] = "attention,linear"
os.environ["VLLM_OPERATOR_BENCHMARK_WARMUP"] = "5"
os.environ["VLLM_OPERATOR_BENCHMARK_AUTO_SAVE"] = "1"
os.environ["VLLM_OPERATOR_BENCHMARK_OUTPUT"] = "/tmp/xxx.json"

# Worker 进程（gpu_model_runner.py）
enable_benchmark = os.environ.get("VLLM_OPERATOR_BENCHMARK_ENABLE", "0")
operators_str = os.environ.get("VLLM_OPERATOR_BENCHMARK_OPS", ...)
```

**关键点**：
- 必须在 `LLM()` 初始化**之前**设置环境变量
- Worker 进程在启动时自动继承环境变量

### Worker 进程 → 主进程

**方式**：临时文件（JSON）

```python
# Worker 进程（operator_profiler.py）
def flush_pending_events(self):
    # ... 处理事件 ...
    if os.environ.get("VLLM_OPERATOR_BENCHMARK_AUTO_SAVE"):
        output_file = os.environ.get("VLLM_OPERATOR_BENCHMARK_OUTPUT")
        self._save_results_internal(output_file)  # 保存 JSON

# 主进程（benchmark_operators.py）
time.sleep(0.5)  # 等待 worker 写入
with open(worker_output_file, 'r') as f:
    worker_results = json.load(f)
```

**关键点**：
- 使用 JSON 格式便于序列化和调试
- 主进程需要等待 worker 写入完成（sleep 0.5 秒）
- 完成后删除临时文件

---

## 关键代码路径

### 1. 注入流程

```
gpu_model_runner.py::load_model()
    └─> _inject_operator_benchmark_if_enabled()
        └─> 读取环境变量
        └─> enable_operator_benchmark(warmup_steps)
        └─> inject_benchmark_hooks(model, operators)
            └─> 遍历 model.named_modules()
                └─> get_operator_type(module)
                └─> create_benchmark_wrapper(forward, module, ...)
                    └─> 替换 module.forward
```

### 2. 数据收集流程

```
模型 forward pass
    └─> module.forward() (被包装)
        └─> benchmark_wrapper(*args, **kwargs)
            └─> start_event.record()
            └─> original_forward(*args, **kwargs)
            └─> end_event.record()
            └─> add_pending_event(PendingEvent(...))
                └─> _pending_events.append(event)

execute_model() 结束
    └─> flush_pending_events()
        └─> torch.cuda.synchronize()  # 单次同步
        └─> 检查 warmup
        └─> 批量计算 elapsed_time
        └─> 批量创建 OperatorCallRecord
        └─> _records.extend(records_to_add)
        └─> 自动保存到临时文件
```

### 3. 结果生成流程

```
get_statistics()
    └─> 按 layer_name 分组
        └─> 计算每层统计（avg, std, min, max, total）
    └─> 按 operator_type 分组
        └─> 计算汇总统计
    └─> 返回 BenchmarkResults

主进程读取
    └─> 从临时文件读取 JSON
    └─> 解析为 BenchmarkResults
    └─> 生成最终报告
    └─> 保存到 output_file
```

---

## 总结

vLLM Operator Benchmark 系统通过以下设计实现了高效、可靠的算子性能分析：

1. **多进程架构**：主进程协调，worker 进程收集数据
2. **非侵入式注入**：动态包装 forward 方法，无需修改模型代码
3. **批量同步优化**：从 O(n) 次同步优化到 O(1) 次同步
4. **环境变量通信**：简单可靠的多进程通信机制
5. **临时文件通信**：Worker 结果通过 JSON 文件传递给主进程
6. **Warmup 机制**：确保性能数据稳定可靠

整个系统设计简洁、高效，能够在实际推理过程中收集详细的算子性能数据，为模型优化提供数据支持。

