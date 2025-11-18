# vLLM Operator Benchmark Guide

本文档介绍如何使用vLLM的operator-level benchmark功能来测量模型中各个算子的性能。

## 概述

vLLM的operator benchmark系统可以在模型推理过程中自动测量各种计算密集型算子（如attention、linear层等）的执行时间。该系统采用装饰器模式，在模型加载后自动注入性能测量代码，使用CUDA Events进行精确的时间测量。

## 主要特性

- **统一的benchmark框架**: 使用装饰器自动对多种算子进行性能测量
- **精确的时间测量**: 使用`torch.cuda.Event`进行微秒级精度测量
- **多算子支持**: 支持attention、linear、layernorm、MLP、MoE等多种算子类型
- **灵活配置**: 通过配置文件或命令行参数调节TP、PP、batch size等参数
- **详细统计**: 提供per-layer和汇总统计信息
- **JSON输出**: 结构化的JSON格式输出，便于后续分析
- **多worker支持**: 自动聚合多个worker的benchmark结果

## 快速开始

### 基本使用

```bash
cd /mnt/disk1/ljm/vllm

# 使用配置文件
python benchmarks/benchmark_operators.py \
    --config benchmarks/operator_configs/quick_test_config.py

# 直接指定参数
python benchmarks/benchmark_operators.py \
    --model /path/to/your/model \
    --tp 1 \
    --batch-sizes 1,4,8 \
    --seq-lengths 512,1024 \
    --operators attention,linear,layernorm \
    --output results.json
```

### 查看结果

```bash
# 查看JSON结果
cat results.json | python -m json.tool

# 或使用Python
python -c "
import json
with open('results.json') as f:
    results = json.load(f)
    for r in results['results']:
        print(f\"Batch: {r['batch_size']}, Seq: {r['seq_length']}\")
        for op, stats in r['summary_stats'].items():
            print(f\"  {op}: {stats['total_time_ms']:.2f}ms ({stats['percentage_of_total']:.1f}%)\")
"
```

## 配置说明

### BenchmarkConfig参数

```python
from vllm.profiler.benchmark_config import BenchmarkConfig

config = BenchmarkConfig(
    # 模型配置
    model_path="path/to/model",          # 模型路径
    tensor_parallel_size=1,               # TP大小
    pipeline_parallel_size=1,             # PP大小
    max_model_len=None,                   # 最大上下文长度
    
    # Benchmark配置
    batch_sizes=[1, 4, 8],                # 要测试的batch大小
    seq_lengths=[512, 1024],              # 要测试的序列长度
    warmup_steps=5,                       # 预热步数
    benchmark_steps=20,                   # benchmark步数
    
    # 算子选择
    operators_to_benchmark=[              # 要benchmark的算子类型
        "attention",
        "linear",
        "layernorm",
        "mlp",
    ],
    
    # 输出配置
    output_file="results.json",           # 输出文件路径
    include_per_layer_stats=True,         # 包含每层统计
    include_summary_stats=True,           # 包含汇总统计
    
    # 高级选项
    enable_cuda_graph=True,               # 启用CUDA graph
    dtype="auto",                         # 数据类型
    gpu_memory_utilization=0.9,           # GPU内存使用率
)
```

### 支持的算子类型

- `attention`: Attention层（包括PagedAttention、FlashAttention等）
- `linear`: Linear层（包括RowParallel、ColumnParallel等）
- `layernorm`: 归一化层（RMSNorm、LayerNorm）
- `mlp`: MLP/FFN层
- `activation`: 激活函数（SiLU、GELU等）
- `embedding`: Embedding层
- `moe`: MoE层
- `mamba`: Mamba/SSM层
- `all`: 所有支持的算子

## 输出格式

### JSON结构

```json
{
  "config": {
    "model_path": "...",
    "tensor_parallel_size": 1,
    ...
  },
  "results": [
    {
      "batch_size": 8,
      "seq_length": 1024,
      "warmup_steps": 5,
      "benchmark_steps": 20,
      "total_wall_time_seconds": 12.5,
      "per_layer_stats": [
        {
          "layer_name": "model.layers.0.self_attn",
          "operator_type": "attention",
          "num_calls": 20,
          "avg_time_ms": 1.23,
          "std_time_ms": 0.05,
          "min_time_ms": 1.18,
          "max_time_ms": 1.35,
          "total_time_ms": 24.6,
          "avg_input_shapes": [[8, 1024, 4096]],
          "avg_output_shapes": [[8, 1024, 4096]]
        },
        ...
      ],
      "summary_stats": {
        "attention": {
          "total_calls": 640,
          "total_time_ms": 450.2,
          "avg_time_per_call_ms": 0.703,
          "percentage_of_total": 45.2
        },
        "linear": {
          "total_calls": 1920,
          "total_time_ms": 380.5,
          "avg_time_per_call_ms": 0.198,
          "percentage_of_total": 38.2
        },
        ...
      },
      "total_forward_time_ms": 995.3
    }
  ],
  "metadata": {
    "timestamp": "2025-11-18 10:30:00",
    "cuda_available": true,
    "cuda_device_count": 8
  }
}
```

## 高级用法

### 环境变量控制

除了使用benchmark脚本，你也可以通过环境变量直接启用operator benchmark：

```bash
# 启用operator benchmark
export VLLM_OPERATOR_BENCHMARK_ENABLE=1

# 指定要benchmark的算子
export VLLM_OPERATOR_BENCHMARK_OPS="attention,linear,layernorm"

# 然后正常运行vLLM
python your_inference_script.py
```

### Python API使用

```python
import os
from vllm import LLM
from vllm.profiler import (
    enable_operator_benchmark,
    get_operator_benchmark,
    inject_benchmark_hooks,
)

# ⚠️ 重要：必须在创建LLM之前设置环境变量
# 这样hooks才能在模型加载时被注入
os.environ["VLLM_OPERATOR_BENCHMARK_ENABLE"] = "1"
os.environ["VLLM_OPERATOR_BENCHMARK_OPS"] = "attention,linear,layernorm,mlp"

# 启用benchmark数据收集
enable_operator_benchmark(warmup_steps=5)

# 创建LLM实例（hooks会在此时自动注入）
llm = LLM(model="your/model")

# 运行推理
for i in range(15):  # warmup + benchmark
    outputs = llm.generate(["Your prompt here"])
    get_operator_benchmark().step()

# 获取benchmark结果
benchmark = get_operator_benchmark()
results = benchmark.get_statistics()

# 打印汇总
benchmark.print_summary()

# 保存结果
benchmark.save_results("results.json")
```

### 多worker结果聚合

```python
from vllm.profiler import (
    load_worker_results_from_temp,
    aggregate_worker_results,
)

# 从多个worker加载结果
worker_results = load_worker_results_from_temp(num_workers=4)

# 聚合结果
aggregated = aggregate_worker_results(worker_results)

# 保存聚合结果
with open("aggregated_results.json", "w") as f:
    json.dump(aggregated.to_dict(), f, indent=2)
```

### 结果比较

```python
from vllm.profiler import compare_results
import json

# 加载两组结果
with open("baseline_results.json") as f:
    baseline = json.load(f)

with open("optimized_results.json") as f:
    optimized = json.load(f)

# 比较结果
comparison = compare_results(
    baseline, 
    optimized,
    output_file="comparison.json"
)

# 查看哪些算子变快了
for config_key, config_diff in comparison["differences"].items():
    print(f"\n{config_key}:")
    for op, diff in config_diff["operator_differences"].items():
        if diff["faster"]:
            print(f"  {op}: {-diff['difference_ms']:.2f}ms faster ({diff['difference_percent']:.1f}%)")
```

## 注意事项

1. ⚠️ **环境变量必须先设置**: 使用Python API时，必须在创建LLM实例**之前**设置 `VLLM_OPERATOR_BENCHMARK_ENABLE=1`，否则hooks不会被注入到模型中
2. **Warmup很重要**: 至少使用5个warmup步骤以确保测量稳定
3. **CUDA同步**: 系统使用`torch.cuda.synchronize()`确保准确测量
4. **子进程测量**: benchmark在worker子进程中自动工作，无需额外配置
5. **内存开销**: benchmark会增加少量内存开销来存储测量数据
6. **性能影响**: benchmark会有约1-2%的性能开销
7. **CUDA Graph兼容**: 与CUDA graph优化兼容

## 故障排除

### 问题：No benchmark records collected

这是最常见的问题，通常是因为hooks没有被注入到模型中。

**解决方案**: 
1. **检查环境变量**（最常见原因）: 
   - 使用Python API时，必须在创建LLM**之前**设置 `VLLM_OPERATOR_BENCHMARK_ENABLE=1`
   - 正确示例：
   ```python
   import os
   os.environ["VLLM_OPERATOR_BENCHMARK_ENABLE"] = "1"  # 必须在import LLM之前
   os.environ["VLLM_OPERATOR_BENCHMARK_OPS"] = "attention,linear"
   
   from vllm import LLM
   llm = LLM(model="...")  # hooks会在此时注入
   ```
   
2. 确保运行了足够的warmup和benchmark步骤
3. 检查是否正确启用了benchmark数据收集（`enable_operator_benchmark()`）
4. 验证operators_to_benchmark配置正确
5. 查看日志中是否有 "Instrumented N operators for benchmarking" 消息

### 问题：Benchmark数据不准确

**解决方案**:
- 增加warmup_steps（建议至少5步）
- 增加benchmark_steps以获得更稳定的平均值
- 确保GPU没有被其他进程占用

### 问题：多worker结果不一致

**解决方案**:
- 使用多worker聚合功能合并结果
- 检查worker_variance_ms指标
- 确保所有workers使用相同配置

## 示例配置文件

查看 `benchmarks/operator_configs/` 目录中的示例配置：

- `quick_test_config.py`: 快速测试配置
- `llama_config.py`: Llama模型配置
- `mixtral_config.py`: Mixtral MoE模型配置

## 贡献

如果你想添加对新算子类型的支持：

1. 在 `operator_injector.py` 的 `OPERATOR_TYPE_MAPPING` 中添加映射
2. 或使用 `register_operator_type()` 函数动态注册

```python
from vllm.profiler import register_operator_type

register_operator_type("YourCustomModule", "custom_operator")
```

## 技术细节

### 实现原理

1. **装饰器注入**: 在模型加载后，系统遍历所有modules，为匹配的算子包装forward方法
2. **CUDA Events**: 使用`torch.cuda.Event`进行精确时间测量
3. **线程安全**: 使用锁机制确保多线程环境下的数据安全
4. **Warmup机制**: 自动跳过warmup阶段的数据收集
5. **聚合统计**: 计算平均值、标准差、最小/最大值等统计信息

### 文件结构

```
vllm/profiler/
  ├── __init__.py                    # 导出接口
  ├── operator_profiler.py           # 核心benchmark逻辑
  ├── operator_injector.py           # 模块注入系统
  ├── benchmark_config.py            # 配置dataclass
  └── multiworker_aggregator.py      # 多worker聚合

benchmarks/
  ├── benchmark_operators.py         # 主执行脚本
  └── operator_configs/              # 示例配置
      ├── README.md
      ├── llama_config.py
      ├── mixtral_config.py
      └── quick_test_config.py
```

## 参考

- [vLLM Documentation](https://docs.vllm.ai/)
- [PyTorch Profiler](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [CUDA Events](https://pytorch.org/docs/stable/cuda.html#torch.cuda.Event)

