# vLLM Operator Benchmark 修改文件清单

本文档列出了为实现 Operator Benchmark 功能而修改或创建的所有文件。

## 核心代码文件（6个）

### 1. `vllm/profiler/operator_profiler.py` ⭐ **核心文件**
**状态**: 新建文件  
**职责**: Operator benchmark 的核心实现
- `OperatorBenchmark` 单例类：管理 benchmark 状态和数据收集
- `OperatorCallRecord` 数据类：存储单个 operator 调用记录
- `PendingEvent` 数据类：存储待处理的 CUDA Event
- `OperatorStats` 数据类：存储统计信息
- `BenchmarkResults` 数据类：存储完整结果
- 关键方法：
  - `add_pending_event()`: 添加待处理事件（批量同步优化）
  - `flush_pending_events()`: 批量处理事件，单次 CUDA 同步
  - `get_statistics()`: 计算统计信息
  - `save_results()`: 保存结果到 JSON

### 2. `vllm/profiler/operator_injector.py` ⭐ **核心文件**
**状态**: 新建文件  
**职责**: 动态注入 benchmark hooks 到模型模块
- `OPERATOR_TYPE_MAPPING`: 算子类型映射表
- `inject_benchmark_hooks()`: 遍历模型并注入 hooks
- `create_benchmark_wrapper()`: 创建包装函数（使用 CUDA Event）
- `get_operator_type()`: 识别算子类型
- `get_tensor_shapes()`: 提取张量形状

### 3. `vllm/profiler/benchmark_config.py`
**状态**: 新建文件  
**职责**: Benchmark 配置数据结构
- `BenchmarkConfig` 数据类：包含所有配置参数
  - 模型配置（model_path, tp_size, pp_size 等）
  - Benchmark 参数（batch_sizes, seq_lengths, warmup_steps 等）
  - 算子选择（operators_to_benchmark）
  - 输出配置（output_file, include_per_layer_stats 等）
- `validate()`: 验证配置有效性
- `to_dict()`: 序列化为字典

### 4. `vllm/profiler/multiworker_aggregator.py`
**状态**: 新建文件  
**职责**: 多 worker 结果聚合和比较
- `aggregate_worker_results()`: 聚合多个 worker 的结果
- `compare_results()`: 比较两次 benchmark 的结果
- `merge_results_by_config()`: 合并不同配置的结果
- `save_worker_results_to_temp()` / `load_worker_results_from_temp()`: 临时文件 I/O

### 5. `vllm/profiler/__init__.py`
**状态**: 修改文件  
**职责**: 导出 profiler 模块的公共 API
**修改内容**:
- 添加了所有新类和函数的导出
- 包括：`OperatorBenchmark`, `enable_operator_benchmark`, `inject_benchmark_hooks`, `BenchmarkConfig` 等

### 6. `vllm/v1/worker/gpu_model_runner.py`
**状态**: 修改文件  
**职责**: 在 worker 进程中集成 benchmark 功能
**修改内容**:
- 在 `load_model()` 方法中添加 `_inject_operator_benchmark_if_enabled()` 调用
- 新增 `_inject_operator_benchmark_if_enabled()` 方法：
  - 读取环境变量 `VLLM_OPERATOR_BENCHMARK_ENABLE`
  - 读取环境变量 `VLLM_OPERATOR_BENCHMARK_OPS`
  - 读取环境变量 `VLLM_OPERATOR_BENCHMARK_WARMUP`
  - 调用 `enable_operator_benchmark()` 和 `inject_benchmark_hooks()`
- 在 `execute_model()` 方法中添加 `flush_pending_events()` 调用（在 forward pass 结束后）

---

## 脚本文件（2个）

### 7. `benchmarks/benchmark_operators.py`
**状态**: 新建文件  
**职责**: Benchmark 主脚本，提供命令行接口
**功能**:
- 解析配置文件或命令行参数
- 设置环境变量（在 LLM 初始化前）
- 初始化 vLLM 引擎
- 运行 warmup 和 benchmark 迭代
- 从临时文件读取 worker 结果
- 生成最终 JSON 报告
- 支持多种配置方式（Python 配置文件或命令行参数）

### 8. `benchmarks/operator_benchmark_example.py`
**状态**: 新建文件  
**职责**: 简单的 API 使用示例
**功能**:
- 演示如何通过 Python API 使用 benchmark 功能
- 展示环境变量设置的重要性
- 展示如何读取 worker 结果

---

## 配置文件（1个）

### 9. `benchmarks/operator_configs/qwen_config.py`
**状态**: 新建文件（可能由用户创建）  
**职责**: Qwen 模型的 benchmark 配置示例

---

## 文档文件（2个）

### 10. `benchmarks/OPERATOR_BENCHMARK.md`
**状态**: 新建文件  
**职责**: 用户使用指南
**内容**:
- 快速开始指南
- 配置说明
- 使用示例
- 结果解读
- 故障排除

### 11. `OPERATOR_BENCHMARK_IMPLEMENTATION.md`
**状态**: 新建文件  
**职责**: 实现文档（详细的技术文档）
**内容**:
- 系统架构概览
- 文件结构与职责
- 核心工作流程
- 关键设计决策
- 数据流分析
- 性能优化机制
- 多进程通信机制
- 关键代码路径

---

## 修改统计

### 新建文件（9个）
1. `vllm/profiler/operator_profiler.py`
2. `vllm/profiler/operator_injector.py`
3. `vllm/profiler/benchmark_config.py`
4. `vllm/profiler/multiworker_aggregator.py`
5. `benchmarks/benchmark_operators.py`
6. `benchmarks/operator_benchmark_example.py`
7. `benchmarks/OPERATOR_BENCHMARK.md`
8. `OPERATOR_BENCHMARK_IMPLEMENTATION.md`
9. `MODIFIED_FILES.md` (本文件)

### 修改文件（2个）
1. `vllm/profiler/__init__.py` - 添加导出
2. `vllm/v1/worker/gpu_model_runner.py` - 添加集成点

### 总计
- **新建文件**: 9 个
- **修改文件**: 2 个
- **总计**: 11 个文件

---

## 文件依赖关系

```
benchmarks/benchmark_operators.py
    ├─> vllm.profiler.BenchmarkConfig
    ├─> vllm.profiler.enable_operator_benchmark
    └─> vllm.profiler.get_operator_benchmark

vllm/v1/worker/gpu_model_runner.py
    ├─> vllm.profiler.enable_operator_benchmark
    └─> vllm.profiler.inject_benchmark_hooks

vllm/profiler/operator_injector.py
    └─> vllm.profiler.operator_profiler.get_operator_benchmark

vllm/profiler/operator_profiler.py
    └─> (独立，无依赖)

vllm/profiler/benchmark_config.py
    └─> (独立，无依赖)

vllm/profiler/multiworker_aggregator.py
    └─> vllm.profiler.operator_profiler.BenchmarkResults
```

---

## 关键修改点总结

### 1. 核心实现
- **operator_profiler.py**: 实现了单例模式的 `OperatorBenchmark` 类，支持批量 CUDA 同步优化
- **operator_injector.py**: 实现了非侵入式的 hook 注入机制

### 2. Worker 集成
- **gpu_model_runner.py**: 
  - 在 `load_model()` 中读取环境变量并注入 hooks
  - 在 `execute_model()` 中调用 `flush_pending_events()`

### 3. 性能优化
- **批量同步**: 从每个 operator 同步改为每个 forward pass 同步一次
- **延迟处理**: 使用 `PendingEvent` 队列，批量处理 CUDA Event

### 4. 多进程通信
- **环境变量**: 主进程 → Worker 进程（配置传递）
- **临时文件**: Worker 进程 → 主进程（结果传递）

---

## 注意事项

1. **环境变量必须在 LLM 初始化前设置**，否则 worker 进程无法读取
2. **临时文件路径**通过环境变量 `VLLM_OPERATOR_BENCHMARK_OUTPUT` 传递
3. **批量同步优化**是关键性能优化，避免了频繁 CUDA 同步导致的性能下降
4. **Warmup 机制**基于 forward pass 次数（`_flush_count`），而非 operator 调用次数

