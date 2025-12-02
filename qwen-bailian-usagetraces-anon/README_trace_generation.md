# Trace File Generation Guide

## 生成的稀疏Trace文件

### 文件信息

**文件名**: `qwen_trace_5min_sparse.jsonl`

**特点**:
- ⏱️ **时长**: 约5分钟 (280.8秒)
- 📊 **请求数**: 60个
- 📈 **请求密度**: ~12.8 req/min（比原始trace稀疏很多）
- 📝 **Input长度**: 175-2369 tokens (平均 676 tokens)
- ✍️ **Output长度**: 5-194 tokens (平均 59 tokens) ✅ **比原始小很多**

### 与原始trace对比

| 特征 | 原始 qwen_traceB_blksz_16.jsonl | 新生成 qwen_trace_5min_sparse.jsonl |
|------|--------------------------------|-------------------------------------|
| 时长 | 0.716秒内25个请求 | 280.8秒内60个请求 |
| 密度 | 极高 (~2100 req/min) | 稀疏 (~12.8 req/min) |
| Output长度 | 1-1161 tokens (avg ~150) | 5-194 tokens (avg ~59) ✓ |
| 适用场景 | 压力测试 | 实际生产负载模拟 |

## 使用方法

### 在E2E Benchmark中使用

编辑配置文件 `e2e_config_template.py`:

```python
config = BenchmarkConfig(
    # ... 其他配置 ...
    
    # 使用新生成的稀疏trace
    trace_file_path="qwen-bailian-usagetraces-anon/qwen_trace_5min_sparse.jsonl",
    trace_time_range_minutes=(0, 5),  # 完整5分钟
    trace_realtime_replay=True,       # 实时回放模式
    
    # ... 其他配置 ...
)
```

### 运行Benchmark

```bash
python benchmarks/e2e_sweep/run_e2e_sweep.py \
    --num-gpus 8 \
    --base-config your_config.py \
    --output-dir e2e_results/5min_sparse_test \
    --gpu-type h100
```

### 预期结果

使用这个稀疏trace，您将能够：

1. **观察到TP/DP配置的真实差异**
   - 稀疏负载下，不同配置的吞吐量可能会有差异
   - TTFT差异会更加明显

2. **更长的测试时间**
   - 5分钟的测试 vs 之前的2分钟
   - 更稳定的metrics统计

3. **更小的output长度**
   - 减少生成负载，更关注首token延迟(TTFT)
   - 适合评估推理性能而非生成性能

## 自定义生成其他Trace

脚本 `generate_sparse_trace.py` 可以自定义参数：

```python
# 编辑脚本底部的参数
requests = generate_requests(
    num_requests=60,    # 请求数量
    duration=300        # 时长(秒)
)
```

### 常见配置示例

**轻负载 (5 req/min)**:
```python
requests = generate_requests(num_requests=25, duration=300)
```

**中等负载 (20 req/min)**:
```python
requests = generate_requests(num_requests=100, duration=300)
```

**重负载 (40 req/min)**:
```python
requests = generate_requests(num_requests=200, duration=300)
```

**10分钟trace**:
```python
requests = generate_requests(num_requests=120, duration=600)
```

运行生成:
```bash
python3 qwen-bailian-usagetraces-anon/generate_sparse_trace.py
```

## Trace格式说明

每行是一个JSON对象:

```json
{
  "chat_id": 0,                    // 对话ID
  "parent_chat_id": -1,            // 父对话ID (-1表示新对话)
  "timestamp": 0.0,                // 时间戳(秒)
  "input_length": 175,             // 输入token数
  "output_length": 55,             // 输出token数
  "type": "api",                   // 类型: "api" 或 "text"
  "turn": 1,                       // 对话轮次
  "hash_ids": [0, 1, 2, ...]      // Token ID映射(用于生成输入)
}
```

## 建议

### 场景1: 评估延迟(TTFT)
```python
# 使用稀疏trace + 小output
trace_file_path="qwen_trace_5min_sparse.jsonl"
trace_realtime_replay=True
```
→ 关注TTFT指标

### 场景2: 评估吞吐量
```python
# 使用稀疏trace + 非实时回放
trace_file_path="qwen_trace_5min_sparse.jsonl"
trace_realtime_replay=False  # 压力测试模式
```
→ 关注throughput指标，能看到DP的优势

### 场景3: 长时间稳定性测试
```python
# 生成30分钟trace
requests = generate_requests(num_requests=400, duration=1800)
```
→ 评估长时间运行稳定性

## 文件位置

所有trace文件在:
```
qwen-bailian-usagetraces-anon/
├── qwen_traceB_blksz_16.jsonl      # 原始密集trace
├── qwen_traceB_test_2min.jsonl     # 2分钟测试trace
├── qwen_trace_5min_sparse.jsonl    # 新生成的5分钟稀疏trace ✨
└── generate_sparse_trace.py        # 生成脚本
```

