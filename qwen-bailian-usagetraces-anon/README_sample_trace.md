# Trace采样工具使用说明

## 功能

`sample_trace.py` 是一个从大型trace文件中提取和采样请求的工具。

**主要功能**：
- 提取前N分钟的请求
- 按时间顺序每k个请求保留1个
- 重新编号chat_id（从0开始连续）
- 显示详细的统计信息

## 使用方法

### 基本用法

```bash
python sample_trace.py --input INPUT_FILE --output OUTPUT_FILE --k SAMPLE_RATE --minutes TIME_LIMIT
```

### 参数说明

| 参数 | 必需 | 默认值 | 说明 |
|------|------|--------|------|
| `--input` | 是 | - | 输入的JSONL trace文件路径 |
| `--output` | 是 | - | 输出的JSONL文件路径 |
| `--k` | 否 | 10 | 采样率：每k个请求保留1个 |
| `--minutes` | 否 | 5 | 提取前N分钟的数据 |

## 使用示例

### 示例1：提取前5分钟，每10个保留1个

```bash
python sample_trace.py \
    --input qwen_traceB_blksz_16.jsonl \
    --output qwen_trace_5min_sampled_k10.jsonl \
    --k 10 \
    --minutes 5
```

**结果**：
- 原始：前5分钟有6852个请求
- 采样后：686个请求（6852 / 10 ≈ 686）
- 请求密度：约137个请求/分钟

### 示例2：提取前3分钟，每20个保留1个

```bash
python sample_trace.py \
    --input qwen_traceB_blksz_16.jsonl \
    --output qwen_trace_3min_sampled_k20.jsonl \
    --k 20 \
    --minutes 3
```

**预期**：
- 更短的时间范围（3分钟）
- 更稀疏的采样（k=20）
- 生成更小的trace文件

### 示例3：提取前10分钟，每5个保留1个

```bash
python sample_trace.py \
    --input qwen_traceB_blksz_16.jsonl \
    --output qwen_trace_10min_sampled_k5.jsonl \
    --k 5 \
    --minutes 10
```

**预期**：
- 更长的时间范围（10分钟）
- 更密集的采样（k=5）
- 生成更大的trace文件

## 输出示例

运行脚本时会显示详细的统计信息：

```
================================================================================
Trace文件采样工具
================================================================================
输入文件: qwen_traceB_blksz_16.jsonl
输出文件: qwen_trace_5min_sampled_k10.jsonl
采样率: 每 10 个保留 1 个
时间范围: 前 5 分钟
================================================================================
📖 正在读取 qwen_traceB_blksz_16.jsonl...
✓ 前 5 分钟内共有 6852 个请求
✓ 采样率 k=10，保留 686 个请求
💾 正在保存到 qwen_trace_5min_sampled_k10.jsonl...
✅ 完成！

📊 采样结果统计:
  时间范围: 0 - 5 分钟
  请求总数: 686
  请求密度: 137.2 请求/分钟
  实际时长: 300.0 秒 (5.0 分钟)

  输入长度统计:
    最小值: 31
    最大值: 14015
    平均值: 796
    中位数: 504

  输出长度统计:
    最小值: 1
    最大值: 2038
    平均值: 78
    中位数: 34

📝 前 5 个请求预览:
  [0] t=0.000s, input=1261, output=102, type=api
  [1] t=0.194s, input=194, output=8, type=api
  [2] t=0.576s, input=108, output=1, type=api
  [3] t=1.175s, input=258, output=18, type=api
  [4] t=1.545s, input=434, output=26, type=api
```

## 采样策略

### 时间顺序采样

脚本采用**时间顺序采样**：
1. 读取前N分钟内的所有请求
2. 按时间戳排序
3. 从第0个开始，每隔k个保留1个

**示例**（k=10）：
```
原始索引: 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20...
保留索引: 0,                   10,                    20...
          ↑                    ↑                     ↑
```

### chat_id重新编号

采样后，所有请求的`chat_id`会被重新编号为0, 1, 2, 3...，确保连续性。

## 使用场景

### 场景1：快速测试

```bash
# 生成小规模测试数据（约60个请求）
python sample_trace.py \
    --input qwen_traceB_blksz_16.jsonl \
    --output test_trace.jsonl \
    --k 100 \
    --minutes 5
```

### 场景2：性能基准测试

```bash
# 生成中等规模测试数据（约340个请求）
python sample_trace.py \
    --input qwen_traceB_blksz_16.jsonl \
    --output benchmark_trace.jsonl \
    --k 20 \
    --minutes 5
```

### 场景3：压力测试

```bash
# 生成大规模测试数据（约1370个请求）
python sample_trace.py \
    --input qwen_traceB_blksz_16.jsonl \
    --output stress_trace.jsonl \
    --k 5 \
    --minutes 5
```

## 选择合适的k值

| k值 | 请求数量（5分钟） | 请求密度 | 适用场景 |
|-----|-----------------|---------|---------|
| 100 | ~69 | ~14/分钟 | 快速功能测试 |
| 50 | ~137 | ~27/分钟 | 单元测试 |
| 20 | ~343 | ~69/分钟 | 集成测试 |
| 10 | ~686 | ~137/分钟 | 性能测试 |
| 5 | ~1370 | ~274/分钟 | 压力测试 |
| 2 | ~3426 | ~685/分钟 | 极限压力测试 |

## 验证采样结果

### 检查文件行数
```bash
wc -l qwen_trace_5min_sampled_k10.jsonl
```

### 查看前几行
```bash
head -5 qwen_trace_5min_sampled_k10.jsonl
```

### 查看最后几行
```bash
tail -5 qwen_trace_5min_sampled_k10.jsonl
```

### 验证时间范围
```bash
python3 -c "
import json
with open('qwen_trace_5min_sampled_k10.jsonl', 'r') as f:
    lines = f.readlines()
    first = json.loads(lines[0])
    last = json.loads(lines[-1])
    print(f'第一个请求时间: {first[\"timestamp\"]:.3f}s')
    print(f'最后一个请求时间: {last[\"timestamp\"]:.3f}s')
    print(f'时间跨度: {last[\"timestamp\"]:.1f}s ({last[\"timestamp\"]/60:.1f}分钟)')
"
```

## 在benchmark中使用

生成采样文件后，可以在benchmark配置中使用：

```python
# benchmarks/e2e_sweep/example_configs/e2e_config_template.py
config = BenchmarkConfig(
    # ...
    trace_file_path="qwen-bailian-usagetraces-anon/qwen_trace_5min_sampled_k10.jsonl",
    trace_time_range_minutes=(0, 5),
    trace_realtime_replay=True,
    # ...
)
```

或在命令行中使用：

```bash
python benchmarks/e2e_sweep/run_e2e_sweep.py \
    --trace-file qwen-bailian-usagetraces-anon/qwen_trace_5min_sampled_k10.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-gpus 4 \
    --base-config benchmarks/e2e_sweep/example_configs/e2e_config_template.py \
    --output-dir e2e_results/sampled_test
```

## 常见问题

### Q1: 为什么采样后的请求数不是正好 total/k？

**A**: 因为整数除法的余数被丢弃。例如6852/10 = 685.2，实际保留686个（0, 10, 20, ..., 6850）。

### Q2: 采样是否保持时间分布？

**A**: 是的。采样按时间顺序进行，保持原始时间戳不变，因此时间分布特性得以保留。

### Q3: hash_ids是否需要重新生成？

**A**: 不需要。脚本保留原始的hash_ids，确保输入数据的一致性。

### Q4: 如何选择合适的k值？

**A**: 根据你的测试目标：
- 功能测试：k=50-100（快速验证）
- 性能测试：k=10-20（代表性负载）
- 压力测试：k=2-5（高负载）

### Q5: 可以采样多个时间段吗？

**A**: 当前版本只支持从开始时间采样。如需其他时间段，可以修改脚本中的`req['timestamp'] <= time_limit_seconds`条件。

## 相关文件

- `qwen_traceB_blksz_16.jsonl` - 原始大型trace文件（172800行）
- `qwen_trace_5min_sparse.jsonl` - 之前生成的稀疏trace（60行）
- `generate_sparse_trace.py` - 生成合成trace的脚本

## 总结

`sample_trace.py`是一个简单但强大的工具，可以：
- ✅ 快速从大文件中提取子集
- ✅ 灵活控制采样率和时间范围
- ✅ 保持时间分布特性
- ✅ 自动重新编号chat_id
- ✅ 提供详细的统计信息

适用于各种测试场景，从快速功能验证到压力测试。







