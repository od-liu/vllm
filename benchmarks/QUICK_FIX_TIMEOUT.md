# 超时问题快速修复指南 🚀

## 🔴 问题：所有请求都超时了

### 立即解决方案（3种方法任选）

#### ⭐ 方法1：使用便捷脚本（推荐）

```bash
cd /mnt/disk1/ljm/vllm

# 2小时超时（适合大多数场景）
./benchmarks/run_with_custom_timeout.sh \
    -t 7200 \
    -c benchmarks/operator_configs/e2e_trace_config.py \
    -p 2

# 4小时超时（超长序列）
./benchmarks/run_with_custom_timeout.sh \
    -t 14400 \
    -c benchmarks/operator_configs/e2e_trace_config.py \
    -p 2
```

#### 方法2：手动设置环境变量

```bash
# 设置2小时超时
export VLLM_BENCHMARK_TIMEOUT=7200

# 运行benchmark
python benchmarks/benchmark_operators.py \
    --config benchmarks/operator_configs/e2e_trace_config.py \
    --phase 2
```

#### 方法3：一行命令

```bash
VLLM_BENCHMARK_TIMEOUT=7200 python benchmarks/benchmark_operators.py \
    --config benchmarks/operator_configs/e2e_trace_config.py \
    --phase 2
```

---

## 📊 如何选择正确的Timeout值？

### 快速参考表

| 最大输出长度 | TP=1-2 | TP=4 | TP=8 |
|-------------|--------|------|------|
| < 100       | 1800s (30min) | 2400s (40min) | 3000s (50min) |
| 100-500     | 3600s (1h) | 4800s (1.3h) | 6000s (1.7h) |
| 500-1000    | 7200s (2h) | 9600s (2.7h) | 12000s (3.3h) |
| > 1000      | 14400s (4h) | 19200s (5.3h) | 24000s (6.7h) |

### 🎯 推荐值

**保守但安全**：`VLLM_BENCHMARK_TIMEOUT=14400`（4小时）

**如果仍然超时**：`VLLM_BENCHMARK_TIMEOUT=28800`（8小时）

**调试模式**：`VLLM_BENCHMARK_TIMEOUT=none`（禁用，不推荐生产环境）

---

## 🔍 验证修复是否生效

运行benchmark后，查看日志应该显示：

### ✅ 正确的日志（修复后）

```
Request 1234: timeout=7200s (120.0min) | input=500, output=1000, TP=8
  ⏱ Request 1234 still running (30s / 7200s)...
  ⏱ Request 1234 still running (60s / 7200s)...
```

**关键指标**：
- `timeout` 应该是一个较大的值（>1800s）
- `TP=` 应该正确显示你的tensor_parallel_size（不应该是1如果你用了TP>1）

### ❌ 错误的日志（未修复）

```
Request 1234: timeout=600s (10.0min) | input=500, output=1000, TP=1
❌ REQUEST TIMEOUT
  Timeout used: 600.0s (10.0 minutes)
```

**问题指标**：
- `timeout=600s` 太小
- `TP=1` 但实际使用了TP=8（检测失败）

---

## 🛠 高级调试

### 检查TP大小是否正确检测

如果日志显示 `TP=1` 但你明确使用了TP=8：

```bash
# 手动设置TP大小
export VLLM_TENSOR_PARALLEL_SIZE=8
export VLLM_BENCHMARK_TIMEOUT=14400

python benchmarks/benchmark_operators.py \
    --config benchmarks/operator_configs/e2e_trace_config.py \
    --phase 2
```

### 监控GPU使用情况

```bash
# 在另一个终端运行
watch -n 1 nvidia-smi

# 观察：
# 1. GPU利用率应该接近100%
# 2. 内存使用应该稳定
# 3. 如果看到0% utilization → vLLM可能卡死
```

### 查看详细请求信息

在trace执行开始时，日志会显示：

```
Running trace requests for operator performance collection...
  Total requests: 30
  Output length - max: 1000, avg: 850.5
```

根据 `max: 1000` 来选择timeout：
- max < 200: 使用 3600s（1小时）
- max < 500: 使用 7200s（2小时）
- max < 1000: 使用 14400s（4小时）
- max >= 1000: 使用 28800s（8小时）

---

## 🎓 完整工作流示例

```bash
#!/bin/bash
# 我的benchmark运行脚本

# 1. 设置GPU
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# 2. 设置超时（4小时）
export VLLM_BENCHMARK_TIMEOUT=14400

# 3. （可选）手动设置TP大小
export VLLM_TENSOR_PARALLEL_SIZE=8

# 4. 运行Phase 2
echo "Starting Phase 2: Operator Benchmarking"
python benchmarks/benchmark_operators.py \
    --config benchmarks/operator_configs/e2e_trace_config.py \
    --phase 2

# 5. 检查结果
if [ $? -eq 0 ]; then
    echo "✓ Benchmark completed successfully!"
    # 查看结果文件
    ls -lh benchmark_results/
else
    echo "✗ Benchmark failed"
    exit 1
fi
```

保存为 `my_benchmark.sh`，然后：

```bash
chmod +x my_benchmark.sh
./my_benchmark.sh
```

---

## ❓ 常见问题

### Q: 我应该用多大的timeout？

**A**: 从4小时（14400s）开始。如果还超时，翻倍到8小时。

### Q: 禁用timeout（设为none）安全吗？

**A**: 不推荐。如果vLLM真的卡死，进程会永远运行。更好的做法是设一个很大的值如24小时（86400s）。

### Q: 为什么即使设置了很大的timeout还是超时？

**A**: 可能是：
1. TP大小检测失败 → 手动设置 `VLLM_TENSOR_PARALLEL_SIZE`
2. GPU性能问题 → 检查 `nvidia-smi`
3. 序列确实太长 → 考虑减少output_length或增加更多GPU

### Q: 如何知道我的请求实际需要多长时间？

**A**: 先用少量请求（5个）测试：
```bash
# 只测试5个请求
# 修改trace配置中的time_range_minutes=1（取前1分钟的数据）
# 或手动截取trace文件的前5行
```

观察最慢请求的时间，然后设置 `timeout = 实际时间 * 1.5`

---

## 📚 详细文档

- **完整配置指南**: `TIMEOUT_CONFIGURATION.md`
- **脚本使用**: `./run_with_custom_timeout.sh --help`
- **原理说明**: 查看 `vllm/profiler/trace_scheduler.py` 中的timeout计算逻辑

---

## 🆘 还是不行？

1. **检查文件**: 确认你使用的是最新修复后的代码
2. **查看日志**: 搜索 "REQUEST TIMEOUT" 查看详细错误信息
3. **监控GPU**: 使用 `nvidia-smi` 确认GPU正在工作
4. **减少负载**: 尝试减少TP大小或output length
5. **联系支持**: 提供完整的错误日志和配置信息

---

最后更新：2025-12-01




