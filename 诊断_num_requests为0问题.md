# 诊断：num_requests=0 问题

## 问题描述

运行benchmark后，结果JSON显示：
```json
{
  "e2e_metrics_aggregated": {
    "num_requests": 0,
    "num_valid_requests": 0
  }
}
```

但终端日志显示请求在被处理：
```
Trace benchmark: 100%|██████████| 171/171 [04:58<00:00, 1.75s/req]
```

## 根本原因

**所有请求都失败了**，导致没有有效输出：

```
[DP0] ERROR Failed to generate for request 680: EngineCore encountered an issue
Valid outputs: 0/171
WARNING No metrics collected! Check if RequestOutput.metrics is populated.
```

## 可能的原因和解决方案

### 原因1：Python缓存问题 ⚠️

修改代码后，Python可能仍在使用旧的缓存版本。

**解决方法**：

```bash
cd /mnt/disk1/ljm/vllm

# 清理所有Python缓存
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete 2>/dev/null

echo "✓ Python缓存已清理"
```

### 原因2：TokensPrompt未正确应用

检查`trace_scheduler.py`是否正确使用TokensPrompt。

**验证方法**：

```bash
cd /mnt/disk1/ljm/vllm

# 检查导入
grep "from vllm.inputs import TokensPrompt" vllm/profiler/trace_scheduler.py

# 检查返回类型
grep "-> TokensPrompt:" vllm/profiler/trace_scheduler.py

# 检查返回语句
grep "return TokensPrompt" vllm/profiler/trace_scheduler.py
```

**预期输出**：
```
from vllm.inputs import TokensPrompt
) -> TokensPrompt:
return TokensPrompt(prompt_token_ids=token_ids)
```

### 原因3：vLLM模块未重新加载

如果使用开发模式安装（`pip install -e .`），修改代码后不需要重新安装，但可能需要重启Python进程。

**解决方法**：

```bash
# 方法1：重新安装（如果使用普通安装）
cd /mnt/disk1/ljm/vllm
pip install -e . --no-build-isolation

# 方法2：确保没有旧的Python进程
pkill -f python
```

### 原因4：AsyncLLM兼容性问题

测试AsyncLLM是否能正确接受TokensPrompt。

**测试脚本**：

```bash
cd /mnt/disk1/ljm/vllm
python3 test_asyncllm_prompt.py
```

这会运行一个简单的测试来验证：
1. ✅ AsyncLLM能够初始化
2. ✅ TokensPrompt能够创建
3. ✅ AsyncLLM.generate()能接受TokensPrompt
4. ✅ 能够成功生成输出

## 诊断步骤

### 步骤1：清理缓存并验证代码

```bash
cd /mnt/disk1/ljm/vllm

# 1. 清理Python缓存
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null
find . -name "*.pyc" -delete

# 2. 验证TokensPrompt导入
python3 -c "from vllm.inputs import TokensPrompt; print('✓ TokensPrompt可导入')"

# 3. 验证trace_scheduler.py的修改
grep -n "TokensPrompt" vllm/profiler/trace_scheduler.py
```

### 步骤2：运行简化测试

```bash
cd /mnt/disk1/ljm/vllm

# 运行TokensPrompt测试
python3 test_asyncllm_prompt.py
```

**预期输出**：
```
================================================================================
测试AsyncLLM + TokensPrompt
================================================================================

1. 初始化AsyncLLM...
✓ AsyncLLM初始化成功

2. 创建TokensPrompt...
✓ TokensPrompt创建成功: {'prompt_token_ids': [1, 2, 3, 4, 5]}

3. 创建SamplingParams...
✓ SamplingParams创建成功

4. 测试异步生成...
   收到输出: finished=False
   收到输出: finished=True
✓ 生成成功！

5. 关闭AsyncLLM...
✓ AsyncLLM已关闭

================================================================================
✅ 所有测试通过！AsyncLLM可以正确接受TokensPrompt
================================================================================
```

### 步骤3：使用小规模trace测试

如果简化测试通过，用小规模trace测试：

```bash
cd /mnt/disk1/ljm/vllm

# 使用很小的trace（约7个请求）
python benchmarks/e2e_sweep/run_e2e_sweep.py \
    --num-gpus 1 \
    --base-config benchmarks/e2e_sweep/example_configs/e2e_config_template.py \
    --output-dir e2e_results/debug_test \
    --gpu-type h100 \
    --gpu-ids "4"
```

注意观察：
- ✅ 是否有"Failed to generate"错误
- ✅ "Valid outputs"是否大于0
- ✅ 结果JSON中`num_requests`是否大于0

### 步骤4：检查完整错误日志

如果仍然失败，查看完整的错误堆栈：

```bash
# 运行benchmark并保存完整日志
python benchmarks/e2e_sweep/run_e2e_sweep.py \
    --num-gpus 1 \
    --base-config benchmarks/e2e_sweep/example_configs/e2e_config_template.py \
    --output-dir e2e_results/debug_test \
    --gpu-type h100 \
    --gpu-ids "4" \
    2>&1 | tee benchmark_debug.log

# 查找错误堆栈
grep -A 20 "Traceback" benchmark_debug.log
grep -A 10 "ERROR" benchmark_debug.log
```

## 常见错误模式

### 错误1：提示格式不兼容

**症状**：
```
ERROR Failed to generate: EngineCore encountered an issue
```

**原因**：使用普通dict而不是TokensPrompt

**解决**：确保使用`TokensPrompt(prompt_token_ids=token_ids)`

### 错误2：属性访问错误

**症状**：
```
DEBUG Could not get TP size: 'AsyncLLM' object has no attribute 'llm_engine'
```

**原因**：尝试访问`llm.llm_engine`（AsyncLLM没有此属性）

**解决**：使用`llm.vllm_config`

### 错误3：模块未重新加载

**症状**：修改代码后问题依然存在

**原因**：Python缓存了旧版本

**解决**：清理`__pycache__`并重启

## 验证修复

修复后，重新运行benchmark：

```bash
cd /mnt/disk1/ljm/vllm

# 清理缓存
find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null

# 运行benchmark
python benchmarks/e2e_sweep/run_e2e_sweep.py \
    --num-gpus 4 \
    --base-config benchmarks/e2e_sweep/example_configs/e2e_config_template.py \
    --output-dir e2e_results/h100_8gpu_5min_k10_fixed \
    --gpu-type h100 \
    --gpu-ids "4,5,6,7"
```

**成功标志**：

1. ✅ 终端日志中**没有** "Failed to generate"错误
2. ✅ 显示 "Valid outputs: X/Y"（X > 0）
3. ✅ 结果JSON中 `num_requests > 0`
4. ✅ 有E2E metrics统计（TTFT, TPOT等）

**结果JSON示例**（成功）：
```json
{
  "e2e_metrics_aggregated": {
    "num_requests": 686,
    "num_valid_requests": 686,
    "ttft_ms": {
      "mean": 123.45,
      "p50": 100.23,
      "p90": 234.56,
      "p99": 456.78
    },
    // ... 其他metrics
  }
}
```

## 总结

`num_requests=0`的问题通常是因为：

1. **所有请求都失败** - 最常见原因
2. **Python缓存** - 修改代码后未生效
3. **TokensPrompt格式** - 使用了错误的输入格式
4. **模块未重新加载** - 开发模式安装后需要重启

按照上述步骤诊断和修复，问题应该能够解决。

## 快速修复检查清单

- [ ] 清理Python缓存
- [ ] 验证TokensPrompt导入
- [ ] 运行`test_asyncllm_prompt.py`测试
- [ ] 检查trace_scheduler.py的修改
- [ ] 使用小规模trace测试
- [ ] 查看完整错误日志
- [ ] 重新运行完整benchmark

