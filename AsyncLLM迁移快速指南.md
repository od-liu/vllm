# AsyncLLM迁移快速指南

## 🎯 一句话总结

将同步的`LLM`替换为完全异步的`AsyncLLM`，从根本上解决线程池阻塞导致的请求卡死问题。

## 📝 主要修改

### 修改的文件
1. ✅ `/mnt/disk1/ljm/vllm/vllm/profiler/trace_scheduler.py`
2. ✅ `/mnt/disk1/ljm/vllm/benchmarks/benchmark_e2e_metrics.py`

### 关键变化

#### 之前（同步+线程池）
```python
from vllm import LLM
llm = LLM(model="...", ...)
result = await loop.run_in_executor(
    executor,
    lambda: llm.generate([prompt], params)
)
```

#### 现在（完全异步+正确的prompt格式）
```python
from vllm.v1.engine.async_llm import AsyncLLM
from vllm.engine.arg_utils import AsyncEngineArgs
from vllm.inputs import TokensPrompt  # 重要！

engine_args = AsyncEngineArgs(model="...", ...)
llm = AsyncLLM.from_engine_args(engine_args)

# 使用TokensPrompt而不是普通dict
prompt = TokensPrompt(prompt_token_ids=token_ids)

async for output in llm.generate(
    request_id=request_id,
    prompt=prompt,  # TokensPrompt类型
    sampling_params=params
):
    if output.finished:
        result = output
        break
```

## 🚀 测试命令

```bash
cd /mnt/disk1/ljm/vllm

# 运行之前卡死的trace文件
python benchmarks/e2e_sweep/run_e2e_sweep.py \
    --trace-file qwen-bailian-usagetraces-anon/qwen_trace_5min_sparse.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-gpus 8 \
    --base-config benchmarks/e2e_sweep/example_configs/e2e_config_template.py \
    --output-dir e2e_results/asyncllm_test
```

## ✅ 成功标志

### 1. 初始化日志变化
**之前**：
```
INFO Initialized TraceScheduler with dedicated ThreadPoolExecutor (max_workers=10, max_concurrent=5)
```

**现在**：
```
INFO Initializing AsyncLLM engine (vLLM v1)...
INFO AsyncLLM engine initialized successfully
INFO Initialized TraceScheduler with AsyncLLM (max_concurrent=5)
```

### 2. **没有heartbeat输出**（最关键！）
**之前**：
```
[DP0] INFO 12-02 08:52:34 trace_scheduler.py:283]   ⏱ Request 58 still running (60s / 960s)...
[DP0] INFO 12-02 08:53:34 trace_scheduler.py:283]   ⏱ Request 58 still running (120s / 960s)...
```

**现在**：
```
（应该没有这些heartbeat消息！）
```

### 3. 顺利完成所有请求
```
Trace benchmark: 100%|██████████| 60/60 [XX:XX<00:00, X.XXs/req]
[DP0] INFO Trace progress: 60/60 requests (100.0%) | ...
INFO Shutting down AsyncLLM engine...
INFO AsyncLLM engine shutdown complete
```

## 🔍 故障排除

### 如果看到错误：`AsyncLLM` 找不到

**原因**：可能使用的vLLM版本太旧，不支持v1引擎。

**解决**：
```bash
# 检查vLLM版本
python -c "import vllm; print(vllm.__version__)"

# 确认AsyncLLM可用
python -c "from vllm.v1.engine.async_llm import AsyncLLM; print('OK')"
```

### 如果仍然卡住

1. **检查日志中是否使用了AsyncLLM**
   - 必须看到 "Initializing AsyncLLM engine (vLLM v1)"
   
2. **增加调试输出**
   ```bash
   export VLLM_LOGGING_LEVEL=DEBUG
   ```

3. **检查模型兼容性**
   - 某些模型可能对v1引擎有特殊要求

## 📊 预期性能提升

| 指标 | 之前（LLM+线程池） | 现在（AsyncLLM） |
|------|-------------------|------------------|
| 请求卡死 | 经常发生 | 应该消除 |
| heartbeat输出 | 持续出现 | 不再出现 |
| 线程开销 | 10个线程 | 0个（asyncio） |
| 延迟 | 较高（线程切换） | 较低（纯异步） |
| 代码复杂度 | 高（线程池+heartbeat） | 低（纯async/await） |

## 🔧 技术细节

### 为什么AsyncLLM能解决问题？

1. **直接信号传递**
   - LLM: `同步调用 → 线程池 → 结果返回`（可能丢失信号）
   - AsyncLLM: `异步生成器 → 直接yield`（保证信号传递）

2. **无线程阻塞**
   - LLM: 可能被线程池调度延迟
   - AsyncLLM: asyncio事件循环直接调度

3. **资源管理清晰**
   - LLM: 线程池资源可能耗尽
   - AsyncLLM: 无线程池，资源使用可预测

### AsyncGenerator工作原理

```python
async for output in llm.generate(...):
    # 每次生成新token时yield一个RequestOutput
    # output.finished==True时表示完成
    if output.finished:
        break
```

这确保了：
- ✅ 每个token生成都会更新状态
- ✅ 完成信号立即传递（无线程延迟）
- ✅ 可以随时中止（通过abort）

## 📚 相关文档

- **详细修改说明**：`迁移到AsyncLLM修复总结.md`
- **之前的修复历史**：`修复完成总结.md`
- **问题诊断分析**：`卡死问题分析.md`

## 💡 后续建议

### 如果成功
- ✅ 保留AsyncLLM实现
- ✅ 可以移除相关的heartbeat和线程池调试代码
- ✅ 考虑将其他同步LLM调用也迁移到AsyncLLM

### 如果失败
- 检查vLLM版本和模型兼容性
- 提供详细的错误日志
- 可能需要针对特定模型调整配置

## 🎉 预期结果

运行benchmark后，应该看到：
1. ✅ 无heartbeat输出
2. ✅ 所有60个请求顺利完成
3. ✅ 更快的执行时间
4. ✅ 更稳定的性能表现

**关键成功指标**：请求58不再卡住，整个benchmark正常完成！

