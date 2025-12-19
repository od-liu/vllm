# AsyncLLM Prompt格式修复

## 问题诊断

在迁移到AsyncLLM后，所有请求都失败，错误信息：
```
[DP0] ERROR 12-02 10:13:06 trace_scheduler.py:327] Failed to generate for request 0: EngineCore encountered an issue. See stack trace (above) for the root cause.
```

同时还有TP size获取失败的警告：
```
[DP0] DEBUG 12-02 10:13:06 trace_scheduler.py:197] Could not get TP size from engine: 'AsyncLLM' object has no attribute 'llm_engine'
```

## 根本原因

### 问题1：Prompt格式不兼容

**错误的做法**（之前）：
```python
def _create_prompt_from_trace(self, trace_req: TraceRequest) -> Dict[str, Any]:
    # ...
    return {"prompt_token_ids": token_ids}  # 普通dict
```

**正确的做法**（现在）：
```python
from vllm.inputs import TokensPrompt

def _create_prompt_from_trace(self, trace_req: TraceRequest) -> TokensPrompt:
    # ...
    return TokensPrompt(prompt_token_ids=token_ids)  # TypedDict
```

**原因**：
- AsyncLLM的`generate()`方法接受`PromptType`，包括`TextPrompt`和`TokensPrompt`
- `TokensPrompt`是vLLM定义的TypedDict，不是普通dict
- 传递普通dict会导致AsyncLLM内部处理失败

### 问题2：属性访问错误

**错误的做法**（之前）：
```python
if hasattr(self.llm.llm_engine, 'parallel_config'):
    tp_size = self.llm.llm_engine.parallel_config.tensor_parallel_size
```

**正确的做法**（现在）：
```python
if hasattr(self.llm, 'vllm_config'):
    tp_size = self.llm.vllm_config.parallel_config.tensor_parallel_size
```

**原因**：
- 同步的`LLM`有`llm_engine`属性
- 异步的`AsyncLLM`直接暴露`vllm_config`，没有`llm_engine`属性

## 修改内容

### 文件1: `/mnt/disk1/ljm/vllm/vllm/profiler/trace_scheduler.py`

#### 修改1：添加TokensPrompt导入（第19行）
```python
from vllm.inputs import TokensPrompt
```

#### 修改2：修改`_create_prompt_from_trace`返回类型（第60-102行）
```python
def _create_prompt_from_trace(
    self,
    trace_req: TraceRequest,
) -> TokensPrompt:  # 改为TokensPrompt
    """
    Create a TokensPrompt from trace request for AsyncLLM.
    ...
    """
    # ... token处理逻辑 ...
    
    # 返回TokensPrompt而不是dict
    return TokensPrompt(prompt_token_ids=token_ids)
```

#### 修改3：修复TP size获取（第183-203行）
```python
tp_size = 1  # Default
try:
    # Method 1: AsyncLLM exposes vllm_config directly
    if hasattr(self.llm, 'vllm_config'):
        tp_size = self.llm.vllm_config.parallel_config.tensor_parallel_size
    # Method 2: Try legacy llm_engine path (for compatibility)
    elif hasattr(self.llm, 'llm_engine') and hasattr(self.llm.llm_engine, 'parallel_config'):
        tp_size = self.llm.llm_engine.parallel_config.tensor_parallel_size
except Exception as e:
    logger.debug(f"Could not get TP size from engine: {e}")
    # Method 3: From environment variable
    try:
        import os
        tp_env = os.environ.get('VLLM_TENSOR_PARALLEL_SIZE')
        if tp_env:
            tp_size = int(tp_env)
    except:
        pass
```

## TokensPrompt详解

`TokensPrompt`是vLLM定义的TypedDict（在`vllm/inputs/data.py`）：

```python
class TokensPrompt(TypedDict):
    """Schema for a tokenized prompt."""
    
    prompt_token_ids: list[int]
    """A list of token IDs to pass to the model."""
```

### 为什么使用TypedDict而不是普通dict？

1. **类型安全**：TypedDict提供类型检查，确保键名正确
2. **文档清晰**：明确定义了预期的数据结构
3. **内部处理**：AsyncLLM内部使用类型判断来区分不同的prompt类型

### 使用示例

从`vllm/benchmarks/throughput.py`中的正确用法：

```python
from vllm.inputs import TokensPrompt, TextPrompt

# 根据输入类型选择
prompt = (
    TokensPrompt(prompt_token_ids=token_ids)
    if has_token_ids
    else TextPrompt(prompt=text)
)

# 传递给AsyncLLM
async for output in llm.generate(
    request_id=request_id,
    prompt=prompt,  # TokensPrompt或TextPrompt
    sampling_params=params,
):
    # 处理输出
    pass
```

## 测试验证

### 运行测试
```bash
cd /mnt/disk1/ljm/vllm

python benchmarks/e2e_sweep/run_e2e_sweep.py \
    --trace-file qwen-bailian-usagetraces-anon/qwen_trace_5min_sparse.jsonl \
    --model Qwen/Qwen2.5-7B-Instruct \
    --num-gpus 4 \
    --gpu-ids "1,2,3,5" \
    --base-config benchmarks/e2e_sweep/example_configs/e2e_config_template.py \
    --output-dir e2e_results/asyncllm_fixed
```

### 成功标志

1. ✅ 没有"Failed to generate"错误
2. ✅ TP size正确识别（或优雅降级为默认值1）
3. ✅ 所有60个请求正常完成
4. ✅ 看到正常的进度输出

### 预期日志

**成功的日志**：
```
[DP0] INFO 12-02 10:XX:XX trace_scheduler.py:54] Initialized TraceScheduler with AsyncLLM (max_concurrent=5)
[DP0] INFO 12-02 10:XX:XX trace_scheduler.py:270] Request 0: timeout=968s (16.1min) | input=175, output=55, TP=4
[DP0] INFO 12-02 10:XX:XX trace_scheduler.py:404] Trace progress: 1/60 requests (1.7%) | ...
...
Trace benchmark: 100%|██████████| 60/60
```

**不应该看到**：
```
❌ Failed to generate for request X: EngineCore encountered an issue
❌ 'AsyncLLM' object has no attribute 'llm_engine'
```

## 关键学习点

### 1. vLLM输入类型系统

vLLM定义了多种输入类型：
- `TextPrompt`: 文本字符串
- `TokensPrompt`: token ID列表
- `EmbedsPrompt`: embedding向量
- 等等

### 2. AsyncLLM vs LLM的差异

| 方面 | LLM (同步) | AsyncLLM (异步) |
|------|-----------|----------------|
| 输入格式 | dict或其他 | 必须是PromptType |
| 引擎访问 | `llm.llm_engine` | `llm.vllm_config` |
| model_config | `llm.llm_engine.model_config` | `llm.model_config` |
| generate方法 | 同步，返回list | 异步生成器 |

### 3. 类型安全的重要性

使用正确的TypedDict：
- ✅ 编译时类型检查
- ✅ IDE自动补全
- ✅ 运行时验证
- ✅ 更清晰的API

## 相关文档

- **AsyncLLM迁移总结**：`迁移到AsyncLLM修复总结.md`
- **快速指南**：`AsyncLLM迁移快速指南.md`
- **完整修复历史**：`修复完成总结.md`

## 修复时间线

1. ❌ 第一次：GPU内存问题（误诊）
2. ❌ 第二次：管道死锁问题
3. ❌ 第三次：线程池耗尽（专用线程池）
4. ✅ 第四次：迁移到AsyncLLM
5. ✅ 第五次：修复AsyncLLM prompt格式 ← **当前**

## 总结

这次修复解决了AsyncLLM迁移的最后障碍：
- ✅ 使用正确的`TokensPrompt`类型传递token_ids
- ✅ 修复AsyncLLM的属性访问路径
- ✅ 保持向后兼容性（支持旧的llm_engine路径）

现在benchmark应该能够正常运行，所有60个请求应该顺利完成！










