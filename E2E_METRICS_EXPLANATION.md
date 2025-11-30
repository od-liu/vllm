# E2E Metrics 指标说明

本文档详细解释端到端（E2E）性能指标的含义和计算逻辑。

## 指标概览

### 1. TTFT (Time To First Token) - 首Token延迟

**含义**：从请求到达（arrival）到生成第一个输出token的时间。

**计算公式**（v1引擎）：
```
TTFT = first_token_latency
```
其中 `first_token_latency` 是vLLM引擎已经计算好的wall-clock时间差：
```
first_token_latency = first_token_time (wall-clock) - arrival_time (wall-clock)
```

**单位**：毫秒（ms）

**你的结果**：
- Mean: 137.88 ms
- P50: 137.30 ms
- P90: 165.96 ms
- P99: 170.95 ms

**解读**：平均来说，用户需要等待约138毫秒才能看到第一个token生成。

---

### 2. TPOT (Time Per Output Token) - 每个输出Token的平均时间

**含义**：生成每个后续输出token的平均时间（不包括第一个token，因为第一个token在prefill阶段生成）。

**计算公式**（v1引擎）：
```
decode_time = last_token_ts - first_token_ts  (monotonic时间戳差)
TPOT = decode_time / (num_generation_tokens - 1)
```

**为什么减1？**
- 第一个token在prefill阶段生成（包含在TTFT中）
- 后续的token在decode阶段生成
- 所以TPOT只计算decode阶段的token生成速度

**单位**：毫秒（ms）

**你的结果**：
- Mean: 103.39 ms
- P50: 103.82 ms
- P90: 129.24 ms
- P99: 132.23 ms

**解读**：平均来说，每个后续token需要约103毫秒生成。这意味着生成速度约为 1000/103 ≈ 9.7 tokens/秒。

---

### 3. E2E Latency (End-to-End Latency) - 端到端延迟

**含义**：从请求到达（arrival）到最后一个token生成完成的总时间。

**计算公式**（v1引擎，近似）：
```
decode_time = last_token_ts - first_token_ts  (monotonic时间戳差)
E2E Latency ≈ first_token_latency + decode_time
```

**为什么是近似值？**
- `first_token_latency` 是wall-clock时间差
- `decode_time` 是monotonic时间戳差
- 我们假设两种时钟的速率相似（通常成立）
- 更准确的方法需要引擎在完成时记录wall-clock时间，但v1引擎的`RequestStateStats`不包含这个信息

**单位**：毫秒（ms）

**你的结果**（修复前）：
- Mean: 58267.12 ms (≈58秒) - **这个值明显异常，说明之前的计算有问题**

**修复后的预期值**：
假设平均生成50个token：
```
E2E Latency ≈ TTFT + (TPOT × (num_tokens - 1))
            ≈ 138ms + (103ms × 49)
            ≈ 138ms + 5047ms
            ≈ 5185ms (约5.2秒)
```

---

### 4. Queued Time - 排队时间

**含义**：请求在调度器队列中等待的时间（从进入队列到被调度执行）。

**计算公式**（v1引擎）：
```
Queued Time = scheduled_ts - queued_ts
```
两个时间戳都是monotonic时间戳，所以差值有效。

**单位**：毫秒（ms）

**你的结果**：
- Mean: 0.19 ms
- P50: 0.20 ms
- P90: 0.23 ms
- P99: 0.23 ms

**解读**：平均排队时间只有0.19毫秒，说明系统负载不高，请求几乎立即被调度。

---

## 时间戳类型说明

vLLM v1引擎使用两种时间戳：

1. **Wall-clock时间**（`arrival_time`）：
   - 使用 `time.time()` 获取
   - 受系统时钟调整影响
   - 用于计算与外部世界相关的时间差

2. **Monotonic时间戳**（`queued_ts`, `scheduled_ts`, `first_token_ts`, `last_token_ts`）：
   - 使用 `time.monotonic()` 获取
   - 不受系统时钟调整影响
   - 只用于计算时间间隔（差值）
   - 不能直接与wall-clock时间相减

## 计算逻辑总结

### v1引擎（RequestStateStats）

```python
# TTFT: 直接使用引擎计算的first_token_latency
ttft_ms = first_token_latency * 1000

# Queued Time: monotonic时间戳差
queued_time_ms = (scheduled_ts - queued_ts) * 1000

# TPOT: decode阶段时间 / (token数 - 1)
decode_time = last_token_ts - first_token_ts
tpot_ms = (decode_time / (num_generation_tokens - 1)) * 1000

# E2E Latency: TTFT + decode_time (近似)
e2e_latency_ms = (first_token_latency + decode_time) * 1000
```

### v0引擎（RequestMetrics）

```python
# TTFT: wall-clock时间差
ttft_ms = (first_token_time - arrival_time) * 1000

# E2E Latency: wall-clock时间差
e2e_latency_ms = (finished_time - arrival_time) * 1000

# TPOT: (finished_time - first_token_time) / (token数 - 1)
tpot_ms = ((finished_time - first_token_time) / (num_generation_tokens - 1)) * 1000

# Queued Time: 直接使用
queued_time_ms = time_in_queue * 1000
```

## 注意事项

1. **E2E Latency是近似值**：由于v1引擎的限制，我们使用 `TTFT + decode_time` 来近似E2E延迟。这在大多数情况下是准确的，但可能略低于实际值（因为不包括一些边缘情况的时间）。

2. **TPOT不包括第一个token**：第一个token在prefill阶段生成，后续token在decode阶段生成。TPOT只反映decode阶段的性能。

3. **时间戳类型**：不能直接混合使用wall-clock和monotonic时间戳。我们只在计算差值时使用monotonic时间戳。

4. **统计值**：每个指标都提供mean、p50、p90、p99统计值，帮助了解分布情况。

## 性能分析示例

基于你的结果：

```
TTFT (mean): 137.88 ms
TPOT (mean): 103.39 ms
Queued Time (mean): 0.19 ms
```

**分析**：
- 首token延迟：138ms（正常范围）
- Token生成速度：约9.7 tokens/秒（103ms/token）
- 排队时间：几乎为0（系统负载低）
- 如果生成50个token，总时间约为：138ms + (103ms × 49) ≈ 5.2秒

**优化建议**：
- TTFT可以通过优化prefill阶段来改善
- TPOT可以通过优化decode阶段来改善（如使用更快的GPU、优化KV cache等）

