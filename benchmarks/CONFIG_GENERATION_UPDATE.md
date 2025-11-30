# Configuration Generation Logic Update

## Summary

Updated the parallel configuration generation to support **all TP×DP combinations** now that multi-process DP is fully implemented.

## What Changed

### Before: TP >= DP Constraint

**Old Logic:**
```python
for tp in tp_values:
    dp = num_gpus // tp
    # Apply constraint: TP must be >= DP
    if tp >= dp:
        configs.append((tp, pp, dp))
```

**4 GPU Example:**
- ✅ TP=4, DP=1 (allowed)
- ✅ TP=2, DP=2 (allowed)
- ❌ TP=1, DP=4 (rejected, TP < DP)

**Rationale:** Single-process `LLM(data_parallel_size>1)` was not supported.

### After: All Combinations Supported

**New Logic:**
```python
for tp in tp_values:
    dp = num_gpus // tp
    # With multiprocessing DP support, all combinations are valid
    configs.append((tp, pp, dp))

# Sort by DP (ascending), then TP (descending)
configs.sort(key=lambda x: (x[2], -x[0]))
```

**4 GPU Example:**
- ✅ DP=1, TP=4 (single process, 4-way TP)
- ✅ DP=2, TP=2 (2 processes, 2-way TP each)
- ✅ DP=4, TP=1 (4 processes, single GPU each)

**Rationale:** Multi-process implementation now supports all DP configurations.

## Configuration Order

### Sorting Logic

```python
configs.sort(key=lambda x: (x[2], -x[0]))
#                           ^^^^  ^^^^
#                           DP↑   TP↓
```

**Priority:**
1. **DP ascending** (1, 2, 4, 8, ...)
2. **TP descending** (8, 4, 2, 1)

### Examples

**2 GPUs:**
1. DP=1, TP=2
2. DP=2, TP=1

**4 GPUs:**
1. DP=1, TP=4
2. DP=2, TP=2
3. DP=4, TP=1

**8 GPUs:**
1. DP=1, TP=8
2. DP=2, TP=4
3. DP=4, TP=2
4. DP=8, TP=1

### Rationale for Sort Order

**Why DP ascending?**
- Start with simplest configuration (single process)
- Gradually increase process count
- Easier to identify where issues start

**Why TP descending within each DP level?**
- Within same DP, test maximum TP first
- TP configurations usually more stable
- Helps identify TP-specific issues first

## Test Output

```bash
$ python3 -c "
from benchmarks.run_parallel_sweep import generate_parallel_configs
configs = generate_parallel_configs(8)
for tp, pp, dp in configs:
    print(f'DP={dp}, TP={tp}, PP={pp}')
"

DP=1, TP=8, PP=1
DP=2, TP=4, PP=1
DP=4, TP=2, PP=1
DP=8, TP=1, PP=1
```

✅ All 4 valid combinations generated  
✅ Correctly sorted by DP, then TP

## Files Modified

### 1. benchmarks/run_parallel_sweep.py

**Function:** `generate_parallel_configs()`

**Changes:**
- Removed `if tp >= dp:` constraint
- Updated comment to reflect multi-process support
- Changed sort key from `configs.sort()` to `configs.sort(key=lambda x: (x[2], -x[0]))`

### 2. benchmarks/PARALLEL_SWEEP_GUIDE.md

**Sections Updated:**
- Overview: Removed TP >= DP mention
- Quick Start: Updated configuration examples
- Example sections: Show all combinations for 4 and 8 GPUs
- Filtering section: Added example filter code

**Key Changes:**
```diff
- with the constraint that **TP >= DP**.
+ **all TP×DP combinations** are now supported.

- TP=4, PP=1, DP=1
- TP=2, PP=1, DP=2
- (TP=1, PP=1, DP=4 excluded)
+ DP=1, TP=4 (single process, 4-way TP)
+ DP=2, TP=2 (2 processes, 2-way TP each)  
+ DP=4, TP=1 (4 processes, single GPU each)
```

## Use Cases

### Compare TP vs Multi-Instance Performance

```bash
# Test all combinations to find optimal balance
python benchmarks/run_parallel_sweep.py --num-gpus 8 ...

# Results will include:
# - DP=1, TP=8: Maximum tensor parallelism
# - DP=8, TP=1: Maximum multi-instance parallelism
# - DP=2, TP=4, DP=4, TP=2: Hybrid approaches
```

### Identify Sweet Spot

Different workloads may benefit from different configurations:
- **Large models**: May need high TP to fit in memory
- **Small models**: May benefit from high DP for throughput
- **Medium models**: Hybrid DP×TP may be optimal

Now you can test **all** configurations to find the best one.

## Custom Filtering

If you want to test specific configurations, you can add filters:

### Example 1: Only Single-Process Configurations

```python
def generate_parallel_configs(num_gpus: int) -> List[Tuple[int, int, int]]:
    configs = []
    pp = 1
    tp_values = get_divisors(num_gpus)
    
    for tp in tp_values:
        dp = num_gpus // tp
        # Filter: only DP=1
        if dp == 1:
            configs.append((tp, pp, dp))
    
    return configs
```

### Example 2: Only Multi-Process Configurations

```python
def generate_parallel_configs(num_gpus: int) -> List[Tuple[int, int, int]]:
    configs = []
    pp = 1
    tp_values = get_divisors(num_gpus)
    
    for tp in tp_values:
        dp = num_gpus // tp
        # Filter: only DP>1
        if dp > 1:
            configs.append((tp, pp, dp))
    
    configs.sort(key=lambda x: (x[2], -x[0]))
    return configs
```

### Example 3: Balanced Configurations Only

```python
def generate_parallel_configs(num_gpus: int) -> List[Tuple[int, int, int]]:
    configs = []
    pp = 1
    tp_values = get_divisors(num_gpus)
    
    for tp in tp_values:
        dp = num_gpus // tp
        # Filter: only configurations where TP and DP are close
        if abs(tp - dp) <= 2:
            configs.append((tp, pp, dp))
    
    configs.sort(key=lambda x: (x[2], -x[0]))
    return configs
```

## Backward Compatibility

✅ **Fully backward compatible**

- Old sweeps only tested TP >= DP configs
- New sweeps test all configs
- Results from old sweeps are still valid
- Can compare new results with old results for TP >= DP configs

## Performance Expectations

### DP=1 Configurations (Single Process)
- Most mature code path
- Lowest memory overhead
- Best for large models that need TP

### DP>1 Configurations (Multi-Process)
- Independent LLM instances
- Higher memory overhead (multiple KV caches)
- Best for throughput with smaller models
- No NCCL communication overhead (simulated DP)

### Comparison

For 8 GPUs:
- **DP=1, TP=8**: Lowest latency per request (if model is very large)
- **DP=8, TP=1**: Highest throughput (if model fits in single GPU)
- **DP=2, TP=4** or **DP=4, TP=2**: Balanced approaches

## Documentation Updates

### Files Created
1. `CONFIG_GENERATION_UPDATE.md` (this file)
2. `REALTIME_OUTPUT_FIX.md` (subprocess output fix)

### Files Updated
1. `run_parallel_sweep.py`
2. `PARALLEL_SWEEP_GUIDE.md`

### Related Documentation
- `DP_SUPPORT_GUIDE.md`: Multi-process DP details
- `PER_RANK_METRICS_GUIDE.md`: Per-rank performance metrics
- `MULTI_PROCESS_FIXES.md`: All multi-process fixes

## Testing

### Verify Configuration Generation

```bash
cd /mnt/disk1/ljm/vllm
python3 -c "
import sys
sys.path.insert(0, 'benchmarks')
from run_parallel_sweep import generate_parallel_configs

for num_gpus in [2, 4, 8]:
    configs = generate_parallel_configs(num_gpus)
    print(f'{num_gpus} GPUs: {len(configs)} configurations')
    for tp, pp, dp in configs:
        print(f'  DP={dp}, TP={tp}, PP={pp}')
    print()
"
```

### Run Small Sweep

```bash
# Test with 2 GPUs (quick test)
CUDA_VISIBLE_DEVICES=0,1 python benchmarks/run_parallel_sweep.py \
    --num-gpus 2 \
    --base-config benchmarks/operator_configs/sweep_test_config.py \
    --output-dir benchmark_results/test_2gpu_all_configs

# Should test both:
# - DP=1, TP=2
# - DP=2, TP=1
```

## Status

✅ **COMPLETE AND TESTED**

- Configuration generation updated
- Sorting logic implemented
- Documentation updated
- Unit tests passing
- Ready for full GPU testing

## Summary

With multi-process DP support now fully implemented:

✅ **All TP×DP combinations supported**  
✅ **Automatic single/multi-process dispatch**  
✅ **Optimal configuration discovery**  
✅ **Better performance insights**  

No more artificial constraints - test everything and find the true optimal configuration for your workload!


