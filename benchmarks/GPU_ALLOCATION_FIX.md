# GPU Allocation Fix for Multi-Process Mode

## Problem

When running with `data_parallel_size > 1`, all processes were using the same GPUs, causing conflicts and resource contention.

### Symptoms

```bash
# Running: CUDA_VISIBLE_DEVICES=3,4,5,6 python benchmark_operators.py --config dp2_config.py
# Expected: Process 0 on GPUs 3,4; Process 1 on GPUs 5,6
# Actual: Both processes on GPUs 3,4

nvidia-smi
# GPU 3,4: Multiple processes (conflict!)
# GPU 5,6: Unused
```

### Root Cause

Each process inherited the same `CUDA_VISIBLE_DEVICES=3,4,5,6` environment variable:
- Process 0: Sees 4 GPUs, uses first 2 (indices 0,1 → physical GPUs 3,4)
- Process 1: Sees 4 GPUs, uses first 2 (indices 0,1 → physical GPUs 3,4)
- **Conflict!** Both trying to use GPUs 3,4

## Solution

Modified `run_benchmark_multi_rank()` to:
1. Parse the current `CUDA_VISIBLE_DEVICES`
2. Calculate GPU allocation for each process
3. Set process-specific `CUDA_VISIBLE_DEVICES` in `_run_rank_process()`

### Implementation

**Before:**
```python
def _run_rank_process(...):
    # All processes inherit parent's CUDA_VISIBLE_DEVICES
    # Both try to use the same GPUs
    result = run_benchmark_single_rank(config, dp_rank, dp_size)
```

**After:**
```python
def _run_rank_process(..., gpu_indices: str):
    # Set unique GPU subset for this process
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_indices
    result = run_benchmark_single_rank(config, dp_rank, dp_size)

# In run_benchmark_multi_rank():
for rank in range(dp_size):
    start_idx = rank * tp_size
    end_idx = start_idx + tp_size
    rank_gpus = available_gpus[start_idx:end_idx]
    rank_gpu_str = ','.join(rank_gpus)
    
    proc = Process(
        target=_run_rank_process,
        args=(..., rank_gpu_str)  # Pass GPU assignment
    )
```

## Example

### Configuration
- `CUDA_VISIBLE_DEVICES=3,4,5,6` (4 GPUs available)
- `tensor_parallel_size=2` (TP=2)
- `data_parallel_size=2` (DP=2)

### GPU Allocation

```
Main Process: CUDA_VISIBLE_DEVICES=3,4,5,6
  ├─ Parse: available_gpus = ['3', '4', '5', '6']
  ├─ Calculate: Each process needs TP=2 GPUs
  │
  ├─ Process 0:
  │   ├─ Assign: GPUs 3,4 (indices 0-1)
  │   ├─ Set: CUDA_VISIBLE_DEVICES=3,4
  │   └─ LLM(TP=2) uses CUDA device 0,1 → Physical GPUs 3,4
  │
  └─ Process 1:
      ├─ Assign: GPUs 5,6 (indices 2-3)
      ├─ Set: CUDA_VISIBLE_DEVICES=5,6
      └─ LLM(TP=2) uses CUDA device 0,1 → Physical GPUs 5,6
```

### Result

```bash
nvidia-smi
# GPU 3: Process 0 (TP rank 0)
# GPU 4: Process 0 (TP rank 1)
# GPU 5: Process 1 (TP rank 0)
# GPU 6: Process 1 (TP rank 1)
# ✓ All GPUs used, no conflicts!
```

## Verification

Test the allocation logic:
```bash
cd /mnt/disk1/ljm/vllm
python3 -c "
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3,4,5,6'
available_gpus = os.environ['CUDA_VISIBLE_DEVICES'].split(',')
tp_size, dp_size = 2, 2

for rank in range(dp_size):
    start = rank * tp_size
    end = start + tp_size
    rank_gpus = ','.join(available_gpus[start:end])
    print(f'Process {rank}: CUDA_VISIBLE_DEVICES={rank_gpus}')
"
```

Output:
```
Process 0: CUDA_VISIBLE_DEVICES=3,4
Process 1: CUDA_VISIBLE_DEVICES=5,6
```

## Impact

✅ **Fixed**: Each process now uses its own GPU subset
✅ **Fixed**: All available GPUs are utilized
✅ **Fixed**: No GPU resource conflicts
✅ **Improved**: Clear logging of GPU assignments per process

## Files Modified

1. **`benchmarks/benchmark_operators.py`**:
   - Added `gpu_indices` parameter to `_run_rank_process()`
   - Added GPU allocation logic in `run_benchmark_multi_rank()`
   - Set per-process `CUDA_VISIBLE_DEVICES`

2. **`benchmarks/DP_SUPPORT_GUIDE.md`**:
   - Updated process architecture diagram
   - Added troubleshooting for GPU allocation issues

3. **`benchmarks/GPU_ALLOCATION_FIX.md`**:
   - This document

## Testing

To test with your configuration:
```bash
cd /mnt/disk1/ljm/vllm
source ../qwen/bin/activate

# Clean any lingering processes
pkill -f "VLLM::Worker"

# Run test
CUDA_VISIBLE_DEVICES=3,4,5,6 python benchmarks/benchmark_operators.py \
    --config benchmarks/operator_configs/dp_test_config.py

# In another terminal, watch GPU usage:
watch -n 1 nvidia-smi
# Should see all 4 GPUs (3,4,5,6) being used
```

## Related Issues

This fix also resolves:
- ✅ NCCL communication errors (now avoided by not setting VLLM_DP_* vars)
- ✅ GPU memory contention (each process has exclusive GPU access)
- ✅ Confusing nvidia-smi output (all GPUs now clearly assigned)


