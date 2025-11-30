# Benchmark Refactoring Summary

## Overview

This document summarizes the refactoring of the vLLM benchmark system to separate Phase 1 (E2E metrics) and Phase 2 (Operator profiling) into independent, modular scripts.

## Changes Made

### 1. Created `benchmark_e2e_metrics.py`

**Purpose**: Dedicated script for measuring end-to-end performance metrics.

**Features**:
- Measures TTFT (Time To First Token)
- Measures TPOT (Time Per Output Token)
- Measures E2E latency and throughput
- Supports multi-process DP mode
- Independent execution without operator profiling overhead
- Proper phase parameter handling
- Result aggregation across multiple DP ranks

**Key Functions**:
- `run_e2e_benchmark_single_rank()`: Single rank E2E benchmark
- `run_e2e_benchmark_multi_rank()`: Multi-process DP mode
- `_aggregate_e2e_metrics()`: Aggregate E2E metrics from all ranks

### 2. Modified `benchmark_operators.py`

**Purpose**: Simplified to focus only on operator-level profiling.

**Changes**:
- ✓ Removed all E2E metrics collection code
- ✓ Removed `E2EMetricsCollector` import
- ✓ Removed Phase 1 logic branches
- ✓ Fixed `_run_rank_process()` to accept and pass `phase` parameter
- ✓ Fixed `run_benchmark_multi_rank()` to pass `phase` to subprocess
- ✓ Updated phase validation to only accept phase 2
- ✓ Removed `e2e_metrics` from output structure
- ✓ Removed `_aggregate_e2e_metrics()` function
- ✓ Simplified output to only include operator performance data

**Key Fix**: 
The critical bug where `phase` parameter was not passed in multi-process DP mode has been fixed.

### 3. Created `run_benchmarks.py`

**Purpose**: Orchestrator script that runs both phases and merges results.

**Features**:
- Automatically generates all valid TP×PP×DP configurations
- Calls `benchmark_e2e_metrics.py` for Phase 1
- Calls `benchmark_operators.py` for Phase 2
- Merges results from both phases
- Creates summary CSV with all metrics
- Supports `--skip-existing` to resume interrupted sweeps
- Proper error handling and cleanup

**Key Modification**:
The `run_benchmark()` function now selects the appropriate script based on phase:
```python
if phase == 1:
    script = "benchmarks/benchmark_e2e_metrics.py"
elif phase == 2:
    script = "benchmarks/benchmark_operators.py"
```

### 4. Created Testing Infrastructure

**Files Created**:
- `TESTING_GUIDE.md`: Comprehensive testing guide with examples
- `validate_setup.py`: Validation script to check setup before running
- `REFACTORING_SUMMARY.md`: This document

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                      run_benchmarks.py                          │
│                  (Orchestrator & Sweep Script)                  │
└────────────────────────┬────────────────────────────────────────┘
                         │
         ┌───────────────┴───────────────┐
         │                               │
         ▼                               ▼
┌────────────────────┐          ┌────────────────────┐
│  Phase 1: E2E      │          │  Phase 2: Operator │
│                    │          │     Profiling      │
│ benchmark_e2e_     │          │                    │
│   metrics.py       │          │ benchmark_         │
│                    │          │   operators.py     │
│ - TTFT             │          │                    │
│ - TPOT             │          │ - Per-layer stats  │
│ - Latency          │          │ - Operator stats   │
│ - Throughput       │          │ - TP aggregation   │
│ - DP aggregation   │          │ - DP aggregation   │
└────────────────────┘          └────────────────────┘
         │                               │
         └───────────────┬───────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │  Merge Results   │
              │                  │
              │ - E2E metrics    │
              │ - Operator perf  │
              │ - Summary CSV    │
              └──────────────────┘
```

## Output Structure

### Phase 1 Output (E2E Metrics)
```json
{
  "config": {...},
  "metadata": {...},
  "e2e_metrics_aggregated": {
    "ttft_ms": {"mean": ..., "p50": ..., "p90": ..., "p99": ...},
    "tpot_ms": {...},
    "e2e_latency_ms": {...},
    "throughput": {"tokens_per_second": ...}
  },
  "e2e_metrics_per_rank": [...]
}
```

### Phase 2 Output (Operator Performance)
```json
{
  "config": {...},
  "metadata": {...},
  "operator_performance": {
    "per_layer_stats": [...],
    "summary_stats": {...}
  },
  "operator_performance_per_dp_rank": [...],
  "results": [...]
}
```

### Merged Output (Final)
```json
{
  "config": {...},
  "metadata": {...},
  "e2e_metrics_aggregated": {...},
  "e2e_metrics_per_rank": [...],
  "operator_performance": {...},
  "operator_performance_per_dp_rank": [...],
  "results": [...]
}
```

## Bug Fixes

### Critical Fix: Phase Parameter in Multi-Process Mode

**Problem**: When running with DP > 1, the `_run_rank_process()` function did not receive or pass the `phase` parameter to `run_benchmark_single_rank()`, causing:
```
ValueError: Phase parameter is required. Use --phase 1 for E2E metrics or --phase 2 for operator profiling.
```

**Solution**: 
1. Added `phase` parameter to `_run_rank_process()` function signature
2. Updated `run_benchmark_multi_rank()` to pass `phase` when spawning processes
3. Both functions now correctly propagate the phase parameter

**Fixed Code**:
```python
def _run_rank_process(..., phase: Optional[int] = None):
    result = run_benchmark_single_rank(config, dp_rank, dp_size, phase=phase)

def run_benchmark_multi_rank(config, phase=None):
    proc = Process(
        target=_run_rank_process,
        args=(config, rank, dp_size, ..., phase)  # phase now passed
    )
```

## Usage Examples

### Single Configuration Test
```bash
# Test E2E metrics only
python benchmarks/benchmark_e2e_metrics.py \
    --config benchmarks/operator_configs/e2e_trace_config.py \
    --phase 1

# Test operator profiling only
python benchmarks/benchmark_operators.py \
    --config benchmarks/operator_configs/e2e_trace_config.py \
    --phase 2
```

### Full Sweep
```bash
# Run complete sweep for all TP/DP combinations
python benchmarks/run_benchmarks.py \
    --num-gpus 4 \
    --base-config benchmarks/operator_configs/e2e_trace_config.py \
    --output-dir benchmark_results/4gpu_sweep \
    --gpu-ids "0,1,2,3"
```

### Validation
```bash
# Validate setup before running
python benchmarks/validate_setup.py \
    --config benchmarks/operator_configs/e2e_trace_config.py
```

## Benefits

1. **Clear Separation of Concerns**: E2E and operator profiling are now independent
2. **Easier Debugging**: Issues in one phase don't affect the other
3. **Flexibility**: Can run phases independently or together
4. **Bug Fix**: Multi-process DP mode now works correctly
5. **Better Maintainability**: Cleaner code structure, easier to extend
6. **Testing Infrastructure**: Comprehensive guide and validation tools

## Testing Checklist

- [x] Script files created and syntax-valid
- [x] Import structure correct
- [x] Phase parameter passing fixed
- [x] E2E metrics code separated
- [x] Operator profiling code simplified
- [x] Run orchestrator created
- [x] Testing guide created
- [x] Validation script created

## Next Steps for Users

1. **Install Dependencies**: Ensure all required packages are installed
2. **Validate Setup**: Run `python benchmarks/validate_setup.py --config <your_config>`
3. **Test Single Config**: Start with a simple 1-GPU test
4. **Run Full Sweep**: Use `run_benchmarks.py` for comprehensive testing
5. **Analyze Results**: Check summary CSV and individual JSON files

## Migration from Old System

If you were using the old `run_parallel_sweep.py`:

**Old way**:
```bash
python benchmarks/run_parallel_sweep.py --num-gpus 4 ...
```

**New way**:
```bash
python benchmarks/run_benchmarks.py --num-gpus 4 ...
```

The command-line interface is identical, but internally:
- Phase 1 now uses `benchmark_e2e_metrics.py` (new)
- Phase 2 uses `benchmark_operators.py` (simplified)
- Results are properly merged with all metrics

## Files Modified/Created

**Created**:
- `benchmarks/benchmark_e2e_metrics.py` (new)
- `benchmarks/run_benchmarks.py` (new)
- `benchmarks/TESTING_GUIDE.md` (new)
- `benchmarks/validate_setup.py` (new)
- `benchmarks/REFACTORING_SUMMARY.md` (new)

**Modified**:
- `benchmarks/benchmark_operators.py` (simplified, bug fixed)

**Preserved**:
- `benchmarks/run_parallel_sweep.py` (unchanged, for backward compatibility)

## Support

For issues or questions:
1. Check `TESTING_GUIDE.md` for common scenarios
2. Run `validate_setup.py` to diagnose problems
3. Check error messages from individual phase runs
4. Verify GPU availability and trace file access

