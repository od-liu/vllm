# Testing Guide for Refactored Benchmark System

This guide provides instructions for testing the refactored two-phase benchmark system.

## Architecture Overview

The refactored system separates E2E metrics and operator profiling into two independent scripts:

- **Phase 1 (E2E Metrics)**: `benchmark_e2e_metrics.py` - Measures TTFT, TPOT, latency, throughput
- **Phase 2 (Operator Profiling)**: `benchmark_operators.py` - Measures per-layer operator performance
- **Orchestrator**: `run_benchmarks.py` - Runs both phases and merges results

## Test Scenarios

### 1. Single GPU Configuration (TP=1, DP=1)

This is the simplest configuration for initial testing.

```bash
# Test Phase 1 only (E2E metrics)
python benchmarks/benchmark_e2e_metrics.py \
    --config benchmarks/operator_configs/e2e_trace_config.py \
    --phase 1

# Test Phase 2 only (Operator profiling)
python benchmarks/benchmark_operators.py \
    --config benchmarks/operator_configs/e2e_trace_config.py \
    --phase 2

# Test full sweep (both phases)
python benchmarks/run_benchmarks.py \
    --num-gpus 1 \
    --base-config benchmarks/operator_configs/e2e_trace_config.py \
    --output-dir test_results/1gpu \
    --gpu-ids "0"
```

### 2. Multi-TP Configuration (TP>1, DP=1)

Tests tensor parallelism without data parallelism.

```bash
# For 4 GPUs with TP=4, DP=1
python benchmarks/run_benchmarks.py \
    --num-gpus 4 \
    --base-config benchmarks/operator_configs/e2e_trace_config.py \
    --output-dir test_results/4gpu_tp4 \
    --gpu-ids "0,1,2,3"
```

Expected configurations:
- TP=4, PP=1, DP=1
- TP=2, PP=1, DP=2
- TP=1, PP=1, DP=4

### 3. Multi-DP Configuration (TP=1, DP>1)

Tests data parallelism without tensor parallelism.

```bash
# For 4 GPUs with TP=1, DP=4
# This will run 4 independent processes, each using 1 GPU
python benchmarks/run_benchmarks.py \
    --num-gpus 4 \
    --base-config benchmarks/operator_configs/e2e_trace_config.py \
    --output-dir test_results/4gpu_dp4 \
    --gpu-ids "0,1,2,3"
```

Note: The configuration TP=1, PP=1, DP=4 will be tested along with other TP/DP combinations.

### 4. Mixed Configuration (TP>1, DP>1)

Tests both tensor and data parallelism simultaneously.

```bash
# For 8 GPUs, this will test multiple TP/DP combinations
python benchmarks/run_benchmarks.py \
    --num-gpus 8 \
    --base-config benchmarks/operator_configs/e2e_trace_config.py \
    --output-dir test_results/8gpu_mixed \
    --gpu-ids "0,1,2,3,4,5,6,7"
```

Expected configurations:
- TP=8, PP=1, DP=1
- TP=4, PP=1, DP=2
- TP=2, PP=1, DP=4
- TP=1, PP=1, DP=8

## Validation Steps

### 1. Verify Script Imports and Syntax

```bash
# Test if scripts can be imported without errors
python -c "import sys; sys.path.insert(0, '.'); import benchmarks.benchmark_e2e_metrics; print('✓ E2E script OK')"
python -c "import sys; sys.path.insert(0, '.'); import benchmarks.benchmark_operators; print('✓ Operator script OK')"
python -c "import sys; sys.path.insert(0, '.'); import benchmarks.run_benchmarks; print('✓ Run benchmarks script OK')"
```

### 2. Check Configuration File

Ensure your configuration file has the required fields:

```python
# Required fields for e2e_trace_config.py
config = BenchmarkConfig(
    model_path="...",
    tensor_parallel_size=1,  # Will be overridden by sweep
    pipeline_parallel_size=1,
    data_parallel_size=1,    # Will be overridden by sweep
    
    use_trace_data=True,
    trace_file_path="...",
    trace_time_range_minutes=(0, 2),
    trace_realtime_replay=True,
    
    enable_e2e_metrics=True,  # Required for Phase 1
    
    operators_to_benchmark=["all"],  # For Phase 2
    
    warmup_steps=5,
    output_file="...",
    # ... other fields
)
```

### 3. Verify Output Structure

After running a test, check the output files:

```bash
# Phase 1 output should contain E2E metrics
cat test_results/1gpu/tp1_pp1_dp1_phase1.json | jq '.e2e_metrics_aggregated'

# Phase 2 output should contain operator performance
cat test_results/1gpu/tp1_pp1_dp1_phase2.json | jq '.operator_performance.summary_stats'

# Final merged output should contain both
cat test_results/1gpu/tp1_pp1_dp1_results.json | jq 'keys'
# Expected: ["config", "metadata", "e2e_metrics_aggregated", "e2e_metrics_per_rank", 
#            "operator_performance", "operator_performance_per_dp_rank", "results"]
```

### 4. Check Summary CSV

```bash
# Verify summary CSV contains all metrics
head test_results/1gpu/summary.csv
```

Expected columns:
- config, tp, pp, dp, status, execution_time
- ttft_ms_mean, ttft_ms_p50, ttft_ms_p90, ttft_ms_p99
- tpot_ms_mean, tpot_ms_p50, tpot_ms_p90, tpot_ms_p99
- throughput_tokens_per_sec
- top1_operator, top1_time_ms, top1_percentage
- ... etc

## Troubleshooting

### Phase 1 Fails

If Phase 1 (E2E metrics) fails:

1. Check if `enable_e2e_metrics=True` in config
2. Check if `use_trace_data=True` in config
3. Verify trace file exists and is readable
4. Check GPU memory is sufficient

### Phase 2 Fails

If Phase 2 (Operator profiling) fails:

1. Check if environment variables are set correctly (should be automatic)
2. Verify operator benchmark is enabled in vllm
3. Check tmp_benchmark directory is writable
4. Verify TP rank files are being created

### DP > 1 Fails

If configurations with DP > 1 fail:

1. Check GPU allocation is correct
2. Verify CUDA_VISIBLE_DEVICES is set properly
3. Check that each process has enough GPUs (TP * DP total)
4. Verify trace sharding is working correctly

### Merge Fails

If result merging fails:

1. Check both phase1 and phase2 JSON files exist
2. Verify JSON files are valid (use `jq` to check)
3. Check file permissions

## Performance Expectations

Typical execution times (will vary based on hardware and trace size):

- **Phase 1 (E2E)**: 2-5 minutes per configuration
- **Phase 2 (Operator)**: 2-5 minutes per configuration
- **Total per config**: 4-10 minutes

For a 4-GPU sweep (3 configurations), expect 12-30 minutes total.

## Quick Validation Script

```bash
# Run this to quickly validate the setup
python benchmarks/validate_setup.py --config benchmarks/operator_configs/e2e_trace_config.py
```

(This script is provided in validate_setup.py)

