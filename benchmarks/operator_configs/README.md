# vLLM Benchmark Configuration Guide

This directory contains configuration files for benchmarking vLLM performance metrics and operator-level profiling.

## Architecture Overview

The benchmark system has been refactored into a **two-phase architecture**:

- **Phase 1 (E2E Metrics)**: Measures end-to-end performance metrics (TTFT, TPOT, latency, throughput)
  - Script: `benchmarks/benchmark_e2e_metrics.py`
  - Output: E2E metrics aggregated across all GPUs/ranks

- **Phase 2 (Operator Profiling)**: Measures per-layer operator performance
  - Script: `benchmarks/benchmark_operators.py`
  - Output: Detailed operator statistics (per-layer and summary)

- **Orchestrator**: Automatically runs both phases and merges results
  - Script: `benchmarks/run_benchmarks.py`
  - Performs parallel configuration sweep (TP×PP×DP combinations)

## Quick Start

### Full Benchmark Sweep (Recommended)

Run a complete benchmark sweep for all valid TP×PP×DP configurations:

```bash
python benchmarks/run_benchmarks.py \
    --num-gpus 4 \
    --base-config benchmarks/operator_configs/e2e_trace_config.py \
    --output-dir benchmark_results/test_run \
    --gpu-ids "2,3,4,5"
```

This will:
1. Generate all valid TP×PP×DP combinations (e.g., TP=4/DP=1, TP=2/DP=2, TP=1/DP=4)
2. Run Phase 1 (E2E metrics) for each configuration
3. Run Phase 2 (Operator profiling) for each configuration
4. Merge results and create a summary CSV

### Individual Phase Execution

#### Phase 1: E2E Metrics Only

Measure TTFT, TPOT, latency, and throughput:

```bash
python benchmarks/benchmark_e2e_metrics.py \
    --config benchmarks/operator_configs/e2e_trace_config.py \
    --phase 1
```

**Output**: JSON file with `e2e_metrics_aggregated` containing:
- `ttft_ms`: Time To First Token (mean, p50, p90, p99)
- `tpot_ms`: Time Per Output Token (mean, p50, p90, p99)
- `e2e_latency_ms`: End-to-end latency
- `queued_time_ms`: Queue wait time
- `throughput`: Tokens per second

#### Phase 2: Operator Profiling Only

Measure per-layer operator performance:

```bash
python benchmarks/benchmark_operators.py \
    --config benchmarks/operator_configs/e2e_trace_config.py \
    --phase 2
```

**Output**: JSON file with `operator_performance` containing:
- `per_layer_stats`: Detailed statistics for each layer
- `summary_stats`: Aggregated statistics by operator type

## Configuration File

### e2e_trace_config.py

The main configuration file for two-phase benchmarking with trace data:

```python
from vllm.profiler import BenchmarkConfig

config = BenchmarkConfig(
    # Model path
    model_path="/path/to/model",
    
    # Parallelism (will be overridden by sweep script)
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    data_parallel_size=1,
    
    # Trace-based benchmarking
    use_trace_data=True,
    trace_file_path="/path/to/trace.jsonl",
    trace_time_range_minutes=(0, 2),  # Time range in minutes
    trace_hash_id_seed=42,
    trace_realtime_replay=True,  # Replay with real-time intervals
    
    # Enable E2E metrics (required for Phase 1)
    enable_e2e_metrics=True,
    
    # Operator profiling (for Phase 2)
    operators_to_benchmark=["all"],  # or ["attention", "linear", "layernorm", "mlp"]
    
    # Warmup
    warmup_steps=5,
    
    # Output settings
    output_file="results.json",  # Will be overridden by sweep script
    include_per_layer_stats=True,
    include_summary_stats=True,
    
    # GPU settings
    gpu_memory_utilization=0.7,
    dtype="auto",
    enable_cuda_graph=False,  # Disable for accurate profiling
)
```

## Command-Line Options

### run_benchmarks.py (Orchestrator)

```bash
python benchmarks/run_benchmarks.py \
    --num-gpus <N>                    # Total number of GPUs
    --base-config <config_file>       # Base configuration file
    --output-dir <directory>          # Output directory for results
    [--gpu-ids "0,1,2,3"]            # Specific GPU IDs (optional)
    [--model-path <path>]             # Override model path
    [--trace-file <path>]             # Override trace file
    [--gpu-type "h100"]               # GPU type for documentation
    [--skip-existing]                 # Skip completed configurations
```

**Example**:
```bash
python benchmarks/run_benchmarks.py \
    --num-gpus 8 \
    --base-config benchmarks/operator_configs/e2e_trace_config.py \
    --output-dir benchmark_results/h100_8gpu_sweep \
    --gpu-ids "0,1,2,3,4,5,6,7" \
    --gpu-type "h100"
```

### benchmark_e2e_metrics.py (Phase 1)

```bash
python benchmarks/benchmark_e2e_metrics.py \
    --config <config_file>            # Configuration file (required)
    --phase 1                          # Must be 1 for E2E metrics
    [--verbose]                        # Enable verbose logging
```

**Example**:
```bash
python benchmarks/benchmark_e2e_metrics.py \
    --config benchmarks/operator_configs/e2e_trace_config.py \
    --phase 1
```

### benchmark_operators.py (Phase 2)

```bash
python benchmarks/benchmark_operators.py \
    --config <config_file>            # Configuration file (required)
    --phase 2                          # Must be 2 for operator profiling
    [--verbose]                        # Enable verbose logging
```

**Example**:
```bash
python benchmarks/benchmark_operators.py \
    --config benchmarks/operator_configs/e2e_trace_config.py \
    --phase 2
```

## Output Structure

### Phase 1 Output (E2E Metrics)

```json
{
  "config": { ... },
  "metadata": { ... },
  "e2e_metrics_aggregated": {
    "ttft_ms": {
      "mean": 45.2,
      "p50": 42.1,
      "p90": 52.3,
      "p99": 58.7
    },
    "tpot_ms": { ... },
    "e2e_latency_ms": { ... },
    "throughput": {
      "tokens_per_second": 1234.5
    }
  },
  "e2e_metrics_per_rank": [ ... ]
}
```

### Phase 2 Output (Operator Performance)

```json
{
  "config": { ... },
  "metadata": { ... },
  "operator_performance": {
    "per_layer_stats": [
      {
        "layer_name": "model.layers.0.self_attn.q_proj",
        "operator_type": "linear",
        "num_calls": 279,
        "avg_time_ms": 0.123,
        "total_time_ms": 34.3,
        "avg_input_shapes": [[57, 4096]],
        "avg_output_shapes": [[57, 4096]]
      },
      ...
    ],
    "summary_stats": {
      "attention": {
        "total_time_ms": 450.2,
        "percentage_of_total": 45.2
      },
      ...
    }
  },
  "operator_performance_per_dp_rank": [ ... ]
}
```

### Merged Output (Final Results)

The orchestrator merges both phases:

```json
{
  "config": { ... },
  "metadata": { ... },
  "e2e_metrics_aggregated": { ... },
  "e2e_metrics_per_rank": [ ... ],
  "operator_performance": { ... },
  "operator_performance_per_dp_rank": [ ... ],
  "results": [ ... ]
}
```

### Summary CSV

The orchestrator also generates `summary.csv` with all configurations:

| config | tp | pp | dp | status | execution_time | ttft_ms_mean | tpot_ms_mean | throughput_tokens_per_sec | top1_operator | ... |
|--------|----|----|----|--------|----------------|--------------|--------------|---------------------------|---------------|-----|
| tp4_pp1_dp1 | 4 | 1 | 1 | success | 245.3 | 45.2 | 12.3 | 1234.5 | attention | ... |

## Supported Operators

- `attention`: Attention layers (PagedAttention, FlashAttention, etc.)
- `linear`: Linear/fully-connected layers (including parallel variants)
- `layernorm`: Normalization layers (RMSNorm, LayerNorm)
- `mlp`: MLP/FFN layers
- `activation`: Activation functions (SiLU, GELU, etc.)
- `embedding`: Embedding layers
- `moe`: Mixture of Experts layers
- `mamba`: Mamba/SSM layers
- `all`: Benchmark all supported operators

## Multi-Process Data Parallelism (DP > 1)

When `data_parallel_size > 1`, the benchmark automatically:

1. **Launches multiple independent processes** (one per DP rank)
2. **Shards trace data** using round-robin distribution
3. **Aggregates results** across all ranks
4. **Writes temporary files** to `tmp_benchmark/tp{TP}_dp{DP}/` for each rank

Each process uses different GPUs:
- DP rank 0: GPUs [0, 1, ..., TP-1]
- DP rank 1: GPUs [TP, TP+1, ..., 2*TP-1]
- etc.

## Tips and Best Practices

1. **Start Small**: Test with a single GPU configuration first
   ```bash
   python benchmarks/run_benchmarks.py --num-gpus 1 ...
   ```

2. **Warmup Steps**: Use at least 5 warmup steps for stable measurements

3. **Trace Data**: Ensure trace file exists and is readable
   - Format: JSONL with request timestamps
   - Time range: Adjust `trace_time_range_minutes` based on your needs

4. **GPU Memory**: Adjust `gpu_memory_utilization` based on model size
   - Small models: 0.9
   - Large models: 0.7-0.8

5. **CUDA Graph**: Disable for accurate profiling (`enable_cuda_graph=False`)

6. **Resume Interrupted Runs**: Use `--skip-existing` to continue from where you left off

7. **Monitor Progress**: Check `tmp_benchmark/` directory for intermediate results

## Troubleshooting

### Phase 1 Fails
- Check if `enable_e2e_metrics=True` in config
- Verify trace file exists and is readable
- Check GPU memory availability

### Phase 2 Fails
- Verify operator benchmark is enabled (should be automatic)
- Check `tmp_benchmark/` directory is writable
- Ensure TP rank files are being created

### DP > 1 Fails
- Verify enough GPUs available (TP × DP)
- Check GPU allocation in logs
- Verify trace sharding is working

## Validation

Before running benchmarks, validate your setup:

```bash
python benchmarks/validate_setup.py \
    --config benchmarks/operator_configs/e2e_trace_config.py
```

This checks:
- Script imports
- Configuration validity
- GPU access
- Directory permissions

## Additional Resources

- **Testing Guide**: `benchmarks/TESTING_GUIDE.md`
- **Refactoring Summary**: `benchmarks/REFACTORING_SUMMARY.md`
- **Operator Benchmark Docs**: `benchmarks/OPERATOR_BENCHMARK.md`
