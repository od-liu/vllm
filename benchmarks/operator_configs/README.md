# vLLM Benchmark Configuration Guide

This directory contains configuration files for benchmarking vLLM E2E performance metrics.

## Architecture Overview

The benchmark system measures end-to-end performance metrics:

- **E2E Metrics**: Measures end-to-end performance metrics (TTFT, TPOT, latency, throughput)
  - Script: `benchmarks/benchmark_e2e_metrics.py`
  - Output: E2E metrics aggregated across all GPUs/ranks

## Quick Start

### E2E Metrics Benchmarking

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

## Configuration File

### e2e_trace_config.py

The main configuration file for E2E metrics benchmarking with trace data:

```python
from vllm.profiler import BenchmarkConfig

config = BenchmarkConfig(
    # Model path
    model_path="/path/to/model",
    
    # Parallelism
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    data_parallel_size=1,
    
    # Trace-based benchmarking
    use_trace_data=True,
    trace_file_path="/path/to/trace.jsonl",
    trace_time_range_minutes=(0, 2),  # Time range in minutes
    trace_hash_id_seed=42,
    trace_realtime_replay=True,  # Replay with real-time intervals
    
    # Enable E2E metrics
    enable_e2e_metrics=True,
    
    # Warmup
    warmup_steps=5,
    
    # Output settings
    output_file="results.json",
    
    # GPU settings
    gpu_memory_utilization=0.7,
    dtype="auto",
    enable_cuda_graph=False,  # Disable for accurate profiling
)
```

## Command-Line Options

### benchmark_e2e_metrics.py

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

## Output Structure

### E2E Metrics Output

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

2. **Warmup Steps**: Use at least 5 warmup steps for stable measurements

3. **Trace Data**: Ensure trace file exists and is readable
   - Format: JSONL with request timestamps
   - Time range: Adjust `trace_time_range_minutes` based on your needs

4. **GPU Memory**: Adjust `gpu_memory_utilization` based on model size
   - Small models: 0.9
   - Large models: 0.7-0.8

5. **CUDA Graph**: Disable for accurate profiling (`enable_cuda_graph=False`)

6. **Monitor Progress**: Check `tmp_benchmark/` directory for intermediate results

## Troubleshooting

### E2E Metrics Collection Fails
- Check if `enable_e2e_metrics=True` in config
- Verify trace file exists and is readable
- Check GPU memory availability

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
