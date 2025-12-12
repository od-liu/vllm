# E2E Metrics Benchmark Sweep

A simplified benchmarking system for vLLM that focuses exclusively on end-to-end (E2E) performance metrics across different parallelization configurations.

## Overview

This tool automatically:
- Generates all valid TP×DP combinations for your GPU count (PP fixed at 1)
- Runs E2E benchmarks for each configuration
- Collects TTFT, TPOT, throughput, and latency metrics
- Generates comprehensive visualizations
- **No operator-level profiling** (faster than full two-phase benchmark)

## Quick Start

### 1. Prepare Your Configuration

Copy and modify the example config template:

```bash
cp benchmarks/e2e_sweep/example_configs/e2e_config_template.py my_config.py
```

Edit `my_config.py` to set:
- `model_path`: Path to your model
- `trace_file_path`: Path to your trace data (JSONL format)
- `trace_time_range_minutes`: Time range to use from trace
- Other parameters as needed

### 2. Run the Sweep

```bash
python benchmarks/e2e_sweep/run_e2e_sweep.py \
    --num-gpus 8 \
    --base-config my_config.py \
    --output-dir results/my_sweep \
    --gpu-type h100
```

This will:
- Generate all TP×DP configs for 8 GPUs (e.g., TP=8/DP=1, TP=4/DP=2, TP=2/DP=4, TP=1/DP=8)
- Run E2E benchmarks for each config
- Save results to `results/my_sweep/`
- Automatically generate visualizations

### 3. View Results

Results are automatically visualized. Check:
- `results/my_sweep/plots/` - All plots (PNG files)
- `results/my_sweep/e2e_summary.csv` - Summary data (CSV)
- `results/my_sweep/plots/e2e_summary.txt` - Text summary

To regenerate visualizations:

```bash
python benchmarks/e2e_sweep/visualize_e2e_results.py \
    --results-dir results/my_sweep
```

## Configuration Options

### run_e2e_sweep.py

| Option | Description | Example |
|--------|-------------|---------|
| `--num-gpus` | Total GPUs available (required) | `8` |
| `--base-config` | Path to config file (required) | `my_config.py` |
| `--output-dir` | Results directory (required) | `results/h100_8gpu` |
| `--model-path` | Override model path from config | `/path/to/model` |
| `--trace-file` | Override trace file from config | `/path/to/trace.jsonl` |
| `--gpu-ids` | Specific GPUs to use | `"0,1,2,3"` |
| `--gpu-type` | GPU type for documentation | `h100`, `a100` |
| `--skip-existing` | Skip already-completed configs | (flag) |
| `--no-visualize` | Disable auto-visualization | (flag) |

### Example Commands

**Basic 8-GPU sweep:**
```bash
python benchmarks/e2e_sweep/run_e2e_sweep.py \
    --num-gpus 8 \
    --base-config benchmarks/e2e_sweep/example_configs/e2e_config_template.py \
    --output-dir results/h100_8gpu \
    --model-path /models/Qwen3-8B \
    --trace-file /data/trace.jsonl \
    --gpu-type h100
```

**Use specific GPUs (GPUs 2-5):**
```bash
python benchmarks/e2e_sweep/run_e2e_sweep.py \
    --num-gpus 4 \
    --base-config my_config.py \
    --output-dir results/test \
    --gpu-ids "2,3,4,5"
```

**Resume interrupted sweep:**
```bash
python benchmarks/e2e_sweep/run_e2e_sweep.py \
    --num-gpus 8 \
    --base-config my_config.py \
    --output-dir results/h100_8gpu \
    --skip-existing
```

## Generated Configurations

For N GPUs, the script generates all divisor-based TP×DP combinations where:
- TP × DP = N
- PP = 1 (pipeline parallelism not included in sweep)

**Examples:**

| GPUs | Configurations Generated |
|------|-------------------------|
| 4 | TP=1/DP=4, TP=2/DP=2, TP=4/DP=1 |
| 8 | TP=1/DP=8, TP=2/DP=4, TP=4/DP=2, TP=8/DP=1 |
| 16 | TP=1/DP=16, TP=2/DP=8, TP=4/DP=4, TP=8/DP=2, TP=16/DP=1 |

## Metrics Collected

### E2E Performance Metrics

- **TTFT (Time to First Token)**: Latency until first token is generated
  - Mean, Median, P90, P99
- **TPOT (Time per Output Token)**: Average time per generated token
  - Mean, Median, P90, P99
- **E2E Latency**: Total request latency
  - Mean, Median, P90, P99
- **Throughput**: Tokens generated per second
- **Queued Time**: Time requests spend in queue
  - Mean, Median, P90, P99

### Output Files

```
results/
└── my_sweep/
    ├── tp8_pp1_dp1_e2e.json     # Individual config results
    ├── tp4_pp1_dp2_e2e.json
    ├── tp2_pp1_dp4_e2e.json
    ├── tp1_pp1_dp8_e2e.json
    ├── e2e_summary.csv           # Summary table (all configs)
    └── plots/
        ├── e2e_summary.txt       # Text summary
        ├── e2e_comparison.png    # Bar charts: TTFT, TPOT, throughput, latency
        ├── percentiles_distribution.png  # P50/P90/P99 distributions
        ├── scaling_analysis.png  # How metrics scale with GPU count
        ├── efficiency_analysis.png  # Throughput per GPU
        └── dp_vs_tp_*.png        # DP vs TP comparison (if multiple configs)
```

## Visualizations

The visualizer generates multiple plots:

### 1. E2E Comparison (`e2e_comparison.png`)
Bar charts comparing all configurations:
- TTFT (lower is better)
- TPOT (lower is better)
- Throughput (higher is better)
- E2E Latency (lower is better)

### 2. Percentiles Distribution (`percentiles_distribution.png`)
P50/P90/P99 distributions for:
- TTFT
- TPOT

### 3. Scaling Analysis (`scaling_analysis.png`)
Line plots showing how metrics scale with GPU count:
- TTFT vs total GPUs
- TPOT vs total GPUs
- Throughput vs total GPUs

### 4. Efficiency Analysis (`efficiency_analysis.png`)
- Total throughput per config
- Throughput per GPU (efficiency metric)

### 5. DP vs TP Comparison (`dp_vs_tp_*.png`)
For GPU counts with multiple configs (e.g., 8 GPUs: TP=8/DP=1 vs TP=4/DP=2):
- Throughput comparison
- TTFT comparison

## Configuration File Format

Your base config file should use `BenchmarkConfig` from vLLM:

```python
from vllm.profiler import BenchmarkConfig

config = BenchmarkConfig(
    # Model settings
    model_path="/path/to/model",
    
    # Parallelism (will be overridden by sweep)
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    data_parallel_size=1,
    
    # Trace data settings
    use_trace_data=True,
    trace_file_path="/path/to/trace.jsonl",
    trace_time_range_minutes=(0, 5),  # Use first 5 minutes
    trace_hash_id_seed=42,
    trace_realtime_replay=True,  # Replay with real timing
    
    # E2E metrics
    enable_e2e_metrics=True,
    
    # Warmup
    warmup_steps=5,
    
    # vLLM engine settings
    gpu_memory_utilization=0.5,
    dtype="auto",
    enable_cuda_graph=False,  # Disable for accurate profiling
    
    # Output (will be overridden by sweep)
    output_file="results.json",
)
```

### Required Settings

- `use_trace_data=True` - E2E benchmark requires trace data
- `enable_e2e_metrics=True` - Enable E2E metrics collection
- `trace_file_path` - Path to your trace JSONL file

### Trace Data Format

The trace file should be JSONL format with entries like:

```json
{"timestamp": 1234567890.123, "prompt_len": 256, "output_len": 128}
{"timestamp": 1234567891.456, "prompt_len": 512, "output_len": 64}
```

## Comparison with Full Two-Phase Benchmark

| Feature | E2E Sweep (This Tool) | Full Two-Phase |
|---------|----------------------|----------------|
| E2E Metrics (TTFT, TPOT) | ✅ Yes | ✅ Yes |
| Operator Profiling | ❌ No | ✅ Yes |
| Speed | 🚀 Faster | Slower |
| Use Case | Quick perf comparison | Detailed analysis |

**Use E2E Sweep when:**
- You want to quickly compare TP/DP configurations
- You only care about user-facing metrics (TTFT, TPOT, throughput)
- You need to optimize parallelization strategy

**Use Full Two-Phase when:**
- You need operator-level profiling
- You're debugging performance issues
- You want detailed layer-by-layer timing

## Tips and Best Practices

### 1. Choose Trace Time Range Carefully
- Shorter range = faster benchmark
- Use representative workload (e.g., peak traffic period)
- Recommended: 2-5 minutes of trace data

### 2. GPU Memory Settings
- Start with `gpu_memory_utilization=0.5`
- Increase if you have OOM errors (out-of-memory)
- Decrease if you want more headroom

### 3. Realtime vs Non-Realtime Replay
- `trace_realtime_replay=True`: Realistic workload with request timing
- `trace_realtime_replay=False`: Maximum throughput test

### 4. Warmup Steps
- Recommended: 3-5 warmup steps
- Helps stabilize metrics
- First few iterations may have cold-start effects

### 5. Interpreting Results
- **High TP, Low DP** (e.g., TP=8, DP=1):
  - Better for large models that don't fit in one GPU
  - Lower throughput per GPU
- **Low TP, High DP** (e.g., TP=1, DP=8):
  - Better for high-throughput scenarios
  - Higher throughput per GPU
  - Requires model fits in one GPU

## Troubleshooting

### "No trace requests loaded"
- Check `trace_file_path` is correct
- Check `trace_time_range_minutes` has data
- Verify trace file format (JSONL)

### "Out of memory"
- Reduce `gpu_memory_utilization`
- Use higher TP (splits model across more GPUs)
- Reduce `trace_time_range_minutes`

### "Process timed out"
- Reduce trace time range
- Check if model is loading correctly
- Check GPU availability

### Visualization fails
- Install matplotlib: `pip install matplotlib`
- Or use text summary: `plots/e2e_summary.txt`

## Advanced Usage

### Custom GPU Assignment
```bash
# Use GPUs 4-7 for 4-GPU sweep
python benchmarks/e2e_sweep/run_e2e_sweep.py \
    --num-gpus 4 \
    --gpu-ids "4,5,6,7" \
    --base-config my_config.py \
    --output-dir results/gpus_4_7
```

### Override Multiple Config Parameters
```bash
python benchmarks/e2e_sweep/run_e2e_sweep.py \
    --num-gpus 8 \
    --base-config base.py \
    --model-path /new/model/path \
    --trace-file /new/trace.jsonl \
    --output-dir results/new_experiment
```

### Disable Auto-Visualization
```bash
python benchmarks/e2e_sweep/run_e2e_sweep.py \
    --num-gpus 8 \
    --base-config my_config.py \
    --output-dir results/test \
    --no-visualize
```

## Support

For issues or questions:
1. Check this README
2. Check vLLM documentation
3. Check trace file format
4. Review error messages in terminal output

## Related Tools

- `benchmarks/benchmark_e2e_metrics.py` - Single-config E2E benchmark
- `benchmarks/visualize_benchmark_results.py` - Full benchmark visualizer


