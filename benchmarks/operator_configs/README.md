# vLLM Operator Benchmark Configurations

This directory contains example configurations for benchmarking vLLM operators.

## Quick Start

Run a benchmark using a config file:

```bash
python benchmarks/benchmark_operators.py --config benchmarks/operator_configs/quick_test_config.py
```

Or specify parameters directly:

```bash
python benchmarks/benchmark_operators.py \
    --model meta-llama/Llama-2-7b-hf \
    --tp 1 \
    --batch-sizes 1,4,8 \
    --seq-lengths 512,1024 \
    --operators attention,linear,layernorm \
    --output results.json
```

## Example Configurations

### quick_test_config.py
Minimal configuration for quick testing and development. Uses a small model (OPT-125M) with reduced iterations.

### llama_config.py
Configuration for benchmarking Llama-style models with comprehensive operator coverage.

### mixtral_config.py
Configuration for benchmarking Mixtral MoE models, including MoE-specific operators.

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

## Creating Custom Configurations

Create a new Python file with a `config` variable:

```python
from vllm.profiler.benchmark_config import BenchmarkConfig

config = BenchmarkConfig(
    model_path="your/model/path",
    tensor_parallel_size=2,
    batch_sizes=[1, 8, 16],
    seq_lengths=[1024, 2048],
    operators_to_benchmark=["attention", "linear"],
    output_file="my_results.json",
)
```

Then run:

```bash
python benchmarks/benchmark_operators.py --config your_config.py
```

## Output Format

Results are saved in JSON format with the following structure:

```json
{
  "config": { ... },
  "results": [
    {
      "batch_size": 8,
      "seq_length": 1024,
      "per_layer_stats": [ ... ],
      "summary_stats": {
        "Attention": {
          "total_time_ms": 450.2,
          "percentage_of_total": 45.2
        },
        ...
      }
    }
  ]
}
```

## Tips

1. **Start small**: Use quick_test_config.py first to verify everything works
2. **Warmup matters**: Use at least 5 warmup steps for stable measurements
3. **Multiple runs**: Run benchmarks multiple times and average results
4. **Monitor GPU**: Watch GPU utilization to ensure operators are being measured correctly
5. **Compare results**: Use the comparison tools in `multiworker_aggregator.py`

