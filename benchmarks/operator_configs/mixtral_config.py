"""Example configuration for benchmarking Mixtral MoE models."""

from vllm.profiler.benchmark_config import BenchmarkConfig

# Configuration for benchmarking a Mixtral model with MoE layers
config = BenchmarkConfig(
    # Model configuration
    model_path="mistralai/Mixtral-8x7B-v0.1",  # Change to your model path
    tensor_parallel_size=2,  # Use TP for large models
    pipeline_parallel_size=1,
    max_model_len=4096,
    
    # Benchmark configuration
    batch_sizes=[1, 4, 8],
    seq_lengths=[1024, 2048],
    warmup_steps=3,
    benchmark_steps=10,
    
    # Operator selection - include MoE layers
    operators_to_benchmark=[
        "attention",
        "linear",
        "layernorm",
        "moe",  # Benchmark MoE layers
        "activation",
    ],
    
    # Output configuration
    output_file="mixtral_benchmark_results.json",
    include_per_layer_stats=True,
    include_summary_stats=True,
    verbose=True,
    
    # Advanced options
    enable_cuda_graph=True,
    dtype="bfloat16",
    gpu_memory_utilization=0.85,
)

