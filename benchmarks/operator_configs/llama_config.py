"""Example configuration for benchmarking Llama models."""

from vllm.profiler.benchmark_config import BenchmarkConfig

# Configuration for benchmarking a Llama model
config = BenchmarkConfig(
    # Model configuration
    model_path="meta-llama/Llama-2-7b-hf",  # Change to your model path
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    max_model_len=2048,
    
    # Benchmark configuration
    batch_sizes=[1, 4, 8, 16],
    seq_lengths=[512, 1024, 2048],
    warmup_steps=5,
    benchmark_steps=20,
    
    # Operator selection
    operators_to_benchmark=[
        "attention",
        "linear",
        "layernorm",
        "mlp",
        "activation",
    ],
    
    # Output configuration
    output_file="llama_benchmark_results.json",
    include_per_layer_stats=True,
    include_summary_stats=True,
    verbose=False,
    
    # Advanced options
    enable_cuda_graph=True,
    dtype="auto",
    gpu_memory_utilization=0.9,
)

