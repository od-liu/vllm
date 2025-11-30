"""Quick test configuration for development and testing.

命令行：
CUDA_VISIBLE_DEVICES=3 python benchmarks/benchmark_operators.py --config benchmarks/operator_configs/qwen_config.py

"""

from vllm.profiler.benchmark_config import BenchmarkConfig

# Minimal configuration for quick testing
config = BenchmarkConfig(
    # Model configuration
    model_path="/mnt/disk1/ljm/qwen_test/models/Qwen/Qwen3-8B",  # model path
    tensor_parallel_size=2,
    pipeline_parallel_size=1,
    max_model_len=512,
    
    # Benchmark configuration - minimal for speed
    batch_sizes=[1, 4],
    seq_lengths=[128, 256],
    warmup_steps=2,   # tmux
    benchmark_steps=10,
    
    # Operator selection - test all operators
    operators_to_benchmark=["all"],
    
    # Output configuration
    output_file="test_results_tp2_1.json",
    include_per_layer_stats=True,
    include_summary_stats=True,
    verbose=True,
    
    # Advanced options
    enable_cuda_graph=False,  # Disable for faster testing
    dtype="float16",
    gpu_memory_utilization=0.7,  # Reduced to avoid OOM errors
)

