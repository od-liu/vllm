# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test configuration for parallel sweep testing.

This is a minimal configuration for quickly testing the parallel sweep functionality
with a small trace file.

Usage:
    python benchmarks/run_parallel_sweep.py \
        --num-gpus 4 \
        --base-config benchmarks/operator_configs/sweep_test_config.py \
        --output-dir benchmark_results/test_sweep
"""

from vllm.profiler import BenchmarkConfig

config = BenchmarkConfig(
    model_path="/mnt/disk1/ljm/qwen_test/models/Qwen/Qwen3-8B",
    tensor_parallel_size=1,  # Will be overridden by sweep script
    pipeline_parallel_size=1,  # Will be overridden by sweep script
    data_parallel_size=1,  # Will be overridden by sweep script
    
    warmup_steps=2,  # Reduced for faster testing
    benchmark_steps=0,
    
    operators_to_benchmark=["all"],
    
    # Trace mode configuration
    use_trace_data=True,
    trace_file_path="/mnt/disk1/ljm/vllm/qwen-bailian-usagetraces-anon/qwen_trace_short.jsonl",
    trace_time_range_minutes=(0, 1),  # Very short for testing
    trace_hash_id_seed=42,
    trace_realtime_replay=False,  # Disable for faster testing
    
    # Enable E2E metrics
    enable_e2e_metrics=True,
    
    output_file="test_results.json",  # Will be overridden by sweep script
    include_per_layer_stats=True,
    include_summary_stats=True,
    verbose=False,  # Reduce output during sweep
    
    gpu_memory_utilization=0.5,  # Balanced for TP/DP configurations
    dtype="auto",
    enable_cuda_graph=False,
    # For TP configurations: 0.35 × 80 GiB = 28 GiB per GPU
    # Sufficient for model weights + KV cache + overhead
)

