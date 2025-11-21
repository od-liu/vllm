# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example configuration for trace-based operator benchmarking.
命令行：
CUDA_VISIBLE_DEVICES=7 python benchmarks/benchmark_operators.py --config benchmarks/operator_configs/trace_config.py

"""

from vllm.profiler import BenchmarkConfig

config = BenchmarkConfig(
    model_path="/mnt/disk1/ljm/qwen_test/models/Qwen/Qwen3-8B",
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    
    # These parameters are ignored in trace mode (can be omitted or set to any value)
    # batch_sizes=[1],  # Optional in trace mode
    # seq_lengths=[512],  # Optional in trace mode
    
    warmup_steps=5,
    benchmark_steps=0,  # Can be 0 in trace mode (not used)
    
    operators_to_benchmark=["all"],
    
    # Trace mode configuration
    use_trace_data=True,
    trace_file_path="/mnt/disk1/ljm/vllm/qwen-bailian-usagetraces-anon/qwen_traceB_blksz_16.jsonl",
    trace_time_range_minutes=(0, 2),  # First 30 minutes
    trace_hash_id_seed=42,
    trace_realtime_replay=True,  # Replay with real-time intervals
    
    output_file="qwen_bailian_2min_results.json",
    include_per_layer_stats=True,
    include_summary_stats=True,
    verbose=True,
    
    gpu_memory_utilization=0.7,
    dtype="auto",
    enable_cuda_graph=False,  # Disable for more accurate profiling
)

