# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example configuration for E2E metrics benchmarking.

This configuration enables end-to-end metrics collection using trace data.

命令行示例：
# 运行 E2E metrics benchmark
python benchmarks/benchmark_e2e_metrics.py \
    --config benchmarks/operator_configs/e2e_trace_config.py \
    --phase 1
"""

from vllm.profiler import BenchmarkConfig

config = BenchmarkConfig(
    model_path="/mnt/disk1/ljm/qwen_test/models/Qwen/Qwen3-8B",
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    data_parallel_size=1,
    
    # These parameters are ignored in trace mode (can be omitted or set to any value)
    # batch_sizes=[1],  # Optional in trace mode
    # seq_lengths=[512],  # Optional in trace mode
    
    warmup_steps=5,
    benchmark_steps=0,  # Can be 0 in trace mode (not used)
    
    # Trace mode configuration
    use_trace_data=True,
    trace_file_path="/mnt/disk1/ljm/vllm/qwen-bailian-usagetraces-anon/qwen_traceB_test_2min20.jsonl",
    trace_time_range_minutes=(0, 2),  # First 2 minutes (adjust as needed)
    trace_hash_id_seed=42,
    trace_realtime_replay=True,  # Replay with real-time intervals
    
    # Enable E2E metrics collection (requires use_trace_data=True)
    enable_e2e_metrics=True,
    
    output_file="e2e_trace_results_accurate.json",
    include_per_layer_stats=True,
    include_summary_stats=True,
    verbose=True,
    
    gpu_memory_utilization=0.5,
    dtype="auto",
    enable_cuda_graph=False,  # Disable for more accurate profiling
)

