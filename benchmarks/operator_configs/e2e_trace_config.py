# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Example configuration for two-phase benchmarking (E2E metrics + Operator performance).

This configuration enables both end-to-end metrics collection and operator performance
benchmarking using trace data. The benchmark will automatically run in two phases:
1. Phase 1: Collect E2E metrics (TTFT, TPOT, latency, throughput) with operator benchmark disabled
2. Phase 2: Collect operator performance with operator benchmark enabled

Both phases use the same trace data to ensure consistency.

命令行：
python benchmarks/run_parallel_sweep.py --num-gpus 4 --base-config benchmarks/operator_configs/e2e_trace_config.py --output-dir benchmark_results/4gpu_sweep --gpu-ids "3,4,5,6" --gpu-type 'h100'
python benchmarks/run_parallel_sweep.py \
    --num-gpus 4 \
    --base-config benchmarks/operator_configs/e2e_trace_config.py \
    --output-dir benchmark_results/4gpu_sweep \
    --gpu-ids "2,3,4,5"
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
    
    operators_to_benchmark=["all"],
    
    # Trace mode configuration
    use_trace_data=True,
    trace_file_path="/mnt/disk1/ljm/vllm/qwen-bailian-usagetraces-anon/qwen_traceB_test_2min.jsonl",
    trace_time_range_minutes=(0, 2),  # First 2 minutes (adjust as needed)
    trace_hash_id_seed=42,
    trace_realtime_replay=True,  # Replay with real-time intervals
    
    # Enable E2E metrics collection (requires use_trace_data=True)
    enable_e2e_metrics=True,
    
    output_file="e2e_trace_results_accurate.json",
    include_per_layer_stats=True,
    include_summary_stats=True,
    verbose=True,
    
    gpu_memory_utilization=0.7,
    dtype="auto",
    enable_cuda_graph=False,  # Disable for more accurate profiling
)

