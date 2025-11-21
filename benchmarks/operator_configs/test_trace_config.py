# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quick test configuration for trace benchmarking (sparse trace).
CUDA_VISIBLE_DEVICES=2,3 python benchmarks/benchmark_operators.py     --config benchmarks/operator_configs/test_trace_config.py
采用了频率更为稀疏的qwen_traceB_test_2min.jsonl数据集，否则在当前配置下很难运行

"""

from vllm.profiler import BenchmarkConfig

config = BenchmarkConfig(
    model_path="/mnt/disk1/ljm/qwen_test/models/Qwen/Qwen3-8B",
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    
    warmup_steps=2,  # Quick warmup
    benchmark_steps=0,
    
    operators_to_benchmark=["all"],
    
    # Test trace configuration
    use_trace_data=True,
    trace_file_path="/mnt/disk1/ljm/vllm/qwen-bailian-usagetraces-anon/qwen_traceB_test_2min.jsonl",
    trace_time_range_minutes=None,  # Use all requests in the test file
    trace_hash_id_seed=42,
    trace_realtime_replay=True,
    
    output_file="test_trace_results.json",
    include_per_layer_stats=True,
    include_summary_stats=True,
    verbose=True,
    
    gpu_memory_utilization=0.7,
    dtype="auto",
    enable_cuda_graph=False,
)

