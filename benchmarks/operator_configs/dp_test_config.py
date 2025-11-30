#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test configuration for Data Parallelism (DP) support.

This config tests DP=2 with a small model and trace to verify:
1. Round-robin trace data sharding across DP ranks
2. Multi-process execution
3. Result aggregation
4. E2E metrics aggregation
"""

from vllm.profiler import BenchmarkConfig

config = BenchmarkConfig(
    # Model configuration
    model_path="/mnt/disk1/ljm/qwen_test/models/Qwen/Qwen3-8B",
    
    # Parallelism configuration - Test DP=2
    tensor_parallel_size=2,
    pipeline_parallel_size=1,
    data_parallel_size=2,  # This will trigger multi-process mode
    
    # Memory configuration
    gpu_memory_utilization=0.5,
    
    # Trace-based benchmarking
    use_trace_data=True,
    trace_file_path="/mnt/disk1/ljm/vllm/qwen-bailian-usagetraces-anon/qwen_traceB_test_2min.jsonl",
    trace_time_range_minutes=(0, 1),  # Use only 1 minute of data for quick test
    trace_realtime_replay=False,  # Run as fast as possible
    
    # Enable E2E metrics
    enable_e2e_metrics=True,
    
    # Warmup
    warmup_steps=2,
    
    # Operator profiling
    operators_to_benchmark=["all"],
    include_per_layer_stats=True,
    include_summary_stats=True,
    
    # Output
    output_file="dp_test_results.json",
    
    # Other settings
    dtype="auto",
    enable_cuda_graph=False,  # Disable CUDA graph for simpler debugging
    verbose=True,
)


