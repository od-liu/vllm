# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Example E2E Benchmark Configuration Template

This configuration template is designed for E2E metrics benchmarking across
different parallelization configurations (TP/DP sweep).

IMPORTANT: When using with run_e2e_sweep.py, the following parameters will be
automatically overridden for each configuration:
  - tensor_parallel_size (will be set based on TP value)
  - pipeline_parallel_size (always set to 1)
  - data_parallel_size (will be set based on DP value)
  - output_file (will be set based on output_dir and config)

Usage with E2E sweep:
    python benchmarks/e2e_sweep/run_e2e_sweep.py \
        --num-gpus 4 \
        --base-config benchmarks/e2e_sweep/example_configs/e2e_config_template.py \
        --output-dir e2e_results/h100_8gpu_2min_test \
        --gpu-type h100\
        --gpu-ids "0,1,2,3"

Usage for single config testing:
    python benchmarks/benchmark_e2e_metrics.py \
        --config benchmarks/e2e_sweep/example_configs/e2e_config_template.py \
        --phase 1
"""

from vllm.profiler import BenchmarkConfig

config = BenchmarkConfig(
    # ============================================================================
    # MODEL CONFIGURATION
    # ============================================================================
    # Path to your model. Can be:
    #   - Local path: "/path/to/model"
    #   - HuggingFace model: "meta-llama/Llama-2-7b-hf"
    # Note: When using run_e2e_sweep.py, you can override this with --model-path
    model_path="/mnt/disk1/ljm/qwen_test/models/Qwen/Qwen3-8B",
    
    # ============================================================================
    # PARALLELISM CONFIGURATION
    # ============================================================================
    # These will be AUTOMATICALLY OVERRIDDEN by run_e2e_sweep.py
    # Set them here for single-config testing only
    tensor_parallel_size=1,      # Will be set by sweep (e.g., 1, 2, 4, 8)
    pipeline_parallel_size=1,    # Always 1 in sweep (PP not supported in sweep)
    data_parallel_size=1,        # Will be set by sweep (e.g., 1, 2, 4, 8)
    
    # ============================================================================
    # TRACE DATA CONFIGURATION (REQUIRED for E2E benchmarks)
    # ============================================================================
    # Enable trace-based benchmarking
    use_trace_data=True,
    
    # Path to trace file (JSONL format)
    # Format: {"timestamp": 1234.567, "prompt_len": 256, "output_len": 128}
    # Note: Can be overridden with --trace-file when using run_e2e_sweep.py
    trace_file_path="/mnt/disk1/ljm/vllm/qwen-bailian-usagetraces-anon/qwen_trace_5min_sparse.jsonl",
    
    # Time range to use from trace (in minutes from start)
    # Example: (0, 5) = use first 5 minutes of trace data
    # Shorter range = faster benchmark, but less data
    # Recommended: 2-5 minutes for quick comparison, 10+ for production testing
    trace_time_range_minutes=(0, 5),
    
    # Random seed for hash ID generation (for deterministic results)
    trace_hash_id_seed=42,
    
    # Whether to replay requests with real-time intervals
    # True:  Realistic workload with request timing from trace
    # False: Maximum throughput test (no delays between requests)
    trace_realtime_replay=True,
    
    # ============================================================================
    # E2E METRICS CONFIGURATION
    # ============================================================================
    # Enable E2E metrics collection (REQUIRED for E2E benchmarks)
    enable_e2e_metrics=True,
    
    # ============================================================================
    # BENCHMARK SETTINGS
    # ============================================================================
    # Number of warmup iterations before actual benchmark
    # Recommended: 3-5 for stable results
    warmup_steps=5,
    
    # Number of benchmark steps (ignored in trace mode, controlled by trace data)
    benchmark_steps=0,
    
    # Operators to benchmark (not used in E2E-only sweep, but required by config)
    operators_to_benchmark=["all"],
    
    # ============================================================================
    # vLLM ENGINE SETTINGS
    # ============================================================================
    # GPU memory utilization (0.0 to 1.0)
    # 0.5 = use 50% of GPU memory (safe default)
    # 0.9 = use 90% of GPU memory (maximize capacity, may cause OOM)
    # Lower values = more stable, higher values = more capacity
    gpu_memory_utilization=0.5,
    
    # Data type for model weights
    # "auto" = automatically detect from model
    # "float16" = FP16 (faster, less memory)
    # "bfloat16" = BF16 (better for large models)
    dtype="auto",
    
    # Enable CUDA graph optimization
    # False = disable CUDA graphs (more accurate profiling)
    # True = enable CUDA graphs (faster inference, less accurate profiling)
    # Recommended: False for benchmarking, True for production
    enable_cuda_graph=False,
    
    # Maximum model length (optional, auto-detected if not set)
    # Uncomment and set if you want to override model's max length
    # max_model_len=4096,
    
    # ============================================================================
    # OUTPUT CONFIGURATION
    # ============================================================================
    # Output file path (will be overridden by run_e2e_sweep.py)
    output_file="e2e_results.json",
    
    # Include detailed per-layer statistics (not used in E2E-only mode)
    include_per_layer_stats=True,
    
    # Include summary statistics
    include_summary_stats=True,
    
    # Verbose output
    verbose=True,
)

# ============================================================================
# CONFIGURATION NOTES
# ============================================================================
# 
# 1. TRACE FILE REQUIREMENTS:
#    - JSONL format (one JSON object per line)
#    - Each line should have: timestamp, prompt_len, output_len
#    - Timestamps should be sorted in ascending order
#
# 2. CHOOSING PARALLELISM STRATEGY (when using sweep):
#    The sweep will automatically test all TP×DP combinations where TP×DP = num_gpus
#    
#    High TP (e.g., TP=8, DP=1):
#      ✓ Can run larger models that don't fit in one GPU
#      ✓ Lower latency for single requests
#      ✗ Lower overall throughput
#      
#    High DP (e.g., TP=1, DP=8):
#      ✓ Higher overall throughput
#      ✓ Better GPU utilization
#      ✗ Model must fit in single GPU
#      ✗ Higher latency for single requests
#
# 3. MEMORY CONSIDERATIONS:
#    - If you get OOM errors, reduce gpu_memory_utilization
#    - Or use higher TP (splits model across more GPUs)
#    - Or reduce trace_time_range_minutes (less concurrent requests)
#
# 4. PERFORMANCE TUNING:
#    - For faster benchmarks: reduce trace_time_range_minutes
#    - For more stable metrics: increase warmup_steps
#    - For realistic workload: use trace_realtime_replay=True
#    - For maximum throughput: use trace_realtime_replay=False
#
# ============================================================================

