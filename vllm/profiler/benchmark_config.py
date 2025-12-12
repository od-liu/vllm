# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration dataclasses for benchmarking."""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple


@dataclass
class BenchmarkConfig:
    """Configuration for benchmarking.
    
    This configuration allows you to control model loading, benchmark parameters,
    and output settings. Used for E2E metrics benchmarking.
    """
    
    # Model configuration
    model_path: str
    """Path to the model to benchmark."""
    
    tensor_parallel_size: int = 1
    """Number of GPUs for tensor parallelism."""
    
    pipeline_parallel_size: int = 1
    """Number of GPUs for pipeline parallelism."""
    
    data_parallel_size: int = 1
    """Number of data parallel replicas (for future use)."""
    
    max_model_len: Optional[int] = None
    """Maximum model context length."""
    
    # Benchmark configuration
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8])
    """List of batch sizes to benchmark.
    
    Note: Ignored in trace mode (use_trace_data=True).
    """
    
    seq_lengths: List[int] = field(default_factory=lambda: [512, 1024])
    """List of sequence lengths to benchmark.
    
    Note: Ignored in trace mode (use_trace_data=True).
    """
    
    warmup_steps: int = 5
    """Number of warmup iterations before collecting statistics."""
    
    benchmark_steps: int = 20
    """Number of benchmark iterations to collect statistics.
    
    Note: Can be 0 in trace mode (use_trace_data=True), where it's not used.
    """
    
    # Operator selection (deprecated - no longer used)
    operators_to_benchmark: List[str] = field(
        default_factory=lambda: ["all"]
    )
    """List of operator types to benchmark (deprecated - no longer used).
    
    This field is kept for backward compatibility but is not used by any
    current benchmark scripts. It can be safely ignored or omitted.
    """
    
    # Output configuration
    output_file: str = "benchmark_results.json"
    """Path to output JSON file."""
    
    include_per_layer_stats: bool = True
    """Include detailed per-layer statistics in output."""
    
    include_summary_stats: bool = True
    """Include aggregated summary statistics in output."""
    
    verbose: bool = False
    """Enable verbose logging."""
    
    # Advanced options
    enable_cuda_graph: bool = True
    """Enable CUDA graph optimization."""
    
    dtype: str = "auto"
    """Data type for model weights (auto, float16, bfloat16, float32)."""
    
    gpu_memory_utilization: float = 0.9
    """Fraction of GPU memory to use."""
    
    # Trace mode configuration
    use_trace_data: bool = False
    """Whether to use trace data for benchmarking."""
    
    trace_file_path: Optional[str] = None
    """Path to trace JSONL file."""
    
    trace_time_range_minutes: Optional[Tuple[float, float]] = None
    """Time range in minutes (start_min, end_min) to filter trace requests.
    
    Example: (0, 30) for first 30 minutes of trace.
    If None, uses all requests in trace.
    """
    
    trace_hash_id_seed: int = 42
    """Seed for deterministic hash_id to token_id mapping."""
    
    trace_realtime_replay: bool = True
    """Whether to replay requests with real-time intervals from trace."""
    
    # End-to-end metrics configuration
    enable_e2e_metrics: bool = False
    """Whether to collect end-to-end metrics (TTFT, TPOT, etc.).
    
    When enabled with use_trace_data=True, the benchmark will collect E2E metrics
    from trace-based request replay.
    """
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.model_path:
            raise ValueError("model_path must be specified")
        
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        
        if self.pipeline_parallel_size < 1:
            raise ValueError("pipeline_parallel_size must be >= 1")
        
        if self.data_parallel_size < 1:
            raise ValueError("data_parallel_size must be >= 1")
        
        if self.enable_e2e_metrics and not self.use_trace_data:
            raise ValueError(
                "enable_e2e_metrics=True requires use_trace_data=True. "
                "E2E metrics collection currently only supports trace-based benchmarking."
            )
        
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        
        # operators_to_benchmark is deprecated and no longer validated
        
        if self.gpu_memory_utilization <= 0 or self.gpu_memory_utilization > 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
        
        # Trace mode validation
        if self.use_trace_data:
            if not self.trace_file_path:
                raise ValueError("trace_file_path must be specified when use_trace_data=True")
            if self.trace_time_range_minutes is not None:
                start_min, end_min = self.trace_time_range_minutes
                if start_min < 0 or end_min <= start_min:
                    raise ValueError(
                        f"trace_time_range_minutes must have start_min >= 0 and "
                        f"end_min > start_min, got ({start_min}, {end_min})"
                    )
            # In trace mode, benchmark_steps can be 0 (not used)
            # batch_sizes and seq_lengths are ignored but can be set to any value
        else:
            # In non-trace mode, these parameters are required
            if self.benchmark_steps < 1:
                raise ValueError("benchmark_steps must be >= 1 when use_trace_data=False")
            if not self.batch_sizes:
                raise ValueError("batch_sizes must not be empty when use_trace_data=False")
            if not self.seq_lengths:
                raise ValueError("seq_lengths must not be empty when use_trace_data=False")
    
    def to_dict(self):
        """Convert config to dictionary."""
        result = {
            "model_path": self.model_path,
            "tensor_parallel_size": self.tensor_parallel_size,
            "pipeline_parallel_size": self.pipeline_parallel_size,
            "max_model_len": self.max_model_len,
            "batch_sizes": self.batch_sizes,
            "seq_lengths": self.seq_lengths,
            "warmup_steps": self.warmup_steps,
            "benchmark_steps": self.benchmark_steps,
            "operators_to_benchmark": self.operators_to_benchmark,
            "output_file": self.output_file,
            "include_per_layer_stats": self.include_per_layer_stats,
            "include_summary_stats": self.include_summary_stats,
            "verbose": self.verbose,
            "enable_cuda_graph": self.enable_cuda_graph,
            "dtype": self.dtype,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "use_trace_data": self.use_trace_data,
            "trace_file_path": self.trace_file_path,
            "trace_time_range_minutes": self.trace_time_range_minutes,
            "trace_hash_id_seed": self.trace_hash_id_seed,
            "trace_realtime_replay": self.trace_realtime_replay,
            "data_parallel_size": self.data_parallel_size,
            "enable_e2e_metrics": self.enable_e2e_metrics,
        }
        return result

