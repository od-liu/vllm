# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Configuration dataclasses for operator benchmarking."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BenchmarkConfig:
    """Configuration for operator benchmarking.
    
    This configuration allows you to control model loading, benchmark parameters,
    operator selection, and output settings.
    """
    
    # Model configuration
    model_path: str
    """Path to the model to benchmark."""
    
    tensor_parallel_size: int = 1
    """Number of GPUs for tensor parallelism."""
    
    pipeline_parallel_size: int = 1
    """Number of GPUs for pipeline parallelism."""
    
    max_model_len: Optional[int] = None
    """Maximum model context length."""
    
    # Benchmark configuration
    batch_sizes: List[int] = field(default_factory=lambda: [1, 4, 8])
    """List of batch sizes to benchmark."""
    
    seq_lengths: List[int] = field(default_factory=lambda: [512, 1024])
    """List of sequence lengths to benchmark."""
    
    warmup_steps: int = 5
    """Number of warmup iterations before collecting statistics."""
    
    benchmark_steps: int = 20
    """Number of benchmark iterations to collect statistics."""
    
    # Operator selection
    operators_to_benchmark: List[str] = field(
        default_factory=lambda: ["attention", "linear", "layernorm", "mlp", "activation"]
    )
    """List of operator types to benchmark. 
    
    Supported types:
    - "attention": Attention layers
    - "linear": Linear/fully-connected layers
    - "layernorm": Layer normalization (RMSNorm, LayerNorm)
    - "mlp": MLP/FFN layers
    - "activation": Activation functions (SiLU, GELU, etc.)
    - "embedding": Embedding layers
    - "moe": Mixture of Experts layers
    - "mamba": Mamba/SSM layers
    - "all": Benchmark all supported operators
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
    
    def validate(self) -> None:
        """Validate configuration parameters."""
        if not self.model_path:
            raise ValueError("model_path must be specified")
        
        if self.tensor_parallel_size < 1:
            raise ValueError("tensor_parallel_size must be >= 1")
        
        if self.pipeline_parallel_size < 1:
            raise ValueError("pipeline_parallel_size must be >= 1")
        
        if self.warmup_steps < 0:
            raise ValueError("warmup_steps must be >= 0")
        
        if self.benchmark_steps < 1:
            raise ValueError("benchmark_steps must be >= 1")
        
        if not self.batch_sizes:
            raise ValueError("batch_sizes must not be empty")
        
        if not self.seq_lengths:
            raise ValueError("seq_lengths must not be empty")
        
        if not self.operators_to_benchmark:
            raise ValueError("operators_to_benchmark must not be empty")
        
        if self.gpu_memory_utilization <= 0 or self.gpu_memory_utilization > 1:
            raise ValueError("gpu_memory_utilization must be in (0, 1]")
    
    def to_dict(self):
        """Convert config to dictionary."""
        return {
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
        }

