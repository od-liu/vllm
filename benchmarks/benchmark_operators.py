#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark script for measuring operator-level performance in vLLM.

This script runs inference with vLLM while collecting detailed performance
metrics for compute-intensive operators like attention, linear layers, etc.

Usage:
    python benchmark_operators.py --model <model_path> --output results.json
    
Or use with a Python config:
    python benchmark_operators.py --config my_config.py
"""

import argparse
import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Add vllm to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

from vllm import LLM, SamplingParams
from vllm.logger import init_logger
from vllm.profiler import (
    BenchmarkConfig,
    enable_operator_benchmark,
    get_operator_benchmark,
)
from vllm.profiler.hash_id_mapper import HashIDMapper
from vllm.profiler.trace_loader import TraceLoader
from vllm.profiler.trace_scheduler import TraceScheduler

logger = init_logger(__name__)


def create_dummy_prompts(
    batch_size: int, seq_length: int, tokenizer_name: Optional[str] = None
) -> List[str]:
    """
    Create dummy prompts for benchmarking.
    
    Args:
        batch_size: Number of prompts to generate
        seq_length: Approximate length of each prompt in tokens
        tokenizer_name: Optional tokenizer to use for accurate token counts
        
    Returns:
        List of prompt strings
    """
    # Create a simple repeating pattern to reach desired length
    # Average English word is ~4-5 characters, so we estimate tokens
    avg_chars_per_token = 4
    target_chars = seq_length * avg_chars_per_token
    
    base_text = "The quick brown fox jumps over the lazy dog. "
    num_repeats = (target_chars // len(base_text)) + 1
    
    prompts = []
    for i in range(batch_size):
        # Add a unique prefix to each prompt
        prompt = f"Prompt {i}: " + (base_text * num_repeats)
        prompts.append(prompt[:target_chars])
    
    return prompts


def run_benchmark_iteration(
    llm: LLM,
    prompts: List[str],
    max_tokens: int = 128,
) -> None:
    """
    Run a single benchmark iteration.
    
    Args:
        llm: The vLLM instance
        prompts: List of prompt strings
        max_tokens: Maximum number of tokens to generate
    """
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic for consistency
        max_tokens=max_tokens,
        ignore_eos=True,  # Generate exactly max_tokens
    )
    
    # Run inference (this will be profiled by the injected hooks)
    _ = llm.generate(prompts, sampling_params)


def run_benchmark(config: BenchmarkConfig) -> Dict[str, Any]:
    """
    Run the complete benchmark suite.
    
    Args:
        config: Benchmark configuration
        
    Returns:
        Dictionary containing benchmark results
    """
    config.validate()
    
    logger.info("=" * 80)
    logger.info("Starting Operator Benchmark")
    logger.info("=" * 80)
    logger.info(f"Model: {config.model_path}")
    logger.info(f"Tensor Parallel Size: {config.tensor_parallel_size}")
    logger.info(f"Pipeline Parallel Size: {config.pipeline_parallel_size}")
    if config.use_trace_data:
        logger.info("Mode: Trace-based benchmarking")
        logger.info(f"Trace File: {config.trace_file_path}")
        if config.trace_time_range_minutes:
            logger.info(f"Time Range: {config.trace_time_range_minutes[0]}-{config.trace_time_range_minutes[1]} minutes")
        logger.info(f"Realtime Replay: {config.trace_realtime_replay}")
    else:
        logger.info("Mode: Synthetic data benchmarking")
        logger.info(f"Batch Sizes: {config.batch_sizes}")
        logger.info(f"Sequence Lengths: {config.seq_lengths}")
        logger.info(f"Benchmark Steps: {config.benchmark_steps}")
    logger.info(f"Warmup Steps: {config.warmup_steps}")
    logger.info(f"Operators: {config.operators_to_benchmark}")
    logger.info("=" * 80)
    
    # Create temporary file for worker process to save results
    temp_file = tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False, dir='/tmp'
    )
    worker_output_file = temp_file.name
    temp_file.close()
    
    logger.info(f"Worker will save results to: {worker_output_file}")
    
    # Set environment variables to enable benchmarking in workers
    os.environ["VLLM_OPERATOR_BENCHMARK_ENABLE"] = "1"
    os.environ["VLLM_OPERATOR_BENCHMARK_OPS"] = ",".join(
        config.operators_to_benchmark
    )
    os.environ["VLLM_OPERATOR_BENCHMARK_WARMUP"] = str(config.warmup_steps)
    os.environ["VLLM_OPERATOR_BENCHMARK_AUTO_SAVE"] = "1"
    os.environ["VLLM_OPERATOR_BENCHMARK_OUTPUT"] = worker_output_file
    
    try:
        # Initialize vLLM
        logger.info("Initializing vLLM engine...")
        llm_kwargs = {
            "model": config.model_path,
            "tensor_parallel_size": config.tensor_parallel_size,
            "pipeline_parallel_size": config.pipeline_parallel_size,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "dtype": config.dtype,
        }
        
        if config.max_model_len is not None:
            llm_kwargs["max_model_len"] = config.max_model_len
        
        if not config.enable_cuda_graph:
            llm_kwargs["enforce_eager"] = True
        
        llm = LLM(**llm_kwargs)
        
        # Enable benchmark in the main process (for coordination)
        enable_operator_benchmark(warmup_steps=config.warmup_steps)
        benchmark = get_operator_benchmark()
        
        logger.info("vLLM engine initialized successfully")
        
        # Reset benchmark state
        benchmark.reset()
        benchmark.enable(warmup_steps=config.warmup_steps)
        
        # Warmup phase: always use simple dummy prompts
        if config.warmup_steps > 0:
            logger.info(f"\nWarmup phase: using simple dummy prompts ({config.warmup_steps} iterations)")
            simple_prompts = create_dummy_prompts(batch_size=4, seq_length=128)
            for i in range(config.warmup_steps):
                run_benchmark_iteration(llm, simple_prompts, max_tokens=16)
                benchmark.step()
            logger.info("Warmup phase completed")
        
        # Benchmark phase
        all_results = []
        
        if config.use_trace_data:
            # Trace-based benchmarking
            logger.info(f"\nBenchmark phase: using trace data from {config.trace_file_path}")
            
            # Load trace requests
            trace_loader = TraceLoader(
                config.trace_file_path,
                time_range_minutes=config.trace_time_range_minutes,
            )
            trace_requests = trace_loader.load_requests()
            
            if not trace_requests:
                raise ValueError("No trace requests loaded after filtering")
            
            logger.info(f"Loaded {len(trace_requests)} trace requests")
            
            # Create hash_id mapper
            vocab_size = llm.llm_engine.model_config.get_vocab_size()
            mapper = HashIDMapper(
                vocab_size=vocab_size,
                seed=config.trace_hash_id_seed,
            )
            logger.info(f"Initialized HashIDMapper with vocab_size={vocab_size}")
            
            # Reset benchmark state for trace phase
            benchmark.reset()
            benchmark.enable(warmup_steps=0)  # No warmup in trace phase
            
            # Create scheduler
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=128,  # Default, will be overridden per request
                ignore_eos=True,
            )
            scheduler = TraceScheduler(llm, mapper, sampling_params)
            
            # Run trace-based benchmark
            logger.info("Starting trace-based benchmark execution...")
            start_time = time.perf_counter()
            
            # Run async scheduler
            outputs = asyncio.run(
                scheduler.schedule_requests(
                    trace_requests,
                    realtime=config.trace_realtime_replay,
                )
            )
            
            end_time = time.perf_counter()
            total_time = end_time - start_time
            
            logger.info(f"Trace benchmark completed in {total_time:.2f} seconds")
            logger.info(f"Processed {len(outputs)} requests")
            
            # Wait for worker to write results
            time.sleep(0.5)
            
            # Collect results from worker process
            results = None
            try:
                if os.path.exists(worker_output_file):
                    with open(worker_output_file, 'r') as f:
                        worker_results = json.load(f)
                    
                    if worker_results.get("per_layer_stats"):
                        logger.info(f"  ✓ Loaded {len(worker_results['per_layer_stats'])} operator stats from worker")
                        results = type('BenchmarkResults', (), {
                            'per_layer_stats': worker_results["per_layer_stats"],
                            'summary_stats': worker_results["summary_stats"],
                            'total_forward_time_ms': worker_results["total_forward_time_ms"],
                        })()
                    else:
                        logger.warning("  Worker results file exists but contains no data")
                else:
                    logger.warning(f"  Worker results file not found: {worker_output_file}")
            except Exception as e:
                logger.warning(f"  Failed to read worker results: {e}")
            
            # Fallback to main process results if worker results are unavailable
            if results is None:
                logger.warning("  Using main process results (fallback)")
                results = benchmark.get_statistics()
            
            # Add configuration metadata
            iteration_result = {
                "mode": "trace",
                "trace_file": config.trace_file_path,
                "trace_time_range_minutes": config.trace_time_range_minutes,
                "num_trace_requests": len(trace_requests),
                "warmup_steps": config.warmup_steps,
                "total_wall_time_seconds": total_time,
                "per_layer_stats": results.per_layer_stats if config.include_per_layer_stats else [],
                "summary_stats": results.summary_stats if config.include_summary_stats else {},
                "total_forward_time_ms": results.total_forward_time_ms,
            }
            
            all_results.append(iteration_result)
        
        else:
            # Original synthetic data benchmarking
            for batch_size in config.batch_sizes:
                for seq_length in config.seq_lengths:
                    logger.info(f"\nBenchmarking batch_size={batch_size}, seq_length={seq_length}")
                    
                    # Reset benchmark state for this iteration
                    benchmark.reset()
                    benchmark.enable(warmup_steps=0)  # Warmup already done
                    
                    # Create prompts
                    prompts = create_dummy_prompts(batch_size, seq_length)
                    
                    # Benchmark phase
                    logger.info(f"  Running {config.benchmark_steps} benchmark iterations...")
                    start_time = time.perf_counter()
                    
                    for i in range(config.benchmark_steps):
                        run_benchmark_iteration(llm, prompts, max_tokens=16)
                        benchmark.step()
                    
                    end_time = time.perf_counter()
                    total_time = end_time - start_time
                    
                    logger.info(f"  Completed in {total_time:.2f} seconds")
                    
                    # Wait for worker to write results
                    time.sleep(0.5)
                    
                    # Collect results from worker process
                    results = None
                    try:
                        if os.path.exists(worker_output_file):
                            with open(worker_output_file, 'r') as f:
                                worker_results = json.load(f)
                            
                            if worker_results.get("per_layer_stats"):
                                logger.info(f"  ✓ Loaded {len(worker_results['per_layer_stats'])} operator stats from worker")
                                results = type('BenchmarkResults', (), {
                                    'per_layer_stats': worker_results["per_layer_stats"],
                                    'summary_stats': worker_results["summary_stats"],
                                    'total_forward_time_ms': worker_results["total_forward_time_ms"],
                                })()
                            else:
                                logger.warning("  Worker results file exists but contains no data")
                        else:
                            logger.warning(f"  Worker results file not found: {worker_output_file}")
                    except Exception as e:
                        logger.warning(f"  Failed to read worker results: {e}")
                    
                    # Fallback to main process results if worker results are unavailable
                    if results is None:
                        logger.warning("  Using main process results (fallback)")
                        results = benchmark.get_statistics()
                    
                    # Add configuration metadata
                    iteration_result = {
                        "mode": "synthetic",
                        "batch_size": batch_size,
                        "seq_length": seq_length,
                        "warmup_steps": config.warmup_steps,
                        "benchmark_steps": config.benchmark_steps,
                        "total_wall_time_seconds": total_time,
                        "per_layer_stats": results.per_layer_stats if config.include_per_layer_stats else [],
                        "summary_stats": results.summary_stats if config.include_summary_stats else {},
                        "total_forward_time_ms": results.total_forward_time_ms,
                    }
                    
                    all_results.append(iteration_result)
        
        # Compile final results
        final_results = {
            "config": config.to_dict(),
            "results": all_results,
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            },
        }
        
        # Save results
        output_path = Path(config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"\nResults saved to: {output_path}")
        
        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("BENCHMARK SUMMARY")
        logger.info("=" * 80)
        
        for result in all_results:
            if result.get('mode') == 'trace':
                logger.info(f"\nTrace-based Benchmark")
                logger.info(f"  Trace requests: {result.get('num_trace_requests', 'N/A')}")
                logger.info(f"  Time range: {result.get('trace_time_range_minutes', 'N/A')} minutes")
            else:
                logger.info(
                    f"\nBatch Size: {result.get('batch_size', 'N/A')}, "
                    f"Seq Length: {result.get('seq_length', 'N/A')}"
                )
            logger.info(f"  Total forward time: {result['total_forward_time_ms']:.2f} ms")
            logger.info(f"  Wall time: {result['total_wall_time_seconds']:.2f} s")
            
            if result['summary_stats']:
                logger.info("  Top operators by time:")
                sorted_ops = sorted(
                    result['summary_stats'].items(),
                    key=lambda x: x[1]['total_time_ms'],
                    reverse=True,
                )[:5]
                
                for op_name, op_stats in sorted_ops:
                    logger.info(
                        f"    {op_name}: {op_stats['total_time_ms']:.2f} ms "
                        f"({op_stats['percentage_of_total']:.1f}%)"
                    )
        
        logger.info("\n" + "=" * 80)
        
        return final_results
    
    finally:
        # Clean up temporary worker output file
        try:
            if os.path.exists(worker_output_file):
                os.unlink(worker_output_file)
                logger.info(f"Cleaned up temporary file: {worker_output_file}")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary file {worker_output_file}: {e}")


def load_config_from_file(config_path: str) -> BenchmarkConfig:
    """
    Load benchmark configuration from a Python file.
    
    Args:
        config_path: Path to Python config file
        
    Returns:
        BenchmarkConfig instance
    """
    import importlib.util
    
    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load config from {config_path}")
    
    config_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(config_module)
    
    if not hasattr(config_module, "config"):
        raise ValueError(
            f"Config file {config_path} must define a 'config' variable "
            "of type BenchmarkConfig"
        )
    
    return config_module.config


def main():
    """Main entry point for the benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM operators",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Config file option
    parser.add_argument(
        "--config",
        type=str,
        help="Path to Python config file containing BenchmarkConfig",
    )
    
    # Individual config options (override config file if both provided)
    parser.add_argument(
        "--model",
        type=str,
        help="Path or name of the model to benchmark",
    )
    parser.add_argument(
        "--tp",
        "--tensor-parallel-size",
        type=int,
        default=1,
        dest="tensor_parallel_size",
        help="Tensor parallel size",
    )
    parser.add_argument(
        "--pp",
        "--pipeline-parallel-size",
        type=int,
        default=1,
        dest="pipeline_parallel_size",
        help="Pipeline parallel size",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default="1,4,8",
        help="Comma-separated list of batch sizes to test",
    )
    parser.add_argument(
        "--seq-lengths",
        type=str,
        default="512,1024",
        help="Comma-separated list of sequence lengths to test",
    )
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--benchmark-steps",
        type=int,
        default=20,
        help="Number of benchmark iterations",
    )
    parser.add_argument(
        "--operators",
        type=str,
        default="attention,linear,layernorm,mlp",
        help="Comma-separated list of operators to benchmark",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model data type",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        help="Maximum model context length",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization fraction",
    )
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="Disable CUDA graph optimization",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    # Trace mode options
    parser.add_argument(
        "--trace-file",
        type=str,
        help="Path to trace JSONL file for trace-based benchmarking",
    )
    parser.add_argument(
        "--trace-time-range",
        type=str,
        help="Time range in minutes, e.g., '0,30' for first 30 minutes",
    )
    parser.add_argument(
        "--trace-seed",
        type=int,
        default=42,
        help="Seed for hash_id to token_id mapping",
    )
    parser.add_argument(
        "--no-trace-realtime",
        action="store_true",
        help="Disable real-time replay (submit all requests immediately)",
    )
    
    args = parser.parse_args()
    
    # Load or create config
    if args.config:
        config = load_config_from_file(args.config)
        logger.info(f"Loaded config from {args.config}")
        
        # Override with command-line arguments if provided
        if args.trace_file:
            config.use_trace_data = True
            config.trace_file_path = args.trace_file
        if args.trace_time_range:
            parts = args.trace_time_range.split(",")
            if len(parts) != 2:
                parser.error("--trace-time-range must be in format 'start,end' (e.g., '0,30')")
            config.trace_time_range_minutes = (float(parts[0].strip()), float(parts[1].strip()))
        if args.trace_seed is not None:
            config.trace_hash_id_seed = args.trace_seed
        if args.no_trace_realtime:
            config.trace_realtime_replay = False
    elif args.model:
        # Create config from command-line arguments
        batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
        seq_lengths = [int(x.strip()) for x in args.seq_lengths.split(",")]
        operators = [x.strip() for x in args.operators.split(",")]
        
        # Determine if trace mode is enabled
        use_trace_data = args.trace_file is not None
        trace_time_range = None
        if args.trace_time_range:
            parts = args.trace_time_range.split(",")
            if len(parts) != 2:
                parser.error("--trace-time-range must be in format 'start,end' (e.g., '0,30')")
            trace_time_range = (float(parts[0].strip()), float(parts[1].strip()))
        
        config = BenchmarkConfig(
            model_path=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            pipeline_parallel_size=args.pipeline_parallel_size,
            batch_sizes=batch_sizes if not use_trace_data else [1],
            seq_lengths=seq_lengths if not use_trace_data else [512],
            warmup_steps=args.warmup_steps,
            benchmark_steps=args.benchmark_steps if not use_trace_data else 0,
            operators_to_benchmark=operators,
            output_file=args.output,
            dtype=args.dtype,
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            enable_cuda_graph=not args.no_cuda_graph,
            verbose=args.verbose,
            use_trace_data=use_trace_data,
            trace_file_path=args.trace_file,
            trace_time_range_minutes=trace_time_range,
            trace_hash_id_seed=args.trace_seed,
            trace_realtime_replay=not args.no_trace_realtime,
        )
    else:
        parser.error("Either --config or --model must be provided")
    
    # Run benchmark
    try:
        results = run_benchmark(config)
        logger.info("\nBenchmark completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

