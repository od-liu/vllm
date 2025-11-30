#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Benchmark script for measuring end-to-end (E2E) performance metrics in vLLM.

This script runs inference with vLLM while collecting E2E performance metrics
such as TTFT, TPOT, end-to-end latency, and throughput.

Usage:
    python benchmark_e2e_metrics.py --config my_config.py --phase 1
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
from multiprocessing import Process
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

# Add vllm to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent))

from vllm import LLM, SamplingParams
from vllm.logger import init_logger
from vllm.profiler import BenchmarkConfig
from vllm.profiler.e2e_metrics import E2EMetricsCollector
from vllm.profiler.hash_id_mapper import HashIDMapper
from vllm.profiler.trace_loader import TraceLoader
from vllm.profiler.trace_scheduler import TraceScheduler

logger = init_logger(__name__)


def run_e2e_benchmark_single_rank(
    config: BenchmarkConfig,
    dp_rank: int = 0,
    dp_size: int = 1,
) -> Dict[str, Any]:
    """
    Run E2E benchmark for a single DP rank.
    
    Args:
        config: Benchmark configuration
        dp_rank: Data parallel rank (0-indexed)
        dp_size: Total number of data parallel ranks
        
    Returns:
        Dictionary containing E2E benchmark results
    """
    # Configure logger with rank prefix
    rank_prefix = f"[DP{dp_rank}]"
    
    # Create custom formatter with rank
    class RankFormatter(logging.Formatter):
        def __init__(self, rank_prefix, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.rank_prefix = rank_prefix
        
        def format(self, record):
            original = super().format(record)
            return f"{self.rank_prefix} {original}"
    
    # Apply formatter to all handlers
    vllm_logger = logging.getLogger("vllm")
    for handler in vllm_logger.handlers:
        handler.setFormatter(RankFormatter(
            rank_prefix,
            fmt='%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s',
            datefmt='%m-%d %H:%M:%S'
        ))
    
    config.validate()
    
    logger.info(f"Starting E2E benchmark for DP rank {dp_rank}/{dp_size}")
    
    # Log parallel execution mode
    if dp_size > 1:
        logger.info(f"Running in multi-process mode: process {dp_rank}/{dp_size}")
        logger.info("Note: Each process runs independently without DP communication")
    
    logger.info("=" * 80)
    logger.info("Starting E2E Metrics Benchmark")
    logger.info("=" * 80)
    logger.info(f"Model: {config.model_path}")
    logger.info(f"Tensor Parallel Size: {config.tensor_parallel_size}")
    logger.info(f"Pipeline Parallel Size: {config.pipeline_parallel_size}")
    
    if not config.use_trace_data:
        raise ValueError("E2E benchmark requires trace data. Set use_trace_data=True")
    
    logger.info("Mode: Trace-based E2E benchmarking")
    logger.info(f"Trace File: {config.trace_file_path}")
    if config.trace_time_range_minutes:
        logger.info(f"Time Range: {config.trace_time_range_minutes[0]}-{config.trace_time_range_minutes[1]} minutes")
    logger.info(f"Realtime Replay: {config.trace_realtime_replay}")
    logger.info(f"Warmup Steps: {config.warmup_steps}")
    logger.info("=" * 80)
    
    # Create temporary file for results
    config_subdir = f"tp{config.tensor_parallel_size}_dp{dp_size}"
    temp_dir = Path(f"./tmp_benchmark/{config_subdir}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Use process-specific prefix to avoid conflicts in multi-process mode
    temp_file_prefix = f'e2e_bench_dp{dp_rank}_' if dp_size > 1 else 'e2e_bench_'
    temp_file = tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False, dir=str(temp_dir), prefix=temp_file_prefix
    )
    temp_output_file = temp_file.name
    temp_file.close()
    
    logger.info(f"Results will be saved to: {temp_output_file}")
    logger.info(f"Configuration directory: tmp_benchmark/{config_subdir}/")
    
    try:
        # Initialize vLLM
        if dp_size > 1:
            logger.info(f"Initializing vLLM engine for DP rank {dp_rank}/{dp_size}...")
        else:
            logger.info("Initializing vLLM engine...")
        
        llm_kwargs = {
            "model": config.model_path,
            "tensor_parallel_size": config.tensor_parallel_size,
            "pipeline_parallel_size": config.pipeline_parallel_size,
            "gpu_memory_utilization": config.gpu_memory_utilization,
            "dtype": config.dtype,
            "disable_log_stats": False,  # Enable metrics collection
        }
        
        if config.max_model_len is not None:
            llm_kwargs["max_model_len"] = config.max_model_len
        
        if not config.enable_cuda_graph:
            llm_kwargs["enforce_eager"] = True
        
        logger.info("Metrics collection enabled for E2E metrics")
        
        llm = LLM(**llm_kwargs)
        
        logger.info("vLLM engine initialized successfully")
        
        # Load trace requests
        trace_loader = TraceLoader(
            config.trace_file_path,
            time_range_minutes=config.trace_time_range_minutes,
        )
        trace_requests = trace_loader.load_requests()
        
        if not trace_requests:
            raise ValueError("No trace requests loaded after filtering")
        
        logger.info(f"Loaded {len(trace_requests)} trace requests")
        
        # Shard data for DP rank using round-robin
        if dp_size > 1:
            original_count = len(trace_requests)
            rank_requests = [
                req for i, req in enumerate(trace_requests) 
                if i % dp_size == dp_rank
            ]
            logger.info(
                f"DP rank {dp_rank}/{dp_size} processing {len(rank_requests)}/{original_count} "
                f"requests (round-robin sharding)"
            )
            trace_requests = rank_requests
            
            if not trace_requests:
                raise ValueError(f"DP rank {dp_rank} has no requests after sharding")
        
        # Create hash_id mapper
        vocab_size = llm.llm_engine.model_config.get_vocab_size()
        mapper = HashIDMapper(
            vocab_size=vocab_size,
            seed=config.trace_hash_id_seed,
        )
        logger.info(f"Initialized HashIDMapper with vocab_size={vocab_size}")
        
        # Warmup phase (optional, using simple prompts)
        if config.warmup_steps > 0:
            logger.info(f"\nWarmup phase: running {config.warmup_steps} iterations")
            warmup_prompts = ["Warmup prompt: The quick brown fox jumps over the lazy dog."] * 4
            warmup_params = SamplingParams(temperature=0.0, max_tokens=16, ignore_eos=True)
            for i in range(config.warmup_steps):
                _ = llm.generate(warmup_prompts, warmup_params)
            logger.info("Warmup phase completed")
        
        # E2E Metrics Collection
        logger.info("\n" + "-" * 80)
        logger.info("E2E Metrics Collection")
        logger.info("-" * 80)
        
        # Create scheduler
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=128,  # Default, will be overridden per request
            ignore_eos=True,
        )
        scheduler = TraceScheduler(llm, mapper, sampling_params)
        
        # Run trace requests for E2E metrics
        logger.info("Running trace requests for E2E metrics collection...")
        e2e_start_time = time.perf_counter()
        
        e2e_outputs = asyncio.run(
            scheduler.schedule_requests(
                trace_requests,
                realtime=config.trace_realtime_replay,
            )
        )
        
        e2e_end_time = time.perf_counter()
        e2e_total_time = e2e_end_time - e2e_start_time
        
        logger.info(f"E2E benchmark completed in {e2e_total_time:.2f} seconds")
        logger.info(f"Processed {len(e2e_outputs)} requests")
        
        # Filter out None outputs
        valid_outputs = [out for out in e2e_outputs if out is not None]
        logger.info(f"Valid outputs: {len(valid_outputs)}/{len(e2e_outputs)}")
        
        # Collect E2E metrics
        e2e_collector = E2EMetricsCollector()
        collected_metrics = e2e_collector.add_requests(valid_outputs)
        logger.info(f"Successfully collected metrics from {len(collected_metrics)} requests")
        
        # Log sample metrics for debugging
        if collected_metrics:
            sample = collected_metrics[0]
            logger.info(f"Sample metrics: TTFT={sample.ttft_ms}ms, TPOT={sample.tpot_ms}ms, "
                      f"E2E={sample.e2e_latency_ms}ms, Queued={sample.queued_time_ms}ms")
        else:
            logger.warning("No metrics collected! Check if RequestOutput.metrics is populated.")
        
        e2e_stats = e2e_collector.calculate_statistics()
        e2e_throughput = e2e_collector.calculate_throughput(e2e_total_time)
        
        if e2e_throughput is not None:
            e2e_stats["throughput"] = {"tokens_per_second": e2e_throughput}
        
        e2e_metrics_result = {
            "num_requests": len(e2e_outputs),
            "num_valid_requests": len(valid_outputs),
            "total_time_seconds": e2e_total_time,
            "num_ranks": 1,  # Single rank result
            **e2e_stats,
        }
        
        logger.info("\nE2E Metrics Summary:")
        if e2e_stats.get("ttft_ms", {}).get("mean") is not None:
            logger.info(f"  TTFT: {e2e_stats['ttft_ms']['mean']:.2f} ms (mean)")
        if e2e_stats.get("tpot_ms", {}).get("mean") is not None:
            logger.info(f"  TPOT: {e2e_stats['tpot_ms']['mean']:.2f} ms (mean)")
        if e2e_stats.get("e2e_latency_ms", {}).get("mean") is not None:
            logger.info(f"  E2E Latency: {e2e_stats['e2e_latency_ms']['mean']:.2f} ms (mean)")
        if e2e_throughput is not None:
            logger.info(f"  Throughput: {e2e_throughput:.2f} tokens/sec")
        
        # Compile final results
        final_results = {
            "config": config.to_dict(),
            "metadata": {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "cuda_available": torch.cuda.is_available(),
                "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "dp_rank": dp_rank,
                "dp_size": dp_size,
            },
            "e2e_metrics": e2e_metrics_result,
        }
        
        # Save results to temporary file
        with open(temp_output_file, 'w') as f:
            json.dump(final_results, f, indent=2)
        
        logger.info(f"\nResults saved to: {temp_output_file}")
        logger.info("=" * 80)
        
        return final_results
    
    except Exception as e:
        logger.error(f"E2E benchmark failed: {e}", exc_info=True)
        raise
    finally:
        # Cleanup is handled by caller
        pass


def _run_rank_process(
    config: BenchmarkConfig,
    dp_rank: int,
    dp_size: int,
    output_file: str,
    gpu_indices: str,
):
    """
    Wrapper function to run in each DP rank process.
    
    Args:
        config: Benchmark configuration
        dp_rank: Data parallel rank (for trace sharding only)
        dp_size: Total number of data parallel ranks
        output_file: Path to save rank results
        gpu_indices: Comma-separated GPU indices for this process
    """
    try:
        # Set CUDA_VISIBLE_DEVICES for this specific process
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_indices
        
        logger.info(f"Starting independent process {dp_rank}/{dp_size}...")
        logger.info(f"Using GPUs: {gpu_indices}")
        
        # Run E2E benchmark for this rank
        result = run_e2e_benchmark_single_rank(config, dp_rank, dp_size)
        
        # Save result to file
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Process {dp_rank}/{dp_size} completed successfully")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Process {dp_rank} failed: {e}", exc_info=True)
        sys.exit(1)


def _aggregate_e2e_metrics(rank_results: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate E2E metrics from all ranks.
    
    Args:
        rank_results: List of result dictionaries from all DP ranks
        
    Returns:
        Aggregated E2E metrics dictionary
    """
    import numpy as np
    
    # Collect all E2E metrics from all ranks
    all_metrics = []
    for result in rank_results:
        e2e = result.get('e2e_metrics', {})
        if e2e and e2e.get('num_requests', 0) > 0:
            all_metrics.append(e2e)
    
    if not all_metrics:
        return {
            'num_requests': 0,
            'num_ranks': len(rank_results),
        }
    
    # Aggregate statistics
    total_requests = sum([m['num_requests'] for m in all_metrics])
    total_valid_requests = sum([m.get('num_valid_requests', 0) for m in all_metrics])
    total_time = max([m.get('total_time_seconds', 0) for m in all_metrics])
    
    # Collect all metric values across ranks
    def collect_metric(metric_name):
        values = []
        for m in all_metrics:
            metric_data = m.get(metric_name, {})
            if metric_data:
                # Collect mean, p50, p90, p99 from each rank
                for stat in ['mean', 'p50', 'p90', 'p99']:
                    val = metric_data.get(stat)
                    if val is not None:
                        values.append(val)
        return values
    
    # Aggregate each metric
    aggregated = {
        'num_requests': total_requests,
        'num_valid_requests': total_valid_requests,
        'num_ranks': len(rank_results),
        'total_time_seconds': total_time,
    }
    
    for metric_name in ['ttft_ms', 'tpot_ms', 'e2e_latency_ms', 'queued_time_ms']:
        values = collect_metric(metric_name)
        if values:
            aggregated[metric_name] = {
                'mean': float(np.mean(values)),
                'p50': float(np.percentile(values, 50)),
                'p90': float(np.percentile(values, 90)),
                'p99': float(np.percentile(values, 99)),
            }
        else:
            aggregated[metric_name] = None
    
    # Aggregate throughput
    total_tokens = sum([m.get('total_generation_tokens', 0) for m in all_metrics])
    if total_time > 0:
        aggregated['throughput'] = {
            'tokens_per_second': total_tokens / total_time
        }
    
    aggregated['total_generation_tokens'] = total_tokens
    
    return aggregated


def run_e2e_benchmark_multi_rank(config: BenchmarkConfig) -> Dict[str, Any]:
    """
    Run E2E benchmark with multiple independent processes (simulated DP).
    
    Args:
        config: Benchmark configuration
        
    Returns:
        Aggregated results dictionary
    """
    dp_size = config.data_parallel_size
    
    logger.info(f"Launching {dp_size} independent processes for E2E benchmark")
    logger.info("Each process will handle a shard of the trace data")
    
    # Determine GPU allocation for each process
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cuda_visible:
        available_gpus = cuda_visible.split(',')
    else:
        # If not set, assume GPUs 0 to (TP * DP - 1)
        total_gpus = config.tensor_parallel_size * dp_size
        available_gpus = [str(i) for i in range(total_gpus)]
    
    tp_size = config.tensor_parallel_size
    
    # Verify we have enough GPUs
    required_gpus = tp_size * dp_size
    if len(available_gpus) < required_gpus:
        raise ValueError(
            f"Not enough GPUs available. Required: {required_gpus} "
            f"(TP={tp_size} * DP={dp_size}), Available: {len(available_gpus)}"
        )
    
    logger.info(f"Available GPUs: {','.join(available_gpus)}")
    logger.info(f"Each process will use {tp_size} GPU(s)")
    
    # Create temporary files for each rank's output
    rank_output_files = []
    rank_gpu_assignments = []
    
    for rank in range(dp_size):
        # Allocate GPUs for this rank
        start_idx = rank * tp_size
        end_idx = start_idx + tp_size
        rank_gpus = available_gpus[start_idx:end_idx]
        rank_gpu_str = ','.join(rank_gpus)
        rank_gpu_assignments.append(rank_gpu_str)
        
        temp_file = tempfile.NamedTemporaryFile(
            mode='w', suffix=f'_dp{rank}.json', delete=False
        )
        rank_output_files.append(temp_file.name)
        temp_file.close()
        
        logger.info(f"Process {rank}: GPUs {rank_gpu_str}, output: {temp_file.name}")
    
    # Start processes
    procs = []
    for rank in range(dp_size):
        proc = Process(
            target=_run_rank_process,
            args=(config, rank, dp_size, rank_output_files[rank], rank_gpu_assignments[rank])
        )
        proc.start()
        procs.append(proc)
        logger.info(f"Started process {rank} (PID: {proc.pid}, GPUs: {rank_gpu_assignments[rank]})")
    
    # Wait for all processes
    all_success = True
    timeout = 3600  # 1 hour timeout per rank
    for rank, proc in enumerate(procs):
        logger.info(f"Waiting for DP rank {rank} to complete...")
        proc.join(timeout=timeout)
        if proc.exitcode is None:
            logger.error(f"DP rank {rank} timed out after {timeout} seconds")
            proc.kill()
            all_success = False
        elif proc.exitcode != 0:
            logger.error(f"DP rank {rank} failed with exit code {proc.exitcode}")
            all_success = False
        else:
            logger.info(f"DP rank {rank} completed successfully")
    
    if not all_success:
        # Clean up temp files even on failure
        for file_path in rank_output_files:
            try:
                if os.path.exists(file_path):
                    os.remove(file_path)
            except:
                pass
        raise RuntimeError("One or more DP ranks failed")
    
    # Load all rank results
    logger.info(f"Aggregating results from {dp_size} DP ranks...")
    rank_results = []
    for rank, file_path in enumerate(rank_output_files):
        with open(file_path, 'r') as f:
            result = json.load(f)
            rank_results.append(result)
    
    # Aggregate E2E metrics
    aggregated_e2e_metrics = _aggregate_e2e_metrics(rank_results)
    
    # Per-rank E2E metrics
    per_rank_e2e_metrics = [
        {
            'dp_rank': i,
            'e2e_metrics': r.get('e2e_metrics', {})
        }
        for i, r in enumerate(rank_results)
    ]
    
    # Construct final result
    final_result = {
        'config': rank_results[0]['config'],
        'metadata': {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'data_parallel_size': dp_size,
        },
        'e2e_metrics_aggregated': aggregated_e2e_metrics,
        'e2e_metrics_per_rank': per_rank_e2e_metrics,
    }
    
    # Save to output file
    output_path = Path(config.output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w") as f:
        json.dump(final_result, f, indent=2)
    
    logger.info(f"\nAggregated results saved to: {output_path}")
    
    # Clean up temp files
    for file_path in rank_output_files:
        try:
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up {file_path}: {e}")
    
    return final_result


def run_e2e_benchmark(config: BenchmarkConfig) -> Dict[str, Any]:
    """
    Run the complete E2E benchmark suite.
    Automatically handles multi-process execution if data_parallel_size > 1.
    
    Args:
        config: Benchmark configuration
        
    Returns:
        Dictionary containing E2E benchmark results
    """
    config.validate()
    
    # Check if we need multi-process execution
    if config.data_parallel_size > 1:
        logger.info(f"=" * 80)
        logger.info(f"Multi-process mode: {config.data_parallel_size} independent processes")
        logger.info(f"(Simulated DP for benchmarking - no inter-process communication)")
        logger.info(f"=" * 80)
        return run_e2e_benchmark_multi_rank(config)
    else:
        # Single process mode
        result = run_e2e_benchmark_single_rank(config, dp_rank=0, dp_size=1)
        
        # Save to output file
        output_path = Path(config.output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Format for consistency with multi-rank output
        final_result = {
            'config': result['config'],
            'metadata': result['metadata'],
            'e2e_metrics_aggregated': result['e2e_metrics'],
            'e2e_metrics_per_rank': [
                {
                    'dp_rank': 0,
                    'e2e_metrics': result['e2e_metrics']
                }
            ],
        }
        
        with open(output_path, "w") as f:
            json.dump(final_result, f, indent=2)
        
        logger.info(f"Results saved to: {output_path}")
        
        return final_result


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
    """Main entry point for the E2E benchmark script."""
    parser = argparse.ArgumentParser(
        description="Benchmark vLLM E2E performance metrics",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Config file option
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to Python config file containing BenchmarkConfig",
    )
    
    parser.add_argument(
        "--phase",
        type=int,
        choices=[1],
        required=True,
        help="Benchmark phase (must be 1 for E2E metrics)",
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    
    args = parser.parse_args()
    
    # Verify phase is correct
    if args.phase != 1:
        parser.error("E2E benchmark script only supports --phase 1")
    
    # Load config
    config = load_config_from_file(args.config)
    logger.info(f"Loaded config from {args.config}")
    
    # Verify E2E metrics are enabled
    if not config.enable_e2e_metrics:
        logger.warning("enable_e2e_metrics is False in config, but running E2E benchmark anyway")
        config.enable_e2e_metrics = True
    
    if not config.use_trace_data:
        parser.error("E2E benchmark requires trace data. Set use_trace_data=True in config")
    
    # Run benchmark
    try:
        results = run_e2e_benchmark(config)
        logger.info("\nE2E Benchmark completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"E2E Benchmark failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

