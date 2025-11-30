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
    disable_operator_benchmark,
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


def run_benchmark_single_rank(config: BenchmarkConfig, dp_rank: int = 0, dp_size: int = 1, phase: Optional[int] = None) -> Dict[str, Any]:
    """
    Run benchmark for a single DP rank.
    
    Args:
        config: Benchmark configuration
        dp_rank: Data parallel rank (0-indexed)
        dp_size: Total number of data parallel ranks
        phase: For two-phase mode: 1=E2E only, 2=Operator only, None=both phases
        
    Returns:
        Dictionary containing benchmark results
    """
    import logging
    
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
    
    logger.info(f"Starting benchmark for DP rank {dp_rank}/{dp_size}")
    
    # Log parallel execution mode
    if dp_size > 1:
        logger.info(f"Running in multi-process mode: process {dp_rank}/{dp_size}")
        logger.info("Note: Each process runs independently without DP communication")
        logger.info("This allows benchmarking different configurations while avoiding NCCL setup complexity")
    
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
    # Use subdirectory per TP/DP configuration for better organization
    config_subdir = f"tp{config.tensor_parallel_size}_dp{dp_size}"
    temp_dir = Path(f"./tmp_benchmark/{config_subdir}")
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    # Use process-specific prefix to avoid conflicts in multi-process mode
    temp_file_prefix = f'vllm_bench_dp{dp_rank}_' if dp_size > 1 else 'vllm_bench_'
    temp_file = tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False, dir=str(temp_dir), prefix=temp_file_prefix
    )
    worker_output_file = temp_file.name
    temp_file.close()
    
    logger.info(f"Worker will save results to: {worker_output_file}")
    logger.info(f"Configuration directory: tmp_benchmark/{config_subdir}/")
    
    # Determine which phase to run (phase parameter is now required)
    if phase is None:
        raise ValueError(
            "Phase parameter is required. Use --phase 2 for operator profiling. "
            "For E2E metrics, use benchmark_e2e_metrics.py instead."
        )
    
    if phase != 2:
        raise ValueError(
            f"Invalid phase: {phase}. This script only supports phase 2 (operator profiling). "
            "For E2E metrics (phase 1), use benchmark_e2e_metrics.py instead."
        )
    
    logger.info(f"Running Phase 2: Operator Profiling")
    
    # Set environment variables BEFORE LLM init so workers inherit them
    # Enable operator profiling for Phase 2
    os.environ["VLLM_OPERATOR_BENCHMARK_ENABLE"] = "1"
    os.environ["VLLM_OPERATOR_BENCHMARK_OPS"] = ",".join(config.operators_to_benchmark)
    os.environ["VLLM_OPERATOR_BENCHMARK_WARMUP"] = str(config.warmup_steps)
    os.environ["VLLM_OPERATOR_BENCHMARK_AUTO_SAVE"] = "1"
    os.environ["VLLM_OPERATOR_BENCHMARK_OUTPUT"] = worker_output_file
    logger.info("Operator profiling enabled for Phase 2")
    
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
        }
        
        # IMPORTANT: Do NOT pass data_parallel_size to LLM when dp_size > 1
        # DP coordination is handled via environment variables (VLLM_DP_*)
        # Passing data_parallel_size would cause "single-process usage" error
        if dp_size == 1 and config.data_parallel_size > 1:
            # Only pass data_parallel_size if we're in single-process mode
            # This will trigger the error, which is what we want to catch
            llm_kwargs["data_parallel_size"] = config.data_parallel_size
        
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
        operator_performance_data = None  # Store operator performance data separately
        
        # Run operator profiling benchmark
        if config.use_trace_data:
            # Operator profiling benchmark with trace data
            logger.info("\n" + "=" * 80)
            logger.info("OPERATOR PROFILING MODE")
            logger.info("=" * 80)
            
            # Load trace requests once (used in both phases)
            trace_loader = TraceLoader(
                config.trace_file_path,
                time_range_minutes=config.trace_time_range_minutes,
            )
            trace_requests = trace_loader.load_requests()
            
            if not trace_requests:
                raise ValueError("No trace requests loaded after filtering")
            
            logger.info(f"Loaded {len(trace_requests)} trace requests")
            
            # Shard data for DP rank using round-robin
            # This ensures each rank processes requests distributed across the entire time range
            # Example: rank 0 gets indices [0, 2, 4, ...], rank 1 gets [1, 3, 5, ...]
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
            
            # ========== Operator Performance Collection ==========
            logger.info("\n" + "-" * 80)
            logger.info("Operator Performance Collection")
            logger.info("-" * 80)
            benchmark = get_operator_benchmark()
            logger.info(f"Benchmark instance status: enabled={benchmark.is_enabled()}")
            
            # Verify profiling is active
            if not benchmark.is_enabled():
                logger.error("ERROR: Benchmark not enabled in Phase 2!")
            else:
                logger.info("✓ Operator profiling is active")
            
            # Create sampling params and scheduler for operator phase
            sampling_params = SamplingParams(
                temperature=0.0,
                max_tokens=128,  # Default, will be overridden per request
                ignore_eos=True,
            )
            scheduler = TraceScheduler(llm, mapper, sampling_params)
            
            # Run trace requests for operator performance
            logger.info("Running trace requests for operator performance collection...")
            op_start_time = time.perf_counter()
            
            op_outputs = asyncio.run(
                scheduler.schedule_requests(
                    trace_requests,
                    realtime=config.trace_realtime_replay,
                )
            )
            
            op_end_time = time.perf_counter()
            op_total_time = op_end_time - op_start_time
            
            logger.info(f"Phase 2 completed in {op_total_time:.2f} seconds")
            logger.info(f"Processed {len(op_outputs)} requests")
            
            # Wait for worker to write results (increased to 2 seconds for reliability)
            time.sleep(2.0)
            
            # Collect operator performance results from all TP ranks
            # Worker writes to either:
            #   - Base file (worker_output_file) when TP=1
            #   - Per-TP files (worker_output_file_base + _tpN.json) when TP>1
            
            # Use absolute path resolution for robust file finding
            worker_file_path = Path(worker_output_file).resolve()
            worker_file_base = worker_file_path.stem
            worker_file_dir = worker_file_path.parent
            worker_file_suffix = worker_file_path.suffix
            
            # Add detailed diagnostic logging
            logger.info(f"Attempting to read worker files:")
            logger.info(f"  Base file: {worker_file_path} (exists: {worker_file_path.exists()})")
            logger.info(f"  Looking for TP files: {worker_file_base}_tp*.json in {worker_file_dir}")
            
            # List all files in directory for debugging
            try:
                all_files = list(worker_file_dir.glob("*.json"))
                logger.info(f"  Files in {worker_file_dir}: {[f.name for f in all_files]}")
            except Exception as e:
                logger.warning(f"  Failed to list directory: {e}")
            
            all_tp_results = []
            
            # First, try to read the base file (for TP=1 case)
            if worker_file_path.exists():
                max_retries = 3
                for retry in range(max_retries):
                    try:
                        stat = worker_file_path.stat()
                        # Check file size is reasonable (should be >1KB if has data)
                        if stat.st_size > 1000:
                            with open(worker_file_path, 'r') as f:
                                base_results = json.load(f)
                            # Check if file has the expected structure
                            if "per_layer_stats" in base_results:
                                logger.info(f"  ✓ Successfully loaded {worker_file_path.name} ({stat.st_size} bytes)")
                                all_tp_results.append(base_results)
                                break
                        elif retry < max_retries - 1:
                            logger.warning(f"  File {worker_file_path.name} too small ({stat.st_size} bytes), retrying...")
                            time.sleep(1.0)
                        else:
                            logger.warning(f"  File {worker_file_path.name} is {stat.st_size} bytes (may be empty)")
                    except Exception as e:
                        if retry < max_retries - 1:
                            logger.warning(f"  Retry {retry+1}/{max_retries} for base file: {e}")
                            time.sleep(1.0)
                        else:
                            logger.warning(f"  Failed to read base worker results after {max_retries} attempts: {e}")
            
            # If no base file, try per-TP files (for TP>1 case)
            if not all_tp_results:
                tp_rank = 0
                found_any = False
                while True:
                    tp_file = worker_file_dir / f"{worker_file_base}_tp{tp_rank}{worker_file_suffix}"
                    if tp_file.exists():
                        found_any = True
                        max_retries = 3
                        for retry in range(max_retries):
                            try:
                                stat = tp_file.stat()
                                # Check file size is reasonable (should be >100KB for operator data)
                                if stat.st_size > 100000:
                                    with open(tp_file, 'r') as f:
                                        tp_results = json.load(f)
                                    # Check if file has the expected structure
                                    if "per_layer_stats" in tp_results:
                                        logger.info(f"  ✓ Successfully loaded {tp_file.name} ({stat.st_size} bytes)")
                                        all_tp_results.append(tp_results)
                                        break
                                elif retry < max_retries - 1:
                                    logger.warning(f"  File {tp_file.name} too small ({stat.st_size} bytes), retrying...")
                                    time.sleep(1.0)
                                else:
                                    logger.warning(f"  File {tp_file.name} is {stat.st_size} bytes (may be incomplete)")
                            except Exception as e:
                                if retry < max_retries - 1:
                                    logger.warning(f"  Retry {retry+1}/{max_retries} for TP rank {tp_rank}: {e}")
                                    time.sleep(1.0)
                                else:
                                    logger.warning(f"  Failed to read TP rank {tp_rank} after {max_retries} attempts: {e}")
                                    break
                        tp_rank += 1
                    else:
                        if tp_rank == 0 and not found_any:
                            logger.warning(f"  No TP rank files found at {tp_file}")
                        break
            
            # Aggregate results from all TP ranks or fallback
            op_results = None
            if all_tp_results:
                # Aggregate across TP ranks (average per-layer, sum totals)
                import numpy as np
                
                # Collect all per_layer_stats (include even if empty, to detect structure)
                all_per_layer = [r.get("per_layer_stats", {}) for r in all_tp_results if "per_layer_stats" in r]
                
                # CRITICAL FIX: per_layer_stats from worker files is a LIST, not a dict!
                # Filter and check the actual data structure
                non_empty_per_layer = [
                    p for p in all_per_layer 
                    if p and (isinstance(p, dict) or isinstance(p, list))
                ]
                
                # Add diagnostic logging
                logger.info(f"  Total per_layer_stats collected: {len(all_per_layer)}")
                logger.info(f"  Non-empty per_layer_stats: {len(non_empty_per_layer)}")
                if all_per_layer and not non_empty_per_layer:
                    logger.warning("  All per_layer_stats are empty!")
                    for i, p in enumerate(all_per_layer):
                        logger.warning(f"    Rank {i}: type={type(p)}, len={len(p) if isinstance(p, (dict, list)) else 'N/A'}")
                
                if non_empty_per_layer:
                    # Check if data is in list format (from worker) or dict format (from main process)
                    first_data = non_empty_per_layer[0]
                    logger.info(f"  First per_layer_stats type: {type(first_data)}")
                    
                    if isinstance(first_data, list):
                        # Worker data format: list of layer records
                        # Convert to the expected dict format for compatibility
                        logger.info(f"  Converting {len(non_empty_per_layer)} list-format results")
                        
                        # Just use the data from first rank (they should be identical for TP)
                        # For now, pass through the list format and convert to dict in aggregation
                        aggregated_per_layer = first_data  # Keep as list
                        
                        aggregated_summary = {}
                        for r in all_tp_results:
                            summary = r.get('summary_stats', {})
                            for op_type, stats in summary.items():
                                if op_type not in aggregated_summary:
                                    aggregated_summary[op_type] = stats.copy()
                                else:
                                    # Aggregate stats across TP ranks
                                    aggregated_summary[op_type]['total_calls'] += stats.get('total_calls', 0)
                                    aggregated_summary[op_type]['total_time_ms'] += stats.get('total_time_ms', 0)
                        
                        # Recalculate averages
                        for op_type in aggregated_summary:
                            if aggregated_summary[op_type]['total_calls'] > 0:
                                aggregated_summary[op_type]['avg_time_per_call_ms'] = (
                                    aggregated_summary[op_type]['total_time_ms'] / 
                                    aggregated_summary[op_type]['total_calls']
                                )
                        
                        total_forward_time = sum([r.get('total_forward_time_ms', 0) for r in all_tp_results])
                        aggregated_summary['num_tp_ranks_aggregated'] = len(all_tp_results)
                        
                        logger.info(f"  ✓ Aggregated {len(aggregated_per_layer)} layer records from {len(all_tp_results)} TP ranks")
                        logger.info(f"  Total forward time: {total_forward_time:.2f}ms")
                        
                        op_results = type('BenchmarkResults', (), {
                            'per_layer_stats': aggregated_per_layer,
                            'summary_stats': aggregated_summary,
                            'total_forward_time_ms': total_forward_time,
                            'tp_rank_results': all_tp_results,  # Keep individual TP rank data
                        })()
                        
                    elif isinstance(first_data, dict):
                        # Main process format: dict of layer names to stats
                        logger.info(f"  Aggregating {len(non_empty_per_layer)} dict-format results")
                        aggregated_per_layer = {}
                        for layer_name in first_data.keys():
                            layer_data = [stats.get(layer_name) for stats in non_empty_per_layer if layer_name in stats and stats[layer_name]]
                            if layer_data:
                                aggregated_per_layer[layer_name] = {
                                    'mean_time_ms': np.mean([d['mean_time_ms'] for d in layer_data]),
                                    'std_time_ms': np.mean([d['std_time_ms'] for d in layer_data]),
                                    'min_time_ms': np.mean([d['min_time_ms'] for d in layer_data]),
                                    'max_time_ms': np.mean([d['max_time_ms'] for d in layer_data]),
                                    'call_count': sum([d['call_count'] for d in layer_data]),
                                }
                        
                        aggregated_summary = {
                            'total_forward_time_ms': sum([r.get('total_forward_time_ms', 0) for r in all_tp_results]),
                            'num_tp_ranks_aggregated': len(all_tp_results),
                        }
                        
                        logger.info(f"  ✓ Aggregated results from {len(all_tp_results)} TP ranks")
                        op_results = type('BenchmarkResults', (), {
                            'per_layer_stats': aggregated_per_layer,
                            'summary_stats': aggregated_summary,
                            'total_forward_time_ms': aggregated_summary['total_forward_time_ms'],
                            'tp_rank_results': all_tp_results,  # Keep individual TP rank data
                        })()
            
            # Add detailed failure diagnostics if no results
            if op_results is None:
                logger.error("Failed to load operator results from worker files!")
                logger.error(f"Expected files pattern: {worker_file_base}_tp*.json or {worker_file_base}.json")
                logger.error(f"Search directory: {worker_file_dir}")
                logger.error(f"Files found: {len(all_tp_results)}")
                
                # Try manual diagnostic check
                logger.error("Diagnostic: Manually checking for TP files...")
                for i in range(8):  # Check up to 8 TP ranks
                    tp_file = worker_file_dir / f"{worker_file_base}_tp{i}.json"
                    if tp_file.exists():
                        logger.error(f"  Found: {tp_file.name} ({tp_file.stat().st_size} bytes)")
                        try:
                            with open(tp_file, 'r') as f:
                                data = json.load(f)
                            logger.error(f"    Keys: {list(data.keys())}")
                            per_layer = data.get('per_layer_stats')
                            logger.error(f"    per_layer_stats type: {type(per_layer)}")
                            if isinstance(per_layer, (dict, list)):
                                logger.error(f"    per_layer_stats len: {len(per_layer)}")
                        except Exception as e:
                            logger.error(f"    Error reading: {e}")
                
                logger.warning("  Using main process results (fallback)")
                op_results = benchmark.get_statistics()
            
            # Store operator performance results separately (will be added to final_results later)
            # Don't add detailed stats to results array to avoid duplication
            iteration_result = {
                "mode": "trace_operator",
                "trace_file": config.trace_file_path,
                "trace_time_range_minutes": config.trace_time_range_minutes,
                "num_trace_requests": len(trace_requests),
                "warmup_steps": config.warmup_steps,
                "total_wall_time_seconds": op_total_time,
                "total_forward_time_ms": op_results.total_forward_time_ms,
            }
            
            all_results.append(iteration_result)
            
            # Store operator performance data separately for final_results
            # This avoids duplication in the output (don't include in results array)
            operator_performance_data = {
                "per_layer_stats": op_results.per_layer_stats if config.include_per_layer_stats else {},
                "summary_stats": op_results.summary_stats if config.include_summary_stats else {},
            }
            
            # Add verification logging
            logger.info(f"Operator performance data prepared:")
            per_layer = operator_performance_data.get('per_layer_stats')
            logger.info(f"  per_layer_stats type: {type(per_layer)}")
            if isinstance(per_layer, dict):
                logger.info(f"  per_layer_stats length (dict): {len(per_layer)}")
            elif isinstance(per_layer, list):
                logger.info(f"  per_layer_stats length (list): {len(per_layer)}")
            else:
                logger.info(f"  per_layer_stats is: {per_layer}")
            logger.info(f"  summary_stats keys: {list(operator_performance_data.get('summary_stats', {}).keys())}")
            
            # Include per-TP-rank data if available
            if hasattr(op_results, 'tp_rank_results') and op_results.tp_rank_results:
                operator_performance_data["per_tp_rank"] = [
                    {
                        "tp_rank": r.get("tp_rank", i),
                        "per_layer_stats": r.get("per_layer_stats", {}) if config.include_per_layer_stats else {},
                        "summary_stats": r.get("summary_stats", {}),
                        "total_forward_time_ms": r.get("total_forward_time_ms", 0),
                    }
                    for i, r in enumerate(op_results.tp_rank_results)
                ]
                logger.info(f"  ✓ Included {len(op_results.tp_rank_results)} per-TP-rank results")
        
        elif config.use_trace_data:
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
                "total_forward_time_ms": results.total_forward_time_ms,
            }
            
            all_results.append(iteration_result)
            
            # Store operator performance data separately to avoid duplication
            operator_performance_data = {
                "per_layer_stats": results.per_layer_stats if config.include_per_layer_stats else [],
                "summary_stats": results.summary_stats if config.include_summary_stats else {},
            }
        
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
        
        # Add operator performance with per-rank structure (consistent format for all DP sizes)
        if operator_performance_data is not None:
            # Two-phase mode: use separately stored data
            final_results["operator_performance"] = operator_performance_data
            final_results["operator_performance_per_dp_rank"] = [
                {
                    "dp_rank": dp_rank,
                    "per_layer_stats": operator_performance_data.get("per_layer_stats", {}),
                    "summary_stats": operator_performance_data.get("summary_stats", {}),
                }
            ]
            
            # Include per-TP-rank data if available
            if "per_tp_rank" in operator_performance_data:
                final_results["operator_performance_per_dp_rank"][0]["per_tp_rank"] = \
                    operator_performance_data["per_tp_rank"]
            
        elif all_results:
            # Single-phase mode: extract from results if available
            # Only add if results contain operator stats (non-trace mode)
            first_result = all_results[0]
            if "per_layer_stats" in first_result or "summary_stats" in first_result:
                operator_perf = {
                    "per_layer_stats": first_result.get("per_layer_stats", []),
                    "summary_stats": first_result.get("summary_stats", {}),
                }
                final_results["operator_performance"] = operator_perf
                final_results["operator_performance_per_dp_rank"] = [
                    {
                        "dp_rank": dp_rank,
                        "per_layer_stats": operator_perf.get("per_layer_stats", []),
                        "summary_stats": operator_perf.get("summary_stats", {}),
                    }
                ]
        
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
        
        # Get summary stats from operator_performance if available (two-phase mode)
        # or from results array (single-phase mode)
        summary_stats_to_display = None
        if operator_performance_data is not None and operator_performance_data.get('summary_stats'):
            summary_stats_to_display = operator_performance_data['summary_stats']
        elif all_results and all_results[0].get('summary_stats'):
            summary_stats_to_display = all_results[0]['summary_stats']
        elif 'operator_performance' in final_results and final_results['operator_performance'].get('summary_stats'):
            summary_stats_to_display = final_results['operator_performance']['summary_stats']
        
        for result in all_results:
            if result.get('mode') in ('trace', 'trace_operator'):
                logger.info(f"\nTrace-based Benchmark")
                logger.info(f"  Trace requests: {result.get('num_trace_requests', 'N/A')}")
                logger.info(f"  Time range: {result.get('trace_time_range_minutes', 'N/A')} minutes")
            else:
                logger.info(
                    f"\nBatch Size: {result.get('batch_size', 'N/A')}, "
                    f"Seq Length: {result.get('seq_length', 'N/A')}"
                )
            logger.info(f"  Total forward time: {result.get('total_forward_time_ms', 0):.2f} ms")
            logger.info(f"  Wall time: {result.get('total_wall_time_seconds', 0):.2f} s")
            
            # Use summary_stats from operator_performance if available, otherwise from result
            stats_to_use = summary_stats_to_display or result.get('summary_stats')
            if stats_to_use:
                logger.info("  Top operators by time:")
                # Filter out non-dict entries and sort by total_time_ms
                sorted_ops = sorted(
                    [(k, v) for k, v in stats_to_use.items() if isinstance(v, dict) and 'total_time_ms' in v],
                    key=lambda x: x[1].get('total_time_ms', 0),
                    reverse=True,
                )[:5]
                
                for op_name, op_stats in sorted_ops:
                    logger.info(
                        f"    {op_name}: {op_stats.get('total_time_ms', 0):.2f} ms "
                        f"({op_stats.get('percentage_of_total', 0):.1f}%)"
                    )
        
        logger.info("\n" + "=" * 80)
        
        return final_results
    
    finally:
        # Clean up temporary worker output files (including all TP rank files)
        try:
            worker_file_base = Path(worker_output_file).stem
            worker_file_dir = Path(worker_output_file).parent
            worker_file_suffix = Path(worker_output_file).suffix
            
            # Clean up all TP rank files
            tp_rank = 0
            cleaned_count = 0
            while True:
                tp_file = worker_file_dir / f"{worker_file_base}_tp{tp_rank}{worker_file_suffix}"
                if tp_file.exists():
                    os.unlink(tp_file)
                    cleaned_count += 1
                    tp_rank += 1
                else:
                    break
            
            if cleaned_count > 0:
                logger.info(f"Cleaned up {cleaned_count} temporary TP rank file(s)")
        except Exception as e:
            logger.warning(f"Failed to clean up temporary files: {e}")


def _run_rank_process(
    config: BenchmarkConfig,
    dp_rank: int,
    dp_size: int,
    dp_master_ip: str,
    dp_master_port: int,
    output_file: str,
    gpu_indices: str,
    phase: Optional[int] = None,
):
    """
    Wrapper function to run in each DP rank process.
    
    Note: We do NOT set VLLM_DP_* environment variables here.
    Each process runs as an independent LLM instance without DP communication.
    This avoids NCCL/TCPStore initialization issues while still allowing
    parallel execution and result aggregation for benchmarking purposes.
    
    Args:
        config: Benchmark configuration
        dp_rank: Data parallel rank (for trace sharding only)
        dp_size: Total number of data parallel ranks
        dp_master_ip: IP address of DP master (unused, kept for API compatibility)
        dp_master_port: Port of DP master (unused, kept for API compatibility)
        output_file: Path to save rank results
        gpu_indices: Comma-separated GPU indices for this process
        phase: Phase number (2 for operator profiling)
    """
    try:
        # Set CUDA_VISIBLE_DEVICES for this specific process
        # This ensures each process uses different GPUs
        os.environ['CUDA_VISIBLE_DEVICES'] = gpu_indices
        
        # Do NOT set VLLM_DP_* environment variables
        # Each process runs independently without DP communication
        # This is sufficient for benchmarking different parallel configurations
        
        logger.info(f"Starting independent process {dp_rank}/{dp_size} (no DP communication)...")
        logger.info(f"Using GPUs: {gpu_indices}")
        
        # Run benchmark for this rank
        result = run_benchmark_single_rank(config, dp_rank, dp_size, phase=phase)
        
        # Save result to file
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Process {dp_rank}/{dp_size} completed successfully")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Process {dp_rank} failed: {e}", exc_info=True)
        sys.exit(1)


def _aggregate_operator_performance(rank_results: List[Dict]) -> Dict[str, Any]:
    """
    Aggregate operator performance from all ranks.
    
    Args:
        rank_results: List of result dictionaries from all DP ranks
        
    Returns:
        Aggregated operator performance dictionary
    """
    import numpy as np
    
    # Collect per_layer_stats from all ranks
    all_per_layer_stats = []
    for result in rank_results:
        op_perf = result.get('operator_performance', {})
        if 'per_layer_stats' in op_perf:
            per_layer = op_perf['per_layer_stats']
            # Check if it's a dict (not an empty list)
            if isinstance(per_layer, dict) and per_layer:
                all_per_layer_stats.append(per_layer)
            elif isinstance(per_layer, list):
                logger.warning(f"Rank has per_layer_stats as list (empty or invalid), skipping")
    
    if not all_per_layer_stats:
        logger.warning("No valid per_layer_stats found across all ranks")
        return {}
    
    # Average per-layer stats across ranks
    aggregated_per_layer = {}
    for layer_name in all_per_layer_stats[0].keys():
        layer_data = [stats[layer_name] for stats in all_per_layer_stats if layer_name in stats]
        if not layer_data:
            continue
        
        aggregated_per_layer[layer_name] = {
            'mean_time_ms': np.mean([d['mean_time_ms'] for d in layer_data]),
            'std_time_ms': np.mean([d['std_time_ms'] for d in layer_data]),
            'min_time_ms': np.mean([d['min_time_ms'] for d in layer_data]),
            'max_time_ms': np.mean([d['max_time_ms'] for d in layer_data]),
            'call_count': sum([d['call_count'] for d in layer_data]),
        }
    
    # Aggregate summary stats
    all_summary_stats = []
    for result in rank_results:
        op_perf = result.get('operator_performance', {})
        summary = op_perf.get('summary_stats', {})
        # Only include if it's a valid dict
        if isinstance(summary, dict) and summary:
            all_summary_stats.append(summary)
    
    if all_summary_stats:
        aggregated_summary = {
            'total_forward_time_ms': sum([s.get('total_forward_time_ms', 0) for s in all_summary_stats]),
            'num_layers_profiled': all_summary_stats[0].get('num_layers_profiled', 0),
            'num_ranks_aggregated': len(rank_results),
            'num_ranks_with_data': len(all_summary_stats),
        }
    else:
        logger.warning("No valid summary_stats found across all ranks")
        aggregated_summary = {
            'total_forward_time_ms': 0,
            'num_layers_profiled': 0,
            'num_ranks_aggregated': len(rank_results),
            'num_ranks_with_data': 0,
        }
    
    return {
        'per_layer_stats': aggregated_per_layer,
        'summary_stats': aggregated_summary,
    }


def _aggregate_dp_results(rank_output_files: List[str], dp_size: int) -> Dict[str, Any]:
    """
    Aggregate results from all DP ranks.
    
    Args:
        rank_output_files: List of file paths containing results from each rank
        dp_size: Total number of DP ranks
        
    Returns:
        Aggregated results dictionary
    """
    logger.info(f"Aggregating results from {dp_size} DP ranks...")
    
    # Load all rank results
    rank_results = []
    for rank, file_path in enumerate(rank_output_files):
        with open(file_path, 'r') as f:
            result = json.load(f)
            result['dp_rank'] = rank
            rank_results.append(result)
    
    # Aggregate operator performance
    aggregated_operator_perf = _aggregate_operator_performance(rank_results)
    
    # Collect per-DP-rank operator performance (including per-TP-rank data)
    per_dp_rank_operator_perf = []
    for result in rank_results:
        dp_rank = result['dp_rank']
        op_perf = result.get('operator_performance', {})
        
        rank_data = {
            'dp_rank': dp_rank,
            'per_layer_stats': op_perf.get('per_layer_stats', {}),
            'summary_stats': op_perf.get('summary_stats', {}),
        }
        
        # Include per-TP-rank data if available
        if 'per_tp_rank' in op_perf:
            rank_data['per_tp_rank'] = op_perf['per_tp_rank']
        
        per_dp_rank_operator_perf.append(rank_data)
    
    # Construct final result
    final_result = {
        'config': rank_results[0]['config'],
        'metadata': {
            'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
            'data_parallel_size': dp_size,
        },
        'operator_performance': aggregated_operator_perf,
        'operator_performance_per_dp_rank': per_dp_rank_operator_perf,
        'results': []  # Empty, all data in top-level fields
    }
    
    # Clean up temp files
    for file_path in rank_output_files:
        try:
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up {file_path}: {e}")
    
    return final_result


def run_benchmark_multi_rank(config: BenchmarkConfig, phase: Optional[int] = None) -> Dict[str, Any]:
    """
    Run benchmark with multiple independent processes (simulated DP).
    
    Note: This does NOT use true data parallelism with inter-process communication.
    Each process runs independently with its own shard of trace data.
    Results are aggregated after all processes complete.
    
    This approach is sufficient for benchmarking different parallel configurations
    without the complexity of NCCL/TCPStore setup.
    
    Args:
        config: Benchmark configuration
        
    Returns:
        Aggregated results dictionary
    """
    from multiprocessing import Process
    
    dp_size = config.data_parallel_size
    
    # These are kept for API compatibility but not actually used
    # since we don't establish DP communication
    dp_master_ip = "127.0.0.1"
    dp_master_port = 0  # Not needed without DP communication
    
    logger.info(f"Launching {dp_size} independent processes (no DP communication)")
    logger.info("Each process will handle a shard of the trace data")
    
    # Determine GPU allocation for each process
    # Get current CUDA_VISIBLE_DEVICES or use all available GPUs
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
            args=(config, rank, dp_size, dp_master_ip, dp_master_port, 
                  rank_output_files[rank], rank_gpu_assignments[rank], phase)
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
    
    # Aggregate results
    return _aggregate_dp_results(rank_output_files, dp_size)


def run_benchmark(config: BenchmarkConfig, phase: Optional[int] = None) -> Dict[str, Any]:
    """
    Run the complete benchmark suite.
    Automatically handles multi-process execution if data_parallel_size > 1.
    
    Note: When data_parallel_size > 1, multiple independent processes are launched
    without actual DP communication. This allows testing different parallel
    configurations without NCCL complexity.
    
    Args:
        config: Benchmark configuration
        phase: For two-phase mode: 1=E2E only, 2=Operator only, None=both (legacy)
        
    Returns:
        Dictionary containing benchmark results
    """
    config.validate()
    
    # Check if we need multi-process execution
    if config.data_parallel_size > 1:
        logger.info(f"=" * 80)
        logger.info(f"Multi-process mode: {config.data_parallel_size} independent processes")
        logger.info(f"(Simulated DP for benchmarking - no inter-process communication)")
        logger.info(f"=" * 80)
        return run_benchmark_multi_rank(config, phase=phase)
    else:
        # Single process mode
        return run_benchmark_single_rank(config, dp_rank=0, dp_size=1, phase=phase)


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
    parser.add_argument(
        "--phase",
        type=int,
        choices=[2],
        required=True,
        help="Benchmark phase: must be 2 for operator profiling (required). For E2E metrics, use benchmark_e2e_metrics.py",
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
        results = run_benchmark(config, phase=args.phase)
        logger.info("\nBenchmark completed successfully!")
        return 0
    except Exception as e:
        logger.error(f"Benchmark failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

