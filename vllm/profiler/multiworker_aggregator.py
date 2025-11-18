# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Multi-worker result aggregation for operator benchmarking."""

import json
import pickle
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from vllm.logger import init_logger
from vllm.profiler.operator_profiler import BenchmarkResults, OperatorStats

logger = init_logger(__name__)


def aggregate_worker_results(
    worker_results: List[BenchmarkResults],
) -> BenchmarkResults:
    """
    Aggregate benchmark results from multiple workers.
    
    This function combines results from multiple workers (e.g., in tensor parallel
    or pipeline parallel setups) into a single unified result.
    
    Args:
        worker_results: List of BenchmarkResults from each worker
        
    Returns:
        Aggregated BenchmarkResults
    """
    if not worker_results:
        raise ValueError("No worker results to aggregate")
    
    if len(worker_results) == 1:
        return worker_results[0]
    
    logger.info(f"Aggregating results from {len(worker_results)} workers")
    
    # Aggregate per-layer statistics
    # Group by layer name and operator type
    layer_groups = defaultdict(list)
    
    for worker_result in worker_results:
        for layer_stat in worker_result.per_layer_stats:
            key = (layer_stat["layer_name"], layer_stat["operator_type"])
            layer_groups[key].append(layer_stat)
    
    # Compute aggregated statistics for each layer
    aggregated_per_layer = []
    
    for (layer_name, operator_type), stats_list in sorted(layer_groups.items()):
        # Average across workers
        num_calls_list = [s["num_calls"] for s in stats_list]
        avg_time_list = [s["avg_time_ms"] for s in stats_list]
        total_time_list = [s["total_time_ms"] for s in stats_list]
        
        aggregated_stat = {
            "layer_name": layer_name,
            "operator_type": operator_type,
            "num_calls": int(np.mean(num_calls_list)),
            "avg_time_ms": float(np.mean(avg_time_list)),
            "std_time_ms": float(np.std(avg_time_list)),  # Std across workers
            "min_time_ms": float(np.min([s["min_time_ms"] for s in stats_list])),
            "max_time_ms": float(np.max([s["max_time_ms"] for s in stats_list])),
            "total_time_ms": float(np.mean(total_time_list)),
            "avg_input_shapes": stats_list[0].get("avg_input_shapes", []),
            "avg_output_shapes": stats_list[0].get("avg_output_shapes", []),
            "worker_variance_ms": float(np.std(avg_time_list)),  # Variance across workers
        }
        
        aggregated_per_layer.append(aggregated_stat)
    
    # Aggregate summary statistics
    operator_groups = defaultdict(list)
    
    for worker_result in worker_results:
        for op_type, op_stats in worker_result.summary_stats.items():
            operator_groups[op_type].append(op_stats)
    
    aggregated_summary = {}
    
    for op_type, stats_list in sorted(operator_groups.items()):
        total_calls_list = [s["total_calls"] for s in stats_list]
        total_time_list = [s["total_time_ms"] for s in stats_list]
        avg_time_list = [s["avg_time_per_call_ms"] for s in stats_list]
        
        aggregated_summary[op_type] = {
            "total_calls": int(np.mean(total_calls_list)),
            "total_time_ms": float(np.mean(total_time_list)),
            "avg_time_per_call_ms": float(np.mean(avg_time_list)),
            "std_time_ms": float(np.std(avg_time_list)),
            "percentage_of_total": 0.0,  # Will be recalculated below
            "worker_variance_ms": float(np.std(total_time_list)),
        }
    
    # Recalculate percentages
    total_time = sum(s["total_time_ms"] for s in aggregated_summary.values())
    
    for op_stats in aggregated_summary.values():
        op_stats["percentage_of_total"] = (
            (op_stats["total_time_ms"] / total_time * 100) if total_time > 0 else 0.0
        )
    
    # Aggregate config
    aggregated_config = worker_results[0].config.copy()
    aggregated_config["num_workers"] = len(worker_results)
    
    # Average total forward time
    avg_total_forward_time = float(
        np.mean([r.total_forward_time_ms for r in worker_results])
    )
    
    return BenchmarkResults(
        per_layer_stats=aggregated_per_layer,
        summary_stats=aggregated_summary,
        total_forward_time_ms=avg_total_forward_time,
        config=aggregated_config,
    )


def save_worker_results_to_temp(
    results: BenchmarkResults,
    worker_rank: int,
    temp_dir: Optional[str] = None,
) -> str:
    """
    Save worker results to a temporary file.
    
    Args:
        results: BenchmarkResults to save
        worker_rank: Rank of this worker
        temp_dir: Optional temporary directory (if None, uses system temp)
        
    Returns:
        Path to the saved file
    """
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    
    temp_path = Path(temp_dir) / f"vllm_benchmark_worker_{worker_rank}.pkl"
    
    with open(temp_path, "wb") as f:
        pickle.dump(results, f)
    
    logger.info(f"Worker {worker_rank} saved results to {temp_path}")
    
    return str(temp_path)


def load_worker_results_from_temp(
    num_workers: int,
    temp_dir: Optional[str] = None,
    timeout_seconds: float = 60.0,
) -> List[BenchmarkResults]:
    """
    Load worker results from temporary files.
    
    Args:
        num_workers: Expected number of workers
        temp_dir: Optional temporary directory (if None, uses system temp)
        timeout_seconds: Maximum time to wait for all worker files
        
    Returns:
        List of BenchmarkResults from all workers
    """
    import time
    
    if temp_dir is None:
        temp_dir = tempfile.gettempdir()
    
    temp_dir_path = Path(temp_dir)
    worker_results = []
    
    start_time = time.time()
    
    for rank in range(num_workers):
        temp_path = temp_dir_path / f"vllm_benchmark_worker_{rank}.pkl"
        
        # Wait for file to exist
        while not temp_path.exists():
            if time.time() - start_time > timeout_seconds:
                raise TimeoutError(
                    f"Timeout waiting for worker {rank} results at {temp_path}"
                )
            time.sleep(0.1)
        
        # Load results
        with open(temp_path, "rb") as f:
            results = pickle.load(f)
        
        worker_results.append(results)
        logger.info(f"Loaded results from worker {rank}")
        
        # Clean up temp file
        try:
            temp_path.unlink()
        except Exception as e:
            logger.warning(f"Failed to delete temp file {temp_path}: {e}")
    
    return worker_results


def merge_results_by_config(
    results_list: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Merge multiple benchmark runs with potentially different configurations.
    
    This is useful for comparing results across different settings.
    
    Args:
        results_list: List of result dictionaries (from JSON files)
        
    Returns:
        Merged results dictionary
    """
    if not results_list:
        return {}
    
    merged = {
        "runs": results_list,
        "num_runs": len(results_list),
        "metadata": {
            "merged_at": __import__("time").strftime("%Y-%m-%d %H:%M:%S"),
        },
    }
    
    return merged


def compare_results(
    baseline_results: Dict[str, Any],
    comparison_results: Dict[str, Any],
    output_file: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Compare two benchmark results and compute differences.
    
    Args:
        baseline_results: Baseline benchmark results
        comparison_results: Results to compare against baseline
        output_file: Optional path to save comparison results
        
    Returns:
        Comparison results dictionary
    """
    comparison = {
        "baseline": baseline_results.get("metadata", {}),
        "comparison": comparison_results.get("metadata", {}),
        "differences": {},
    }
    
    # Compare summary statistics
    baseline_summary = {}
    comparison_summary = {}
    
    for result in baseline_results.get("results", []):
        key = (result["batch_size"], result["seq_length"])
        baseline_summary[key] = result.get("summary_stats", {})
    
    for result in comparison_results.get("results", []):
        key = (result["batch_size"], result["seq_length"])
        comparison_summary[key] = result.get("summary_stats", {})
    
    # Compute differences for common configurations
    for key in set(baseline_summary.keys()) & set(comparison_summary.keys()):
        batch_size, seq_length = key
        base_stats = baseline_summary[key]
        comp_stats = comparison_summary[key]
        
        config_diff = {
            "batch_size": batch_size,
            "seq_length": seq_length,
            "operator_differences": {},
        }
        
        for op_type in set(base_stats.keys()) & set(comp_stats.keys()):
            base_time = base_stats[op_type]["total_time_ms"]
            comp_time = comp_stats[op_type]["total_time_ms"]
            
            diff_ms = comp_time - base_time
            diff_percent = ((comp_time - base_time) / base_time * 100) if base_time > 0 else 0.0
            
            config_diff["operator_differences"][op_type] = {
                "baseline_time_ms": base_time,
                "comparison_time_ms": comp_time,
                "difference_ms": diff_ms,
                "difference_percent": diff_percent,
                "faster": diff_ms < 0,
            }
        
        comparison["differences"][f"bs{batch_size}_sl{seq_length}"] = config_diff
    
    if output_file:
        with open(output_file, "w") as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"Comparison results saved to {output_file}")
    
    return comparison

