#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Parallel Configuration Sweep Script for Two-Phase Benchmarking.

This script automatically generates and runs benchmarks for all valid TP×PP×DP combinations
that satisfy: TP × PP × DP = num_gpus.

Phase 1: E2E metrics (TTFT, TPOT, latency, throughput) using benchmark_e2e_metrics.py
Phase 2: Operator-level profiling using benchmark_operators.py

Example usage:
    python benchmarks/run_benchmarks.py \
        --num-gpus 8 \
        --base-config benchmarks/operator_configs/e2e_trace_config.py \
        --output-dir benchmark_results/h100_8gpu_sweep \
        --model-path /path/to/model \
        --trace-file /path/to/trace.jsonl
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple
import csv
from datetime import datetime


def get_divisors(n: int) -> List[int]:
    """Get all divisors of n in ascending order."""
    divisors = []
    for i in range(1, int(n**0.5) + 1):
        if n % i == 0:
            divisors.append(i)
            if i != n // i:
                divisors.append(n // i)
    return sorted(divisors)


def generate_parallel_configs(num_gpus: int, skip_data_parallel: bool = True) -> List[Tuple[int, int, int]]:
    """
    Generate all valid (TP, PP, DP) combinations where TP × PP × DP = num_gpus.
    
    Current implementation:
    - PP is fixed to 1 (no pipeline parallelism sweep due to complex memory requirements)
    - Only TP and DP are varied: TP × DP = num_gpus
    - Constraint: TP >= DP to ensure valid configurations
    
    Args:
        num_gpus: Total number of GPUs available
        skip_data_parallel: Deprecated parameter, kept for compatibility.
                           All configs with TP >= DP are now included.
        
    Returns:
        List of (tp, pp, dp) tuples where pp is always 1
    """
    configs = []
    pp = 1  # Fixed: No pipeline parallelism in sweep
    
    tp_values = get_divisors(num_gpus)
    
    for tp in tp_values:
        dp = num_gpus // tp
        
        # With multiprocessing DP support, all TP/DP combinations are valid
        # No constraints needed - each combination will work correctly:
        # - DP=1: Single process with TP
        # - DP>1: Multiple processes with TP per process
        configs.append((tp, pp, dp))
    
    # Sort configs by DP first (ascending), then by TP (descending)
    # This gives us: DP=1,TP=4 -> DP=2,TP=2 -> DP=4,TP=1
    configs.sort(key=lambda x: (x[2], -x[0]))
    return configs


def create_temp_config(
    base_config_path: str,
    tp: int,
    pp: int,
    dp: int,
    output_file: str,
    model_path: str = None,
    trace_file: str = None,
    phase: int = None,
) -> str:
    """
    Create a temporary configuration file with specified TP/PP/DP settings.
    
    Args:
        base_config_path: Path to base configuration file
        tp: Tensor parallel size
        pp: Pipeline parallel size
        dp: Data parallel size
        output_file: Output file name for results
        model_path: Optional model path override
        trace_file: Optional trace file path override
        phase: Phase number (1 or 2) to include in temp filename for uniqueness
        
    Returns:
        Path to temporary config file
    """
    # Read the base config file
    with open(base_config_path, 'r') as f:
        config_content = f.read()
    
    # Create a modified config
    # Parse and modify the config
    temp_config_lines = []
    for line in config_content.split('\n'):
        if 'tensor_parallel_size=' in line:
            temp_config_lines.append(f"    tensor_parallel_size={tp},")
        elif 'pipeline_parallel_size=' in line:
            temp_config_lines.append(f"    pipeline_parallel_size={pp},")
        elif 'data_parallel_size=' in line:
            temp_config_lines.append(f"    data_parallel_size={dp},")
        elif 'output_file=' in line:
            temp_config_lines.append(f'    output_file="{output_file}",')
        elif model_path and 'model_path=' in line:
            temp_config_lines.append(f'    model_path="{model_path}",')
        elif trace_file and 'trace_file_path=' in line:
            temp_config_lines.append(f'    trace_file_path="{trace_file}",')
        else:
            temp_config_lines.append(line)
    
    # Write temporary config
    temp_config_content = '\n'.join(temp_config_lines)
    
    # Include phase in filename to avoid conflicts between phase 1 and phase 2
    phase_suffix = f"_phase{phase}" if phase is not None else ""
    temp_config_path = f"/tmp/temp_config_tp{tp}_pp{pp}_dp{dp}{phase_suffix}.py"
    
    with open(temp_config_path, 'w') as f:
        f.write(temp_config_content)
    
    return temp_config_path


def run_benchmark(
    config_path: str,
    num_gpus: int,
    gpu_ids: str = None,
    phase: int = None,
) -> Tuple[bool, str, float]:
    """
    Run a single benchmark with the given configuration.
    
    Args:
        config_path: Path to configuration file
        num_gpus: Number of GPUs to use
        gpu_ids: Comma-separated GPU IDs to use (e.g., "0,1,2,3")
        phase: Phase to run (1 for E2E metrics, 2 for operator profiling)
        
    Returns:
        Tuple of (success, error_message, execution_time)
    """
    # Set up environment
    env = os.environ.copy()
    
    if gpu_ids:
        env['CUDA_VISIBLE_DEVICES'] = gpu_ids
    else:
        # Use first N GPUs
        env['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, range(num_gpus)))
    
    # Select script based on phase
    if phase == 1:
        script = "benchmarks/benchmark_e2e_metrics.py"
    elif phase == 2:
        script = "benchmarks/benchmark_operators.py"
    else:
        return False, f"Invalid phase: {phase}. Must be 1 (E2E) or 2 (Operator).", 0
    
    # Build command with unbuffered output
    cmd = [
        sys.executable,
        "-u",  # Unbuffered output for real-time logging
        script,
        "--config",
        config_path,
        "--phase",
        str(phase),
    ]
    
    print(f"\nRunning command: {' '.join(cmd)}")
    print(f"CUDA_VISIBLE_DEVICES: {env['CUDA_VISIBLE_DEVICES']}")
    print("-" * 80)
    print("Live output from benchmark:")
    print("-" * 80)
    
    start_time = time.time()
    
    try:
        # Use Popen to stream output in real-time
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Merge stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
        )
        
        # Stream output line by line
        output_lines = []
        for line in process.stdout:
            print(line, end='')  # Print to terminal in real-time
            output_lines.append(line)
        
        # Wait for process to complete
        return_code = process.wait(timeout=3600)
        execution_time = time.time() - start_time
        
        print("-" * 80)
        print(f"Process completed in {execution_time:.2f} seconds")
        print("-" * 80)
        
        if return_code == 0:
            return True, "", execution_time
        else:
            error_msg = f"Command failed with return code {return_code}\n"
            error_msg += "Output:\n" + "".join(output_lines)
            return False, error_msg, execution_time
            
    except subprocess.TimeoutExpired:
        execution_time = time.time() - start_time
        process.kill()
        return False, "Benchmark timed out after 1 hour", execution_time
    except Exception as e:
        execution_time = time.time() - start_time
        return False, f"Exception occurred: {str(e)}", execution_time


def merge_phase_results(phase1_file: str, phase2_file: str, output_file: str):
    """
    Merge Phase 1 (E2E metrics) and Phase 2 (operator performance) results.
    
    Args:
        phase1_file: Path to Phase 1 result JSON
        phase2_file: Path to Phase 2 result JSON  
        output_file: Path to merged output JSON
    """
    try:
        with open(phase1_file, 'r') as f:
            phase1_data = json.load(f)
        
        with open(phase2_file, 'r') as f:
            phase2_data = json.load(f)
        
        # Merge: use config from phase1, add operator_performance from phase2
        merged = {
            "config": phase1_data.get("config", {}),
            "metadata": phase1_data.get("metadata", {}),
            "e2e_metrics_aggregated": phase1_data.get("e2e_metrics_aggregated"),
            "e2e_metrics_per_rank": phase1_data.get("e2e_metrics_per_rank", []),
            "operator_performance": phase2_data.get("operator_performance", {}),
            "operator_performance_per_dp_rank": phase2_data.get("operator_performance_per_dp_rank", []),
            "results": phase2_data.get("results", []),
        }
        
        with open(output_file, 'w') as f:
            json.dump(merged, f, indent=2)
        
        print(f"✓ Merged Phase 1 and Phase 2 results into: {output_file}")
        
    except Exception as e:
        print(f"✗ Failed to merge phase results: {e}")
        raise


def extract_summary_metrics(result_file: str) -> dict:
    """
    Extract summary metrics from a result JSON file.
    
    Args:
        result_file: Path to result JSON file
        
    Returns:
        Dictionary with summary metrics
    """
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        metrics = {}
        
        # Extract E2E metrics if available
        # Prefer aggregated metrics for DP runs, fallback to regular metrics
        e2e = None
        if 'e2e_metrics_aggregated' in data:
            e2e = data['e2e_metrics_aggregated']
            metrics['is_aggregated'] = True
        elif 'e2e_metrics' in data:
            e2e = data['e2e_metrics']
            metrics['is_aggregated'] = False
        
        if e2e is not None:
            for metric_name in ['ttft_ms', 'tpot_ms', 'e2e_latency_ms', 'queued_time_ms']:
                if metric_name in e2e:
                    metric_data = e2e[metric_name]
                    if metric_data:  # Check if not None
                        metrics[f"{metric_name}_mean"] = metric_data.get('mean', None)
                        metrics[f"{metric_name}_p50"] = metric_data.get('p50', None)
                        metrics[f"{metric_name}_p90"] = metric_data.get('p90', None)
                        metrics[f"{metric_name}_p99"] = metric_data.get('p99', None)
            
            metrics['num_requests'] = e2e.get('num_requests', 0)
            metrics['total_generation_tokens'] = e2e.get('total_generation_tokens', 0)
            metrics['num_ranks'] = e2e.get('num_ranks', 1)
            
            if 'throughput' in e2e:
                throughput_data = e2e['throughput']
                if isinstance(throughput_data, dict):
                    metrics['throughput_tokens_per_sec'] = throughput_data.get('tokens_per_second', None)
                else:
                    metrics['throughput_tokens_per_sec'] = throughput_data
        
        # Extract operator performance summary if available
        if 'operator_performance' in data and 'summary_stats' in data['operator_performance']:
            summary_stats = data['operator_performance']['summary_stats']
            
            # Get top 3 operators by time
            sorted_ops = sorted(
                summary_stats.items(),
                key=lambda x: x[1].get('total_time_ms', 0) if isinstance(x[1], dict) else 0,
                reverse=True
            )[:3]
            
            for i, (op_name, op_stats) in enumerate(sorted_ops, 1):
                if isinstance(op_stats, dict):
                    metrics[f'top{i}_operator'] = op_name
                    metrics[f'top{i}_time_ms'] = op_stats.get('total_time_ms', 0)
                    metrics[f'top{i}_percentage'] = op_stats.get('percentage_of_total', 0)
        
        return metrics
        
    except Exception as e:
        print(f"Warning: Failed to extract metrics from {result_file}: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Run parallel configuration sweep for two-phase benchmarking (E2E + Operator)"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        required=True,
        help="Total number of GPUs available"
    )
    parser.add_argument(
        "--base-config",
        type=str,
        required=True,
        help="Path to base configuration file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to store benchmark results"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Override model path from base config"
    )
    parser.add_argument(
        "--trace-file",
        type=str,
        default=None,
        help="Override trace file path from base config"
    )
    parser.add_argument(
        "--gpu-ids",
        type=str,
        default=None,
        help="Comma-separated GPU IDs to use (e.g., '0,1,2,3')"
    )
    parser.add_argument(
        "--gpu-type",
        type=str,
        default="unknown",
        help="GPU type for documentation (e.g., 'h100', 'a100')"
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip configurations that already have result files"
    )
    parser.add_argument(
        "--include-data-parallel",
        action="store_true",
        help="Include configurations with DP > 1"
    )
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.base_config):
        print(f"Error: Base config file not found: {args.base_config}")
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Generate all parallel configurations
    configs = generate_parallel_configs(args.num_gpus)
    
    print("=" * 80)
    print(f"Parallel Configuration Sweep - Two-Phase Benchmarking")
    print("=" * 80)
    print(f"Total GPUs: {args.num_gpus}")
    print(f"GPU Type: {args.gpu_type}")
    print(f"Base Config: {args.base_config}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Pipeline Parallel: Fixed at PP=1 (no PP sweep)")
    print(f"  Note: PP > 1 has complex memory requirements and is excluded from sweep")
    print(f"Number of configurations: {len(configs)}")
    print("\nConfigurations to run:")
    for tp, pp, dp in configs:
        print(f"  TP={tp}, PP={pp}, DP={dp}")
    print("=" * 80)
    
    # Run benchmarks for each configuration
    results_summary = []
    total_start_time = time.time()
    
    for i, (tp, pp, dp) in enumerate(configs, 1):
        config_name = f"tp{tp}_pp{pp}_dp{dp}"
        
        # Separate output files for each phase
        output_file_p1 = os.path.join(args.output_dir, f"{config_name}_phase1.json")
        output_file_p2 = os.path.join(args.output_dir, f"{config_name}_phase2.json")
        final_output_file = os.path.join(args.output_dir, f"{config_name}_results.json")
        
        print(f"\n{'=' * 80}")
        print(f"Configuration {i}/{len(configs)}: TP={tp}, PP={pp}, DP={dp}")
        print(f"{'=' * 80}")
        
        # Check if final result already exists
        if args.skip_existing and os.path.exists(final_output_file):
            print(f"Skipping: Result file already exists: {final_output_file}")
            
            # Extract metrics from existing file
            metrics = extract_summary_metrics(final_output_file)
            results_summary.append({
                'config': config_name,
                'tp': tp,
                'pp': pp,
                'dp': dp,
                'status': 'skipped',
                'execution_time': 0,
                **metrics
            })
            continue
        
        # Create temporary configs for both phases
        temp_config_p1 = create_temp_config(
            args.base_config,
            tp, pp, dp,
            output_file_p1,
            args.model_path,
            args.trace_file,
            phase=1,  # Phase 1: E2E metrics
        )
        
        temp_config_p2 = create_temp_config(
            args.base_config,
            tp, pp, dp,
            output_file_p2,
            args.model_path,
            args.trace_file,
            phase=2,  # Phase 2: Operator profiling
        )
        
        # ========== Run Phase 1: E2E Metrics ==========
        print(f"\n--- Phase 1: E2E Metrics ---")
        success_p1, error_p1, time_p1 = run_benchmark(
            temp_config_p1,
            args.num_gpus,
            args.gpu_ids,
            phase=1,
        )
        
        if not success_p1:
            print(f"✗ Phase 1 failed: {error_p1}")
            # Clean up temp configs
            try:
                os.remove(temp_config_p1)
                os.remove(temp_config_p2)
            except:
                pass
            
            results_summary.append({
                'config': config_name,
                'tp': tp,
                'pp': pp,
                'dp': dp,
                'status': 'failed',
                'execution_time': time_p1,
                'error': f"Phase 1 failed: {error_p1}"
            })
            continue
        
        print(f"✓ Phase 1 completed in {time_p1:.1f} seconds")
        
        # ========== Run Phase 2: Operator Performance ==========
        print(f"\n--- Phase 2: Operator Performance ---")
        success_p2, error_p2, time_p2 = run_benchmark(
            temp_config_p2,
            args.num_gpus,
            args.gpu_ids,
            phase=2,
        )
        
        if not success_p2:
            print(f"✗ Phase 2 failed: {error_p2}")
            # Clean up temp configs and phase 1 result
            try:
                os.remove(temp_config_p1)
                os.remove(temp_config_p2)
                os.remove(output_file_p1)
            except:
                pass
            
            results_summary.append({
                'config': config_name,
                'tp': tp,
                'pp': pp,
                'dp': dp,
                'status': 'failed',
                'execution_time': time_p1 + time_p2,
                'error': f"Phase 2 failed: {error_p2}"
            })
            continue
        
        print(f"✓ Phase 2 completed in {time_p2:.1f} seconds")
        
        # ========== Merge Results ==========
        try:
            merge_phase_results(output_file_p1, output_file_p2, final_output_file)
            total_exec_time = time_p1 + time_p2
            print(f"✓ SUCCESS: Total time {total_exec_time:.1f} seconds")
            
            # Extract metrics from merged file
            metrics = extract_summary_metrics(final_output_file)
            
            results_summary.append({
                'config': config_name,
                'tp': tp,
                'pp': pp,
                'dp': dp,
                'status': 'success',
                'execution_time': total_exec_time,
                **metrics
            })
            
            # Clean up intermediate files
            try:
                os.remove(output_file_p1)
                os.remove(output_file_p2)
            except:
                pass
            
        except Exception as e:
            print(f"✗ Failed to merge results: {e}")
            results_summary.append({
                'config': config_name,
                'tp': tp,
                'pp': pp,
                'dp': dp,
                'status': 'failed',
                'execution_time': time_p1 + time_p2,
                'error': f"Failed to merge results: {e}"
            })
        
        # Clean up temp configs
        try:
            os.remove(temp_config_p1)
            os.remove(temp_config_p2)
        except:
            pass
    
    # Calculate total time
    total_time = time.time() - total_start_time
    
    # Write summary CSV
    summary_csv_path = os.path.join(args.output_dir, "summary.csv")
    
    if results_summary:
        # Collect all unique field names from all results
        all_fieldnames = set()
        for result in results_summary:
            all_fieldnames.update(result.keys())
        
        # Sort fieldnames for consistent CSV output
        # Put basic fields first, then metrics fields
        basic_fields = ['config', 'tp', 'pp', 'dp', 'status', 'execution_time']
        metric_fields = sorted([f for f in all_fieldnames if f not in basic_fields and f != 'error'])
        error_field = ['error'] if 'error' in all_fieldnames else []
        fieldnames = [f for f in basic_fields if f in all_fieldnames] + metric_fields + error_field
        
        with open(summary_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_summary)
        
        print(f"\n{'=' * 80}")
        print(f"Summary saved to: {summary_csv_path}")
    
    # Print final summary
    print(f"\n{'=' * 80}")
    print("FINAL SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total configurations: {len(configs)}")
    print(f"Successful: {sum(1 for r in results_summary if r['status'] == 'success')}")
    print(f"Failed: {sum(1 for r in results_summary if r['status'] == 'failed')}")
    print(f"Skipped: {sum(1 for r in results_summary if r['status'] == 'skipped')}")
    print(f"Total execution time: {total_time/60:.1f} minutes")
    print(f"\nResults directory: {args.output_dir}")
    print(f"Summary CSV: {summary_csv_path}")
    print(f"\nTo visualize results, run:")
    print(f"  python benchmarks/visualize_results.py --results-dir {args.output_dir}")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()

