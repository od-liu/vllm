#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
E2E Metrics Configuration Sweep Script.

This script automatically generates and runs E2E benchmarks for all valid TP×DP combinations
that satisfy: TP × DP = num_gpus (with PP fixed at 1).

Unlike the full two-phase benchmark, this script ONLY collects E2E metrics (TTFT, TPOT, 
throughput, latency) without operator-level profiling, making it faster and simpler.

Example usage:
    python benchmarks/e2e_sweep/run_e2e_sweep.py \
        --num-gpus 8 \
        --base-config benchmarks/e2e_sweep/example_configs/e2e_config_template.py \
        --output-dir e2e_results/h100_8gpu \
        --model-path /path/to/model \
        --trace-file /path/to/trace.jsonl \
        --gpu-type h100


        python benchmarks/e2e_sweep/run_e2e_sweep.py \
        --num-gpus 4 \
        --base-config benchmarks/e2e_sweep/example_configs/e2e_config_template.py \
        --output-dir e2e_results/h100_4gpu_600prompts \
        --gpu-ids "0,1,2,3"

For visualization:
    python benchmarks/e2e_sweep/visualize_e2e_results.py \
        --results-dir e2e_results/h100_8gpu
"""

import argparse
import json
import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any
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


def generate_parallel_configs(num_gpus: int) -> List[Tuple[int, int, int]]:
    """
    Generate all valid (TP, PP, DP) combinations where TP × PP × DP = num_gpus.
    
    Current implementation:
    - PP is fixed to 1 (no pipeline parallelism sweep)
    - Only TP and DP are varied: TP × DP = num_gpus
    
    Args:
        num_gpus: Total number of GPUs available
        
    Returns:
        List of (tp, pp, dp) tuples where pp is always 1
    """
    configs = []
    pp = 1  # Fixed: No pipeline parallelism in sweep
    
    tp_values = get_divisors(num_gpus)
    
    for tp in tp_values:
        dp = num_gpus // tp
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
        
    Returns:
        Path to temporary config file
    """
    # Read the base config file
    with open(base_config_path, 'r') as f:
        config_content = f.read()
    
    # Create a modified config
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
        elif trace_file and 'trace_file_path=' in line and not line.strip().startswith('#'):
            # Replace trace_file_path only if it's not commented out
            temp_config_lines.append(f'    trace_file_path="{trace_file}",')
        else:
            temp_config_lines.append(line)
    
    # Write temporary config
    temp_config_content = '\n'.join(temp_config_lines)
    temp_config_path = f"/tmp/e2e_temp_config_tp{tp}_pp{pp}_dp{dp}.py"
    
    with open(temp_config_path, 'w') as f:
        f.write(temp_config_content)
    
    return temp_config_path


def run_e2e_benchmark(
    config_path: str,
    num_gpus: int,
    gpu_ids: str = None,
) -> Tuple[bool, str, float]:
    """
    Run a single E2E benchmark with the given configuration.
    
    Args:
        config_path: Path to configuration file
        num_gpus: Number of GPUs to use
        gpu_ids: Comma-separated GPU IDs to use (e.g., "0,1,2,3")
        
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
    
    # Use benchmark_e2e_metrics.py script
    script = "benchmarks/benchmark_e2e_metrics.py"
    
    # Build command with unbuffered output
    cmd = [
        sys.executable,
        "-u",  # Unbuffered output for real-time logging
        script,
        "--config",
        config_path,
        "--phase",
        "1",  # E2E metrics phase
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
        
        # Use a queue and thread to avoid blocking on stdout
        output_queue = queue.Queue()
        output_lines = []
        
        def reader_thread():
            try:
                for line in process.stdout:
                    output_queue.put(('line', line))
            except Exception as e:
                output_queue.put(('error', str(e)))
            finally:
                output_queue.put(('done', None))
        
        reader = threading.Thread(target=reader_thread, daemon=True)
        reader.start()
        
        # Process output from queue without blocking
        done = False
        while not done:
            try:
                msg_type, data = output_queue.get(timeout=1)
                if msg_type == 'line':
                    print(data, end='')  # Print to terminal in real-time
                    output_lines.append(data)
                elif msg_type == 'error':
                    print(f"Error reading output: {data}", file=sys.stderr)
                elif msg_type == 'done':
                    done = True
            except queue.Empty:
                # Check if process is still running
                if process.poll() is not None:
                    # Process finished, drain any remaining output
                    while not output_queue.empty():
                        try:
                            msg_type, data = output_queue.get_nowait()
                            if msg_type == 'line':
                                print(data, end='')
                                output_lines.append(data)
                        except queue.Empty:
                            break
                    break
        
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


def extract_e2e_metrics(result_file: str) -> dict:
    """
    Extract E2E metrics from a result JSON file.
    
    Args:
        result_file: Path to result JSON file
        
    Returns:
        Dictionary with E2E metrics
    """
    try:
        with open(result_file, 'r') as f:
            data = json.load(f)
        
        metrics = {}
        
        # Extract E2E metrics - prefer aggregated metrics for DP runs
        e2e = None
        if 'e2e_metrics_aggregated' in data:
            e2e = data['e2e_metrics_aggregated']
            metrics['is_aggregated'] = True
        elif 'e2e_metrics' in data:
            e2e = data['e2e_metrics']
            metrics['is_aggregated'] = False
        
        if e2e is not None:
            for metric_name in ['ttft_ms', 'tpot_ms', 'e2e_latency_ms', 'queued_time_ms']:
                if metric_name in e2e and e2e[metric_name]:
                    metric_data = e2e[metric_name]
                    metrics[f"{metric_name}_mean"] = metric_data.get('mean', None)
                    metrics[f"{metric_name}_median"] = metric_data.get('median', metric_data.get('p50', None))
                    metrics[f"{metric_name}_p90"] = metric_data.get('p90', None)
                    metrics[f"{metric_name}_p99"] = metric_data.get('p99', None)
            
            metrics['num_requests'] = e2e.get('num_requests', 0)
            metrics['num_valid_requests'] = e2e.get('num_valid_requests', 0)
            metrics['total_prompt_tokens'] = e2e.get('total_prompt_tokens', 0)
            metrics['total_generation_tokens'] = e2e.get('total_generation_tokens', 0)
            metrics['num_ranks'] = e2e.get('num_ranks', 1)
            metrics['total_time_seconds'] = e2e.get('total_time_seconds', 0)
            
            if 'throughput' in e2e:
                throughput_data = e2e['throughput']
                if isinstance(throughput_data, dict):
                    metrics['throughput_tokens_per_sec'] = throughput_data.get('tokens_per_second', None)
                else:
                    metrics['throughput_tokens_per_sec'] = throughput_data
        
        return metrics
        
    except Exception as e:
        print(f"Warning: Failed to extract metrics from {result_file}: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(
        description="Run E2E metrics configuration sweep (simplified, no operator profiling)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run E2E sweep on 8 GPUs
  python benchmarks/e2e_sweep/run_e2e_sweep.py \\
      --num-gpus 8 \\
      --base-config benchmarks/e2e_sweep/example_configs/e2e_config_template.py \\
      --output-dir e2e_results/h100_8gpu \\
      --model-path /path/to/model \\
      --trace-file /path/to/trace.jsonl \\
      --gpu-type h100
  
  # Use specific GPUs and skip existing results
  python benchmarks/e2e_sweep/run_e2e_sweep.py \\
      --num-gpus 4 \\
      --base-config benchmarks/e2e_sweep/example_configs/e2e_config_template.py \\
      --output-dir e2e_results/test \\
      --gpu-ids "0,1,2,3" \\
      --skip-existing
        """
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
        "--auto-visualize",
        action="store_true",
        default=True,
        help="Automatically generate visualizations after sweep completes (default: True)"
    )
    parser.add_argument(
        "--no-visualize",
        action="store_true",
        help="Disable automatic visualization"
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
    print(f"E2E Metrics Configuration Sweep")
    print("=" * 80)
    print(f"Total GPUs: {args.num_gpus}")
    print(f"GPU Type: {args.gpu_type}")
    print(f"Base Config: {args.base_config}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Pipeline Parallel: Fixed at PP=1 (no PP sweep)")
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
        output_file = os.path.join(args.output_dir, f"{config_name}_e2e.json")
        
        print(f"\n{'=' * 80}")
        print(f"Configuration {i}/{len(configs)}: TP={tp}, PP={pp}, DP={dp}")
        print(f"{'=' * 80}")
        
        # Check if result already exists
        if args.skip_existing and os.path.exists(output_file):
            print(f"Skipping: Result file already exists: {output_file}")
            
            # Extract metrics from existing file
            metrics = extract_e2e_metrics(output_file)
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
        
        # Create temporary config
        temp_config = create_temp_config(
            args.base_config,
            tp, pp, dp,
            output_file,
            args.model_path,
            args.trace_file,
        )
        
        # Run E2E benchmark
        success, error, exec_time = run_e2e_benchmark(
            temp_config,
            args.num_gpus,
            args.gpu_ids,
        )
        
        # Clean up temp config
        try:
            os.remove(temp_config)
        except:
            pass
        
        if not success:
            print(f"✗ Benchmark failed: {error}")
            results_summary.append({
                'config': config_name,
                'tp': tp,
                'pp': pp,
                'dp': dp,
                'status': 'failed',
                'execution_time': exec_time,
                'error': error[:200]  # Truncate error message
            })
            continue
        
        print(f"✓ SUCCESS: Completed in {exec_time:.1f} seconds")
        
        # Extract metrics
        metrics = extract_e2e_metrics(output_file)
        
        results_summary.append({
            'config': config_name,
            'tp': tp,
            'pp': pp,
            'dp': dp,
            'status': 'success',
            'execution_time': exec_time,
            **metrics
        })
    
    # Calculate total time
    total_time = time.time() - total_start_time
    
    # Write summary CSV
    summary_csv_path = os.path.join(args.output_dir, "e2e_summary.csv")
    
    if results_summary:
        # Collect all unique field names
        all_fieldnames = set()
        for result in results_summary:
            all_fieldnames.update(result.keys())
        
        # Sort fieldnames for consistent CSV output
        basic_fields = ['config', 'tp', 'pp', 'dp', 'status', 'execution_time']
        metric_fields = sorted([f for f in all_fieldnames if f not in basic_fields and f != 'error'])
        error_field = ['error'] if 'error' in all_fieldnames else []
        fieldnames = [f for f in basic_fields if f in all_fieldnames] + metric_fields + error_field
        
        with open(summary_csv_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results_summary)
        
        print(f"\n{'=' * 80}")
        print(f"Summary CSV saved to: {summary_csv_path}")
    
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
    
    # Auto-generate visualizations
    if args.auto_visualize and not args.no_visualize:
        print(f"\n{'=' * 80}")
        print("Generating visualizations...")
        print(f"{'=' * 80}")
        
        viz_script = os.path.join(
            os.path.dirname(__file__), 
            "visualize_e2e_results.py"
        )
        
        if os.path.exists(viz_script):
            try:
                subprocess.run(
                    [sys.executable, viz_script, "--results-dir", args.output_dir],
                    check=True
                )
                print(f"\n✓ Visualizations generated successfully!")
            except subprocess.CalledProcessError as e:
                print(f"\n✗ Visualization failed: {e}")
                print(f"You can manually run: python {viz_script} --results-dir {args.output_dir}")
        else:
            print(f"\nVisualization script not found: {viz_script}")
    else:
        print(f"\nTo visualize results, run:")
        print(f"  python benchmarks/e2e_sweep/visualize_e2e_results.py --results-dir {args.output_dir}")
    
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()


