#!/usr/bin/env python3
"""
Visualization tool for vLLM benchmark results.

This script generates comprehensive visualizations from benchmark results,
including E2E metrics (TTFT, TPOT, throughput) and operator-level performance.

Usage:
    python benchmarks/visualize_benchmark_results.py --results-dir benchmark_results/test_run
    python benchmarks/visualize_benchmark_results.py --results-dir benchmark_results/test_run --output-dir plots
    python benchmarks/visualize_benchmark_results.py --summary-csv benchmark_results/test_run/summary.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import csv

# Check if matplotlib is available
try:
    import matplotlib
    matplotlib.use('Agg')  # Use non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not found. Will generate text-based visualization only.")
    print("Install with: pip install matplotlib")


class BenchmarkVisualizer:
    """Visualizer for benchmark results."""
    
    def __init__(self, results_dir: str, output_dir: Optional[str] = None):
        """
        Initialize the visualizer.
        
        Args:
            results_dir: Directory containing benchmark results
            output_dir: Directory to save plots (default: results_dir/plots)
        """
        self.results_dir = Path(results_dir)
        self.output_dir = Path(output_dir) if output_dir else self.results_dir / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results_files = []
        self.summary_data = []
        self.e2e_data = {}
        self.operator_data = {}
        
    def load_results(self):
        """Load all JSON result files and summary CSV."""
        print(f"Loading results from {self.results_dir}...")
        
        # Load summary CSV if exists
        summary_csv = self.results_dir / "summary.csv"
        if summary_csv.exists():
            self._load_summary_csv(summary_csv)
        
        # Load all result JSON files
        for json_file in sorted(self.results_dir.glob("*.json")):
            if json_file.name.startswith("tp") and "results" in json_file.name:
                self._load_result_file(json_file)
        
        print(f"Loaded {len(self.results_files)} result files")
        print(f"Found {len(self.e2e_data)} configurations with E2E data")
        print(f"Found {len(self.operator_data)} configurations with operator data")
        
    def _load_summary_csv(self, csv_path: Path):
        """Load summary CSV file."""
        print(f"Loading summary from {csv_path}")
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            self.summary_data = list(reader)
        print(f"Loaded {len(self.summary_data)} summary entries")
    
    def _load_result_file(self, json_path: Path):
        """Load a single result JSON file."""
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            self.results_files.append({
                'path': json_path,
                'data': data
            })
            
            # Extract configuration from filename
            filename = json_path.stem  # e.g., "tp4_pp1_dp1_results"
            config_key = filename.replace("_results", "")
            
            # Store E2E data (try multiple possible keys)
            if 'e2e_metrics_aggregated' in data and data['e2e_metrics_aggregated']:
                self.e2e_data[config_key] = data['e2e_metrics_aggregated']
            elif 'e2e_metrics' in data and data['e2e_metrics']:
                self.e2e_data[config_key] = data['e2e_metrics']
            
            # Store operator data (try multiple possible keys)
            # operator_performance is a list of dicts, convert to dict for easier processing
            if 'operator_performance' in data and data['operator_performance']:
                op_list = data['operator_performance']
                if isinstance(op_list, list):
                    # Convert list to dict keyed by layer_name
                    self.operator_data[config_key] = {
                        f"{op['layer_name']}" if 'layer_name' in op else f"op_{i}": op 
                        for i, op in enumerate(op_list)
                    }
                else:
                    self.operator_data[config_key] = op_list
            elif 'operator_metrics' in data and data['operator_metrics']:
                self.operator_data[config_key] = data['operator_metrics']
                
        except Exception as e:
            print(f"Warning: Failed to load {json_path}: {e}")
    
    def generate_text_summary(self):
        """Generate text-based summary."""
        output_file = self.output_dir / "summary.txt"
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("VLLM BENCHMARK RESULTS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            # E2E Metrics Summary
            if self.e2e_data:
                f.write("\n" + "=" * 80 + "\n")
                f.write("E2E PERFORMANCE METRICS\n")
                f.write("=" * 80 + "\n\n")
                
                for config, metrics in sorted(self.e2e_data.items()):
                    f.write(f"\n{config.upper()}\n")
                    f.write("-" * 40 + "\n")
                    
                    def format_metric(value, decimals=2):
                        """Format metric value, handling None/N/A cases."""
                        if value is None or value == 'N/A':
                            return 'N/A'
                        try:
                            return f"{float(value):.{decimals}f}"
                        except (ValueError, TypeError):
                            return str(value)
                    
                    if 'ttft_ms' in metrics:
                        f.write(f"  TTFT (Time to First Token):\n")
                        f.write(f"    Mean:   {format_metric(metrics['ttft_ms'].get('mean'))} ms\n")
                        f.write(f"    Median: {format_metric(metrics['ttft_ms'].get('median'))} ms\n")
                        f.write(f"    P90:    {format_metric(metrics['ttft_ms'].get('p90'))} ms\n")
                        f.write(f"    P99:    {format_metric(metrics['ttft_ms'].get('p99'))} ms\n")
                    
                    if 'tpot_ms' in metrics:
                        f.write(f"\n  TPOT (Time per Output Token):\n")
                        f.write(f"    Mean:   {format_metric(metrics['tpot_ms'].get('mean'))} ms\n")
                        f.write(f"    Median: {format_metric(metrics['tpot_ms'].get('median'))} ms\n")
                    
                    if 'e2e_latency_ms' in metrics:
                        f.write(f"\n  E2E Latency:\n")
                        f.write(f"    Mean:   {format_metric(metrics['e2e_latency_ms'].get('mean'))} ms\n")
                        f.write(f"    Median: {format_metric(metrics['e2e_latency_ms'].get('median'))} ms\n")
                    
                    if 'throughput' in metrics:
                        f.write(f"\n  Throughput:\n")
                        f.write(f"    Tokens/s: {format_metric(metrics['throughput'].get('tokens_per_second'))}\n")
                        f.write(f"    Requests/s: {format_metric(metrics['throughput'].get('requests_per_second'))}\n")
            
            # Operator Metrics Summary
            if self.operator_data:
                f.write("\n\n" + "=" * 80 + "\n")
                f.write("TOP OPERATOR PERFORMANCE (by total time)\n")
                f.write("=" * 80 + "\n")
                
                for config, metrics in sorted(self.operator_data.items()):
                    f.write(f"\n{config.upper()}\n")
                    f.write("-" * 40 + "\n")
                    
                    # Get all operators sorted by total time
                    operators = []
                    for op_name, op_data in metrics.items():
                        if isinstance(op_data, dict) and 'total_time_ms' in op_data:
                            operators.append((op_name, op_data))
                    
                    operators.sort(key=lambda x: x[1].get('total_time_ms', 0), reverse=True)
                    
                    # Show top 10
                    for i, (op_name, op_data) in enumerate(operators[:10], 1):
                        total_time = op_data.get('total_time_ms', 0)
                        avg_time = op_data.get('avg_time_ms', 0)
                        count = op_data.get('count', 0)
                        f.write(f"  {i:2d}. {op_name:40s} | Total: {total_time:8.2f}ms | Avg: {avg_time:6.2f}ms | Count: {count:5d}\n")
        
        print(f"✓ Text summary saved to {output_file}")
    
    def plot_e2e_comparison(self):
        """Plot E2E metrics comparison across configurations."""
        if not MATPLOTLIB_AVAILABLE or not self.e2e_data:
            return
        
        configs = list(self.e2e_data.keys())
        
        # Extract metrics
        ttft_means = []
        tpot_means = []
        throughput_tokens = []
        
        for config in configs:
            metrics = self.e2e_data[config]
            ttft_means.append(metrics.get('ttft_ms', {}).get('mean', 0))
            tpot_means.append(metrics.get('tpot_ms', {}).get('mean', 0))
            throughput_tokens.append(metrics.get('throughput', {}).get('tokens_per_second', 0))
        
        # Create subplots
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('E2E Performance Metrics Comparison', fontsize=16, fontweight='bold')
        
        # TTFT comparison
        ax = axes[0, 0]
        bars = ax.bar(range(len(configs)), ttft_means, color='steelblue', alpha=0.8)
        ax.set_xlabel('Configuration', fontweight='bold')
        ax.set_ylabel('TTFT (ms)', fontweight='bold')
        ax.set_title('Time to First Token (Lower is Better)', fontweight='bold')
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, ttft_means)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        # TPOT comparison
        ax = axes[0, 1]
        bars = ax.bar(range(len(configs)), tpot_means, color='coral', alpha=0.8)
        ax.set_xlabel('Configuration', fontweight='bold')
        ax.set_ylabel('TPOT (ms)', fontweight='bold')
        ax.set_title('Time per Output Token (Lower is Better)', fontweight='bold')
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        for i, (bar, val) in enumerate(zip(bars, tpot_means)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                   f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        # Throughput comparison
        ax = axes[1, 0]
        bars = ax.bar(range(len(configs)), throughput_tokens, color='seagreen', alpha=0.8)
        ax.set_xlabel('Configuration', fontweight='bold')
        ax.set_ylabel('Tokens/Second', fontweight='bold')
        ax.set_title('Throughput (Higher is Better)', fontweight='bold')
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        for i, (bar, val) in enumerate(zip(bars, throughput_tokens)):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                   f'{val:.0f}', ha='center', va='bottom', fontsize=9)
        
        # TTFT percentiles
        ax = axes[1, 1]
        p50_list = [self.e2e_data[c].get('ttft_ms', {}).get('median', 0) for c in configs]
        p90_list = [self.e2e_data[c].get('ttft_ms', {}).get('p90', 0) for c in configs]
        p99_list = [self.e2e_data[c].get('ttft_ms', {}).get('p99', 0) for c in configs]
        
        x = np.arange(len(configs))
        width = 0.25
        ax.bar(x - width, p50_list, width, label='P50', color='lightblue', alpha=0.8)
        ax.bar(x, p90_list, width, label='P90', color='steelblue', alpha=0.8)
        ax.bar(x + width, p99_list, width, label='P99', color='darkblue', alpha=0.8)
        
        ax.set_xlabel('Configuration', fontweight='bold')
        ax.set_ylabel('TTFT (ms)', fontweight='bold')
        ax.set_title('TTFT Percentiles Distribution', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "e2e_comparison.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ E2E comparison plot saved to {output_file}")
    
    def plot_operator_performance(self, top_n: int = 10):
        """Plot top operator performance across configurations."""
        if not MATPLOTLIB_AVAILABLE or not self.operator_data:
            return
        
        # Collect all unique operators
        all_operators = set()
        for metrics in self.operator_data.values():
            for op_name in metrics.keys():
                if isinstance(metrics[op_name], dict) and 'total_time_ms' in metrics[op_name]:
                    all_operators.add(op_name)
        
        # For each configuration, get top operators
        configs = list(self.operator_data.keys())
        
        for config in configs:
            metrics = self.operator_data[config]
            
            # Get operators sorted by total time
            operators = []
            for op_name, op_data in metrics.items():
                if isinstance(op_data, dict) and 'total_time_ms' in op_data:
                    operators.append({
                        'name': op_name,
                        'total_time_ms': op_data.get('total_time_ms', 0),
                        'avg_time_ms': op_data.get('avg_time_ms', 0),
                        'count': op_data.get('count', 0),
                    })
            
            operators.sort(key=lambda x: x['total_time_ms'], reverse=True)
            top_ops = operators[:top_n]
            
            if not top_ops:
                continue
            
            # Create plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
            fig.suptitle(f'Top {top_n} Operators - {config.upper()}', fontsize=14, fontweight='bold')
            
            # Plot 1: Total time
            names = [op['name'].split('.')[-1][:30] for op in top_ops]  # Shorten names
            total_times = [op['total_time_ms'] for op in top_ops]
            
            bars = ax1.barh(range(len(names)), total_times, color='steelblue', alpha=0.8)
            ax1.set_yticks(range(len(names)))
            ax1.set_yticklabels(names)
            ax1.set_xlabel('Total Time (ms)', fontweight='bold')
            ax1.set_title('Total Execution Time', fontweight='bold')
            ax1.grid(axis='x', alpha=0.3)
            ax1.invert_yaxis()
            
            # Add value labels
            for i, (bar, val) in enumerate(zip(bars, total_times)):
                ax1.text(val, bar.get_y() + bar.get_height()/2, 
                        f' {val:.1f}ms', va='center', fontsize=9)
            
            # Plot 2: Average time
            avg_times = [op['avg_time_ms'] for op in top_ops]
            
            bars = ax2.barh(range(len(names)), avg_times, color='coral', alpha=0.8)
            ax2.set_yticks(range(len(names)))
            ax2.set_yticklabels(names)
            ax2.set_xlabel('Average Time (ms)', fontweight='bold')
            ax2.set_title('Average Execution Time per Call', fontweight='bold')
            ax2.grid(axis='x', alpha=0.3)
            ax2.invert_yaxis()
            
            # Add value labels with count
            for i, (bar, avg_val, count) in enumerate(zip(bars, avg_times, [op['count'] for op in top_ops])):
                ax2.text(avg_val, bar.get_y() + bar.get_height()/2, 
                        f' {avg_val:.2f}ms (×{count})', va='center', fontsize=9)
            
            plt.tight_layout()
            output_file = self.output_dir / f"operators_{config}.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ Operator performance plot saved to {output_file}")
    
    def plot_parallelism_scaling(self):
        """Plot how performance scales with TP and DP."""
        if not MATPLOTLIB_AVAILABLE or not self.e2e_data:
            return
        
        # Parse configurations to extract TP and DP
        scaling_data = []
        for config, metrics in self.e2e_data.items():
            # Parse config like "tp4_pp1_dp1"
            parts = config.split('_')
            tp = int(parts[0].replace('tp', ''))
            dp = int(parts[2].replace('dp', ''))
            
            scaling_data.append({
                'tp': tp,
                'dp': dp,
                'total_gpus': tp * dp,
                'ttft_mean': metrics.get('ttft_ms', {}).get('mean', 0),
                'tpot_mean': metrics.get('tpot_ms', {}).get('mean', 0),
                'throughput': metrics.get('throughput', {}).get('tokens_per_second', 0),
                'config': config
            })
        
        # Sort by total GPUs
        scaling_data.sort(key=lambda x: x['total_gpus'])
        
        # Create plot
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Performance Scaling Analysis', fontsize=16, fontweight='bold')
        
        configs = [d['config'] for d in scaling_data]
        total_gpus = [d['total_gpus'] for d in scaling_data]
        
        # TTFT vs GPUs
        ax = axes[0]
        ttft_values = [d['ttft_mean'] for d in scaling_data]
        ax.plot(total_gpus, ttft_values, 'o-', color='steelblue', linewidth=2, markersize=8)
        ax.set_xlabel('Total GPUs (TP × DP)', fontweight='bold')
        ax.set_ylabel('TTFT Mean (ms)', fontweight='bold')
        ax.set_title('TTFT vs GPU Count', fontweight='bold')
        ax.grid(True, alpha=0.3)
        for i, (gpu, ttft, cfg) in enumerate(zip(total_gpus, ttft_values, configs)):
            ax.annotate(cfg, (gpu, ttft), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=8)
        
        # TPOT vs GPUs
        ax = axes[1]
        tpot_values = [d['tpot_mean'] for d in scaling_data]
        ax.plot(total_gpus, tpot_values, 'o-', color='coral', linewidth=2, markersize=8)
        ax.set_xlabel('Total GPUs (TP × DP)', fontweight='bold')
        ax.set_ylabel('TPOT Mean (ms)', fontweight='bold')
        ax.set_title('TPOT vs GPU Count', fontweight='bold')
        ax.grid(True, alpha=0.3)
        for i, (gpu, tpot, cfg) in enumerate(zip(total_gpus, tpot_values, configs)):
            ax.annotate(cfg, (gpu, tpot), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=8)
        
        # Throughput vs GPUs
        ax = axes[2]
        throughput_values = [d['throughput'] for d in scaling_data]
        ax.plot(total_gpus, throughput_values, 'o-', color='seagreen', linewidth=2, markersize=8)
        ax.set_xlabel('Total GPUs (TP × DP)', fontweight='bold')
        ax.set_ylabel('Throughput (tokens/s)', fontweight='bold')
        ax.set_title('Throughput vs GPU Count', fontweight='bold')
        ax.grid(True, alpha=0.3)
        for i, (gpu, thru, cfg) in enumerate(zip(total_gpus, throughput_values, configs)):
            ax.annotate(cfg, (gpu, thru), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=8)
        
        plt.tight_layout()
        output_file = self.output_dir / "scaling_analysis.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Scaling analysis plot saved to {output_file}")
    
    def generate_all_visualizations(self):
        """Generate all available visualizations."""
        print("\n" + "=" * 80)
        print("GENERATING VISUALIZATIONS")
        print("=" * 80 + "\n")
        
        # Always generate text summary
        self.generate_text_summary()
        
        if not MATPLOTLIB_AVAILABLE:
            print("\n⚠ Matplotlib not available. Only text summary was generated.")
            print("Install matplotlib to generate plots: pip install matplotlib")
            return
        
        # Generate plots
        self.plot_e2e_comparison()
        self.plot_operator_performance(top_n=10)
        self.plot_parallelism_scaling()
        
        print("\n" + "=" * 80)
        print(f"✓ All visualizations saved to: {self.output_dir}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize vLLM benchmark results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize results from a directory
  python benchmarks/visualize_benchmark_results.py --results-dir benchmark_results/test_run
  
  # Specify custom output directory for plots
  python benchmarks/visualize_benchmark_results.py --results-dir benchmark_results/test_run --output-dir my_plots
  
  # Generate text summary only (if matplotlib not available)
  python benchmarks/visualize_benchmark_results.py --results-dir benchmark_results/test_run
        """
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Directory containing benchmark results (JSON files and summary.csv)'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Directory to save plots (default: <results-dir>/plots)'
    )
    
    args = parser.parse_args()
    
    # Validate results directory
    if not os.path.isdir(args.results_dir):
        print(f"Error: Results directory not found: {args.results_dir}")
        sys.exit(1)
    
    # Create visualizer and generate plots
    visualizer = BenchmarkVisualizer(args.results_dir, args.output_dir)
    visualizer.load_results()
    visualizer.generate_all_visualizations()
    
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()

