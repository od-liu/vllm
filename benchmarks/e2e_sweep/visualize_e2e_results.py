#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Visualization tool for E2E benchmark sweep results.

This script generates comprehensive visualizations from E2E benchmark results,
including TTFT, TPOT, throughput, latency, and efficiency metrics across
different TP/DP configurations.

Usage:
    python benchmarks/e2e_sweep/visualize_e2e_results.py --results-dir e2e_results/h100_8gpu
    python benchmarks/e2e_sweep/visualize_e2e_results.py --results-dir e2e_results/h100_8gpu --output-dir my_plots
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


class E2EVisualizer:
    """Visualizer for E2E benchmark sweep results."""
    
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
        
    def load_results(self):
        """Load all JSON result files and summary CSV."""
        print(f"Loading results from {self.results_dir}...")
        
        # Load summary CSV if exists
        summary_csv = self.results_dir / "e2e_summary.csv"
        if summary_csv.exists():
            self._load_summary_csv(summary_csv)
        
        # Load all result JSON files
        for json_file in sorted(self.results_dir.glob("*.json")):
            if json_file.name.startswith("tp") and "_e2e.json" in json_file.name:
                self._load_result_file(json_file)
        
        print(f"Loaded {len(self.results_files)} result files")
        print(f"Found {len(self.e2e_data)} configurations with E2E data")
        
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
            
            # Extract configuration from filename (e.g., "tp4_pp1_dp1_e2e.json")
            filename = json_path.stem
            config_key = filename.replace("_e2e", "")
            
            # Store E2E data
            if 'e2e_metrics_aggregated' in data and data['e2e_metrics_aggregated']:
                self.e2e_data[config_key] = data['e2e_metrics_aggregated']
            elif 'e2e_metrics' in data and data['e2e_metrics']:
                self.e2e_data[config_key] = data['e2e_metrics']
                
        except Exception as e:
            print(f"Warning: Failed to load {json_path}: {e}")
    
    def generate_text_summary(self):
        """Generate text-based summary."""
        output_file = self.output_dir / "e2e_summary.txt"
        
        with open(output_file, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("E2E BENCHMARK SWEEP RESULTS SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            
            if self.e2e_data:
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
                    
                    # Basic stats
                    f.write(f"  Requests: {metrics.get('num_requests', 'N/A')}\n")
                    f.write(f"  Total Time: {format_metric(metrics.get('total_time_seconds'), 1)} seconds\n")
                    
                    if 'ttft_ms' in metrics and metrics['ttft_ms']:
                        f.write(f"\n  TTFT (Time to First Token):\n")
                        f.write(f"    Mean:   {format_metric(metrics['ttft_ms'].get('mean'))} ms\n")
                        f.write(f"    Median: {format_metric(metrics['ttft_ms'].get('median', metrics['ttft_ms'].get('p50')))} ms\n")
                        f.write(f"    P90:    {format_metric(metrics['ttft_ms'].get('p90'))} ms\n")
                        f.write(f"    P99:    {format_metric(metrics['ttft_ms'].get('p99'))} ms\n")
                    
                    if 'tpot_ms' in metrics and metrics['tpot_ms']:
                        f.write(f"\n  TPOT (Time per Output Token):\n")
                        f.write(f"    Mean:   {format_metric(metrics['tpot_ms'].get('mean'))} ms\n")
                        f.write(f"    Median: {format_metric(metrics['tpot_ms'].get('median', metrics['tpot_ms'].get('p50')))} ms\n")
                        f.write(f"    P90:    {format_metric(metrics['tpot_ms'].get('p90'))} ms\n")
                        f.write(f"    P99:    {format_metric(metrics['tpot_ms'].get('p99'))} ms\n")
                    
                    if 'e2e_latency_ms' in metrics and metrics['e2e_latency_ms']:
                        f.write(f"\n  E2E Latency:\n")
                        f.write(f"    Mean:   {format_metric(metrics['e2e_latency_ms'].get('mean'))} ms\n")
                        f.write(f"    Median: {format_metric(metrics['e2e_latency_ms'].get('median', metrics['e2e_latency_ms'].get('p50')))} ms\n")
                        f.write(f"    P90:    {format_metric(metrics['e2e_latency_ms'].get('p90'))} ms\n")
                        f.write(f"    P99:    {format_metric(metrics['e2e_latency_ms'].get('p99'))} ms\n")
                    
                    if 'throughput' in metrics:
                        f.write(f"\n  Throughput:\n")
                        throughput_data = metrics['throughput']
                        if isinstance(throughput_data, dict):
                            f.write(f"    Tokens/s: {format_metric(throughput_data.get('tokens_per_second'))}\n")
                        else:
                            f.write(f"    Tokens/s: {format_metric(throughput_data)}\n")
        
        print(f"✓ Text summary saved to {output_file}")
    
    def plot_e2e_comparison(self):
        """Plot E2E metrics comparison across configurations."""
        if not MATPLOTLIB_AVAILABLE or not self.e2e_data:
            return
        
        configs = sorted(self.e2e_data.keys())
        
        # Extract metrics
        ttft_means = []
        tpot_means = []
        throughput_tokens = []
        e2e_latency_means = []
        
        for config in configs:
            metrics = self.e2e_data[config]
            ttft_means.append(metrics.get('ttft_ms', {}).get('mean', 0) if metrics.get('ttft_ms') else 0)
            tpot_means.append(metrics.get('tpot_ms', {}).get('mean', 0) if metrics.get('tpot_ms') else 0)
            
            throughput_data = metrics.get('throughput', {})
            if isinstance(throughput_data, dict):
                throughput_tokens.append(throughput_data.get('tokens_per_second', 0))
            else:
                throughput_tokens.append(throughput_data if throughput_data else 0)
            
            e2e_latency_means.append(metrics.get('e2e_latency_ms', {}).get('mean', 0) if metrics.get('e2e_latency_ms') else 0)
        
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
        for bar, val in zip(bars, ttft_means):
            if val > 0:
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
        for bar, val in zip(bars, tpot_means):
            if val > 0:
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
        for bar, val in zip(bars, throughput_tokens):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'{val:.0f}', ha='center', va='bottom', fontsize=9)
        
        # E2E Latency comparison
        ax = axes[1, 1]
        bars = ax.bar(range(len(configs)), e2e_latency_means, color='mediumpurple', alpha=0.8)
        ax.set_xlabel('Configuration', fontweight='bold')
        ax.set_ylabel('E2E Latency (ms)', fontweight='bold')
        ax.set_title('End-to-End Latency (Lower is Better)', fontweight='bold')
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, e2e_latency_means):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'{val:.1f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_file = self.output_dir / "e2e_comparison.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ E2E comparison plot saved to {output_file}")
    
    def plot_percentiles(self):
        """Plot percentile distributions for TTFT and TPOT."""
        if not MATPLOTLIB_AVAILABLE or not self.e2e_data:
            return
        
        configs = sorted(self.e2e_data.keys())
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('E2E Metrics Percentile Distributions', fontsize=16, fontweight='bold')
        
        # TTFT percentiles
        ax = axes[0]
        p50_list = []
        p90_list = []
        p99_list = []
        
        for c in configs:
            ttft = self.e2e_data[c].get('ttft_ms', {})
            if ttft:
                p50_list.append(ttft.get('median', ttft.get('p50', 0)))
                p90_list.append(ttft.get('p90', 0))
                p99_list.append(ttft.get('p99', 0))
            else:
                p50_list.append(0)
                p90_list.append(0)
                p99_list.append(0)
        
        x = np.arange(len(configs))
        width = 0.25
        ax.bar(x - width, p50_list, width, label='P50/Median', color='lightblue', alpha=0.8)
        ax.bar(x, p90_list, width, label='P90', color='steelblue', alpha=0.8)
        ax.bar(x + width, p99_list, width, label='P99', color='darkblue', alpha=0.8)
        
        ax.set_xlabel('Configuration', fontweight='bold')
        ax.set_ylabel('TTFT (ms)', fontweight='bold')
        ax.set_title('TTFT Percentiles Distribution', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        # TPOT percentiles
        ax = axes[1]
        p50_list = []
        p90_list = []
        p99_list = []
        
        for c in configs:
            tpot = self.e2e_data[c].get('tpot_ms', {})
            if tpot:
                p50_list.append(tpot.get('median', tpot.get('p50', 0)))
                p90_list.append(tpot.get('p90', 0))
                p99_list.append(tpot.get('p99', 0))
            else:
                p50_list.append(0)
                p90_list.append(0)
                p99_list.append(0)
        
        x = np.arange(len(configs))
        ax.bar(x - width, p50_list, width, label='P50/Median', color='lightcoral', alpha=0.8)
        ax.bar(x, p90_list, width, label='P90', color='coral', alpha=0.8)
        ax.bar(x + width, p99_list, width, label='P99', color='darkred', alpha=0.8)
        
        ax.set_xlabel('Configuration', fontweight='bold')
        ax.set_ylabel('TPOT (ms)', fontweight='bold')
        ax.set_title('TPOT Percentiles Distribution', fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_file = self.output_dir / "percentiles_distribution.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Percentiles distribution plot saved to {output_file}")
    
    def plot_scaling_analysis(self):
        """Plot how performance scales with TP and DP."""
        if not MATPLOTLIB_AVAILABLE or not self.e2e_data:
            return
        
        # Parse configurations to extract TP and DP
        scaling_data = []
        for config, metrics in self.e2e_data.items():
            # Parse config like "tp4_pp1_dp1"
            parts = config.split('_')
            tp = int(parts[0].replace('tp', ''))
            pp = int(parts[1].replace('pp', ''))
            dp = int(parts[2].replace('dp', ''))
            
            ttft_mean = 0
            tpot_mean = 0
            throughput = 0
            
            if metrics.get('ttft_ms'):
                ttft_mean = metrics['ttft_ms'].get('mean', 0)
            if metrics.get('tpot_ms'):
                tpot_mean = metrics['tpot_ms'].get('mean', 0)
            
            throughput_data = metrics.get('throughput', {})
            if isinstance(throughput_data, dict):
                throughput = throughput_data.get('tokens_per_second', 0)
            else:
                throughput = throughput_data if throughput_data else 0
            
            scaling_data.append({
                'tp': tp,
                'dp': dp,
                'total_gpus': tp * dp,
                'ttft_mean': ttft_mean,
                'tpot_mean': tpot_mean,
                'throughput': throughput,
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
        for gpu, ttft, cfg in zip(total_gpus, ttft_values, configs):
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
        for gpu, tpot, cfg in zip(total_gpus, tpot_values, configs):
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
        for gpu, thru, cfg in zip(total_gpus, throughput_values, configs):
            ax.annotate(cfg, (gpu, thru), textcoords="offset points", 
                       xytext=(0,10), ha='center', fontsize=8)
        
        plt.tight_layout()
        output_file = self.output_dir / "scaling_analysis.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Scaling analysis plot saved to {output_file}")
    
    def plot_efficiency_analysis(self):
        """Plot efficiency metrics (throughput per GPU)."""
        if not MATPLOTLIB_AVAILABLE or not self.e2e_data:
            return
        
        # Parse configurations and calculate efficiency
        efficiency_data = []
        for config, metrics in self.e2e_data.items():
            parts = config.split('_')
            tp = int(parts[0].replace('tp', ''))
            dp = int(parts[2].replace('dp', ''))
            total_gpus = tp * dp
            
            throughput_data = metrics.get('throughput', {})
            if isinstance(throughput_data, dict):
                throughput = throughput_data.get('tokens_per_second', 0)
            else:
                throughput = throughput_data if throughput_data else 0
            
            throughput_per_gpu = throughput / total_gpus if total_gpus > 0 else 0
            
            efficiency_data.append({
                'config': config,
                'tp': tp,
                'dp': dp,
                'total_gpus': total_gpus,
                'throughput': throughput,
                'throughput_per_gpu': throughput_per_gpu
            })
        
        # Sort by config
        efficiency_data.sort(key=lambda x: x['config'])
        
        # Create plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        fig.suptitle('GPU Efficiency Analysis', fontsize=16, fontweight='bold')
        
        configs = [d['config'] for d in efficiency_data]
        
        # Total throughput
        ax = ax1
        throughputs = [d['throughput'] for d in efficiency_data]
        bars = ax.bar(range(len(configs)), throughputs, color='seagreen', alpha=0.8)
        ax.set_xlabel('Configuration', fontweight='bold')
        ax.set_ylabel('Total Throughput (tokens/s)', fontweight='bold')
        ax.set_title('Total Throughput', fontweight='bold')
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, throughputs):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'{val:.0f}', ha='center', va='bottom', fontsize=9)
        
        # Throughput per GPU
        ax = ax2
        throughputs_per_gpu = [d['throughput_per_gpu'] for d in efficiency_data]
        bars = ax.bar(range(len(configs)), throughputs_per_gpu, color='orange', alpha=0.8)
        ax.set_xlabel('Configuration', fontweight='bold')
        ax.set_ylabel('Throughput per GPU (tokens/s/GPU)', fontweight='bold')
        ax.set_title('GPU Efficiency (Throughput per GPU)', fontweight='bold')
        ax.set_xticks(range(len(configs)))
        ax.set_xticklabels(configs, rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, throughputs_per_gpu):
            if val > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                       f'{val:.0f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        output_file = self.output_dir / "efficiency_analysis.png"
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✓ Efficiency analysis plot saved to {output_file}")
    
    def plot_dp_vs_tp_comparison(self):
        """Plot comparison of different DP vs TP strategies."""
        if not MATPLOTLIB_AVAILABLE or not self.e2e_data:
            return
        
        # Group configs by total GPUs
        gpu_groups = {}
        for config, metrics in self.e2e_data.items():
            parts = config.split('_')
            tp = int(parts[0].replace('tp', ''))
            dp = int(parts[2].replace('dp', ''))
            total_gpus = tp * dp
            
            if total_gpus not in gpu_groups:
                gpu_groups[total_gpus] = []
            
            throughput_data = metrics.get('throughput', {})
            if isinstance(throughput_data, dict):
                throughput = throughput_data.get('tokens_per_second', 0)
            else:
                throughput = throughput_data if throughput_data else 0
            
            ttft_mean = 0
            if metrics.get('ttft_ms'):
                ttft_mean = metrics['ttft_ms'].get('mean', 0)
            
            gpu_groups[total_gpus].append({
                'config': config,
                'tp': tp,
                'dp': dp,
                'throughput': throughput,
                'ttft_mean': ttft_mean
            })
        
        # Only create plot if we have multiple configs with same GPU count
        if not any(len(configs) > 1 for configs in gpu_groups.values()):
            print("  Skipping DP vs TP comparison (need multiple configs with same GPU count)")
            return
        
        # Create plot for each GPU count that has multiple configs
        for total_gpus, configs_list in sorted(gpu_groups.items()):
            if len(configs_list) <= 1:
                continue
            
            configs_list.sort(key=lambda x: x['tp'], reverse=True)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
            fig.suptitle(f'DP vs TP Comparison ({total_gpus} GPUs)', fontsize=14, fontweight='bold')
            
            configs = [c['config'] for c in configs_list]
            tp_values = [c['tp'] for c in configs_list]
            dp_values = [c['dp'] for c in configs_list]
            
            # Throughput comparison
            ax = ax1
            throughputs = [c['throughput'] for c in configs_list]
            x = np.arange(len(configs))
            bars = ax.bar(x, throughputs, color='seagreen', alpha=0.8)
            ax.set_xlabel('Configuration', fontweight='bold')
            ax.set_ylabel('Throughput (tokens/s)', fontweight='bold')
            ax.set_title('Throughput Comparison', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f'TP={tp}\nDP={dp}' for tp, dp in zip(tp_values, dp_values)])
            ax.grid(axis='y', alpha=0.3)
            for bar, val in zip(bars, throughputs):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                           f'{val:.0f}', ha='center', va='bottom', fontsize=10)
            
            # TTFT comparison
            ax = ax2
            ttfts = [c['ttft_mean'] for c in configs_list]
            bars = ax.bar(x, ttfts, color='steelblue', alpha=0.8)
            ax.set_xlabel('Configuration', fontweight='bold')
            ax.set_ylabel('TTFT Mean (ms)', fontweight='bold')
            ax.set_title('TTFT Comparison', fontweight='bold')
            ax.set_xticks(x)
            ax.set_xticklabels([f'TP={tp}\nDP={dp}' for tp, dp in zip(tp_values, dp_values)])
            ax.grid(axis='y', alpha=0.3)
            for bar, val in zip(bars, ttfts):
                if val > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(), 
                           f'{val:.1f}', ha='center', va='bottom', fontsize=10)
            
            plt.tight_layout()
            output_file = self.output_dir / f"dp_vs_tp_{total_gpus}gpus.png"
            plt.savefig(output_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"✓ DP vs TP comparison plot saved to {output_file}")
    
    def generate_all_visualizations(self):
        """Generate all available visualizations."""
        print("\n" + "=" * 80)
        print("GENERATING E2E VISUALIZATIONS")
        print("=" * 80 + "\n")
        
        # Always generate text summary
        self.generate_text_summary()
        
        if not MATPLOTLIB_AVAILABLE:
            print("\n⚠ Matplotlib not available. Only text summary was generated.")
            print("Install matplotlib to generate plots: pip install matplotlib")
            return
        
        if not self.e2e_data:
            print("\n⚠ No E2E data found to visualize.")
            return
        
        # Generate all plots
        self.plot_e2e_comparison()
        self.plot_percentiles()
        self.plot_scaling_analysis()
        self.plot_efficiency_analysis()
        self.plot_dp_vs_tp_comparison()
        
        print("\n" + "=" * 80)
        print(f"✓ All visualizations saved to: {self.output_dir}")
        print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description="Visualize E2E benchmark sweep results",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Visualize results from a directory
  python benchmarks/e2e_sweep/visualize_e2e_results.py --results-dir e2e_results/h100_8gpu
  
  # Specify custom output directory for plots
  python benchmarks/e2e_sweep/visualize_e2e_results.py --results-dir e2e_results/h100_8gpu --output-dir my_plots
        """
    )
    
    parser.add_argument(
        '--results-dir',
        type=str,
        required=True,
        help='Directory containing E2E benchmark results (JSON files and summary CSV)'
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
    visualizer = E2EVisualizer(args.results_dir, args.output_dir)
    visualizer.load_results()
    visualizer.generate_all_visualizations()
    
    print("\n✓ Visualization complete!")


if __name__ == "__main__":
    main()


