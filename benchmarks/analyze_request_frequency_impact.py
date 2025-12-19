#!/usr/bin/env python3
"""
分析不同请求频次对TTFT、TPOT等指标的影响。

从不同prompts数量的实验数据中提取指标，绘制随请求频次变化的曲线。
"""

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not found. Install with: pip install matplotlib")


def extract_prompt_count_from_dir(dir_name: str) -> Optional[int]:
    """从目录名中提取prompts数量，如 'h100_4gpu_200prompts' -> 200"""
    match = re.search(r'(\d+)prompts', dir_name)
    if match:
        return int(match.group(1))
    return None


def extract_config_from_filename(filename: str) -> Optional[str]:
    """从文件名中提取配置，如 'tp1_pp1_dp4_e2e.json' -> 'tp1_dp4'"""
    match = re.search(r'tp(\d+)_pp\d+_dp(\d+)_e2e', filename)
    if match:
        return f"tp{match.group(1)}_dp{match.group(2)}"
    return None


def load_e2e_metrics(json_path: Path) -> Optional[Dict]:
    """从JSON文件中加载E2E指标"""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # 优先使用聚合指标
        if 'e2e_metrics_aggregated' in data and data['e2e_metrics_aggregated']:
            return data['e2e_metrics_aggregated']
        elif 'e2e_metrics' in data and data['e2e_metrics']:
            return data['e2e_metrics']
        return None
    except Exception as e:
        print(f"Error loading {json_path}: {e}")
        return None


def collect_data_from_directories(base_dir: Path, prompt_dirs: List[str]) -> Dict:
    """
    从多个实验目录中收集数据
    
    Returns:
        Dict[config][metric][stat] = [(request_freq, value), ...]
    """
    data = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    
    for prompt_dir_name in prompt_dirs:
        prompt_dir = base_dir / prompt_dir_name
        if not prompt_dir.exists():
            print(f"Warning: Directory {prompt_dir} does not exist, skipping")
            continue
        
        # 提取请求数量
        prompt_count = extract_prompt_count_from_dir(prompt_dir_name)
        if prompt_count is None:
            print(f"Warning: Could not extract prompt count from {prompt_dir_name}")
            continue
        
        # 计算请求频次 (requests per minute, 假设是5分钟的实验)
        request_freq = prompt_count / 5.0  # requests per minute
        
        print(f"\nProcessing {prompt_dir_name}: {prompt_count} prompts -> {request_freq:.1f} req/min")
        
        # 查找所有JSON文件
        json_files = list(prompt_dir.glob("*_e2e.json"))
        if not json_files:
            print(f"  No JSON files found in {prompt_dir}")
            continue
        
        for json_file in json_files:
            config = extract_config_from_filename(json_file.name)
            if config is None:
                print(f"  Warning: Could not extract config from {json_file.name}")
                continue
            
            metrics = load_e2e_metrics(json_file)
            if metrics is None:
                print(f"  Warning: No metrics found in {json_file.name}")
                continue
            
            # 提取各种指标
            for metric_name in ['ttft_ms', 'tpot_ms', 'e2e_latency_ms', 'queued_time_ms']:
                if metric_name in metrics and metrics[metric_name]:
                    metric_data = metrics[metric_name]
                    for stat in ['mean', 'p50', 'p90', 'p99']:
                        if stat in metric_data:
                            value = metric_data[stat]
                            if value is not None:
                                data[config][metric_name][stat].append((request_freq, value))
            
            # 提取吞吐量
            if 'throughput' in metrics and metrics['throughput']:
                if 'tokens_per_second' in metrics['throughput']:
                    tps = metrics['throughput']['tokens_per_second']
                    data[config]['throughput_tokens_per_second']['mean'].append((request_freq, tps))
            
            print(f"  Loaded {config} from {json_file.name}")
    
    return data


def plot_metrics_vs_frequency(data: Dict, output_dir: Path):
    """绘制指标随请求频次变化的曲线"""
    if not MATPLOTLIB_AVAILABLE:
        print("matplotlib not available, skipping plots")
        return
    
    # 为每个指标创建图表
    metrics_to_plot = {
        'ttft_ms': 'TTFT (ms)',
        'tpot_ms': 'TPOT (ms)',
        'e2e_latency_ms': 'E2E Latency (ms)',
        'queued_time_ms': 'Queued Time (ms)',
        'throughput_tokens_per_second': 'Throughput (tokens/s)'
    }
    
    stats_to_plot = ['mean', 'p50', 'p90', 'p99']
    
    for metric_name, metric_label in metrics_to_plot.items():
        if metric_name not in data or not any(data[metric_name].values()):
            continue
        
        # 创建图表
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'{metric_label} vs Request Frequency', fontsize=16, fontweight='bold')
        
        for idx, stat in enumerate(stats_to_plot):
            ax = axes[idx // 2, idx % 2]
            
            # 为每个配置绘制曲线
            for config in sorted(data.keys()):
                if metric_name not in data[config] or stat not in data[config][metric_name]:
                    continue
                
                points = data[config][metric_name][stat]
                if not points:
                    continue
                
                # 按请求频次排序
                points.sort(key=lambda x: x[0])
                freqs = [p[0] for p in points]
                values = [p[1] for p in points]
                
                # 绘制曲线
                ax.plot(freqs, values, marker='o', label=config, linewidth=2, markersize=6)
            
            ax.set_xlabel('Request Frequency (requests/min)', fontsize=11)
            ax.set_ylabel(f'{metric_label} ({stat})', fontsize=11)
            ax.set_title(f'{stat.upper()}', fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=9)
        
        plt.tight_layout()
        output_file = output_dir / f'{metric_name}_vs_frequency.png'
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {output_file}")
        plt.close()
    
    # 创建综合对比图（所有指标在同一图中，使用mean值）
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Key Metrics vs Request Frequency (Mean Values)', fontsize=16, fontweight='bold')
    
    key_metrics = [
        ('ttft_ms', 'TTFT (ms)', axes[0, 0]),
        ('tpot_ms', 'TPOT (ms)', axes[0, 1]),
        ('e2e_latency_ms', 'E2E Latency (ms)', axes[0, 2]),
        ('queued_time_ms', 'Queued Time (ms)', axes[1, 0]),
        ('throughput_tokens_per_second', 'Throughput (tokens/s)', axes[1, 1])
    ]
    
    # 隐藏最后一个空白子图
    axes[1, 2].axis('off')
    
    for metric_name, metric_label, ax in key_metrics:
        for config in sorted(data.keys()):
            if metric_name not in data[config] or 'mean' not in data[config][metric_name]:
                continue
            
            points = data[config][metric_name]['mean']
            if not points:
                continue
            
            points.sort(key=lambda x: x[0])
            freqs = [p[0] for p in points]
            values = [p[1] for p in points]
            
            ax.plot(freqs, values, marker='o', label=config, linewidth=2.5, markersize=8)
        
        ax.set_xlabel('Request Frequency (requests/min)', fontsize=12)
        ax.set_ylabel(metric_label, fontsize=12)
        ax.set_title(metric_label, fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10)
    
    plt.tight_layout()
    output_file = output_dir / 'all_metrics_vs_frequency.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {output_file}")
    plt.close()


def generate_summary_table(data: Dict, output_dir: Path):
    """生成汇总表格"""
    output_file = output_dir / 'request_frequency_analysis.txt'
    
    with open(output_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Request Frequency Impact Analysis\n")
        f.write("=" * 80 + "\n\n")
        
        for config in sorted(data.keys()):
            f.write(f"\nConfiguration: {config}\n")
            f.write("-" * 80 + "\n")
            
            for metric_name in ['ttft_ms', 'tpot_ms', 'e2e_latency_ms', 'queued_time_ms']:
                if metric_name not in data[config]:
                    continue
                
                f.write(f"\n{metric_name}:\n")
                if 'mean' in data[config][metric_name]:
                    points = sorted(data[config][metric_name]['mean'], key=lambda x: x[0])
                    f.write("  Request Freq (req/min) | Mean Value\n")
                    f.write("  " + "-" * 40 + "\n")
                    for freq, value in points:
                        f.write(f"  {freq:>20.1f} | {value:>10.2f}\n")
    
    print(f"Saved summary: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='分析不同请求频次对E2E指标的影响'
    )
    parser.add_argument(
        '--base-dir',
        type=str,
        default='e2e_results',
        help='基准目录路径（包含各个prompts目录）'
    )
    parser.add_argument(
        '--prompt-dirs',
        type=str,
        nargs='+',
        default=[
            'h100_4gpu_200prompts',
            'h100_4gpu_483prompts',
            'h100_4gpu_4851prompts',
            'h100_4gpu_804prompts',
            'h100_4gpu_1197prompts',
            'h100_4gpu_1592prompts',
            'h100_4gpu_1991prompts',
            'h100_4gpu_1991prompts_1',
            'h100_4gpu_4034prompts',
            'h100_4gpu_2409prompts',
            'h100_4gpu_2801prompts',
            'h100_4gpu_1592prompts_p'
        ],
        help='要分析的prompts目录列表'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='输出目录（默认：base_dir/request_frequency_analysis）'
    )
    
    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)
    if not base_dir.exists():
        print(f"Error: Base directory {base_dir} does not exist")
        return
    
    output_dir = Path(args.output_dir) if args.output_dir else base_dir / 'request_frequency_analysis'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Analyzing {len(args.prompt_dirs)} directories...")
    
    # 收集数据
    data = collect_data_from_directories(base_dir, args.prompt_dirs)
    
    if not data:
        print("No data collected!")
        return
    
    print(f"\nCollected data for {len(data)} configurations:")
    for config in sorted(data.keys()):
        print(f"  {config}")
    
    # 重新组织数据结构以便绘图（数据已经是正确格式，直接使用）
    plot_data_dict = data
    
    # 绘制图表
    if MATPLOTLIB_AVAILABLE:
        plot_metrics_vs_frequency(plot_data_dict, output_dir)
    
    # 生成汇总表格
    generate_summary_table(plot_data_dict, output_dir)
    
    print("\nAnalysis complete!")


if __name__ == '__main__':
    main()
