#!/usr/bin/env python3
"""
从trace文件中提取前N分钟的prompts，并进行采样。

使用方法:
    python sample_trace.py --input qwen_traceB_blksz_16.jsonl --output sampled_trace.jsonl --k 10 --minutes 5
    
参数:
    --input: 输入的JSONL文件路径
    --output: 输出的JSONL文件路径
    --k: 采样率，每k个请求保留1个 (默认: 10)
    --minutes: 提取前N分钟的数据 (默认: 5)
"""

import json
import argparse
import numpy as np


def sample_trace(input_file, output_file, k=10, time_limit_minutes=5):
    """
    从trace文件中采样前N分钟的请求。
    
    Args:
        input_file: 输入JSONL文件路径
        output_file: 输出JSONL文件路径
        k: 采样率，每k个请求保留1个
        time_limit_minutes: 时间限制（分钟）
    """
    time_limit_seconds = time_limit_minutes * 60
    
    # 读取前N分钟内的所有请求
    all_requests = []
    
    print(f"📖 正在读取 {input_file}...")
    with open(input_file, 'r') as f:
        for line in f:
            req = json.loads(line.strip())
            if req['timestamp'] <= time_limit_seconds:
                all_requests.append(req)
            else:
                break  # 已经超过时间限制，停止读取
    
    print(f"✓ 前 {time_limit_minutes} 分钟内共有 {len(all_requests)} 个请求")
    
    # 按时间戳排序（确保按时间顺序）
    all_requests.sort(key=lambda x: x['timestamp'])
    
    # 采样：每k个保留1个
    sampled_requests = []
    for i in range(0, len(all_requests), k):
        sampled_requests.append(all_requests[i])
    
    print(f"✓ 采样率 k={k}，保留 {len(sampled_requests)} 个请求")
    
    # 重新分配chat_id（从0开始连续编号）
    for i, req in enumerate(sampled_requests):
        req['chat_id'] = i
    
    # 保存到输出文件
    print(f"💾 正在保存到 {output_file}...")
    with open(output_file, 'w') as f:
        for req in sampled_requests:
            f.write(json.dumps(req) + '\n')
    
    print(f"✅ 完成！")
    
    # 打印统计信息
    print_statistics(sampled_requests, time_limit_minutes)
    
    return sampled_requests


def print_statistics(requests, time_minutes):
    """打印采样后的统计信息。"""
    if not requests:
        print("⚠️  警告：没有请求被采样")
        return
    
    timestamps = [r['timestamp'] for r in requests]
    input_lengths = [r['input_length'] for r in requests]
    output_lengths = [r['output_length'] for r in requests]
    
    print(f"\n📊 采样结果统计:")
    print(f"  时间范围: 0 - {time_minutes} 分钟")
    print(f"  请求总数: {len(requests)}")
    print(f"  请求密度: {len(requests) / time_minutes:.1f} 请求/分钟")
    print(f"  实际时长: {timestamps[-1]:.1f} 秒 ({timestamps[-1]/60:.1f} 分钟)")
    print(f"\n  输入长度统计:")
    print(f"    最小值: {min(input_lengths)}")
    print(f"    最大值: {max(input_lengths)}")
    print(f"    平均值: {np.mean(input_lengths):.0f}")
    print(f"    中位数: {np.median(input_lengths):.0f}")
    print(f"\n  输出长度统计:")
    print(f"    最小值: {min(output_lengths)}")
    print(f"    最大值: {max(output_lengths)}")
    print(f"    平均值: {np.mean(output_lengths):.0f}")
    print(f"    中位数: {np.median(output_lengths):.0f}")
    
    # 显示前几个请求
    print(f"\n📝 前 5 个请求预览:")
    for i in range(min(5, len(requests))):
        req = requests[i]
        print(f"  [{i}] t={req['timestamp']:.3f}s, "
              f"input={req['input_length']}, "
              f"output={req['output_length']}, "
              f"type={req['type']}")


def main():
    parser = argparse.ArgumentParser(
        description="从trace文件中采样前N分钟的请求",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 每10个请求保留1个，提取前5分钟
  python sample_trace.py --input qwen_traceB_blksz_16.jsonl --output sampled_5min_k10.jsonl --k 10 --minutes 5
  
  # 每20个请求保留1个，提取前3分钟
  python sample_trace.py --input qwen_traceB_blksz_16.jsonl --output sampled_3min_k20.jsonl --k 20 --minutes 3
  
  # 每5个请求保留1个，提取前10分钟
  python sample_trace.py --input qwen_traceB_blksz_16.jsonl --output sampled_10min_k5.jsonl --k 5 --minutes 10
        """
    )
    
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='输入的JSONL trace文件路径'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='输出的JSONL文件路径'
    )
    parser.add_argument(
        '--k',
        type=int,
        default=10,
        help='采样率：每k个请求保留1个 (默认: 10)'
    )
    parser.add_argument(
        '--minutes',
        type=int,
        default=5,
        help='提取前N分钟的数据 (默认: 5)'
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if args.k < 1:
        print("❌ 错误: k 必须大于等于 1")
        return 1
    
    if args.minutes < 1:
        print("❌ 错误: minutes 必须大于等于 1")
        return 1
    
    # 执行采样
    print("=" * 80)
    print(f"Trace文件采样工具")
    print("=" * 80)
    print(f"输入文件: {args.input}")
    print(f"输出文件: {args.output}")
    print(f"采样率: 每 {args.k} 个保留 1 个")
    print(f"时间范围: 前 {args.minutes} 分钟")
    print("=" * 80)
    
    try:
        sample_trace(
            input_file=args.input,
            output_file=args.output,
            k=args.k,
            time_limit_minutes=args.minutes
        )
        return 0
    except FileNotFoundError:
        print(f"❌ 错误: 找不到输入文件 {args.input}")
        return 1
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

