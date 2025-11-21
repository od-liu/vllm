#!/usr/bin/env python3
"""Create a sparse test trace file from original trace."""

import json
import argparse
from pathlib import Path

def create_sparse_trace(
    input_file: str,
    output_file: str,
    time_range_minutes: tuple[float, float] = (0, 2),
    target_request_count: int = 30,
):
    """
    Create a sparse test trace by sampling from original trace.
    
    Args:
        input_file: Path to original trace JSONL
        output_file: Path to output test trace JSONL
        time_range_minutes: Time range (start_min, end_min)
        target_request_count: Target number of requests in output
    """
    start_sec = time_range_minutes[0] * 60
    end_sec = time_range_minutes[1] * 60
    
    # Load requests in time range
    requests_in_range = []
    with open(input_file, 'r') as f:
        for line in f:
            req = json.loads(line)
            if start_sec <= req['timestamp'] < end_sec:
                requests_in_range.append(req)
    
    print(f"Found {len(requests_in_range)} requests in range [{start_sec}s, {end_sec}s)")
    
    # Calculate sampling interval
    if len(requests_in_range) <= target_request_count:
        # Keep all if already sparse enough
        sampled = requests_in_range
        print(f"Keeping all {len(sampled)} requests")
    else:
        # Sample evenly
        interval = len(requests_in_range) // target_request_count
        sampled = requests_in_range[::interval][:target_request_count]
        print(f"Sampled {len(sampled)} requests (1 every {interval} requests)")
    
    # Write to output file
    with open(output_file, 'w') as f:
        for req in sampled:
            f.write(json.dumps(req) + '\n')
    
    print(f"\nCreated test trace: {output_file}")
    print(f"  Time range: {start_sec}s - {end_sec}s ({time_range_minutes[1]-time_range_minutes[0]} minutes)")
    print(f"  Total requests: {len(sampled)}")
    if end_sec > start_sec:
        print(f"  Avg rate: {len(sampled)/(end_sec-start_sec):.2f} req/s")
    
    # Show first and last requests
    if sampled:
        print(f"\nFirst request: chat_id={sampled[0]['chat_id']}, timestamp={sampled[0]['timestamp']:.3f}s")
        print(f"Last request: chat_id={sampled[-1]['chat_id']}, timestamp={sampled[-1]['timestamp']:.3f}s")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create sparse test trace")
    parser.add_argument(
        "--input",
        default="qwen-bailian-usagetraces-anon/qwen_traceB_blksz_16.jsonl",
        help="Input trace file"
    )
    parser.add_argument(
        "--output",
        default="qwen-bailian-usagetraces-anon/qwen_traceB_test_2min.jsonl",
        help="Output test trace file"
    )
    parser.add_argument(
        "--time-range",
        type=str,
        default="0,2",
        help="Time range in minutes (start,end)"
    )
    parser.add_argument(
        "--count",
        type=int,
        default=30,
        help="Target number of requests"
    )
    
    args = parser.parse_args()
    
    # Parse time range
    start_min, end_min = map(float, args.time_range.split(','))
    
    create_sparse_trace(
        input_file=args.input,
        output_file=args.output,
        time_range_minutes=(start_min, end_min),
        target_request_count=args.count,
    )

