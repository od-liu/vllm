#!/usr/bin/env python3
"""
Generate a sparse 5-minute trace file with smaller output lengths.

Characteristics:
- Duration: 5 minutes (300 seconds)
- Output length: smaller (5-120 tokens, avg ~40)
- Request density: ~10-15 requests per minute (50-75 total)
- Input length: moderate (100-2000 tokens)
"""

import json
import random
import numpy as np

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)

def generate_hash_ids(input_length, output_length):
    """Generate hash_ids list based on input and output length."""
    # Approximate: 1.3 tokens per hash_id for Chinese text
    num_hash_ids = int((input_length + output_length) / 1.3)
    return list(range(num_hash_ids))

def generate_requests(num_requests=600, duration=300):
    """
    Generate trace requests with sparse timing.
    
    Args:
        num_requests: Number of requests to generate (default: 60 for ~12 req/min)
        duration: Total duration in seconds (default: 300 = 5 minutes)
    """
    requests = []
    
    # Generate timestamps with some clustering (realistic traffic pattern)
    # Use exponential distribution for inter-arrival times
    timestamps = []
    current_time = 0.0
    
    for i in range(num_requests):
        if i == 0:
            current_time = 0.0
        else:
            # Average 5 seconds between requests, with variation
            inter_arrival = np.random.exponential(duration / num_requests)
            current_time += inter_arrival
        
        if current_time < duration:
            timestamps.append(round(current_time, 3))
    
    timestamps.sort()
    
    # Request type distribution
    request_types = ['api'] * 80 + ['text'] * 20  # 80% api, 20% text
    
    hash_id_counter = 0
    
    for i, timestamp in enumerate(timestamps):
        # Input length: moderate, with some variety
        # Biased towards shorter inputs (200-800), with some longer ones
        if random.random() < 0.7:
            input_length = random.randint(150, 800)
        elif random.random() < 0.9:
            input_length = random.randint(800, 1500)
        else:
            input_length = random.randint(1500, 2500)
        
        # Output length: SMALLER than original
        # Most outputs are short (10-60 tokens), some medium (60-120)
        rand = random.random()
        if rand < 0.5:
            output_length = random.randint(5, 40)  # Very short
        elif rand < 0.8:
            output_length = random.randint(40, 80)  # Short
        elif rand < 0.95:
            output_length = random.randint(80, 120)  # Medium
        else:
            output_length = random.randint(120, 200)  # Occasionally longer
        
        # Generate hash_ids
        num_hash_ids = int((input_length + output_length) / 1.5)
        hash_ids = list(range(hash_id_counter, hash_id_counter + num_hash_ids))
        hash_id_counter += num_hash_ids
        
        request = {
            "chat_id": i,
            "parent_chat_id": -1,
            "timestamp": timestamp,
            "input_length": input_length,
            "output_length": output_length,
            "type": random.choice(request_types),
            "turn": 1,
            "hash_ids": hash_ids
        }
        
        requests.append(request)
    
    return requests

def save_trace(requests, output_file):
    """Save requests to JSONL file."""
    with open(output_file, 'w') as f:
        for req in requests:
            f.write(json.dumps(req) + '\n')
    
    print(f"✓ Generated {len(requests)} requests")
    print(f"✓ Saved to: {output_file}")
    
    # Print statistics
    timestamps = [r['timestamp'] for r in requests]
    input_lengths = [r['input_length'] for r in requests]
    output_lengths = [r['output_length'] for r in requests]
    
    print(f"\n📊 Statistics:")
    print(f"  Duration: {timestamps[-1]:.1f} seconds ({timestamps[-1]/60:.1f} minutes)")
    print(f"  Requests: {len(requests)}")
    print(f"  Request rate: {len(requests)/(timestamps[-1]/60):.1f} req/min")
    print(f"  Input length:  min={min(input_lengths)}, max={max(input_lengths)}, avg={np.mean(input_lengths):.0f}")
    print(f"  Output length: min={min(output_lengths)}, max={max(output_lengths)}, avg={np.mean(output_lengths):.0f}")
    
    # Show first few requests
    print(f"\n📝 First 3 requests preview:")
    for i in range(min(3, len(requests))):
        req = requests[i]
        print(f"  {i}: t={req['timestamp']:.3f}s, input={req['input_length']}, output={req['output_length']}, type={req['type']}")

if __name__ == "__main__":
    # Generate sparse 5-minute trace
    num_requests = 6000
    requests = generate_requests(num_requests, duration=300)
    
    output_file = f"qwen_trace_5min_{num_requests}prompts.jsonl"
    save_trace(requests, output_file)
    
    print(f"\n✅ Done! Use this trace with:")
    print(f"   trace_file_path='qwen-bailian-usagetraces-anon/{output_file}'")
    print(f"   trace_time_range_minutes=(0, 5)")

