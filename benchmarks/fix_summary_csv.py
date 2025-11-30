#!/usr/bin/env python3
"""
Fix incomplete summary.csv files by extracting metrics from JSON result files.
"""

import json
import csv
import os
import sys
from pathlib import Path

def extract_metrics_from_json(json_path: str) -> dict:
    """Extract E2E metrics from a result JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Look for e2e_metrics in the JSON
        if 'e2e_metrics' in data:
            e2e = data['e2e_metrics']
            return {
                'num_requests': e2e.get('num_requests', 0),
                'total_generation_tokens': e2e.get('total_generation_tokens', 0),
                'ttft_ms_mean': e2e.get('ttft_ms', {}).get('mean'),
                'ttft_ms_p50': e2e.get('ttft_ms', {}).get('p50'),
                'ttft_ms_p90': e2e.get('ttft_ms', {}).get('p90'),
                'ttft_ms_p99': e2e.get('ttft_ms', {}).get('p99'),
                'tpot_ms_mean': e2e.get('tpot_ms', {}).get('mean'),
                'tpot_ms_p50': e2e.get('tpot_ms', {}).get('p50'),
                'tpot_ms_p90': e2e.get('tpot_ms', {}).get('p90'),
                'tpot_ms_p99': e2e.get('tpot_ms', {}).get('p99'),
                'e2e_latency_ms_mean': e2e.get('e2e_latency_ms', {}).get('mean'),
                'e2e_latency_ms_p50': e2e.get('e2e_latency_ms', {}).get('p50'),
                'e2e_latency_ms_p90': e2e.get('e2e_latency_ms', {}).get('p90'),
                'e2e_latency_ms_p99': e2e.get('e2e_latency_ms', {}).get('p99'),
                'queued_time_ms_mean': e2e.get('queued_time_ms', {}).get('mean'),
                'queued_time_ms_p50': e2e.get('queued_time_ms', {}).get('p50'),
                'queued_time_ms_p90': e2e.get('queued_time_ms', {}).get('p90'),
                'queued_time_ms_p99': e2e.get('queued_time_ms', {}).get('p99'),
            }
    except Exception as e:
        print(f"Warning: Failed to extract metrics from {json_path}: {e}")
        return {}

def fix_summary_csv(results_dir: str):
    """Fix summary.csv by adding metrics from JSON result files."""
    results_dir = Path(results_dir)
    summary_csv = results_dir / "summary.csv"
    
    if not summary_csv.exists():
        print(f"Error: {summary_csv} not found")
        return False
    
    # Read existing CSV
    with open(summary_csv, 'r') as f:
        reader = csv.DictReader(f)
        existing_rows = list(reader)
    
    print(f"Found {len(existing_rows)} existing rows in summary.csv")
    
    # Update rows with metrics from JSON files
    updated_rows = []
    for row in existing_rows:
        config = row['config']
        json_path = results_dir / f"{config}_results.json"
        
        if json_path.exists() and row['status'] == 'success':
            print(f"Extracting metrics from {json_path.name}...")
            metrics = extract_metrics_from_json(str(json_path))
            row.update(metrics)
        
        updated_rows.append(row)
    
    # Collect all unique fieldnames
    all_fieldnames = set()
    for row in updated_rows:
        all_fieldnames.update(row.keys())
    
    # Sort fieldnames: basic fields first, then metrics, then error
    basic_fields = ['config', 'tp', 'pp', 'dp', 'status', 'execution_time']
    metric_fields = sorted([f for f in all_fieldnames if f not in basic_fields and f != 'error'])
    error_field = ['error'] if 'error' in all_fieldnames else []
    fieldnames = [f for f in basic_fields if f in all_fieldnames] + metric_fields + error_field
    
    # Write updated CSV
    backup_path = summary_csv.with_suffix('.csv.backup')
    summary_csv.rename(backup_path)
    print(f"Backed up original CSV to {backup_path.name}")
    
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(updated_rows)
    
    print(f"✓ Successfully updated {summary_csv}")
    print(f"  Total rows: {len(updated_rows)}")
    print(f"  Fields: {len(fieldnames)}")
    return True

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python fix_summary_csv.py <results_directory>")
        print("Example: python fix_summary_csv.py benchmark_results/test_4gpu")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    if not os.path.isdir(results_dir):
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)
    
    success = fix_summary_csv(results_dir)
    sys.exit(0 if success else 1)

