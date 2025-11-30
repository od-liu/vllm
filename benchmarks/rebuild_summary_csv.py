#!/usr/bin/env python3
"""
Rebuild summary.csv from JSON result files in a directory.
"""

import json
import csv
import os
import sys
from pathlib import Path
import re

def extract_config_from_filename(filename: str):
    """Extract TP, PP, DP from filename like 'tp4_pp1_dp1_results.json'."""
    match = re.match(r'tp(\d+)_pp(\d+)_dp(\d+)_results\.json', filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return None, None, None

def extract_metrics_from_json(json_path: str) -> dict:
    """Extract E2E metrics from a result JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Look for e2e_metrics in the JSON
        if 'e2e_metrics' in data:
            e2e = data['e2e_metrics']
            # Also get execution time from metadata if available
            exec_time = data.get('metadata', {}).get('total_time_seconds', None)
            
            metrics = {
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
            
            if exec_time is not None:
                metrics['execution_time'] = exec_time
                
            return metrics
    except Exception as e:
        print(f"Warning: Failed to extract metrics from {json_path}: {e}")
        return {}

def rebuild_summary_csv(results_dir: str, include_failed: bool = True):
    """Rebuild summary.csv from JSON result files."""
    results_dir = Path(results_dir)
    
    # Find all result JSON files
    json_files = list(results_dir.glob("*_results.json"))
    
    if not json_files:
        print(f"No result files found in {results_dir}")
        return False
    
    print(f"Found {len(json_files)} result file(s)")
    
    # Build rows from JSON files
    rows = []
    for json_file in sorted(json_files):
        tp, pp, dp = extract_config_from_filename(json_file.name)
        if tp is None:
            print(f"Skipping {json_file.name} (can't parse config)")
            continue
        
        config_name = f"tp{tp}_pp{pp}_dp{dp}"
        print(f"Processing {config_name}...")
        
        metrics = extract_metrics_from_json(str(json_file))
        
        row = {
            'config': config_name,
            'tp': tp,
            'pp': pp,
            'dp': dp,
            'status': 'success',
        }
        row.update(metrics)
        rows.append(row)
    
    # Optionally add failed configs from old CSV
    if include_failed:
        old_csv = results_dir / "summary.csv"
        if old_csv.exists():
            try:
                with open(old_csv, 'r') as f:
                    reader = csv.DictReader(f)
                    for old_row in reader:
                        if old_row.get('status') == 'failed':
                            # Check if this config already exists in rows
                            config = old_row.get('config', '')
                            if not any(r['config'] == config for r in rows):
                                print(f"Adding failed config from old CSV: {config}")
                                rows.append(old_row)
            except Exception as e:
                print(f"Warning: Could not read old CSV: {e}")
    
    if not rows:
        print("No data to write")
        return False
    
    # Sort rows by tp, pp, dp
    rows.sort(key=lambda r: (int(r['tp']), int(r['pp']), int(r['dp'])))
    
    # Collect all unique fieldnames
    all_fieldnames = set()
    for row in rows:
        all_fieldnames.update(row.keys())
    
    # Sort fieldnames: basic fields first, then metrics, then error
    basic_fields = ['config', 'tp', 'pp', 'dp', 'status', 'execution_time']
    metric_fields = sorted([f for f in all_fieldnames if f not in basic_fields and f != 'error'])
    error_field = ['error'] if 'error' in all_fieldnames else []
    fieldnames = [f for f in basic_fields if f in all_fieldnames] + metric_fields + error_field
    
    # Backup old CSV if it exists
    summary_csv = results_dir / "summary.csv"
    if summary_csv.exists():
        backup_path = summary_csv.with_suffix('.csv.old')
        summary_csv.rename(backup_path)
        print(f"Backed up old CSV to {backup_path.name}")
    
    # Write new CSV
    with open(summary_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    
    print(f"\n✓ Successfully created {summary_csv}")
    print(f"  Total rows: {len(rows)}")
    print(f"  Total fields: {len(fieldnames)}")
    print(f"  Successful configs: {sum(1 for r in rows if r['status'] == 'success')}")
    print(f"  Failed configs: {sum(1 for r in rows if r['status'] == 'failed')}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python rebuild_summary_csv.py <results_directory> [--skip-failed]")
        print("Example: python rebuild_summary_csv.py benchmark_results/test_4gpu")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    include_failed = '--skip-failed' not in sys.argv
    
    if not os.path.isdir(results_dir):
        print(f"Error: Directory not found: {results_dir}")
        sys.exit(1)
    
    success = rebuild_summary_csv(results_dir, include_failed=include_failed)
    sys.exit(0 if success else 1)

