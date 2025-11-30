#!/bin/bash
# Test script to verify the worker file reading fix

echo "=========================================="
echo "Worker File Reading Fix - Test Script"
echo "=========================================="
echo ""

cd /mnt/disk1/ljm/vllm
source ../qwen/bin/activate

echo "Step 1: Clean up old data"
rm -rf tmp_benchmark benchmark_results/test_fix
echo "  ✓ Cleaned"
echo ""

echo "Step 2: Run benchmark (TP=4, DP=1)"
echo "  This will test the critical fix for LIST format per_layer_stats"
echo ""

python benchmarks/run_parallel_sweep.py \
    --num-gpus 4 \
    --base-config benchmarks/operator_configs/e2e_trace_config.py \
    --output-dir benchmark_results/test_fix \
    --gpu-ids "3,4,5,6" \
    2>&1 | tee test_fix_log.txt

echo ""
echo "Step 3: Check worker files"
if [ -d "tmp_benchmark" ]; then
    echo "  Worker files in tmp_benchmark/:"
    ls -lh tmp_benchmark/*.json | tail -10
else
    echo "  ERROR: tmp_benchmark directory not found!"
fi
echo ""

echo "Step 4: Verify worker file structure"
python3 << 'EOF'
from pathlib import Path
import json

worker_files = sorted(Path('tmp_benchmark').glob('vllm_bench_*.json'), 
                      key=lambda p: p.stat().st_mtime, reverse=True)[:5]

if not worker_files:
    print("  ERROR: No worker files found!")
else:
    for f in worker_files:
        if f.stat().st_size > 1000:
            print(f"\n  {f.name} ({f.stat().st_size} bytes):")
            with open(f) as file:
                data = json.load(file)
            
            per_layer = data.get('per_layer_stats')
            print(f"    per_layer_stats type: {type(per_layer)}")
            
            if isinstance(per_layer, list):
                print(f"    per_layer_stats length (list): {len(per_layer)}")
                if per_layer:
                    print(f"    First record keys: {list(per_layer[0].keys()) if isinstance(per_layer[0], dict) else 'N/A'}")
            elif isinstance(per_layer, dict):
                print(f"    per_layer_stats length (dict): {len(per_layer)}")
            
            summary = data.get('summary_stats', {})
            print(f"    summary_stats keys: {list(summary.keys())}")
EOF
echo ""

echo "Step 5: Check final results"
python3 << 'EOF'
import json
from pathlib import Path

result_files = list(Path('benchmark_results/test_fix').glob('*_results.json'))
if not result_files:
    print("  ERROR: No result files found!")
else:
    for result_file in result_files:
        print(f"\n  {result_file.name}:")
        with open(result_file) as f:
            data = json.load(f)
        
        op = data.get('operator_performance', {})
        per_layer = op.get('per_layer_stats', {})
        
        print(f"    per_layer_stats type: {type(per_layer)}")
        if isinstance(per_layer, list):
            print(f"    ✓ per_layer_stats length (list): {len(per_layer)}")
            if per_layer:
                print(f"    ✓ Sample record: {list(per_layer[0].keys())[:5] if isinstance(per_layer[0], dict) else 'N/A'}")
        elif isinstance(per_layer, dict):
            print(f"    per_layer_stats keys (dict): {len(per_layer)}")
        else:
            print(f"    ✗ per_layer_stats is EMPTY: {per_layer}")
        
        summary = op.get('summary_stats', {})
        if summary:
            print(f"    ✓ summary_stats keys: {list(summary.keys())}")
        else:
            print(f"    ✗ summary_stats is EMPTY")
        
        print(f"")
        print(f"    E2E metrics:")
        e2e = data.get('e2e_metrics_aggregated', {})
        if e2e:
            print(f"      num_requests: {e2e.get('num_requests')}")
            print(f"      throughput: {e2e.get('throughput_tokens_per_sec', 0):.2f} tokens/s")
        else:
            print(f"      No E2E metrics found")
EOF
echo ""

echo "Step 6: Check key logs"
echo "  Looking for diagnostic messages..."
grep -E "Attempting to read worker files|Successfully loaded|per_layer_stats type|per_layer_stats length|Operator performance data prepared" test_fix_log.txt | tail -20
echo ""

echo "=========================================="
echo "Test Complete!"
echo "=========================================="
echo ""
echo "Expected Success Indicators:"
echo "  ✓ Worker files exist and are >100KB"
echo "  ✓ per_layer_stats type is <class 'list'>"
echo "  ✓ per_layer_stats length is >1000"
echo "  ✓ summary_stats contains keys like 'attention', 'linear', etc."
echo "  ✓ Final JSON has operator_performance data"
echo ""
echo "If you see empty operator_performance, check test_fix_log.txt for:"
echo "  - 'Failed to load operator results'"
echo "  - File size warnings"
echo "  - Data structure mismatches"
echo ""


