#!/bin/bash
# Test script for Two-Phase LLM Reinitialization Fix

echo "=========================================="
echo "Two-Phase LLM Reinitialization Test"
echo "=========================================="
echo ""

cd /mnt/disk1/ljm/vllm
source ../qwen/bin/activate

echo "清理旧数据..."
rm -rf tmp_benchmark benchmark_results/test_two_phase
echo "✓ 清理完成"
echo ""

echo "运行Two-Phase Benchmark测试..."
echo "配置: TP=4, DP=1, trace-based, two-phase mode"
echo ""

python benchmarks/run_parallel_sweep.py \
    --num-gpus 4 \
    --base-config benchmarks/operator_configs/e2e_trace_config.py \
    --output-dir benchmark_results/test_two_phase \
    --gpu-ids "3,4,5,6" \
    --gpu-type 'h100' \
    2>&1 | tee test_two_phase_log.txt

echo ""
echo "=========================================="
echo "检查结果"
echo "=========================================="
echo ""

echo "1. 检查Phase 1后是否无operator数据（重新初始化前）"
echo "   查找日志中的关键信息..."
grep -A 5 "PHASE 1: E2E Metrics Collection" test_two_phase_log.txt | head -10
echo ""

echo "2. 检查LLM重新初始化日志"
grep -A 10 "Reinitializing LLM for Phase 2" test_two_phase_log.txt | head -15
echo ""

echo "3. 检查Phase 2 Warmup日志"
grep -A 5 "Executing warmup for Phase 2" test_two_phase_log.txt | head -10
echo ""

echo "4. 检查Phase 2 Operator Profiling日志"
grep -A 5 "Operator benchmarking enabled" test_two_phase_log.txt | head -10
echo ""

echo "5. 检查临时文件目录结构"
if [ -d "tmp_benchmark/tp4_dp1" ]; then
    echo "✓ 找到配置目录: tmp_benchmark/tp4_dp1/"
    ls -lh tmp_benchmark/tp4_dp1/*.json 2>/dev/null | head -10
else
    echo "✗ 未找到配置目录"
fi
echo ""

echo "6. 检查worker文件内容"
python3 << 'EOF'
from pathlib import Path
import json

worker_files = list(Path('tmp_benchmark/tp4_dp1').glob('*.json'))
if not worker_files:
    print("✗ 未找到worker文件")
else:
    for f in sorted(worker_files)[:4]:
        if f.stat().st_size > 1000:
            with open(f) as file:
                data = json.load(file)
            per_layer = data.get('per_layer_stats', [])
            print(f"✓ {f.name}: {f.stat().st_size} bytes, {len(per_layer)} records")
        else:
            print(f"✗ {f.name}: {f.stat().st_size} bytes (too small)")
EOF
echo ""

echo "7. 检查最终结果文件"
python3 << 'EOF'
import json
from pathlib import Path

result_file = Path('benchmark_results/test_two_phase/tp4_pp1_dp1_results.json')
if not result_file.exists():
    print("✗ 结果文件不存在")
else:
    with open(result_file) as f:
        data = json.load(f)
    
    print(f"✓ 结果文件存在")
    
    # Check E2E metrics
    e2e = data.get('e2e_metrics_aggregated', {})
    if e2e:
        print(f"  E2E Metrics: {e2e.get('num_requests')} requests")
        print(f"  Throughput: {e2e.get('throughput_tokens_per_sec', 0):.2f} tokens/s")
    else:
        print("  ✗ 无E2E metrics")
    
    # Check operator performance
    op = data.get('operator_performance', {})
    per_layer = op.get('per_layer_stats', [])
    summary = op.get('summary_stats', {})
    
    if isinstance(per_layer, list) and len(per_layer) > 0:
        print(f"  ✓ Operator data: {len(per_layer)} layer records")
    else:
        print(f"  ✗ Operator data: empty or invalid")
    
    if summary:
        print(f"  ✓ Summary stats: {list(summary.keys())}")
    else:
        print(f"  ✗ Summary stats: empty")
EOF
echo ""

echo "=========================================="
echo "测试完成总结"
echo "=========================================="
echo ""
echo "预期结果："
echo "  ✓ Phase 1日志显示 'No Operator Profiling'"
echo "  ✓ 看到 'Reinitializing LLM for Phase 2' 日志"
echo "  ✓ 看到 'Executing warmup for Phase 2' 日志"
echo "  ✓ Phase 2显示 'Operator benchmarking enabled'"
echo "  ✓ Worker文件有数据 (>100KB)"
echo "  ✓ 最终JSON包含operator_performance数据"
echo ""
echo "如果以上都满足，说明Two-Phase重新初始化修复成功！"
echo ""


