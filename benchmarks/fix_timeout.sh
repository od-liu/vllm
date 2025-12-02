#!/bin/bash
# 快速修复超时问题脚本
# 用法: ./benchmarks/fix_timeout.sh [配置文件路径] [phase]

set -e

CONFIG_PATH="${1:-benchmarks/operator_configs/e2e_trace_config.py}"
PHASE="${2:-2}"

echo "======================================================================"
echo "🔧 修复Benchmark超时问题"
echo "======================================================================"
echo ""
echo "检测到的问题："
echo "  ❌ 请求超时时间设置过短（~22分钟）"
echo "  ❌ 部分请求运行时间异常长（输入长，但卡住）"
echo "  ❌ TP大小可能检测失败（显示TP=1）"
echo ""
echo "应用的修复："
echo "  ✅ 设置超时为2小时（7200秒）"
echo "  ✅ 手动设置TP大小（如果适用）"
echo "  ✅ 启用详细日志"
echo ""
echo "======================================================================"
echo ""

# 1. 设置超时为2小时
export VLLM_BENCHMARK_TIMEOUT=7200
echo "✓ VLLM_BENCHMARK_TIMEOUT=7200 (2小时)"

# 2. 检测GPU数量并设置TP大小
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
    echo "✓ 检测到 $GPU_COUNT 个GPU"
    
    if [ "$GPU_COUNT" -ge 8 ]; then
        export VLLM_TENSOR_PARALLEL_SIZE=8
        echo "✓ VLLM_TENSOR_PARALLEL_SIZE=8"
    elif [ "$GPU_COUNT" -ge 4 ]; then
        export VLLM_TENSOR_PARALLEL_SIZE=4
        echo "✓ VLLM_TENSOR_PARALLEL_SIZE=4"
    elif [ "$GPU_COUNT" -ge 2 ]; then
        export VLLM_TENSOR_PARALLEL_SIZE=2
        echo "✓ VLLM_TENSOR_PARALLEL_SIZE=2"
    else
        export VLLM_TENSOR_PARALLEL_SIZE=1
        echo "✓ VLLM_TENSOR_PARALLEL_SIZE=1"
    fi
else
    echo "⚠  无法检测GPU，请手动设置 VLLM_TENSOR_PARALLEL_SIZE"
fi

# 3. 启用INFO级别日志
export VLLM_LOGGING_LEVEL=INFO
echo "✓ VLLM_LOGGING_LEVEL=INFO"

echo ""
echo "======================================================================"
echo "🚀 开始运行Benchmark"
echo "======================================================================"
echo "  配置文件: $CONFIG_PATH"
echo "  Phase: $PHASE"
echo "  TP大小: ${VLLM_TENSOR_PARALLEL_SIZE:-配置文件默认值}"
echo "  超时设置: 7200秒 (2小时)"
echo "  输出目录: ./tmp_benchmark/tp${VLLM_TENSOR_PARALLEL_SIZE:-1}_dp1/"
echo "  预计运行时间: 根据trace大小，可能需要30分钟到2小时"
echo "======================================================================"
echo ""

# 4. 运行benchmark
# 根据检测到的GPU数量，自动设置tensor-parallel-size参数
if [ -n "$VLLM_TENSOR_PARALLEL_SIZE" ]; then
    echo "  使用 TP size: $VLLM_TENSOR_PARALLEL_SIZE"
    python benchmarks/benchmark_operators.py \
        --config "$CONFIG_PATH" \
        --phase "$PHASE" \
        --tensor-parallel-size "$VLLM_TENSOR_PARALLEL_SIZE"
else
    echo "  使用配置文件中的默认 TP size"
    python benchmarks/benchmark_operators.py \
        --config "$CONFIG_PATH" \
        --phase "$PHASE"
fi

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo "======================================================================"
    echo "✅ Benchmark完成"
    echo "======================================================================"
else
    echo "======================================================================"
    echo "❌ Benchmark失败 (退出码: $exit_code)"
    echo "======================================================================"
    echo ""
    echo "💡 如果仍然超时，尝试："
    echo "   1. 进一步增加超时: export VLLM_BENCHMARK_TIMEOUT=14400 (4小时)"
    echo "   2. 禁用超时: export VLLM_BENCHMARK_TIMEOUT=none"
    echo "   3. 检查GPU性能: watch -n 1 nvidia-smi"
    echo "   4. 减少trace数据中的输出长度"
fi

exit $exit_code

