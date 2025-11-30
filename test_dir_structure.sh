#!/bin/bash
# 快速测试新的目录结构

echo "清理旧的tmp_benchmark..."
rm -rf tmp_benchmark

echo ""
echo "运行sweep测试..."
python benchmarks/run_parallel_sweep.py \
    --num-gpus 4 \
    --base-config benchmarks/operator_configs/e2e_trace_config.py \
    --output-dir benchmark_results/test_dir \
    --gpu-ids "3,4,5,6" \
    2>&1 | head -100

echo ""
echo "检查目录结构..."
tree -L 2 tmp_benchmark/ 2>/dev/null || find tmp_benchmark -type f -o -type d | sort

echo ""
echo "各配置的文件数量:"
for dir in tmp_benchmark/*/; do
    if [ -d "$dir" ]; then
        count=$(ls "$dir"*.json 2>/dev/null | wc -l)
        echo "  $(basename $dir): $count files"
    fi
done
