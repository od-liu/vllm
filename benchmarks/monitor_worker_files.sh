#!/bin/bash
# 实时监控worker文件的写入情况

echo "==================================="
echo "Worker File Monitor"
echo "==================================="
echo "Monitoring: ./tmp_benchmark/"
echo "Press Ctrl+C to stop"
echo ""

watch -n 2 '
echo "$(date)"
echo "-----------------------------------"
echo "Worker files in ./tmp_benchmark/:"
if [ -d "./tmp_benchmark" ]; then
    ls -lh ./tmp_benchmark/*.json 2>/dev/null || echo "  No files yet"
    echo ""
    echo "File sizes and layer counts:"
    for f in ./tmp_benchmark/*.json 2>/dev/null; do
        if [ -f "$f" ]; then
            size=$(stat -c%s "$f" 2>/dev/null || stat -f%z "$f" 2>/dev/null)
            echo "  $(basename $f): $size bytes"
            if [ "$size" -gt 0 ]; then
                python3 -c "
import json, sys
try:
    with open(\"$f\") as file:
        d = json.load(file)
        layers = len(d.get(\"per_layer_stats\", {}))
        records = len(d.get(\"config\", {}).get(\"records\", []))
        total_time = d.get(\"total_forward_time_ms\", 0)
        print(f\"    → {layers} layers, {total_time:.2f}ms total\")
except Exception as e:
    print(f\"    → Error reading: {e}\")
" 2>/dev/null
            fi
        fi
    done
else
    echo "  Directory not created yet"
fi
echo ""
echo "Latest operator profiling logs:"
grep -h "Auto-saving\|Saved benchmark" /tmp/*.log 2>/dev/null | tail -5 || echo "  No logs yet"
'

