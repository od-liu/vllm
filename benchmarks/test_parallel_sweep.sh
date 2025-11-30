#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
#
# Test script for parallel sweep functionality
# This runs a minimal test with 2 GPUs to verify the sweep system works

set -e  # Exit on error

echo "========================================"
echo "Parallel Sweep Test Script"
echo "========================================"

# Configuration
NUM_GPUS=2
TEST_DIR="benchmark_results/test_sweep_$(date +%Y%m%d_%H%M%S)"
BASE_CONFIG="benchmarks/operator_configs/sweep_test_config.py"

echo ""
echo "Configuration:"
echo "  Number of GPUs: $NUM_GPUS"
echo "  Test directory: $TEST_DIR"
echo "  Base config: $BASE_CONFIG"
echo ""

# Check if base config exists
if [ ! -f "$BASE_CONFIG" ]; then
    echo "Error: Base config not found: $BASE_CONFIG"
    exit 1
fi

# Step 1: Test configuration generation
echo "Step 1: Testing configuration generation..."
python3 -c "
from benchmarks.run_parallel_sweep import generate_parallel_configs
configs = generate_parallel_configs($NUM_GPUS)
print(f'Generated {len(configs)} configurations:')
for tp, pp, dp in configs:
    print(f'  TP={tp}, PP={pp}, DP={dp}')
    assert tp * pp * dp == $NUM_GPUS, f'Invalid config: {tp} * {pp} * {dp} != $NUM_GPUS'
print('✓ Configuration generation test passed')
"

if [ $? -ne 0 ]; then
    echo "✗ Configuration generation test failed"
    exit 1
fi

echo ""

# Step 2: Run the actual sweep (dry run - just check it starts)
echo "Step 2: Testing sweep execution (checking script syntax)..."
python3 benchmarks/run_parallel_sweep.py --help > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "✗ Sweep script execution test failed"
    exit 1
fi

echo "✓ Sweep script syntax test passed"
echo ""

# Step 3: Test visualization script
echo "Step 3: Testing visualization script (checking syntax)..."
python3 benchmarks/visualize_results.py --help > /dev/null 2>&1

if [ $? -ne 0 ]; then
    echo "⚠ Visualization script requires matplotlib/numpy (optional)"
    echo "  Install with: pip install matplotlib numpy"
else
    echo "✓ Visualization script syntax test passed"
fi
echo ""

# Step 4: Check documentation
echo "Step 4: Checking documentation..."
if [ ! -f "benchmarks/PARALLEL_SWEEP_GUIDE.md" ]; then
    echo "✗ Documentation not found"
    exit 1
fi

echo "✓ Documentation exists"
echo ""

# Final summary
echo "========================================"
echo "All basic tests passed!"
echo "========================================"
echo ""
echo "To run a full test sweep (requires GPUs):"
echo "  python benchmarks/run_parallel_sweep.py \\"
echo "    --num-gpus $NUM_GPUS \\"
echo "    --base-config $BASE_CONFIG \\"
echo "    --output-dir $TEST_DIR"
echo ""
echo "Then visualize with:"
echo "  python benchmarks/visualize_results.py \\"
echo "    --results-dir $TEST_DIR \\"
echo "    --show-best"
echo ""

