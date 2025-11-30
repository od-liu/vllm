#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Component test script for parallel sweep functionality.
Tests individual components without requiring GPUs.
"""

import sys
import os

# Add parent directory to path to enable imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_config_generation():
    """Test parallel configuration generation."""
    print("Test 1: Configuration Generation")
    print("-" * 40)
    
    from run_parallel_sweep import generate_parallel_configs
    
    # Test different GPU counts
    test_cases = [2, 4, 8]
    
    for num_gpus in test_cases:
        configs = generate_parallel_configs(num_gpus)
        print(f"\n{num_gpus} GPUs: {len(configs)} configurations")
        
        for tp, pp, dp in configs:
            product = tp * pp * dp
            print(f"  TP={tp}, PP={pp}, DP={dp} (product={product})")
            assert product == num_gpus, f"Invalid config: {tp}×{pp}×{dp}={product} != {num_gpus}"
    
    print("\n✓ Configuration generation test passed\n")
    return True


def test_divisors():
    """Test divisor calculation."""
    print("Test 2: Divisor Calculation")
    print("-" * 40)
    
    from run_parallel_sweep import get_divisors
    
    test_cases = {
        8: [1, 2, 4, 8],
        12: [1, 2, 3, 4, 6, 12],
        16: [1, 2, 4, 8, 16],
    }
    
    for n, expected in test_cases.items():
        result = get_divisors(n)
        print(f"  Divisors of {n}: {result}")
        assert result == expected, f"Expected {expected}, got {result}"
    
    print("\n✓ Divisor calculation test passed\n")
    return True


def test_result_parsing():
    """Test result file name parsing."""
    print("Test 3: Result File Parsing")
    print("-" * 40)
    
    try:
        from visualize_results import parse_config_name
        
        test_cases = {
            "tp1_pp2_dp4_results.json": (1, 2, 4),
            "tp8_pp1_dp1_results.json": (8, 1, 1),
            "tp2_pp2_dp2_results.json": (2, 2, 2),
            "invalid_filename.json": None,
        }
        
        for filename, expected in test_cases.items():
            result = parse_config_name(filename)
            print(f"  {filename} -> {result}")
            assert result == expected, f"Expected {expected}, got {result}"
        
        print("\n✓ Result file parsing test passed\n")
        return True
    except ImportError as e:
        print(f"\n⚠ Skipping visualization tests (missing dependencies: {e})\n")
        return True


def test_temp_config_creation():
    """Test temporary config file creation."""
    print("Test 4: Temporary Config Creation")
    print("-" * 40)
    
    from run_parallel_sweep import create_temp_config
    import tempfile
    import os
    
    # Create a minimal base config
    base_config_content = """
from vllm.profiler import BenchmarkConfig

config = BenchmarkConfig(
    model_path="/path/to/model",
    tensor_parallel_size=1,
    pipeline_parallel_size=1,
    data_parallel_size=1,
    output_file="results.json",
)
"""
    
    # Write to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
        f.write(base_config_content)
        base_config_path = f.name
    
    try:
        # Test config modification
        temp_config = create_temp_config(
            base_config_path,
            tp=2,
            pp=2,
            dp=2,
            output_file="test_output.json",
        )
        
        print(f"  Created temp config: {temp_config}")
        
        # Verify the temp config was created
        assert os.path.exists(temp_config), "Temp config file not created"
        
        # Read and verify content
        with open(temp_config, 'r') as f:
            content = f.read()
            assert "tensor_parallel_size=2" in content, "TP not updated"
            assert "pipeline_parallel_size=2" in content, "PP not updated"
            assert "data_parallel_size=2" in content, "DP not updated"
            assert "test_output.json" in content, "Output file not updated"
        
        print(f"  ✓ Config properly modified")
        
        # Clean up
        os.remove(temp_config)
        
    finally:
        os.remove(base_config_path)
    
    print("\n✓ Temporary config creation test passed\n")
    return True


def main():
    """Run all component tests."""
    print("=" * 60)
    print("Parallel Sweep Component Tests")
    print("=" * 60)
    print()
    
    tests = [
        test_config_generation,
        test_divisors,
        test_temp_config_creation,
        test_result_parsing,
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"✗ Test failed with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("=" * 60)
    print(f"Test Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    if failed == 0:
        print("\n✓ All tests passed!\n")
        return 0
    else:
        print(f"\n✗ {failed} test(s) failed\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())

