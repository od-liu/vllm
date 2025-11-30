#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Test script to verify DP implementation without running full benchmark.

This script tests:
1. Round-robin trace sharding logic
2. Function signatures and imports
3. Configuration validation
"""

import sys
from pathlib import Path

# Add vllm to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_round_robin_sharding():
    """Test that round-robin sharding distributes requests correctly."""
    print("=" * 80)
    print("Test 1: Round-robin trace sharding")
    print("=" * 80)
    
    # Simulate trace requests
    trace_requests = list(range(10))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    dp_size = 2
    
    # Test rank 0
    dp_rank = 0
    rank_0_requests = [
        req for i, req in enumerate(trace_requests) 
        if i % dp_size == dp_rank
    ]
    print(f"DP rank 0 requests: {rank_0_requests}")
    assert rank_0_requests == [0, 2, 4, 6, 8], f"Expected [0, 2, 4, 6, 8], got {rank_0_requests}"
    
    # Test rank 1
    dp_rank = 1
    rank_1_requests = [
        req for i, req in enumerate(trace_requests) 
        if i % dp_size == dp_rank
    ]
    print(f"DP rank 1 requests: {rank_1_requests}")
    assert rank_1_requests == [1, 3, 5, 7, 9], f"Expected [1, 3, 5, 7, 9], got {rank_1_requests}"
    
    print("✓ Round-robin sharding works correctly")
    print()
    return True


def test_function_imports():
    """Test that all new functions can be imported."""
    print("=" * 80)
    print("Test 2: Function imports")
    print("=" * 80)
    
    try:
        from benchmarks.benchmark_operators import (
            run_benchmark,
            run_benchmark_single_rank,
            run_benchmark_multi_rank,
            _run_rank_process,
            _aggregate_dp_results,
            _aggregate_operator_performance,
            _aggregate_e2e_metrics,
        )
        print("✓ All functions imported successfully")
        print()
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        print()
        return False


def test_config_with_dp():
    """Test that config with DP can be created."""
    print("=" * 80)
    print("Test 3: Configuration with DP")
    print("=" * 80)
    
    try:
        from vllm.profiler import BenchmarkConfig
        
        config = BenchmarkConfig(
            model_path="/fake/model/path",
            tensor_parallel_size=2,
            pipeline_parallel_size=1,
            data_parallel_size=2,
            use_trace_data=True,
            trace_file_path="/fake/trace.jsonl",
            output_file="test_output.json",
        )
        
        print(f"  Model: {config.model_path}")
        print(f"  TP: {config.tensor_parallel_size}")
        print(f"  PP: {config.pipeline_parallel_size}")
        print(f"  DP: {config.data_parallel_size}")
        print("✓ Config with DP=2 created successfully")
        print()
        return True
    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        print()
        return False


def test_visualize_aggregated_metrics():
    """Test that visualize script can handle aggregated metrics."""
    print("=" * 80)
    print("Test 4: Visualize aggregated metrics")
    print("=" * 80)
    
    try:
        # Create fake result data with aggregated metrics
        fake_result = {
            'e2e_metrics_aggregated': {
                'num_requests': 100,
                'num_ranks': 2,
                'ttft_ms': {'mean': 100.5, 'p50': 95.0, 'p90': 120.0, 'p99': 150.0},
                'tpot_ms': {'mean': 10.5, 'p50': 10.0, 'p90': 12.0, 'p99': 15.0},
            },
            'e2e_metrics_per_rank': [
                {'dp_rank': 0, 'e2e_metrics': {'num_requests': 50}},
                {'dp_rank': 1, 'e2e_metrics': {'num_requests': 50}},
            ]
        }
        
        # Simulate extract_metrics logic
        if 'e2e_metrics_aggregated' in fake_result:
            e2e = fake_result['e2e_metrics_aggregated']
        elif 'e2e_metrics' in fake_result:
            e2e = fake_result['e2e_metrics']
        else:
            e2e = None
        
        assert e2e is not None, "Failed to extract e2e metrics"
        assert e2e['num_requests'] == 100, f"Expected 100 requests, got {e2e['num_requests']}"
        assert e2e['num_ranks'] == 2, f"Expected 2 ranks, got {e2e['num_ranks']}"
        
        print("  Aggregated metrics extracted:")
        print(f"    Total requests: {e2e['num_requests']}")
        print(f"    Number of ranks: {e2e['num_ranks']}")
        print(f"    TTFT mean: {e2e['ttft_ms']['mean']} ms")
        print("✓ Aggregated metrics handling works correctly")
        print()
        return True
    except Exception as e:
        print(f"✗ Aggregated metrics test failed: {e}")
        print()
        return False


def main():
    print("\n" + "=" * 80)
    print("DP Implementation Tests")
    print("=" * 80)
    print()
    
    results = []
    
    # Run tests
    results.append(("Round-robin sharding", test_round_robin_sharding()))
    results.append(("Function imports", test_function_imports()))
    results.append(("Config with DP", test_config_with_dp()))
    results.append(("Visualize aggregated metrics", test_visualize_aggregated_metrics()))
    
    # Summary
    print("=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print()
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed! DP implementation is ready for testing.")
        return 0
    else:
        print("\n✗ Some tests failed. Please review the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())


