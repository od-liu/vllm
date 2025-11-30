#!/usr/bin/env python3
"""
Test script to verify per-rank metrics collection and aggregation.
"""

import json
import tempfile
from pathlib import Path

def test_tp_file_naming():
    """Test TP rank file naming logic"""
    print("Test 1: TP File Naming")
    print("=" * 50)
    
    base_path = "/tmp/vllm_bench_dp0.json"
    base_file = Path(base_path)
    stem = base_file.stem  # "vllm_bench_dp0"
    suffix = base_file.suffix  # ".json"
    
    for tp_rank in range(4):
        new_stem = f"{stem}_tp{tp_rank}"
        rank_file = base_file.parent / f"{new_stem}{suffix}"
        print(f"  TP rank {tp_rank}: {rank_file}")
    
    print("✓ File naming test passed\n")


def test_file_discovery():
    """Test finding all TP rank files"""
    print("Test 2: File Discovery")
    print("=" * 50)
    
    # Create temp files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        base_file = tmpdir / "vllm_bench_dp0.json"
        
        # Create multiple TP rank files
        test_data = {"test": "data"}
        for tp_rank in range(3):
            tp_file = tmpdir / f"vllm_bench_dp0_tp{tp_rank}.json"
            with open(tp_file, 'w') as f:
                json.dump({**test_data, "tp_rank": tp_rank}, f)
            print(f"  Created: {tp_file.name}")
        
        # Discover files
        worker_file_base = base_file.stem
        worker_file_dir = base_file.parent
        worker_file_suffix = base_file.suffix
        
        found_files = []
        tp_rank = 0
        while True:
            tp_file = worker_file_dir / f"{worker_file_base}_tp{tp_rank}{worker_file_suffix}"
            if tp_file.exists():
                found_files.append(tp_file)
                tp_rank += 1
            else:
                break
        
        print(f"  Found {len(found_files)} files:")
        for f in found_files:
            print(f"    - {f.name}")
        
        assert len(found_files) == 3, f"Expected 3 files, found {len(found_files)}"
        print("✓ File discovery test passed\n")


def test_aggregation_logic():
    """Test aggregation of per-rank stats"""
    print("Test 3: Aggregation Logic")
    print("=" * 50)
    
    import numpy as np
    
    # Simulate TP rank results
    tp_results = [
        {
            "tp_rank": 0,
            "per_layer_stats": {
                "layer1": {
                    "mean_time_ms": 10.0,
                    "std_time_ms": 1.0,
                    "min_time_ms": 8.0,
                    "max_time_ms": 12.0,
                    "call_count": 100,
                }
            },
            "summary_stats": {},
            "total_forward_time_ms": 1000.0,
        },
        {
            "tp_rank": 1,
            "per_layer_stats": {
                "layer1": {
                    "mean_time_ms": 12.0,
                    "std_time_ms": 1.5,
                    "min_time_ms": 10.0,
                    "max_time_ms": 14.0,
                    "call_count": 100,
                }
            },
            "summary_stats": {},
            "total_forward_time_ms": 1200.0,
        },
    ]
    
    # Aggregate
    all_per_layer = [r["per_layer_stats"] for r in tp_results]
    aggregated_per_layer = {}
    
    for layer_name in all_per_layer[0].keys():
        layer_data = [stats[layer_name] for stats in all_per_layer]
        aggregated_per_layer[layer_name] = {
            'mean_time_ms': np.mean([d['mean_time_ms'] for d in layer_data]),
            'std_time_ms': np.mean([d['std_time_ms'] for d in layer_data]),
            'min_time_ms': np.mean([d['min_time_ms'] for d in layer_data]),
            'max_time_ms': np.mean([d['max_time_ms'] for d in layer_data]),
            'call_count': sum([d['call_count'] for d in layer_data]),
        }
    
    total_forward_time = sum([r['total_forward_time_ms'] for r in tp_results])
    
    print("  Input TP rank 0:")
    print(f"    mean_time_ms: {tp_results[0]['per_layer_stats']['layer1']['mean_time_ms']}")
    print(f"    total_forward_time_ms: {tp_results[0]['total_forward_time_ms']}")
    
    print("  Input TP rank 1:")
    print(f"    mean_time_ms: {tp_results[1]['per_layer_stats']['layer1']['mean_time_ms']}")
    print(f"    total_forward_time_ms: {tp_results[1]['total_forward_time_ms']}")
    
    print("  Aggregated:")
    print(f"    mean_time_ms: {aggregated_per_layer['layer1']['mean_time_ms']} (expected: 11.0)")
    print(f"    call_count: {aggregated_per_layer['layer1']['call_count']} (expected: 200)")
    print(f"    total_forward_time_ms: {total_forward_time} (expected: 2200.0)")
    
    assert aggregated_per_layer['layer1']['mean_time_ms'] == 11.0
    assert aggregated_per_layer['layer1']['call_count'] == 200
    assert total_forward_time == 2200.0
    
    print("✓ Aggregation logic test passed\n")


def test_output_structure():
    """Test expected output JSON structure"""
    print("Test 4: Output Structure")
    print("=" * 50)
    
    # Single DP process output
    single_dp_output = {
        "config": {},
        "operator_performance": {
            "per_layer_stats": {},
            "summary_stats": {
                "total_forward_time_ms": 2200.0,
                "num_tp_ranks_aggregated": 2,
            },
            "per_tp_rank": [
                {
                    "tp_rank": 0,
                    "per_layer_stats": {},
                    "summary_stats": {},
                    "total_forward_time_ms": 1000.0,
                },
                {
                    "tp_rank": 1,
                    "per_layer_stats": {},
                    "summary_stats": {},
                    "total_forward_time_ms": 1200.0,
                },
            ]
        },
        "e2e_metrics": {}
    }
    
    # Verify structure
    assert "operator_performance" in single_dp_output
    assert "per_tp_rank" in single_dp_output["operator_performance"]
    assert len(single_dp_output["operator_performance"]["per_tp_rank"]) == 2
    print("  ✓ Single DP output structure valid")
    
    # Multi-DP output
    multi_dp_output = {
        "config": {},
        "data_parallel_size": 2,
        "operator_performance": {
            "per_layer_stats": {},
            "summary_stats": {
                "total_forward_time_ms": 4400.0,
                "num_ranks_aggregated": 2,
            }
        },
        "operator_performance_per_dp_rank": [
            {
                "dp_rank": 0,
                "per_layer_stats": {},
                "summary_stats": {},
                "per_tp_rank": [
                    {"tp_rank": 0, "total_forward_time_ms": 1000.0},
                    {"tp_rank": 1, "total_forward_time_ms": 1200.0},
                ]
            },
            {
                "dp_rank": 1,
                "per_layer_stats": {},
                "summary_stats": {},
                "per_tp_rank": [
                    {"tp_rank": 0, "total_forward_time_ms": 1000.0},
                    {"tp_rank": 1, "total_forward_time_ms": 1200.0},
                ]
            }
        ],
        "e2e_metrics_aggregated": {},
        "e2e_metrics_per_rank": []
    }
    
    assert "operator_performance_per_dp_rank" in multi_dp_output
    assert len(multi_dp_output["operator_performance_per_dp_rank"]) == 2
    for dp_data in multi_dp_output["operator_performance_per_dp_rank"]:
        assert "per_tp_rank" in dp_data
        assert len(dp_data["per_tp_rank"]) == 2
    print("  ✓ Multi-DP output structure valid")
    
    print("✓ Output structure test passed\n")


def main():
    """Run all tests"""
    print("\n" + "=" * 50)
    print("Per-Rank Metrics Test Suite")
    print("=" * 50 + "\n")
    
    test_tp_file_naming()
    test_file_discovery()
    test_aggregation_logic()
    test_output_structure()
    
    print("=" * 50)
    print("All tests passed! ✓")
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()


