#!/usr/bin/env python3
"""
Simple example demonstrating operator benchmark API usage.

This script shows how to use the operator benchmark functionality
programmatically without using the full benchmark script.
"""

from vllm import LLM, SamplingParams
from vllm.profiler import enable_operator_benchmark, get_operator_benchmark


def main():
    print("=" * 80)
    print("vLLM Operator Benchmark Example")
    print("=" * 80)
    
    # Configuration
    model_path = "/mnt/disk1/ljm/qwen_test/models/Qwen/Qwen3-8B"  # Use a small model for quick testing
    warmup_steps = 3
    benchmark_steps = 10
    
    print(f"\nModel: {model_path}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Benchmark steps: {benchmark_steps}")
    
    # IMPORTANT: Set environment variables BEFORE creating LLM instance
    # This ensures hooks are injected into the model during loading
    import os
    import tempfile
    
    # Create a temporary file for worker to save results
    temp_output = tempfile.NamedTemporaryFile(
        mode='w', suffix='.json', delete=False, dir='/tmp'
    )
    output_file = temp_output.name
    temp_output.close()
    
    os.environ["VLLM_OPERATOR_BENCHMARK_ENABLE"] = "1"
    os.environ["VLLM_OPERATOR_BENCHMARK_OPS"] = "attention,linear,layernorm,mlp,activation"
    os.environ["VLLM_OPERATOR_BENCHMARK_WARMUP"] = str(warmup_steps)
    os.environ["VLLM_OPERATOR_BENCHMARK_AUTO_SAVE"] = "1"
    os.environ["VLLM_OPERATOR_BENCHMARK_OUTPUT"] = output_file
    
    print(f"Worker will save results to: {output_file}")
    
    # Enable operator benchmarking (for coordination, though worker manages its own)
    enable_operator_benchmark(warmup_steps=warmup_steps)
    
    # Create LLM instance
    print("\nInitializing vLLM...")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=1,
        max_model_len=512,
        enforce_eager=True,  # Disable CUDA graph for simpler example
    )
    
    # Prepare test prompts
    prompts = [
        "The future of artificial intelligence is",
        "Machine learning has revolutionized",
        "Deep learning models can now",
    ]
    
    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=16,
        ignore_eos=True,
    )
    
    # Get benchmark instance
    benchmark = get_operator_benchmark()
    
    # Run warmup + benchmark iterations
    total_iterations = warmup_steps + benchmark_steps
    print(f"\nRunning {total_iterations} iterations...")
    
    for i in range(total_iterations):
        if i < warmup_steps:
            print(f"  Warmup iteration {i+1}/{warmup_steps}")
        else:
            print(f"  Benchmark iteration {i-warmup_steps+1}/{benchmark_steps}")
        
        # Run inference
        _ = llm.generate(prompts, sampling_params)
        
        # Signal step completion
        benchmark.step()
    
    print("\nBenchmark completed!")
    
    # Give worker a moment to finish saving
    import time
    time.sleep(1)
    
    # Read results from worker's output file
    import json
    print(f"\nReading results from worker process: {output_file}")
    
    try:
        with open(output_file, 'r') as f:
            worker_results = json.load(f)
        
        # Check if we got any results
        if not worker_results.get("per_layer_stats"):
            print("⚠️  WARNING: Worker process did not collect any benchmark data!")
            print("   This might mean the benchmark didn't run long enough,")
            print("   or there was an issue with data collection.")
            
            # Try to get results from main process (might be empty)
            benchmark = get_operator_benchmark()
            results = benchmark.get_statistics()
        else:
            print(f"✅ Successfully loaded {len(worker_results['per_layer_stats'])} operator stats from worker")
            
            # Convert back to results format for display
            from vllm.profiler.operator_profiler import BenchmarkResults
            results = BenchmarkResults(
                per_layer_stats=worker_results["per_layer_stats"],
                summary_stats=worker_results["summary_stats"],
                total_forward_time_ms=worker_results["total_forward_time_ms"],
                config=worker_results["config"],
            )
            
            # Also save to final location
            final_output = "example_benchmark_results.json"
            with open(final_output, 'w') as f:
                json.dump(worker_results, f, indent=2)
            print(f"Results also saved to: {final_output}")
    except FileNotFoundError:
        print(f"⚠️  WARNING: Could not find worker results at {output_file}")
        print("   Worker process may not have collected enough data yet.")
        benchmark = get_operator_benchmark()
        results = benchmark.get_statistics()
    except Exception as e:
        print(f"⚠️  ERROR reading worker results: {e}")
        benchmark = get_operator_benchmark()
        results = benchmark.get_statistics()
    
    print("\n" + "=" * 80)
    print("Quick Summary:")
    print("=" * 80)
    print(f"Total forward time: {results.total_forward_time_ms:.2f} ms")
    print(f"Total operators measured: {len(results.per_layer_stats)}")
    
    # Show top 3 slowest layers
    sorted_layers = sorted(
        results.per_layer_stats,
        key=lambda x: x["total_time_ms"],
        reverse=True,
    )[:3]
    
    print("\nTop 3 slowest operators:")
    for i, layer in enumerate(sorted_layers, 1):
        print(f"{i}. {layer['layer_name']}")
        print(f"   Type: {layer['operator_type']}")
        print(f"   Average time: {layer['avg_time_ms']:.4f} ms")
        print(f"   Total time: {layer['total_time_ms']:.2f} ms")
    
    print("\n" + "=" * 80)
    print("Example completed successfully!")
    print("=" * 80)


if __name__ == "__main__":
    main()

