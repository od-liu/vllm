#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""
Validation script to test the refactored benchmark setup.

This script performs various checks to ensure:
1. All required scripts can be imported
2. Configuration files are valid
3. Required dependencies are available
4. GPU access is working

Usage:
    python benchmarks/validate_setup.py --config benchmarks/operator_configs/e2e_trace_config.py
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

# Add vllm to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def check_imports():
    """Check if all required modules can be imported."""
    print("=" * 80)
    print("Checking Imports...")
    print("=" * 80)
    
    errors = []
    
    # Check main benchmark scripts
    scripts = [
        "benchmarks.benchmark_e2e_metrics",
    ]
    
    for script in scripts:
        try:
            __import__(script)
            print(f"✓ {script}")
        except Exception as e:
            print(f"✗ {script}: {e}")
            errors.append(f"Failed to import {script}: {e}")
    
    # Check vllm modules
    vllm_modules = [
        "vllm",
        "vllm.profiler",
        "vllm.profiler.e2e_metrics",
        "vllm.profiler.hash_id_mapper",
        "vllm.profiler.trace_loader",
        "vllm.profiler.trace_scheduler",
    ]
    
    for module in vllm_modules:
        try:
            __import__(module)
            print(f"✓ {module}")
        except Exception as e:
            print(f"✗ {module}: {e}")
            errors.append(f"Failed to import {module}: {e}")
    
    return errors


def check_config_file(config_path: str):
    """Check if configuration file is valid."""
    print("\n" + "=" * 80)
    print(f"Checking Configuration File: {config_path}")
    print("=" * 80)
    
    errors = []
    
    # Check if file exists
    if not os.path.exists(config_path):
        error = f"Configuration file not found: {config_path}"
        print(f"✗ {error}")
        errors.append(error)
        return errors
    
    print(f"✓ File exists: {config_path}")
    
    # Try to load config
    try:
        spec = importlib.util.spec_from_file_location("config", config_path)
        if spec is None or spec.loader is None:
            raise ValueError(f"Cannot load config from {config_path}")
        
        config_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(config_module)
        
        if not hasattr(config_module, "config"):
            raise ValueError("Config file must define a 'config' variable")
        
        config = config_module.config
        print(f"✓ Config loaded successfully")
        
        # Check required fields
        required_fields = [
            'model_path',
            'tensor_parallel_size',
            'pipeline_parallel_size',
            'data_parallel_size',
            'use_trace_data',
            'enable_e2e_metrics',
            'output_file',
        ]
        
        for field in required_fields:
            if hasattr(config, field):
                value = getattr(config, field)
                print(f"✓ {field}: {value}")
            else:
                error = f"Missing required field: {field}"
                print(f"✗ {error}")
                errors.append(error)
        
        # Check trace-specific fields if use_trace_data is True
        if hasattr(config, 'use_trace_data') and config.use_trace_data:
            trace_fields = ['trace_file_path', 'trace_time_range_minutes']
            for field in trace_fields:
                if hasattr(config, field):
                    value = getattr(config, field)
                    print(f"✓ {field}: {value}")
                    
                    # Check if trace file exists
                    if field == 'trace_file_path' and not os.path.exists(value):
                        error = f"Trace file not found: {value}"
                        print(f"⚠ {error}")
                        errors.append(error)
                else:
                    error = f"Missing trace field: {field}"
                    print(f"✗ {error}")
                    errors.append(error)
        
        # Validate config
        try:
            config.validate()
            print("✓ Config validation passed")
        except Exception as e:
            error = f"Config validation failed: {e}"
            print(f"✗ {error}")
            errors.append(error)
        
    except Exception as e:
        error = f"Failed to load config: {e}"
        print(f"✗ {error}")
        errors.append(error)
    
    return errors


def check_gpu_access():
    """Check if GPU access is working."""
    print("\n" + "=" * 80)
    print("Checking GPU Access...")
    print("=" * 80)
    
    errors = []
    
    try:
        import torch
        
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            print(f"✓ CUDA available: {gpu_count} GPU(s)")
            
            for i in range(gpu_count):
                gpu_name = torch.cuda.get_device_name(i)
                print(f"  GPU {i}: {gpu_name}")
        else:
            error = "CUDA not available"
            print(f"✗ {error}")
            errors.append(error)
    
    except Exception as e:
        error = f"Failed to check GPU access: {e}"
        print(f"✗ {error}")
        errors.append(error)
    
    return errors


def check_directories():
    """Check if required directories exist or can be created."""
    print("\n" + "=" * 80)
    print("Checking Directories...")
    print("=" * 80)
    
    errors = []
    
    # Check tmp_benchmark directory
    tmp_dir = Path("./tmp_benchmark")
    try:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        print(f"✓ tmp_benchmark directory: {tmp_dir.absolute()}")
    except Exception as e:
        error = f"Failed to create tmp_benchmark directory: {e}"
        print(f"✗ {error}")
        errors.append(error)
    
    # Check if benchmarks directory exists
    benchmarks_dir = Path("./benchmarks")
    if benchmarks_dir.exists():
        print(f"✓ benchmarks directory: {benchmarks_dir.absolute()}")
    else:
        error = "benchmarks directory not found"
        print(f"✗ {error}")
        errors.append(error)
    
    return errors


def check_scripts_exist():
    """Check if all required scripts exist."""
    print("\n" + "=" * 80)
    print("Checking Script Files...")
    print("=" * 80)
    
    errors = []
    
    scripts = [
        "benchmarks/benchmark_e2e_metrics.py",
    ]
    
    for script in scripts:
        if os.path.exists(script):
            print(f"✓ {script}")
        else:
            error = f"Script not found: {script}"
            print(f"✗ {error}")
            errors.append(error)
    
    return errors


def main():
    parser = argparse.ArgumentParser(
        description="Validate benchmark setup"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to configuration file to validate"
    )
    parser.add_argument(
        "--skip-gpu-check",
        action="store_true",
        help="Skip GPU access check"
    )
    
    args = parser.parse_args()
    
    print("\n")
    print("=" * 80)
    print("BENCHMARK SETUP VALIDATION")
    print("=" * 80)
    print("\n")
    
    all_errors = []
    
    # Check script files
    errors = check_scripts_exist()
    all_errors.extend(errors)
    
    # Check imports
    errors = check_imports()
    all_errors.extend(errors)
    
    # Check config if provided
    if args.config:
        errors = check_config_file(args.config)
        all_errors.extend(errors)
    
    # Check GPU access
    if not args.skip_gpu_check:
        errors = check_gpu_access()
        all_errors.extend(errors)
    
    # Check directories
    errors = check_directories()
    all_errors.extend(errors)
    
    # Print summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    if all_errors:
        print(f"\n✗ Validation failed with {len(all_errors)} error(s):\n")
        for i, error in enumerate(all_errors, 1):
            print(f"  {i}. {error}")
        print("\nPlease fix these errors before running benchmarks.")
        return 1
    else:
        print("\n✓ All checks passed! Setup is ready for benchmarking.")
        print("\nYou can now run:")
        if args.config:
            print(f"  python benchmarks/benchmark_e2e_metrics.py --config {args.config} --phase 1")
        else:
            print("  python benchmarks/benchmark_e2e_metrics.py --config <config> --phase 1")
        return 0


if __name__ == "__main__":
    sys.exit(main())

