# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Operator-level profiler for benchmarking compute-intensive operators."""

import json
import threading
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class OperatorCallRecord:
    """Record for a single operator call."""
    
    layer_name: str
    operator_type: str
    elapsed_time_ms: float
    input_shapes: List[Tuple[int, ...]]
    output_shapes: List[Tuple[int, ...]]
    device: str


@dataclass
class PendingEvent:
    """Pending CUDA event waiting to be synchronized and processed."""
    
    start_event: torch.cuda.Event
    end_event: torch.cuda.Event
    layer_name: str
    operator_type: str
    input_shapes: List[Tuple[int, ...]]
    output_shapes: List[Tuple[int, ...]]
    device: str


@dataclass
class OperatorStats:
    """Statistics for an operator across multiple calls."""
    
    layer_name: str
    operator_type: str
    num_calls: int
    avg_time_ms: float
    std_time_ms: float
    min_time_ms: float
    max_time_ms: float
    total_time_ms: float
    avg_input_shapes: List[List[int]]
    avg_output_shapes: List[List[int]]


@dataclass
class BenchmarkResults:
    """Complete benchmark results including per-layer and summary statistics."""
    
    per_layer_stats: List[Dict[str, Any]]
    summary_stats: Dict[str, Dict[str, Any]]
    total_forward_time_ms: float
    config: Dict[str, Any]


class OperatorBenchmark:
    """
    A singleton class to manage operator-level benchmarking.
    
    This class tracks performance metrics for all operators during model execution
    using CUDA events for precise timing measurements.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the benchmark tracker."""
        if self._initialized:
            return
        
        self._initialized = True
        self._enabled = False
        self._warmup_mode = True
        self._warmup_count = 0
        self._warmup_steps = 5
        self._benchmark_steps = 0
        self._records_lock = threading.Lock()
        self._records: List[OperatorCallRecord] = []
        self._call_counts = defaultdict(int)
        
        # Batch synchronization support
        self._pending_events: List[PendingEvent] = []
        self._pending_lock = threading.Lock()
        self._flush_count = 0  # Track forward pass count
        
        logger.info("OperatorBenchmark initialized")
    
    @classmethod
    def get_instance(cls) -> "OperatorBenchmark":
        """Get the singleton instance."""
        return cls()
    
    def enable(self, warmup_steps: int = 5) -> None:
        """Enable benchmarking with specified warmup steps."""
        self._enabled = True
        self._warmup_mode = True
        self._warmup_count = 0
        self._warmup_steps = warmup_steps
        self._benchmark_steps = 0
        logger.info(
            f"Operator benchmarking enabled with {warmup_steps} warmup steps"
        )
    
    def disable(self) -> None:
        """Disable benchmarking."""
        self._enabled = False
        logger.info("Operator benchmarking disabled")
    
    def is_enabled(self) -> bool:
        """Check if benchmarking is enabled."""
        return self._enabled
    
    def is_warmup(self) -> bool:
        """Check if in warmup phase."""
        return self._warmup_mode
    
    def step(self) -> None:
        """
        Signal completion of a forward pass.
        
        This transitions from warmup to benchmark mode after warmup_steps.
        """
        if not self._enabled:
            return
        
        if self._warmup_mode:
            self._warmup_count += 1
            if self._warmup_count >= self._warmup_steps:
                self._warmup_mode = False
                logger.info(
                    f"Warmup complete after {self._warmup_count} steps. "
                    "Starting benchmark collection."
                )
        else:
            self._benchmark_steps += 1
    
    def record_operator_call(
        self,
        layer_name: str,
        operator_type: str,
        elapsed_time_ms: float,
        input_shapes: List[Tuple[int, ...]],
        output_shapes: List[Tuple[int, ...]],
        device: str = "cuda",
    ) -> None:
        """
        Record a single operator call.
        
        This method is called by the wrapped forward methods in worker processes.
        It automatically tracks warmup based on call counts.
        
        Args:
            layer_name: Full name of the layer/module
            operator_type: Type of operator (e.g., "Attention", "Linear")
            elapsed_time_ms: Execution time in milliseconds
            input_shapes: List of input tensor shapes
            output_shapes: List of output tensor shapes
            device: Device where operator executed
        """
        if not self._enabled:
            return
        
        # Auto-track warmup in worker process based on call counts
        # IMPORTANT: Always increment call count, even during warmup!
        with self._records_lock:
            # Increment call count FIRST
            self._call_counts[layer_name] += 1
            total_calls = sum(self._call_counts.values())
            
            # Check if we're still in warmup phase
            # Use warmup_steps * expected_operators_per_forward as threshold
            # Assume ~100 operators per forward pass
            warmup_threshold = self._warmup_steps * 100
            
            if total_calls <= warmup_threshold:
                # Still in warmup, don't record data
                return
        
        # Past warmup, create and record the data
        record = OperatorCallRecord(
            layer_name=layer_name,
            operator_type=operator_type,
            elapsed_time_ms=elapsed_time_ms,
            input_shapes=input_shapes,
            output_shapes=output_shapes,
            device=device,
        )
        
        with self._records_lock:
            self._records.append(record)
            
            # Auto-save results periodically in worker process
            # Check if we should save based on environment variable
            import os
            if os.environ.get("VLLM_OPERATOR_BENCHMARK_AUTO_SAVE"):
                # Save every 1000 records to avoid too frequent I/O
                if len(self._records) % 1000 == 0:
                    output_file = os.environ.get(
                        "VLLM_OPERATOR_BENCHMARK_OUTPUT",
                        "/tmp/vllm_operator_benchmark.json"
                    )
                    try:
                        self._save_results_internal(output_file)
                        logger.debug(
                            f"Auto-saved {len(self._records)} benchmark records to {output_file}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to auto-save benchmark results: {e}")
    
    def add_pending_event(self, event: PendingEvent) -> None:
        """
        Add a pending CUDA event to the queue for batch processing.
        
        This method is called by the wrapped forward methods. Events are stored
        without synchronization for better performance. They will be processed
        in batch by flush_pending_events().
        
        Args:
            event: PendingEvent containing CUDA events and metadata
        """
        if not self._enabled:
            return
        
        with self._pending_lock:
            self._pending_events.append(event)
    
    def flush_pending_events(self) -> None:
        """
        Synchronize and process all pending CUDA events in batch.
        
        This method should be called after each forward pass to:
        1. Synchronize CUDA operations once (instead of per-operator)
        2. Calculate elapsed times for all pending events
        3. Record the operator calls
        4. Clear the pending queue
        
        This dramatically improves performance by reducing synchronization
        overhead from O(num_operators) to O(1) per forward pass.
        """
        with self._pending_lock:
            if not self._pending_events:
                return
            
            # Single synchronization for all pending events
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            # Increment flush count (tracks forward pass count)
            self._flush_count += 1
            
            # Check warmup status based on flush count (forward passes)
            in_warmup = self._flush_count <= self._warmup_steps
            
            # Update call counts for tracking
            with self._records_lock:
                for event in self._pending_events:
                    self._call_counts[event.layer_name] += 1
            
            # If still in warmup, skip recording but clear queue
            if in_warmup:
                self._pending_events.clear()
                return
            
            # Past warmup: batch process all events
            records_to_add = []
            for event in self._pending_events:
                try:
                    # Calculate elapsed time
                    elapsed_time_ms = event.start_event.elapsed_time(event.end_event)
                    
                    # Create record
                    record = OperatorCallRecord(
                        layer_name=event.layer_name,
                        operator_type=event.operator_type,
                        elapsed_time_ms=elapsed_time_ms,
                        input_shapes=event.input_shapes,
                        output_shapes=event.output_shapes,
                        device=event.device,
                    )
                    records_to_add.append(record)
                except Exception as e:
                    logger.debug(f"Failed to process pending event for {event.layer_name}: {e}")
            
            # Add all records at once
            if records_to_add:
                with self._records_lock:
                    self._records.extend(records_to_add)
                
                # Auto-save after each flush if enabled
                import os
                if os.environ.get("VLLM_OPERATOR_BENCHMARK_AUTO_SAVE"):
                    output_file = os.environ.get(
                        "VLLM_OPERATOR_BENCHMARK_OUTPUT",
                        "/tmp/vllm_operator_benchmark.json"
                    )
                    try:
                        self._save_results_internal(output_file)
                        logger.debug(
                            f"Auto-saved {len(self._records)} benchmark records to {output_file}"
                        )
                    except Exception as e:
                        logger.warning(f"Failed to auto-save benchmark results: {e}")
            
            # Clear the queue
            self._pending_events.clear()
    
    def get_statistics(self) -> BenchmarkResults:
        """
        Compute statistics from collected records.
        
        Returns:
            BenchmarkResults containing per-layer and summary statistics
        """
        with self._records_lock:
            if not self._records:
                logger.warning("No benchmark records collected")
                return BenchmarkResults(
                    per_layer_stats=[],
                    summary_stats={},
                    total_forward_time_ms=0.0,
                    config={},
                )
            
            # Group records by layer name
            layer_records = defaultdict(list)
            for record in self._records:
                layer_records[record.layer_name].append(record)
            
            # Compute per-layer statistics
            per_layer_stats = []
            for layer_name, records in sorted(layer_records.items()):
                times = [r.elapsed_time_ms for r in records]
                operator_type = records[0].operator_type
                
                # Average input/output shapes
                avg_input_shapes = []
                avg_output_shapes = []
                
                if records[0].input_shapes:
                    num_inputs = len(records[0].input_shapes)
                    for i in range(num_inputs):
                        shapes = [r.input_shapes[i] for r in records if i < len(r.input_shapes)]
                        if shapes:
                            avg_shape = [int(np.mean([s[j] for s in shapes])) 
                                        for j in range(len(shapes[0]))]
                            avg_input_shapes.append(avg_shape)
                
                if records[0].output_shapes:
                    num_outputs = len(records[0].output_shapes)
                    for i in range(num_outputs):
                        shapes = [r.output_shapes[i] for r in records if i < len(r.output_shapes)]
                        if shapes:
                            avg_shape = [int(np.mean([s[j] for s in shapes])) 
                                        for j in range(len(shapes[0]))]
                            avg_output_shapes.append(avg_shape)
                
                stats = {
                    "layer_name": layer_name,
                    "operator_type": operator_type,
                    "num_calls": len(times),
                    "avg_time_ms": float(np.mean(times)),
                    "std_time_ms": float(np.std(times)),
                    "min_time_ms": float(np.min(times)),
                    "max_time_ms": float(np.max(times)),
                    "total_time_ms": float(np.sum(times)),
                    "avg_input_shapes": avg_input_shapes,
                    "avg_output_shapes": avg_output_shapes,
                }
                per_layer_stats.append(stats)
            
            # Compute summary statistics by operator type
            operator_records = defaultdict(list)
            for record in self._records:
                operator_records[record.operator_type].append(record)
            
            total_time = sum(r.elapsed_time_ms for r in self._records)
            summary_stats = {}
            
            for op_type, records in sorted(operator_records.items()):
                times = [r.elapsed_time_ms for r in records]
                op_total_time = sum(times)
                
                summary_stats[op_type] = {
                    "total_calls": len(times),
                    "total_time_ms": float(op_total_time),
                    "avg_time_per_call_ms": float(np.mean(times)),
                    "std_time_ms": float(np.std(times)),
                    "percentage_of_total": float(
                        (op_total_time / total_time * 100) if total_time > 0 else 0
                    ),
                }
            
            return BenchmarkResults(
                per_layer_stats=per_layer_stats,
                summary_stats=summary_stats,
                total_forward_time_ms=float(total_time),
                config={
                    "warmup_steps": self._warmup_steps,
                    "benchmark_steps": self._benchmark_steps,
                    "total_records": len(self._records),
                },
            )
    
    def _save_results_internal(self, output_path: str) -> None:
        """Internal method to save results without extra logging."""
        results = self.get_statistics()
        
        output_dict = {
            "config": results.config,
            "per_layer_stats": results.per_layer_stats,
            "summary_stats": results.summary_stats,
            "total_forward_time_ms": results.total_forward_time_ms,
        }
        
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_file, "w") as f:
            json.dump(output_dict, f, indent=2)
    
    def save_results(self, output_path: str) -> None:
        """
        Save benchmark results to a JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        self._save_results_internal(output_path)
        logger.info(f"Benchmark results saved to {output_path}")
    
    def reset(self) -> None:
        """Reset all collected benchmark data."""
        with self._records_lock:
            self._records.clear()
            self._call_counts.clear()
            self._warmup_count = 0
            self._benchmark_steps = 0
            self._warmup_mode = True
        with self._pending_lock:
            self._pending_events.clear()
            self._flush_count = 0
        logger.info("Benchmark data reset")
    
    def print_summary(self) -> None:
        """Print a summary of benchmark results to console."""
        results = self.get_statistics()
        
        print("\n" + "=" * 80)
        print("OPERATOR BENCHMARK SUMMARY")
        print("=" * 80)
        print(f"Total forward time: {results.total_forward_time_ms:.2f} ms")
        print(f"Total records: {results.config.get('total_records', 0)}")
        print(f"Benchmark steps: {results.config.get('benchmark_steps', 0)}")
        print("\n" + "-" * 80)
        print("Summary by Operator Type:")
        print("-" * 80)
        
        for op_type, stats in sorted(
            results.summary_stats.items(),
            key=lambda x: x[1]["total_time_ms"],
            reverse=True,
        ):
            print(f"\n{op_type}:")
            print(f"  Total calls: {stats['total_calls']}")
            print(f"  Total time: {stats['total_time_ms']:.2f} ms")
            print(f"  Avg time per call: {stats['avg_time_per_call_ms']:.4f} ms")
            print(f"  Percentage of total: {stats['percentage_of_total']:.1f}%")
        
        print("\n" + "=" * 80)
        print(f"Top 10 slowest operators:")
        print("-" * 80)
        
        sorted_layers = sorted(
            results.per_layer_stats,
            key=lambda x: x["total_time_ms"],
            reverse=True,
        )[:10]
        
        for i, layer_stats in enumerate(sorted_layers, 1):
            print(f"\n{i}. {layer_stats['layer_name']}")
            print(f"   Type: {layer_stats['operator_type']}")
            print(f"   Calls: {layer_stats['num_calls']}")
            print(f"   Avg: {layer_stats['avg_time_ms']:.4f} ms")
            print(f"   Total: {layer_stats['total_time_ms']:.2f} ms")
        
        print("\n" + "=" * 80 + "\n")


# Global convenience functions
def get_operator_benchmark() -> OperatorBenchmark:
    """Get the global OperatorBenchmark instance."""
    return OperatorBenchmark.get_instance()


def enable_operator_benchmark(warmup_steps: int = 5) -> None:
    """Enable operator benchmarking globally."""
    get_operator_benchmark().enable(warmup_steps)


def disable_operator_benchmark() -> None:
    """Disable operator benchmarking globally."""
    get_operator_benchmark().disable()

