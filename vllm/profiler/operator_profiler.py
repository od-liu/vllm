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
    
    # 确保如果多次初始化类，返回同一个实例
    def __new__(cls):  
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize the benchmark tracker."""
        if self._initialized:  # 如果已经初始化过，直接返回
            return
        # 只初始化一次
        self._initialized = True
        self._enabled = False
        self._warmup_mode = True
        self._warmup_count = 0
        self._warmup_steps = 5
        self._benchmark_steps = 0
        self._records_lock = threading.Lock()
        self._records: List[OperatorCallRecord] = []  # 存储OperatorCallRecord
        self._call_counts = defaultdict(int)  # 存储各层调用次数
        
        # Batch synchronization support
        self._pending_events: List[PendingEvent] = []  # 待同步的CUDA事件队列
        self._pending_lock = threading.Lock()
        self._flush_count = 0  # 前向传播次数计数（用于warmup判断）
        
        logger.info("OperatorBenchmark initialized")
    
    @classmethod  # 类方法，获取单例实例
    def get_instance(cls) -> "OperatorBenchmark":
        """Get the singleton instance."""
        return cls()
    
    # 启用benchmark，参数warmup_steps默认5不
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
    
    # 禁用benchmark
    def disable(self) -> None:
        """Disable benchmarking."""
        self._enabled = False
        logger.info("Operator benchmarking disabled")
    
    # 检查是否启用benchmark
    def is_enabled(self) -> bool:
        """Check if benchmarking is enabled."""
        return self._enabled
    
    # 检查是否处于warmup阶段
    def is_warmup(self) -> bool:
        """Check if in warmup phase."""
        return self._warmup_mode
    
    # 标记一次前向传播完成，用于 warmup 到 benchmark 的切换
    def step(self) -> None:
        """
        Signal completion of a forward pass.
        
        This transitions from warmup to benchmark mode after warmup_steps.
        """
        if not self._enabled:
            return
        
        # 如果在warm_up阶段
        # 增加warm_up_count，达到阈值后退出warmup
        if self._warmup_mode:
            self._warmup_count += 1
            if self._warmup_count >= self._warmup_steps:
                self._warmup_mode = False
                # 退出warmup阶段，开始benchmark
                logger.info(
                    f"Warmup complete after {self._warmup_count} steps. "
                    "Starting benchmark collection."
                )
        else:
            # 如果在benchmark阶段
            # 增加benchmark_steps
            self._benchmark_steps += 1
    
    # 记录单个operator调用(CPU路径或旧接口)
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
            self._call_counts[layer_name] += 1  # 更新call_counts，记录各层调用次数
            total_calls = sum(self._call_counts.values())
            
            # Check if we're still in warmup phase
            # Use warmup_steps * expected_operators_per_forward as threshold
            # Assume ~100 operators per forward pass
            warmup_threshold = self._warmup_steps * 100  # 基于总调用数判断warmup阈值，假设每个forward pass有100个operators
            
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
                if len(self._records) % 1000 == 0:  # 每1000条记录保存一次，该方法主要用于CPU或旧接口，CUDA路径优先使用add_pending_event()
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
    
    # 将待同步的CUDA事件加入队列，并不同步事件(CUDA路径)
    def add_pending_event(self, event: PendingEvent) -> None:
        """
        Add a pending CUDA event to the queue for batch processing.
        
        This method is called by the wrapped forward methods. Events are stored
        without synchronization for better performance. They will be processed
        in batch by flush_pending_events().
        
        Args:
            event: PendingEvent containing CUDA events and metadata
        """
        if not self._enabled:   # 检查是否启用benchmark
            return
        
        with self._pending_lock:  # 使用锁保证线程安全
            self._pending_events.append(event)
    
    # 批量同步并处理所有待处理的CUDA事件（核心方法）
    # 在每次前向传播后调用
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
        import sys
        
        with self._pending_lock:  # 使用锁保证线程安全
            if not self._pending_events:  # 检查队列是否为空
                return
            
            num_pending = len(self._pending_events)
            logger.info(f"Flushing {num_pending} pending CUDA events...")
            
            # Single synchronization for all pending events
            if torch.cuda.is_available():  # 检查CUDA是否可用
                import time
                sync_start = time.perf_counter()
                torch.cuda.synchronize()  # 同步所有待处理的CUDA事件
                sync_time = time.perf_counter() - sync_start
                logger.debug(f"  CUDA sync: {sync_time:.2f}s")
            
            # Increment flush count (tracks forward pass count)
            self._flush_count += 1  # 增加前向传播次数计数
            
            # Check warmup status based on flush count (forward passes)
            in_warmup = self._flush_count <= self._warmup_steps  # 检查是否处于warmup阶段
            
            # Update call counts for tracking
            with self._records_lock:  # 使用锁保证线程安全
                for event in self._pending_events:
                    self._call_counts[event.layer_name] += 1
            
            # If still in warmup, skip recording but clear queue
            if in_warmup:  # 如果还在warmup阶段，跳过记录并清空队列
                self._pending_events.clear()
                return
            
            # Past warmup: batch process all events
            records_to_add = []  # 存储待添加的记录
            
            # Process events (no progress reporting for cleaner logs)
            for event in self._pending_events:
                try:
                    # Calculate elapsed time
                    elapsed_time_ms = event.start_event.elapsed_time(event.end_event)  # 计算事件持续时间
                    
                    # Create record
                    record = OperatorCallRecord(
                        layer_name=event.layer_name,
                        operator_type=event.operator_type,
                        elapsed_time_ms=elapsed_time_ms,
                        input_shapes=event.input_shapes,
                        output_shapes=event.output_shapes,
                        device=event.device,
                    )
                    records_to_add.append(record)  # 将记录添加到待添加的记录列表
                except Exception as e:
                    logger.debug(f"Failed to process pending event for {event.layer_name}: {e}")
            
            # Add all records at once
            if records_to_add:
                with self._records_lock:
                    self._records.extend(records_to_add)  # 将待添加的记录列表添加到记录列表
                
                # Auto-save after each flush if enabled
                import os
                if os.environ.get("VLLM_OPERATOR_BENCHMARK_AUTO_SAVE"):
                    # 如果启用自动保存，保存结果到临时文件
                    output_file = os.environ.get(
                        "VLLM_OPERATOR_BENCHMARK_OUTPUT",
                        "/tmp/vllm_operator_benchmark.json"
                    )
                    try:
                        logger.debug(
                            f"Auto-saving {len(self._records)} records after flush "
                            f"({len(records_to_add)} new) to {output_file}"
                        )
                        self._save_results_internal(output_file)  # 保存结果到临时文件
                    except Exception as e:
                        logger.warning(f"Failed to auto-save benchmark results: {e}")
            
            # Clear the queue
            self._pending_events.clear()  # 清空队列
            logger.info(f"✓ Flush #{self._flush_count} complete: processed {len(records_to_add)} events, total {len(self._records)} records")
    
    # 计算统计信息，返回类型为BenchmarkResults
    def get_statistics(self) -> BenchmarkResults:
        """
        Compute statistics from collected records.
        
        Returns:
            BenchmarkResults containing per-layer and summary statistics
        """
        with self._records_lock:
            if not self._records:   # 如果没有记录，返回空结果
                logger.warning("No benchmark records collected")
                return BenchmarkResults(
                    per_layer_stats=[],
                    summary_stats={},
                    total_forward_time_ms=0.0,
                    config={},
                )
            
            # Group records by layer name
            layer_records = defaultdict(list)  # 按layer_name分组
            for record in self._records:
                layer_records[record.layer_name].append(record)
            
            # Compute per-layer statistics
            per_layer_stats = []  # 存储每层统计信息
            for layer_name, records in sorted(layer_records.items()):
                times = [r.elapsed_time_ms for r in records]  # 整理每层调用时间
                operator_type = records[0].operator_type  # 获取operator类型
                
                # Average input/output shapes
                avg_input_shapes = []  # 存储每层输入形状
                avg_output_shapes = []  # 存储每层输出形状
                
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
                    "avg_time_ms": float(np.mean(times)),  # 计算平均时间
                    "std_time_ms": float(np.std(times)),
                    "min_time_ms": float(np.min(times)),
                    "max_time_ms": float(np.max(times)),
                    "total_time_ms": float(np.sum(times)),
                    "avg_input_shapes": avg_input_shapes,
                    "avg_output_shapes": avg_output_shapes,
                }
                per_layer_stats.append(stats)  # 将每层统计信息添加到列表
            
            # Compute summary statistics by operator type
            operator_records = defaultdict(list)  # 按operator类型分组
            for record in self._records:
                operator_records[record.operator_type].append(record)  # 将记录添加到对应的operator类型列表
            
            total_time = sum(r.elapsed_time_ms for r in self._records)  # 计算总时间
            summary_stats = {}
            
            for op_type, records in sorted(operator_records.items()):
                times = [r.elapsed_time_ms for r in records]  # 整理每种operator类型的调用时间
                op_total_time = sum(times)
                
                summary_stats[op_type] = {
                    "total_calls": len(times),
                    "total_time_ms": float(op_total_time),
                    "avg_time_per_call_ms": float(np.mean(times)),  # 计算平均时间
                    "std_time_ms": float(np.std(times)),
                    "percentage_of_total": float(
                        (op_total_time / total_time * 100) if total_time > 0 else 0  # 计算百分比
                    ),
                }
            
            return BenchmarkResults(  # 返回统计结果
                per_layer_stats=per_layer_stats,
                summary_stats=summary_stats,
                total_forward_time_ms=float(total_time),
                config={
                    "warmup_steps": self._warmup_steps,
                    "benchmark_steps": self._benchmark_steps,
                    "total_records": len(self._records),
                },
            )
    
    # 内部保存方法（无额外日志）
    def _save_results_internal(self, output_path: str) -> None:
        """Internal method to save results without extra logging."""
        results = self.get_statistics()  # 获取统计结果
        
        output_dict = {  # 将统计结果转换为字典
            "config": results.config,
            "per_layer_stats": results.per_layer_stats,
            "summary_stats": results.summary_stats,
            "total_forward_time_ms": results.total_forward_time_ms,
        }
        
        output_file_path = Path(output_path)
        
        # IMPORTANT: For TP>1, each worker must write to a separate file to avoid conflicts
        # Check if this is a TP worker and add rank suffix if needed
        tp_rank_info = "unknown"
        try:
            from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
            tp_rank = get_tensor_model_parallel_rank()
            tp_size = get_tensor_model_parallel_world_size()
            tp_rank_info = f"TP{tp_rank}/{tp_size}"
            
            # Only add suffix if TP>1 (multiple workers)
            if tp_size > 1:
                # Modify filename to include TP rank: file.json -> file_tp{rank}.json
                stem = output_file_path.stem
                suffix = output_file_path.suffix
                parent = output_file_path.parent
                output_file_path = parent / f"{stem}_tp{tp_rank}{suffix}"
        except Exception:
            # If we can't get TP info, proceed with original path
            pass
        
        # Debug info
        num_layers = len(results.per_layer_stats) if isinstance(results.per_layer_stats, dict) else 0
        total_records = len(self._records)
        
        output_file_path.parent.mkdir(parents=True, exist_ok=True)
        # 将字典保存为JSON文件
        with open(output_file_path, "w") as f:
            json.dump(output_dict, f, indent=2)
        
        # Use DEBUG level to reduce log spam during auto-save
        logger.debug(
            f"[{tp_rank_info}] Saved benchmark results to {output_file_path}: "
            f"{total_records} records, {num_layers} layers, "
            f"{results.total_forward_time_ms:.2f}ms total"
        )
    
    # 公开的保存方式（带日志）
    def save_results(self, output_path: str) -> None:
        """
        Save benchmark results to a JSON file.
        
        Args:
            output_path: Path to output JSON file
        """
        self._save_results_internal(output_path)
        logger.info(f"Benchmark results saved to {output_path}")
    
    # 重置所有数据（用于重新开始benchmark）
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
    
    # 在控制台打印统计摘要
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

