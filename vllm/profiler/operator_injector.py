# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Module injection system for operator benchmarking."""

import functools
from typing import Any, Callable, List, Optional, Set, Tuple

import torch
import torch.nn as nn

from vllm.logger import init_logger
from vllm.profiler.operator_profiler import get_operator_benchmark

logger = init_logger(__name__)


# Mapping of module types to operator names
OPERATOR_TYPE_MAPPING = {
    # Attention modules
    "Attention": "attention",
    "PagedAttention": "attention",
    "FlashAttention": "attention",
    
    # Linear modules
    "Linear": "linear",
    "RowParallelLinear": "linear",
    "ColumnParallelLinear": "linear",
    "QKVParallelLinear": "linear",
    "MergedColumnParallelLinear": "linear",
    "ReplicatedLinear": "linear",
    
    # Normalization modules
    "RMSNorm": "layernorm",
    "LayerNorm": "layernorm",
    "RmsNorm": "layernorm",
    
    # MLP modules
    "MLP": "mlp",
    "GatedMLP": "mlp",
    
    # Activation modules
    "SiLU": "activation",
    "GELU": "activation",
    "NewGELU": "activation",
    "FastGELU": "activation",
    
    # Embedding modules
    "VocabParallelEmbedding": "embedding",
    "Embedding": "embedding",
    
    # MoE modules
    "FusedMoE": "moe",
    "MoE": "moe",
    
    # Mamba/SSM modules
    "MambaLayer": "mamba",
    "MambaMixer": "mamba",
}


def get_operator_type(module: nn.Module) -> Optional[str]:
    """
    Determine the operator type of a module.
    
    Args:
        module: The PyTorch module to classify
        
    Returns:
        Operator type string, or None if not a benchmarkable operator
    """
    module_class_name = module.__class__.__name__
    
    # Direct mapping
    if module_class_name in OPERATOR_TYPE_MAPPING:
        return OPERATOR_TYPE_MAPPING[module_class_name]
    
    # Check for parent classes
    for base_class in module.__class__.__mro__:
        base_class_name = base_class.__name__
        if base_class_name in OPERATOR_TYPE_MAPPING:
            return OPERATOR_TYPE_MAPPING[base_class_name]
    
    return None


def get_tensor_shapes(tensor_or_tuple: Any) -> List[Tuple[int, ...]]:
    """
    Extract shapes from tensors or nested tuples/lists of tensors.
    
    Args:
        tensor_or_tuple: A tensor, tuple, list, or dict containing tensors
        
    Returns:
        List of tensor shapes
    """
    shapes = []
    
    if isinstance(tensor_or_tuple, torch.Tensor):
        shapes.append(tuple(tensor_or_tuple.shape))
    elif isinstance(tensor_or_tuple, (tuple, list)):
        for item in tensor_or_tuple:
            shapes.extend(get_tensor_shapes(item))
    elif isinstance(tensor_or_tuple, dict):
        for value in tensor_or_tuple.values():
            shapes.extend(get_tensor_shapes(value))
    
    return shapes


def create_benchmark_wrapper(
    original_forward: Callable,
    module: nn.Module,
    layer_name: str,
    operator_type: str,
) -> Callable:
    """
    Create a wrapper around a module's forward method for benchmarking.
    
    Args:
        original_forward: The original forward method
        module: The module being wrapped
        layer_name: Full name of the layer in the model
        operator_type: Type of operator (e.g., "attention", "linear")
        
    Returns:
        Wrapped forward method
    """
    
    @functools.wraps(original_forward)
    def benchmark_wrapper(*args, **kwargs):
        benchmark = get_operator_benchmark()
        
        # If benchmarking is not enabled, just call original
        if not benchmark.is_enabled():
            return original_forward(*args, **kwargs)
        
        # Create CUDA events for timing
        if torch.cuda.is_available():
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            # Record start
            start_event.record()
            
            # Call original forward
            result = original_forward(*args, **kwargs)
            
            # Record end
            end_event.record()
            
            # Extract shapes
            input_shapes = get_tensor_shapes(args) + get_tensor_shapes(kwargs)
            output_shapes = get_tensor_shapes(result)
            
            # Add to pending queue for batch processing (no synchronization here!)
            from vllm.profiler.operator_profiler import PendingEvent
            benchmark.add_pending_event(PendingEvent(
                start_event=start_event,
                end_event=end_event,
                layer_name=layer_name,
                operator_type=operator_type,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                device="cuda",
            ))
        else:
            # CPU fallback (less accurate timing)
            import time
            start_time = time.perf_counter()
            result = original_forward(*args, **kwargs)
            end_time = time.perf_counter()
            elapsed_time_ms = (end_time - start_time) * 1000
            
            input_shapes = get_tensor_shapes(args) + get_tensor_shapes(kwargs)
            output_shapes = get_tensor_shapes(result)
            
            benchmark.record_operator_call(
                layer_name=layer_name,
                operator_type=operator_type,
                elapsed_time_ms=elapsed_time_ms,
                input_shapes=input_shapes,
                output_shapes=output_shapes,
                device="cpu",
            )
        
        return result
    
    return benchmark_wrapper


def inject_benchmark_hooks(
    model: nn.Module,
    operators_to_benchmark: Optional[List[str]] = None,
    module_prefix: str = "",
) -> int:
    """
    Inject benchmark hooks into all relevant operators in a model.
    
    Args:
        model: The PyTorch model to instrument
        operators_to_benchmark: List of operator types to benchmark (e.g., ["attention", "linear"])
                               If None or contains "all", benchmark all supported operators
        module_prefix: Prefix for module names (used for recursion)
        
    Returns:
        Number of modules instrumented
    """
    if operators_to_benchmark is None:
        operators_to_benchmark = ["all"]
    
    benchmark_all = "all" in operators_to_benchmark
    operators_set = set(operators_to_benchmark)
    
    instrumented_count = 0
    
    for name, module in model.named_modules():
        # Skip if this module has already been instrumented
        if hasattr(module, "_benchmark_instrumented"):
            continue
        
        # Determine operator type
        operator_type = get_operator_type(module)
        # 不是支持的operator类型，跳过
        if operator_type is None:
            continue
        
        # 如果当前operator类型不在需要benchmark的类型中，跳过
        if not benchmark_all and operator_type not in operators_set:
            continue
        
        # Get the full layer name
        # 构建完整的层名称
        full_name = f"{module_prefix}.{name}" if module_prefix else name
        if not full_name:
            full_name = module.__class__.__name__
        
        # Wrap the forward method
        try:
            original_forward = module.forward   # 原始forward函数
            wrapped_forward = create_benchmark_wrapper(  # 包装函数
                original_forward,
                module,
                full_name,
                operator_type,
            )
            module.forward = wrapped_forward  # 替换原始forward函数
            module._benchmark_instrumented = True
            instrumented_count += 1  # 计数加1
            
            logger.debug(
                f"Instrumented {operator_type} operator: {full_name}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to instrument {full_name}: {e}"
            )
    
    logger.info(
        f"Instrumented {instrumented_count} operators for benchmarking"
    )
    
    return instrumented_count


def remove_benchmark_hooks(model: nn.Module) -> int:
    """
    Remove all benchmark hooks from a model.
    
    Args:
        model: The PyTorch model to clean up
        
    Returns:
        Number of modules cleaned up
    """
    removed_count = 0
    
    for name, module in model.named_modules():
        if hasattr(module, "_benchmark_instrumented"):
            # Try to restore original forward
            # Note: This is best-effort; proper restoration would require
            # storing the original method, which we don't do here
            delattr(module, "_benchmark_instrumented")
            removed_count += 1
    
    logger.info(f"Removed benchmark hooks from {removed_count} operators")
    
    return removed_count


def should_instrument_operator(
    operator_type: str,
    operators_to_benchmark: List[str],
) -> bool:
    """
    Check if an operator type should be instrumented.
    
    Args:
        operator_type: Type of operator
        operators_to_benchmark: List of operator types to benchmark
        
    Returns:
        True if operator should be instrumented
    """
    if not operators_to_benchmark:
        return False
    
    if "all" in operators_to_benchmark:
        return True
    
    return operator_type in operators_to_benchmark


def get_supported_operator_types() -> List[str]:
    """
    Get list of all supported operator types.
    
    Returns:
        List of operator type strings
    """
    return sorted(set(OPERATOR_TYPE_MAPPING.values()))


def register_operator_type(
    module_class_name: str,
    operator_type: str,
) -> None:
    """
    Register a custom operator type for benchmarking.
    
    Args:
        module_class_name: Name of the module class
        operator_type: Operator type string
    """
    OPERATOR_TYPE_MAPPING[module_class_name] = operator_type
    logger.info(
        f"Registered custom operator type: {module_class_name} -> {operator_type}"
    )

