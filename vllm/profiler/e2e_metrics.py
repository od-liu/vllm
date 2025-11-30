# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""End-to-end metrics collection for vLLM benchmarking."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from vllm.logger import init_logger
from vllm.outputs import RequestOutput
from vllm.sequence import RequestMetrics
from vllm.v1.metrics.stats import RequestStateStats

logger = init_logger(__name__)


@dataclass
class E2EMetrics:
    """End-to-end metrics for a single request."""
    
    request_id: str
    """Request ID."""
    
    ttft_ms: Optional[float] = None
    """Time to first token in milliseconds."""
    
    tpot_ms: Optional[float] = None
    """Time per output token in milliseconds."""
    
    e2e_latency_ms: Optional[float] = None
    """End-to-end latency in milliseconds."""
    
    queued_time_ms: Optional[float] = None
    """Time spent in queue in milliseconds."""
    
    num_prompt_tokens: int = 0
    """Number of prompt tokens."""
    
    num_generation_tokens: int = 0
    """Number of generated tokens."""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "request_id": self.request_id,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "e2e_latency_ms": self.e2e_latency_ms,
            "queued_time_ms": self.queued_time_ms,
            "num_prompt_tokens": self.num_prompt_tokens,
            "num_generation_tokens": self.num_generation_tokens,
        }


class E2EMetricsCollector:
    """Collects and aggregates end-to-end metrics from multiple requests."""
    
    def __init__(self):
        """Initialize the collector."""
        self.metrics: List[E2EMetrics] = []
    
    def add_request(self, request_output: RequestOutput) -> Optional[E2EMetrics]:
        """
        Extract metrics from a RequestOutput and add to collection.
        
        Args:
            request_output: RequestOutput from LLM.generate()
            
        Returns:
            E2EMetrics object if metrics were successfully extracted, None otherwise
        """
        if request_output is None:
            return None
        
        metrics_obj = request_output.metrics
        if metrics_obj is None:
            logger.warning(
                f"Request {request_output.request_id} has no metrics"
            )
            return None
        
        # Extract token counts
        num_prompt_tokens = 0
        if request_output.prompt_token_ids:
            num_prompt_tokens = len(request_output.prompt_token_ids)
        
        num_generation_tokens = 0
        if request_output.outputs:
            # Sum tokens across all outputs
            for output in request_output.outputs:
                if output.token_ids:
                    num_generation_tokens += len(output.token_ids)
        
        # Extract timing metrics based on metrics type
        ttft_ms = None
        tpot_ms = None
        e2e_latency_ms = None
        queued_time_ms = None
        
        if isinstance(metrics_obj, RequestMetrics):
            # v0 engine format
            if metrics_obj.first_token_time is not None and metrics_obj.arrival_time is not None:
                ttft_ms = (metrics_obj.first_token_time - metrics_obj.arrival_time) * 1000
            
            if metrics_obj.finished_time is not None and metrics_obj.arrival_time is not None:
                e2e_latency_ms = (metrics_obj.finished_time - metrics_obj.arrival_time) * 1000
            
            if metrics_obj.finished_time is not None and metrics_obj.first_token_time is not None:
                if num_generation_tokens > 1:
                    # TPOT = (finished_time - first_token_time) / (num_generation_tokens - 1)
                    tpot_ms = (
                        (metrics_obj.finished_time - metrics_obj.first_token_time) 
                        / (num_generation_tokens - 1)
                    ) * 1000
                elif num_generation_tokens == 1:
                    # If only one token, use e2e latency as TPOT
                    if e2e_latency_ms is not None:
                        tpot_ms = e2e_latency_ms
            
            if metrics_obj.time_in_queue is not None:
                queued_time_ms = metrics_obj.time_in_queue * 1000
        
        elif isinstance(metrics_obj, RequestStateStats):
            # v1 engine format
            # RequestStateStats uses monotonic timestamps for engine events
            # and wall-clock time for arrival_time
            
            # TTFT: Use first_token_latency (already calculated as wall-clock difference)
            if metrics_obj.first_token_latency > 0:
                ttft_ms = metrics_obj.first_token_latency * 1000
            
            # Queued time: scheduled_ts - queued_ts (both are monotonic, so difference is valid)
            if metrics_obj.scheduled_ts > 0 and metrics_obj.queued_ts > 0:
                queued_time_ms = (metrics_obj.scheduled_ts - metrics_obj.queued_ts) * 1000
            
            # TPOT: Calculate from decode time (last_token_ts - first_token_ts)
            # Both are monotonic timestamps, so difference is valid
            # Note: decode_time = last_token_ts - first_token_ts
            # TPOT = decode_time / (num_generation_tokens - 1)
            # We subtract 1 because the first token is generated during prefill
            if (metrics_obj.last_token_ts > 0 and 
                metrics_obj.first_token_ts > 0 and 
                num_generation_tokens > 1):
                decode_time = metrics_obj.last_token_ts - metrics_obj.first_token_ts
                tpot_ms = (decode_time / (num_generation_tokens - 1)) * 1000
            elif num_generation_tokens == 1:
                # If only one token, we can't calculate TPOT properly
                # Set to None to indicate it's not applicable
                tpot_ms = None
            
            # E2E latency: Calculate from arrival_time to finish time
            # Since RequestStateStats doesn't have finish wall-clock time,
            # we approximate E2E latency as: first_token_latency + decode_time
            # This is because:
            # - first_token_latency = first_token_time (wall-clock) - arrival_time (wall-clock)
            # - decode_time = last_token_ts (monotonic) - first_token_ts (monotonic)
            # - We assume monotonic and wall-clock clocks progress at similar rates
            # Note: This is an approximation, but more accurate than using current time
            # which would include the time between request completion and metrics collection
            if (metrics_obj.first_token_latency > 0 and 
                metrics_obj.last_token_ts > 0 and 
                metrics_obj.first_token_ts > 0):
                decode_time = metrics_obj.last_token_ts - metrics_obj.first_token_ts
                # E2E latency ≈ TTFT (wall-clock) + decode_time (monotonic, but similar rate)
                # This gives us: arrival -> first_token -> last_token
                e2e_latency_ms = (metrics_obj.first_token_latency + decode_time) * 1000
            elif metrics_obj.first_token_latency > 0:
                # If we only have TTFT, use it as a lower bound for E2E latency
                e2e_latency_ms = metrics_obj.first_token_latency * 1000
        
        e2e_metrics = E2EMetrics(
            request_id=request_output.request_id,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            e2e_latency_ms=e2e_latency_ms,
            queued_time_ms=queued_time_ms,
            num_prompt_tokens=num_prompt_tokens,
            num_generation_tokens=num_generation_tokens,
        )
        
        self.metrics.append(e2e_metrics)
        return e2e_metrics
    
    def add_requests(self, request_outputs: List[RequestOutput]) -> List[E2EMetrics]:
        """
        Extract metrics from multiple RequestOutputs.
        
        Args:
            request_outputs: List of RequestOutput objects
            
        Returns:
            List of E2EMetrics objects (None entries filtered out)
        """
        results = []
        for req_output in request_outputs:
            metrics = self.add_request(req_output)
            if metrics is not None:
                results.append(metrics)
        return results
    
    def calculate_statistics(self) -> Dict[str, Any]:
        """
        Calculate aggregated statistics from collected metrics.
        
        Returns:
            Dictionary containing mean, p50, p90, p99 for each metric
        """
        if not self.metrics:
            return {
                "num_requests": 0,
            }
        
        # Extract metric arrays
        ttft_values = [m.ttft_ms for m in self.metrics if m.ttft_ms is not None]
        tpot_values = [m.tpot_ms for m in self.metrics if m.tpot_ms is not None]
        e2e_latency_values = [
            m.e2e_latency_ms for m in self.metrics 
            if m.e2e_latency_ms is not None
        ]
        queued_time_values = [
            m.queued_time_ms for m in self.metrics 
            if m.queued_time_ms is not None
        ]
        
        # Calculate total tokens
        total_prompt_tokens = sum(m.num_prompt_tokens for m in self.metrics)
        total_generation_tokens = sum(m.num_generation_tokens for m in self.metrics)
        
        def calc_stats(values: List[float]) -> Dict[str, float]:
            """Calculate statistics for a list of values."""
            if not values:
                return {"mean": None, "p50": None, "p90": None, "p99": None}
            
            arr = np.array(values)
            return {
                "mean": float(np.mean(arr)),
                "p50": float(np.percentile(arr, 50)),
                "p90": float(np.percentile(arr, 90)),
                "p99": float(np.percentile(arr, 99)),
            }
        
        stats = {
            "num_requests": len(self.metrics),
            "ttft_ms": calc_stats(ttft_values),
            "tpot_ms": calc_stats(tpot_values),
            "e2e_latency_ms": calc_stats(e2e_latency_values),
            "queued_time_ms": calc_stats(queued_time_values),
            "total_prompt_tokens": total_prompt_tokens,
            "total_generation_tokens": total_generation_tokens,
        }
        
        return stats
    
    def calculate_throughput(
        self, 
        total_time_seconds: float
    ) -> Optional[float]:
        """
        Calculate throughput in tokens per second.
        
        Args:
            total_time_seconds: Total wall-clock time for all requests
            
        Returns:
            Throughput in tokens per second, or None if calculation is not possible
        """
        if total_time_seconds <= 0:
            return None
        
        total_tokens = sum(
            m.num_prompt_tokens + m.num_generation_tokens 
            for m in self.metrics
        )
        
        if total_tokens == 0:
            return None
        
        return total_tokens / total_time_seconds
    
    def reset(self):
        """Reset the collector, clearing all collected metrics."""
        self.metrics.clear()

