# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trace-based request scheduler for operator benchmarking."""

import asyncio
import time
from typing import Any, Dict, List, Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

from vllm import LLM, SamplingParams
from vllm.logger import init_logger
from vllm.profiler.hash_id_mapper import HashIDMapper
from vllm.profiler.trace_loader import TraceRequest

logger = init_logger(__name__)


class TraceScheduler:
    """Schedules trace requests with real-time replay support."""
    
    def __init__(
        self,
        llm: LLM,
        mapper: HashIDMapper,
        sampling_params: SamplingParams,
    ):
        """
        Initialize trace scheduler.
        
        Args:
            llm: vLLM instance for generating outputs
            mapper: HashIDMapper for converting hash_ids to token_ids
            sampling_params: Sampling parameters for generation
        """
        self.llm = llm
        self.mapper = mapper
        self.sampling_params = sampling_params
    
    def _create_prompt_from_trace(
        self,
        trace_req: TraceRequest,
    ) -> Dict[str, Any]:
        """
        Create a prompt dictionary from trace request.
        
        Args:
            trace_req: TraceRequest object
            
        Returns:
            Dictionary with 'prompt_token_ids' key containing token IDs
        """
        # Map hash_ids to token_ids
        token_ids = self.mapper.map_hash_ids(trace_req.hash_ids)
        
        # Truncate to input_length if specified and valid
        if trace_req.input_length > 0:
            token_ids = token_ids[:trace_req.input_length]
        
        # Validate token_ids is a non-empty list
        # vLLM v1 engine requires prompt_token_ids to be list[int] (not empty)
        if not isinstance(token_ids, list):
            token_ids = list(token_ids)
        
        if len(token_ids) == 0:
            # If token_ids is empty, use a minimal valid prompt
            logger.warning(
                f"Request {trace_req.chat_id} has empty token_ids "
                f"(hash_ids={trace_req.hash_ids}, input_length={trace_req.input_length}). "
                f"Using default token [1]"
            )
            token_ids = [1]  # Use BOS token as default
        
        # Ensure all elements are integers
        if not all(isinstance(tid, int) for tid in token_ids):
            logger.error(
                f"Request {trace_req.chat_id} has non-integer tokens: "
                f"{[type(t).__name__ for t in token_ids[:5]]}"
            )
            token_ids = [int(t) for t in token_ids]
        
        # Return format compatible with LLM.generate()
        # Format: {"prompt_token_ids": list[int]}
        return {
            "prompt_token_ids": token_ids,
        }
    
    async def _schedule_single_request(
        self,
        trace_req: TraceRequest,
        base_time: float,
        realtime: bool,
    ) -> Any:
        """
        Schedule and execute a single request.
        
        Args:
            trace_req: TraceRequest to execute
            base_time: Base time for calculating delays
            realtime: Whether to wait for real-time intervals
            
        Returns:
            Request output from LLM
        """
        # Wait until the scheduled time if realtime mode
        if realtime:
            # 计算目标时间 = 基准时间 + trace中的时间戳
            target_time = base_time + trace_req.timestamp
            # 获取当前时间
            current_time = time.time()
            # 计算等待时间 = 目标时间 - 当前时间
            wait_time = target_time - current_time
            
            if wait_time > 0:
                await asyncio.sleep(wait_time)   # 使用 asyncio.sleep 非阻塞等待
            elif wait_time < -1.0:
                # If we're more than 1 second behind, log a warning
                logger.warning(
                    f"Request {trace_req.chat_id} is {abs(wait_time):.2f}s behind schedule"
                )
        
        # Create prompt from trace
        prompt_dict = self._create_prompt_from_trace(trace_req)
        
        # Determine max_tokens from trace output_length
        max_tokens = trace_req.output_length if trace_req.output_length > 0 else 128
        
        # Create sampling params with the specified max_tokens
        sampling_params = SamplingParams(
            temperature=self.sampling_params.temperature,
            max_tokens=max_tokens,
            ignore_eos=self.sampling_params.ignore_eos,
            top_p=self.sampling_params.top_p,
            top_k=self.sampling_params.top_k,
        )
        
        # Submit request
        # Note: LLM.generate() is synchronous, so we run it in executor
        # to avoid blocking the event loop
        # Disable tqdm progress bar for individual requests to avoid clutter
        loop = asyncio.get_event_loop()
        try:
            result = await loop.run_in_executor(
                None,
                lambda: self.llm.generate(
                    [prompt_dict],
                    sampling_params=sampling_params,
                    use_tqdm=False,  # Disable individual progress bars
                )
            )
            return result[0] if result else None
        except Exception as e:
            logger.error(
                f"Failed to generate for request {trace_req.chat_id}: {e}"
            )
            return None
    
    async def schedule_requests(
        self,
        trace_requests: List[TraceRequest],
        realtime: bool = True,
    ) -> List[Any]:
        """
        Schedule and execute all trace requests.
        
        Args:
            trace_requests: List of TraceRequest objects
            realtime: Whether to replay with real-time intervals
            
        Returns:
            List of request outputs
        """
        if not trace_requests:
            logger.warning("No trace requests to schedule")
            return []
        
        total_count = len(trace_requests)
        logger.info(
            f"Scheduling {total_count} trace requests "
            f"(realtime={'enabled' if realtime else 'disabled'})"
        )
        
        # Record base time
        base_time = time.time()
        
        # Track completed requests for progress reporting
        completed_count = 0
        completed_lock = asyncio.Lock()
        
        # Initialize progress bar if tqdm is available
        pbar = None
        if tqdm is not None:
            pbar = tqdm(
                total=total_count,
                desc="Trace benchmark",
                unit="req",
                dynamic_ncols=True,
                miniters=max(1, total_count // 100),  # Update at least every 1%
                mininterval=1.0,  # Update at least every second
            )
        
        async def update_progress():
            """Update progress bar and log status."""
            nonlocal completed_count
            async with completed_lock:
                completed_count += 1
                current_count = completed_count
                
                # Update progress bar
                if pbar is not None:
                    pbar.update(1)
                    elapsed = time.time() - base_time
                    if current_count > 0:
                        avg_time = elapsed / current_count
                        remaining = (total_count - current_count) * avg_time
                        pbar.set_postfix({
                            'elapsed': f'{elapsed:.1f}s',
                            'remaining': f'{remaining:.1f}s',
                            'avg': f'{avg_time:.3f}s/req'
                        })
                
                # Log progress at regular intervals
                log_interval = max(1, total_count // 50)  # Log every 2%
                if current_count % log_interval == 0 or current_count == total_count:
                    elapsed = time.time() - base_time
                    percentage = (current_count / total_count) * 100
                    if current_count > 0:
                        avg_time = elapsed / current_count
                        remaining = (total_count - current_count) * avg_time
                        logger.info(
                            f"Trace progress: {current_count}/{total_count} requests "
                            f"({percentage:.1f}%) | "
                            f"elapsed: {elapsed:.1f}s | "
                            f"avg: {avg_time:.3f}s/req | "
                            f"est. remaining: {remaining:.1f}s"
                        )
        
        # If not realtime, we'll submit all requests immediately
        # but still allow them to process concurrently
        if not realtime:
            logger.info("Submitting all requests concurrently (non-realtime mode)")
            
            async def process_with_progress(req):
                result = await self._schedule_single_request(req, base_time, realtime=False)
                await update_progress()
                return result
            
            tasks = [process_with_progress(req) for req in trace_requests]
            outputs = await asyncio.gather(*tasks)
            
            if pbar is not None:
                pbar.close()
            
            total_time = time.time() - base_time
            logger.info(
                f"Trace benchmark completed: {len(outputs)}/{total_count} requests "
                f"in {total_time:.2f}s (avg: {total_time/len(outputs):.3f}s per request)"
            )
            return outputs
        
        # Realtime mode: schedule requests at their specified times
        logger.info("Scheduling requests with real-time intervals")
        
        # Create tasks for all requests
        async def process_with_progress_realtime(req, req_index):
            result = await self._schedule_single_request(req, base_time, realtime=True)
            await update_progress()
            return result
        
        # 所有任务在开始时同时创建
        # 每个任务独立运行，但会按时间等待
        # 使用asyncio.create_task()实现并发
        tasks = [
            asyncio.create_task(
                process_with_progress_realtime(req, i)
            )
            for i, req in enumerate(trace_requests)
        ]
        
        # Wait for all tasks to complete
        logger.info("Executing trace requests...")
        outputs = await asyncio.gather(*tasks)
        
        if pbar is not None:
            pbar.close()
        
        total_time = time.time() - base_time
        logger.info(
            f"Trace benchmark completed: {len(outputs)}/{total_count} requests "
            f"in {total_time:.2f}s (avg: {total_time/len(outputs):.3f}s per request)"
        )
        
        return outputs

