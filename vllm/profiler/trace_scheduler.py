# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trace-based request scheduler for operator benchmarking."""

import asyncio
import threading
import time
from typing import Any, Dict, List, Optional

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

import torch

from vllm import LLM, SamplingParams
from vllm.logger import init_logger
from vllm.profiler.hash_id_mapper import HashIDMapper
from vllm.profiler.trace_loader import TraceRequest

logger = init_logger(__name__)

# Maximum output tokens to prevent extremely long sequences
MAX_OUTPUT_TOKENS = 4096


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
        
        # Determine max_tokens from trace output_length with safety cap
        requested_max_tokens = trace_req.output_length if trace_req.output_length > 0 else 128
        max_tokens = min(requested_max_tokens, MAX_OUTPUT_TOKENS)
        
        if requested_max_tokens > MAX_OUTPUT_TOKENS:
            logger.warning(
                f"Request {trace_req.chat_id} output length {requested_max_tokens} "
                f"exceeds limit, capped at {MAX_OUTPUT_TOKENS}"
            )
        
        # Check GPU memory before starting long requests
        if torch.cuda.is_available() and max_tokens > 512:
            try:
                free_mem_gb = torch.cuda.mem_get_info()[0] / 1024**3
                logger.info(
                    f"Request {trace_req.chat_id}: GPU free memory = {free_mem_gb:.2f} GB "
                    f"(input={trace_req.input_length}, output={max_tokens})"
                )
                if free_mem_gb < 2.0:
                    logger.warning(
                        f"⚠️  Low GPU memory ({free_mem_gb:.2f} GB), "
                        f"request may be slow or fail"
                    )
            except Exception as e:
                logger.debug(f"Could not check GPU memory: {e}")
        
        # Log request start (especially useful for debugging hangs)
        logger.debug(
            f"Starting request {trace_req.chat_id}: "
            f"input_length={trace_req.input_length}, max_tokens={max_tokens}"
        )
        
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
        
        # Calculate timeout with conservative estimates for long sequences
        # Get TP size for overhead calculation (try multiple methods)
        tp_size = 1  # Default
        try:
            # Method 1: From parallel_config (most reliable)
            if hasattr(self.llm.llm_engine, 'parallel_config'):
                tp_size = self.llm.llm_engine.parallel_config.tensor_parallel_size
            # Method 2: From model_config
            elif hasattr(self.llm.llm_engine, 'model_config'):
                if hasattr(self.llm.llm_engine.model_config, 'tensor_parallel_size'):
                    tp_size = self.llm.llm_engine.model_config.tensor_parallel_size
        except Exception as e:
            logger.debug(f"Could not get TP size from engine: {e}")
            # Method 3: From environment variable
            try:
                import os
                tp_env = os.environ.get('VLLM_TENSOR_PARALLEL_SIZE')
                if tp_env:
                    tp_size = int(tp_env)
            except:
                pass
        
        # Check for timeout override via environment variable
        import os
        timeout_override = os.environ.get('VLLM_BENCHMARK_TIMEOUT')
        
        if timeout_override:
            if timeout_override.lower() == 'none' or timeout_override == '0':
                # Extremely large timeout (effectively infinite)
                timeout = 86400  # 24 hours
                logger.warning(
                    f"⚠️  Request {trace_req.chat_id}: Timeout disabled "
                    f"(VLLM_BENCHMARK_TIMEOUT={timeout_override})"
                )
            else:
                try:
                    timeout = float(timeout_override)
                    logger.info(
                        f"Request {trace_req.chat_id}: Using timeout override "
                        f"from env: {timeout:.0f}s"
                    )
                except ValueError:
                    logger.warning(
                        f"Invalid VLLM_BENCHMARK_TIMEOUT value: {timeout_override}, "
                        f"using calculated timeout"
                    )
                    timeout_override = None
        
        if not timeout_override:
            # Ultra-conservative timeout calculation for long sequences:
            # - Input processing: 0.2s per token (doubled)
            # - Output generation: 2.0s per token for short, 3.0s for long (20-30x more conservative!)
            # - Base buffer: 600s (10 minutes)
            # - TP overhead: multiply by (1 + tp_size * 0.3)
            
            # For very long sequences, use even more conservative estimates
            if max_tokens > 1000:
                output_time_per_token = 3.0  # 3 seconds per token for very long sequences
                base_buffer = 1200  # 20 minutes buffer
            elif max_tokens > 500:
                output_time_per_token = 2.5  # 2.5 seconds per token
                base_buffer = 900  # 15 minutes buffer
            else:
                output_time_per_token = 2.0  # 2 seconds per token
                base_buffer = 600  # 10 minutes buffer
            
            estimated_time = (
                trace_req.input_length * 0.2 +
                max_tokens * output_time_per_token +
                base_buffer
            )
            
            # TP overhead increases with TP size
            tp_overhead = 1.0 + (tp_size * 0.3)
            
            # Minimum timeout based on output length
            if max_tokens > 1000:
                min_timeout = 3600  # 1 hour for very long sequences
            elif max_tokens > 500:
                min_timeout = 1800  # 30 minutes for long sequences
            else:
                min_timeout = 900  # 15 minutes for normal sequences
            
            timeout = max(min_timeout, estimated_time * tp_overhead)
        
        logger.info(
            f"Request {trace_req.chat_id}: timeout={timeout:.0f}s ({timeout/60:.1f}min) | "
            f"input={trace_req.input_length}, output={max_tokens}, TP={tp_size}"
        )
        
        # Start heartbeat monitor thread
        heartbeat_stop = threading.Event()
        
        def heartbeat_monitor():
            """Log heartbeat messages every 30 seconds to show request is still running."""
            start = time.time()
            interval = 30
            while not heartbeat_stop.is_set():
                if heartbeat_stop.wait(interval):
                    break
                elapsed = time.time() - start
                logger.info(
                    f"  ⏱ Request {trace_req.chat_id} still running "
                    f"({elapsed:.0f}s / {timeout:.0f}s)..."
                )
        
        heartbeat_thread = threading.Thread(target=heartbeat_monitor, daemon=True)
        heartbeat_thread.start()
        
        try:
            result = await asyncio.wait_for(
                loop.run_in_executor(
                None,
                lambda: self.llm.generate(
                    [prompt_dict],
                    sampling_params=sampling_params,
                    use_tqdm=False,  # Disable individual progress bars
                )
                ),
                timeout=timeout
            )
            heartbeat_stop.set()
            return result[0] if result else None
        except asyncio.TimeoutError:
            heartbeat_stop.set()
            logger.error(
                f"\n{'='*80}\n"
                f"❌ REQUEST TIMEOUT\n"
                f"{'='*80}\n"
                f"  Request ID: {trace_req.chat_id}\n"
                f"  Input length: {trace_req.input_length} tokens\n"
                f"  Output length: {max_tokens} tokens\n"
                f"  TP size detected: {tp_size}\n"
                f"  Timeout used: {timeout:.1f}s ({timeout/60:.1f} minutes)\n"
                f"  \n"
                f"  💡 Possible solutions:\n"
                f"     1. Set VLLM_BENCHMARK_TIMEOUT=7200 (2 hours) or 'none' (disable)\n"
                f"     2. Reduce max output length in trace data\n"
                f"     3. Check GPU performance (memory, compute utilization)\n"
                f"     4. Reduce TP size if communication overhead is high\n"
                f"{'='*80}\n"
            )
            return None
        except Exception as e:
            heartbeat_stop.set()
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
        
        # Limit concurrent requests to avoid GPU OOM
        max_concurrent = 5  # Maximum concurrent requests (adjust based on GPU memory)
        semaphore = asyncio.Semaphore(max_concurrent)
        logger.info(f"  Maximum concurrent requests: {max_concurrent}")
        
        # Create tasks for all requests with concurrency limit
        async def process_with_progress_realtime(req, req_index):
            async with semaphore:  # Limit concurrency
                result = await self._schedule_single_request(req, base_time, realtime=True)
            await update_progress()
            return result
        
        # 所有任务在开始时同时创建
        # 但通过Semaphore限制同时执行的数量
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

