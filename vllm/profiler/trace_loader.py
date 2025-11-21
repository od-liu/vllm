# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Trace data loader for operator benchmarking."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from vllm.logger import init_logger

logger = init_logger(__name__)


@dataclass
class TraceRequest:
    """Represents a single request from a trace file."""
    
    chat_id: int
    """Randomized chat identifier."""
    
    parent_chat_id: int
    """Parent chat ID, -1 for root requests."""
    
    timestamp: float
    """Relative timestamp in seconds since trace start."""
    
    input_length: int
    """Input token count."""
    
    output_length: int
    """Output token count."""
    
    type: str
    """Request type: text, search, image, file, api."""
    
    turn: int
    """Conversation turn number."""
    
    hash_ids: List[int]
    """Salted SipHash blocks (16 tokens per block)."""


class TraceLoader:
    """Loads and filters trace data from JSONL files."""
    
    def __init__(
        self,
        trace_path: str,
        time_range_minutes: Optional[Tuple[float, float]] = None,
    ):
        """
        Initialize trace loader.
        
        Args:
            trace_path: Path to JSONL trace file
            time_range_minutes: Optional tuple (start_min, end_min) to filter requests.
                              If None, loads all requests.
        """
        self.trace_path = Path(trace_path)
        self.time_range_minutes = time_range_minutes
        
        if not self.trace_path.exists():
            raise FileNotFoundError(f"Trace file not found: {trace_path}")
        
        if not self.trace_path.is_file():
            raise ValueError(f"Trace path is not a file: {trace_path}")
    
    def load_requests(self) -> List[TraceRequest]:
        """
        Load all requests from trace file.
        
        Returns:
            List of TraceRequest objects, sorted by timestamp.
        """
        logger.info(f"Loading trace data from {self.trace_path}")
        
        requests = []
        line_count = 0
        
        try:
            with open(self.trace_path, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        data = json.loads(line)
                        request = TraceRequest(
                            chat_id=data.get("chat_id", -1),
                            parent_chat_id=data.get("parent_chat_id", -1),
                            timestamp=data.get("timestamp", 0.0),
                            input_length=data.get("input_length", 0),
                            output_length=data.get("output_length", 0),
                            type=data.get("type", "text"),
                            turn=data.get("turn", 1),
                            hash_ids=data.get("hash_ids", []),
                        )
                        requests.append(request)
                        line_count += 1
                    except json.JSONDecodeError as e:
                        logger.warning(
                            f"Skipping invalid JSON at line {line_num}: {e}"
                        )
                        continue
                    except KeyError as e:
                        logger.warning(
                            f"Skipping line {line_num} with missing field: {e}"
                        )
                        continue
        
        except Exception as e:
            raise RuntimeError(f"Failed to load trace file: {e}") from e
        
        logger.info(f"Loaded {line_count} requests from trace file")
        
        # Sort by timestamp
        requests.sort(key=lambda r: r.timestamp)
        
        # Apply time range filter if specified
        if self.time_range_minutes is not None:
            requests = self.filter_by_time_range(requests)
        
        logger.info(
            f"Final request count after filtering: {len(requests)}"
        )
        
        if len(requests) == 0:
            logger.warning("No requests loaded after filtering!")
        
        return requests
    
    def filter_by_time_range(
        self,
        requests: List[TraceRequest],
    ) -> List[TraceRequest]:
        """
        Filter requests by time range.
        
        Args:
            requests: List of TraceRequest objects
            
        Returns:
            Filtered list of requests within the specified time range.
        """
        if self.time_range_minutes is None:
            return requests
        
        start_min, end_min = self.time_range_minutes
        start_seconds = start_min * 60.0
        end_seconds = end_min * 60.0
        
        filtered = [
            req for req in requests
            if start_seconds <= req.timestamp < end_seconds
        ]
        
        logger.info(
            f"Filtered requests: {len(requests)} -> {len(filtered)} "
            f"(time range: {start_min:.1f} - {end_min:.1f} minutes)"
        )
        
        return filtered

