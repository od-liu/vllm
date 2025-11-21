# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hash ID to token ID mapping for trace-based benchmarking."""

import random
from typing import List

from vllm.logger import init_logger

logger = init_logger(__name__)


class HashIDMapper:
    """Maps hash IDs to deterministic token IDs.
    
    Each hash_id represents a 16-token block. This mapper generates
    a fixed sequence of 16 token IDs for each unique hash_id, ensuring
    reproducibility while maintaining realistic token distributions.
    """
    
    def __init__(self, vocab_size: int, seed: int = 42):
        """
        Initialize hash ID mapper.
        
        Args:
            vocab_size: Size of the vocabulary (from model config)
            seed: Base seed for random number generation
        """
        self.vocab_size = vocab_size
        self.base_seed = seed
        
        # Cache for already mapped hash_ids
        self._cache: dict[int, List[int]] = {}
        
        # Validate vocab_size
        if vocab_size < 2:
            raise ValueError(f"vocab_size must be >= 2, got {vocab_size}")
        
        logger.info(
            f"Initialized HashIDMapper with vocab_size={vocab_size}, seed={seed}"
        )
    
    def map_hash_id(self, hash_id: int) -> List[int]:
        """
        Map a single hash_id to 16 token IDs.
        
        Args:
            hash_id: The hash ID to map
            
        Returns:
            List of 16 token IDs in range [1, vocab_size-1]
        """
        # Check cache first
        if hash_id in self._cache:
            return self._cache[hash_id]
        
        # Generate deterministic token IDs for this hash_id
        # Use hash_id as part of the seed to ensure determinism
        rng = random.Random(self.base_seed + hash_id)
        
        # Generate 16 token IDs in range [1, vocab_size-1]
        # Avoid 0 and special tokens at the beginning/end of vocab
        token_ids = [
            rng.randint(1, self.vocab_size - 1)
            for _ in range(16)
        ]
        
        # Cache the result
        self._cache[hash_id] = token_ids
        
        return token_ids
    
    def map_hash_ids(self, hash_ids: List[int]) -> List[int]:
        """
        Map a list of hash_ids to a flat list of token IDs.
        
        Args:
            hash_ids: List of hash IDs (each represents 16 tokens)
            
        Returns:
            Flattened list of token IDs
        """
        token_ids = []
        for hash_id in hash_ids:
            token_ids.extend(self.map_hash_id(hash_id))
        
        return token_ids
    
    def get_cache_size(self) -> int:
        """Get the number of cached hash_id mappings."""
        return len(self._cache)
    
    def clear_cache(self) -> None:
        """Clear the mapping cache."""
        self._cache.clear()
        logger.debug("Cleared HashIDMapper cache")

