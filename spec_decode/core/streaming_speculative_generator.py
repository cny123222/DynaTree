"""
Streaming Speculative Generator

This module implements a speculative decoding generator with StreamingLLM
integration for efficient long-context generation. It combines speculative
decoding with KV cache compression to support infinite-length generation
with constant memory usage.

Key Features:
- StreamingLLM attention sink preservation
- Sliding window KV cache compression
- Automatic compression triggering at threshold
- Draft model cache synchronization
"""

import torch
from typing import Tuple, Optional, List, Union
from transformers import PreTrainedModel, AutoTokenizer
from transformers.cache_utils import DynamicCache

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from kvcompress.methods.streaming_llm import streaming_llm_compress, evict_for_space
from spec_decode.core.speculative_generator import SpeculativeGenerator


def list_to_dynamic_cache(cache_list: List[Tuple[torch.Tensor, torch.Tensor]]) -> DynamicCache:
    """Convert list of (key, value) tuples to DynamicCache."""
    cache = DynamicCache()
    for layer_idx, (keys, values) in enumerate(cache_list):
        cache.update(keys, values, layer_idx)
    return cache


class StreamingSpeculativeGenerator(SpeculativeGenerator):
    """
    Speculative Decoding Generator with StreamingLLM KV cache compression.
    
    This generator extends SpeculativeGenerator with automatic KV cache
    compression using the StreamingLLM approach. When the cache size exceeds
    a threshold, it preserves attention sinks (initial tokens) and maintains
    a sliding window of recent tokens.
    
    This enables:
    1. Infinite-length generation without OOM
    2. Constant memory usage regardless of sequence length
    3. Maintained quality through attention sink preservation
    
    Args:
        target_model: Large model for verification
        draft_model: Small model for drafting  
        tokenizer: Tokenizer for the models
        K: Number of tokens to draft per round (default: 5)
        max_len: Maximum sequence length before truncation (default: 2048)
        device: Device to run on (default: "cuda")
        use_compile: Whether to use torch.compile (default: False)
        start_size: Number of initial tokens to keep as attention sinks (default: 4)
        recent_size: Number of recent tokens to keep (default: 1020)
        compress_threshold: Fraction of max_cache_len to trigger compression (default: 0.9)
        max_cache_len: Maximum cache length (default: 1024)
    
    Example:
        >>> generator = StreamingSpeculativeGenerator(
        ...     target_model=target_model,
        ...     draft_model=draft_model,
        ...     tokenizer=tokenizer,
        ...     K=5,
        ...     start_size=4,
        ...     recent_size=1020,
        ...     max_cache_len=1024
        ... )
        >>> 
        >>> # Generate very long text - memory stays constant
        >>> output = generator.generate(prompt, max_new_tokens=10000)
    """
    
    def __init__(
        self,
        target_model: PreTrainedModel,
        draft_model: PreTrainedModel,
        tokenizer: AutoTokenizer,
        K: int = 5,
        max_len: int = 2048,
        device: str = "cuda",
        use_compile: bool = False,
        start_size: int = 4,
        recent_size: int = 1020,
        compress_threshold: float = 0.9,
        max_cache_len: int = 1024
    ):
        super().__init__(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            K=K,
            max_len=max_len,
            device=device,
            use_compile=use_compile
        )
        
        # StreamingLLM parameters
        self.start_size = start_size
        self.recent_size = recent_size
        self.max_cache_len = max_cache_len
        self.compress_threshold = compress_threshold
        
        # Effective cache size
        self._effective_cache_size = start_size + recent_size
        
        # Statistics for compression events
        self._compression_count = 0
        self._tokens_evicted = 0
        
        # Valid token IDs for Draft model prefill
        self._valid_token_ids: Optional[torch.Tensor] = None
        
    def reset(self):
        """Reset generator state including compression stats."""
        super().reset()
        self._compression_count = 0
        self._tokens_evicted = 0
        self._valid_token_ids = None
    
    def _get_cache_length(self) -> int:
        """Get current cache length."""
        if self.target_cache is None:
            return 0
        try:
            return self.target_cache.get_seq_length()
        except:
            # Fallback for different cache implementations
            if len(self.target_cache) > 0:
                return self.target_cache[0][0].shape[2]
            return 0
    
    def _maybe_compress_cache(self):
        """Check and apply StreamingLLM compression if needed."""
        cache_len = self._get_cache_length()
        threshold = int(self.max_cache_len * self.compress_threshold)
        
        if cache_len > threshold:
            # Record tokens to evict
            target_size = self.start_size + self.recent_size
            tokens_to_evict = cache_len - target_size
            
            if tokens_to_evict > 0:
                # Apply StreamingLLM compression
                compressed_list = streaming_llm_compress(
                    self.target_cache,
                    start_size=self.start_size,
                    recent_size=self.recent_size
                )
                
                # Convert back to DynamicCache
                self.target_cache = list_to_dynamic_cache(compressed_list)
                
                # Update statistics
                self._compression_count += 1
                self._tokens_evicted += tokens_to_evict
                
                # Update valid token IDs for draft prefill
                self._update_valid_token_ids(tokens_to_evict)
    
    def _update_valid_token_ids(self, tokens_evicted: int):
        """Update valid token IDs after compression."""
        if self.current_ids is None or self.current_ids.shape[1] == 0:
            return
            
        # After compression, we keep: [0:start_size] + [-recent_size:]
        total_len = self.current_ids.shape[1]
        
        if total_len <= self._effective_cache_size:
            # No need to update if within cache size
            return
        
        # Build valid token IDs: initial + recent
        all_tokens = self.current_ids[0].tolist()
        initial_tokens = all_tokens[:self.start_size]
        recent_tokens = all_tokens[-self.recent_size:]
        
        valid_tokens = initial_tokens + recent_tokens
        self._valid_token_ids = torch.tensor(valid_tokens, device=self.device).unsqueeze(0)
    
    @torch.inference_mode()
    def _draft_tokens(self) -> torch.Tensor:
        """
        Generate K draft tokens using the draft model.
        
        Override to use valid_token_ids for prefill when cache is compressed.
        """
        # Use valid token IDs if available (after compression)
        if self._valid_token_ids is not None:
            prefill_tokens = self._valid_token_ids
        else:
            prefill_tokens = self.current_ids
        
        # Re-prefill draft model with current tokens
        with torch.inference_mode():
            draft_outputs = self.draft_model(
                input_ids=prefill_tokens,
                use_cache=True,
                return_dict=True
            )
        
        draft_cache = draft_outputs.past_key_values
        draft_logits = draft_outputs.logits[:, -1:, :]  # [1, 1, vocab]
        
        # Generate K tokens autoregressively
        draft_tokens = []
        
        for _ in range(self.K):
            next_token = draft_logits.argmax(dim=-1)  # [1, 1]
            draft_tokens.append(next_token)
            
            # Early stop on EOS
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            
            # Get next token logits
            draft_outputs = self.draft_model(
                input_ids=next_token,
                past_key_values=draft_cache,
                use_cache=True,
                return_dict=True
            )
            draft_cache = draft_outputs.past_key_values
            draft_logits = draft_outputs.logits
        
        # Stack draft tokens: [1, num_drafted]
        return torch.cat(draft_tokens, dim=1)
    
    @torch.inference_mode()
    def _update_cache_and_logits(
        self,
        num_accepted: int,
        accepted_tokens: torch.Tensor,
        all_accepted: bool,
        original_len: int,
        verify_outputs=None
    ):
        """
        Update cache and logits, then check for compression.
        
        Extends parent implementation with StreamingLLM compression check.
        """
        # Call parent implementation
        super()._update_cache_and_logits(
            num_accepted, accepted_tokens, all_accepted, original_len, verify_outputs
        )
        
        # Check if compression is needed after cache update
        self._maybe_compress_cache()
    
    def get_stats(self) -> dict:
        """Get generation statistics including compression stats."""
        stats = super().get_stats()
        stats.update({
            "compression_count": self._compression_count,
            "tokens_evicted": self._tokens_evicted,
            "current_cache_len": self._get_cache_length(),
            "max_cache_len": self.max_cache_len,
            "effective_cache_size": self._effective_cache_size
        })
        return stats
    
    def get_compression_config(self) -> dict:
        """Get StreamingLLM compression configuration."""
        return {
            "start_size": self.start_size,
            "recent_size": self.recent_size,
            "max_cache_len": self.max_cache_len,
            "compress_threshold": self.compress_threshold,
            "effective_cache_size": self._effective_cache_size
        }


class StreamingSpeculativeGeneratorV2(StreamingSpeculativeGenerator):
    """
    Optimized version with proactive eviction for incoming tokens.
    
    This version evicts tokens before they overflow, rather than after,
    which can improve cache hit rates and reduce compression overhead.
    """
    
    def _evict_for_incoming(self, num_incoming: int):
        """Proactively evict tokens to make space for incoming tokens."""
        cache_len = self._get_cache_length()
        
        # Check if eviction is needed for incoming tokens
        if cache_len + num_incoming <= self.max_cache_len:
            return
        
        # Evict to make space
        evicted_list = evict_for_space(
            self.target_cache,
            num_coming=num_incoming,
            start_size=self.start_size,
            recent_size=self.recent_size
        )
        
        # Convert back to DynamicCache
        self.target_cache = list_to_dynamic_cache(evicted_list)
        
        # Update statistics
        self._compression_count += 1
        new_cache_len = self._get_cache_length()
        self._tokens_evicted += (cache_len - new_cache_len)
    
    @torch.inference_mode()
    def _verify_tokens(self, draft_tokens: torch.Tensor):
        """Verify tokens with proactive eviction."""
        num_draft = draft_tokens.shape[1]
        
        # Proactively evict before verification to ensure space
        self._evict_for_incoming(num_draft + 1)  # +1 for potential bonus token
        
        # Call parent verification
        return super()._verify_tokens(draft_tokens)


__all__ = [
    'StreamingSpeculativeGenerator',
    'StreamingSpeculativeGeneratorV2',
    'list_to_dynamic_cache'
]

