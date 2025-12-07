"""
KV Cache Compression using L2 Norm-Based Strategy (KnormPress)

This module implements the core compression algorithm from the paper:
"A Simple and Effective L2 Norm-Based Strategy for KV Cache Compression"

Key insight: Tokens with low L2 norm in their key embeddings correlate
strongly with high attention scores. We keep these important tokens
and remove the less important ones (high L2 norm).
"""

from typing import List, Tuple, Union
from math import ceil
import torch
from transformers import DynamicCache


def l2_compress(
    past_key_values: Union[List[Tuple[torch.Tensor, torch.Tensor]], DynamicCache],
    keep_ratio: float = 1.0,
    prune_after: int = 1000,
    skip_layers: List[int] = [0, 1],
    **kwargs
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compress KV cache by keeping tokens with the lowest L2 norms.
    
    The algorithm:
    1. Compute L2 norm of each token's key embedding
    2. Sort tokens by norm (ascending order, low = important)
    3. Keep top keep_ratio percentage with lowest norms
    4. CRITICAL: Sort selected indices to maintain temporal order
    
    Args:
        past_key_values: KV cache, either as list of (key, value) tuples
                        or as DynamicCache object.
                        Shape: (batch_size, num_heads, seq_len, head_dim)
        keep_ratio: Fraction of tokens to keep (0.0 to 1.0).
                   Default is 1.0 (no compression).
        prune_after: Only compress if sequence length > this value.
                    Default is 1000 (matching paper recommendation).
        skip_layers: Layer indices to skip compression.
                    Default is [0, 1] (matching paper).
    
    Returns:
        Compressed KV cache as list of (key, value) tuples.
    
    Example:
        >>> compressed = l2_compress(
        ...     past_key_values,
        ...     keep_ratio=0.8,      # Keep 80%
        ...     prune_after=1000,    # Only compress after 1000 tokens
        ...     skip_layers=[0, 1]   # Skip first 2 layers
        ... )
    """
    # Convert DynamicCache to list format if needed
    if hasattr(past_key_values, 'to_legacy_cache'):
        past_key_values = past_key_values.to_legacy_cache()
    
    # Ensure it's a list (mutable)
    past_key_values = list(past_key_values)
    
    # If keep_ratio is 1.0, return unchanged
    if keep_ratio >= 1.0:
        return past_key_values
    
    # Process each layer
    for layer_idx, (keys, values) in enumerate(past_key_values):
        seq_len = keys.size(2)
        
        # Skip if below threshold
        if seq_len <= prune_after:
            continue
        
        # Skip specified layers
        if layer_idx in skip_layers:
            continue
        
        # Calculate tokens to keep
        tokens_to_keep = ceil(keep_ratio * seq_len)
        
        # Skip if no compression needed
        if tokens_to_keep >= seq_len:
            continue
        
        # Get dimensions
        batch_size, num_heads, seq_len, head_dim = keys.shape
        
        # Compute L2 norm for each token
        # Shape: (batch_size, num_heads, seq_len)
        token_norms = torch.norm(keys, p=2, dim=-1)
        
        # Get indices sorted by norm (ascending = most important first)
        sorted_indices = token_norms.argsort(dim=-1)
        
        # Select top tokens_to_keep most important tokens
        indices_to_keep = sorted_indices[:, :, :tokens_to_keep]
        
        # CRITICAL: Sort indices to maintain temporal order
        # This preserves positional encoding correctness
        indices_to_keep_sorted, _ = torch.sort(indices_to_keep, dim=-1)
        
        # Expand indices for gather
        indices_expanded = indices_to_keep_sorted.unsqueeze(-1).expand(
            batch_size, num_heads, tokens_to_keep, head_dim
        )
        
        # Gather keys and values
        compressed_keys = torch.gather(keys, dim=2, index=indices_expanded)
        compressed_values = torch.gather(values, dim=2, index=indices_expanded)
        
        # Update cache
        past_key_values[layer_idx] = (compressed_keys, compressed_values)
    
    return past_key_values


def to_dynamic_cache(
    past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]
) -> DynamicCache:
    """
    Convert list of (key, value) tuples to DynamicCache object.
    
    Args:
        past_key_values: List of (key, value) tuples
    
    Returns:
        DynamicCache object
    """
    cache = DynamicCache()
    for layer_idx, (keys, values) in enumerate(past_key_values):
        cache.update(keys, values, layer_idx)
    return cache


def get_cache_size_mb(
    past_key_values: Union[List[Tuple[torch.Tensor, torch.Tensor]], DynamicCache]
) -> float:
    """
    Calculate KV cache size in megabytes.
    
    Args:
        past_key_values: KV cache
    
    Returns:
        Size in MB
    """
    if hasattr(past_key_values, 'to_legacy_cache'):
        past_key_values = past_key_values.to_legacy_cache()
    
    total_size = 0
    for keys, values in past_key_values:
        total_size += keys.element_size() * keys.nelement()
        total_size += values.element_size() * values.nelement()
    return total_size / (1024 ** 2)


def get_cache_info(
    past_key_values: Union[List[Tuple[torch.Tensor, torch.Tensor]], DynamicCache]
) -> dict:
    """
    Get detailed information about KV cache.
    
    Args:
        past_key_values: KV cache
    
    Returns:
        Dict with cache information
    """
    if hasattr(past_key_values, 'to_legacy_cache'):
        past_key_values = past_key_values.to_legacy_cache()
    
    if not past_key_values:
        return {"num_layers": 0, "seq_lengths": [], "total_size_mb": 0}
    
    seq_lengths = [keys.size(2) for keys, values in past_key_values]
    
    return {
        "num_layers": len(past_key_values),
        "seq_lengths": seq_lengths,
        "min_seq_len": min(seq_lengths),
        "max_seq_len": max(seq_lengths),
        "avg_seq_len": sum(seq_lengths) / len(seq_lengths),
        "total_size_mb": get_cache_size_mb(past_key_values)
    }

