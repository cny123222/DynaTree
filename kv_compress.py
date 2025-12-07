"""
KV Cache Compression Module for KnormPress Algorithm

This module implements the L2 norm-based KV cache compression strategy.
The key insight is that tokens with low L2 norm in their key embeddings
correlate strongly with high attention scores and should be retained.

Reference: "A Simple and Effective L2 Norm-Based Strategy for KV Cache Compression"
"""

from typing import List, Tuple
from math import ceil
import torch


def l2_compress(
    past_key_values: List[Tuple[torch.Tensor, torch.Tensor]],
    keep_ratio: float = 1.0,
    prune_after: int = 2048,
    skip_layers: List[int] = [],
    **kwargs
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    """
    Compress KV cache by keeping tokens with the lowest L2 norms in their key embeddings.
    
    The algorithm works as follows:
    1. For each layer's KV cache, compute the L2 norm of each token's key embedding
    2. Sort tokens by their norm in ascending order (low norm = high importance)
    3. Keep only the top keep_ratio percentage of tokens with lowest norms
    4. Only apply compression if cache size exceeds prune_after threshold
    
    Args:
        past_key_values: List of tuples, where each tuple contains (keys, values).
                        Keys and values have shape (batch_size, num_heads, seq_len, head_dim)
        keep_ratio: Ratio of tokens to keep (0.0 to 1.0). Default is 1.0 (no compression).
                   For example, 0.8 means keep 80% of tokens, compress 20%.
        prune_after: Only prune if sequence length exceeds this value. Default is 2048.
        skip_layers: List of layer indices to skip compression. Default is empty list.
                    Common practice is to skip first layer (layer 0).
        **kwargs: Additional arguments (for compatibility)
    
    Returns:
        past_key_values: Compressed KV cache with same structure as input
    
    Example:
        >>> # Compress to 80% of original size
        >>> compressed_cache = l2_compress(
        ...     past_key_values, 
        ...     keep_ratio=0.8, 
        ...     prune_after=512,
        ...     skip_layers=[0]
        ... )
    """
    
    # Convert to list if it's a tuple (for mutability)
    past_key_values = list(past_key_values)
    
    # Process each layer's KV cache
    for layer_idx, (keys, values) in enumerate(past_key_values):
        
        # Skip compression if sequence length is below threshold
        seq_len = keys.size(2)
        if seq_len < prune_after:
            continue
        
        # Skip compression for specified layers
        if layer_idx in skip_layers:
            continue
        
        # Calculate how many tokens to keep
        tokens_to_keep = ceil(keep_ratio * seq_len)
        
        # If keep_ratio is 1.0, no compression needed
        if tokens_to_keep >= seq_len:
            continue
        
        # Get dimensions
        # keys shape: (batch_size, num_heads, seq_len, head_dim)
        batch_size, num_heads, seq_len, head_dim = keys.shape
        
        # Compute L2 norm for each token across the head dimension
        # Result shape: (batch_size, num_heads, seq_len)
        token_norms = torch.norm(keys, p=2, dim=-1)
        
        # Sort tokens by norm in ascending order (lowest norms first)
        # These are the indices of tokens sorted by importance
        # sorted_indices shape: (batch_size, num_heads, seq_len)
        sorted_indices = token_norms.argsort(dim=-1)
        
        # Get indices of top tokens_to_keep most important tokens
        # Shape: (batch_size, num_heads, tokens_to_keep)
        indices_to_keep = sorted_indices[:, :, :tokens_to_keep]
        
        # CRITICAL: Sort these indices to maintain temporal order!
        # This preserves the original token sequence, which is essential
        # for positional encodings to work correctly
        indices_to_keep_sorted, _ = torch.sort(indices_to_keep, dim=-1)
        
        # Expand indices to match the full key/value tensor shape
        # Shape: (batch_size, num_heads, tokens_to_keep, head_dim)
        indices_expanded = indices_to_keep_sorted.unsqueeze(-1).expand(
            batch_size, num_heads, tokens_to_keep, head_dim
        )
        
        # Gather keys and values using the sorted indices
        # This keeps the tokens in their original temporal order
        compressed_keys = torch.gather(keys, dim=2, index=indices_expanded)
        compressed_values = torch.gather(values, dim=2, index=indices_expanded)
        
        # Update the KV cache for this layer
        past_key_values[layer_idx] = (compressed_keys, compressed_values)
    
    return past_key_values


def get_cache_size_mb(past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]) -> float:
    """
    Calculate the total memory size of KV cache in MB.
    
    Args:
        past_key_values: List of tuples containing (keys, values)
    
    Returns:
        float: Total size in megabytes
    """
    total_size = 0
    for keys, values in past_key_values:
        total_size += keys.element_size() * keys.nelement()
        total_size += values.element_size() * values.nelement()
    return total_size / (1024 ** 2)


def get_cache_info(past_key_values: List[Tuple[torch.Tensor, torch.Tensor]]) -> dict:
    """
    Get detailed information about the KV cache structure.
    
    Args:
        past_key_values: List of tuples containing (keys, values)
    
    Returns:
        dict: Information including num_layers, seq_len per layer, total size, etc.
    """
    if not past_key_values:
        return {"num_layers": 0, "seq_lengths": [], "total_size_mb": 0}
    
    seq_lengths = [keys.size(2) for keys, values in past_key_values]
    total_size_mb = get_cache_size_mb(past_key_values)
    
    return {
        "num_layers": len(past_key_values),
        "seq_lengths": seq_lengths,
        "min_seq_len": min(seq_lengths),
        "max_seq_len": max(seq_lengths),
        "avg_seq_len": sum(seq_lengths) / len(seq_lengths),
        "total_size_mb": total_size_mb
    }

