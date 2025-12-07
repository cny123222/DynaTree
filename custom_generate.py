"""
Custom Generation Function with KV Cache Compression

This module implements a custom text generation loop that applies
KnormPress (L2 norm-based) KV cache compression after each forward pass.

This is designed to work with the Pythia-70M model (GPT-NeoX architecture).
"""

import torch
from typing import Optional, List, Tuple
import time
from kv_compress import l2_compress
from transformers import DynamicCache


def generate_with_compression(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    keep_ratio: float = 1.0,
    prune_after: int = 512,
    skip_layers: List[int] = [0],
    do_sample: bool = False,
    temperature: float = 1.0,
    top_k: int = 50,
    top_p: float = 1.0,
    device: torch.device = None,
) -> torch.Tensor:
    """
    Generate text with KV cache compression applied after each forward pass.
    
    Args:
        model: The language model (e.g., GPTNeoXForCausalLM)
        tokenizer: The tokenizer
        input_ids: Input token IDs, shape (batch_size, seq_len)
        max_new_tokens: Maximum number of tokens to generate
        keep_ratio: Ratio of KV cache to keep (0.0-1.0). 1.0 means no compression.
        prune_after: Only compress if cache size exceeds this threshold
        skip_layers: List of layer indices to skip compression (e.g., [0] for first layer)
        do_sample: Whether to use sampling (True) or greedy decoding (False)
        temperature: Sampling temperature (only used if do_sample=True)
        top_k: Top-k sampling parameter (only used if do_sample=True)
        top_p: Top-p (nucleus) sampling parameter (only used if do_sample=True)
        device: Device to run on
    
    Returns:
        torch.Tensor: Generated token IDs, shape (batch_size, seq_len + max_new_tokens)
    """
    
    if device is None:
        device = input_ids.device
    
    model.eval()
    
    # Initialize
    past_key_values = None
    generated_tokens = input_ids.clone()
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            # Prepare input for this step
            if step == 0:
                # First step: use full input
                model_input = input_ids
            else:
                # Subsequent steps: only use the last generated token
                model_input = generated_tokens[:, -1:].to(device)
            
            # Forward pass
            outputs = model(
                model_input,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            
            # Get logits for the last token
            next_token_logits = outputs.logits[:, -1, :]
            
            # Apply temperature if sampling
            if do_sample and temperature != 1.0:
                next_token_logits = next_token_logits / temperature
            
            # Generate next token
            if do_sample:
                # Apply top-k filtering
                if top_k > 0:
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p filtering
                if top_p < 1.0:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = 0
                    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Sample from the filtered distribution
                probs = torch.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
            else:
                # Greedy decoding
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Append generated token
            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
            
            # Get the KV cache and apply compression
            past_key_values = outputs.past_key_values
            
            # Apply KV cache compression using KnormPress
            if keep_ratio < 1.0 and past_key_values is not None:
                # Convert to list for compression
                if hasattr(past_key_values, 'to_legacy_cache'):
                    # It's a DynamicCache object, convert to legacy format
                    kv_list = past_key_values.to_legacy_cache()
                else:
                    kv_list = past_key_values
                
                # Apply compression
                compressed_kv = l2_compress(
                    kv_list,
                    keep_ratio=keep_ratio,
                    prune_after=prune_after,
                    skip_layers=skip_layers,
                )
                
                # Convert back to DynamicCache
                new_cache = DynamicCache()
                for layer_keys, layer_values in compressed_kv:
                    new_cache.update(layer_keys, layer_values, 0)  # layer_idx is managed internally
                
                past_key_values = new_cache
            
            # Check for EOS token
            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                break
    
    return generated_tokens


def generate_with_compression_and_timing(
    model,
    tokenizer,
    input_ids: torch.Tensor,
    max_new_tokens: int = 100,
    keep_ratio: float = 1.0,
    prune_after: int = 512,
    skip_layers: List[int] = [0],
    device: torch.device = None,
) -> dict:
    """
    Generate text with KV cache compression and detailed timing information.
    
    This function tracks TTFT (Time To First Token) and per-token generation times
    for performance benchmarking.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        input_ids: Input token IDs
        max_new_tokens: Maximum number of tokens to generate
        keep_ratio: KV cache compression ratio
        prune_after: Compression threshold
        skip_layers: Layers to skip compression
        device: Device to run on
    
    Returns:
        dict: Contains 'output_ids', 'ttft', 'token_times', and other metrics
    """
    
    if device is None:
        device = input_ids.device
    
    model.eval()
    
    # Initialize
    past_key_values = None
    generated_tokens = input_ids.clone()
    token_times = []
    ttft = None
    
    # Start timing
    start_time = time.time()
    
    with torch.no_grad():
        for step in range(max_new_tokens):
            step_start = time.time()
            
            # Prepare input for this step
            if step == 0:
                model_input = input_ids
            else:
                model_input = generated_tokens[:, -1:].to(device)
            
            # Forward pass
            outputs = model(
                model_input,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            
            # Get next token (greedy decoding)
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            
            # Record TTFT (time to first token)
            if step == 0:
                ttft = time.time() - start_time
            
            # Append generated token
            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
            
            # Apply KV cache compression
            past_key_values = outputs.past_key_values
            if keep_ratio < 1.0 and past_key_values is not None:
                # Convert to list for compression
                if hasattr(past_key_values, 'to_legacy_cache'):
                    kv_list = past_key_values.to_legacy_cache()
                else:
                    kv_list = past_key_values
                
                # Apply compression
                compressed_kv = l2_compress(
                    kv_list,
                    keep_ratio=keep_ratio,
                    prune_after=prune_after,
                    skip_layers=skip_layers,
                )
                
                # Convert back to DynamicCache
                new_cache = DynamicCache()
                for layer_keys, layer_values in compressed_kv:
                    new_cache.update(layer_keys, layer_values, 0)
                
                past_key_values = new_cache
            
            # Record time for this token
            token_times.append(time.time() - step_start)
            
            # Check for EOS
            if tokenizer.eos_token_id is not None and next_token.item() == tokenizer.eos_token_id:
                break
    
    total_time = time.time() - start_time
    num_generated = generated_tokens.shape[1] - input_ids.shape[1]
    
    return {
        'output_ids': generated_tokens,
        'ttft': ttft,
        'token_times': token_times,
        'total_time': total_time,
        'num_generated_tokens': num_generated,
        'tpot': total_time / num_generated if num_generated > 0 else 0,
        'throughput': num_generated / total_time if total_time > 0 else 0,
    }


def batch_generate_with_compression(
    model,
    tokenizer,
    input_texts: List[str],
    max_new_tokens: int = 100,
    keep_ratio: float = 1.0,
    prune_after: int = 512,
    skip_layers: List[int] = [0],
    device: torch.device = None,
) -> List[str]:
    """
    Generate text for multiple prompts with KV cache compression.
    
    Note: Currently only supports batch_size=1 due to KV cache compression complexity.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        input_texts: List of input text prompts
        max_new_tokens: Maximum number of tokens to generate per prompt
        keep_ratio: KV cache compression ratio
        prune_after: Compression threshold
        skip_layers: Layers to skip compression
        device: Device to run on
    
    Returns:
        List[str]: Generated texts
    """
    
    if device is None:
        device = next(model.parameters()).device
    
    generated_texts = []
    
    for input_text in input_texts:
        # Tokenize input
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)
        
        # Generate
        output_ids = generate_with_compression(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            keep_ratio=keep_ratio,
            prune_after=prune_after,
            skip_layers=skip_layers,
            device=device,
        )
        
        # Decode
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        generated_texts.append(generated_text)
    
    return generated_texts

