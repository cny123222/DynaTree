"""
Benchmark Module for KnormPress

This module provides functions to measure generation performance metrics:
- TTFT (Time To First Token)
- TPOT (Time Per Output Token)
- Throughput (tokens/sec)
- PPL (Perplexity)
- Accuracy (Next token prediction accuracy)
"""

import time
from typing import Dict, List, Optional, Union
import torch
from transformers import DynamicCache

from .compress import l2_compress, to_dynamic_cache
from .evaluate import evaluate_with_compression


def measure_generation_metrics(
    model,
    tokenizer,
    text: str,
    max_new_tokens: int = 1000,
    keep_ratio: float = 1.0,
    prune_after: int = 1024,  # Much smaller for generation metrics
    skip_layers: List[int] = [0, 1],
    device: Optional[torch.device] = None,
    max_input_tokens: int = 3000,
) -> Dict[str, float]:
    """
    Measure generation performance metrics with KV cache compression.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text (prompt)
        max_new_tokens: Number of tokens to generate
        keep_ratio: Fraction of tokens to keep (0.0 to 1.0)
        prune_after: Only compress if cache length > this value
        skip_layers: Layer indices to skip compression
        device: Device to use
        max_input_tokens: Maximum input tokens (to prevent OOM)
    
    Returns:
        Dict containing:
        - ttft: Time to first token (seconds)
        - tpot: Time per output token (seconds)
        - throughput: Tokens per second
        - total_time: Total generation time (seconds)
        - num_tokens: Number of tokens generated
        - input_length: Input sequence length
    """
    if device is None:
        device = next(model.parameters()).device
    
    # Tokenize input (truncate to prevent OOM)
    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_ids = input_ids[:, :max_input_tokens].to(device)
    input_length = input_ids.shape[1]
    
    # Pad token handling
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    model.eval()
    
    generated_tokens = []
    past_key_values = None
    ttft = None
    
    total_start = time.perf_counter()
    
    with torch.inference_mode():
        # First forward pass (prefill) - measure TTFT
        first_start = time.perf_counter()
        outputs = model(
            input_ids,
            use_cache=True,
            return_dict=True,
        )
        
        # Get first token
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated_tokens.append(next_token)
        
        ttft = time.perf_counter() - first_start
        
        # Get and compress KV cache
        past_key_values = outputs.past_key_values
        if keep_ratio < 1.0 and past_key_values is not None:
            if hasattr(past_key_values, 'to_legacy_cache'):
                kv_list = past_key_values.to_legacy_cache()
            else:
                kv_list = list(past_key_values)
            
            compressed_kv = l2_compress(
                kv_list,
                keep_ratio=keep_ratio,
                prune_after=prune_after,
                skip_layers=skip_layers,
            )
            past_key_values = to_dynamic_cache(compressed_kv)
        
        # Generate remaining tokens
        for _ in range(max_new_tokens - 1):
            outputs = model(
                next_token,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            
            # Get next token
            next_token_logits = outputs.logits[:, -1, :]
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_tokens.append(next_token)
            
            # Check for EOS
            if next_token.item() == tokenizer.eos_token_id:
                break
            
            # Get and compress KV cache
            past_key_values = outputs.past_key_values
            if keep_ratio < 1.0 and past_key_values is not None:
                if hasattr(past_key_values, 'to_legacy_cache'):
                    kv_list = past_key_values.to_legacy_cache()
                else:
                    kv_list = list(past_key_values)
                
                compressed_kv = l2_compress(
                    kv_list,
                    keep_ratio=keep_ratio,
                    prune_after=prune_after,
                    skip_layers=skip_layers,
                )
                past_key_values = to_dynamic_cache(compressed_kv)
    
    total_time = time.perf_counter() - total_start
    num_generated = len(generated_tokens)
    
    # Calculate metrics
    tpot = (total_time - ttft) / max(num_generated - 1, 1)
    throughput = num_generated / total_time if total_time > 0 else 0
    
    return {
        "ttft": ttft,
        "tpot": tpot,
        "throughput": throughput,
        "total_time": total_time,
        "num_tokens": num_generated,
        "input_length": input_length,
    }


def benchmark(
    model,
    tokenizer,
    text: str,
    max_new_tokens: int = 1000,
    keep_ratio: float = 1.0,
    prune_after: int = 1024,  # Start compressing early
    skip_layers: List[int] = [0, 1],
    eval_tokens: int = 3000,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Run full benchmark including generation metrics and quality metrics.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text
        max_new_tokens: Number of tokens to generate for speed benchmark
        keep_ratio: Fraction of tokens to keep (0.0 to 1.0)
        prune_after: Only compress if cache length > this value
        skip_layers: Layer indices to skip compression
        eval_tokens: Number of tokens for PPL/Accuracy evaluation
        device: Device to use
    
    Returns:
        Dict containing all metrics:
        - ttft, tpot, throughput, total_time, num_tokens, input_length
        - perplexity, accuracy, eval_tokens, final_cache_size
    """
    # Measure generation metrics
    gen_metrics = measure_generation_metrics(
        model=model,
        tokenizer=tokenizer,
        text=text,
        max_new_tokens=max_new_tokens,
        keep_ratio=keep_ratio,
        prune_after=prune_after,
        skip_layers=skip_layers,
        device=device,
    )
    
    # Measure quality metrics (PPL and Accuracy)
    qual_metrics = evaluate_with_compression(
        model=model,
        tokenizer=tokenizer,
        text=text,
        keep_ratio=keep_ratio,
        prune_after=prune_after,
        skip_layers=skip_layers,
        max_tokens=eval_tokens,
        device=device,
        show_progress=True,
    )
    
    # Combine metrics
    result = {
        "keep_ratio": keep_ratio,
        "compression_pct": int((1 - keep_ratio) * 100),
        # Generation metrics
        "ttft": gen_metrics["ttft"],
        "tpot": gen_metrics["tpot"],
        "throughput": gen_metrics["throughput"],
        "total_time": gen_metrics["total_time"],
        "num_generated": gen_metrics["num_tokens"],
        "input_length": gen_metrics["input_length"],
        # Quality metrics
        "perplexity": qual_metrics["perplexity"],
        "accuracy": qual_metrics["accuracy"],
        "eval_tokens": qual_metrics["num_tokens"],
        "final_cache_size": qual_metrics["final_cache_size"],
    }
    
    return result


def run_benchmark_suite(
    model,
    tokenizer,
    text: str,
    keep_ratios: List[float] = [1.0, 0.9, 0.8, 0.7, 0.5, 0.3, 0.1],
    max_new_tokens: int = 1000,
    prune_after: int = 1024,  # Start compressing early
    skip_layers: List[int] = [0, 1],
    eval_tokens: int = 3000,
    device: Optional[torch.device] = None,
) -> List[Dict[str, float]]:
    """
    Run complete benchmark suite across multiple compression levels.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        text: Input text
        keep_ratios: List of keep_ratio values to test
        max_new_tokens: Number of tokens to generate
        prune_after: Only compress if cache length > this value
        skip_layers: Layer indices to skip compression
        eval_tokens: Number of tokens for PPL evaluation
        device: Device to use
    
    Returns:
        List of result dicts, one per keep_ratio
    """
    results = []
    
    for keep_ratio in keep_ratios:
        compression_pct = int((1 - keep_ratio) * 100)
        print(f"\n{'='*60}")
        print(f"Testing keep_ratio={keep_ratio:.1f} ({compression_pct}% compression)")
        print('='*60)
        
        result = benchmark(
            model=model,
            tokenizer=tokenizer,
            text=text,
            max_new_tokens=max_new_tokens,
            keep_ratio=keep_ratio,
            prune_after=prune_after,
            skip_layers=skip_layers,
            eval_tokens=eval_tokens,
            device=device,
        )
        
        results.append(result)
        
        # Print results
        print(f"\nGeneration Metrics:")
        print(f"  TTFT:       {result['ttft']:.4f} seconds")
        print(f"  TPOT:       {result['tpot']:.4f} seconds")
        print(f"  Throughput: {result['throughput']:.2f} tokens/sec")
        
        print(f"\nQuality Metrics:")
        print(f"  PPL:        {result['perplexity']:.2f}")
        print(f"  Accuracy:   {result['accuracy']:.2%}")
        print(f"  Cache size: {result['final_cache_size']} tokens")
    
    return results


def print_benchmark_summary(results: List[Dict[str, float]]) -> None:
    """
    Print a summary table of benchmark results.
    
    Args:
        results: List of result dicts from run_benchmark_suite
    """
    print("\n" + "="*80)
    print("BENCHMARK SUMMARY")
    print("="*80)
    
    # Header
    print(f"{'Keep':>6} {'Comp%':>6} {'TTFT(s)':>10} {'TPOT(s)':>10} "
          f"{'Thruput':>10} {'PPL':>10} {'Acc':>10} {'Cache':>8}")
    print("-"*80)
    
    # Baseline values for comparison
    baseline = results[0] if results else None
    
    for r in results:
        # Calculate improvements
        ttft_imp = ""
        if baseline and baseline['ttft'] > 0 and r['keep_ratio'] < 1.0:
            imp = (1 - r['ttft'] / baseline['ttft']) * 100
            ttft_imp = f"({imp:+.0f}%)"
        
        ppl_change = ""
        if baseline and baseline['perplexity'] > 0 and r['keep_ratio'] < 1.0:
            change = (r['perplexity'] / baseline['perplexity'] - 1) * 100
            ppl_change = f"({change:+.0f}%)"
        
        print(f"{r['keep_ratio']:>6.1f} {r['compression_pct']:>6}% "
              f"{r['ttft']:>10.4f} {r['tpot']:>10.4f} "
              f"{r['throughput']:>10.2f} "
              f"{r['perplexity']:>10.2f} {r['accuracy']:>10.2%} "
              f"{r['final_cache_size']:>8}")
    
    print("="*80)
    
    # Print comparison with baseline
    if baseline and len(results) > 1:
        print("\nComparison with baseline (keep_ratio=1.0):")
        for r in results[1:]:
            ttft_imp = (1 - r['ttft'] / baseline['ttft']) * 100
            ppl_change = (r['perplexity'] / baseline['perplexity'] - 1) * 100
            acc_change = (r['accuracy'] / baseline['accuracy'] - 1) * 100
            
            print(f"  keep_ratio={r['keep_ratio']:.1f}: "
                  f"TTFT {ttft_imp:+.1f}%, "
                  f"PPL {ppl_change:+.1f}%, "
                  f"Acc {acc_change:+.1f}%")

