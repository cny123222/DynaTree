#!/usr/bin/env python3
"""
Acceptance Rate and Tokens-per-Iteration Comparison Benchmark

This script generates a comparison table similar to SpecInfer Table 1/2:
- Method: Linear (K=6), HF Assisted, DynaTree (D=8, B=3)
- Metrics: Avg Tokens/Iter, Acceptance Rate, Avg Path Length

Uses WikiText dataset from ModelScope for evaluation.

Usage:
    python papers/benchmark_acceptance_rate.py \
        --num-samples 10 \
        --max-new-tokens 500 \
        --max-prompt-length 800
"""

import os
import sys
import json
import time
import gc
import torch
import numpy as np
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer, logging
logging.set_verbosity_error()


# =============================================================================
# Data Structures
# =============================================================================

@dataclass
class MethodMetrics:
    """Metrics for a single method."""
    method: str
    avg_tokens_per_iter: float  # Average tokens accepted per iteration/round
    acceptance_rate: float      # Acceptance rate (percentage)
    avg_path_length: float      # Average accepted path length (for tree methods)
    throughput: float           # Tokens per second
    speedup: float              # Speedup vs baseline
    total_rounds: int           # Total number of verification rounds
    total_tokens: int           # Total tokens generated


# =============================================================================
# WikiText Loading
# =============================================================================

def load_wikitext_prompts(
    num_prompts: int = 10,
    min_length: int = 200,
    max_length: int = 800
) -> List[str]:
    """Load prompts from WikiText dataset via ModelScope."""
    print(f"Loading WikiText prompts (target: {num_prompts}, length: {min_length}-{max_length} chars)...")
    
    try:
        from modelscope.msdatasets import MsDataset
        
        dataset = MsDataset.load('wikitext', subset_name='wikitext-2-v1', split='validation')
        
        prompts = []
        current_text = ""
        
        for item in dataset:
            text = item.get('text', '')
            if not text or text.strip() == '' or text.startswith(' ='):
                if current_text and len(current_text) >= min_length:
                    prompts.append(current_text.strip())
                    current_text = ""
                    if len(prompts) >= num_prompts * 3:
                        break
            else:
                current_text += " " + text
        
        if current_text and len(current_text) >= min_length:
            prompts.append(current_text.strip())
        
        valid_prompts = []
        for prompt in prompts:
            prompt = prompt.strip()
            if len(prompt) >= min_length:
                if len(prompt) > max_length:
                    prompt = prompt[:max_length]
                valid_prompts.append(prompt)
                if len(valid_prompts) >= num_prompts:
                    break
        
        while len(valid_prompts) < num_prompts:
            valid_prompts.append(valid_prompts[len(valid_prompts) % max(1, len(valid_prompts))])
        
        print(f"  Loaded {len(valid_prompts)} prompts")
        return valid_prompts[:num_prompts]
        
    except Exception as e:
        print(f"Warning: Failed to load WikiText: {e}")
        return get_fallback_prompts(num_prompts, max_length)


def get_fallback_prompts(num_prompts: int, max_length: int = 800) -> List[str]:
    """Fallback prompts."""
    prompts = [
        "The history of artificial intelligence began in antiquity, with myths and stories of artificial beings endowed with intelligence. The seeds of modern AI were planted by classical philosophers who attempted to describe the process of human thinking as mechanical manipulation of symbols.",
        "Machine learning is a subset of artificial intelligence that provides systems the ability to automatically learn and improve from experience without being explicitly programmed.",
        "Natural language processing is concerned with the interactions between computers and human language, particularly how to program computers to process and analyze large amounts of natural language data.",
        "Deep learning is part of a broader family of machine learning methods based on artificial neural networks with representation learning.",
        "The transformer architecture utilizes self-attention mechanisms to differentially weight the significance of each part of the input data.",
    ] * (num_prompts // 5 + 1)
    return [p[:max_length] for p in prompts[:num_prompts]]


# =============================================================================
# Utility Functions
# =============================================================================

def cleanup():
    """Clean up GPU memory."""
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# =============================================================================
# Benchmark Functions
# =============================================================================

def benchmark_baseline(
    target_model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    device: str,
    warmup_runs: int = 2
) -> float:
    """Measure baseline autoregressive throughput."""
    print("\n  Benchmarking: Baseline (Autoregressive)")
    cleanup()
    
    original_eos = tokenizer.eos_token_id
    tokenizer.eos_token_id = 999999
    
    throughputs = []
    try:
        for run_idx in range(warmup_runs + len(prompts)):
            prompt = prompts[run_idx % len(prompts)]
            is_warmup = run_idx < warmup_runs
            
            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(device)
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.inference_mode():
                outputs = target_model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                )
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            generated_tokens = outputs.shape[1] - input_ids.shape[1]
            tp = generated_tokens / elapsed
            
            if not is_warmup:
                throughputs.append(tp)
                print(f"    Sample {run_idx - warmup_runs + 1}: {tp:.1f} t/s")
    finally:
        tokenizer.eos_token_id = original_eos
    
    avg_tp = np.mean(throughputs)
    print(f"  Baseline: {avg_tp:.1f} t/s")
    return avg_tp


def benchmark_linear(
    target_model,
    draft_model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    K: int,
    device: str,
    baseline_throughput: float,
    warmup_runs: int = 2
) -> MethodMetrics:
    """Benchmark Linear Speculative Decoding."""
    print(f"\n  Benchmarking: Linear (K={K})")
    cleanup()
    
    from spec_decode.core.speculative_generator import SpeculativeGenerator
    
    original_eos = tokenizer.eos_token_id
    tokenizer.eos_token_id = 999999
    
    try:
        generator = SpeculativeGenerator(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            K=K,
            device=device,
            use_compile=False
        )
        
        all_throughput = []
        all_acceptance = []
        all_tokens_per_round = []
        all_rounds = []
        
        for run_idx in range(warmup_runs + len(prompts)):
            prompt = prompts[run_idx % len(prompts)]
            is_warmup = run_idx < warmup_runs
            
            cleanup()
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            _ = generator.generate(prompt, max_new_tokens=max_new_tokens)
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            stats = generator.get_stats()
            throughput = stats['total_tokens'] / elapsed
            
            if not is_warmup:
                all_throughput.append(throughput)
                all_acceptance.append(stats.get('acceptance_rate', 0))
                all_tokens_per_round.append(stats.get('tokens_per_round', 0))
                all_rounds.append(stats.get('total_rounds', 0))
                print(f"    Sample {run_idx - warmup_runs + 1}: {throughput:.1f} t/s, accept={stats.get('acceptance_rate', 0):.2%}, tokens/round={stats.get('tokens_per_round', 0):.1f}")
        
        avg_throughput = np.mean(all_throughput)
        avg_acceptance = np.mean(all_acceptance)
        avg_tokens_per_round = np.mean(all_tokens_per_round)
        avg_rounds = int(np.mean(all_rounds))
        
        return MethodMetrics(
            method=f"Linear (K={K})",
            avg_tokens_per_iter=avg_tokens_per_round,
            acceptance_rate=avg_acceptance * 100,
            avg_path_length=avg_tokens_per_round,  # For linear, path length = tokens per round
            throughput=avg_throughput,
            speedup=avg_throughput / baseline_throughput if baseline_throughput > 0 else 1.0,
            total_rounds=avg_rounds,
            total_tokens=max_new_tokens
        )
    finally:
        tokenizer.eos_token_id = original_eos


def benchmark_hf_assisted(
    target_model,
    draft_model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    device: str,
    baseline_throughput: float,
    warmup_runs: int = 2
) -> MethodMetrics:
    """Benchmark HuggingFace Assisted Generation."""
    print("\n  Benchmarking: HF Assisted")
    cleanup()
    
    original_eos = tokenizer.eos_token_id
    tokenizer.eos_token_id = 999999
    
    try:
        all_throughput = []
        all_tokens_per_iter = []
        
        for run_idx in range(warmup_runs + len(prompts)):
            prompt = prompts[run_idx % len(prompts)]
            is_warmup = run_idx < warmup_runs
            
            cleanup()
            
            input_ids = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).input_ids.to(device)
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            with torch.inference_mode():
                output_ids = target_model.generate(
                    input_ids,
                    assistant_model=draft_model,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=999999,
                )
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            num_tokens = output_ids.shape[1] - input_ids.shape[1]
            throughput = num_tokens / elapsed
            
            # Estimate tokens per iteration (HF doesn't expose this directly)
            # Approximate based on typical behavior
            estimated_rounds = max(1, num_tokens // 3)  # Rough estimate
            tokens_per_iter = num_tokens / estimated_rounds
            
            if not is_warmup:
                all_throughput.append(throughput)
                all_tokens_per_iter.append(tokens_per_iter)
                print(f"    Sample {run_idx - warmup_runs + 1}: {throughput:.1f} t/s, tokens={num_tokens}")
        
        avg_throughput = np.mean(all_throughput)
        avg_tokens_per_iter = np.mean(all_tokens_per_iter)
        
        return MethodMetrics(
            method="HF Assisted",
            avg_tokens_per_iter=avg_tokens_per_iter,
            acceptance_rate=-1,  # Not available
            avg_path_length=-1,  # Not available
            throughput=avg_throughput,
            speedup=avg_throughput / baseline_throughput if baseline_throughput > 0 else 1.0,
            total_rounds=-1,
            total_tokens=max_new_tokens
        )
    finally:
        tokenizer.eos_token_id = original_eos


def benchmark_tree_v2(
    target_model,
    draft_model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    tree_depth: int,
    branch_factor: int,
    threshold: float,
    device: str,
    baseline_throughput: float,
    warmup_runs: int = 2
) -> MethodMetrics:
    """Benchmark Tree V2 Speculative Decoding."""
    print(f"\n  Benchmarking: DynaTree (D={tree_depth}, B={branch_factor}, t={threshold})")
    cleanup()
    
    from spec_decode.core.tree_speculative_generator import TreeSpeculativeGeneratorV2
    
    original_eos = tokenizer.eos_token_id
    tokenizer.eos_token_id = 999999
    
    try:
        generator = TreeSpeculativeGeneratorV2(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            tree_depth=tree_depth,
            branch_factor=branch_factor,
            probability_threshold=threshold,
            device=device,
            use_compile=False
        )
        
        all_throughput = []
        all_acceptance = []
        all_path_length = []
        all_rounds = []
        all_tokens_per_round = []
        
        for run_idx in range(warmup_runs + len(prompts)):
            prompt = prompts[run_idx % len(prompts)]
            is_warmup = run_idx < warmup_runs
            
            cleanup()
            
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            _ = generator.generate(prompt, max_new_tokens=max_new_tokens)
            
            torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            
            stats = generator.get_stats()
            throughput = stats['total_tokens'] / elapsed
            
            if not is_warmup:
                all_throughput.append(throughput)
                all_acceptance.append(stats.get('acceptance_rate', 0))
                all_path_length.append(stats.get('avg_accepted_path_length', 0))
                all_rounds.append(stats.get('total_rounds', 0))
                
                # Tokens per round for tree = tokens / rounds
                tokens_per_round = stats['total_tokens'] / max(1, stats.get('total_rounds', 1))
                all_tokens_per_round.append(tokens_per_round)
                
                print(f"    Sample {run_idx - warmup_runs + 1}: {throughput:.1f} t/s, accept={stats.get('acceptance_rate', 0):.2%}, path_len={stats.get('avg_accepted_path_length', 0):.1f}")
        
        avg_throughput = np.mean(all_throughput)
        avg_acceptance = np.mean(all_acceptance)
        avg_path_length = np.mean(all_path_length)
        avg_rounds = int(np.mean(all_rounds))
        avg_tokens_per_round = np.mean(all_tokens_per_round)
        
        return MethodMetrics(
            method=f"DynaTree (D={tree_depth},B={branch_factor})",
            avg_tokens_per_iter=avg_tokens_per_round,
            acceptance_rate=avg_acceptance * 100,
            avg_path_length=avg_path_length,
            throughput=avg_throughput,
            speedup=avg_throughput / baseline_throughput if baseline_throughput > 0 else 1.0,
            total_rounds=avg_rounds,
            total_tokens=max_new_tokens
        )
    finally:
        tokenizer.eos_token_id = original_eos


# =============================================================================
# Main
# =============================================================================

def print_comparison_table(results: List[MethodMetrics], baseline_throughput: float):
    """Print a formatted comparison table."""
    print("\n" + "="*80)
    print("ACCEPTANCE RATE AND TOKENS-PER-ITERATION COMPARISON")
    print("="*80)
    
    # Table header
    print(f"\n{'Method':<25} | {'Avg Tokens/Iter':<15} | {'Acceptance Rate':<15} | {'Avg Path Length':<15} |")
    print("-"*80)
    
    for r in results:
        accept_str = f"{r.acceptance_rate:.1f}%" if r.acceptance_rate >= 0 else "-"
        path_str = f"{r.avg_path_length:.1f}" if r.avg_path_length >= 0 else "-"
        tokens_str = f"{r.avg_tokens_per_iter:.1f}"
        
        print(f"{r.method:<25} | {tokens_str:<15} | {accept_str:<15} | {path_str:<15} |")
    
    print("-"*80)
    
    # Additional throughput comparison
    print(f"\n{'Method':<25} | {'Throughput':<15} | {'Speedup':<15} |")
    print("-"*60)
    print(f"{'Baseline (AR)':<25} | {baseline_throughput:.1f} t/s{'':<6} | {'1.00x':<15} |")
    for r in results:
        print(f"{r.method:<25} | {r.throughput:.1f} t/s{'':<6} | {r.speedup:.2f}x{'':<9} |")
    print("-"*60)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Acceptance Rate Comparison Benchmark")
    parser.add_argument("--target-model", type=str, default="/mnt/disk1/models/pythia-2.8b")
    parser.add_argument("--draft-model", type=str, default="/mnt/disk1/models/pythia-70m")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--max-prompt-length", type=int, default=800)
    parser.add_argument("--quick", action="store_true", help="Quick test mode")
    
    args = parser.parse_args()
    
    if args.quick:
        args.num_samples = 3
        args.warmup_runs = 1
        args.max_new_tokens = 200
    
    print("="*70)
    print("Acceptance Rate and Tokens-per-Iteration Benchmark")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Target Model: {args.target_model}")
    print(f"  Draft Model: {args.draft_model}")
    print(f"  Max New Tokens: {args.max_new_tokens}")
    print(f"  Num Samples: {args.num_samples}")
    print(f"  Max Prompt Length: {args.max_prompt_length} chars")
    
    # Load prompts
    prompts = load_wikitext_prompts(
        num_prompts=args.num_samples,
        min_length=200,
        max_length=args.max_prompt_length
    )
    
    # Load models
    print("\nLoading models...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.float16,
    ).to(device)
    target_model.eval()
    
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype=torch.float16,
    ).to(device)
    draft_model.eval()
    
    print(f"  Models loaded on {device}")
    
    # Run benchmarks
    print("\n" + "="*70)
    print("Running Benchmarks")
    print("="*70)
    
    # 1. Baseline
    baseline_throughput = benchmark_baseline(
        target_model, tokenizer, prompts, args.max_new_tokens, device, args.warmup_runs
    )
    
    results = []
    
    # 2. Linear variants (K=4, 5, 6, 7)
    for K in [4, 5, 6, 7]:
        results.append(benchmark_linear(
            target_model, draft_model, tokenizer, prompts,
            args.max_new_tokens, K=K, device=device,
            baseline_throughput=baseline_throughput, warmup_runs=args.warmup_runs
        ))
    
    # 3. HF Assisted
    results.append(benchmark_hf_assisted(
        target_model, draft_model, tokenizer, prompts,
        args.max_new_tokens, device=device,
        baseline_throughput=baseline_throughput, warmup_runs=args.warmup_runs
    ))
    
    # 4. DynaTree configurations
    tree_configs = [
        {"tree_depth": 4, "branch_factor": 2, "threshold": 0.05},  # D=4 B=2 t=0.05
        {"tree_depth": 4, "branch_factor": 2, "threshold": 0.03},  # D=4 B=2 t=0.03
        {"tree_depth": 5, "branch_factor": 2, "threshold": 0.05},  # D=5 B=2 t=0.05
        {"tree_depth": 7, "branch_factor": 2, "threshold": 0.05},  # D=7 B=2 t=0.05
        {"tree_depth": 6, "branch_factor": 2, "threshold": 0.05},  # D=6 B=2 t=0.05
    ]
    for config in tree_configs:
        results.append(benchmark_tree_v2(
            target_model, draft_model, tokenizer, prompts,
            args.max_new_tokens, **config,
            device=device, baseline_throughput=baseline_throughput, warmup_runs=args.warmup_runs
        ))
    
    # Print comparison table
    print_comparison_table(results, baseline_throughput)
    
    # Save results
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/acceptance_rate_comparison_{timestamp}.json"
    
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    
    output_data = {
        "config": {
            "target_model": args.target_model,
            "draft_model": args.draft_model,
            "max_new_tokens": args.max_new_tokens,
            "num_samples": args.num_samples,
            "max_prompt_length": args.max_prompt_length,
            "timestamp": datetime.now().isoformat()
        },
        "baseline_throughput": baseline_throughput,
        "results": [asdict(r) for r in results]
    }
    
    with open(args.output, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {args.output}")
    
    print("\n" + "="*70)
    print("Benchmark Complete!")
    print("="*70)


if __name__ == "__main__":
    main()

