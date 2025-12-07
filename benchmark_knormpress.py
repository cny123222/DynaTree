#!/usr/bin/env python3
"""
KnormPress Benchmark Script

This script benchmarks the KnormPress KV cache compression algorithm
on the PG-19 long text dataset.

Usage:
    python benchmark_knormpress.py --keep_ratios 1.0,0.9,0.8,0.7,0.5,0.3
    python benchmark_knormpress.py --num_samples 5 --max_tokens 3000

Expected Results (matching paper):
    - keep_ratio >= 0.7: PPL and Accuracy ~= baseline
    - keep_ratio < 0.7: PPL increases, Accuracy decreases gradually
"""

import os
import sys
import argparse
import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knormpress import (
    evaluate_with_compression,
    benchmark,
    run_benchmark_suite,
    print_benchmark_summary,
)


# Local PG-19 dataset path
LOCAL_PG19_PATH = "/Users/od/Desktop/NLP/CS2602-LLM-Inference-Acceleration/data/pg19.parquet"


def get_device():
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model_and_tokenizer(model_id: str = "EleutherAI/pythia-70m-deduped"):
    """Load model and tokenizer."""
    print(f"Loading model: {model_id}")
    
    device = get_device()
    print(f"Using device: {device}")
    
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float32,
    ).to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer, device


def load_pg19_samples(num_samples: int = 3):
    """Load samples from PG-19 dataset."""
    print("\nLoading PG-19 dataset...")
    
    # Try local file first
    if os.path.exists(LOCAL_PG19_PATH):
        print(f"  Found local file: {LOCAL_PG19_PATH}")
        try:
            dataset = load_dataset(
                "parquet",
                data_files={'test': LOCAL_PG19_PATH},
                split="test"
            )
            print(f"  Loaded {len(dataset)} samples from local file")
        except Exception as e:
            print(f"  Failed to load local file: {e}")
            dataset = None
    else:
        dataset = None
    
    # Fallback to HuggingFace
    if dataset is None:
        try:
            print("  Loading from HuggingFace...")
            dataset = load_dataset("pg19", split="test")
        except Exception as e:
            print(f"  Failed to load from HuggingFace: {e}")
            return []
    
    # Extract text samples
    samples = []
    for i, sample in enumerate(dataset):
        if i >= num_samples:
            break
        text = sample.get("text", "")
        if len(text) > 10000:  # Ensure long enough for meaningful evaluation
            samples.append(text)
            print(f"  Sample {i+1}: {len(text)} characters")
    
    print(f"  Total: {len(samples)} samples loaded")
    return samples


def run_single_sample_benchmark(
    model,
    tokenizer,
    text: str,
    keep_ratios: list,
    prune_after: int,
    skip_layers: list,
    max_tokens: int,
    max_new_tokens: int,
    device,
):
    """Run benchmark on a single text sample."""
    results = []
    
    for keep_ratio in keep_ratios:
        compression_pct = int((1 - keep_ratio) * 100)
        print(f"\n  Testing keep_ratio={keep_ratio:.2f} ({compression_pct}% compression)...")
        
        result = benchmark(
            model=model,
            tokenizer=tokenizer,
            text=text,
            max_new_tokens=max_new_tokens,
            keep_ratio=keep_ratio,
            prune_after=prune_after,
            skip_layers=skip_layers,
            eval_tokens=max_tokens,
            device=device,
        )
        
        results.append(result)
        
        print(f"    TTFT: {result['ttft']:.4f}s | "
              f"TPOT: {result['tpot']:.4f}s | "
              f"PPL: {result['perplexity']:.2f} | "
              f"Acc: {result['accuracy']:.2%}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark KnormPress on PG-19 dataset"
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="EleutherAI/pythia-70m-deduped",
        help="Model ID from HuggingFace"
    )
    parser.add_argument(
        "--keep_ratios",
        type=str,
        default="1.0,0.9,0.8,0.7,0.5,0.3",
        help="Comma-separated list of keep_ratio values to test"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of PG-19 samples to test"
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=2048,
        help="Maximum tokens for PPL evaluation"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=2048,
        help="Number of tokens to generate for TTFT/TPOT"
    )
    parser.add_argument(
        "--prune_after",
        type=int,
        default=1024,
        help="Only compress if cache length > this value (use small value for meaningful evaluation)"
    )
    parser.add_argument(
        "--skip_layers",
        type=str,
        default="0,1",
        help="Comma-separated list of layer indices to skip"
    )
    
    args = parser.parse_args()
    
    # Parse arguments
    keep_ratios = [float(x) for x in args.keep_ratios.split(",")]
    skip_layers = [int(x) for x in args.skip_layers.split(",")]
    
    print("="*70)
    print("KnormPress Benchmark")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_id}")
    print(f"  Keep ratios: {keep_ratios}")
    print(f"  Prune after: {args.prune_after} tokens")
    print(f"  Skip layers: {skip_layers}")
    print(f"  Max eval tokens: {args.max_tokens}")
    print(f"  Max new tokens: {args.max_new_tokens}")
    
    # Load model
    model, tokenizer, device = load_model_and_tokenizer(args.model_id)
    
    # Load PG-19 samples
    samples = load_pg19_samples(args.num_samples)
    if not samples:
        print("No samples loaded. Exiting.")
        return
    
    # Run benchmark on each sample
    all_results = {kr: [] for kr in keep_ratios}
    
    for i, text in enumerate(samples):
        print(f"\n{'='*70}")
        print(f"Sample {i+1}/{len(samples)} ({len(text)} characters)")
        print("="*70)
        
        results = run_single_sample_benchmark(
            model=model,
            tokenizer=tokenizer,
            text=text,
            keep_ratios=keep_ratios,
            prune_after=args.prune_after,
            skip_layers=skip_layers,
            max_tokens=args.max_tokens,
            max_new_tokens=args.max_new_tokens,
            device=device,
        )
        
        for r in results:
            all_results[r['keep_ratio']].append(r)
    
    # Print aggregated results
    print("\n" + "="*70)
    print("AGGREGATED RESULTS (averaged across samples)")
    print("="*70)
    
    print(f"\n{'Keep':>6} {'Comp%':>6} {'TTFT(s)':>10} {'TPOT(s)':>10} "
          f"{'Thruput':>10} {'PPL':>10} {'Acc':>10}")
    print("-"*70)
    
    baseline_ppl = None
    baseline_acc = None
    baseline_ttft = None
    
    for keep_ratio in keep_ratios:
        results = all_results[keep_ratio]
        if not results:
            continue
        
        avg_ttft = np.mean([r['ttft'] for r in results])
        avg_tpot = np.mean([r['tpot'] for r in results])
        avg_throughput = np.mean([r['throughput'] for r in results])
        avg_ppl = np.mean([r['perplexity'] for r in results])
        avg_acc = np.mean([r['accuracy'] for r in results])
        
        if keep_ratio == 1.0:
            baseline_ppl = avg_ppl
            baseline_acc = avg_acc
            baseline_ttft = avg_ttft
        
        compression_pct = int((1 - keep_ratio) * 100)
        
        print(f"{keep_ratio:>6.2f} {compression_pct:>6}% "
              f"{avg_ttft:>10.4f} {avg_tpot:>10.4f} "
              f"{avg_throughput:>10.2f} "
              f"{avg_ppl:>10.2f} {avg_acc:>10.2%}")
    
    print("="*70)
    
    # Print comparison
    if baseline_ppl is not None:
        print("\nComparison with baseline (keep_ratio=1.0):")
        for keep_ratio in keep_ratios:
            if keep_ratio == 1.0:
                continue
            
            results = all_results[keep_ratio]
            if not results:
                continue
            
            avg_ttft = np.mean([r['ttft'] for r in results])
            avg_ppl = np.mean([r['perplexity'] for r in results])
            avg_acc = np.mean([r['accuracy'] for r in results])
            
            ttft_imp = (1 - avg_ttft / baseline_ttft) * 100 if baseline_ttft > 0 else 0
            ppl_change = (avg_ppl / baseline_ppl - 1) * 100 if baseline_ppl > 0 else 0
            acc_change = (avg_acc / baseline_acc - 1) * 100 if baseline_acc > 0 else 0
            
            print(f"  keep_ratio={keep_ratio:.2f}: "
                  f"TTFT {ttft_imp:+.1f}%, "
                  f"PPL {ppl_change:+.1f}%, "
                  f"Acc {acc_change:+.1f}%")
    
    print("\nBenchmark completed!")


if __name__ == "__main__":
    main()

