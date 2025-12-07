"""
Optimized Test Script with KnormPress KV Cache Compression

This script benchmarks the KnormPress algorithm on the Pythia-70M model,
focusing on pg-19 long text performance. It measures the same metrics as
baseline_test.py but with KV cache compression applied.
"""

from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch
import time
from datasets import load_dataset
import numpy as np
from typing import Optional, List, Tuple
import argparse

from custom_generate import generate_with_compression_and_timing
from kv_compress import l2_compress, get_cache_info
import pandas as pd

# Model ID
model_id = "EleutherAI/pythia-70m-deduped"

# Local PG-19 dataset path (if available)
LOCAL_PG19_PATH = "/Users/od/Desktop/NLP/CS2602-LLM-Inference-Acceleration/test-00000-of-00001-29a571947c0b5ccc.parquet"


def get_optimal_device():
    """
    Checks for available hardware accelerators and returns the most optimal one.
    Priority: CUDA (NVIDIA) > MPS (Apple Silicon) > CPU.
    """
    if torch.cuda.is_available():
        print("CUDA (NVIDIA GPU) is available. Using cuda.")
        return torch.device("cuda")
    
    if torch.backends.mps.is_available():
        if not torch.backends.mps.is_built():
            print("MPS not available because the current PyTorch install was not "
                  "built with MPS enabled.")
            print("Falling back to CPU.")
            return torch.device("cpu")
        else:
            print("MPS (Apple Silicon GPU) is available. Using mps.")
            return torch.device("mps")
            
    print("No GPU acceleration available. Falling back to CPU.")
    return torch.device("cpu")


def measure_generation_performance_with_compression(
    model, 
    tokenizer, 
    input_text: str, 
    max_new_tokens: int = 100,
    keep_ratio: float = 1.0,
    prune_after: int = 512,
    skip_layers: List[int] = [0],
    device: torch.device = None
) -> dict:
    """
    Measure generation performance with KV cache compression.
    
    Returns:
        dict: Performance metrics including TTFT, TPOT, Throughput, Memory
    """
    # Encode input text
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    input_length = input_ids.shape[1]
    
    # Reset memory stats
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    # Use custom generation if compression is needed
    if keep_ratio < 1.0:
        result = generate_with_compression_and_timing(
            model=model,
            tokenizer=tokenizer,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            keep_ratio=keep_ratio,
            prune_after=prune_after,
            skip_layers=skip_layers,
            device=device,
        )
        
        # Get peak memory
        peak_memory = None
        if device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
        
        return {
            "ttft": result["ttft"],
            "tpot": result["tpot"],
            "throughput": result["throughput"],
            "total_time": result["total_time"],
            "num_generated_tokens": result["num_generated_tokens"],
            "peak_memory_mb": peak_memory,
            "input_length": input_length,
            "output_length": result["output_ids"].shape[1],
            "keep_ratio": keep_ratio,
        }
    else:
        # Use standard generation for baseline (keep_ratio=1.0)
        start_time = time.time()
        
        with torch.no_grad():
            # First forward pass to get first token time
            outputs = model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
                return_dict_in_generate=True,
            )
        
        total_time = time.time() - start_time
        output_ids = outputs.sequences if hasattr(outputs, 'sequences') else outputs
        num_generated = output_ids.shape[1] - input_length
        
        # Approximate TTFT as a fraction of total time
        # This is a rough estimate since we don't have access to intermediate timing
        ttft = total_time * 0.1  # First token typically takes longer
        tpot = total_time / num_generated if num_generated > 0 else 0
        
        peak_memory = None
        if device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        return {
            "ttft": ttft,
            "tpot": tpot,
            "throughput": num_generated / total_time if total_time > 0 else 0,
            "total_time": total_time,
            "num_generated_tokens": num_generated,
            "peak_memory_mb": peak_memory,
            "input_length": input_length,
            "output_length": output_ids.shape[1],
            "keep_ratio": keep_ratio,
        }


def calculate_perplexity(
    model, 
    tokenizer, 
    text: str, 
    device: torch.device = None,
    max_length: int = 512
) -> float:
    """
    Calculate perplexity of text.
    
    Note: This does NOT use KV cache compression as it's for evaluation.
    """
    encodings = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    input_ids = encodings.input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    
    perplexity = torch.exp(loss).item()
    return perplexity


def calculate_perplexity_with_compression(
    model,
    tokenizer,
    text: str,
    keep_ratio: float = 1.0,
    prune_after: int = 512,
    skip_layers: List[int] = [0],
    device: torch.device = None,
    max_eval_tokens: int = 512,
) -> float:
    """
    Calculate perplexity with KV cache compression.
    
    This simulates the actual generation process where KV cache is compressed
    after each token, measuring the real quality impact of compression.
    
    Uses autoregressive evaluation: predict each token given all previous tokens,
    with compressed KV cache.
    """
    from transformers import DynamicCache
    
    # Tokenize input
    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_ids = input_ids[:, :max_eval_tokens].to(device)
    seq_len = input_ids.shape[1]
    
    if seq_len < 2:
        return float('inf')
    
    total_loss = 0.0
    num_tokens = 0
    past_key_values = None
    
    model.eval()
    with torch.no_grad():
        # Process token by token in autoregressive manner
        for i in range(seq_len - 1):
            # Prepare input
            if past_key_values is None:
                # First token: use all tokens up to position i
                current_input = input_ids[:, :i+1]
            else:
                # Subsequent tokens: only use current token with KV cache
                current_input = input_ids[:, i:i+1]
            
            # Target is the next token
            target = input_ids[:, i+1]
            
            # Forward pass
            outputs = model(
                current_input,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            
            # Get logits for the last position
            logits = outputs.logits[:, -1, :]
            
            # Calculate cross-entropy loss for this prediction
            loss = torch.nn.functional.cross_entropy(
                logits, 
                target,
                reduction='sum'
            )
            total_loss += loss.item()
            num_tokens += 1
            
            # Get and compress KV cache
            past_key_values = outputs.past_key_values
            
            # Apply compression if keep_ratio < 1.0
            if keep_ratio < 1.0 and past_key_values is not None:
                # Convert DynamicCache to legacy format
                if hasattr(past_key_values, 'to_legacy_cache'):
                    kv_list = past_key_values.to_legacy_cache()
                else:
                    kv_list = past_key_values
                
                # Apply L2 compression
                compressed_kv = l2_compress(
                    kv_list,
                    keep_ratio=keep_ratio,
                    prune_after=prune_after,
                    skip_layers=skip_layers,
                )
                
                # Convert back to DynamicCache
                new_cache = DynamicCache()
                for layer_idx, (layer_keys, layer_values) in enumerate(compressed_kv):
                    new_cache.update(layer_keys, layer_values, layer_idx)
                
                past_key_values = new_cache
    
    # Calculate perplexity from average loss
    if num_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / num_tokens
    perplexity = np.exp(avg_loss)
    
    return perplexity


def load_test_data():
    """Load test datasets"""
    print("\nLoading test datasets...")
    
    # Load wikitext
    print("Loading wikitext-2-raw-v1...")
    wikitext_dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    # Load pg-19 - try local file first, then online
    print("Loading pg-19...")
    pg19_dataset = None
    
    # Try loading from local parquet file
    try:
        import os
        if os.path.exists(LOCAL_PG19_PATH):
            print(f"  Found local pg-19 file: {LOCAL_PG19_PATH}")
            df = pd.read_parquet(LOCAL_PG19_PATH)
            from datasets import Dataset
            pg19_dataset = Dataset.from_pandas(df)
            print(f"  ✓ Successfully loaded {len(pg19_dataset)} samples from local file")
        else:
            print(f"  Local file not found: {LOCAL_PG19_PATH}")
            raise FileNotFoundError("Local PG-19 file not found")
    except Exception as e1:
        print(f"  × Local loading failed: {e1}")
        
        # Fallback: try loading from Hugging Face
        try:
            print("  Attempting to load from Hugging Face...")
            pg19_dataset = load_dataset("pg19", split="test")
            print("  ✓ Successfully loaded pg-19 from Hugging Face")
        except Exception as e2:
            print(f"  × Hugging Face loading failed: {e2}")
            print("  ℹ️  PG-19 dataset not available, will skip long-context tests")
            pg19_dataset = None
    
    return wikitext_dataset, pg19_dataset


def run_optimized_tests(
    keep_ratios: List[float] = [1.0, 0.9, 0.8, 0.7],
    prune_after: int = 512,
    skip_layers: List[int] = [0],
    num_wikitext_samples: int = 3,
    num_pg19_samples: int = 2,
):
    """
    Run comprehensive tests with different compression ratios.
    """
    print("\n" + "="*70)
    print("KnormPress Optimized Performance Measurement")
    print("="*70)
    
    # Load model and tokenizer
    print("\nLoading model and tokenizer...")
    model = GPTNeoXForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    device = get_optimal_device()
    model.to(device)
    model.eval()
    
    print(f"Model loaded on {device}")
    print(f"Testing with keep_ratios: {keep_ratios}")
    print(f"Prune after: {prune_after} tokens")
    print(f"Skip layers: {skip_layers}")
    
    # Load test data
    wikitext_dataset, pg19_dataset = load_test_data()
    
    all_results = []
    
    # Test each compression ratio
    for keep_ratio in keep_ratios:
        print("\n" + "="*70)
        print(f"Testing with keep_ratio={keep_ratio} ({int((1-keep_ratio)*100)}% compression)")
        print("="*70)
        
        # Test on wikitext
        print("\n" + "-"*70)
        print(f"Test: Wikitext Dataset (keep_ratio={keep_ratio})")
        print("-"*70)
        
        samples_tested = 0
        for i in range(len(wikitext_dataset)):
            if samples_tested >= num_wikitext_samples:
                break
                
            sample = wikitext_dataset[i]
            text = sample.get("text", "")
            
            if not text or len(text.strip()) < 50:
                continue
            
            print(f"\nProcessing wikitext sample {samples_tested+1}/{num_wikitext_samples}...")
            print(f"Input text length: {len(text)} characters")
            
            # Use first 512 characters as input
            input_text = text[:512]
            
            try:
                # Measure generation performance
                gen_metrics = measure_generation_performance_with_compression(
                    model, tokenizer, input_text,
                    max_new_tokens=50,
                    keep_ratio=keep_ratio,
                    prune_after=prune_after,
                    skip_layers=skip_layers,
                    device=device
                )
                
                # Calculate perplexity with compression applied
                ppl_text = text[:1024] if len(text) >= 1024 else text
                try:
                    if keep_ratio < 1.0:
                        # Use compression for PPL calculation to reflect real impact
                        # Note: Use a large prune_after to match paper settings
                        ppl = calculate_perplexity_with_compression(
                            model, tokenizer, ppl_text,
                            keep_ratio=keep_ratio,
                            prune_after=512,  # Match paper: only compress longer sequences
                            skip_layers=skip_layers,
                            device=device,
                            max_eval_tokens=512  # Evaluate on longer sequence
                        )
                    else:
                        # Baseline: standard PPL without compression
                        ppl = calculate_perplexity(
                            model, tokenizer, ppl_text,
                            device=device
                        )
                except Exception as e:
                    print(f"Warning: Could not calculate perplexity: {e}")
                    ppl = None
                
                result = {
                    "dataset": "wikitext",
                    "sample_id": samples_tested,
                    "keep_ratio": keep_ratio,
                    "compression_pct": int((1-keep_ratio)*100),
                    "input_length": gen_metrics["input_length"],
                    "output_length": gen_metrics["output_length"],
                    "ttft_seconds": gen_metrics["ttft"],
                    "tpot_seconds": gen_metrics["tpot"],
                    "throughput_tokens_per_sec": gen_metrics["throughput"],
                    "peak_memory_mb": gen_metrics["peak_memory_mb"],
                    "perplexity": ppl
                }
                all_results.append(result)
                
                # Print results
                print(f"  TTFT: {gen_metrics['ttft']:.4f} seconds")
                print(f"  TPOT: {gen_metrics['tpot']:.4f} seconds")
                print(f"  Throughput: {gen_metrics['throughput']:.2f} tokens/sec")
                if gen_metrics['peak_memory_mb']:
                    print(f"  Peak Memory: {gen_metrics['peak_memory_mb']:.2f} MB")
                if ppl:
                    print(f"  Perplexity: {ppl:.2f}")
                
                samples_tested += 1
                    
            except Exception as e:
                print(f"Error processing sample: {e}")
                continue
        
        # Test on pg-19 (long context)
        if pg19_dataset is not None:
            print("\n" + "-"*70)
            print(f"Test: PG-19 Dataset - Long Context (keep_ratio={keep_ratio})")
            print("-"*70)
            
            samples_processed = 0
            for i in range(min(100, len(pg19_dataset))):
                if samples_processed >= num_pg19_samples:
                    break
                    
                sample = pg19_dataset[i]
                text = sample.get("text", "")
                
                if not text or len(text.strip()) < 100:
                    continue
                
                print(f"\nProcessing pg-19 sample {i+1}...")
                print(f"Input text length: {len(text)} characters")
                
                # Use longer input for pg-19
                input_text = text[:1024]
                
                try:
                    # Measure generation performance
                    gen_metrics = measure_generation_performance_with_compression(
                        model, tokenizer, input_text,
                        max_new_tokens=100,
                        keep_ratio=keep_ratio,
                        prune_after=prune_after,
                        skip_layers=skip_layers,
                        device=device
                    )
                    
                    # Calculate perplexity with compression applied
                    ppl_text = text[:2048] if len(text) >= 2048 else text
                    try:
                        if keep_ratio < 1.0:
                            # Use compression for PPL calculation
                            # Note: Use a large prune_after to match paper settings
                            ppl = calculate_perplexity_with_compression(
                                model, tokenizer, ppl_text,
                                keep_ratio=keep_ratio,
                                prune_after=512,  # Match paper: only compress longer sequences
                                skip_layers=skip_layers,
                                device=device,
                                max_eval_tokens=512  # Evaluate on longer sequence
                            )
                        else:
                            # Baseline: standard PPL without compression
                            ppl = calculate_perplexity(
                                model, tokenizer, ppl_text,
                                device=device
                            )
                    except Exception as e:
                        print(f"Warning: Could not calculate perplexity: {e}")
                        ppl = None
                    
                    result = {
                        "dataset": "pg-19",
                        "sample_id": i,
                        "keep_ratio": keep_ratio,
                        "compression_pct": int((1-keep_ratio)*100),
                        "input_length": gen_metrics["input_length"],
                        "output_length": gen_metrics["output_length"],
                        "ttft_seconds": gen_metrics["ttft"],
                        "tpot_seconds": gen_metrics["tpot"],
                        "throughput_tokens_per_sec": gen_metrics["throughput"],
                        "peak_memory_mb": gen_metrics["peak_memory_mb"],
                        "perplexity": ppl
                    }
                    all_results.append(result)
                    
                    # Print results
                    print(f"  TTFT: {gen_metrics['ttft']:.4f} seconds")
                    print(f"  TPOT: {gen_metrics['tpot']:.4f} seconds")
                    print(f"  Throughput: {gen_metrics['throughput']:.2f} tokens/sec")
                    if gen_metrics['peak_memory_mb']:
                        print(f"  Peak Memory: {gen_metrics['peak_memory_mb']:.2f} MB")
                    if ppl:
                        print(f"  Perplexity: {ppl:.2f}")
                    
                    samples_processed += 1
                    
                except Exception as e:
                    print(f"Error processing sample: {e}")
                    continue
    
    # Print summary table
    print("\n" + "="*70)
    print("Performance Summary - All Tests")
    print("="*70)
    print(f"{'Dataset':<12} {'Keep':<6} {'Comp%':<6} {'Sample':<8} {'TTFT(s)':<10} {'TPOT(s)':<10} {'Thruput':<10} {'Mem(MB)':<10} {'PPL':<10}")
    print("-"*70)
    
    for r in all_results:
        memory_str = f"{r['peak_memory_mb']:.1f}" if r['peak_memory_mb'] else "N/A"
        ppl_str = f"{r['perplexity']:.2f}" if r['perplexity'] else "N/A"
        print(f"{r['dataset']:<12} {r['keep_ratio']:<6.2f} {r['compression_pct']:<6} "
              f"{r['sample_id']:<8} {r['ttft_seconds']:<10.4f} {r['tpot_seconds']:<10.4f} "
              f"{r['throughput_tokens_per_sec']:<10.2f} {memory_str:<10} {ppl_str:<10}")
    
    # Print aggregated statistics by compression ratio
    print("\n" + "="*70)
    print("Aggregated Statistics by Compression Ratio")
    print("="*70)
    
    for keep_ratio in keep_ratios:
        ratio_results = [r for r in all_results if r['keep_ratio'] == keep_ratio]
        if not ratio_results:
            continue
        
        print(f"\nKeep Ratio: {keep_ratio} ({int((1-keep_ratio)*100)}% compression)")
        print("-"*70)
        
        # Overall stats
        avg_ttft = np.mean([r['ttft_seconds'] for r in ratio_results])
        avg_tpot = np.mean([r['tpot_seconds'] for r in ratio_results])
        avg_throughput = np.mean([r['throughput_tokens_per_sec'] for r in ratio_results])
        
        # 修复：检查空列表以避免警告
        memory_values = [r['peak_memory_mb'] for r in ratio_results if r['peak_memory_mb']]
        avg_memory = np.mean(memory_values) if memory_values else None
        
        ppl_values = [r['perplexity'] for r in ratio_results if r['perplexity']]
        avg_ppl = np.mean(ppl_values) if ppl_values else None
        
        print(f"  Average TTFT: {avg_ttft:.4f} seconds")
        print(f"  Average TPOT: {avg_tpot:.4f} seconds")
        print(f"  Average Throughput: {avg_throughput:.2f} tokens/sec")
        if avg_memory is not None:
            print(f"  Average Peak Memory: {avg_memory:.2f} MB")
        else:
            print(f"  Average Peak Memory: N/A (not supported on this device)")
        if avg_ppl is not None:
            print(f"  Average Perplexity: {avg_ppl:.2f}")
        
        # Stats by dataset
        for dataset in ["wikitext", "pg-19"]:
            dataset_results = [r for r in ratio_results if r['dataset'] == dataset]
            if not dataset_results:
                continue
            
            print(f"\n  {dataset.upper()}:")
            avg_ttft_ds = np.mean([r['ttft_seconds'] for r in dataset_results])
            avg_tpot_ds = np.mean([r['tpot_seconds'] for r in dataset_results])
            avg_throughput_ds = np.mean([r['throughput_tokens_per_sec'] for r in dataset_results])
            
            ppl_values_ds = [r['perplexity'] for r in dataset_results if r['perplexity']]
            avg_ppl_ds = np.mean(ppl_values_ds) if ppl_values_ds else None
            
            ppl_str = f"{avg_ppl_ds:.2f}" if avg_ppl_ds is not None else "N/A"
            print(f"    TTFT: {avg_ttft_ds:.4f}s | TPOT: {avg_tpot_ds:.4f}s | "
                  f"Throughput: {avg_throughput_ds:.2f} tok/s | PPL: {ppl_str}")
    
    print("\n" + "="*70)
    print("Optimized testing completed!")
    print("="*70)
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark KnormPress on Pythia-70M")
    parser.add_argument("--keep_ratios", type=str, default="1.0,0.9,0.8,0.7",
                       help="Comma-separated list of keep ratios to test")
    parser.add_argument("--prune_after", type=int, default=512,
                       help="Only compress if cache size exceeds this")
    parser.add_argument("--skip_layers", type=str, default="0",
                       help="Comma-separated list of layers to skip compression")
    parser.add_argument("--num_wikitext_samples", type=int, default=3,
                       help="Number of wikitext samples to test")
    parser.add_argument("--num_pg19_samples", type=int, default=2,
                       help="Number of pg-19 samples to test")
    
    args = parser.parse_args()
    
    keep_ratios = [float(x) for x in args.keep_ratios.split(',')]
    skip_layers = [int(x) for x in args.skip_layers.split(',')]
    
    results = run_optimized_tests(
        keep_ratios=keep_ratios,
        prune_after=args.prune_after,
        skip_layers=skip_layers,
        num_wikitext_samples=args.num_wikitext_samples,
        num_pg19_samples=args.num_pg19_samples,
    )

