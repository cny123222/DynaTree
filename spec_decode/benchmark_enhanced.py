"""
Enhanced Benchmark: Comprehensive Speculative Decoding Performance Analysis

This script provides in-depth performance analysis with the following features:

1. Per-phase timing breakdown (prefill, draft, verify, accept/reject)
2. Statistical analysis (mean, std, median, P90, P99)
3. Different input/output length analysis
4. Acceptance rate trends over generation
5. Memory profiling
6. Comparison between DynamicCache and StaticCache
7. Visualization with multiple plots

Usage:
    python benchmark_enhanced.py \
        --target-model /path/to/pythia-2.8b \
        --draft-model /path/to/pythia-70m \
        --k-values 3 5 7 \
        --num-samples 5 \
        --max-new-tokens 100
"""

import torch
import time
import argparse
import json
import gc
import os
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field, asdict
from tqdm import tqdm
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

# Suppress HuggingFace warnings
logging.set_verbosity_error()

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not found. Plotting will be disabled.")

from core import SpeculativeGenerator, SpeculativeGeneratorWithStaticCache


@dataclass
class PerRoundStats:
    """Statistics for a single speculative decoding round."""
    round_idx: int
    draft_time_ms: float
    verify_time_ms: float
    update_time_ms: float
    total_time_ms: float
    num_drafted: int
    num_accepted: int
    acceptance_rate: float


@dataclass
class EnhancedMetrics:
    """Enhanced metrics from a benchmark run."""
    method: str
    k_value: int
    input_length: int = 0
    output_length: int = 0
    
    # Throughput metrics
    throughput_mean: float = 0.0
    throughput_std: float = 0.0
    throughput_median: float = 0.0
    throughput_p90: float = 0.0
    throughput_p99: float = 0.0
    
    # TTFT metrics (ms)
    ttft_mean: float = 0.0
    ttft_std: float = 0.0
    ttft_median: float = 0.0
    ttft_p90: float = 0.0
    ttft_p99: float = 0.0
    
    # TPOT metrics (ms)
    tpot_mean: float = 0.0
    tpot_std: float = 0.0
    tpot_median: float = 0.0
    tpot_p90: float = 0.0
    tpot_p99: float = 0.0
    
    # Phase timing (ms) - Custom only
    prefill_time_mean: float = 0.0
    draft_time_mean: float = 0.0
    verify_time_mean: float = 0.0
    update_time_mean: float = 0.0
    
    # Acceptance metrics
    acceptance_rate: float = 0.0
    acceptance_rate_std: float = 0.0
    tokens_per_round_mean: float = 0.0
    tokens_per_round_std: float = 0.0
    total_rounds: int = 0
    
    # Memory metrics (MB)
    peak_memory_mb: float = 0.0
    allocated_memory_mb: float = 0.0
    
    # Token counts
    total_tokens: int = 0
    num_samples: int = 0
    total_time_s: float = 0.0
    
    # Per-sample raw data
    per_sample_throughput: List[float] = field(default_factory=list)
    per_sample_ttft: List[float] = field(default_factory=list)
    per_sample_tpot: List[float] = field(default_factory=list)
    per_sample_acceptance: List[float] = field(default_factory=list)
    
    # Per-round data (for trend analysis)
    per_round_acceptance: List[float] = field(default_factory=list)


def cleanup():
    """Clean up GPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def compute_percentiles(data: List[float]) -> Tuple[float, float, float, float, float]:
    """Compute mean, std, median, P90, P99 from data."""
    if not data:
        return 0.0, 0.0, 0.0, 0.0, 0.0
    arr = np.array(data)
    return (
        float(np.mean(arr)),
        float(np.std(arr)),
        float(np.median(arr)),
        float(np.percentile(arr, 90)),
        float(np.percentile(arr, 99))
    )


class InstrumentedGenerator(SpeculativeGenerator):
    """
    Instrumented version of SpeculativeGenerator that records per-phase timing.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.phase_times = {
            "prefill": [],
            "draft": [],
            "verify": [],
            "update": [],
            "total_round": []
        }
        self.round_stats: List[PerRoundStats] = []
        self._round_acceptance_rates = []
    
    def reset(self):
        """Reset including instrumentation data."""
        super().reset()
        self.phase_times = {
            "prefill": [],
            "draft": [],
            "verify": [],
            "update": [],
            "total_round": []
        }
        self.round_stats = []
        self._round_acceptance_rates = []
    
    @torch.inference_mode()
    def generate_instrumented(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        verbose: bool = False
    ) -> Tuple[str, Dict]:
        """
        Generate text with detailed timing instrumentation.
        
        Returns:
            text: Generated text
            timing: Dictionary with per-phase timing data
        """
        # Reset state
        self.reset()
        
        # Tokenize prompt
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_len - max_new_tokens
        ).input_ids.to(self.device)
        
        # Prefill with timing
        torch.cuda.synchronize()
        prefill_start = time.perf_counter()
        self._prefill(input_ids)
        torch.cuda.synchronize()
        prefill_time = (time.perf_counter() - prefill_start) * 1000
        self.phase_times["prefill"].append(prefill_time)
        
        if verbose:
            print(f"Prefilled {input_ids.shape[1]} tokens in {prefill_time:.2f}ms")
        
        generated = 0
        eos_token_id = self.tokenizer.eos_token_id
        round_idx = 0
        
        while generated < max_new_tokens:
            if self.current_ids.shape[1] + self.K >= self.max_len:
                break
            
            round_start = time.perf_counter()
            original_cache_len = self.target_cache.get_seq_length()
            
            # Phase 1: Draft
            torch.cuda.synchronize()
            draft_start = time.perf_counter()
            draft_tokens, draft_logits = self._draft_k_tokens()
            torch.cuda.synchronize()
            draft_time = (time.perf_counter() - draft_start) * 1000
            self.phase_times["draft"].append(draft_time)
            self.stats["total_draft_tokens"] += self.K
            
            # Phase 2: Verify
            torch.cuda.synchronize()
            verify_start = time.perf_counter()
            target_logits, verify_outputs = self._verify_tokens(draft_tokens)
            torch.cuda.synchronize()
            verify_time = (time.perf_counter() - verify_start) * 1000
            self.phase_times["verify"].append(verify_time)
            
            # Phase 3: Accept/Reject
            accepted_tokens, num_accepted, all_accepted = self._accept_reject_greedy(
                draft_tokens, target_logits
            )
            
            # Limit to max_new_tokens
            remaining = max_new_tokens - generated
            if num_accepted > remaining:
                accepted_tokens = accepted_tokens[:, :remaining]
                num_accepted = remaining
                all_accepted = False
            
            if num_accepted == 0:
                break
            
            # Phase 4: Update cache
            torch.cuda.synchronize()
            update_start = time.perf_counter()
            self._update_cache_and_logits(
                num_accepted, accepted_tokens, all_accepted, original_cache_len,
                verify_outputs=verify_outputs
            )
            torch.cuda.synchronize()
            update_time = (time.perf_counter() - update_start) * 1000
            self.phase_times["update"].append(update_time)
            
            # Round timing
            torch.cuda.synchronize()
            round_time = (time.perf_counter() - round_start) * 1000
            self.phase_times["total_round"].append(round_time)
            
            # Record round stats
            round_acceptance = num_accepted / (self.K + 1) if all_accepted else (num_accepted - 1) / self.K
            self._round_acceptance_rates.append(round_acceptance)
            
            self.round_stats.append(PerRoundStats(
                round_idx=round_idx,
                draft_time_ms=draft_time,
                verify_time_ms=verify_time,
                update_time_ms=update_time,
                total_time_ms=round_time,
                num_drafted=self.K,
                num_accepted=num_accepted,
                acceptance_rate=round_acceptance
            ))
            
            # Update sequence
            self.current_ids = torch.cat([self.current_ids, accepted_tokens], dim=-1)
            self.stats["total_accepted"] += num_accepted
            self.stats["total_tokens"] += num_accepted
            self.stats["total_rounds"] += 1
            generated += num_accepted
            round_idx += 1
            
            if verbose:
                print(f"Round {round_idx}: draft={draft_time:.1f}ms, verify={verify_time:.1f}ms, "
                      f"update={update_time:.1f}ms, accepted={num_accepted}/{self.K+1}")
            
            # Check for EOS
            if accepted_tokens[0, -1].item() == eos_token_id:
                break
        
        timing = {
            "prefill_ms": prefill_time,
            "avg_draft_ms": np.mean(self.phase_times["draft"]) if self.phase_times["draft"] else 0,
            "avg_verify_ms": np.mean(self.phase_times["verify"]) if self.phase_times["verify"] else 0,
            "avg_update_ms": np.mean(self.phase_times["update"]) if self.phase_times["update"] else 0,
            "avg_round_ms": np.mean(self.phase_times["total_round"]) if self.phase_times["total_round"] else 0,
            "per_round_acceptance": self._round_acceptance_rates.copy()
        }
        
        return self.tokenizer.decode(self.current_ids[0], skip_special_tokens=True), timing


def measure_baseline_enhanced(
    model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    device: str
) -> EnhancedMetrics:
    """Measure baseline performance with enhanced metrics."""
    metrics = EnhancedMetrics(method="Baseline", k_value=0)
    
    throughputs = []
    ttfts = []
    tpots = []
    total_tokens = 0
    total_time = 0.0
    
    # Warmup
    cleanup()
    dummy = tokenizer("Warmup", return_tensors="pt").to(device)
    with torch.inference_mode():
        _ = model.generate(**dummy, max_new_tokens=5, do_sample=False, 
                          pad_token_id=tokenizer.eos_token_id)
    torch.cuda.synchronize()
    cleanup()
    
    for prompt in tqdm(prompts, desc="Baseline", leave=False):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(device)
        input_len = inputs.input_ids.shape[1]
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        # TTFT measurement
        with torch.inference_mode():
            first_output = model.generate(
                **inputs, max_new_tokens=1, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        torch.cuda.synchronize()
        ttft = time.perf_counter() - start_time
        ttfts.append(ttft * 1000)
        
        # Full generation
        with torch.inference_mode():
            full_output = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        torch.cuda.synchronize()
        sample_time = time.perf_counter() - start_time
        
        new_tokens = full_output.shape[1] - input_len
        total_tokens += new_tokens
        total_time += sample_time
        
        if new_tokens > 1:
            tpot = ((sample_time - ttft) / (new_tokens - 1)) * 1000
            tpots.append(tpot)
        
        throughput = new_tokens / sample_time if sample_time > 0 else 0
        throughputs.append(throughput)
    
    # Compute percentiles
    metrics.throughput_mean, metrics.throughput_std, metrics.throughput_median, \
        metrics.throughput_p90, metrics.throughput_p99 = compute_percentiles(throughputs)
    metrics.ttft_mean, metrics.ttft_std, metrics.ttft_median, \
        metrics.ttft_p90, metrics.ttft_p99 = compute_percentiles(ttfts)
    metrics.tpot_mean, metrics.tpot_std, metrics.tpot_median, \
        metrics.tpot_p90, metrics.tpot_p99 = compute_percentiles(tpots)
    
    metrics.total_tokens = total_tokens
    metrics.num_samples = len(prompts)
    metrics.total_time_s = total_time
    metrics.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    metrics.allocated_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    metrics.per_sample_throughput = throughputs
    metrics.per_sample_ttft = ttfts
    metrics.per_sample_tpot = tpots
    
    return metrics


def measure_huggingface_enhanced(
    target_model,
    draft_model,
    tokenizer,
    prompts: List[str],
    k_value: int,
    max_new_tokens: int,
    device: str
) -> EnhancedMetrics:
    """Measure HuggingFace's native speculative decoding with enhanced metrics."""
    metrics = EnhancedMetrics(method="HuggingFace", k_value=k_value)
    
    throughputs = []
    ttfts = []
    tpots = []
    total_tokens = 0
    total_time = 0.0
    
    # Warmup
    cleanup()
    dummy = tokenizer("Warmup", return_tensors="pt").to(device)
    with torch.inference_mode():
        _ = target_model.generate(
            **dummy, assistant_model=draft_model, max_new_tokens=5,
            num_assistant_tokens=k_value, do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )
    torch.cuda.synchronize()
    cleanup()
    
    for prompt in tqdm(prompts, desc=f"HF (K={k_value})", leave=False):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1500).to(device)
        input_len = inputs.input_ids.shape[1]
        
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        
        # TTFT
        with torch.inference_mode():
            first_output = target_model.generate(
                **inputs, assistant_model=draft_model, max_new_tokens=1,
                num_assistant_tokens=k_value, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        torch.cuda.synchronize()
        ttft = time.perf_counter() - start_time
        ttfts.append(ttft * 1000)
        
        # Full generation
        with torch.inference_mode():
            full_output = target_model.generate(
                **inputs, assistant_model=draft_model, max_new_tokens=max_new_tokens,
                num_assistant_tokens=k_value, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )
        torch.cuda.synchronize()
        sample_time = time.perf_counter() - start_time
        
        new_tokens = full_output.shape[1] - input_len
        total_tokens += new_tokens
        total_time += sample_time
        
        if new_tokens > 1:
            tpot = ((sample_time - ttft) / (new_tokens - 1)) * 1000
            tpots.append(tpot)
        
        throughput = new_tokens / sample_time if sample_time > 0 else 0
        throughputs.append(throughput)
    
    # Compute percentiles
    metrics.throughput_mean, metrics.throughput_std, metrics.throughput_median, \
        metrics.throughput_p90, metrics.throughput_p99 = compute_percentiles(throughputs)
    metrics.ttft_mean, metrics.ttft_std, metrics.ttft_median, \
        metrics.ttft_p90, metrics.ttft_p99 = compute_percentiles(ttfts)
    metrics.tpot_mean, metrics.tpot_std, metrics.tpot_median, \
        metrics.tpot_p90, metrics.tpot_p99 = compute_percentiles(tpots)
    
    metrics.total_tokens = total_tokens
    metrics.num_samples = len(prompts)
    metrics.total_time_s = total_time
    metrics.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    metrics.allocated_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    metrics.per_sample_throughput = throughputs
    metrics.per_sample_ttft = ttfts
    metrics.per_sample_tpot = tpots
    
    return metrics


def measure_custom_enhanced(
    target_model,
    draft_model,
    tokenizer,
    prompts: List[str],
    k_value: int,
    max_new_tokens: int,
    device: str,
    use_compile: bool = False
) -> EnhancedMetrics:
    """Measure custom implementation with enhanced instrumentation."""
    metrics = EnhancedMetrics(method="Custom", k_value=k_value)
    
    # Create instrumented generator
    generator = InstrumentedGenerator(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        K=k_value,
        max_len=2048,
        device=device,
        use_compile=use_compile
    )
    
    throughputs = []
    ttfts = []
    tpots = []
    acceptances = []
    tokens_per_round_list = []
    all_round_acceptances = []
    
    prefill_times = []
    draft_times = []
    verify_times = []
    update_times = []
    
    total_tokens = 0
    total_rounds = 0
    total_time = 0.0
    
    # Warmup
    cleanup()
    generator.reset()
    _ = generator.generate("Warmup", max_new_tokens=5)
    torch.cuda.synchronize()
    cleanup()
    
    for prompt in tqdm(prompts, desc=f"Custom (K={k_value})", leave=False):
        # TTFT measurement
        generator.reset()
        torch.cuda.synchronize()
        ttft_start = time.perf_counter()
        _, _ = generator.generate_instrumented(prompt, max_new_tokens=1)
        torch.cuda.synchronize()
        ttft = (time.perf_counter() - ttft_start) * 1000
        ttfts.append(ttft)
        
        # Full generation with instrumentation
        generator.reset()
        torch.cuda.synchronize()
        start_time = time.perf_counter()
        _, timing = generator.generate_instrumented(prompt, max_new_tokens=max_new_tokens)
        torch.cuda.synchronize()
        sample_time = time.perf_counter() - start_time
        
        stats = generator.get_stats()
        new_tokens = stats["total_tokens"]
        total_tokens += new_tokens
        total_rounds += stats["total_rounds"]
        total_time += sample_time
        
        # Phase times
        prefill_times.append(timing["prefill_ms"])
        draft_times.append(timing["avg_draft_ms"])
        verify_times.append(timing["avg_verify_ms"])
        update_times.append(timing["avg_update_ms"])
        
        # Round acceptance trend
        all_round_acceptances.extend(timing["per_round_acceptance"])
        
        if new_tokens > 1:
            tpot = ((sample_time * 1000 - ttft) / (new_tokens - 1))
            tpots.append(tpot)
        
        throughput = new_tokens / sample_time if sample_time > 0 else 0
        throughputs.append(throughput)
        
        acc = stats["acceptance_rate"]
        acceptances.append(acc)
        
        if stats["total_rounds"] > 0:
            tokens_per_round_list.append(stats["total_tokens"] / stats["total_rounds"])
    
    # Compute percentiles
    metrics.throughput_mean, metrics.throughput_std, metrics.throughput_median, \
        metrics.throughput_p90, metrics.throughput_p99 = compute_percentiles(throughputs)
    metrics.ttft_mean, metrics.ttft_std, metrics.ttft_median, \
        metrics.ttft_p90, metrics.ttft_p99 = compute_percentiles(ttfts)
    metrics.tpot_mean, metrics.tpot_std, metrics.tpot_median, \
        metrics.tpot_p90, metrics.tpot_p99 = compute_percentiles(tpots)
    
    # Phase timing means
    metrics.prefill_time_mean = np.mean(prefill_times) if prefill_times else 0
    metrics.draft_time_mean = np.mean(draft_times) if draft_times else 0
    metrics.verify_time_mean = np.mean(verify_times) if verify_times else 0
    metrics.update_time_mean = np.mean(update_times) if update_times else 0
    
    # Acceptance metrics
    metrics.acceptance_rate = np.mean(acceptances) if acceptances else 0
    metrics.acceptance_rate_std = np.std(acceptances) if acceptances else 0
    metrics.tokens_per_round_mean = np.mean(tokens_per_round_list) if tokens_per_round_list else 0
    metrics.tokens_per_round_std = np.std(tokens_per_round_list) if tokens_per_round_list else 0
    metrics.total_rounds = total_rounds
    
    metrics.total_tokens = total_tokens
    metrics.num_samples = len(prompts)
    metrics.total_time_s = total_time
    metrics.peak_memory_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    metrics.allocated_memory_mb = torch.cuda.memory_allocated() / (1024 ** 2)
    
    metrics.per_sample_throughput = throughputs
    metrics.per_sample_ttft = ttfts
    metrics.per_sample_tpot = tpots
    metrics.per_sample_acceptance = acceptances
    metrics.per_round_acceptance = all_round_acceptances
    
    return metrics


def print_enhanced_summary(
    baseline: EnhancedMetrics,
    hf_results: List[EnhancedMetrics],
    custom_results: List[EnhancedMetrics]
):
    """Print comprehensive summary of benchmark results."""
    print("\n" + "=" * 110)
    print("ENHANCED BENCHMARK SUMMARY")
    print("=" * 110)
    
    # Throughput table with percentiles
    print("\n📊 Throughput (tokens/sec)")
    print("-" * 110)
    print(f"{'Method':<12} {'K':<4} {'Mean':<12} {'Std':<10} {'Median':<10} {'P90':<10} {'P99':<10} {'Speedup':<10}")
    print("-" * 110)
    
    print(f"{'Baseline':<12} {'-':<4} {baseline.throughput_mean:<12.1f} {baseline.throughput_std:<10.1f} "
          f"{baseline.throughput_median:<10.1f} {baseline.throughput_p90:<10.1f} {baseline.throughput_p99:<10.1f} {'1.00x':<10}")
    
    for m in sorted(hf_results + custom_results, key=lambda x: (x.method, x.k_value)):
        speedup = m.throughput_mean / baseline.throughput_mean if baseline.throughput_mean > 0 else 0
        print(f"{m.method:<12} {m.k_value:<4} {m.throughput_mean:<12.1f} {m.throughput_std:<10.1f} "
              f"{m.throughput_median:<10.1f} {m.throughput_p90:<10.1f} {m.throughput_p99:<10.1f} {speedup:.2f}x")
    
    # Latency table
    print("\n⏱️ Latency (ms)")
    print("-" * 110)
    print(f"{'Method':<12} {'K':<4} {'TTFT Mean':<12} {'TTFT P90':<10} {'TPOT Mean':<12} {'TPOT P90':<10}")
    print("-" * 110)
    
    print(f"{'Baseline':<12} {'-':<4} {baseline.ttft_mean:<12.1f} {baseline.ttft_p90:<10.1f} "
          f"{baseline.tpot_mean:<12.1f} {baseline.tpot_p90:<10.1f}")
    
    for m in sorted(hf_results + custom_results, key=lambda x: (x.method, x.k_value)):
        print(f"{m.method:<12} {m.k_value:<4} {m.ttft_mean:<12.1f} {m.ttft_p90:<10.1f} "
              f"{m.tpot_mean:<12.1f} {m.tpot_p90:<10.1f}")
    
    # Phase breakdown for Custom
    if custom_results:
        print("\n⚙️ Phase Timing Breakdown - Custom (ms per round)")
        print("-" * 90)
        print(f"{'K':<5} {'Prefill':<12} {'Draft':<12} {'Verify':<12} {'Update':<12} {'Total Round':<12}")
        print("-" * 90)
        
        for m in sorted(custom_results, key=lambda x: x.k_value):
            total = m.draft_time_mean + m.verify_time_mean + m.update_time_mean
            print(f"{m.k_value:<5} {m.prefill_time_mean:<12.2f} {m.draft_time_mean:<12.2f} "
                  f"{m.verify_time_mean:<12.2f} {m.update_time_mean:<12.2f} {total:<12.2f}")
    
    # Acceptance rate analysis
    if custom_results:
        print("\n🎯 Acceptance Rate Analysis - Custom")
        print("-" * 70)
        print(f"{'K':<5} {'Accept Rate':<15} {'Std':<10} {'Tokens/Round':<15} {'Total Rounds':<12}")
        print("-" * 70)
        
        for m in sorted(custom_results, key=lambda x: x.k_value):
            print(f"{m.k_value:<5} {m.acceptance_rate:<15.2%} {m.acceptance_rate_std:<10.2%} "
                  f"{m.tokens_per_round_mean:<15.2f} {m.total_rounds:<12}")
    
    # Memory usage
    print("\n💾 Memory Usage (MB)")
    print("-" * 50)
    print(f"{'Method':<12} {'K':<4} {'Peak VRAM':<15} {'Allocated':<15}")
    print("-" * 50)
    
    print(f"{'Baseline':<12} {'-':<4} {baseline.peak_memory_mb:<15.1f} {baseline.allocated_memory_mb:<15.1f}")
    
    for m in sorted(hf_results + custom_results, key=lambda x: (x.method, x.k_value)):
        print(f"{m.method:<12} {m.k_value:<4} {m.peak_memory_mb:<15.1f} {m.allocated_memory_mb:<15.1f}")
    
    print("=" * 110)


def plot_results(
    baseline: EnhancedMetrics,
    hf_results: List[EnhancedMetrics],
    custom_results: List[EnhancedMetrics],
    save_path: str = "benchmark_enhanced.png"
):
    """Generate comprehensive visualization of benchmark results."""
    if not HAS_MATPLOTLIB:
        print("Skipping plots (matplotlib not available)")
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Enhanced Speculative Decoding Benchmark", fontsize=16, fontweight='bold')
    
    # Extract K values
    k_values = sorted(set(m.k_value for m in hf_results + custom_results if m.k_value > 0))
    
    # Plot 1: Throughput comparison
    ax1 = axes[0, 0]
    ax1.set_title("Throughput vs K Value", fontsize=12, fontweight='bold')
    ax1.axhline(y=baseline.throughput_mean, color='gray', linestyle='--', linewidth=2, label='Baseline')
    
    hf_throughputs = [next((m.throughput_mean for m in hf_results if m.k_value == k), None) for k in k_values]
    custom_throughputs = [next((m.throughput_mean for m in custom_results if m.k_value == k), None) for k in k_values]
    
    if any(t is not None for t in hf_throughputs):
        valid_k = [k for k, t in zip(k_values, hf_throughputs) if t is not None]
        valid_t = [t for t in hf_throughputs if t is not None]
        ax1.plot(valid_k, valid_t, 'b-o', linewidth=2, markersize=8, label='HuggingFace')
    
    if any(t is not None for t in custom_throughputs):
        valid_k = [k for k, t in zip(k_values, custom_throughputs) if t is not None]
        valid_t = [t for t in custom_throughputs if t is not None]
        ax1.plot(valid_k, valid_t, 'r-s', linewidth=2, markersize=8, label='Custom')
    
    ax1.set_xlabel("K (num assistant tokens)")
    ax1.set_ylabel("Throughput (tokens/sec)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Speedup comparison
    ax2 = axes[0, 1]
    ax2.set_title("Speedup vs K Value", fontsize=12, fontweight='bold')
    ax2.axhline(y=1.0, color='gray', linestyle='--', linewidth=2, label='Baseline')
    
    if any(t is not None for t in hf_throughputs):
        valid_k = [k for k, t in zip(k_values, hf_throughputs) if t is not None]
        speedups = [t / baseline.throughput_mean for t in hf_throughputs if t is not None]
        ax2.plot(valid_k, speedups, 'b-o', linewidth=2, markersize=8, label='HuggingFace')
    
    if any(t is not None for t in custom_throughputs):
        valid_k = [k for k, t in zip(k_values, custom_throughputs) if t is not None]
        speedups = [t / baseline.throughput_mean for t in custom_throughputs if t is not None]
        ax2.plot(valid_k, speedups, 'r-s', linewidth=2, markersize=8, label='Custom')
    
    ax2.set_xlabel("K (num assistant tokens)")
    ax2.set_ylabel("Speedup (x)")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Acceptance rate vs K
    ax3 = axes[0, 2]
    ax3.set_title("Acceptance Rate vs K Value", fontsize=12, fontweight='bold')
    
    if custom_results:
        acc_rates = [m.acceptance_rate for m in sorted(custom_results, key=lambda x: x.k_value)]
        acc_stds = [m.acceptance_rate_std for m in sorted(custom_results, key=lambda x: x.k_value)]
        custom_k = [m.k_value for m in sorted(custom_results, key=lambda x: x.k_value)]
        ax3.errorbar(custom_k, acc_rates, yerr=acc_stds, fmt='g-^', linewidth=2, 
                     markersize=8, capsize=4, label='Custom')
        ax3.set_xlabel("K (num assistant tokens)")
        ax3.set_ylabel("Acceptance Rate")
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        ax3.set_ylim(0, 1)
    
    # Plot 4: TTFT comparison
    ax4 = axes[1, 0]
    ax4.set_title("TTFT (Time to First Token)", fontsize=12, fontweight='bold')
    
    methods = ['Baseline'] + [f'HF K={k}' for k in k_values] + [f'Custom K={k}' for k in k_values]
    ttfts = [baseline.ttft_mean]
    ttfts += [next((m.ttft_mean for m in hf_results if m.k_value == k), 0) for k in k_values]
    ttfts += [next((m.ttft_mean for m in custom_results if m.k_value == k), 0) for k in k_values]
    
    colors = ['gray'] + ['blue'] * len(k_values) + ['red'] * len(k_values)
    bars = ax4.bar(range(len(methods)), ttfts, color=colors, alpha=0.7)
    ax4.set_xticks(range(len(methods)))
    ax4.set_xticklabels(methods, rotation=45, ha='right')
    ax4.set_ylabel("TTFT (ms)")
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: TPOT comparison
    ax5 = axes[1, 1]
    ax5.set_title("TPOT (Time per Output Token)", fontsize=12, fontweight='bold')
    
    tpots = [baseline.tpot_mean]
    tpots += [next((m.tpot_mean for m in hf_results if m.k_value == k), 0) for k in k_values]
    tpots += [next((m.tpot_mean for m in custom_results if m.k_value == k), 0) for k in k_values]
    
    bars = ax5.bar(range(len(methods)), tpots, color=colors, alpha=0.7)
    ax5.set_xticks(range(len(methods)))
    ax5.set_xticklabels(methods, rotation=45, ha='right')
    ax5.set_ylabel("TPOT (ms)")
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Phase timing breakdown (stacked bar)
    ax6 = axes[1, 2]
    ax6.set_title("Phase Timing Breakdown (Custom)", fontsize=12, fontweight='bold')
    
    if custom_results:
        custom_k = [m.k_value for m in sorted(custom_results, key=lambda x: x.k_value)]
        draft_times = [m.draft_time_mean for m in sorted(custom_results, key=lambda x: x.k_value)]
        verify_times = [m.verify_time_mean for m in sorted(custom_results, key=lambda x: x.k_value)]
        update_times = [m.update_time_mean for m in sorted(custom_results, key=lambda x: x.k_value)]
        
        x = np.arange(len(custom_k))
        width = 0.6
        
        ax6.bar(x, draft_times, width, label='Draft', color='#2ecc71')
        ax6.bar(x, verify_times, width, bottom=draft_times, label='Verify', color='#3498db')
        ax6.bar(x, update_times, width, 
                bottom=[d + v for d, v in zip(draft_times, verify_times)], 
                label='Update', color='#e74c3c')
        
        ax6.set_xticks(x)
        ax6.set_xticklabels([f'K={k}' for k in custom_k])
        ax6.set_ylabel("Time per Round (ms)")
        ax6.legend()
        ax6.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n✅ Plot saved to: {save_path}")


def save_results(
    baseline: EnhancedMetrics,
    hf_results: List[EnhancedMetrics],
    custom_results: List[EnhancedMetrics],
    output_path: str
):
    """Save results to JSON file."""
    results = {
        "baseline": asdict(baseline),
        "huggingface": [asdict(m) for m in hf_results],
        "custom": [asdict(m) for m in custom_results]
    }
    
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"✅ Results saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Enhanced Speculative Decoding Benchmark")
    parser.add_argument("--target-model", type=str, default="/mnt/disk1/models/pythia-2.8b",
                        help="Path to target model")
    parser.add_argument("--draft-model", type=str, default="/mnt/disk1/models/pythia-70m",
                        help="Path to draft model")
    parser.add_argument("--k-values", type=int, nargs="+", default=[3, 5, 7],
                        help="K values to test")
    parser.add_argument("--num-samples", type=int, default=5,
                        help="Number of test samples")
    parser.add_argument("--max-new-tokens", type=int, default=100,
                        help="Maximum new tokens to generate")
    parser.add_argument("--device", type=str, default="cuda:0",
                        help="Device to use")
    parser.add_argument("--output-json", type=str, default="benchmark_enhanced_results.json",
                        help="Output JSON filename")
    parser.add_argument("--output-plot", type=str, default="benchmark_enhanced.png",
                        help="Output plot filename")
    parser.add_argument("--skip-hf", action="store_true",
                        help="Skip HuggingFace benchmark")
    parser.add_argument("--skip-custom", action="store_true",
                        help="Skip Custom benchmark")
    parser.add_argument("--use-compile", action="store_true",
                        help="Enable torch.compile for Custom")
    args = parser.parse_args()
    
    print("=" * 80)
    print("🚀 ENHANCED SPECULATIVE DECODING BENCHMARK")
    print("=" * 80)
    print(f"   Target Model: {args.target_model}")
    print(f"   Draft Model:  {args.draft_model}")
    print(f"   K values:     {args.k_values}")
    print(f"   Samples:      {args.num_samples}")
    print(f"   Max tokens:   {args.max_new_tokens}")
    print(f"   Device:       {args.device}")
    print(f"   Compile:      {args.use_compile}")
    print("=" * 80)
    
    # Test prompts
    prompts = [
        "The quick brown fox jumps over the lazy dog. This is a test of",
        "In the beginning of time, there was nothing but darkness and",
        "Machine learning is a subset of artificial intelligence that",
        "The capital of France is Paris, which is known for its",
        "Once upon a time in a land far away, there lived a",
        "The theory of relativity, proposed by Albert Einstein,",
        "Python is a popular programming language because it is",
        "Climate change is one of the most pressing issues facing",
        "The human brain contains approximately 86 billion neurons that",
        "Quantum computing represents a paradigm shift in how we",
    ][:args.num_samples]
    
    device = args.device
    
    # Load models
    print(f"\n📦 Loading models...")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.float16
    ).to(device)
    
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype=torch.float16
    ).to(device)
    
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    target_model.eval()
    draft_model.eval()
    
    print("   Models loaded successfully!")
    
    # Baseline benchmark
    print(f"\n🏃 Running Baseline benchmark...")
    baseline = measure_baseline_enhanced(target_model, tokenizer, prompts, args.max_new_tokens, device)
    print(f"   Throughput: {baseline.throughput_mean:.1f} ± {baseline.throughput_std:.1f} t/s")
    print(f"   TTFT: {baseline.ttft_mean:.1f} ± {baseline.ttft_std:.1f} ms")
    print(f"   TPOT: {baseline.tpot_mean:.1f} ± {baseline.tpot_std:.1f} ms")
    
    # HuggingFace benchmarks
    hf_results = []
    if not args.skip_hf:
        print(f"\n🏃 Running HuggingFace benchmarks...")
        for k in args.k_values:
            result = measure_huggingface_enhanced(
                target_model, draft_model, tokenizer, prompts,
                k, args.max_new_tokens, device
            )
            hf_results.append(result)
            speedup = result.throughput_mean / baseline.throughput_mean
            print(f"   K={k}: {result.throughput_mean:.1f} t/s ({speedup:.2f}x), "
                  f"TTFT={result.ttft_mean:.1f}ms, TPOT={result.tpot_mean:.1f}ms")
    
    # Custom implementation benchmarks
    custom_results = []
    if not args.skip_custom:
        print(f"\n🏃 Running Custom implementation benchmarks...")
        for k in args.k_values:
            result = measure_custom_enhanced(
                target_model, draft_model, tokenizer, prompts,
                k, args.max_new_tokens, device, args.use_compile
            )
            custom_results.append(result)
            speedup = result.throughput_mean / baseline.throughput_mean
            print(f"   K={k}: {result.throughput_mean:.1f} t/s ({speedup:.2f}x), "
                  f"acceptance={result.acceptance_rate:.1%}, "
                  f"TTFT={result.ttft_mean:.1f}ms, TPOT={result.tpot_mean:.1f}ms")
            print(f"         Phase times: draft={result.draft_time_mean:.1f}ms, "
                  f"verify={result.verify_time_mean:.1f}ms, update={result.update_time_mean:.1f}ms")
    
    # Print summary
    print_enhanced_summary(baseline, hf_results, custom_results)
    
    # Plot results
    plot_results(baseline, hf_results, custom_results, args.output_plot)
    
    # Save results
    save_results(baseline, hf_results, custom_results, args.output_json)


if __name__ == "__main__":
    main()

