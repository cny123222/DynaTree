"""
Comprehensive Benchmark: Full metrics for Speculative Decoding with StreamingLLM

Metrics included:
1. Throughput (tokens/s)
2. TTFT (Time to First Token, ms)
3. TPOT (Time per Output Token, ms)
4. PPL (Perplexity) - measure generation quality
5. Acceptance Rate
6. Memory Usage
7. Compression Statistics (for StreamingLLM)

Usage:
    python benchmark_comprehensive.py \
        --max-new-tokens 500 1000 2000 \
        --max-cache-lens 256 512 1024 \
        --num-samples 3
"""

import torch
import torch.nn.functional as F
import time
import argparse
import json
import gc
import math
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, logging

logging.set_verbosity_error()

from core import SpeculativeGenerator, StreamingSpeculativeGenerator


@dataclass 
class ComprehensiveMetrics:
    """Comprehensive metrics for benchmark."""
    method: str
    target_tokens: int
    actual_tokens: int
    max_cache_len: int
    
    # Throughput metrics
    throughput_mean: float = 0.0
    throughput_std: float = 0.0
    
    # Latency metrics (ms)
    ttft_mean: float = 0.0
    ttft_std: float = 0.0
    tpot_mean: float = 0.0
    tpot_std: float = 0.0
    total_time_mean: float = 0.0
    total_time_std: float = 0.0
    
    # Quality metrics
    ppl_mean: float = 0.0
    ppl_std: float = 0.0
    
    # Speculative decoding metrics
    acceptance_rate: float = 0.0  # Draft acceptance (capped at 100%)
    acceptance_with_bonus: float = 0.0  # Including bonus tokens (can be > 100%)
    tokens_per_round: float = 0.0
    total_rounds: int = 0
    
    # Memory metrics
    peak_memory_mb: float = 0.0
    memory_growth_mb: float = 0.0
    
    # StreamingLLM metrics
    compression_count: int = 0
    tokens_evicted: int = 0
    
    # Number of samples
    num_samples: int = 0


def cleanup():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def get_test_prompts() -> List[str]:
    """Get diverse test prompts for comprehensive evaluation."""
    return [
        "The history of artificial intelligence began in the 1950s when",
        "In modern computer science, the most important algorithms include",
        "The theory of relativity, proposed by Albert Einstein, explains that",
        "Machine learning models are trained using large datasets to",
        "Climate change is affecting global ecosystems in several ways including",
    ]


def compute_perplexity_sliding_window(
    model, 
    tokenizer, 
    text: str, 
    device: str,
    stride: int = 512
) -> float:
    """
    Compute perplexity using sliding window method.
    
    This measures how well the model predicts the generated text,
    which indicates generation quality/coherence.
    
    Note: For speculative decoding, output should be identical to autoregressive,
    so PPL should be the same. However, StreamingLLM with small cache may 
    degrade quality due to lost context.
    """
    encodings = tokenizer(text, return_tensors="pt", truncation=True, max_length=4096)
    seq_len = encodings.input_ids.size(1)
    
    if seq_len < 2:
        return float('nan')
    
    max_length = min(2048, model.config.max_position_embeddings if hasattr(model.config, 'max_position_embeddings') else 2048)
    
    nlls = []
    prev_end_loc = 0
    
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # target length for this window
        
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        # Only compute loss on the new tokens (not the context)
        target_ids[:, :-trg_len] = -100
        
        with torch.inference_mode():
            outputs = model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss * trg_len
        
        nlls.append(neg_log_likelihood.item())
        
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    
    total_nll = sum(nlls)
    total_tokens = prev_end_loc
    
    ppl = math.exp(total_nll / total_tokens) if total_tokens > 0 else float('inf')
    return ppl if ppl < 1e6 else float('inf')


def measure_ttft(
    generator,
    prompt: str,
    device: str
) -> float:
    """Measure Time to First Token by generating just 1 token."""
    generator.reset()
    
    # Disable EOS temporarily
    original_eos = generator.tokenizer.eos_token_id
    generator.tokenizer.eos_token_id = 999999
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    try:
        _ = generator.generate(prompt, max_new_tokens=1)
    finally:
        generator.tokenizer.eos_token_id = original_eos
    
    torch.cuda.synchronize()
    ttft = (time.perf_counter() - start) * 1000  # ms
    
    return ttft


def measure_full_generation(
    generator,
    prompt: str,
    max_new_tokens: int,
    target_model,
    tokenizer,
    device: str
) -> Dict:
    """Measure full generation with all metrics."""
    generator.reset()
    
    # Disable EOS temporarily
    original_eos = tokenizer.eos_token_id
    tokenizer.eos_token_id = 999999
    
    torch.cuda.synchronize()
    start = time.perf_counter()
    
    try:
        output_text = generator.generate(prompt, max_new_tokens=max_new_tokens)
    finally:
        tokenizer.eos_token_id = original_eos
    
    torch.cuda.synchronize()
    total_time = time.perf_counter() - start
    
    stats = generator.get_stats()
    tokens_generated = stats["total_tokens"]
    
    # Compute PPL using sliding window method
    ppl = compute_perplexity_sliding_window(target_model, tokenizer, output_text, device)
    
    # Compute CORRECT acceptance rate:
    # total_accepted includes bonus tokens, so we need to exclude them
    # When all K drafts accepted: num_accepted = K+1 (with bonus)
    # Correct acceptance = (total_accepted - bonus_tokens) / total_draft
    # bonus_tokens ≈ rounds where all K were accepted
    # A simpler approximation: acceptance_rate = min(total_accepted/total_draft, 1.0)
    # Or compute: accepted_draft_only = total_tokens - total_rounds (each round gets at least 1 token)
    total_rounds = stats.get('total_rounds', 0)
    total_draft = stats.get('total_draft_tokens', 0)
    total_accepted = stats.get('total_accepted', 0)
    
    # Each round: drafts K tokens, accepts 1 to K+1 tokens
    # Bonus tokens = total_accepted - (accepted_drafts)
    # accepted_drafts ≤ total_draft (K per round)
    # So correct acceptance rate = min(total_accepted, total_draft) / total_draft
    if total_draft > 0:
        # Cap at 100% for meaningful interpretation
        correct_acceptance = min(total_accepted, total_draft) / total_draft
    else:
        correct_acceptance = 0
    
    return {
        'total_time': total_time,
        'tokens_generated': tokens_generated,
        'output_text': output_text,
        'ppl': ppl,
        'acceptance_rate': correct_acceptance,  # Capped at 100%
        'acceptance_rate_with_bonus': total_accepted / total_draft if total_draft > 0 else 0,  # Original
        'total_rounds': total_rounds,
        'total_accepted': total_accepted,
        'total_draft': total_draft,
        'compression_count': stats.get('compression_count', 0),
        'tokens_evicted': stats.get('tokens_evicted', 0),
    }


def measure_standard(
    target_model,
    draft_model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    k_value: int,
    device: str
) -> ComprehensiveMetrics:
    """Measure standard speculative decoding with all metrics."""
    cleanup()
    memory_start = torch.cuda.memory_allocated() / (1024**2)
    torch.cuda.reset_peak_memory_stats()
    
    ttfts = []
    tpots = []
    throughputs = []
    ppls = []
    total_times = []
    all_tokens = []
    total_rounds = 0
    total_accepted = 0
    total_draft = 0
    acceptance_rates = []
    acceptance_with_bonus_list = []
    
    for prompt in prompts:
        generator = SpeculativeGenerator(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            K=k_value,
            max_len=8192,
            device=device,
            use_compile=False
        )
        
        # Measure TTFT
        ttft = measure_ttft(generator, prompt, device)
        ttfts.append(ttft)
        
        # Measure full generation
        result = measure_full_generation(
            generator, prompt, max_new_tokens,
            target_model, tokenizer, device
        )
        
        total_times.append(result['total_time'])
        tokens = result['tokens_generated']
        all_tokens.append(tokens)
        
        # TPOT = (total_time - ttft/1000) / (tokens - 1) * 1000 if tokens > 1
        if tokens > 1:
            tpot = (result['total_time'] - ttft/1000) / (tokens - 1) * 1000
            tpots.append(tpot)
        
        throughput = tokens / result['total_time'] if result['total_time'] > 0 else 0
        throughputs.append(throughput)
        
        ppls.append(result['ppl'])
        total_rounds += result['total_rounds']
        total_accepted += result['total_accepted']
        total_draft += result['total_draft']
        acceptance_rates.append(result['acceptance_rate'])
        acceptance_with_bonus_list.append(result['acceptance_rate_with_bonus'])
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
    total_tokens = sum(all_tokens)
    
    # Filter valid PPL values (not inf or nan)
    valid_ppls = [p for p in ppls if p < 1e6 and not math.isnan(p)]
    
    return ComprehensiveMetrics(
        method="standard",
        target_tokens=max_new_tokens,
        actual_tokens=int(np.mean(all_tokens)),
        max_cache_len=0,
        throughput_mean=np.mean(throughputs),
        throughput_std=np.std(throughputs),
        ttft_mean=np.mean(ttfts),
        ttft_std=np.std(ttfts),
        tpot_mean=np.mean(tpots) if tpots else 0,
        tpot_std=np.std(tpots) if tpots else 0,
        total_time_mean=np.mean(total_times),
        total_time_std=np.std(total_times),
        ppl_mean=np.mean(valid_ppls) if valid_ppls else float('nan'),
        ppl_std=np.std(valid_ppls) if len(valid_ppls) > 1 else 0,
        acceptance_rate=np.mean(acceptance_rates),  # Capped at 100%
        acceptance_with_bonus=np.mean(acceptance_with_bonus_list),  # Can be > 100%
        tokens_per_round=total_tokens / total_rounds if total_rounds > 0 else 0,
        total_rounds=total_rounds,
        peak_memory_mb=peak_memory,
        memory_growth_mb=peak_memory - memory_start,
        compression_count=0,
        tokens_evicted=0,
        num_samples=len(prompts)
    )


def measure_streaming(
    target_model,
    draft_model,
    tokenizer,
    prompts: List[str],
    max_new_tokens: int,
    max_cache_len: int,
    k_value: int,
    device: str
) -> ComprehensiveMetrics:
    """Measure streaming speculative decoding with all metrics."""
    cleanup()
    memory_start = torch.cuda.memory_allocated() / (1024**2)
    torch.cuda.reset_peak_memory_stats()
    
    ttfts = []
    tpots = []
    throughputs = []
    ppls = []
    total_times = []
    all_tokens = []
    total_rounds = 0
    total_accepted = 0
    total_draft = 0
    total_compressions = 0
    total_evicted = 0
    acceptance_rates = []
    acceptance_with_bonus_list = []
    
    for prompt in prompts:
        generator = StreamingSpeculativeGenerator(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            K=k_value,
            max_len=8192,
            device=device,
            use_compile=False,
            max_cache_len=max_cache_len,
            start_size=4,
            recent_size=max_cache_len - 4
        )
        
        # Measure TTFT
        ttft = measure_ttft(generator, prompt, device)
        ttfts.append(ttft)
        
        # Measure full generation
        result = measure_full_generation(
            generator, prompt, max_new_tokens,
            target_model, tokenizer, device
        )
        
        total_times.append(result['total_time'])
        tokens = result['tokens_generated']
        all_tokens.append(tokens)
        
        if tokens > 1:
            tpot = (result['total_time'] - ttft/1000) / (tokens - 1) * 1000
            tpots.append(tpot)
        
        throughput = tokens / result['total_time'] if result['total_time'] > 0 else 0
        throughputs.append(throughput)
        
        ppls.append(result['ppl'])
        total_rounds += result['total_rounds']
        total_accepted += result['total_accepted']
        total_draft += result['total_draft']
        total_compressions += result['compression_count']
        total_evicted += result['tokens_evicted']
        acceptance_rates.append(result['acceptance_rate'])
        acceptance_with_bonus_list.append(result['acceptance_rate_with_bonus'])
    
    peak_memory = torch.cuda.max_memory_allocated() / (1024**2)
    total_tokens = sum(all_tokens)
    
    # Filter valid PPL values
    valid_ppls = [p for p in ppls if p < 1e6 and not math.isnan(p)]
    
    return ComprehensiveMetrics(
        method="streaming",
        target_tokens=max_new_tokens,
        actual_tokens=int(np.mean(all_tokens)),
        max_cache_len=max_cache_len,
        throughput_mean=np.mean(throughputs),
        throughput_std=np.std(throughputs),
        ttft_mean=np.mean(ttfts),
        ttft_std=np.std(ttfts),
        tpot_mean=np.mean(tpots) if tpots else 0,
        tpot_std=np.std(tpots) if tpots else 0,
        total_time_mean=np.mean(total_times),
        total_time_std=np.std(total_times),
        ppl_mean=np.mean(valid_ppls) if valid_ppls else float('nan'),
        ppl_std=np.std(valid_ppls) if len(valid_ppls) > 1 else 0,
        acceptance_rate=np.mean(acceptance_rates),  # Capped at 100%
        acceptance_with_bonus=np.mean(acceptance_with_bonus_list),  # Can be > 100%
        tokens_per_round=total_tokens / total_rounds if total_rounds > 0 else 0,
        total_rounds=total_rounds,
        peak_memory_mb=peak_memory,
        memory_growth_mb=peak_memory - memory_start,
        compression_count=total_compressions,
        tokens_evicted=total_evicted,
        num_samples=len(prompts)
    )


def plot_comprehensive_results(results: List[ComprehensiveMetrics], save_path: str):
    """Plot comprehensive benchmark results."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.gridspec as gridspec
    except ImportError:
        print("matplotlib not available, skipping plot")
        return
    
    fig = plt.figure(figsize=(16, 12))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Separate results
    standard = [r for r in results if r.method == "standard"]
    streaming = [r for r in results if r.method == "streaming"]
    standard.sort(key=lambda x: x.target_tokens)
    
    cache_sizes = sorted(set(r.max_cache_len for r in streaming if r.max_cache_len > 0))
    colors = {'standard': '#2C3E50', 256: '#E74C3C', 512: '#3498DB', 1024: '#2ECC71'}
    
    # Plot 1: Throughput
    ax1 = fig.add_subplot(gs[0, 0])
    if standard:
        x = [r.target_tokens for r in standard]
        y = [r.throughput_mean for r in standard]
        yerr = [r.throughput_std for r in standard]
        ax1.errorbar(x, y, yerr=yerr, fmt='o-', color=colors['standard'], 
                    linewidth=2, markersize=8, capsize=4, label='Standard')
    
    for cache in cache_sizes:
        stream_cache = sorted([r for r in streaming if r.max_cache_len == cache], 
                             key=lambda x: x.target_tokens)
        if stream_cache:
            x = [r.target_tokens for r in stream_cache]
            y = [r.throughput_mean for r in stream_cache]
            yerr = [r.throughput_std for r in stream_cache]
            ax1.errorbar(x, y, yerr=yerr, fmt='o--', color=colors.get(cache, '#888'),
                        linewidth=2, markersize=8, capsize=4, label=f'Stream({cache})')
    
    ax1.set_xlabel('Target Tokens')
    ax1.set_ylabel('Throughput (t/s)')
    ax1.set_title('Throughput')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: TTFT
    ax2 = fig.add_subplot(gs[0, 1])
    if standard:
        x = [r.target_tokens for r in standard]
        y = [r.ttft_mean for r in standard]
        yerr = [r.ttft_std for r in standard]
        ax2.errorbar(x, y, yerr=yerr, fmt='o-', color=colors['standard'],
                    linewidth=2, markersize=8, capsize=4, label='Standard')
    
    for cache in cache_sizes:
        stream_cache = sorted([r for r in streaming if r.max_cache_len == cache],
                             key=lambda x: x.target_tokens)
        if stream_cache:
            x = [r.target_tokens for r in stream_cache]
            y = [r.ttft_mean for r in stream_cache]
            yerr = [r.ttft_std for r in stream_cache]
            ax2.errorbar(x, y, yerr=yerr, fmt='o--', color=colors.get(cache, '#888'),
                        linewidth=2, markersize=8, capsize=4, label=f'Stream({cache})')
    
    ax2.set_xlabel('Target Tokens')
    ax2.set_ylabel('TTFT (ms)')
    ax2.set_title('Time to First Token')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: TPOT
    ax3 = fig.add_subplot(gs[0, 2])
    if standard:
        x = [r.target_tokens for r in standard]
        y = [r.tpot_mean for r in standard]
        yerr = [r.tpot_std for r in standard]
        ax3.errorbar(x, y, yerr=yerr, fmt='o-', color=colors['standard'],
                    linewidth=2, markersize=8, capsize=4, label='Standard')
    
    for cache in cache_sizes:
        stream_cache = sorted([r for r in streaming if r.max_cache_len == cache],
                             key=lambda x: x.target_tokens)
        if stream_cache:
            x = [r.target_tokens for r in stream_cache]
            y = [r.tpot_mean for r in stream_cache]
            yerr = [r.tpot_std for r in stream_cache]
            ax3.errorbar(x, y, yerr=yerr, fmt='o--', color=colors.get(cache, '#888'),
                        linewidth=2, markersize=8, capsize=4, label=f'Stream({cache})')
    
    ax3.set_xlabel('Target Tokens')
    ax3.set_ylabel('TPOT (ms/token)')
    ax3.set_title('Time per Output Token')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: PPL (Perplexity)
    ax4 = fig.add_subplot(gs[1, 0])
    if standard:
        x = [r.target_tokens for r in standard]
        y = [r.ppl_mean for r in standard]
        yerr = [r.ppl_std for r in standard]
        ax4.errorbar(x, y, yerr=yerr, fmt='o-', color=colors['standard'],
                    linewidth=2, markersize=8, capsize=4, label='Standard')
    
    for cache in cache_sizes:
        stream_cache = sorted([r for r in streaming if r.max_cache_len == cache],
                             key=lambda x: x.target_tokens)
        if stream_cache:
            x = [r.target_tokens for r in stream_cache]
            y = [r.ppl_mean for r in stream_cache]
            yerr = [r.ppl_std for r in stream_cache]
            ax4.errorbar(x, y, yerr=yerr, fmt='o--', color=colors.get(cache, '#888'),
                        linewidth=2, markersize=8, capsize=4, label=f'Stream({cache})')
    
    ax4.set_xlabel('Target Tokens')
    ax4.set_ylabel('Perplexity')
    ax4.set_title('Generation Quality (PPL, lower=better)')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Acceptance Rate
    ax5 = fig.add_subplot(gs[1, 1])
    if standard:
        x = [r.target_tokens for r in standard]
        y = [r.acceptance_rate * 100 for r in standard]
        ax5.plot(x, y, 'o-', color=colors['standard'], linewidth=2, markersize=8, label='Standard')
    
    for cache in cache_sizes:
        stream_cache = sorted([r for r in streaming if r.max_cache_len == cache],
                             key=lambda x: x.target_tokens)
        if stream_cache:
            x = [r.target_tokens for r in stream_cache]
            y = [r.acceptance_rate * 100 for r in stream_cache]
            ax5.plot(x, y, 'o--', color=colors.get(cache, '#888'), linewidth=2, markersize=8,
                    label=f'Stream({cache})')
    
    ax5.set_xlabel('Target Tokens')
    ax5.set_ylabel('Acceptance Rate (%)')
    ax5.set_title('Draft Token Acceptance Rate')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Memory Growth
    ax6 = fig.add_subplot(gs[1, 2])
    if standard:
        x = [r.target_tokens for r in standard]
        y = [r.memory_growth_mb for r in standard]
        ax6.plot(x, y, 'o-', color=colors['standard'], linewidth=2, markersize=8, label='Standard')
    
    for cache in cache_sizes:
        stream_cache = sorted([r for r in streaming if r.max_cache_len == cache],
                             key=lambda x: x.target_tokens)
        if stream_cache:
            x = [r.target_tokens for r in stream_cache]
            y = [r.memory_growth_mb for r in stream_cache]
            ax6.plot(x, y, 'o--', color=colors.get(cache, '#888'), linewidth=2, markersize=8,
                    label=f'Stream({cache})')
    
    ax6.set_xlabel('Target Tokens')
    ax6.set_ylabel('Memory Growth (MB)')
    ax6.set_title('Memory Usage')
    ax6.legend(fontsize=8)
    ax6.grid(True, alpha=0.3)
    
    # Plot 7: Compression Events (StreamingLLM only)
    ax7 = fig.add_subplot(gs[2, 0])
    for cache in cache_sizes:
        stream_cache = sorted([r for r in streaming if r.max_cache_len == cache],
                             key=lambda x: x.target_tokens)
        if stream_cache:
            x = [r.target_tokens for r in stream_cache]
            y = [r.compression_count for r in stream_cache]
            ax7.plot(x, y, 'o--', color=colors.get(cache, '#888'), linewidth=2, markersize=8,
                    label=f'Stream({cache})')
    
    ax7.set_xlabel('Target Tokens')
    ax7.set_ylabel('Compression Count')
    ax7.set_title('KV Cache Compressions')
    ax7.legend(fontsize=8)
    ax7.grid(True, alpha=0.3)
    
    # Plot 8: Summary Table
    ax8 = fig.add_subplot(gs[2, 1:])
    ax8.axis('off')
    
    headers = ['Config', 'Tokens', 'Throughput', 'TTFT (ms)', 'TPOT (ms)', 
               'PPL', 'Accept%', 'Mem (MB)', 'Compress']
    
    table_data = []
    for r in results:
        config = r.method if r.max_cache_len == 0 else f"stream({r.max_cache_len})"
        table_data.append([
            config,
            str(r.actual_tokens),
            f'{r.throughput_mean:.1f}±{r.throughput_std:.1f}',
            f'{r.ttft_mean:.1f}±{r.ttft_std:.1f}',
            f'{r.tpot_mean:.2f}±{r.tpot_std:.2f}',
            f'{r.ppl_mean:.1f}±{r.ppl_std:.1f}',
            f'{r.acceptance_rate*100:.1f}',
            f'{r.memory_growth_mb:.0f}',
            str(r.compression_count) if r.compression_count > 0 else '-'
        ])
    
    table = ax8.table(cellText=table_data, colLabels=headers, loc='center',
                     cellLoc='center', colColours=['#4ECDC4']*len(headers))
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.0, 1.4)
    
    for i in range(len(headers)):
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    plt.suptitle('Comprehensive Benchmark: Standard vs StreamingLLM', fontsize=14, fontweight='bold')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✅ Plot saved to: {save_path}")
    plt.close()


def print_results(results: List[ComprehensiveMetrics]):
    """Print comprehensive results table."""
    print("\n" + "=" * 170)
    print("COMPREHENSIVE BENCHMARK RESULTS")
    print("=" * 170)
    
    print(f"\n{'Config':<15} {'Tokens':<8} {'Throughput':<18} {'TTFT (ms)':<15} {'TPOT (ms)':<15} "
          f"{'PPL':<15} {'Accept%':<10} {'T/Round':<10} {'Mem MB':<10} {'Compress':<10}")
    print("-" * 170)
    
    for r in results:
        config = r.method if r.max_cache_len == 0 else f"stream({r.max_cache_len})"
        ppl_str = f"{r.ppl_mean:.1f}±{r.ppl_std:.1f}" if not math.isnan(r.ppl_mean) else "N/A"
        print(f"{config:<15} {r.actual_tokens:<8} "
              f"{r.throughput_mean:.1f}±{r.throughput_std:.1f}{'':<8} "
              f"{r.ttft_mean:.1f}±{r.ttft_std:.1f}{'':<5} "
              f"{r.tpot_mean:.2f}±{r.tpot_std:.2f}{'':<5} "
              f"{ppl_str:<15} "
              f"{r.acceptance_rate*100:.1f}%{'':<5} "
              f"{r.tokens_per_round:.2f}{'':<6} "
              f"{r.memory_growth_mb:.0f}{'':<6} "
              f"{r.compression_count if r.compression_count > 0 else '-':<10}")
    
    print("=" * 170)
    print("\nNote: Accept% is draft token acceptance (capped at 100%).")
    print("      T/Round is avg tokens generated per speculative round (can be > K due to bonus tokens).")


def main():
    parser = argparse.ArgumentParser(description="Comprehensive Speculative Decoding Benchmark")
    parser.add_argument("--target-model", type=str, default="/mnt/disk1/models/pythia-2.8b")
    parser.add_argument("--draft-model", type=str, default="/mnt/disk1/models/pythia-70m")
    parser.add_argument("--max-new-tokens", type=int, nargs="+", default=[500, 1000, 2000])
    parser.add_argument("--max-cache-lens", type=int, nargs="+", default=[256, 512, 1024])
    parser.add_argument("--k-value", type=int, default=5)
    parser.add_argument("--num-samples", type=int, default=3)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output-json", type=str, default="benchmark_comprehensive_results.json")
    parser.add_argument("--output-plot", type=str, default="benchmark_comprehensive.png")
    args = parser.parse_args()
    
    print("=" * 80)
    print("📊 COMPREHENSIVE SPECULATIVE DECODING BENCHMARK")
    print("=" * 80)
    print(f"   Target Model:  {args.target_model}")
    print(f"   Draft Model:   {args.draft_model}")
    print(f"   Token Lengths: {args.max_new_tokens}")
    print(f"   Cache Sizes:   {args.max_cache_lens}")
    print(f"   K Value:       {args.k_value}")
    print(f"   Num Samples:   {args.num_samples}")
    print("=" * 80)
    print("\n📏 Metrics: Throughput, TTFT, TPOT, PPL, Acceptance Rate, Memory, Compression")
    
    # Load models
    print("\n🔄 Loading models...")
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model, torch_dtype=torch.float16
    ).to(args.device)
    
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model, torch_dtype=torch.float16
    ).to(args.device)
    
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    target_model.eval()
    draft_model.eval()
    
    prompts = get_test_prompts()[:args.num_samples]
    results = []
    
    # Standard speculative decoding
    print("\n🏃 Running Standard Speculative Decoding...")
    for max_tokens in args.max_new_tokens:
        print(f"  Testing max_new_tokens={max_tokens}...")
        result = measure_standard(
            target_model, draft_model, tokenizer,
            prompts, max_tokens, args.k_value, args.device
        )
        results.append(result)
        print(f"    Throughput: {result.throughput_mean:.1f}±{result.throughput_std:.1f} t/s, "
              f"TTFT: {result.ttft_mean:.1f}ms, TPOT: {result.tpot_mean:.2f}ms, "
              f"PPL: {result.ppl_mean:.1f}")
    
    # StreamingLLM speculative decoding
    print("\n🏃 Running StreamingLLM Speculative Decoding...")
    for max_tokens in args.max_new_tokens:
        for cache_len in args.max_cache_lens:
            print(f"  Testing max_new_tokens={max_tokens}, max_cache_len={cache_len}...")
            result = measure_streaming(
                target_model, draft_model, tokenizer,
                prompts, max_tokens, cache_len, args.k_value, args.device
            )
            results.append(result)
            print(f"    Throughput: {result.throughput_mean:.1f}±{result.throughput_std:.1f} t/s, "
                  f"TTFT: {result.ttft_mean:.1f}ms, TPOT: {result.tpot_mean:.2f}ms, "
                  f"PPL: {result.ppl_mean:.1f}, Compress: {result.compression_count}")
    
    # Print results
    print_results(results)
    
    # Save results
    with open(args.output_json, 'w') as f:
        json.dump([asdict(r) for r in results], f, indent=2)
    print(f"\n✅ Results saved to: {args.output_json}")
    
    # Plot
    plot_comprehensive_results(results, args.output_plot)


if __name__ == "__main__":
    main()
