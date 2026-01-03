#!/usr/bin/env python3
"""
Comprehensive Benchmark Script for Speculative Decoding Methods

This script benchmarks multiple speculative decoding configurations measuring:
1. TTFT (Time To First Token) - Prefill latency
2. TPOT (Time Per Output Token) - Per-token generation latency
3. Throughput (tokens/second)
4. FLOPs estimation (based on model architecture)
5. Token acceptance rates comparison (Tree vs Linear)

Tested configurations (from Tree_Speculative_Decoding_实验报告.md):
1. Tree V2 (D=8, B=3, t=0.03) - Best performer
2. HuggingFace Assisted Generation
3. Linear K=6
4. Streaming K=6 cache=1024
5. Linear K=7, K=8, K=5
6. Baseline (Autoregressive)
7. Streaming K=6 cache=512

Uses pg19.parquet dataset for realistic text prompts.
"""

import os
import sys
import json
import time
import gc
import torch
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
from contextlib import contextmanager

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from transformers import AutoModelForCausalLM, AutoTokenizer


@dataclass
class BenchmarkMetrics:
    """Comprehensive benchmark metrics."""
    method: str
    config: Dict[str, Any]
    
    # Timing metrics
    ttft_ms: float = 0.0           # Time To First Token (ms)
    tpot_ms: float = 0.0           # Time Per Output Token (ms)
    total_latency_ms: float = 0.0  # Total generation time (ms)
    throughput_tps: float = 0.0    # Throughput (tokens/second)
    
    # Token metrics
    total_tokens_generated: int = 0
    prompt_tokens: int = 0
    
    # Acceptance metrics (for spec decode)
    total_draft_tokens: int = 0
    total_accepted_tokens: int = 0
    acceptance_rate: float = 0.0
    tokens_per_round: float = 0.0
    total_rounds: int = 0
    
    # Tree-specific metrics
    avg_path_length: float = 0.0
    max_path_length: int = 0
    
    # FLOPs estimation
    prefill_flops: float = 0.0
    decode_flops: float = 0.0
    total_flops: float = 0.0
    flops_per_token: float = 0.0
    
    # Memory
    peak_memory_mb: float = 0.0
    
    # Speedup
    speedup: float = 1.0


@dataclass
class ModelFLOPsConfig:
    """Model configuration for FLOPs estimation."""
    hidden_size: int
    num_layers: int
    num_attention_heads: int
    intermediate_size: int
    vocab_size: int
    
    def estimate_forward_flops(self, seq_len: int, batch_size: int = 1) -> float:
        """
        Estimate FLOPs for a single forward pass.
        
        Based on: https://arxiv.org/abs/2001.08361 (Scaling Laws for Neural Language Models)
        Approximate FLOPs per token ≈ 6 * N (where N is model parameters)
        
        More detailed estimation:
        - Attention: 4 * seq_len * d_model^2 + 2 * seq_len^2 * d_model (per layer)
        - FFN: 16 * seq_len * d_model^2 (per layer, assuming 4x intermediate)
        - Total per layer: 20 * seq_len * d_model^2 + 2 * seq_len^2 * d_model
        """
        d = self.hidden_size
        L = self.num_layers
        s = seq_len
        
        # Per layer FLOPs (batch_size = 1)
        # QKV projections: 3 * 2 * s * d^2
        # Attention scores: 2 * s^2 * d  
        # Attention output: 2 * s * d^2
        # FFN: 2 * 4 * 2 * s * d^2 = 16 * s * d^2
        attention_flops = 6 * s * d * d + 2 * s * s * d
        ffn_flops = 16 * s * d * d
        per_layer_flops = attention_flops + ffn_flops
        
        # Total for all layers
        total_flops = L * per_layer_flops * batch_size
        
        # Output projection
        output_flops = 2 * s * d * self.vocab_size * batch_size
        
        return total_flops + output_flops


class FLOPsEstimator:
    """Estimate FLOPs for different model operations."""
    
    def __init__(self, target_config: ModelFLOPsConfig, draft_config: Optional[ModelFLOPsConfig] = None):
        self.target_config = target_config
        self.draft_config = draft_config
    
    def estimate_prefill_flops(self, prompt_len: int) -> float:
        """Estimate FLOPs for prefill phase."""
        return self.target_config.estimate_forward_flops(prompt_len)
    
    def estimate_ar_decode_flops(self, num_tokens: int, prompt_len: int) -> float:
        """Estimate FLOPs for autoregressive decoding."""
        total = 0.0
        for i in range(num_tokens):
            # Each step processes 1 new token with full KV cache
            total += self.target_config.estimate_forward_flops(1)
        return total
    
    def estimate_spec_decode_flops(
        self, 
        num_tokens: int, 
        k: int, 
        num_rounds: int,
        prompt_len: int
    ) -> float:
        """Estimate FLOPs for speculative decoding."""
        if self.draft_config is None:
            return self.estimate_ar_decode_flops(num_tokens, prompt_len)
        
        total = 0.0
        for _ in range(num_rounds):
            # Draft model: K forward passes (each processes 1 token)
            for j in range(k):
                total += self.draft_config.estimate_forward_flops(1)
            
            # Target model: 1 forward pass for K tokens
            total += self.target_config.estimate_forward_flops(k)
        
        return total
    
    def estimate_tree_decode_flops(
        self,
        num_tokens: int,
        tree_depth: int,
        branch_factor: int,
        num_rounds: int,
        prompt_len: int
    ) -> float:
        """Estimate FLOPs for tree-based speculative decoding."""
        if self.draft_config is None:
            return self.estimate_ar_decode_flops(num_tokens, prompt_len)
        
        # Estimate number of tree nodes
        max_nodes = sum(branch_factor ** i for i in range(tree_depth + 1))
        
        total = 0.0
        for _ in range(num_rounds):
            # Draft model: multiple forward passes to build tree
            # Approximate: tree_depth levels, each level has more nodes
            for d in range(tree_depth):
                nodes_at_level = min(branch_factor ** d, max_nodes)
                for _ in range(nodes_at_level):
                    total += self.draft_config.estimate_forward_flops(1)
            
            # Target model: 1 forward pass for all tree nodes
            total += self.target_config.estimate_forward_flops(max_nodes)
        
        return total


class ComprehensiveBenchmark:
    """Comprehensive benchmark runner for speculative decoding methods."""
    
    def __init__(
        self,
        target_model_path: str,
        draft_model_path: str,
        dataset_path: str,
        device: str = "cuda",
        num_samples: int = 5,
        max_new_tokens: int = 500,
        warmup_runs: int = 2,
        use_dataset: bool = False  # 默认使用短 prompt 以获得可复现结果
    ):
        self.target_model_path = target_model_path
        self.draft_model_path = draft_model_path
        self.dataset_path = dataset_path
        self.device = device
        self.num_samples = num_samples
        self.max_new_tokens = max_new_tokens
        self.warmup_runs = warmup_runs
        self.use_dataset = use_dataset
        
        # Load models
        print("Loading models...")
        self.tokenizer = AutoTokenizer.from_pretrained(target_model_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load models on single GPU to avoid device mismatch
        self.target_model = AutoModelForCausalLM.from_pretrained(
            target_model_path,
            torch_dtype=torch.float16,
        ).to(device)
        self.target_model.eval()
        
        self.draft_model = AutoModelForCausalLM.from_pretrained(
            draft_model_path,
            torch_dtype=torch.float16,
        ).to(device)
        self.draft_model.eval()
        
        # Setup FLOPs estimator
        target_config = ModelFLOPsConfig(
            hidden_size=self.target_model.config.hidden_size,
            num_layers=self.target_model.config.num_hidden_layers,
            num_attention_heads=self.target_model.config.num_attention_heads,
            intermediate_size=self.target_model.config.intermediate_size,
            vocab_size=self.target_model.config.vocab_size
        )
        draft_config = ModelFLOPsConfig(
            hidden_size=self.draft_model.config.hidden_size,
            num_layers=self.draft_model.config.num_hidden_layers,
            num_attention_heads=self.draft_model.config.num_attention_heads,
            intermediate_size=self.draft_model.config.intermediate_size,
            vocab_size=self.draft_model.config.vocab_size
        )
        self.flops_estimator = FLOPsEstimator(target_config, draft_config)
        
        # Load dataset
        print(f"Loading dataset from {dataset_path}...")
        self.dataset = pd.read_parquet(dataset_path)
        self.prompts = self._prepare_prompts(use_dataset=self.use_dataset)
        
        if self.use_dataset:
            print(f"Using {len(self.prompts)} prompts from dataset")
        else:
            print(f"Using standard short prompt (reproducible mode)")
        
        # Results storage
        self.results: List[BenchmarkMetrics] = []
        self.baseline_throughput: Optional[float] = None
    
    def _prepare_prompts(self, use_dataset: bool = True) -> List[str]:
        """Prepare prompts from dataset or use standard short prompts.
        
        Args:
            use_dataset: If True, use pg19 dataset. If False, use short standard prompts.
        """
        if not use_dataset:
            # 使用与原始测试相同的短 prompt，以获得可复现的结果
            standard_prompt = """Write a detailed technical explanation about the development of large language models. 
Cover the history, architecture innovations, training techniques, and future directions.
Begin your explanation:

Large language models have become"""
            return [standard_prompt] * self.num_samples
        
        # 使用数据集
        prompts = []
        for idx, row in self.dataset.iterrows():
            if idx >= self.num_samples:
                break
            text = row['text']
            # Use first 500-1000 characters as prompt
            prompt = text[:800] if len(text) > 800 else text
            prompts.append(prompt)
        return prompts
    
    @contextmanager
    def _memory_tracking(self):
        """Context manager for GPU memory tracking."""
        torch.cuda.reset_peak_memory_stats()
        try:
            yield
        finally:
            pass
    
    def _cleanup(self):
        """Clean up GPU memory."""
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    def _disable_eos(self):
        """Disable EOS token to force long generation."""
        original_eos = self.tokenizer.eos_token_id
        self.tokenizer.eos_token_id = 999999
        return original_eos
    
    def _restore_eos(self, original_eos):
        """Restore original EOS token."""
        self.tokenizer.eos_token_id = original_eos
    
    @torch.inference_mode()
    def _measure_prefill_ttft(self, prompt: str, model) -> float:
        """
        Measure Time To First Token (TTFT) - the prefill latency.
        
        Args:
            prompt: Input prompt string
            model: Model to measure prefill for
            
        Returns:
            TTFT in milliseconds
        """
        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).input_ids.to(self.device)
        
        torch.cuda.synchronize()
        start_prefill = time.perf_counter()
        
        # Run prefill (first forward pass)
        with torch.no_grad():
            _ = model(input_ids=input_ids, use_cache=True, return_dict=True)
        
        torch.cuda.synchronize()
        ttft_ms = (time.perf_counter() - start_prefill) * 1000
        
        return ttft_ms
    
    @torch.inference_mode()
    def benchmark_baseline(self) -> BenchmarkMetrics:
        """Benchmark pure autoregressive generation."""
        print("\n" + "="*60)
        print("Benchmarking: Baseline (Autoregressive)")
        print("="*60)
        
        metrics = BenchmarkMetrics(method="Baseline (AR)", config={})
        
        original_eos = self._disable_eos()
        
        all_ttft = []
        all_tpot = []
        all_throughput = []
        all_tokens = []
        
        for run_idx in range(self.warmup_runs + len(self.prompts)):
            prompt_idx = run_idx % len(self.prompts)
            prompt = self.prompts[prompt_idx]
            is_warmup = run_idx < self.warmup_runs
            
            self._cleanup()
            
            input_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).input_ids.to(self.device)
            
            prompt_len = input_ids.shape[1]
            
            with self._memory_tracking():
                # Measure TTFT (prefill)
                torch.cuda.synchronize()
                start_prefill = time.perf_counter()
                
                outputs = self.target_model(
                    input_ids=input_ids,
                    use_cache=True,
                    return_dict=True
                )
                
                torch.cuda.synchronize()
                ttft = (time.perf_counter() - start_prefill) * 1000  # ms
                
                # Generate tokens
                past_kv = outputs.past_key_values
                next_token_logits = outputs.logits[:, -1, :]
                generated_ids = [input_ids]
                
                torch.cuda.synchronize()
                start_decode = time.perf_counter()
                
                for _ in range(self.max_new_tokens):
                    next_token = next_token_logits.argmax(dim=-1, keepdim=True)
                    generated_ids.append(next_token)
                    
                    outputs = self.target_model(
                        input_ids=next_token,
                        past_key_values=past_kv,
                        use_cache=True,
                        return_dict=True
                    )
                    past_kv = outputs.past_key_values
                    next_token_logits = outputs.logits[:, -1, :]
                
                torch.cuda.synchronize()
                decode_time = (time.perf_counter() - start_decode) * 1000  # ms
                
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024  # MB
            
            total_time = ttft + decode_time
            num_tokens = self.max_new_tokens
            tpot = decode_time / num_tokens
            throughput = num_tokens / (total_time / 1000)
            
            if not is_warmup:
                all_ttft.append(ttft)
                all_tpot.append(tpot)
                all_throughput.append(throughput)
                all_tokens.append(num_tokens)
                print(f"  Sample {run_idx - self.warmup_runs + 1}: {throughput:.1f} t/s, TTFT={ttft:.1f}ms, TPOT={tpot:.2f}ms")
        
        self._restore_eos(original_eos)
        
        # Average metrics
        metrics.ttft_ms = sum(all_ttft) / len(all_ttft)
        metrics.tpot_ms = sum(all_tpot) / len(all_tpot)
        metrics.throughput_tps = sum(all_throughput) / len(all_throughput)
        metrics.total_tokens_generated = int(sum(all_tokens) / len(all_tokens))
        metrics.total_latency_ms = metrics.ttft_ms + metrics.tpot_ms * metrics.total_tokens_generated
        metrics.peak_memory_mb = peak_memory
        
        # FLOPs estimation
        avg_prompt_len = 512  # approximate
        metrics.prefill_flops = self.flops_estimator.estimate_prefill_flops(avg_prompt_len)
        metrics.decode_flops = self.flops_estimator.estimate_ar_decode_flops(
            metrics.total_tokens_generated, avg_prompt_len
        )
        metrics.total_flops = metrics.prefill_flops + metrics.decode_flops
        metrics.flops_per_token = metrics.decode_flops / metrics.total_tokens_generated
        
        # Set baseline throughput
        self.baseline_throughput = metrics.throughput_tps
        metrics.speedup = 1.0
        
        print(f"\nBaseline Results:")
        print(f"  Throughput: {metrics.throughput_tps:.1f} t/s")
        print(f"  TTFT: {metrics.ttft_ms:.1f} ms")
        print(f"  TPOT: {metrics.tpot_ms:.2f} ms")
        print(f"  Total FLOPs: {metrics.total_flops:.2e}")
        
        self.results.append(metrics)
        return metrics
    
    @torch.inference_mode()
    def benchmark_hf_assisted(self) -> BenchmarkMetrics:
        """Benchmark HuggingFace's assisted generation."""
        print("\n" + "="*60)
        print("Benchmarking: HuggingFace Assisted Generation")
        print("="*60)
        
        metrics = BenchmarkMetrics(method="HF Assisted", config={"num_assistant_tokens": 5})
        
        original_eos = self._disable_eos()
        
        all_ttft = []
        all_tpot = []
        all_throughput = []
        all_tokens = []
        
        for run_idx in range(self.warmup_runs + len(self.prompts)):
            prompt_idx = run_idx % len(self.prompts)
            prompt = self.prompts[prompt_idx]
            is_warmup = run_idx < self.warmup_runs
            
            self._cleanup()
            
            # Measure TTFT (prefill time for target model)
            ttft = self._measure_prefill_ttft(prompt, self.target_model)
            
            input_ids = self.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=1024
            ).input_ids.to(self.device)
            
            with self._memory_tracking():
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                # Use HF's native assisted generation
                output_ids = self.target_model.generate(
                    input_ids,
                    assistant_model=self.draft_model,
                    max_new_tokens=self.max_new_tokens,
                    do_sample=False,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=999999,  # Disable EOS
                )
                
                torch.cuda.synchronize()
                total_time = (time.perf_counter() - start_time) * 1000  # ms
                
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            num_tokens = output_ids.shape[1] - input_ids.shape[1]
            throughput = num_tokens / (total_time / 1000)
            # TPOT = average time per token during decode (exclude prefill)
            tpot = (total_time - ttft) / num_tokens if num_tokens > 0 else 0
            
            if not is_warmup:
                all_throughput.append(throughput)
                all_tokens.append(num_tokens)
                all_ttft.append(ttft)
                all_tpot.append(tpot)
                print(f"  Sample {run_idx - self.warmup_runs + 1}: {throughput:.1f} t/s, TTFT={ttft:.1f}ms, TPOT={tpot:.2f}ms, tokens={num_tokens}")
        
        self._restore_eos(original_eos)
        
        metrics.throughput_tps = sum(all_throughput) / len(all_throughput)
        metrics.total_tokens_generated = int(sum(all_tokens) / len(all_tokens))
        metrics.ttft_ms = sum(all_ttft) / len(all_ttft)
        metrics.tpot_ms = sum(all_tpot) / len(all_tpot)
        metrics.peak_memory_mb = peak_memory
        
        if self.baseline_throughput:
            metrics.speedup = metrics.throughput_tps / self.baseline_throughput
        
        print(f"\nHF Assisted Results:")
        print(f"  Throughput: {metrics.throughput_tps:.1f} t/s")
        print(f"  TTFT: {metrics.ttft_ms:.1f} ms")
        print(f"  TPOT: {metrics.tpot_ms:.2f} ms")
        print(f"  Speedup: {metrics.speedup:.2f}x")
        
        self.results.append(metrics)
        return metrics
    
    @torch.inference_mode()
    def benchmark_linear_spec_decode(self, K: int = 6) -> BenchmarkMetrics:
        """Benchmark linear speculative decoding."""
        print("\n" + "="*60)
        print(f"Benchmarking: Linear Speculative Decoding (K={K})")
        print("="*60)
        
        from spec_decode.core.speculative_generator import SpeculativeGenerator
        
        metrics = BenchmarkMetrics(method=f"Linear K={K}", config={"K": K})
        
        original_eos = self._disable_eos()
        
        generator = SpeculativeGenerator(
            target_model=self.target_model,
            draft_model=self.draft_model,
            tokenizer=self.tokenizer,
            K=K,
            device=self.device,
            use_compile=False
        )
        
        all_throughput = []
        all_acceptance = []
        all_tokens_per_round = []
        all_rounds = []
        all_ttft = []
        all_tpot = []
        
        for run_idx in range(self.warmup_runs + len(self.prompts)):
            prompt_idx = run_idx % len(self.prompts)
            prompt = self.prompts[prompt_idx]
            is_warmup = run_idx < self.warmup_runs
            
            self._cleanup()
            
            # Measure TTFT (prefill time for target model)
            ttft = self._measure_prefill_ttft(prompt, self.target_model)
            
            with self._memory_tracking():
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                _ = generator.generate(prompt, max_new_tokens=self.max_new_tokens)
                
                torch.cuda.synchronize()
                total_time = (time.perf_counter() - start_time) * 1000
                
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            stats = generator.get_stats()
            num_tokens = stats['total_tokens']
            throughput = num_tokens / (total_time / 1000)
            # TPOT = average time per token during decode (exclude prefill)
            tpot = (total_time - ttft) / num_tokens if num_tokens > 0 else 0
            
            if not is_warmup:
                all_throughput.append(throughput)
                all_acceptance.append(stats.get('acceptance_rate', 0))
                all_tokens_per_round.append(stats.get('tokens_per_round', 0))
                all_rounds.append(stats.get('total_rounds', 0))
                all_ttft.append(ttft)
                all_tpot.append(tpot)
                print(f"  Sample {run_idx - self.warmup_runs + 1}: {throughput:.1f} t/s, TTFT={ttft:.1f}ms, TPOT={tpot:.2f}ms, accept={stats.get('acceptance_rate', 0):.2%}")
        
        self._restore_eos(original_eos)
        
        metrics.throughput_tps = sum(all_throughput) / len(all_throughput)
        metrics.acceptance_rate = sum(all_acceptance) / len(all_acceptance)
        metrics.tokens_per_round = sum(all_tokens_per_round) / len(all_tokens_per_round)
        metrics.total_rounds = int(sum(all_rounds) / len(all_rounds))
        metrics.total_tokens_generated = self.max_new_tokens
        metrics.peak_memory_mb = peak_memory
        metrics.ttft_ms = sum(all_ttft) / len(all_ttft)
        metrics.tpot_ms = sum(all_tpot) / len(all_tpot)
        
        # FLOPs estimation
        avg_prompt_len = 512
        metrics.prefill_flops = self.flops_estimator.estimate_prefill_flops(avg_prompt_len)
        metrics.decode_flops = self.flops_estimator.estimate_spec_decode_flops(
            metrics.total_tokens_generated, K, metrics.total_rounds, avg_prompt_len
        )
        metrics.total_flops = metrics.prefill_flops + metrics.decode_flops
        metrics.flops_per_token = metrics.decode_flops / metrics.total_tokens_generated
        
        if self.baseline_throughput:
            metrics.speedup = metrics.throughput_tps / self.baseline_throughput
        
        print(f"\nLinear K={K} Results:")
        print(f"  Throughput: {metrics.throughput_tps:.1f} t/s")
        print(f"  TTFT: {metrics.ttft_ms:.1f} ms")
        print(f"  TPOT: {metrics.tpot_ms:.2f} ms")
        print(f"  Acceptance Rate: {metrics.acceptance_rate:.2%}")
        print(f"  Tokens/Round: {metrics.tokens_per_round:.1f}")
        print(f"  Speedup: {metrics.speedup:.2f}x")
        
        self.results.append(metrics)
        return metrics
    
    @torch.inference_mode()
    def benchmark_streaming_spec_decode(self, K: int = 6, max_cache_len: int = 1024) -> BenchmarkMetrics:
        """Benchmark streaming speculative decoding."""
        print("\n" + "="*60)
        print(f"Benchmarking: Streaming Speculative Decoding (K={K}, cache={max_cache_len})")
        print("="*60)
        
        from spec_decode.core.streaming_speculative_generator import StreamingSpeculativeGenerator
        
        metrics = BenchmarkMetrics(
            method=f"Streaming K={K} cache={max_cache_len}",
            config={"K": K, "max_cache_len": max_cache_len}
        )
        
        original_eos = self._disable_eos()
        
        generator = StreamingSpeculativeGenerator(
            target_model=self.target_model,
            draft_model=self.draft_model,
            tokenizer=self.tokenizer,
            K=K,
            max_cache_len=max_cache_len,
            device=self.device,
            use_compile=False
        )
        
        all_throughput = []
        all_acceptance = []
        all_ttft = []
        all_tpot = []
        
        for run_idx in range(self.warmup_runs + len(self.prompts)):
            prompt_idx = run_idx % len(self.prompts)
            prompt = self.prompts[prompt_idx]
            is_warmup = run_idx < self.warmup_runs
            
            self._cleanup()
            
            # Measure TTFT (prefill time for target model)
            ttft = self._measure_prefill_ttft(prompt, self.target_model)
            
            with self._memory_tracking():
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                _ = generator.generate(prompt, max_new_tokens=self.max_new_tokens)
                
                torch.cuda.synchronize()
                total_time = (time.perf_counter() - start_time) * 1000
                
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            stats = generator.get_stats()
            num_tokens = stats['total_tokens']
            throughput = num_tokens / (total_time / 1000)
            # TPOT = average time per token during decode (exclude prefill)
            tpot = (total_time - ttft) / num_tokens if num_tokens > 0 else 0
            
            if not is_warmup:
                all_throughput.append(throughput)
                all_acceptance.append(stats.get('acceptance_rate', 0))
                all_ttft.append(ttft)
                all_tpot.append(tpot)
                print(f"  Sample {run_idx - self.warmup_runs + 1}: {throughput:.1f} t/s, TTFT={ttft:.1f}ms, TPOT={tpot:.2f}ms")
        
        self._restore_eos(original_eos)
        
        metrics.throughput_tps = sum(all_throughput) / len(all_throughput)
        metrics.acceptance_rate = sum(all_acceptance) / len(all_acceptance)
        metrics.total_tokens_generated = self.max_new_tokens
        metrics.peak_memory_mb = peak_memory
        metrics.ttft_ms = sum(all_ttft) / len(all_ttft)
        metrics.tpot_ms = sum(all_tpot) / len(all_tpot)
        
        if self.baseline_throughput:
            metrics.speedup = metrics.throughput_tps / self.baseline_throughput
        
        print(f"\nStreaming K={K} cache={max_cache_len} Results:")
        print(f"  Throughput: {metrics.throughput_tps:.1f} t/s")
        print(f"  TTFT: {metrics.ttft_ms:.1f} ms")
        print(f"  TPOT: {metrics.tpot_ms:.2f} ms")
        print(f"  Speedup: {metrics.speedup:.2f}x")
        
        self.results.append(metrics)
        return metrics
    
    @torch.inference_mode()
    def benchmark_tree_v2(
        self,
        tree_depth: int = 8,
        branch_factor: int = 3,
        probability_threshold: float = 0.03
    ) -> BenchmarkMetrics:
        """Benchmark Tree V2 speculative decoding."""
        print("\n" + "="*60)
        print(f"Benchmarking: Tree V2 (D={tree_depth}, B={branch_factor}, t={probability_threshold})")
        print("="*60)
        
        from spec_decode.core.tree_speculative_generator import TreeSpeculativeGeneratorV2
        
        metrics = BenchmarkMetrics(
            method=f"Tree V2 (D={tree_depth}, B={branch_factor}, t={probability_threshold})",
            config={
                "tree_depth": tree_depth,
                "branch_factor": branch_factor,
                "probability_threshold": probability_threshold
            }
        )
        
        original_eos = self._disable_eos()
        
        generator = TreeSpeculativeGeneratorV2(
            target_model=self.target_model,
            draft_model=self.draft_model,
            tokenizer=self.tokenizer,
            tree_depth=tree_depth,
            branch_factor=branch_factor,
            probability_threshold=probability_threshold,
            device=self.device,
            use_compile=False
        )
        
        all_throughput = []
        all_acceptance = []
        all_path_lengths = []
        all_rounds = []
        all_ttft = []
        all_tpot = []
        
        for run_idx in range(self.warmup_runs + len(self.prompts)):
            prompt_idx = run_idx % len(self.prompts)
            prompt = self.prompts[prompt_idx]
            is_warmup = run_idx < self.warmup_runs
            
            self._cleanup()
            
            # Measure TTFT (prefill time for target model)
            ttft = self._measure_prefill_ttft(prompt, self.target_model)
            
            with self._memory_tracking():
                torch.cuda.synchronize()
                start_time = time.perf_counter()
                
                _ = generator.generate(prompt, max_new_tokens=self.max_new_tokens)
                
                torch.cuda.synchronize()
                total_time = (time.perf_counter() - start_time) * 1000
                
                peak_memory = torch.cuda.max_memory_allocated() / 1024 / 1024
            
            stats = generator.get_stats()
            num_tokens = stats['total_tokens']
            throughput = num_tokens / (total_time / 1000)
            # TPOT = average time per token during decode (exclude prefill)
            tpot = (total_time - ttft) / num_tokens if num_tokens > 0 else 0
            
            if not is_warmup:
                all_throughput.append(throughput)
                all_acceptance.append(stats.get('acceptance_rate', 0))
                all_path_lengths.append(stats.get('avg_accepted_path_length', 0))
                all_rounds.append(stats.get('total_rounds', 0))
                all_ttft.append(ttft)
                all_tpot.append(tpot)
                print(f"  Sample {run_idx - self.warmup_runs + 1}: {throughput:.1f} t/s, TTFT={ttft:.1f}ms, TPOT={tpot:.2f}ms, accept={stats.get('acceptance_rate', 0):.2%}, path_len={stats.get('avg_accepted_path_length', 0):.1f}")
        
        self._restore_eos(original_eos)
        
        metrics.throughput_tps = sum(all_throughput) / len(all_throughput)
        metrics.acceptance_rate = sum(all_acceptance) / len(all_acceptance)
        metrics.avg_path_length = sum(all_path_lengths) / len(all_path_lengths)
        metrics.total_rounds = int(sum(all_rounds) / len(all_rounds))
        metrics.total_tokens_generated = self.max_new_tokens
        metrics.peak_memory_mb = peak_memory
        metrics.ttft_ms = sum(all_ttft) / len(all_ttft)
        metrics.tpot_ms = sum(all_tpot) / len(all_tpot)
        
        # FLOPs estimation
        avg_prompt_len = 512
        metrics.prefill_flops = self.flops_estimator.estimate_prefill_flops(avg_prompt_len)
        metrics.decode_flops = self.flops_estimator.estimate_tree_decode_flops(
            metrics.total_tokens_generated, tree_depth, branch_factor, 
            metrics.total_rounds, avg_prompt_len
        )
        metrics.total_flops = metrics.prefill_flops + metrics.decode_flops
        metrics.flops_per_token = metrics.decode_flops / metrics.total_tokens_generated
        
        if self.baseline_throughput:
            metrics.speedup = metrics.throughput_tps / self.baseline_throughput
        
        print(f"\nTree V2 Results:")
        print(f"  Throughput: {metrics.throughput_tps:.1f} t/s")
        print(f"  TTFT: {metrics.ttft_ms:.1f} ms")
        print(f"  TPOT: {metrics.tpot_ms:.2f} ms")
        print(f"  Acceptance Rate: {metrics.acceptance_rate:.2%}")
        print(f"  Avg Path Length: {metrics.avg_path_length:.1f}")
        print(f"  Speedup: {metrics.speedup:.2f}x")
        
        self.results.append(metrics)
        return metrics
    
    def compare_acceptance_rates(self) -> Dict[str, Any]:
        """
        Compare token acceptance rates between Tree and Linear methods.
        
        For Tree-based methods, acceptance rate = accepted_tokens / total_draft_tokens
        - Tree generates multiple candidate tokens per round (branching)
        - Only one path is accepted
        
        For Linear methods, acceptance rate = accepted_tokens / K
        - Linear generates K tokens sequentially
        - Accept consecutive tokens until mismatch
        """
        print("\n" + "="*60)
        print("Acceptance Rate Comparison: Tree vs Linear")
        print("="*60)
        
        comparison = {
            "linear_methods": [],
            "tree_methods": [],
            "analysis": {}
        }
        
        for metrics in self.results:
            if metrics.method.startswith("Linear"):
                comparison["linear_methods"].append({
                    "method": metrics.method,
                    "acceptance_rate": metrics.acceptance_rate,
                    "tokens_per_round": metrics.tokens_per_round,
                    "total_rounds": metrics.total_rounds,
                })
            elif metrics.method.startswith("Tree"):
                comparison["tree_methods"].append({
                    "method": metrics.method,
                    "acceptance_rate": metrics.acceptance_rate,
                    "avg_path_length": metrics.avg_path_length,
                    "total_rounds": metrics.total_rounds,
                })
        
        # Calculate averages
        if comparison["linear_methods"]:
            avg_linear_acceptance = sum(m["acceptance_rate"] for m in comparison["linear_methods"]) / len(comparison["linear_methods"])
            avg_linear_tokens_per_round = sum(m["tokens_per_round"] for m in comparison["linear_methods"]) / len(comparison["linear_methods"])
        else:
            avg_linear_acceptance = 0
            avg_linear_tokens_per_round = 0
        
        if comparison["tree_methods"]:
            avg_tree_acceptance = sum(m["acceptance_rate"] for m in comparison["tree_methods"]) / len(comparison["tree_methods"])
            avg_tree_path_length = sum(m["avg_path_length"] for m in comparison["tree_methods"]) / len(comparison["tree_methods"])
        else:
            avg_tree_acceptance = 0
            avg_tree_path_length = 0
        
        comparison["analysis"] = {
            "avg_linear_acceptance_rate": avg_linear_acceptance,
            "avg_linear_tokens_per_round": avg_linear_tokens_per_round,
            "avg_tree_acceptance_rate": avg_tree_acceptance,
            "avg_tree_path_length": avg_tree_path_length,
            "tree_vs_linear_acceptance_ratio": avg_tree_acceptance / avg_linear_acceptance if avg_linear_acceptance > 0 else 0,
            "explanation": """
Acceptance Rate Interpretation:

Linear Speculative Decoding:
- Acceptance Rate = (Total Accepted Tokens) / (Total Draft Tokens)
- Draft generates K tokens sequentially
- Accept consecutive tokens until mismatch with target
- Higher acceptance rate → more tokens accepted per verification

Tree Speculative Decoding:
- Acceptance Rate = (Accepted Path Tokens) / (Total Tree Nodes)
- Draft generates a tree of candidates (multiple branches)
- Only ONE path (longest valid) is accepted
- Lower raw acceptance rate is expected due to branching
- Avg Path Length is more meaningful metric for efficiency

Key Insight:
- Tree methods have lower raw acceptance rates because they generate 
  more candidate tokens (tree nodes) but only accept one path
- However, the accepted path is often LONGER than linear's acceptance
- This results in higher throughput despite lower raw acceptance rate
"""
        }
        
        print("\nLinear Methods:")
        for m in comparison["linear_methods"]:
            print(f"  {m['method']}: acceptance={m['acceptance_rate']:.2%}, tokens/round={m['tokens_per_round']:.1f}")
        
        print("\nTree Methods:")
        for m in comparison["tree_methods"]:
            print(f"  {m['method']}: acceptance={m['acceptance_rate']:.2%}, path_len={m['avg_path_length']:.1f}")
        
        print(f"\nSummary:")
        print(f"  Avg Linear Acceptance: {avg_linear_acceptance:.2%}")
        print(f"  Avg Linear Tokens/Round: {avg_linear_tokens_per_round:.1f}")
        print(f"  Avg Tree Acceptance: {avg_tree_acceptance:.2%}")
        print(f"  Avg Tree Path Length: {avg_tree_path_length:.1f}")
        
        return comparison
    
    def run_all_benchmarks(self) -> Dict[str, Any]:
        """Run all benchmarks from the report."""
        print("\n" + "#"*70)
        print("# Comprehensive Speculative Decoding Benchmark")
        print(f"# Dataset: {self.dataset_path}")
        print(f"# Target Model: {self.target_model_path}")
        print(f"# Draft Model: {self.draft_model_path}")
        print(f"# Max New Tokens: {self.max_new_tokens}")
        print(f"# Samples: {self.num_samples}")
        print("#"*70)
        
        # 1. Baseline
        self.benchmark_baseline()
        
        # 2. Tree V2 configurations (from parameter search results)
        tree_configs = [
            {"tree_depth": 4, "branch_factor": 2, "probability_threshold": 0.05},  # D=4 B=2 t=0.05
            {"tree_depth": 4, "branch_factor": 2, "probability_threshold": 0.03},  # D=4 B=2 t=0.03
            {"tree_depth": 5, "branch_factor": 2, "probability_threshold": 0.05},  # D=5 B=2 t=0.05
            {"tree_depth": 7, "branch_factor": 2, "probability_threshold": 0.05},  # D=7 B=2 t=0.05
            {"tree_depth": 6, "branch_factor": 2, "probability_threshold": 0.05},  # D=6 B=2 t=0.05
        ]
        for config in tree_configs:
            self.benchmark_tree_v2(**config)
        
        # 3. HuggingFace Assisted
        self.benchmark_hf_assisted()
        
        # 4. Linear variants (K roughly matches tree depth for comparison)
        for K in [4, 5, 6, 7]:
            self.benchmark_linear_spec_decode(K=K)
        
        # 5. Streaming variants
        self.benchmark_streaming_spec_decode(K=5, max_cache_len=1024)
        self.benchmark_streaming_spec_decode(K=6, max_cache_len=512)
        
        # 6. Compare acceptance rates
        acceptance_comparison = self.compare_acceptance_rates()
        
        # Generate summary
        summary = self._generate_summary(acceptance_comparison)
        
        return summary
    
    def _generate_summary(self, acceptance_comparison: Dict) -> Dict[str, Any]:
        """Generate comprehensive summary."""
        summary = {
            "config": {
                "target_model": self.target_model_path,
                "draft_model": self.draft_model_path,
                "dataset": self.dataset_path,
                "max_new_tokens": self.max_new_tokens,
                "num_samples": self.num_samples,
                "warmup_runs": self.warmup_runs,
            },
            "results": [asdict(m) for m in self.results],
            "acceptance_comparison": acceptance_comparison,
            "rankings": {},
            "timestamp": datetime.now().isoformat(),
        }
        
        # Generate rankings
        sorted_by_throughput = sorted(self.results, key=lambda x: x.throughput_tps, reverse=True)
        summary["rankings"]["by_throughput"] = [
            {"rank": i+1, "method": m.method, "throughput": m.throughput_tps, "speedup": m.speedup}
            for i, m in enumerate(sorted_by_throughput)
        ]
        
        sorted_by_acceptance = sorted(
            [m for m in self.results if m.acceptance_rate > 0],
            key=lambda x: x.acceptance_rate, reverse=True
        )
        summary["rankings"]["by_acceptance_rate"] = [
            {"rank": i+1, "method": m.method, "acceptance_rate": m.acceptance_rate}
            for i, m in enumerate(sorted_by_acceptance)
        ]
        
        # Print summary table
        print("\n" + "="*100)
        print("FINAL RESULTS SUMMARY")
        print("="*100)
        print(f"\n{'Rank':<5} {'Method':<35} {'Throughput':<12} {'Speedup':<10} {'TTFT(ms)':<10} {'TPOT(ms)':<10} {'Accept%':<10}")
        print("-"*100)
        for i, m in enumerate(sorted_by_throughput):
            accept_str = f"{m.acceptance_rate:.1%}" if m.acceptance_rate > 0 else "N/A"
            print(f"{i+1:<5} {m.method:<35} {m.throughput_tps:>8.1f} t/s {m.speedup:>7.2f}x {m.ttft_ms:>8.1f} {m.tpot_ms:>8.2f} {accept_str:>10}")
        
        print("\n" + "="*80)
        print("FLOPs COMPARISON")
        print("="*80)
        print(f"\n{'Method':<35} {'Total FLOPs':<15} {'FLOPs/Token':<15} {'Memory(MB)':<12}")
        print("-"*80)
        for m in sorted_by_throughput:
            if m.total_flops > 0:
                print(f"{m.method:<35} {m.total_flops:>12.2e} {m.flops_per_token:>12.2e} {m.peak_memory_mb:>10.0f}")
        
        return summary
    
    def save_results(self, output_path: str):
        """Save results to JSON file."""
        summary = self._generate_summary(self.compare_acceptance_rates())
        
        with open(output_path, 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        print(f"\nResults saved to: {output_path}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Comprehensive Speculative Decoding Benchmark")
    parser.add_argument("--target-model", type=str, default="/mnt/disk1/models/pythia-2.8b")
    parser.add_argument("--draft-model", type=str, default="/mnt/disk1/models/pythia-70m")
    parser.add_argument("--dataset", type=str, default="/mnt/disk1/ljm/LLM-Efficient-Reasoning/data/pg19.parquet")
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=500)
    parser.add_argument("--num-samples", type=int, default=5)
    parser.add_argument("--warmup-runs", type=int, default=2)
    parser.add_argument("--quick", action="store_true", help="Quick test with reduced samples")
    parser.add_argument("--use-dataset", action="store_true", default=False,
                        help="Use pg19 dataset prompts instead of standard short prompts")
    
    args = parser.parse_args()
    
    if args.quick:
        args.num_samples = 2
        args.warmup_runs = 1
        args.max_new_tokens = 200
    
    benchmark = ComprehensiveBenchmark(
        target_model_path=args.target_model,
        draft_model_path=args.draft_model,
        dataset_path=args.dataset,
        num_samples=args.num_samples,
        max_new_tokens=args.max_new_tokens,
        warmup_runs=args.warmup_runs,
        use_dataset=args.use_dataset
    )
    
    benchmark.run_all_benchmarks()
    
    # Save results
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = f"results/comprehensive_benchmark_{timestamp}.json"
    
    os.makedirs(os.path.dirname(args.output) if os.path.dirname(args.output) else ".", exist_ok=True)
    benchmark.save_results(args.output)


if __name__ == "__main__":
    main()

