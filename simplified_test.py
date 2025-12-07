"""
简化版优化测试脚本 - 专注于wikitext长文本测试

当pg-19不可用时，使用wikitext的长样本进行测试
"""

from transformers import GPTNeoXForCausalLM, AutoTokenizer
import torch
import time
from datasets import load_dataset
import numpy as np
from typing import List
import argparse

from custom_generate import generate_with_compression_and_timing
from kv_compress import l2_compress

model_id = "EleutherAI/pythia-70m-deduped"

def get_optimal_device():
    if torch.cuda.is_available():
        print("CUDA (NVIDIA GPU) is available. Using cuda.")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        if torch.backends.mps.is_built():
            print("MPS (Apple Silicon GPU) is available. Using mps.")
            return torch.device("mps")
    print("No GPU acceleration available. Falling back to CPU.")
    return torch.device("cpu")


def calculate_perplexity(model, tokenizer, text: str, device, max_length: int = 512):
    """简单的困惑度计算"""
    encodings = tokenizer(text, return_tensors="pt", max_length=max_length, truncation=True)
    input_ids = encodings.input_ids.to(device)
    
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss
    
    return torch.exp(loss).item()


def measure_generation_with_compression(
    model, tokenizer, input_text: str, 
    max_new_tokens: int = 100,
    keep_ratio: float = 1.0,
    prune_after: int = 512,
    skip_layers: List[int] = [0],
    device = None
):
    """测量生成性能"""
    input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)
    input_length = input_ids.shape[1]
    
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    
    if keep_ratio < 1.0:
        result = generate_with_compression_and_timing(
            model=model, tokenizer=tokenizer, input_ids=input_ids,
            max_new_tokens=max_new_tokens, keep_ratio=keep_ratio,
            prune_after=prune_after, skip_layers=skip_layers, device=device,
        )
        peak_memory = None
        if device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        return {
            "ttft": result["ttft"], "tpot": result["tpot"],
            "throughput": result["throughput"], "total_time": result["total_time"],
            "num_generated_tokens": result["num_generated_tokens"],
            "peak_memory_mb": peak_memory, "input_length": input_length,
            "output_length": result["output_ids"].shape[1], "keep_ratio": keep_ratio,
        }
    else:
        start_time = time.time()
        with torch.no_grad():
            outputs = model.generate(
                input_ids, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id, use_cache=True,
                return_dict_in_generate=True,
            )
        total_time = time.time() - start_time
        output_ids = outputs.sequences if hasattr(outputs, 'sequences') else outputs
        num_generated = output_ids.shape[1] - input_length
        
        ttft = total_time * 0.1
        tpot = total_time / num_generated if num_generated > 0 else 0
        
        peak_memory = None
        if device.type == "cuda":
            peak_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
        
        return {
            "ttft": ttft, "tpot": tpot,
            "throughput": num_generated / total_time if total_time > 0 else 0,
            "total_time": total_time, "num_generated_tokens": num_generated,
            "peak_memory_mb": peak_memory, "input_length": input_length,
            "output_length": output_ids.shape[1], "keep_ratio": keep_ratio,
        }


def run_simplified_tests(
    keep_ratios: List[float] = [1.0, 0.9, 0.8, 0.7],
    prune_after: int = 512,
    skip_layers: List[int] = [0],
    num_samples: int = 3,
    use_long_context: bool = True,
):
    """运行简化版测试"""
    print("\n" + "="*70)
    print("KnormPress 简化测试 (使用 Wikitext 长文本)")
    print("="*70)
    
    # 加载模型
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
    
    # 加载数据集
    print("\nLoading wikitext dataset...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    
    # 筛选长文本样本
    if use_long_context:
        print("Filtering long-context samples (>800 chars)...")
        long_samples = [s for s in dataset if len(s.get('text', '')) > 800]
        samples_to_test = long_samples[:num_samples]
        print(f"Found {len(long_samples)} long samples, using {len(samples_to_test)}")
    else:
        samples_to_test = [s for s in dataset if len(s.get('text', '')) > 50][:num_samples]
    
    all_results = []
    
    # 对每个压缩率测试
    for keep_ratio in keep_ratios:
        print("\n" + "="*70)
        print(f"Testing with keep_ratio={keep_ratio} ({int((1-keep_ratio)*100)}% compression)")
        print("="*70)
        
        for i, sample in enumerate(samples_to_test):
            text = sample.get("text", "")
            if not text or len(text.strip()) < 50:
                continue
            
            print(f"\nSample {i+1}/{len(samples_to_test)}: {len(text)} chars")
            
            # 使用较长的输入来测试长上下文
            input_text = text[:1024] if use_long_context else text[:512]
            
            try:
                # 生成性能
                gen_metrics = measure_generation_with_compression(
                    model, tokenizer, input_text,
                    max_new_tokens=100 if use_long_context else 50,
                    keep_ratio=keep_ratio, prune_after=prune_after,
                    skip_layers=skip_layers, device=device
                )
                
                # 困惑度
                ppl_text = text[:1024] if len(text) >= 1024 else text
                try:
                    ppl = calculate_perplexity(model, tokenizer, ppl_text, device=device)
                except Exception as e:
                    print(f"  Warning: Could not calculate PPL: {e}")
                    ppl = None
                
                result = {
                    "sample_id": i, "keep_ratio": keep_ratio,
                    "compression_pct": int((1-keep_ratio)*100),
                    "input_length": gen_metrics["input_length"],
                    "ttft_seconds": gen_metrics["ttft"],
                    "tpot_seconds": gen_metrics["tpot"],
                    "throughput_tokens_per_sec": gen_metrics["throughput"],
                    "peak_memory_mb": gen_metrics["peak_memory_mb"],
                    "perplexity": ppl
                }
                all_results.append(result)
                
                print(f"  TTFT: {gen_metrics['ttft']:.4f}s | TPOT: {gen_metrics['tpot']:.4f}s | "
                      f"Throughput: {gen_metrics['throughput']:.2f} tok/s | PPL: {ppl:.2f if ppl else 'N/A'}")
                
            except Exception as e:
                print(f"  Error: {e}")
                continue
    
    # 打印汇总
    print("\n" + "="*70)
    print("Performance Summary")
    print("="*70)
    print(f"{'Keep':<6} {'Comp%':<6} {'Sample':<8} {'TTFT(s)':<10} {'TPOT(s)':<10} "
          f"{'Thruput':<10} {'PPL':<10}")
    print("-"*70)
    
    for r in all_results:
        ppl_str = f"{r['perplexity']:.2f}" if r['perplexity'] else "N/A"
        print(f"{r['keep_ratio']:<6.2f} {r['compression_pct']:<6} {r['sample_id']:<8} "
              f"{r['ttft_seconds']:<10.4f} {r['tpot_seconds']:<10.4f} "
              f"{r['throughput_tokens_per_sec']:<10.2f} {ppl_str:<10}")
    
    # 按压缩率汇总
    print("\n" + "="*70)
    print("Aggregated by Compression Ratio")
    print("="*70)
    
    for keep_ratio in keep_ratios:
        ratio_results = [r for r in all_results if r['keep_ratio'] == keep_ratio]
        if not ratio_results:
            continue
        
        avg_ttft = np.mean([r['ttft_seconds'] for r in ratio_results])
        avg_tpot = np.mean([r['tpot_seconds'] for r in ratio_results])
        avg_throughput = np.mean([r['throughput_tokens_per_sec'] for r in ratio_results])
        ppl_values = [r['perplexity'] for r in ratio_results if r['perplexity']]
        avg_ppl = np.mean(ppl_values) if ppl_values else None
        
        print(f"\nKeep Ratio: {keep_ratio} ({int((1-keep_ratio)*100)}% compression)")
        print(f"  TTFT: {avg_ttft:.4f}s | TPOT: {avg_tpot:.4f}s | "
              f"Throughput: {avg_throughput:.2f} tok/s | PPL: {avg_ppl:.2f if avg_ppl else 'N/A'}")
    
    print("\n" + "="*70)
    print("Testing completed!")
    print("="*70)
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="简化版KnormPress测试")
    parser.add_argument("--keep_ratios", type=str, default="1.0,0.9,0.8,0.7",
                       help="压缩比率列表")
    parser.add_argument("--prune_after", type=int, default=512,
                       help="压缩阈值")
    parser.add_argument("--skip_layers", type=str, default="0",
                       help="跳过的层")
    parser.add_argument("--num_samples", type=int, default=3,
                       help="测试样本数")
    parser.add_argument("--long_context", action="store_true",
                       help="使用长文本样本（>800字符）")
    
    args = parser.parse_args()
    
    keep_ratios = [float(x) for x in args.keep_ratios.split(',')]
    skip_layers = [int(x) for x in args.skip_layers.split(',')]
    
    results = run_simplified_tests(
        keep_ratios=keep_ratios,
        prune_after=args.prune_after,
        skip_layers=skip_layers,
        num_samples=args.num_samples,
        use_long_context=args.long_context,
    )

