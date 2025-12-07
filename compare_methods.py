"""
对照实验：比较KnormPress与直接截断方法在PG-19数据集上的表现

三条曲线：
1. 基线（无压缩）：水平线
2. KnormPress（L2范数压缩）：PPL vs keep_ratio
3. 直接截断（删除最前面的tokens）：PPL vs keep_ratio
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import GPTNeoXForCausalLM, AutoTokenizer, DynamicCache
from datasets import load_dataset
from typing import List, Tuple
from math import ceil
import os
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def l2_compress(past_key_values, keep_ratio: float = 1.0, prune_after: int = 256, skip_layers: List[int] = []):
    """
    KnormPress: 基于L2范数的KV cache压缩
    保留L2范数最低的token（对应高attention分数）
    """
    past_key_values = list(past_key_values)
    
    for layer_idx, (keys, values) in enumerate(past_key_values):
        seq_len = keys.size(2)
        if seq_len < prune_after:
            continue
        if layer_idx in skip_layers:
            continue
        
        tokens_to_keep = ceil(keep_ratio * seq_len)
        if tokens_to_keep >= seq_len:
            continue
        
        batch_size, num_heads, seq_len, head_dim = keys.shape
        
        # 计算L2范数
        token_norms = torch.norm(keys, p=2, dim=-1)
        sorted_indices = token_norms.argsort(dim=-1)
        
        # 保留L2范数最低的tokens
        indices_to_keep = sorted_indices[:, :, :tokens_to_keep]
        # 保持时序
        indices_to_keep_sorted, _ = torch.sort(indices_to_keep, dim=-1)
        
        indices_expanded = indices_to_keep_sorted.unsqueeze(-1).expand(
            batch_size, num_heads, tokens_to_keep, head_dim
        )
        
        compressed_keys = torch.gather(keys, dim=2, index=indices_expanded)
        compressed_values = torch.gather(values, dim=2, index=indices_expanded)
        
        past_key_values[layer_idx] = (compressed_keys, compressed_values)
    
    return past_key_values


def truncate_compress(past_key_values, keep_ratio: float = 1.0, prune_after: int = 256, skip_layers: List[int] = []):
    """
    直接截断：删除最前面的tokens，保留最新的tokens
    这是最简单的压缩方法
    """
    past_key_values = list(past_key_values)
    
    for layer_idx, (keys, values) in enumerate(past_key_values):
        seq_len = keys.size(2)
        if seq_len < prune_after:
            continue
        if layer_idx in skip_layers:
            continue
        
        tokens_to_keep = ceil(keep_ratio * seq_len)
        if tokens_to_keep >= seq_len:
            continue
        
        # 直接保留最新的tokens（删除最前面的）
        compressed_keys = keys[:, :, -tokens_to_keep:, :]
        compressed_values = values[:, :, -tokens_to_keep:, :]
        
        past_key_values[layer_idx] = (compressed_keys, compressed_values)
    
    return past_key_values


def calculate_ppl_with_method(
    model, tokenizer, text: str, 
    keep_ratio: float, 
    compress_method: str,  # "none", "knorm", "truncate"
    prune_after: int = 256,
    skip_layers: List[int] = [0],
    device = None,
    max_eval_tokens: int = 1024,
) -> float:
    """
    使用指定方法计算PPL
    """
    # Tokenize
    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_ids = input_ids[:, :max_eval_tokens].to(device)
    seq_len = input_ids.shape[1]
    
    # 调整prune_after
    effective_prune_after = min(prune_after, max(seq_len - 50, seq_len // 2))
    effective_prune_after = max(effective_prune_after, 32)
    
    if seq_len < effective_prune_after + 10:
        with torch.no_grad():
            outputs = model(input_ids, labels=input_ids)
        return torch.exp(outputs.loss).item()
    
    total_loss = 0.0
    num_tokens = 0
    past_kv = None
    
    model.eval()
    with torch.no_grad():
        # Phase 1: 预填充
        prefill_input = input_ids[:, :effective_prune_after]
        outputs = model(prefill_input, use_cache=True, return_dict=True)
        past_kv = outputs.past_key_values
        
        # Phase 2: 带压缩的评估
        for i in range(effective_prune_after, seq_len - 1):
            current_input = input_ids[:, i:i+1]
            target = input_ids[:, i+1]
            
            outputs = model(current_input, past_key_values=past_kv, use_cache=True, return_dict=True)
            
            logits = outputs.logits[:, -1, :]
            loss = torch.nn.functional.cross_entropy(logits, target, reduction='sum')
            total_loss += loss.item()
            num_tokens += 1
            
            past_kv = outputs.past_key_values
            
            # 应用压缩
            if compress_method != "none" and keep_ratio < 1.0 and past_kv is not None:
                if hasattr(past_kv, 'to_legacy_cache'):
                    kv_list = past_kv.to_legacy_cache()
                else:
                    kv_list = list(past_kv)
                
                # 选择压缩方法
                if compress_method == "knorm":
                    compressed_kv = l2_compress(kv_list, keep_ratio, effective_prune_after, skip_layers)
                elif compress_method == "truncate":
                    compressed_kv = truncate_compress(kv_list, keep_ratio, effective_prune_after, skip_layers)
                else:
                    compressed_kv = kv_list
                
                # 转回DynamicCache
                new_cache = DynamicCache()
                for layer_idx, (k, v) in enumerate(compressed_kv):
                    new_cache.update(k, v, layer_idx)
                past_kv = new_cache
    
    if num_tokens == 0:
        return float('inf')
    
    return np.exp(total_loss / num_tokens)


def run_comparison_experiment():
    """运行对照实验"""
    print("="*70)
    print("KnormPress vs 直接截断 对照实验")
    print("数据集: PG-19 (长文本)")
    print("="*70)
    
    # 加载模型
    print("\n加载模型...")
    model_id = "EleutherAI/pythia-70m-deduped"
    model = GPTNeoXForCausalLM.from_pretrained(model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else 
                         "cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    print(f"模型加载完成，使用设备: {device}")
    
    # 加载PG-19数据集
    print("\n加载PG-19数据集...")
    local_pg19_path = "/Users/od/Desktop/NLP/CS2602-LLM-Inference-Acceleration/data/pg19.parquet"
    
    # 检查本地文件
    old_path = "/Users/od/Desktop/NLP/CS2602-LLM-Inference-Acceleration/test-00000-of-00001-29a571947c0b5ccc.parquet"
    if os.path.exists(old_path) and not os.path.exists(local_pg19_path):
        # 移动文件到data目录
        os.rename(old_path, local_pg19_path)
        print(f"数据文件已移动到: {local_pg19_path}")
    
    if os.path.exists(local_pg19_path):
        pg19_dataset = load_dataset("parquet", data_files={'test': local_pg19_path}, split="test")
    else:
        pg19_dataset = load_dataset("pg19", split="test")
    
    print(f"加载了 {len(pg19_dataset)} 个样本")
    
    # 选择测试样本（长文本）
    test_samples = []
    for i, sample in enumerate(pg19_dataset):
        text = sample.get("text", "")
        if len(text) > 50000:  # 选择长文本
            test_samples.append(text)
            if len(test_samples) >= 5:
                break
    
    print(f"选择了 {len(test_samples)} 个长文本样本进行测试")
    
    # 测试的压缩率
    keep_ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    
    # 存储结果
    results = {
        "knorm": [],
        "truncate": [],
        "baseline": None
    }
    
    # 首先计算基线PPL
    print("\n计算基线PPL (无压缩)...")
    baseline_ppls = []
    for idx, text in enumerate(test_samples):
        ppl = calculate_ppl_with_method(
            model, tokenizer, text,
            keep_ratio=1.0,
            compress_method="none",
            device=device,
            max_eval_tokens=1024
        )
        baseline_ppls.append(ppl)
        print(f"  样本 {idx+1}: PPL = {ppl:.2f}")
    
    baseline_ppl = np.mean(baseline_ppls)
    results["baseline"] = baseline_ppl
    print(f"基线平均PPL: {baseline_ppl:.2f}")
    
    # 测试KnormPress
    print("\n测试 KnormPress (L2范数压缩)...")
    for keep_ratio in keep_ratios:
        if keep_ratio == 1.0:
            results["knorm"].append(baseline_ppl)
            continue
        
        ppls = []
        for text in test_samples:
            ppl = calculate_ppl_with_method(
                model, tokenizer, text,
                keep_ratio=keep_ratio,
                compress_method="knorm",
                device=device,
                max_eval_tokens=1024
            )
            ppls.append(ppl)
        
        avg_ppl = np.mean(ppls)
        results["knorm"].append(avg_ppl)
        print(f"  keep_ratio={keep_ratio:.1f}: PPL = {avg_ppl:.2f}")
    
    # 测试直接截断
    print("\n测试 直接截断 (删除最前面的tokens)...")
    for keep_ratio in keep_ratios:
        if keep_ratio == 1.0:
            results["truncate"].append(baseline_ppl)
            continue
        
        ppls = []
        for text in test_samples:
            ppl = calculate_ppl_with_method(
                model, tokenizer, text,
                keep_ratio=keep_ratio,
                compress_method="truncate",
                device=device,
                max_eval_tokens=1024
            )
            ppls.append(ppl)
        
        avg_ppl = np.mean(ppls)
        results["truncate"].append(avg_ppl)
        print(f"  keep_ratio={keep_ratio:.1f}: PPL = {avg_ppl:.2f}")
    
    # 绘制对比图
    print("\n生成对比图...")
    plot_comparison(keep_ratios, results)
    
    # 打印汇总表格
    print("\n" + "="*70)
    print("实验结果汇总")
    print("="*70)
    print(f"{'Keep Ratio':<12} {'KnormPress PPL':<18} {'Truncate PPL':<18} {'差异':<12}")
    print("-"*60)
    for i, kr in enumerate(keep_ratios):
        knorm_ppl = results["knorm"][i]
        trunc_ppl = results["truncate"][i]
        diff = trunc_ppl - knorm_ppl
        print(f"{kr:<12.1f} {knorm_ppl:<18.2f} {trunc_ppl:<18.2f} {diff:+.2f}")
    
    print(f"\n基线PPL (无压缩): {baseline_ppl:.2f}")
    print("="*70)
    
    return results


def plot_comparison(keep_ratios, results):
    """绘制对比图"""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 压缩率（x轴）
    compression_ratios = [(1 - kr) * 100 for kr in keep_ratios]
    
    # 绘制基线（水平线）
    ax.axhline(y=results["baseline"], color='green', linestyle='--', linewidth=2, 
               label=f'Baseline (No Compression): PPL={results["baseline"]:.2f}')
    
    # 绘制KnormPress曲线
    ax.plot(compression_ratios, results["knorm"], 'b-o', linewidth=2, markersize=8,
            label='KnormPress (L2 Norm)')
    
    # 绘制直接截断曲线
    ax.plot(compression_ratios, results["truncate"], 'r-s', linewidth=2, markersize=8,
            label='Direct Truncation (Remove Oldest)')
    
    # 设置坐标轴
    ax.set_xlabel('Compression Ratio (%)', fontsize=14)
    ax.set_ylabel('Perplexity (PPL)', fontsize=14)
    ax.set_title('KnormPress vs Direct Truncation on PG-19 Long Text Dataset\n'
                 '(Pythia-70M, Lower PPL is Better)', fontsize=14)
    
    # 设置x轴范围
    ax.set_xlim(-5, 95)
    ax.set_xticks([0, 10, 20, 30, 40, 50, 60, 70, 80, 90])
    
    # 添加网格
    ax.grid(True, alpha=0.3)
    
    # 添加图例
    ax.legend(loc='upper left', fontsize=11)
    
    # 添加注释区域
    ax.fill_between([0, 30], [0, 0], [1000, 1000], alpha=0.1, color='green',
                    label='_Safe Zone (≤30% compression)')
    
    # 保存图片
    output_path = "/Users/od/Desktop/NLP/CS2602-LLM-Inference-Acceleration/results/compression_comparison.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"对比图已保存到: {output_path}")
    
    plt.close()


if __name__ == "__main__":
    results = run_comparison_experiment()

