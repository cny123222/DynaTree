#!/bin/bash
# ====================================================================
# Tree-based Speculative Decoding - Paper Experiments Runner
# ====================================================================
# 这个脚本运行所有论文需要的实验
# 所有参数都基于实际实验结果 (papers/Tree_Speculative_Decoding_实验报告.md)
# ====================================================================

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}======================================================================"
echo "           Tree-based Speculative Decoding"
echo "                 论文实验运行脚本"
echo "======================================================================${NC}"
echo ""
echo "这个脚本会运行以下实验："
echo "  1. 主要性能对比 (Baseline vs Linear vs Tree-based)"
echo "  2. 参数影响分析 (深度D、分支B、阈值τ)"
echo "  3. 消融实验 (动态剪枝的效果)"
echo "  4. 序列长度扩展 (100/200/300/500/1000 tokens)"
echo ""

# ====================================================================
# Step 1: 检测模型路径
# ====================================================================
echo -e "${BLUE}[步骤 1/5] 检测模型路径...${NC}"
python3 check_models.py > /tmp/model_check.txt 2>&1

# Extract model paths from check output
if grep -q "/mnt/disk1/models/pythia-2.8b" /tmp/model_check.txt; then
    TARGET_MODEL="/mnt/disk1/models/pythia-2.8b"
elif [ -d "./models/pythia-2.8b" ]; then
    TARGET_MODEL="./models/pythia-2.8b"
else
    TARGET_MODEL="EleutherAI/pythia-2.8b"
fi

if grep -q "/mnt/disk1/models/pythia-70m" /tmp/model_check.txt; then
    DRAFT_MODEL="/mnt/disk1/models/pythia-70m"
elif [ -d "./models/pythia-70m" ]; then
    DRAFT_MODEL="./models/pythia-70m"
else
    DRAFT_MODEL="EleutherAI/pythia-70m"
fi

echo "  ✓ Target Model: $TARGET_MODEL"
echo "  ✓ Draft Model: $DRAFT_MODEL"
echo ""

# ====================================================================
# Step 2: 创建输出目录
# ====================================================================
echo -e "${BLUE}[步骤 2/5] 创建输出目录...${NC}"
mkdir -p results/final_experiments
mkdir -p papers/figures/final

# Timestamp for this run
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR="results/final_experiments/${TIMESTAMP}"
mkdir -p "$RESULTS_DIR"

echo "  ✓ 结果将保存到: $RESULTS_DIR"
echo ""

# ====================================================================
# Step 3: 显示实验参数说明
# ====================================================================
echo -e "${BLUE}[步骤 3/5] 实验参数说明${NC}"
echo ""
echo -e "${YELLOW}我们使用的参数来自实际实验结果:${NC}"
echo "  参考文件: papers/Tree_Speculative_Decoding_实验报告.md"
echo "  参数搜索文件: results/tree_param_search_20251231_140952.json"
echo ""
echo "最优参数 (Optimal Parameters - 基于实际测试):"
echo "  ┌─────────────┬───────┬─────────┬──────────┬──────────────────┐"
echo "  │ Token Length│ Depth │ Branch  │ Threshold│ Speedup (实测)   │"
echo "  ├─────────────┼───────┼─────────┼──────────┼──────────────────┤"
echo "  │  100 tokens │   7   │    3    │   0.03   │    1.43x         │"
echo "  │  200 tokens │   7   │    3    │   0.03   │    1.54x         │"
echo "  │  300 tokens │   7   │    3    │   0.03   │    1.60x         │"
echo "  │  500 tokens │   8   │    3    │   0.03   │ 1.62x-1.79x ⭐  │"
echo "  │ 1000 tokens │   6   │    3    │   0.05   │    1.71x         │"
echo "  └─────────────┴───────┴─────────┴──────────┴──────────────────┘"
echo ""
echo -e "${YELLOW}关键发现:${NC}"
echo "  • 500 tokens 是最优测试长度，加速比最高"
echo "  • D=8, B=3, τ=0.03 是 500 tokens 的最优配置"
echo "  • 树深度需要随生成长度调整 (短序列用D=7，长序列用D=8)"
echo ""
echo -e "${YELLOW}参数含义:${NC}"
echo "  • Depth (D): 树的最大深度，控制推测的步数"
echo "  • Branch (B): 每个节点的分支数，控制每步的候选数量"
echo "  • Threshold (τ): 动态剪枝阈值，概率低于此值的分支会被剪掉"
echo ""

# ====================================================================
# Experiment 1: 主要性能对比 (500 tokens - 最优长度)
# ====================================================================
echo ""
echo -e "${GREEN}======================================================================"
echo "实验 1: 主要性能对比 (Main Performance Comparison)"
echo -e "======================================================================${NC}"
echo ""
echo "这个实验对比所有方法在最优配置下的性能"
echo ""
echo "测试的方法:"
echo "  1. Baseline: 标准自回归生成 (无推测解码)"
echo "  2. HuggingFace Assisted: HF 官方实现"
echo "  3. Linear Speculative Decoding: K=5,6,7,8"
echo "  4. Tree V2 (D=8, B=3, τ=0.03): 我们的方法 ⭐"
echo ""
echo "测试配置:"
echo "  • 生成长度: 500 tokens (最优长度)"
echo "  • 样本数量: 5 prompts"
echo "  • 最优参数: D=8, B=3, τ=0.03"
echo ""
echo "评估指标:"
echo "  • Throughput (吞吐量): tokens/second"
echo "  • TPOT (每token时间): ms/token"
echo "  • Acceptance Rate (接受率): 推测token被接受的比例"
echo "  • Speedup (加速比): vs Baseline"
echo ""
echo "预期结果 (基于实际实验报告):"
echo "  • Tree V2:       193.4 t/s (1.62x) ⭐ 最快"
echo "  • HF Assisted:   161.9 t/s (1.36x)"
echo "  • Linear K=6:    133.1 t/s (1.11x)"
echo "  • Baseline:      119.4 t/s (1.00x)"
echo ""
echo "预计时间: ~15 分钟 (5次运行，跳过首次warmup)"
echo "对应论文: Table 2 (主实验结果表)"
echo ""

read -p "运行实验 1? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}开始运行实验 1...${NC}"
    python spec_decode/benchmark_tree_vs_linear.py \
        --target-model "$TARGET_MODEL" \
        --draft-model "$DRAFT_MODEL" \
        --max-new-tokens 500 \
        --num-prompts 5 \
        --device cuda \
        --save \
        --output-dir "$RESULTS_DIR"
    
    # Rename the output file
    LATEST_RESULT=$(ls -t "$RESULTS_DIR"/tree_vs_linear_benchmark_*.json 2>/dev/null | head -1)
    if [ -f "$LATEST_RESULT" ]; then
        mv "$LATEST_RESULT" "$RESULTS_DIR/exp1_main_comparison_500tokens.json"
        echo ""
        echo -e "${GREEN}✓ 实验 1 完成!${NC}"
        echo "  输出文件: $RESULTS_DIR/exp1_main_comparison_500tokens.json"
    fi
else
    echo -e "${YELLOW}⊘ 跳过实验 1${NC}"
fi

# ====================================================================
# Experiment 2: 参数影响分析
# ====================================================================
echo ""
echo -e "${GREEN}======================================================================"
echo "实验 2: 参数影响分析 (Parameter Sweep Visualization)"
echo -e "======================================================================${NC}"
echo ""
echo "这个实验可视化超参数对性能的影响"
echo ""
echo "参数搜索范围 (已完成的实验):"
echo "  • 深度 (Depth):     [3, 4, 5, 6, 7, 8]"
echo "  • 分支因子 (Branch): [2, 3, 4]"
echo "  • 剪枝阈值 (Threshold): [0.01, 0.02, 0.03, 0.05, 0.1]"
echo "  • Token长度:        [100, 200, 300, 500, 1000]"
echo "  • 总配置数: 6 × 3 × 5 × 5 = 450 组"
echo ""
echo "关键发现:"
echo "  • B=3 是最优分支因子 (平均加速 1.31x)"
echo "  • τ=0.03 是最优阈值 (最大加速 1.79x)"
echo "  • 深度需要根据token长度调整"
echo ""
echo "数据来源:"
echo "  • 文件: results/tree_param_search_20251231_140952.json"
echo "  • 包含 450 组参数配置的完整测试结果"
echo ""
echo "生成图表:"
echo "  • 热力图: 展示不同参数组合的吞吐量"
echo "  • 曲线图: 展示单个参数的影响趋势"
echo "  • Top-10 最优配置列表"
echo ""
echo "预计时间: ~5 分钟"
echo "对应论文: Figure 2 (参数影响图)"
echo ""

read -p "运行实验 2? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    # Check if we have existing param search data
    PARAM_SEARCH_FILE=$(ls -t results/tree_param_search_*.json 2>/dev/null | head -1)
    
    if [ -f "$PARAM_SEARCH_FILE" ]; then
        echo ""
        echo -e "${YELLOW}使用已有参数搜索数据: $PARAM_SEARCH_FILE${NC}"
        
        # Check if plotting script exists
        if [ -f "papers/plot_param_sweep_publication.py" ]; then
            python papers/plot_param_sweep_publication.py \
                --input "$PARAM_SEARCH_FILE" \
                --output "$RESULTS_DIR/exp2_param_sweep.pdf" \
                --style publication
            
            echo ""
            echo -e "${GREEN}✓ 实验 2 完成!${NC}"
            echo "  输出文件: $RESULTS_DIR/exp2_param_sweep.pdf"
        else
            echo -e "${YELLOW}警告: papers/plot_param_sweep_publication.py 不存在${NC}"
            echo "使用分析脚本生成文本报告..."
            
            # Use the analysis script as fallback
            python papers/analyze_tree_search_results.py "$PARAM_SEARCH_FILE" \
                > "$RESULTS_DIR/exp2_param_analysis.txt"
            
            echo ""
            echo -e "${GREEN}✓ 实验 2 完成 (文本分析)${NC}"
            echo "  输出文件: $RESULTS_DIR/exp2_param_analysis.txt"
        fi
    else
        echo ""
        echo -e "${RED}✗ 错误: 未找到参数搜索数据!${NC}"
        echo "请先运行参数搜索:"
        echo "  python papers/tree_param_search.py"
    fi
else
    echo -e "${YELLOW}⊘ 跳过实验 2${NC}"
fi

# ====================================================================
# Experiment 3: 消融实验 (Pruning Ablation)
# ====================================================================
echo ""
echo -e "${GREEN}======================================================================"
echo "实验 3: 消融实验 - 动态剪枝效果 (Ablation Study)"
echo -e "======================================================================${NC}"
echo ""
echo "这个实验测试动态剪枝(Dynamic Pruning)的有效性"
echo ""
echo "测试的三个变体:"
echo "  1. No Pruning (无剪枝)"
echo "     • threshold=0.0, max_nodes=9999"
echo "     • 树会非常大，包含所有可能的分支"
echo "     • 预期: 速度慢，显存占用高"
echo ""
echo "  2. Static Pruning (静态剪枝)"
echo "     • threshold=0.0, max_nodes=64"
echo "     • 通过固定节点数限制树的大小"
echo "     • 预期: 中等性能"
echo ""
echo "  3. Dynamic Pruning (动态剪枝) ← 我们的方法"
echo "     • threshold=0.03, max_nodes=128"
echo "     • 根据概率动态剪掉不太可能的分支"
echo "     • 预期: 最佳性能 (1.62x加速)"
echo ""
echo "评估指标:"
echo "  • Throughput: 吞吐量"
echo "  • Avg Nodes: 平均树节点数"
echo "  • Avg Path Length: 平均接受路径长度"
echo "  • Acceptance Rate: 接受率"
echo ""
echo "预计时间: ~20 分钟"
echo "对应论文: Table 3 (消融实验表)"
echo ""

read -p "运行实验 3? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${YELLOW}开始运行实验 3...${NC}"
    
    # Create ablation script if it doesn't exist
    if [ ! -f "spec_decode/ablation_pruning.py" ]; then
        echo "创建消融实验脚本..."
        cat > spec_decode/ablation_pruning.py << 'ABLATION_SCRIPT'
#!/usr/bin/env python3
"""
Ablation Study: Dynamic Pruning in Tree-based Speculative Decoding

This script compares three pruning strategies:
1. No Pruning: Allow tree to grow without limits
2. Static Pruning: Fixed maximum node count
3. Dynamic Pruning: Probability-based pruning (our method)
"""
import argparse
import json
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spec_decode.core import TreeSpeculativeGeneratorV2
import time

def run_ablation(args):
    print("=" * 70)
    print("Ablation Study: Dynamic Pruning Effect")
    print("=" * 70)
    print("\nLoading models...")
    
    target_model = AutoModelForCausalLM.from_pretrained(
        args.target_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    target_model.eval()
    
    draft_model = AutoModelForCausalLM.from_pretrained(
        args.draft_model,
        torch_dtype=torch.float16,
        device_map="auto"
    )
    draft_model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.target_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Test prompts - same as in the report
    prompts = [
        "Write a detailed technical explanation about the development of large language models. Cover the history, architecture innovations, training techniques, and future directions.",
        "Explain the concept of neural networks and deep learning, including their applications in computer vision and natural language processing.",
        "Discuss the challenges and opportunities in artificial intelligence research, focusing on ethical considerations and societal impact.",
        "Describe the evolution of programming languages from assembly to modern high-level languages, highlighting key innovations.",
        "Analyze the impact of quantum computing on cryptography and information security in the coming decades.",
    ]
    
    # Three pruning strategies - matching the actual experiment configuration
    variants = {
        "no_prune": {
            "name": "No Pruning",
            "threshold": 0.0,
            "max_nodes": 9999,
            "description": "Allow unlimited tree growth"
        },
        "static_prune": {
            "name": "Static Pruning",
            "threshold": 0.0,
            "max_nodes": 64,
            "description": "Fixed max nodes limit"
        },
        "dynamic_prune": {
            "name": "Dynamic Pruning (Ours)",
            "threshold": 0.03,  # Optimal from actual experiments
            "max_nodes": 128,   # Optimal from actual experiments
            "description": "Probability-based pruning"
        },
    }
    
    results = {}
    
    for variant_id, variant_config in variants.items():
        print(f"\n{'=' * 70}")
        print(f"Testing: {variant_config['name']}")
        print(f"  Description: {variant_config['description']}")
        print(f"  Threshold: {variant_config['threshold']}")
        print(f"  Max Nodes: {variant_config['max_nodes']}")
        print("=" * 70)
        
        generator = TreeSpeculativeGeneratorV2(
            target_model=target_model,
            draft_model=draft_model,
            tokenizer=tokenizer,
            tree_depth=args.depth,
            branch_factor=args.branch,
            probability_threshold=variant_config["threshold"],
            max_tree_nodes=variant_config["max_nodes"],
            device="cuda",
            use_compile=False
        )
        
        times = []
        stats_list = []
        
        # Run multiple times, skip first warmup
        for run_idx in range(args.num_runs + 1):
            for prompt in tqdm(prompts, desc=f"Run {run_idx+1}/{args.num_runs+1}"):
                generator.reset()
                
                torch.cuda.synchronize()
                start = time.perf_counter()
                
                output = generator.generate(prompt, max_new_tokens=args.max_new_tokens)
                
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                
                stats = generator.get_stats()
                
                # Skip first run (warmup)
                if run_idx > 0:
                    times.append(elapsed)
                    stats_list.append(stats)
        
        # Calculate average results
        avg_time = sum(times) / len(times)
        avg_tokens = sum(s['total_tokens'] for s in stats_list) / len(stats_list)
        avg_rounds = sum(s['total_rounds'] for s in stats_list) / len(stats_list)
        avg_nodes = sum(s['total_tree_nodes'] for s in stats_list) / avg_rounds if avg_rounds > 0 else 0
        avg_path = sum(s['avg_accepted_path_length'] for s in stats_list) / len(stats_list)
        avg_acceptance = sum(s.get('acceptance_rate', 0) for s in stats_list) / len(stats_list)
        throughput = avg_tokens / avg_time if avg_time > 0 else 0
        tpot_ms = (avg_time / avg_tokens) * 1000 if avg_tokens > 0 else 0
        
        results[variant_id] = {
            "name": variant_config["name"],
            "config": {
                "threshold": variant_config["threshold"],
                "max_nodes": variant_config["max_nodes"]
            },
            "metrics": {
                "throughput": round(throughput, 2),
                "tpot_ms": round(tpot_ms, 2),
                "avg_nodes_per_round": round(avg_nodes, 2),
                "avg_path_length": round(avg_path, 2),
                "acceptance_rate": round(avg_acceptance * 100, 1),
                "avg_time": round(avg_time, 3),
                "avg_tokens": round(avg_tokens, 1)
            }
        }
        
        print(f"\n{variant_config['name']} Results:")
        print(f"  Throughput: {throughput:.1f} tokens/s")
        print(f"  TPOT: {tpot_ms:.2f} ms")
        print(f"  Avg Nodes/Round: {avg_nodes:.1f}")
        print(f"  Avg Path Length: {avg_path:.2f}")
        print(f"  Acceptance Rate: {avg_acceptance*100:.1f}%")
    
    # Save results
    output_file = args.output
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary comparison
    print(f"\n{'=' * 70}")
    print("SUMMARY COMPARISON")
    print("=" * 70)
    print(f"{'Method':<30} {'Throughput':>12} {'TPOT':>10} {'Nodes':>8} {'Path':>8}")
    print("-" * 70)
    
    baseline_throughput = results.get("dynamic_prune", {}).get("metrics", {}).get("throughput", 1)
    
    for variant_id in ["no_prune", "static_prune", "dynamic_prune"]:
        res = results[variant_id]
        metrics = res['metrics']
        speedup = metrics['throughput'] / baseline_throughput if baseline_throughput > 0 else 1.0
        print(f"{res['name']:<30} {metrics['throughput']:>10.1f} t/s "
              f"{metrics['tpot_ms']:>8.2f} ms "
              f"{metrics['avg_nodes_per_round']:>6.1f} "
              f"{metrics['avg_path_length']:>6.2f}")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-model", required=True, help="Target model path")
    parser.add_argument("--draft-model", required=True, help="Draft model path")
    parser.add_argument("--depth", type=int, default=8, help="Tree depth (default: 8)")
    parser.add_argument("--branch", type=int, default=3, help="Branch factor (default: 3)")
    parser.add_argument("--max-new-tokens", type=int, default=500, help="Tokens to generate (default: 500)")
    parser.add_argument("--num-runs", type=int, default=4, help="Number of runs (default: 4, +1 warmup)")
    parser.add_argument("--output", default="results/ablation_pruning.json", help="Output file")
    args = parser.parse_args()
    
    run_ablation(args)
ABLATION_SCRIPT
        chmod +x spec_decode/ablation_pruning.py
        echo "✓ 消融实验脚本创建完成"
    fi
    
    python spec_decode/ablation_pruning.py \
        --target-model "$TARGET_MODEL" \
        --draft-model "$DRAFT_MODEL" \
        --depth 8 \
        --branch 3 \
        --max-new-tokens 500 \
        --num-runs 4 \
        --output "$RESULTS_DIR/exp3_ablation_pruning.json"
    
    echo ""
    echo -e "${GREEN}✓ 实验 3 完成!${NC}"
    echo "  输出文件: $RESULTS_DIR/exp3_ablation_pruning.json"
else
    echo -e "${YELLOW}⊘ 跳过实验 3${NC}"
fi

# ====================================================================
# Experiment 4: 序列长度扩展 (Scalability)
# ====================================================================
echo ""
echo -e "${GREEN}======================================================================"
echo "实验 4: 序列长度扩展测试 (Sequence Length Scaling)"
echo -e "======================================================================${NC}"
echo ""
echo "这个实验测试不同生成长度下的性能，验证方法的可扩展性"
echo ""
echo "测试配置 (使用各自的最优参数):"
echo "  • 100 tokens:  D=7, B=3, τ=0.03 (预期 1.43x)"
echo "  • 200 tokens:  D=7, B=3, τ=0.03 (预期 1.54x)"
echo "  • 300 tokens:  D=7, B=3, τ=0.03 (预期 1.60x)"
echo "  • 500 tokens:  D=8, B=3, τ=0.03 (预期 1.62x) ⭐ 最优"
echo "  • 1000 tokens: D=6, B=3, τ=0.05 (预期 1.71x)"
echo ""
echo "关键发现:"
echo "  • 生成长度越长，Tree-based方法的优势一般越明显"
echo "  • 500 tokens 达到最佳平衡点"
echo "  • 不同长度需要不同的树深度"
echo ""
echo "预计时间: ~60 分钟 (五个长度分别测试)"
echo "对应论文: Figure 3 或 Table 4 (扩展性分析)"
echo ""

read -p "运行实验 4? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo ""
    echo -e "${YELLOW}开始运行实验 4...${NC}"
    echo ""
    
    # Create length scaling script with optimal params for each length
    cat > /tmp/run_length_scaling.py << 'LENGTH_SCALING_SCRIPT'
#!/usr/bin/env python3
"""
Sequence Length Scaling Experiment
Tests performance across different generation lengths with optimal parameters.
Based on actual experimental results from papers/Tree_Speculative_Decoding_实验报告.md
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer
from spec_decode.core import TreeSpeculativeGeneratorV2

# Optimal parameters from actual experiments for each length
configs = [
    {"length": 100, "depth": 7, "branch": 3, "threshold": 0.03},
    {"length": 200, "depth": 7, "branch": 3, "threshold": 0.03},
    {"length": 300, "depth": 7, "branch": 3, "threshold": 0.03},
    {"length": 500, "depth": 8, "branch": 3, "threshold": 0.03},
    {"length": 1000, "depth": 6, "branch": 3, "threshold": 0.05},
]

target_model_path = sys.argv[1]
draft_model_path = sys.argv[2]
output_dir = sys.argv[3]

print("=" * 70)
print("Sequence Length Scaling Experiment")
print("=" * 70)
print("\nLoading models...")

tokenizer = AutoTokenizer.from_pretrained(target_model_path)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

target_model = AutoModelForCausalLM.from_pretrained(
    target_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
target_model.eval()

draft_model = AutoModelForCausalLM.from_pretrained(
    draft_model_path,
    torch_dtype=torch.float16,
    device_map="auto"
)
draft_model.eval()

test_prompt = "Write a detailed technical explanation about the development of large language models."

all_length_results = {}

for config in configs:
    length = config["length"]
    depth = config["depth"]
    branch = config["branch"]
    threshold = config["threshold"]
    
    print(f"\n{'=' * 70}")
    print(f"Testing {length} tokens")
    print(f"  Optimal params: D={depth}, B={branch}, τ={threshold}")
    print("=" * 70)
    
    # Test Baseline
    print("\n[1/2] Testing Baseline...")
    baseline_times = []
    for run in range(5):
        torch.cuda.empty_cache()
        input_ids = tokenizer(test_prompt, return_tensors="pt").input_ids.to("cuda")
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        with torch.inference_mode():
            outputs = target_model.generate(
                input_ids,
                max_new_tokens=length,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id
            )
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        if run > 0:  # Skip warmup
            baseline_times.append(elapsed)
    
    baseline_throughput = length / (sum(baseline_times) / len(baseline_times))
    print(f"  Baseline: {baseline_throughput:.1f} t/s")
    
    # Test Tree V2 with optimal params
    print("\n[2/2] Testing Tree V2...")
    generator = TreeSpeculativeGeneratorV2(
        target_model=target_model,
        draft_model=draft_model,
        tokenizer=tokenizer,
        tree_depth=depth,
        branch_factor=branch,
        probability_threshold=threshold,
        max_tree_nodes=128,
        device="cuda",
        use_compile=False
    )
    
    tree_times = []
    stats_list = []
    
    for run in range(5):
        generator.reset()
        torch.cuda.empty_cache()
        
        torch.cuda.synchronize()
        start = time.perf_counter()
        
        output = generator.generate(test_prompt, max_new_tokens=length)
        
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        
        if run > 0:  # Skip warmup
            tree_times.append(elapsed)
            stats_list.append(generator.get_stats())
    
    tree_throughput = length / (sum(tree_times) / len(tree_times))
    speedup = tree_throughput / baseline_throughput
    avg_acceptance = sum(s.get('acceptance_rate', 0) for s in stats_list) / len(stats_list)
    
    print(f"  Tree V2: {tree_throughput:.1f} t/s ({speedup:.2f}x speedup)")
    
    all_length_results[f"{length}_tokens"] = {
        "config": {
            "length": length,
            "depth": depth,
            "branch": branch,
            "threshold": threshold
        },
        "results": {
            "baseline_throughput": round(baseline_throughput, 2),
            "tree_throughput": round(tree_throughput, 2),
            "speedup": round(speedup, 2),
            "acceptance_rate": round(avg_acceptance * 100, 1)
        }
    }

# Save combined results
output_file = f"{output_dir}/exp4_length_scaling.json"
with open(output_file, 'w') as f:
    json.dump(all_length_results, f, indent=2)

# Print summary
print(f"\n{'=' * 70}")
print("SUMMARY")
print("=" * 70)
print(f"{'Length':<10} {'Baseline':>12} {'Tree V2':>12} {'Speedup':>10}")
print("-" * 70)

for length in [100, 200, 300, 500, 1000]:
    key = f"{length}_tokens"
    if key in all_length_results:
        res = all_length_results[key]["results"]
        print(f"{length:>5} tok {res['baseline_throughput']:>10.1f} t/s "
              f"{res['tree_throughput']:>10.1f} t/s {res['speedup']:>8.2f}x")

print(f"\n{'=' * 70}")
print(f"Results saved to: {output_file}")
print("=" * 70)
LENGTH_SCALING_SCRIPT
    
    python /tmp/run_length_scaling.py \
        "$TARGET_MODEL" \
        "$DRAFT_MODEL" \
        "$RESULTS_DIR"
    
    echo ""
    echo -e "${GREEN}✓ 实验 4 完成!${NC}"
    echo "  输出文件: $RESULTS_DIR/exp4_length_scaling.json"
else
    echo -e "${YELLOW}⊘ 跳过实验 4${NC}"
fi

# ====================================================================
# Summary
# ====================================================================
echo ""
echo -e "${GREEN}======================================================================"
echo "                    实验完成总结"
echo "======================================================================${NC}"
echo ""
echo "所有结果已保存到: $RESULTS_DIR"
echo ""

# List generated files
echo "生成的文件:"
for file in "$RESULTS_DIR"/*; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        filesize=$(du -h "$file" | cut -f1)
        echo "  ✓ $filename ($filesize)"
    fi
done

echo ""
echo -e "${YELLOW}下一步操作:${NC}"
echo ""
echo "1. 查看实验结果:"
echo "   cd $RESULTS_DIR"
echo "   cat exp1_main_comparison_500tokens.json | jq '.results[] | {method, avg_throughput, tpot_ms}'"
echo ""
echo "2. 生成论文图表:"
echo "   python papers/generate_all_figures.py --results-dir $RESULTS_DIR"
echo ""
echo "3. 开始写论文:"
echo "   cd papers/"
echo "   # 查看 PAPER_PLAN.md 获取详细的写作指南"
echo ""
echo -e "${GREEN}实验总结 (基于实际实验报告):${NC}"
echo "  • 实验 1: 主要性能对比 → 论文 Table 2"
echo "  • 实验 2: 参数影响分析 → 论文 Figure 2"
echo "  • 实验 3: 消融实验     → 论文 Table 3"
echo "  • 实验 4: 长度扩展性   → 论文 Figure 3/Table 4"
echo ""
echo -e "${BLUE}论文关键数据 (实际测试结果):${NC}"
echo "  🏆 Tree V2 (D=8, B=3, τ=0.03):  193.4 t/s (1.62x)"
echo "  🥈 HuggingFace Assisted:         161.9 t/s (1.36x)"
echo "  🥉 Linear K=6:                   133.1 t/s (1.11x)"
echo "  📊 Baseline:                     119.4 t/s (1.00x)"
echo ""
echo "  参数搜索最佳结果: 1.79x (500 tokens, D=8, B=3, τ=0.03)"
echo ""
echo -e "${GREEN}========================================== 完成 ==========================================${NC}"
echo ""
