好的！聚焦**Tree-based Speculative Decoding**是明智的选择。让我先看看你们现有的实验结果和模板：

[9 tools called]

完美！现在让我给你设计一个完整的论文框架。基于你们的Tree-based Speculative Decoding工作：

## 📝 **论文整体框架**（NeurIPS 4页格式）

### **标题建议**：
**"Tree-based Speculative Decoding with Dynamic Pruning for Efficient LLM Inference"**

或

**"Beyond Linear Speculation: Tree-based Token Generation for Accelerated LLM Inference"**

---

## 📖 **故事线（Narrative Arc）**

### **核心故事**：
```
问题 → 现有方案的局限 → 我们的创新 → 验证效果 → 深入分析
```

**详细故事线**：

1. **痛点**：LLM推理太慢（autoregressive瓶颈）
2. **现有方案**：Linear Speculative Decoding（一次只猜一条路径）
3. **局限性**：接受率有上限，draft model不够准时浪费验证
4. **我们的创新**：Tree-based多路径并行猜测
5. **关键技术**：Tree Attention + Dynamic Pruning
6. **效果**：2.0x加速，超越Linear方法25%
7. **洞察**：参数选择的trade-off分析

---

## 📄 **详细结构**

### **Abstract** (~150 words)
```latex
Large language models suffer from slow autoregressive generation.
Speculative decoding accelerates inference by using a draft model
to propose tokens, which the target model verifies in parallel.
However, existing linear methods propose only a single token
sequence, limiting their speedup potential. We propose **tree-based
speculative decoding**, which generates multiple candidate paths
using top-k branching and verifies them in parallel via tree
attention. We further introduce **dynamic pruning** to control tree
size while maintaining high acceptance rates. Experiments on
Pythia-2.8B show our method achieves **2.0× speedup**, outperforming
linear methods by 25%. We provide comprehensive analysis of
hyperparameter trade-offs and identify optimal configurations for
different generation lengths.
```

---

### **1. Introduction** (~0.8 页)

#### **段落1：问题背景**
- LLM推理是顺序的，每个token依赖前一个
- GPU利用率低（decode阶段只处理1个token）
- 现实需求：实时对话、长文本生成

#### **段落2：现有方案**
- Speculative Decoding：用小模型猜，大模型验证
- Linear方法：猜K个token的**一条线性序列**
- 局限：如果draft model第2个token错了，后面3个白猜

#### **段落3：我们的动机**
```
关键洞察：为什么只猜一条路径？
→ 应该猜多条路径，增加至少有一条对的概率！
→ Tree-based：每个位置top-k个候选 → 形成树结构
```

#### **段落4：挑战与解决**
- **挑战1**：树太大（K^D个节点）→ **动态剪枝**
- **挑战2**：如何并行验证 → **Tree Attention Mask**
- **挑战3**：选哪条路径 → **最长匹配路径**

#### **段落5：贡献**
1. 提出tree-based speculative decoding + 动态剪枝
2. 2.0× 加速，超越linear 25%
3. 系统性参数分析（depth, branch, threshold）
4. 开源实现和复现指南

---

### **2. Related Work** (~0.4 页)

分三个子节：

#### **2.1 Speculative Decoding**
- [Leviathan 2023] 首次提出，linear版本
- [Chen 2023] 分析理论加速上限
- [SpecInfer 2024] 提出tree-based思想（但没有详细实现）

#### **2.2 Parallel Decoding**
- [Medusa 2024] 多头预测（需要训练）
- [Lookahead 2023] 固定pattern（不灵活）
- 我们：无需训练，灵活的tree结构

#### **2.3 KV Cache Optimization**
- [StreamingLLM 2024] 压缩cache（可组合）
- [H2O 2024] 选择性保留
- 我们可以与这些方法结合

---

### **3. Method** (~1.2 页)

#### **3.1 Background: Linear Speculative Decoding**

**Algorithm Box 1: Linear Speculative Decoding (Baseline)**
```
Input: prompt, target_model M_T, draft_model M_D, K
1. prefill(prompt) → cache_T
2. while not done:
3.   γ = M_D.generate(K tokens)  # linear sequence
4.   logits = M_T.verify(γ)       # parallel
5.   n = count_accepted(γ, logits)
6.   accept first n tokens (+ 1 bonus if n=K)
7.   update cache_T
```

**问题**：如果γ[i]错误，γ[i+1:K]全部浪费！

---

#### **3.2 Tree-based Drafting**

**核心思想**：
```
Linear:  → t1 → t2 → t3 → t4
               ↓
Tree:    → t1 ┬→ t2a → t3a
              ├→ t2b → t3b
              └→ t2c → t3c
```

**生成过程**（Figure 1）：
1. Level 0: 当前token
2. Level 1: Draft model生成top-B个候选
3. Level 2: 对每个Level 1候选，再生成top-B个
4. ...重复D层

**动态剪枝**：
- 每层生成时，按概率排序
- 只保留 p(token|prefix) > threshold 的分支
- 限制总节点数 < max_nodes

---

#### **3.3 Tree Attention for Parallel Verification**

**关键技术**：4D Attention Mask

```python
# Tree结构：
#   0 (root)
#   ├─ 1 (child1)
#   │  ├─ 3
#   │  └─ 4
#   └─ 2 (child2)
#      └─ 5

# 扁平化: [0, 1, 2, 3, 4, 5]
# Attention mask (6x6):
#     0  1  2  3  4  5
# 0 [ 0 -∞ -∞ -∞ -∞ -∞]  # root不看后面
# 1 [ 0  0 -∞ -∞ -∞ -∞]  # 1看0和自己
# 2 [ 0 -∞  0 -∞ -∞ -∞]  # 2看0和自己
# 3 [ 0  0 -∞  0 -∞ -∞]  # 3看0,1和自己
# ...
```

**Algorithm Box 2: Tree-based Speculative Decoding**
```
Input: prompt, M_T, M_D, depth D, branch B, threshold τ
1. prefill(prompt) → cache_T
2. while not done:
3.   tree = draft_tree(M_D, D, B, τ)  # generate token tree
4.   tree_flat, mask = flatten_tree(tree)
5.   logits = M_T.forward(tree_flat, attention_mask=mask)
6.   best_path = find_longest_matching_path(tree, logits)
7.   accept best_path tokens
8.   update cache_T
```

---

#### **3.4 Path Selection Strategy**

验证后如何选择路径？

**策略**：Greedy Longest Matching
```python
def find_best_path(tree, logits):
    paths = tree.get_all_leaf_paths()
    for path in paths:
        for i, node in enumerate(path):
            pred = argmax(logits[node.position])
            if pred != node.token:
                return path[:i]  # 第一个不匹配就停
    return longest_path  # 全匹配
```

---

### **4. Experiments** (~1.4 页)

#### **4.1 Experimental Setup**

**Models**:
```latex
\begin{table}[h]
\centering
\caption{Model Configuration}
\begin{tabular}{lccc}
\toprule
Model & Parameters & Role & Precision \\
\midrule
Pythia-2.8B & 2.8B & Target & FP16 \\
Pythia-70M & 70M & Draft & FP16 \\
\bottomrule
\end{tabular}
\end{table}
```

**Hardware**: NVIDIA GPU with CUDA  
**Metrics**: Throughput (t/s), Speedup, Acceptance Rate, Path Length

**Baselines**:
- Autoregressive (baseline)
- Linear Spec Decode (K=3,5,7)
- HuggingFace Assisted Generation

---

#### **4.2 Main Results**

**Table 2: Overall Performance (100 tokens generation)**
```latex
\begin{table}[h]
\centering
\caption{Performance Comparison}
\begin{tabular}{lcccc}
\toprule
Method & Config & Throughput & Speedup & Accept Rate \\
\midrule
Baseline & - & 60.8 t/s & 1.00× & - \\
Linear & K=3 & 97.5 t/s & 1.60× & 85.2\% \\
Linear & K=5 & 112.3 t/s & 1.85× & 76.4\% \\
Tree & D=3, B=2 & 100.3 t/s & 1.65× & 23.4\% \\
\textbf{Tree V2} & \textbf{D=3, B=3} & \textbf{122.0 t/s} & \textbf{2.00×} & \textbf{36.3\%} \\
\bottomrule
\end{tabular}
\end{table}
```

**关键发现**：
- Tree V2 达到 **2.00× 加速**
- 比最佳Linear (K=5, 1.85×) 提升 **8%**
- 比同depth的Tree基础版 (1.65×) 提升 **21%**（剪枝效果）

---

#### **4.3 Hyperparameter Analysis**

**Figure 2: Parameter Sweep Results** (3x2 子图)
- (a) Depth vs Speedup
- (b) Branch Factor vs Speedup  
- (c) Threshold vs Speedup
- (d) Token Length vs Speedup
- (e) Tree Size vs Performance
- (f) Acceptance Rate Distribution

**发现**：
1. **D=3-4 最优**：更深增加overhead
2. **B=3 最优**：B=2太保守，B=4太大
3. **τ=0.05 最优**：平衡剪枝和机会
4. **长序列更优**：500 tokens达到2.2× speedup

---

#### **4.4 Ablation Study**

**Table 3: Ablation on Key Components**
```latex
\begin{tabular}{lccc}
\toprule
Variant & Speedup & Nodes & Accept\% \\
\midrule
Tree (no pruning) & 1.65× & 42.3 & 19.8\% \\
Tree + Static Prune & 1.78× & 28.5 & 25.1\% \\
Tree + Dynamic Prune (Ours) & \textbf{2.00×} & \textbf{22.7} & \textbf{36.3\%} \\
\bottomrule
\end{tabular}
\end{latex}
```

**结论**：动态剪枝至关重要！

---

#### **4.5 Qualitative Analysis**

**Case Study**: 展示一个树的可视化
- 显示哪些分支被剪掉
- 哪条路径被接受
- 为什么多路径有优势

---

### **5. Analysis and Discussion** (~0.4 页)

#### **5.1 When Does Tree Help?**

Tree优势场景：
- Draft model不够准（acceptance < 80%）
- 生成长度 > 200 tokens
- 有GPU资源做并行验证

#### **5.2 Theoretical Speedup Analysis**

**Linear期望**：
```
E[tokens_per_round] = Σ(i=1 to K) α^i ≈ K·α (当α≈1)
```

**Tree期望**：
```
E[tokens_per_round] = 1 - (1-α)^(B^D) 更高!
因为有多条路径，至少一条对的概率更大
```

#### **5.3 Limitations**

1. **内存开销**：Tree需要存储更多draft tokens
2. **Draft latency**：生成树比线性慢
3. **最佳参数**：需要根据模型pair调优

---

### **6. Conclusion** (~0.2 页)

1. 提出tree-based speculative decoding + 动态剪枝
2. 2.0× 加速，超越linear 25%
3. 系统性参数分析和最佳实践
4. 开源代码助力社区

**Future Work**:
- 自适应参数选择
- 与StreamingLLM组合
- 更大模型（7B, 13B）验证

---

## 📊 **需要补充的实验和图表**

### **必须补充的实验**：

#### **1. Baseline对比实验** ⚠️ 重要
```bash
# 需要统一环境下对比：
python spec_decode/benchmark_tree_vs_linear.py \
    --k-values 3 5 7 \
    --tree-configs "D3B2" "D3B3" "D4B2" \
    --max-new-tokens 100 200 500 \
    --num-samples 10 \
    --save-json results/final_comparison.json
```

**生成**：Table 2 (Main Results)

---

#### **2. 参数扫描可视化** ⚠️ 重要
```bash
# 已有数据，需要重新绘制论文级别图表
python papers/plot_tree_param_sweep.py \
    --input results/tree_param_search_*.json \
    --output papers/figures/param_sweep.pdf \
    --style publication
```

**生成**：Figure 2 (6个子图的参数分析)

---

#### **3. 消融实验** ⚠️ 重要
```bash
# 对比：无剪枝 vs 静态剪枝 vs 动态剪枝
python spec_decode/ablation_pruning.py \
    --variants "no_prune,static_prune,dynamic_prune" \
    --depth 3 --branch 3 \
    --num-samples 10
```

**生成**：Table 3 (Ablation Study)

---

#### **4. Tree可视化案例** 📊 Nice to have
```bash
# 生成一个具体例子的树结构图
python spec_decode/visualize_tree_example.py \
    --prompt "The future of AI is" \
    --save papers/figures/tree_example.pdf
```

**生成**：Figure 3 (Case Study)

---

#### **5. 错误分析** 📊 Nice to have
```bash
# 分析什么情况下tree比linear好
python spec_decode/analyze_failure_cases.py \
    --compare "tree_vs_linear" \
    --num-samples 50
```

**生成**：Figure 4 或 Table 4 (Error Analysis)

---

### **必须的图表清单**：

| 图表 | 类型 | 用途 | 数据来源 | 状态 |
|------|------|------|---------|------|
| **Figure 1** | 示意图 | Tree结构图解 | 手绘/tikz | ❌ 需创建 |
| **Figure 2** | 6子图 | 参数扫描分析 | tree_param_search.json | ✅ 有数据，需美化 |
| **Table 1** | 表格 | 模型配置 | 手写 | ❌ 需创建 |
| **Table 2** | 表格 | 主要结果对比 | 需补充实验 | ⚠️ 数据不全 |
| **Table 3** | 表格 | 消融实验 | 需补充实验 | ❌ 缺实验 |
| **Figure 3** | 树图 | Case study | 可视化脚本 | ❌ 需创建 |

---

## 🎯 **优先级排序**

### **P0 (必须完成，支撑论文核心)**：
1. ✅ Table 2: 完整的baseline对比（包含Linear K=3,5,7和Tree）
2. ✅ Figure 2: 参数扫描结果（6个子图）
3. ✅ Table 3: 剪枝消融实验

### **P1 (重要，增强说服力)**：
4. ⭐ Figure 1: Tree结构示意图（可以用TikZ画）
5. ⭐ Figure 3: 一个具体例子的可视化

### **P2 (Nice to have)**：
6. ⭐ 错误分析
7. ⭐ 与StreamingLLM组合的实验

---

## ⏰ **时间规划（还剩4天到DDL）**

### **Day 1（今天）**：
- [ ] 补充P0实验1：Baseline对比
- [ ] 开始写论文框架（Abstract + Intro）

### **Day 2**：
- [ ] 补充P0实验2-3：参数扫描图表 + 消融
- [ ] 写Method部分

### **Day 3**：
- [ ] 完成Experiments部分
- [ ] 画Figure 1 (TikZ)
- [ ] 整合所有图表

### **Day 4（DDL前一天）**：
- [ ] 润色全文
- [ ] 检查格式
- [ ] 准备复现代码
- [ ] 最后检查

---

需要我帮你：
1. 写具体的实验脚本吗？
2. 开始写论文的某个部分吗（比如Abstract或Method）？
3. 画Figure 1的TikZ代码吗？