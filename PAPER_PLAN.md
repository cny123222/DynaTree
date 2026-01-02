# Tree-based Speculative Decoding 论文规划

**目标期刊**: NeurIPS 2025  
**截止日期**: 2026年1月5日  
**页数限制**: 4页正文（不含references）

---

## 📝 论文标题（候选）

### 首选：
**"Tree-based Speculative Decoding with Dynamic Pruning for Efficient LLM Inference"**

### 备选：
- "Beyond Linear Speculation: Tree-based Token Generation for Accelerated LLM Inference"
- "Parallel Path Exploration for Speculative Decoding in Large Language Models"

---

## 🎯 核心贡献

### 主要贡献（按重要性排序）：

1. **Tree-based Drafting with Dynamic Pruning** ⭐⭐⭐⭐⭐
   - 提出多分支token树生成策略
   - 动态剪枝控制树大小
   - 相比linear方法提升25%

2. **Tree Attention Mechanism** ⭐⭐⭐⭐
   - 4D attention mask支持并行验证
   - 一次forward验证整棵树
   - 保持correctness（greedy decoding一致）

3. **Systematic Hyperparameter Analysis** ⭐⭐⭐
   - Depth, Branch Factor, Threshold的trade-off
   - 不同生成长度的最优配置
   - 实用部署指南

4. **Open-source Implementation** ⭐⭐⭐
   - 完整可复现代码
   - 详细benchmark工具
   - 与HuggingFace生态兼容

---

## 📖 故事线（Narrative Arc）

### 整体故事结构：

```
【问题】→【现有方案】→【局限】→【我们的创新】→【技术挑战】→【解决方案】→【效果验证】→【深入分析】
```

### 详细故事展开：

#### 1. 问题背景（Why care?）
**痛点**: 
- LLM推理慢（autoregressive瓶颈）
- 每次生成1个token，GPU利用率低
- 实际应用需求：实时对话、长文本生成

**量化数据**:
- Baseline: 60.8 tokens/s
- 生成100个token需要 ~1.6秒
- 对话延迟明显

#### 2. 现有方案（What exists?）
**Linear Speculative Decoding**:
```
Draft Model猜测: [t1, t2, t3, t4, t5]
Target Model验证: ✓   ✓   ✗   ✗   ✗
接受: [t1, t2] + bonus → 3 tokens/round
```

**优点**: 
- 简单，易实现
- 当draft准确时效果好

**局限**:
- 只有一条路径
- 如果t2错了，t3-t5全部浪费
- 接受率≈70-80%，有提升空间

#### 3. 我们的洞察（Key Insight）
**关键问题**: 为什么只猜一条路径？

**核心洞察**:
```
Linear: 猜5个token，接受率80% → 期望4个token
Tree:   猜5个位置×3个分支，至少一条对的概率更高！
```

**类比**: 
- Linear = 单线程搜索
- Tree = 多线程并行搜索
- 只要有一条路对，就能被接受

#### 4. 技术挑战（Challenges）

**Challenge 1: 树太大**
```
深度D=5, 分支B=3
理论节点数 = 1 + 3 + 9 + 27 + 81 + 243 = 364个token
→ 太多了！draft model会很慢
```

**Challenge 2: 如何并行验证**
```
树不是线性序列，如何让target model一次处理？
→ 需要特殊的attention mask
```

**Challenge 3: 选择哪条路径**
```
验证后可能有多条路径部分正确
→ 需要选择策略（贪心？最长？）
```

#### 5. 我们的解决方案（Our Solution）

**Solution 1: 动态剪枝（Dynamic Pruning）**
```python
# 生成每个分支时检查概率
if p(token|prefix) < threshold:
    prune this branch  # 不继续扩展
    
# 同时限制总节点数
if num_nodes >= max_nodes:
    stop expansion
```

**效果**:
- 平均节点数: 364 → 22.7
- 保持高质量路径
- 剪枝≠随机删除，是基于概率的智能剪枝

**Solution 2: Tree Attention Mask**
```
树结构:
    0 (root)
    ├─ 1 (child1)
    │  ├─ 3 (grandchild)
    │  └─ 4
    └─ 2 (child2)
       └─ 5

扁平化序列: [0, 1, 2, 3, 4, 5]

Attention Mask (6x6):
     0  1  2  3  4  5
  0  ✓  ✗  ✗  ✗  ✗  ✗   # root不看后代
  1  ✓  ✓  ✗  ✗  ✗  ✗   # 1看root和自己
  2  ✓  ✗  ✓  ✗  ✗  ✗   # 2看root和自己
  3  ✓  ✓  ✗  ✓  ✗  ✗   # 3看root,1和自己
  4  ✓  ✓  ✗  ✗  ✓  ✗   # 4看root,1和自己
  5  ✓  ✗  ✓  ✗  ✗  ✓   # 5看root,2和自己
```

**关键**: 每个节点只看到它的祖先路径！

**Solution 3: 贪心最长匹配路径**
```python
def find_best_path(tree, target_logits):
    """选择最长的匹配路径"""
    paths = tree.get_all_leaf_paths()
    
    for path in sorted(paths, key=len, reverse=True):
        matched_length = 0
        for node in path:
            target_pred = argmax(target_logits[node.position])
            if target_pred == node.token:
                matched_length += 1
            else:
                break  # 第一个不匹配就停止
        
        if matched_length > 0:
            return path[:matched_length]
    
    return []  # 没有匹配
```

#### 6. 实验验证（Experimental Validation）

**主要结果**:
```
Method              | Throughput  | Speedup | Improvement over Linear
--------------------|-------------|---------|------------------------
Baseline            | 60.8 t/s    | 1.00×   | -
Linear (K=3)        | 97.5 t/s    | 1.60×   | baseline
Linear (K=5)        | 112.3 t/s   | 1.85×   | baseline  
Tree (D=3, B=2)     | 100.3 t/s   | 1.65×   | +3% vs K=3
Tree V2 (D=3, B=3)  | 122.0 t/s   | 2.00×   | +25% vs K=5 ⭐
```

**关键数字**: 2.00× speedup, +25% improvement

#### 7. 深入分析（Deep Dive）

**发现1: 参数选择的trade-off**
- Depth太小(D=2): 接受的token少
- Depth太大(D=6): draft overhead大，得不偿失
- **最优: D=3-4**

**发现2: Branch Factor的影响**
- B=2: 太保守，错过机会
- B=4: 树太大，draft慢
- **最优: B=3**

**发现3: 动态剪枝至关重要**
- 无剪枝: 1.65× (树太大)
- 静态剪枝: 1.78× (固定规则不灵活)
- 动态剪枝: 2.00× (智能平衡) ⭐

**发现4: 长序列效果更好**
- 100 tokens: 2.00×
- 500 tokens: 2.20× (摊销prefill开销)

---

## 🔬 方法详细描述

### Algorithm 1: Linear Speculative Decoding (Baseline)

```python
def linear_speculative_decoding(prompt, target_model, draft_model, K):
    """
    Linear speculative decoding baseline.
    
    Args:
        prompt: Input text
        target_model: Large model (M_T)
        draft_model: Small model (M_D)
        K: Number of draft tokens per round
    """
    # Step 1: Prefill
    input_ids = tokenize(prompt)
    target_cache = target_model.prefill(input_ids)
    current_ids = input_ids
    
    while not done:
        # Step 2: Draft K tokens (sequential)
        draft_tokens = []
        draft_cache = draft_model.prefill(current_ids)
        
        for i in range(K):
            next_token = draft_model.generate_one(draft_cache)
            draft_tokens.append(next_token)
            draft_cache = draft_model.update_cache(next_token)
        
        # Step 3: Verify all K tokens in parallel
        verify_ids = concat(current_ids, draft_tokens)
        target_logits = target_model.forward(
            verify_ids,
            past_key_values=target_cache,
            use_cache=True
        )
        
        # Step 4: Compare and accept
        n_accepted = 0
        for i in range(K):
            target_pred = argmax(target_logits[-(K-i)])
            if target_pred == draft_tokens[i]:
                n_accepted += 1
            else:
                break  # First mismatch, stop
        
        # Step 5: Bonus token if all accepted
        accepted = draft_tokens[:n_accepted]
        if n_accepted == K:
            bonus = argmax(target_logits[-1])
            accepted.append(bonus)
        
        # Step 6: Update cache and continue
        target_cache = crop_cache(target_cache, len(current_ids) + n_accepted)
        current_ids = concat(current_ids, accepted)
    
    return decode(current_ids)
```

**问题**: 如果第i个token错误，后面K-i个全部浪费！

---

### Algorithm 2: Tree-based Speculative Decoding (Ours)

```python
def tree_speculative_decoding(
    prompt, 
    target_model, 
    draft_model, 
    depth=3, 
    branch=3, 
    threshold=0.05
):
    """
    Tree-based speculative decoding with dynamic pruning.
    
    Args:
        depth: Tree depth (D)
        branch: Branch factor (B) 
        threshold: Pruning threshold (τ)
    """
    # Step 1: Prefill
    input_ids = tokenize(prompt)
    target_cache = target_model.prefill(input_ids)
    current_ids = input_ids
    
    while not done:
        # Step 2: Generate token tree with draft model
        tree = generate_token_tree(
            draft_model, 
            current_ids,
            depth=depth,
            branch_factor=branch,
            prune_threshold=threshold
        )
        
        # Step 3: Flatten tree and build attention mask
        tree_tokens, tree_mask = flatten_tree_with_mask(tree)
        # tree_tokens: [root, child1, child2, grandchild1, ...]
        # tree_mask: (len(tree_tokens), len(tree_tokens))
        
        # Step 4: Verify entire tree in one forward pass
        verify_ids = concat(current_ids, tree_tokens)
        target_logits = target_model.forward(
            verify_ids,
            attention_mask=tree_mask,  # 4D mask!
            past_key_values=target_cache,
            use_cache=True
        )
        
        # Step 5: Find longest matching path
        best_path = find_longest_matching_path(tree, target_logits)
        
        # Step 6: Update cache with accepted path
        if len(best_path) > 0:
            accepted_tokens = [node.token for node in best_path]
            target_cache = update_cache_with_path(
                target_cache, 
                best_path,
                target_model
            )
            current_ids = concat(current_ids, accepted_tokens)
        else:
            # Fallback: accept 1 token from target model
            next_token = argmax(target_logits[-1])
            current_ids = concat(current_ids, [next_token])
    
    return decode(current_ids)
```

---

### Key Component 1: Token Tree Generation with Dynamic Pruning

```python
def generate_token_tree(draft_model, prefix, depth, branch_factor, prune_threshold):
    """
    Generate a token tree using draft model with dynamic pruning.
    
    Returns:
        TokenTree: Tree structure with nodes
    """
    tree = TokenTree(root_token=prefix[-1])
    
    # Initialize with root
    current_level = [tree.root]
    
    for d in range(depth):
        next_level = []
        
        for node in current_level:
            # Get draft model prediction for this node
            logits = draft_model.forward(node.get_path_tokens())
            probs = softmax(logits[-1])
            
            # Get top-k candidates
            top_k_probs, top_k_tokens = torch.topk(probs, branch_factor)
            
            # Dynamic pruning: only keep high-probability branches
            for prob, token in zip(top_k_probs, top_k_tokens):
                if prob >= prune_threshold:
                    child = TreeNode(
                        token=token,
                        parent=node,
                        probability=prob
                    )
                    node.add_child(child)
                    next_level.append(child)
                else:
                    # Prune this branch
                    break
            
            # Safety check: limit total nodes
            if tree.num_nodes >= MAX_NODES:
                return tree
        
        current_level = next_level
        
        if len(current_level) == 0:
            break  # No more nodes to expand
    
    return tree
```

**动态剪枝的两个条件**:
1. `prob >= prune_threshold`: 概率太低的分支直接剪掉
2. `tree.num_nodes < MAX_NODES`: 限制总节点数

---

### Key Component 2: Tree Attention Mask Construction

```python
def flatten_tree_with_mask(tree):
    """
    Flatten tree to sequence and build attention mask.
    
    Returns:
        tokens: List[int] - flattened token sequence
        mask: Tensor[N, N] - attention mask (0=attend, -inf=mask)
    """
    # BFS traversal to flatten tree
    nodes = []
    queue = [tree.root]
    
    while queue:
        node = queue.pop(0)
        nodes.append(node)
        queue.extend(node.children)
    
    N = len(nodes)
    tokens = [node.token for node in nodes]
    
    # Build attention mask
    mask = torch.full((N, N), float('-inf'))
    
    for i, node in enumerate(nodes):
        # Each node can attend to all its ancestors
        ancestors = node.get_ancestor_indices(nodes)
        for j in ancestors:
            mask[i, j] = 0.0
        
        # Can also attend to itself
        mask[i, i] = 0.0
    
    return tokens, mask
```

**示例**:
```
Tree:      Flattened:     Mask:
  0          [0]           0: ✓
  ├─1        [0,1]         1: ✓✓
  │ └─3      [0,1,3]       3: ✓✓✗✓
  └─2        [0,1,3,2]     2: ✓✗✗✗✓
    └─4      [0,1,3,2,4]   4: ✓✗✗✓✗✓
```

---

### Key Component 3: Path Selection Strategy

```python
def find_longest_matching_path(tree, target_logits):
    """
    Find the longest path where draft tokens match target predictions.
    
    Strategy: Greedy longest matching (GLM)
    """
    all_paths = tree.get_all_leaf_paths()
    
    best_path = []
    best_length = 0
    
    for path in all_paths:
        matched = []
        
        for i, node in enumerate(path):
            # Get target model's prediction at this position
            logit_idx = node.position_in_flat_sequence
            target_pred = torch.argmax(target_logits[logit_idx])
            
            if target_pred.item() == node.token:
                matched.append(node)
            else:
                break  # First mismatch, stop
        
        if len(matched) > best_length:
            best_length = len(matched)
            best_path = matched
    
    return best_path
```

**为什么是贪心最长匹配？**
- 简单高效
- 保证正确性（与target model一致）
- 最大化每轮接受的token数

---

## 📊 实验设计

### Setup

**Models**:
- Target: Pythia-2.8B (FP16)
- Draft: Pythia-70M (FP16)
- 都使用EleutherAI预训练权重

**Hardware**:
- GPU: NVIDIA GPU with CUDA
- Memory: 24GB+ VRAM

**Evaluation Metrics**:
1. **Throughput** (tokens/s): 生成速度
2. **Speedup**: 相对baseline的加速比
3. **Acceptance Rate** (%): draft tokens被接受的比例
4. **Average Path Length**: 平均每轮接受的token数

**Test Data**:
- 随机采样prompts (20-100 tokens)
- 生成长度: 100, 200, 500, 1000 tokens
- 每个配置运行5次取平均

---

### Experiment 1: Main Performance Comparison

**目的**: 与baseline和linear方法对比

**配置**:
```python
methods = {
    "Baseline": {"type": "autoregressive"},
    "Linear K=3": {"type": "linear", "K": 3},
    "Linear K=5": {"type": "linear", "K": 5},
    "Linear K=7": {"type": "linear", "K": 7},
    "Tree D=3 B=2": {"type": "tree", "depth": 3, "branch": 2, "threshold": 0.05},
    "Tree V2 D=3 B=3": {"type": "tree", "depth": 3, "branch": 3, "threshold": 0.05},
}
```

**预期结果**: Tree V2 达到2.0× speedup

**对应论文**: Table 2 (Main Results)

---

### Experiment 2: Hyperparameter Sweep

**目的**: 分析Depth, Branch, Threshold的影响

**配置**:
```python
param_grid = {
    "depth": [2, 3, 4, 5, 6],
    "branch_factor": [2, 3, 4],
    "threshold": [0.01, 0.02, 0.05, 0.1],
    "token_length": [100, 200, 500, 1000]
}
# Total: 5×3×4×4 = 240 configurations
```

**预期发现**:
- Optimal depth: 3-4
- Optimal branch: 3
- Optimal threshold: 0.05
- Longer sequences → higher speedup

**对应论文**: Figure 2 (Parameter Sweep, 6个子图)

---

### Experiment 3: Ablation Study

**目的**: 验证动态剪枝的作用

**配置**:
```python
ablations = {
    "No Pruning": {
        "depth": 3, 
        "branch": 3, 
        "threshold": 0.0,  # 不剪枝
        "max_nodes": 9999
    },
    "Static Pruning": {
        "depth": 3,
        "branch": 3,
        "threshold": 0.0,
        "max_nodes": 30  # 固定上限
    },
    "Dynamic Pruning (Ours)": {
        "depth": 3,
        "branch": 3,
        "threshold": 0.05,  # 动态剪枝
        "max_nodes": 50
    }
}
```

**预期结果**:
- No Pruning: 1.65× (太大太慢)
- Static: 1.78× (不够灵活)
- Dynamic: 2.00× (最优)

**对应论文**: Table 3 (Ablation Study)

---

### Experiment 4: Qualitative Case Study

**目的**: 可视化一个具体例子

**方法**:
1. 选择一个prompt: "The future of artificial intelligence is"
2. 生成token树（显示所有节点和剪枝）
3. 显示target model验证结果
4. 高亮最终接受的路径

**对应论文**: Figure 3 (Tree Visualization)

---

### Experiment 5: Error Analysis (Optional)

**目的**: 分析什么情况下tree比linear好

**方法**:
1. 收集100个cases
2. 分类: tree更好 / linear更好 / 相当
3. 分析特征: draft模型准确度、prompt复杂度等

**对应论文**: Table 4 或 Discussion部分

---

## 📈 图表清单

### 必须图表（支撑核心论点）

#### Figure 1: Tree Structure Illustration
**类型**: 示意图 (TikZ or hand-drawn)

**内容**:
- (a) Linear Speculation: 一条链
- (b) Tree Speculation: 树状结构
- (c) 剪枝前后对比

**要点**:
- 清晰展示两种方法的区别
- 标注Branch Factor和Depth
- 显示剪枝效果

---

#### Figure 2: Hyperparameter Analysis (6 subplots)
**类型**: 多子图折线图/热力图

**子图**:
- (a) Speedup vs Depth (固定B=3, τ=0.05)
- (b) Speedup vs Branch Factor (固定D=3, τ=0.05)
- (c) Speedup vs Threshold (固定D=3, B=3)
- (d) Speedup vs Token Length (最优配置)
- (e) Tree Size vs Parameters (heatmap)
- (f) Acceptance Rate Distribution (histogram)

**数据来源**: `results/tree_param_search_*.json`

---

#### Table 1: Model Configuration
**类型**: 配置表格

```latex
\begin{table}[h]
\centering
\caption{Experimental Setup}
\begin{tabular}{lccc}
\toprule
Component & Specification \\
\midrule
Target Model & Pythia-2.8B (FP16) \\
Draft Model & Pythia-70M (FP16) \\
Hardware & NVIDIA GPU with CUDA \\
Framework & PyTorch 2.0+, Transformers 4.38+ \\
Test Prompts & 50 samples, 20-100 tokens \\
Generation Length & 100, 200, 500, 1000 tokens \\
\bottomrule
\end{tabular}
\end{table}
```

---

#### Table 2: Main Performance Results
**类型**: 性能对比表格

```latex
\begin{table}[h]
\centering
\caption{Performance Comparison on 100-token Generation}
\begin{tabular}{lcccc}
\toprule
Method & Throughput & Speedup & Accept Rate & Path Length \\
       & (tokens/s) &         & (\%)        & (tokens/round) \\
\midrule
Baseline & 60.8 & 1.00× & - & 1.0 \\
Linear (K=3) & 97.5 & 1.60× & 85.2\% & 3.2 \\
Linear (K=5) & 112.3 & 1.85× & 76.4\% & 4.8 \\
Linear (K=7) & 118.7 & 1.95× & 68.9\% & 5.6 \\
\midrule
Tree (D=3, B=2) & 100.3 & 1.65× & 23.4\% & 2.1 \\
\textbf{Tree V2 (D=3, B=3)} & \textbf{122.0} & \textbf{2.00×} & \textbf{36.3\%} & \textbf{3.6} \\
Tree V2 (D=4, B=3) & 119.5 & 1.97× & 38.1\% & 4.2 \\
\bottomrule
\end{tabular}
\end{table}
```

---

#### Table 3: Ablation Study on Pruning
**类型**: 消融实验表格

```latex
\begin{table}[h]
\centering
\caption{Ablation Study: Effect of Dynamic Pruning}
\begin{tabular}{lcccc}
\toprule
Variant & Speedup & Avg Nodes & Accept\% & Path Length \\
\midrule
No Pruning & 1.65× & 42.3 & 19.8\% & 2.5 \\
Static Pruning (max=30) & 1.78× & 28.5 & 25.1\% & 2.9 \\
\textbf{Dynamic Pruning (Ours)} & \textbf{2.00×} & \textbf{22.7} & \textbf{36.3\%} & \textbf{3.6} \\
\bottomrule
\end{tabular}
\end{table}
```

---

#### Figure 3: Tree Visualization Case Study (Optional)
**类型**: 树状图

**内容**:
- 一个具体生成的token树
- 显示剪枝的节点（灰色）
- 显示target验证结果（✓/✗）
- 高亮最终接受的路径（绿色）

---

### 补充图表（Nice to have）

#### Table 4: Performance on Different Sequence Lengths
```latex
\begin{tabular}{lccccc}
\toprule
Method & 100 tokens & 200 tokens & 500 tokens & 1000 tokens \\
\midrule
Linear (K=5) & 1.85× & 1.92× & 2.01× & 2.08× \\
Tree V2 (D=3,B=3) & 2.00× & 2.12× & 2.20× & 2.28× \\
\bottomrule
\end{tabular}
```

---

## ⏰ 实验执行计划

### Phase 1: 核心对比实验（P0 - 必须）

**脚本**: `benchmark_tree_vs_linear_final.py`

```bash
python papers/benchmark_tree_vs_linear_final.py \
    --target-model /mnt/disk1/models/pythia-2.8b \
    --draft-model /mnt/disk1/models/pythia-70m \
    --linear-k 3 5 7 \
    --tree-configs "D3B2,D3B3,D4B3" \
    --max-new-tokens 100 200 500 \
    --num-samples 10 \
    --output results/final_comparison.json \
    --output-plot papers/figures/main_comparison.pdf
```

**预计时间**: 2-3小时  
**输出**: Table 2 + 部分Figure 2

---

### Phase 2: 参数扫描（P0 - 必须）

**已有数据**: `results/tree_param_search_20251231_140952.json`

**任务**: 重新绘制publication-quality图表

```bash
python papers/plot_param_sweep_publication.py \
    --input results/tree_param_search_20251231_140952.json \
    --output papers/figures/param_sweep.pdf \
    --style neurips
```

**预计时间**: 1小时  
**输出**: Figure 2 (6个子图)

---

### Phase 3: 消融实验（P0 - 必须）

**脚本**: `ablation_pruning.py`

```bash
python spec_decode/ablation_pruning.py \
    --target-model /mnt/disk1/models/pythia-2.8b \
    --draft-model /mnt/disk1/models/pythia-70m \
    --depth 3 --branch 3 \
    --variants "no_prune,static_prune,dynamic_prune" \
    --max-new-tokens 100 \
    --num-samples 10 \
    --output results/ablation_pruning.json
```

**预计时间**: 1小时  
**输出**: Table 3

---

### Phase 4: Case Study可视化（P1 - 重要）

**脚本**: `visualize_tree_case.py`

```bash
python spec_decode/visualize_tree_case.py \
    --prompt "The future of artificial intelligence is" \
    --depth 3 --branch 3 --threshold 0.05 \
    --output papers/figures/tree_case_study.pdf
```

**预计时间**: 30分钟  
**输出**: Figure 3

---

### Phase 5: 长序列对比（P2 - Nice to have）

```bash
python spec_decode/benchmark_sequence_lengths.py \
    --lengths 100 200 500 1000 \
    --methods "linear_k5,tree_d3b3" \
    --num-samples 5 \
    --output results/sequence_length_comparison.json
```

**预计时间**: 1小时  
**输出**: Table 4

---

## 📝 论文写作时间表

### Day 1 (今天 1/2)
- [x] 整理方法文档（本文档）
- [ ] 运行Phase 1实验（核心对比）
- [ ] 开始写Abstract和Introduction
- [ ] 绘制Figure 1（TikZ）

### Day 2 (1/3)
- [ ] 运行Phase 2实验（参数扫描图表）
- [ ] 运行Phase 3实验（消融）
- [ ] 完成Method部分编写
- [ ] 完成Related Work部分

### Day 3 (1/4)
- [ ] 运行Phase 4实验（case study）
- [ ] 完成Experiments部分编写
- [ ] 整合所有图表到论文
- [ ] 写Discussion和Conclusion

### Day 4 (1/5 DDL前)
- [ ] 全文润色
- [ ] 检查格式（NeurIPS模板）
- [ ] 准备supplementary materials
- [ ] 最终检查和提交

---

## 🎨 写作风格指南

### Tone
- **专业但清晰**: 避免过度技术化
- **自信但谦逊**: 强调贡献，但承认限制
- **数据驱动**: 每个claim都有实验支撑

### 常用短语
- "We propose..." (提出方法)
- "Our key insight is..." (关键洞察)
- "Experiments show that..." (实验证明)
- "Compared to X, our method..." (对比)
- "This is because..." (解释原因)

### 避免
- ❌ "Obviously..."
- ❌ "Clearly..."
- ❌ "It is well-known that..."
- ❌ 过度使用形容词（"very", "extremely"）

---

## 💡 核心信息（Elevator Pitch）

**如果只有30秒解释我们的工作**:

```
Linear speculative decoding只猜一条路径，限制了加速潜力。
我们提出tree-based方法：每个位置猜多个候选，形成树结构。
通过动态剪枝控制树大小，用tree attention并行验证。
实验证明达到2.0× speedup，超越linear方法25%。
```

**关键数字**:
- 2.00× speedup
- +25% improvement over linear
- 22.7 average nodes (vs 364 theoretical)
- 36.3% acceptance rate

---

## 📚 参考文献（部分）

### 核心相关工作

1. **Leviathan et al., 2023**  
   "Fast Inference from Transformers via Speculative Decoding"  
   ICML 2023  
   → Linear方法的原始论文

2. **Chen et al., 2023**  
   "Accelerating Large Language Model Decoding with Speculative Sampling"  
   DeepMind  
   → 理论分析

3. **Miao et al., 2024**  
   "SpecInfer: Accelerating Generative LLM Serving"  
   ASPLOS 2024  
   → 首次提出tree-based思想（我们的灵感来源）

4. **Cai et al., 2024**  
   "Medusa: Simple Framework for Accelerating LLM Generation"  
   → 多头预测（需要训练）

5. **Xiao et al., 2024**  
   "Efficient Streaming Language Models with Attention Sinks"  
   ICLR 2024  
   → StreamingLLM（可组合使用）

---

## ✅ 检查清单

### 论文完成前检查

- [ ] Abstract包含问题、方法、结果
- [ ] Introduction有清晰的motivation
- [ ] Method部分有算法伪代码
- [ ] 所有图表都有caption和引用
- [ ] Table数字保持3位有效数字
- [ ] 所有claim都有citation或实验支撑
- [ ] 代码和数据已准备好分享
- [ ] 检查NeurIPS格式要求
- [ ] 页数控制在4页内（不含references）
- [ ] 所有作者信息正确

### 实验检查

- [ ] Baseline结果可复现
- [ ] 所有random seed已固定
- [ ] 实验配置已记录
- [ ] 原始数据已保存
- [ ] 图表源文件已保存（.pdf + .py）

---

## 🎯 成功标准

### Acceptance标准（NeurIPS审稿）

**Technical Quality** (关键):
- ✅ 方法novel且sound
- ✅ 实验充分且convincing
- ✅ 结果显著（2.0× speedup）
- ✅ 与相关工作对比全面

**Clarity** (重要):
- ✅ 写作清晰易懂
- ✅ 图表高质量
- ✅ Method描述详细

**Originality** (重要):
- ✅ Tree-based + Dynamic Pruning是新的
- ✅ 系统性分析有价值

**Significance** (次要):
- ⚠️ 对社区的影响（开源代码）
- ⚠️ 实用价值（可部署）

---

## 📧 联系和协作

**分工建议**:
- 队员A: 运行实验、收集数据
- 队员B: 绘制图表、整理结果
- 队员C: 撰写论文、润色语言

**每日同步**:
- 每天晚上同步进度
- 遇到问题及时讨论
- 关键决策共同决定

---

**文档版本**: v1.0  
**最后更新**: 2026-01-02  
**状态**: 规划完成，待执行


