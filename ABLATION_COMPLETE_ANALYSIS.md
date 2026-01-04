# 完整消融实验数据分析

## ✅ **数据齐全！**

所有数据来源于WikiText-2数据集，500 tokens生成：

### **数据来源**

1. **Baseline & Linear K=6**: `results/不同生成token长度性能对比/wikitext_benchmark_500tokens.json`
2. **Tree (无剪枝 & 有剪枝)**: `results/剪枝消融实验结果.json`

### **Baseline一致性**
- WikiText benchmark: 127.9 t/s
- 剪枝消融实验: 128.6 t/s
- 差异: 0.7 t/s (0.5%) ✅ 可接受

---

## 📊 **完整消融实验数据（4步）**

使用剪枝消融实验的Baseline (128.6 t/s) 作为基准：

### **Step 1: Baseline (AR only)**
```
Throughput: 128.6 t/s
Speedup: 1.00×
```
- 纯自回归解码
- 无draft model，无speculation

### **Step 2: + Draft Model (Linear K=6)**
```
Throughput: 174.2 t/s
Speedup: 1.35×
Gain: +35.5%
```
- 添加draft model和线性投机解码
- **贡献最大的组件**

### **Step 3: + Tree Structure (D=7, B=2, 无剪枝 t=0.0)**
```
Throughput: 63.4 t/s
Speedup: 0.49×
Gain over Linear: -63.6% ⚠️
```
- 添加树形结构，但**无剪枝**
- **性能大幅下降！**
- 原因：树太大，验证开销过高

### **Step 4: + Adaptive Pruning (τ=0.05)**
```
Throughput: 182.7 t/s
Speedup: 1.42×
Gain over no-prune: +188%
Total gain over baseline: +42.1%
```
- 添加自适应剪枝
- **剪枝是关键！**
- 最终性能超越Linear

---

## ⚠️ **关键发现**

### **Tree without pruning is SLOWER than Linear!**

```
Linear K=6:          174.2 t/s (1.35×)
Tree (no prune):     63.4 t/s (0.49×)  ❌ 比Baseline还慢！
Tree (with prune):   182.7 t/s (1.42×)  ✅ 最快
```

**原因分析**：
1. **Tree D=7, B=2, 无剪枝** 会生成非常大的树
   - 理论节点数：1 + 2 + 4 + 8 + 16 + 32 + 64 = 127个节点
   - 实际可能更少（因为有些路径短），但仍然很大

2. **大树的验证开销非常高**
   - 需要对所有节点做forward pass
   - 内存占用大
   - 计算开销大

3. **Acceptance rate也低**
   - 无剪枝：60.4%
   - 有剪枝：72.6%
   - 说明剪枝后的树质量更好

---

## 🎯 **两种消融实验方案**

### **方案 A: 标准4步消融（展示剪枝的关键性）** ⭐ 推荐

```
Step 1: Baseline               128.6 t/s (1.00×)
Step 2: + Draft (Linear)       174.2 t/s (1.35×) [+35%]
Step 3: + Tree (no prune)      63.4 t/s (0.49×)  [-64%] ⚠️
Step 4: + Adaptive Pruning     182.7 t/s (1.42×) [+188%]
```

**优点**：
- ✅ 展示剪枝的**关键重要性**
- ✅ 说明为什么pruning是核心贡献
- ✅ 所有数据100%真实

**缺点**：
- ⚠️ Step 3性能下降，需要解释
- ⚠️ 不是"单调递增"的消融实验

**文字说明**：
```
Table X shows a 4-step ablation study. Starting from autoregressive 
decoding (128.6 t/s), adding a draft model yields 35% improvement 
(174.2 t/s). However, naively expanding to a tree structure **without 
pruning** (t=0.0) significantly degrades performance (63.4 t/s, 0.49×), 
as the large unpruned tree introduces excessive verification overhead. 
This demonstrates that **adaptive pruning is essential** for tree-based 
speculation. With pruning (t=0.05), DynaTree achieves 182.7 t/s (1.42×), 
a 188% improvement over the unpruned tree and 5% faster than linear 
speculation, validating the effectiveness of probability-based branch 
pruning.
```

---

### **方案 B: 简化3步消融（跳过无剪枝步骤）**

```
Step 1: Baseline               128.6 t/s (1.00×)
Step 2: + Draft (Linear)       174.2 t/s (1.35×) [+35%]
Step 3: + Tree + Pruning       182.7 t/s (1.42×) [+5%]
```

**优点**：
- ✅ 单调递增，逻辑清晰
- ✅ 避免解释"为什么Step 3变慢"

**缺点**：
- ❌ 无法突出剪枝的重要性
- ❌ Tree和Pruning合并，贡献不明确

---

## 📊 **推荐：方案A（4步消融）**

### **为什么推荐方案A？**

1. **突出核心贡献**: 证明adaptive pruning是DynaTree的关键创新
2. **真实反映研究过程**: 直接展示"为什么需要pruning"
3. **增强论文可信度**: 不隐藏负面结果，反而更有说服力
4. **回答审稿人问题**: 如果审稿人问"为什么需要pruning"，这个数据就是答案

### **LaTeX表格（方案A）**

```latex
\begin{table}[t]
\centering
\caption{\textbf{Ablation study: progressive component addition.} Starting from autoregressive decoding, we incrementally add (i)~draft-based speculation, (ii)~tree structure, and (iii)~adaptive pruning. Notably, an unpruned tree (Step~3) severely degrades performance due to excessive verification overhead, demonstrating that probability-based pruning is essential for efficient tree-based speculation.}
\label{tab:ablation}
\begin{tabular}{llccc}
\toprule
Step & Components & Configuration & Throughput & Speedup \\
\midrule
1 & Baseline & AR only & 128.6 & 1.00\(\times\) \\
2 & + Draft model & Linear K=6 & 174.2 & 1.35\(\times\) \\
3 & + Tree structure & D=7, B=2, \(\tau\)=0.0 & 63.4 & 0.49\(\times\) \\
\textbf{4} & \textbf{+ Adaptive pruning} & \(\tau\)=\textbf{0.05} & \textbf{182.7} & \textbf{1.42\(\times\)} \\
\bottomrule
\end{tabular}
\end{table}
```

### **文字说明（正文）**

```latex
\subsection{Ablation Study}

To isolate the contribution of each algorithmic component, we conduct a 
4-step ablation study by progressively adding features to the baseline 
autoregressive decoder. Table~\ref{tab:ablation} summarizes the results 
on WikiText-2 generating 500 tokens. Starting from pure autoregressive 
generation (128.6 tokens/s), introducing speculative decoding with a draft 
model (Linear K=6) yields a 35\% improvement (174.2 tokens/s), demonstrating 
the core benefit of parallel verification.

Expanding to a tree structure without pruning ($\tau$=0.0, Step 3), however, 
\emph{severely degrades performance} to 63.4 tokens/s (0.49$\times$ speedup). 
This counterintuitive result occurs because an unpruned tree at depth 7 
generates up to 127 nodes, creating excessive verification overhead that 
outweighs the benefits of multi-path exploration. The low acceptance rate 
(60.4\%) further indicates that many low-probability branches waste computation.

Adding adaptive pruning ($\tau$=0.05, Step 4) recovers and exceeds the 
performance, achieving 182.7 tokens/s (1.42$\times$ speedup), a \textbf{188\% 
improvement} over the unpruned tree. This dramatic recovery demonstrates that 
\textbf{probability-based pruning is essential} for tree-based speculative 
decoding. By dynamically removing low-probability branches, pruning reduces 
the average tree size while maintaining high-quality paths, resulting in 
5\% faster throughput than linear speculation.

The ablation study confirms that both draft-based speculation (+35\%) and 
adaptive pruning (+188\% over unpruned) contribute significantly to DynaTree's 
final performance, with pruning being the critical factor that makes tree-based 
approaches practical.
```

---

## 📈 **可视化建议**

### **柱状图（显示负增长）**

```
     200┤                              ╭─────╮
        │                              │     │
     150┤             ╭─────╮          │ 1.42│
        │             │     │          │     │
     100┤   ╭─────╮   │ 1.35│   ╭─┐   ╰─────╯
        │   │     │   │     │   │ │
      50┤   │ 1.00│   ╰─────╯   │0.49
        │   │     │              │ │
       0└───┴─────┴──────────────┴─┴──────────────
           Step1   Step2        Step3   Step4
```

**关键**：
- Step 3的柱子明显低于Step 2
- 用不同颜色标注（如红色）
- 箭头或标注说明"Pruning is essential"

---

## 🎯 **下一步行动**

### **Option 1: 立即添加4步消融实验** ⭐ 推荐

我可以帮你：
1. 创建绘图脚本 (`plot_ablation_4steps.py`)
2. 生成消融图表 (显示负增长)
3. 更新LaTeX表格和文字
4. 重新编译PDF

预计时间：30分钟

### **Option 2: 使用简化的3步消融**

跳过"无剪枝"步骤，只展示：
- Baseline → Linear → Tree (pruned)

预计时间：20分钟

---

## 💡 **我的建议**

**选择方案A（4步消融）**，因为：

1. **突出核心贡献**: Adaptive pruning是你们的关键创新
2. **回答审稿人质疑**: 证明pruning不是"锦上添花"，而是"必不可少"
3. **增强可信度**: 不隐藏负面结果，反而显得更诚实
4. **故事性强**: "We tried tree structure, it failed, then we added pruning, it succeeded"

这是一个**非常有说服力**的消融实验！

---

**你想选哪个方案？告诉我，我立即开始实现！**

