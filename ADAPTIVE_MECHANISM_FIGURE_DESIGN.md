# DynaTree Three-Phase Adaptive Mechanism Figure Design

**目标**: 创建一张清晰展示DynaTree三阶段自适应机制的示意图

---

## 🎯 图表目标

### 核心要传达的信息
1. **Phase 1**: Draft model的置信度如何决定分支数（1-3）
2. **Phase 2**: 累积概率如何决定深度（早停/正常/深扩展）
3. **Phase 3**: 历史接受率如何调整参数（反馈循环）

### 与Figure 1的区别
- **Figure 1**: 展示整体流程（6个阶段）
- **新图**: 聚焦自适应决策机制（3个phase的内部逻辑）

---

## 📐 图表设计

### 布局建议: 3行×1列（垂直布局）

```
┌────────────────────────────────────────────────────────┐
│                                                        │
│  Phase 1: Confidence-Based Adaptive Branching         │
│  [输入] → [决策逻辑] → [输出效果]                      │
│                                                        │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Phase 2: Dynamic Depth Control                        │
│  [输入] → [决策逻辑] → [输出效果]                      │
│                                                        │
├────────────────────────────────────────────────────────┤
│                                                        │
│  Phase 3: Historical Acceptance Adjustment             │
│  [输入] → [决策逻辑] → [输出效果] → [反馈]             │
│                                                        │
└────────────────────────────────────────────────────────┘
```

---

## 🎨 详细设计

### Phase 1: Confidence-Based Adaptive Branching

#### 左侧：输入
```
Draft Model Logits
       ↓
   Softmax
       ↓
P = [0.75, 0.15, 0.05, 0.03, 0.02]
       ↓
Confidence = max(P) = 0.75
```

#### 中间：决策树
```
         Confidence
              ↓
    ┌─────────┼─────────┐
    ↓         ↓         ↓
  > 0.8    0.3-0.8    < 0.3
    ↓         ↓         ↓
   B=1       B=2       B=3
```

#### 右侧：可视化效果
```
High Conf (C=0.9):     Medium Conf (C=0.6):    Low Conf (C=0.2):
      u0                     u0                      u0
       ↓                   ↙   ↘                  ↙  ↓  ↘
      u1                 u1    u2              u1  u2  u3
  (1 branch)          (2 branches)          (3 branches)
  
  省计算               正常探索              多探索
```

#### 标注文字
- **High Confidence (>0.8)**: Draft model is certain → Use 1 branch (save computation)
- **Medium Confidence (0.3-0.8)**: Normal uncertainty → Use 2 branches (default)
- **Low Confidence (<0.3)**: Draft model is uncertain → Use 3 branches (explore more)

---

### Phase 2: Dynamic Depth Control

#### 左侧：输入
```
Node Path: u0 → u1 → u2
Log Probs: [-0.1, -0.2, -0.3]
       ↓
Cumulative Logit = -0.6
       ↓
Cumulative Prob = exp(-0.6) = 0.55
```

#### 中间：决策树
```
    Cumulative Probability
              ↓
    ┌─────────┼─────────┐
    ↓         ↓         ↓
  < 0.1    0.1-0.5    > 0.5
    ↓         ↓         ↓
Early Stop  Normal   Deep Expand
 (停止)    (D=4)      (D≤8)
```

#### 右侧：深度可视化
```
Early Stop (P=0.05):   Normal (P=0.3):      Deep Expand (P=0.7):
Depth 3                Depth 4              Depth 8
  u0                     u0                    u0
  u1  ⛔                 u1                    u1
  u2  (stop here)       u2                    u2
                        u3                    u3
                        u4 ✓                  u4
                                             u5
                                             u6
                                             u7
                                             u8 ✓
```

#### 标注文字
- **Very Low Prob (<0.1)**: Branch is unlikely → Early stop at depth 3 (save computation)
- **Medium Prob (0.1-0.5)**: Normal quality → Stop at base depth 4
- **High Prob (>0.5)**: High-quality path → Allow deep expansion up to depth 8

---

### Phase 3: Historical Acceptance Adjustment

#### 左侧：历史记录
```
Last 10 Rounds:
Accept Rates: [0.85, 0.88, 0.82, 0.90, 0.87, ...]
       ↓
Avg Accept Rate = 0.86
```

#### 中间：决策逻辑
```
     Avg Accept Rate
            ↓
    ┌───────┼───────┐
    ↓       ↓       ↓
  > 0.8   0.6-0.8  < 0.6
    ↓       ↓       ↓
 Too High  Good   Too Low
    ↓       ↓       ↓
More     Keep    More
Aggr.   Current  Cons.
```

#### 右侧：参数调整
```
Too High (>0.8):          Good (0.6-0.8):       Too Low (<0.6):
Draft too accurate        Balanced              Draft inaccurate

Adjustments:              No change             Adjustments:
base_depth: 4→5           base_depth: 4         base_depth: 4→3
high_conf: 0.8→0.75       high_conf: 0.8        high_conf: 0.8→0.85
deep_expand: 0.5→0.4      deep_expand: 0.5      deep_expand: 0.5→0.6

↓ More exploration        ↓ Maintain            ↓ Less exploration
↑ Deeper trees            ↓ Current config      ↑ Shallower trees
```

#### 反馈循环（用箭头）
```
Parameters → Tree Generation → Verification → Accept Rate → [loop back to Parameters]
```

#### 标注文字
- **High Accept Rate (>0.8)**: Draft model very accurate → Be more aggressive (deeper, more branches)
- **Target Range (0.6-0.8)**: Balanced performance → Keep current parameters
- **Low Accept Rate (<0.6)**: Draft model struggles → Be more conservative (shallower, fewer branches)

---

## 🖼️ 绘图建议

### 工具选择
1. **draw.io** / **diagrams.net** (推荐，免费在线)
2. **PowerPoint** + 导出高分辨率PNG
3. **Inkscape** (矢量图，可导出PDF)
4. **Python matplotlib** (编程生成)

### 颜色方案
```
Phase 1 (Branching):
  - High Conf: 🟢 Green (#4CAF50)
  - Medium Conf: 🟡 Yellow (#FFC107)
  - Low Conf: 🔴 Red (#F44336)

Phase 2 (Depth):
  - Early Stop: ⛔ Red (#F44336)
  - Normal: 🟦 Blue (#2196F3)
  - Deep Expand: 🟩 Green (#4CAF50)

Phase 3 (Adjustment):
  - More Aggressive: 🟢 Green (#4CAF50)
  - Keep: 🟦 Blue (#2196F3)
  - More Conservative: 🟠 Orange (#FF9800)

Arrows:
  - Input flow: Black (#000000)
  - Feedback loop: Dashed blue (#2196F3)
```

### 字体和样式
```
Title: Bold, 14pt
Phase headers: Bold, 12pt
Decision labels: Regular, 10pt
Annotations: Italic, 9pt

Tree nodes: Circles (diameter 20px)
Decision boxes: Rectangles with rounded corners
Arrows: 2px width, solid or dashed
```

---

## 📝 Caption建议

```latex
\caption{\textbf{DynaTree's three-phase adaptive mechanism.} 
\textbf{(Top)} Phase 1: Confidence-based adaptive branching adjusts 
the number of child nodes (1--3) per node based on draft model 
confidence: high confidence uses 1 branch to save computation, 
low confidence uses 3 branches to explore more options. 
\textbf{(Middle)} Phase 2: Dynamic depth control implements early 
stopping for low cumulative probability branches (<0.1) and deep 
expansion for high-probability paths (>0.5), adaptively balancing 
tree depth between 3 and 8 layers. 
\textbf{(Bottom)} Phase 3: Historical acceptance adjustment maintains 
a sliding window of recent acceptance rates and dynamically tunes 
confidence thresholds and base depth—being more aggressive when 
draft accuracy is high (>0.8) and more conservative when it is 
low (<0.6). This three-phase mechanism enables training-free 
adaptation to varying text complexity and draft model performance.}
\label{fig:adaptive-mechanism}
```

---

## 🎯 插入位置建议

### 选项1: 在Methodology Section 3.3之后
```latex
\subsection{Confidence-Aware Adaptive Branching}
[新增subsection描述算法]

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{../figures/adaptive_mechanism.pdf}
  \caption{...}
  \label{fig:adaptive-mechanism}
\end{figure}
```

### 选项2: 在Experiments Section开始之前
```latex
\section{Methodology}
...

\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{../figures/adaptive_mechanism.pdf}
  \caption{...}
  \label{fig:adaptive-mechanism}
\end{figure}

\section{Experiments}
```

**推荐选项1**: 与算法描述紧密结合

---

## 📐 尺寸建议

### 图片尺寸
- **宽度**: 与论文列宽一致（约6.5英寸）
- **高度**: 建议6-8英寸（三个phase垂直排列）
- **分辨率**: 300 DPI（PDF矢量图更佳）

### 各Phase高度分配
```
Total height: 8 inches

Phase 1: 2.5 inches (需要展示树结构)
Phase 2: 3.0 inches (需要展示不同深度)
Phase 3: 2.5 inches (需要展示反馈循环)
```

---

## ✅ 检查清单

创建图表时确保：

- [ ] 三个Phase清晰分隔（用横线或背景色区分）
- [ ] 每个Phase有明确的输入→决策→输出流程
- [ ] 关键阈值数字清晰可见（0.8, 0.3, 0.1, 0.5等）
- [ ] 树结构可视化准确（不同分支数和深度）
- [ ] 颜色使用一致（相同含义用相同颜色）
- [ ] 箭头方向明确（因果关系清楚）
- [ ] 文字标注简洁（不要过度拥挤）
- [ ] Phase 3的反馈循环清晰（虚线箭头）
- [ ] 图例说明充分（如果需要）
- [ ] 与Caption描述完全一致

---

## 🚀 快速开始步骤

1. **使用draw.io** (最简单)
   - 访问 https://app.diagrams.net/
   - 选择空白画布
   - 创建3个区域（用矩形框区分）
   
2. **绘制Phase 1**
   - 左: 添加文本框显示置信度计算
   - 中: 添加决策流程图（菱形或矩形）
   - 右: 用圆圈画树节点，展示1/2/3个分支

3. **绘制Phase 2**
   - 左: 显示累积概率计算
   - 中: 决策流程图（3个分支）
   - 右: 画不同深度的树（3/4/8层）

4. **绘制Phase 3**
   - 左: 显示历史记录
   - 中: 决策流程图
   - 右: 显示参数调整方向
   - 底部: 添加反馈循环箭头

5. **导出**
   - 文件 → 导出为 → PNG (300 DPI)
   - 或导出为 → PDF (矢量图，推荐)

---

## 📚 参考示例

类似风格的图表可参考：
- AdaEAGLE论文的Figure 2 (Draft Length Prediction)
- CM-ASD论文的Figure 1 (Confidence Modulation)
- EAGLE论文的Figure 3 (Tree Construction)

我们的图需要更清晰地展示**三个独立但相互关联的决策流程**。

---

## 💡 关键提示

### 这张新图解决的问题
1. ❌ Figure 1显示不出"为什么分支数不同"
2. ❌ Figure 1看不出"为什么深度不同"
3. ❌ Figure 1没有展示"参数如何调整"

### 新图的价值
1. ✅ 让读者理解**adaptive的机制**
2. ✅ 展示**三个phase如何协同**
3. ✅ 突出**training-free的关键**（都基于自然信号）
4. ✅ 可视化**决策逻辑**（不只是流程）

这张图是论文的**核心创新可视化**，非常重要！

