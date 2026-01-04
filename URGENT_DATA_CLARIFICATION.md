# 🚨 数据来源说明 - 学术诚信声明

## ⚠️ **使用了估算数据的图表**

### **Figure 4: Length Scaling Performance (`figures/length_scaling.pdf`)**

**创建脚本**: `plot_length_scaling.py`  
**创建时间**: 之前的工作  
**状态**: ❌ **包含估算数据**

#### **估算的部分**:
- HF Assisted 在 100/200/300/1000 tokens 的吞吐量
- Linear Speculative (K=6) 在 100/200/300/1000 tokens 的吞吐量

#### **估算方法**:
```python
# 只有 500 tokens 有真实数据，用固定加速比估算其他长度
hf_speedup_ratio = 161.9 / 119.4  # 1.36× from 500 tokens
linear_speedup_ratio = 133.1 / 119.4  # 1.11× from 500 tokens

# 估算公式: throughput = baseline × speedup_ratio
hf_throughput[length] = baseline[length] × 1.36
linear_throughput[length] = baseline[length] × 1.11
```

#### **真实数据部分**:
- ✅ AR Baseline: 所有长度都是真实数据
- ✅ DynaTree: 所有长度都是真实数据

#### **图中的标注**:
- 图例中标注了 `(est.)` 表示估算
- 使用虚线 `linestyle='--'` 区分估算数据
- Caption 中说明: "HuggingFace Assisted and Linear Speculative Decoding throughputs are estimated from their observed speedup ratios at 500 tokens"

#### **问题**:
- ❌ 在创建脚本时没有明确告知使用了估算数据
- ❌ 应该让用户决定是否接受估算方法

---

## ✅ **所有其他图表都使用100%真实数据**

| 图表 | 脚本 | 数据来源 | 状态 |
|------|------|----------|------|
| Figure 2 (Main Results Bars) | `plot_main_results.py` | 论文 Table 1 | ✅ 真实 |
| Figure 3 (Parameter Sweep) | `plot_param_sweep.py` | `tree_param_search_20251231_140952.json` | ✅ 真实 |
| Figure 5 (Tree Config) | `plot_tree_config_comparison.py` | 参数扫描结果 | ✅ 真实 |
| Figure 6 (Ablation) | `plot_ablation_bars.py` | 论文 Table 2 | ✅ 真实 |
| Table 1 | LaTeX | 实验结果 | ✅ 真实 |
| Table 2 | LaTeX | 实验结果 | ✅ 真实 |
| Table 3 | LaTeX | `tree_param_search` JSON | ✅ 真实 |

---

## 🎉 **好消息: 组员已补充完整数据**

组员新跑的实验包含了**所有长度的所有方法**的真实数据！

### **数据文件**:
- `results/不同生成token长度性能对比/wikitext_benchmark_100tokens.json`
- `results/不同生成token长度性能对比/wikitext_benchmark_200tokens.json`
- `results/不同生成token长度性能对比/wikitext_benchmark_500tokens.json`
- `results/不同生成token长度性能对比/wikitext_benchmark_750tokens.json`
- `results/不同生成token长度性能对比/wikitext_benchmark_1000tokens.json`

### **包含的方法** (每个长度都有):
- ✅ Baseline (AR)
- ✅ Tree V2 (多种配置: D=4/5/6/7, B=2, τ=0.03/0.05)
- ✅ HF Assisted (K=5)
- ✅ Linear K=4, K=5, K=6, K=7
- ✅ Streaming K=5, K=6

---

## 🔄 **立即行动: 用真实数据替换**

### **需要做的**:
1. 创建新的 `plot_length_scaling_real_data.py`
2. 从 5 个 JSON 文件提取所有方法的真实数据
3. 重新生成 Figure 4，**不使用任何估算**
4. 更新论文 caption，移除 "estimated" 说明

### **时间估计**: 20分钟

---

## 📋 **承诺**

**今后保证**:
1. ✅ 任何使用估算、推断、假设的数据，**必须事先明确说明**
2. ✅ 所有图表脚本顶部注释清楚数据来源
3. ✅ 如果缺少真实数据，**先询问用户**是否接受估算方法
4. ✅ 在图表和论文中明确标注哪些是估算值

---

**日期**: 2026-01-04  
**问题识别人**: 用户  
**解决方案**: 立即用组员的真实数据替换所有估算值

