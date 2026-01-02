# Tree-based Speculative Decoding 实验报告

## 📋 概述

本报告记录了 Tree-based Speculative Decoding (树形投机解码) 的实验过程和结果。实验表明，在最优参数配置下，Tree V2 方法实现了 **1.62x 加速比**，显著优于 HuggingFace 原生 Assisted Generation (1.36x) 和 Linear Speculative Decoding (1.11x)。

---

## 1. 实验环境

### 1.1 硬件配置

| 项目 | 配置 |
|------|------|
| GPU | NVIDIA GPU (CUDA) |
| 显存 | 足够运行 Pythia-2.8B + Pythia-70M |
| 系统 | Linux 5.15.0-126-generic |

### 1.2 软件环境

| 项目 | 版本 |
|------|------|
| Python | 3.x |
| PyTorch | 2.0+ (支持 torch.compile) |
| Transformers | 4.x (支持 DynamicCache) |
| CUDA | 兼容版本 |

### 1.3 模型配置

| 模型角色 | 模型名称 | 参数量 | 路径 |
|---------|---------|--------|------|
| Target Model | Pythia-2.8B | 2.8B | `/mnt/disk1/models/pythia-2.8b` |
| Draft Model | Pythia-70M | 70M | `/mnt/disk1/models/pythia-70m` |

---

## 2. 实验方法

### 2.1 测试方法列表

本实验对比了以下 7 种推理方法：

1. **Baseline (AR)** - 纯自回归生成，作为基准
2. **HuggingFace Assisted Generation** - HuggingFace 官方实现的辅助生成
3. **Linear Speculative Decoding** - 线性投机解码 (K=5,6,7,8)
4. **Tree V2 Speculative Decoding** - 树形投机解码 V2 版本
5. **StreamingLLM + Spec Decode** - 结合 StreamingLLM 的投机解码

### 2.2 Tree-based Speculative Decoding 原理

Tree-based Speculative Decoding 的核心思想是：

```
传统 Linear 方法:
  Draft 模型生成: t1 -> t2 -> t3 -> t4 -> t5 (线性序列)
  Target 验证: 逐个验证

Tree-based 方法:
  Draft 模型生成树形结构:
                    t1
                 /  |  \
               t2a t2b t2c
              / |   |   | \
           t3a t3b t3c t3d t3e
           ...

  Target 一次验证整棵树的所有分支
```

**优势：**
- 并行验证多个候选路径
- 提高找到正确 token 序列的概率
- 更充分利用 GPU 并行计算能力

### 2.3 关键参数说明

| 参数 | 含义 | 最优值 |
|------|------|--------|
| **D (tree_depth)** | 树的最大深度 | 8 |
| **B (branch_factor)** | 每个节点的分支数 | 3 |
| **t (probability_threshold)** | 概率剪枝阈值 | 0.03 |

---

## 3. 实验配置

### 3.1 参数搜索配置

```python
# 参数搜索范围
depths = [3, 4, 5, 6, 7, 8]      # 树深度
branches = [2, 3, 4]             # 分支因子
thresholds = [0.01, 0.02, 0.03, 0.05, 0.1]  # 概率阈值
token_lengths = [100, 200, 300, 500, 1000]  # 生成长度

# 总配置数: 6 × 3 × 5 × 5 = 450 种组合
```

### 3.2 性能测试配置

```python
MAX_NEW_TOKENS = 500        # 生成 token 数
NUM_RUNS = 5                # 每个方法运行次数
SKIP_FIRST = True           # 跳过首次 warmup
WARMUP_ROUNDS = 10          # 预热轮数

# 测试 prompt
PROMPT = """Write a detailed technical explanation about the development 
of large language models. Cover the history, architecture innovations, 
training techniques, and future directions..."""
```

### 3.3 最优 Tree V2 配置

```python
TREE_DEPTH = 8              # 树深度
TREE_BRANCH = 3             # 分支因子  
TREE_THRESHOLD = 0.03       # 概率阈值
MAX_TREE_NODES = 128        # 最大树节点数
```

---

## 4. 实验脚本

### 4.1 参数搜索脚本

**路径**: `papers/tree_param_search.py`

```python
# 核心搜索逻辑
for depth in depths:
    for branch in branches:
        for threshold in thresholds:
            for tokens in token_lengths:
                # 创建 Tree V2 生成器
                gen = TreeSpeculativeGeneratorV2(
                    target_model, draft_model, tokenizer,
                    tree_depth=depth,
                    branch_factor=branch,
                    probability_threshold=threshold,
                    max_tree_nodes=128
                )
                
                # 测量性能
                throughput, stats = measure_performance(gen, tokens)
                
                # 记录结果
                results.append({
                    'depth': depth,
                    'branch': branch,
                    'threshold': threshold,
                    'tokens': tokens,
                    'throughput': throughput,
                    'speedup': throughput / baseline
                })
```

**运行命令**:
```bash
cd /mnt/disk1/ljm/LLM-Efficient-Reasoning
python papers/tree_param_search.py
```

### 4.2 性能对比脚本

**路径**: `papers/benchmark_optimal_config.py`

```python
# 测试所有方法
methods = [
    ("Baseline", run_baseline),
    ("HF Assisted", run_hf_assisted),
    ("Linear K=5", run_linear_k5),
    ("Linear K=6", run_linear_k6),
    ("Linear K=7", run_linear_k7),
    ("Linear K=8", run_linear_k8),
    ("Tree V2 D=8 B=3 t=0.03", run_tree_v2),
    ("Streaming K=6 cache=512", run_streaming_512),
    ("Streaming K=6 cache=1024", run_streaming_1024),
]

for name, run_fn in methods:
    results = []
    for i in range(NUM_RUNS):
        cleanup()
        torch.cuda.synchronize()
        start = time.perf_counter()
        tokens, stats = run_fn()
        elapsed = time.perf_counter() - start
        throughput = tokens / elapsed
        if i > 0:  # 跳过首次 warmup
            results.append(throughput)
    
    avg_throughput = sum(results) / len(results)
    print(f"{name}: {avg_throughput:.1f} t/s")
```

**运行命令**:
```bash
cd /mnt/disk1/ljm/LLM-Efficient-Reasoning
python papers/benchmark_optimal_config.py
```

### 4.3 结果分析脚本

**路径**: `papers/analyze_tree_search_results.py`

```python
# 分析参数搜索结果
def analyze_results(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    results = data['results']
    
    # 按加速比排序
    sorted_results = sorted(results, key=lambda x: x['speedup'], reverse=True)
    
    # 输出 Top 10
    print("Top 10 配置:")
    for i, r in enumerate(sorted_results[:10]):
        print(f"{i+1}. D={r['depth']} B={r['branch']} t={r['threshold']}")
        print(f"   {r['speedup']:.2f}x ({r['throughput']:.1f} t/s)")
```

**运行命令**:
```bash
cd /mnt/disk1/ljm/LLM-Efficient-Reasoning
python papers/analyze_tree_search_results.py results/tree_param_search_20251231_140952.json
```

---

## 5. 实验结果

### 5.1 参数搜索结果

参数搜索共测试了 450 种配置组合，结果保存在:
`results/tree_param_search_20251231_140952.json`

#### 5.1.1 Top 10 最优配置

| 排名 | Tokens | D | B | t | 吞吐量 | Baseline | 加速比 |
|-----|--------|---|---|------|--------|----------|--------|
| 1 | 500 | 8 | 3 | 0.03 | 221.4 t/s | 123.9 t/s | **1.79x** |
| 2 | 500 | 7 | 3 | 0.03 | 217.4 t/s | 123.9 t/s | 1.76x |
| 3 | 500 | 8 | 3 | 0.02 | 217.2 t/s | 123.9 t/s | 1.75x |
| 4 | 500 | 8 | 4 | 0.02 | 212.5 t/s | 123.9 t/s | 1.72x |
| 5 | 500 | 6 | 3 | 0.03 | 212.1 t/s | 123.9 t/s | 1.71x |
| 6 | 1000 | 6 | 3 | 0.05 | 212.3 t/s | 124.5 t/s | 1.71x |
| 7 | 1000 | 6 | 3 | 0.10 | 211.2 t/s | 124.5 t/s | 1.70x |
| 8 | 500 | 7 | 3 | 0.02 | 209.5 t/s | 123.9 t/s | 1.69x |
| 9 | 1000 | 7 | 3 | 0.10 | 210.0 t/s | 124.5 t/s | 1.69x |
| 10 | 1000 | 8 | 2 | 0.10 | 208.7 t/s | 124.5 t/s | 1.68x |

#### 5.1.2 各 Token 长度最优配置

| Token 长度 | 最优配置 | 吞吐量 | 加速比 |
|-----------|----------|--------|--------|
| 100 | D=7, B=3, t=0.03 | 150.7 t/s | 1.43x |
| 200 | D=7, B=3, t=0.03 | 193.2 t/s | 1.54x |
| 300 | D=7, B=3, t=0.03 | 199.0 t/s | 1.60x |
| **500** | **D=8, B=3, t=0.03** | **221.4 t/s** | **1.79x** |
| 1000 | D=6, B=3, t=0.05 | 212.3 t/s | 1.71x |

#### 5.1.3 参数敏感性分析

**Branch Factor (B) 的影响:**
| B | 平均加速比 | 最大加速比 |
|---|-----------|-----------|
| 2 | 1.11x | 1.68x |
| **3** | **1.31x** | **1.79x** |
| 4 | 1.19x | 1.72x |

**结论**: B=3 是最优分支因子

**Probability Threshold (t) 的影响:**
| t | 平均加速比 | 最大加速比 |
|---|-----------|-----------|
| 0.01 | 1.09x | 1.57x |
| 0.02 | 1.28x | 1.75x |
| **0.03** | **1.31x** | **1.79x** |
| 0.05 | 1.17x | 1.71x |
| 0.10 | 1.19x | 1.70x |

**结论**: t=0.03 是最优阈值

### 5.2 性能对比结果

在 500 tokens、最优配置 (D=8, B=3, t=0.03) 下的性能对比：

| 排名 | 方法 | 吞吐量 | 加速比 | 备注 |
|-----|------|--------|--------|------|
| 🥇 | **Tree V2 (D=8, B=3, t=0.03)** | **193.4 t/s** | **1.62x** | 最优 |
| 🥈 | HuggingFace Assisted | 161.9 t/s | 1.36x | 官方实现 |
| 🥉 | Linear K=6 | 133.1 t/s | 1.11x | 自定义实现 |
| 4 | Streaming K=6 cache=1024 | 132.9 t/s | 1.11x | StreamingLLM |
| 5 | Linear K=7 | 131.9 t/s | 1.10x | |
| 6 | Linear K=8 | 128.9 t/s | 1.08x | |
| 7 | Linear K=5 | 125.2 t/s | 1.05x | |
| 8 | Baseline (AR) | 119.4 t/s | 1.00x | 基准 |
| 9 | Streaming K=6 cache=512 | 114.2 t/s | 0.96x | 开销过大 |

### 5.3 详细运行数据

#### Tree V2 (D=8, B=3, t=0.03)
```
Run 1: 500 tokens, 2.83s, 176.8 t/s (warmup, 跳过)
Run 2: 500 tokens, 2.57s, 194.5 t/s
Run 3: 500 tokens, 2.59s, 192.9 t/s
Run 4: 500 tokens, 2.60s, 192.3 t/s
Run 5: 500 tokens, 2.58s, 194.1 t/s
>>> 平均: 193.4 t/s (1.62x)
    接受率: 29.6%
```

#### HuggingFace Assisted
```
Run 1: 500 tokens, 3.16s, 158.2 t/s (warmup, 跳过)
Run 2: 500 tokens, 3.08s, 162.3 t/s
Run 3: 500 tokens, 3.10s, 161.2 t/s
Run 4: 500 tokens, 3.10s, 161.4 t/s
Run 5: 500 tokens, 3.08s, 162.6 t/s
>>> 平均: 161.9 t/s (1.36x)
```

#### Linear K=6
```
Run 1: 500 tokens, 4.04s, 123.9 t/s (warmup, 跳过)
Run 2: 500 tokens, 3.76s, 132.8 t/s
Run 3: 500 tokens, 3.76s, 133.0 t/s
Run 4: 500 tokens, 3.75s, 133.2 t/s
Run 5: 500 tokens, 3.76s, 133.1 t/s
>>> 平均: 133.1 t/s (1.11x)
    接受率: 68.3%, 每轮 tokens: 4.10
```

---

## 6. 结论与分析

### 6.1 核心发现

1. **Tree V2 是最快的方法**
   - 实现了 **1.62x 加速比** (193.4 t/s)
   - 比 HuggingFace 官方实现快 **19%**
   - 比 Linear 方法快 **45%**

2. **最优参数配置**
   - 树深度 D = 8
   - 分支因子 B = 3
   - 概率阈值 t = 0.03
   - 生成长度 = 500 tokens

3. **Tree V2 的优势来源**
   - 并行验证多个候选分支
   - 概率剪枝减少无效计算
   - 更高效利用 GPU 并行能力

### 6.2 方法对比总结

| 方法 | 加速比 | 优点 | 缺点 |
|------|--------|------|------|
| **Tree V2** | **1.62x** | 最快、可定制 | 实现复杂 |
| HF Assisted | 1.36x | 官方支持、稳定 | 不可定制 |
| Linear | 1.11x | 简单、易理解 | 加速有限 |
| Streaming | 1.11x | 支持长序列 | 短序列开销大 |

### 6.3 适用场景建议

| 场景 | 推荐方法 | 理由 |
|------|----------|------|
| 追求最大速度 | Tree V2 (D=8, B=3, t=0.03) | 加速比最高 |
| 生产环境稳定性 | HuggingFace Assisted | 官方维护、稳定 |
| 超长序列生成 | Streaming + Spec Decode | 内存效率高 |
| 快速原型验证 | Linear K=6 | 实现简单 |

### 6.4 未来优化方向

1. **Tree + StreamingLLM 融合** - 结合两者优势用于超长序列
2. **动态参数调整** - 根据生成内容动态调整树结构
3. **torch.compile 优化** - 利用 PyTorch 2.0 编译加速
4. **更大模型验证** - 在 7B、13B 模型上验证效果

---

## 7. 复现指南

### 7.1 环境准备

```bash
# 克隆仓库
git clone <repository_url>
cd LLM-Efficient-Reasoning

# 安装依赖
pip install torch transformers matplotlib numpy
```

### 7.2 运行参数搜索

```bash
# 完整参数搜索 (约 2-3 小时)
python papers/tree_param_search.py

# 结果保存在 results/tree_param_search_*.json
```

### 7.3 运行性能对比

```bash
# 在最优配置下对比所有方法 (约 10 分钟)
python papers/benchmark_optimal_config.py
```

### 7.4 分析结果

```bash
# 分析参数搜索结果
python papers/analyze_tree_search_results.py results/tree_param_search_20251231_140952.json
```

---

## 8. 附录

### 8.1 相关文件列表

| 文件 | 说明 |
|------|------|
| `spec_decode/core/token_tree.py` | TokenTree 数据结构实现 |
| `spec_decode/core/tree_speculative_generator.py` | Tree V2 生成器实现 |
| `papers/tree_param_search.py` | 参数搜索脚本 |
| `papers/benchmark_optimal_config.py` | 性能对比脚本 |
| `papers/analyze_tree_search_results.py` | 结果分析脚本 |
| `results/tree_param_search_20251231_140952.json` | 参数搜索原始数据 |

### 8.2 参考文献

1. Leviathan et al., "Fast Inference from Transformers via Speculative Decoding", ICML 2023
2. Miao et al., "SpecInfer: Accelerating Generative Large Language Model Serving with Tree-based Speculative Inference", 2024
3. Xiao et al., "Efficient Streaming Language Models with Attention Sinks", ICLR 2024

---

**报告生成时间**: 2026年1月2日  
**实验环境**: Pythia-2.8B + Pythia-70M on CUDA

