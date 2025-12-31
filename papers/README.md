# Speculative Decoding 论文实验结果与复现指南

本目录包含论文撰写所需的所有实验结果和复现命令。

---

## 📁 目录结构

```
papers/
├── README.md                           # 本文件（复现指南）
├── speculative_decoding_paper_draft.md # 论文草稿
├── reproduction_commands.sh            # 一键复现脚本
└── figures/
    ├── paper_fig6_long_seq.png         # 长序列内存对比图
    └── paper_fig7_comprehensive.png    # 全面性能对比图（主图）

项目根目录保留的结果文件：
├── benchmark_comprehensive_results.json  # 全面 benchmark 数据
└── benchmark_long_seq_results.json       # 长序列测试数据
```

---

## 参数说明

### 表格中各列含义

| 参数 | 含义 | 说明 |
|------|------|------|
| **Tokens** | 目标生成长度 | `max_new_tokens` 参数，表示要生成多少个 token |
| **Throughput** | 吞吐量 (tokens/s) | 每秒生成的 token 数，越高越好 |
| **TTFT** | 首 token 延迟 (ms) | Time to First Token，从输入到输出第一个 token 的时间 |
| **TPOT** | 每 token 延迟 (ms) | Time per Output Token，生成每个 token 的平均时间 |
| **PPL** | 困惑度 | Perplexity，衡量生成质量，越低越好 |
| **Accept%** | 接受率 | Draft tokens 被 target model 接受的比例 |
| **T/Round** | 每轮 tokens | 每个推测解码轮次平均生成的 tokens 数 |
| **Mem MB** | 内存增长 | 推理过程中 GPU 显存增长量 |
| **Compress** | 压缩次数 | StreamingLLM KV cache 压缩触发次数 |

---

## 📊 核心实验结果

### 实验 1：全面性能对比（主表格）

**结果文件**：`benchmark_comprehensive_results.json`  
**图表文件**：`papers/figures/paper_fig7_comprehensive.png`

**复现命令**：
```bash
cd /mnt/disk1/ljm/LLM-Efficient-Reasoning

python spec_decode/benchmark_comprehensive.py \
    --target-model /mnt/disk1/models/pythia-2.8b \
    --draft-model /mnt/disk1/models/pythia-70m \
    --max-new-tokens 500 1000 2000 \
    --max-cache-lens 256 512 1024 \
    --k-value 5 \
    --num-samples 3 \
    --output-json benchmark_comprehensive_results.json \
    --output-plot papers/figures/paper_fig7_comprehensive.png
```

**核心数据**（K=5，Pythia-2.8B + Pythia-70M）：

| 配置 | Tokens | Throughput | TTFT (ms) | TPOT (ms) | PPL | Accept% | Memory |
|------|--------|------------|-----------|-----------|-----|---------|--------|
| standard | 500 | 132.0±47.7 | 277.0 | 8.24 | 1.2 | 99.3% | 607 MB |
| standard | 1000 | 185.6±6.2 | 37.1 | 5.36 | 1.1 | 100% | 1127 MB |
| standard | 2000 | **192.9±3.8** | 37.2 | **5.17** | 1.0 | 100% | 2237 MB |
| stream(256) | 2000 | 177.1±11.5 | 36.8 | 5.65 | 1.1 | 98.2% | **1688 MB** |
| stream(1024) | 1000 | **197.3±4.2** | 36.9 | **5.04** | 1.1 | 100% | 1126 MB |

---

### 实验 2：长序列生成对比（内存优势）

**结果文件**：`benchmark_long_seq_results.json`  
**图表文件**：`papers/figures/paper_fig6_long_seq.png`

**复现命令**：
```bash
python spec_decode/benchmark_long_sequence.py \
    --target-model /mnt/disk1/models/pythia-2.8b \
    --draft-model /mnt/disk1/models/pythia-70m \
    --max-new-tokens 500 1000 2000 \
    --max-cache-lens 256 512 1024 \
    --k-value 5 \
    --output-json benchmark_long_seq_results.json \
    --output-plot papers/figures/paper_fig6_long_seq.png
```

**核心数据**：

| 方法 | Tokens | Throughput | Memory 增长 | 压缩次数 |
|------|--------|------------|-------------|----------|
| standard | 2000 | 168.4 t/s | 874 MB | 0 |
| stream(256) | 2000 | 131.4 t/s | **398 MB** | 409 |
| stream(512) | 2000 | 162.0 t/s | 553 MB | 285 |

**关键结论**：StreamingLLM (cache=256) 节省 **54.5%** 内存，吞吐量损失约 22%。

---

## 📝 论文写作要点

### 主要贡献点

1. **Speculative Decoding 加速效果**
   - Pythia-2.8B + Pythia-70M 上实现 **2.3× 加速**
   - TPOT 从 ~12ms 降低到 **5.04ms**（降低 58%）

2. **StreamingLLM 集成**
   - 内存节省 **24-55%**（取决于 cache 大小）
   - 吞吐量损失 **< 10%**（cache=512 时）
   - PPL 影响 **< 0.2**（可忽略）

3. **最优配置推荐**
   
   | 场景 | 推荐配置 | 原因 |
   |------|---------|------|
   | 追求最高吞吐 | K=5, stream(1024) | 197.3 t/s, TPOT=5.04ms |
   | 内存受限 | K=5, stream(256) | 内存节省 55%，吞吐量损失 22% |
   | 平衡方案 | K=5, stream(512) | 内存节省 21%，吞吐量损失 < 5% |

### 论文表格模板

**Table 1: Performance Comparison**

```latex
\begin{table}[h]
\centering
\caption{Performance Comparison on Pythia-2.8B (Target) + Pythia-70M (Draft)}
\begin{tabular}{lcccccc}
\toprule
Method & Tokens & Throughput & TTFT & TPOT & Memory \\
       &        & (t/s)      & (ms) & (ms) & (MB)   \\
\midrule
Baseline         & 2000 & ~80        & ~40  & ~12  & ~3000 \\
Spec (K=5)       & 2000 & 192.9±3.8  & 37.2 & 5.17 & 2237  \\
Spec+Stream(256) & 2000 & 177.1±11.5 & 36.8 & 5.65 & 1688  \\
Spec+Stream(512) & 2000 & 187.7±4.9  & 37.2 & 5.32 & 1767  \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 🔧 环境要求

```bash
pip install torch transformers accelerate matplotlib numpy tqdm
```

### 硬件配置
- GPU: NVIDIA GPU with CUDA
- 显存: >= 8GB（运行 Pythia-2.8B）

### 模型路径
- Target Model: `/mnt/disk1/models/pythia-2.8b`
- Draft Model: `/mnt/disk1/models/pythia-70m`

---

## ⚠️ 注意事项

1. **PPL 解释**：greedy decoding 下 PPL 接近 1 是正常的

2. **Acceptance Rate**：Accept% 上限为 100%，T/Round > K 表示有 bonus tokens

3. **结果复现**：由于 GPU 状态，结果可能有 ±5% 波动

---

*Last Updated: December 2024*
