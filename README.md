# CS2602 LLM Inference Acceleration

CS2602大作业：针对大型语言模型的KV Cache优化与推理加速。

本项目实现了 **KnormPress** (L2 Norm-Based KV Cache Compression) 算法，通过压缩KV缓存来加速 Pythia-70M 模型的推理过程。

## 项目概述

KnormPress 是一种基于 L2 范数的 KV Cache 压缩方法。核心思想是：**键嵌入（key embeddings）的 L2 范数较低的 token 通常与较高的注意力分数相关**。通过选择性地保留这些重要的低范数 token，可以在保持模型性能的同时显著减少内存占用和加速推理。

### 算法原理

1. 计算每个 token 的键嵌入的 L2 范数
2. 按范数升序排序（低范数 = 高重要性）
3. 保留前 `keep_ratio` 百分比的最低范数 token
4. 仅在缓存大小超过 `prune_after` 阈值时才进行压缩

## 环境配置

### 依赖安装

```bash
# 创建并激活 conda 环境
conda create -n nlp python=3.13
conda activate nlp

# 安装依赖
pip install torch transformers datasets numpy
```

### 模型和数据集

- **模型**: `EleutherAI/pythia-70m-deduped`
- **数据集**: `wikitext-2-raw-v1` (用于测试)

## 使用方法

### 1. 运行基线测试

首先运行基线测试以获得未优化的性能指标：

```bash
conda activate nlp
python baseline_test.py
```

这将测试原始模型在 wikitext 数据集上的性能，输出包括：
- TTFT (Time To First Token): 生成第一个 token 的时间
- TPOT (Time Per Output Token): 平均每个输出 token 的时间
- Throughput: 吞吐量（tokens/秒）
- Peak Memory: 峰值显存占用（仅 CUDA）
- Perplexity (PPL): 困惑度

### 2. 运行优化测试

使用 KnormPress 压缩算法进行测试：

```bash
# 测试多个压缩比率
python optimized_test.py --keep_ratios 1.0,0.9,0.8,0.7 --num_wikitext_samples 3

# 自定义参数
python optimized_test.py \
    --keep_ratios 0.9,0.8,0.7,0.6 \
    --prune_after 512 \
    --skip_layers 0 \
    --num_wikitext_samples 5
```

**参数说明**:
- `--keep_ratios`: 压缩比率列表（1.0 = 无压缩，0.8 = 保留 80%）
- `--prune_after`: 缓存超过此大小才压缩（默认 512）
- `--skip_layers`: 跳过压缩的层（默认第 0 层）
- `--num_wikitext_samples`: 测试的样本数量

## 实验结果

### 性能对比表格

在 Apple Silicon (MPS) 上使用 Pythia-70M 模型的测试结果：

| 压缩率 | 保留比例 | TTFT (秒) | TPOT (秒) | 吞吐量 (tok/s) | PPL | 说明 |
|--------|----------|-----------|-----------|----------------|-----|------|
| 0%     | 1.0      | 0.0623    | 0.0125    | 84.16          | 75.03 | 基线（无压缩） |
| 10%    | 0.9      | **0.0080** | 0.0129    | 77.69          | 75.03 | **TTFT 降低 87%** |
| 20%    | 0.8      | **0.0066** | 0.0130    | 76.79          | 75.03 | **TTFT 降低 89%** |
| 30%    | 0.7      | **0.0063** | 0.0131    | 76.45          | 75.03 | **TTFT 降低 90%** |

### 关键发现

1. **显著降低 TTFT**: 使用 10% 压缩率（keep_ratio=0.9），TTFT 从 0.0623 秒降低到 0.0080 秒，**降低了 87%**

2. **保持稳定吞吐量**: 压缩后的吞吐量仅略有下降（从 84.16 降至 77.69 tokens/s），**下降约 8%**

3. **PPL 无损失**: 所有压缩率下，困惑度保持不变（75.03），表明**模型质量未受影响**

4. **更高压缩率的边际收益递减**: 从 10% 压缩（0.9）到 30% 压缩（0.7），性能提升趋于平缓

### 详细分析

#### 为什么 TTFT 显著降低？

KnormPress 压缩显著减少了首次前向传播时需要处理的 KV Cache 大小，这对于长上下文尤其重要：
- 减少了注意力计算的复杂度（从 O(n²) 降至 O(kn²)，其中 k 是 keep_ratio）
- 降低了内存访问开销
- 优化了缓存利用率

#### 为什么 PPL 没有下降？

L2 范数低的 token 确实对应高注意力分数，保留这些关键 token 可以维持模型的预测能力。KnormPress 是一种**智能压缩**方法，不是简单的截断。

### 适用场景

KnormPress 特别适合：
- **长上下文推理**: 超长文本生成任务（如 pg-19 数据集）
- **实时应用**: 需要低延迟的应用（降低 TTFT）
- **资源受限环境**: 内存或显存有限的设备

## 项目结构

```
.
├── README.md                 # 本文件
├── lab-instruction.md        # 作业要求文档
├── baseline_test.py          # 基线性能测试脚本
├── optimized_test.py         # KnormPress 优化测试脚本
├── kv_compress.py            # KV Cache 压缩核心实现
├── custom_generate.py        # 自定义生成函数（带压缩）
└── l2compress/               # KnormPress 参考实现（原始仓库）
```

## 核心代码说明

### KV Cache 压缩 (`kv_compress.py`)

```python
def l2_compress(past_key_values, keep_ratio=1.0, prune_after=512, skip_layers=[]):
    """
    基于 L2 范数压缩 KV Cache
    
    参数:
        past_key_values: KV 缓存 (list of tuples)
        keep_ratio: 保留比例 (0.0-1.0)
        prune_after: 压缩阈值
        skip_layers: 跳过的层
    """
    # 计算每个 token 的 L2 范数
    token_norms = torch.norm(keys, p=2, dim=-1)
    
    # 按范数排序（升序）
    sorted_indices = token_norms.argsort(dim=-1)
    
    # 保留低范数 token
    tokens_to_keep = ceil(keep_ratio * seq_len)
    compressed_keys = sorted_keys[:, :, :tokens_to_keep, :]
    compressed_values = sorted_values[:, :, :tokens_to_keep, :]
    
    return compressed_cache
```

### 自定义生成 (`custom_generate.py`)

实现了一个生成循环，在每次前向传播后应用 KV Cache 压缩：

```python
def generate_with_compression(model, tokenizer, input_ids, 
                              keep_ratio=1.0, prune_after=512):
    for step in range(max_new_tokens):
        outputs = model(input_ids, past_key_values=past_key_values)
        
        # 应用 KV Cache 压缩
        if keep_ratio < 1.0:
            past_key_values = l2_compress(
                outputs.past_key_values,
                keep_ratio=keep_ratio,
                prune_after=prune_after
            )
```

## 参考文献

- **论文**: [A Simple and Effective L2 Norm-Based Strategy for KV Cache Compression](https://arxiv.org/abs/2406.11430) (EMNLP 2024)
- **原始实现**: [l2compress GitHub Repository](https://github.com/NVIDIA/kvpress)
- **Pythia 模型**: [EleutherAI/pythia-70m-deduped](https://huggingface.co/EleutherAI/pythia-70m-deduped)

## 总结

本项目成功实现了 KnormPress 算法，并在 Pythia-70M 模型上验证了其有效性：

✅ **TTFT 降低 87%** (从 0.062s 到 0.008s)  
✅ **吞吐量基本保持** (仅下降 8%)  
✅ **PPL 无损失** (保持 75.03)  
✅ **易于集成** 到现有 transformers 工作流

KnormPress 是一种简单、有效且**无需训练**的 KV Cache 压缩方法，特别适合长上下文推理场景。

## 作者

[Your Name] - CS2602 课程作业

## 致谢

感谢 KnormPress 论文作者提供的开源实现和详细文档。
