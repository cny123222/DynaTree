# 大作业：语言模型高效推理

## 目标

在不改变模型参数的情况下，让语言模型推理更快、更省显存。

## 任务内容

1.  了解当前最前沿的推理加速技术。
2.  动手实现所选技术。
3.  以 NeurIPS 风格撰写小论文，并完成实验、对比和分析。

## 推理加速技术分类

### 1. Attention 机制层面

针对 Attention 机制本身涉及的各个要素（如 Q、K、V、softmax 计算方式等）的改进。

*   Multi-Query Attention
*   Group-Query Attention
*   Multi-head Latent Attention
*   Palu
*   Performers

### 2. 逐层 KV Cache 压缩层面

对每一层的 KV Cache 序列维度的长度进行压缩。

*   Streaming LLM、LM-Infinite
*   InfLLM
*   H2O、Scissorhands、TOVA
*   TreeKV
*   SnapKV
*   PyramidKV

### 3. 层间 KV Cache 压缩层面

通过在不同层复用同一套 KV Cache，降低显存开销。

*   Layer-Condensed KV
*   Cross-Layer Attention
*   You Only Cache Once
*   Context Expansion with Parallel Encoding

### 4. 宏观计算架构适配层面

不改变模型本身，而是从模型运行的计算环境特点出发寻求解决方案。

*   Flash Attention 系列
*   Flash Decoding 系列
*   Speculative Decoding

## 参考加速方法及快速上手

*   **参考项目：** [https://github.com/NVIDIA/kvpress](https://github.com/NVIDIA/kvpress)
*   **内容：** 包含多种 KV 压缩算法和评估指标，可作为实现参考。

## 个人部分

### 1. 算法实现

*   在 KVPress 或其他加速优化方法中选择并复现。

### 2. Baseline 设置

*   **所有实验都应该在 Pythia-2.8B 模型上用无训练方法进行优化。**
*   **有条件的小组在实验设计有需要的情况下，可以在跑完 2.8B 的基础上可以跑 7B，评分将按照 2.8B 的情况来评分，7B 只在方法本身需要的情况下考虑（比如 2.8B 效果不好，但是你设计的方法模型越大效果越好，所以你补了 7B 的实验，这种情况下才会考虑）。**
*   在 pg-19, wikitext 等数据集上进行 ppl 测试和加速测试。
    *   **注：** pg-19 为超长文本数据集，可取单一 sample 进行测试即可。
*   Pythia-2.8B 模型链接：[https://huggingface.co/EleutherAI/pythia-2.8b](https://huggingface.co/EleutherAI/pythia-2.8b)

### 3. 创建个人独立的公开 Github 仓库

*   包含你实现的全部代码和 `README`。
*   建议使用 git 的相关功能进行代码历史跟踪。

### 4. `README` 至少需要包含

*   如何运行你的代码。
*   一个简短报告展示加速/优化效果。

### 5. 评分标准

*   根据所实现算法和提交的仓库可复现性进行评分。

## 小组部分

### 1. 算法实现

*   综合实现集成一系列加速方法。
*   报告加速/优化的效果和性能提升。

### 2. Baseline 设置

*   **所有实验都应该在 Pythia-2.8B 模型上用无训练方法进行优化。**
*   **有条件的小组在实验设计有需要的情况下，可以在跑完 2.8B 的基础上可以跑 7B，评分将按照 2.8B 的情况来评分，7B 只在方法本身需要的情况下考虑（比如 2.8B 效果不好，但是你设计的方法模型越大效果越好，所以你补了 7B 的实验，这种情况下才会考虑）。**
*   在 pg-19, wikitext 等数据集上进行 ppl 测试和加速测试。
    *   **注：** pg-19 为超长文本数据集，可取单一 sample 进行测试即可。
*   Pythia-2.8B 模型链接：[https://huggingface.co/EleutherAI/pythia-2.8b](https://huggingface.co/EleutherAI/pythia-2.8b)

### 3. 参考加速指标

*   **TTFT:** Time To First Token (首个 Token 生成时间)
*   **TPOT:** Time Per Output Token (每个 Token 生成时间)
*   **Throughput:** (吞吐量)
*   **Total/average FLOPs:** Floating Point Operations over the sequence (序列上的总/平均浮点运算次数)

### 4. 论文要求

*   使用 NeurIPS 模板撰写英文论文（在压缩包提供），正文不超过 4 页。
*   包含 abstract, intro, method, experiments。
*   随论文提交可以复现实验结果的代码（可以是链接附在论文内；个人部分的链接也贴在论文里即可）。

### 5. 评分标准（以 NeurIPS 会议 Review 的标准对论文质量评分，小组内成员共享）

*   **创新：** 设计出相应的改进算法，将根据改进算法的质量和有效性进行评分；若是在已有方法上进行的原创性改进，应在报告中汇报其本身算法的参考文献。
*   **严谨：** 相关的实验或者错误分析，进行严谨的论证，而非凭空猜测。
*   **诚信：** 不追求高性能以及大的提升，但要求学术诚信以及研究的规范。

## 附加要求

### 1. 创新

*   设计出相应的改进算法，将根据改进算法的质量和有效性进行评分。
*   可以在某个已有的算法基础上进行改进，若是在已有方法上进行的原创性改进，应在报告中汇报其本身算法的参考文献。

### 2. 分析与讨论

*   开放式的讨论，包括但不限于对于性能瓶颈的讨论、对于某种方法为什么会没有效果的讨论等。
*   建议提供相关的实验或者错误分析，进行严谨的论证，而非凭空猜测。

### 3. 分组完成

*   每小组人数不超过 3 人，只需一个人提交。
*   需在报告中注明分工和工作量。

### 4. 性能问题

*   性能不是决定最终得分的唯一指标，不追求高性能以及大的提升，但要求学术诚信以及研究的规范，希望大家可以抱着探索的精神完成这次作业。

## 大作业 DDL

**2026年1月5日23:59 (17周周一晚)**