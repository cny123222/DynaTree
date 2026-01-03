# SpecInfer 论文图表总结 (供论文 Agent 参考)

本文档总结了 "SpecInfer: Accelerating Large Language Model Serving with Tree-based Speculative Inference and Verification" 论文中的所有图表，以提供论文图表设计的参考。

---

## 1. Figure 1: Decoding Method Comparison

### (a) Incremental decoding. (示意图)
*   **目的:** 形象地展示传统增量解码（自回归解码）的工作原理。
*   **内容:**
    *   以一个 LLM 方框为中心。
    *   展示输入 token ($t_0$)，然后 LLM 逐次生成 $t_1, t_2, t_3$。
    *   强调这是一个迭代过程，每次生成一个 token。
*   **核心发现/要点:** 直观展现传统解码的“一步一吐”特性，为后续对比做铺垫。

### (b) Timeline Comparison. (时间轴对比图)
*   **目的:** 对比三种解码方法的执行时间线和并行度。
*   **内容:**
    *   **Incremental Decoding Timeline:** LLM 逐个解码 $t_1, t_2, t_3, t_4$，每个解码操作串行。
    *   **Sequence-based Speculative Inference Timeline:** SSM 猜测一串 $t_1, t_2, t_3, t_4$，然后 LLM 一次性并行验证 (SequenceVerify)，接受 $t_1, t_2$，然后继续下一轮猜测。
    *   **Tree-based Speculative Inference Timeline:** SSM 猜测一个 token tree ($t_1, t_2, t_3$ 等多分支)，LLM 一次性并行验证整个 token tree (TreeVerify)，接受更多 token，然后继续下一轮猜测。
*   **核心发现/要点:**
    *   清晰地展示了从串行到并行验证的演变。
    *   突出树形推测解码在单轮中能够验证更多 token 的潜力，从而减少总迭代次数。
    *   强调了 LLM 的 TreeVerify 过程是关键的并行化步骤。

---

## 2. Figure 2: SpecInfer's Overview

### (a) Learning-based Speculator (系统概览图 - 左上角)
*   **目的:** 展示 SpecInfer 中 speculator（即 draft model）如何生成 token tree 的两种方法：Expansion-based 和 Merge-based。
*   **内容:**
    *   **Expansion-based Token Tree Construction:** 一个 SSM 接收输入，通过“扩展”生成一个多分支的 token tree（Output 0, 1, 2）。
    *   **Merge-based Token Tree Construction:** 多个 SSMs（SSM 0, 1, 2）分别生成输出，然后通过“合并”机制构建一个统一的 token tree。
*   **核心发现/要点:** 阐述了 SpecInfer 构造 token tree 的两种策略，强调了利用单个或多个 SSM 产生多样化猜测的能力。

### (b) Token Tree Verifier (系统概览图 - 下方)
*   **目的:** 展示 SpecInfer 的 token tree verifier（即 target model）如何验证 token tree。
*   **内容:**
    *   输入：Speculated Token Tree 和 LLM。
    *   核心过程：**Tree-based Parallel Decoding**，LLM 对整个 token tree 进行并行验证。
    *   输出：Verified Output (一个序列，如 "machine learning system optimization")。
*   **核心发现/要点:** 突出 LLM 作为“验证器”的角色，能够一次性并行处理整个树，而非传统的增量解码器。

### (c) Sequence Representation of Speculated Tokens (示意图 - 中间部分)
*   **目的:** 展示 token tree 如何扁平化为线性序列供 LLM 处理。
*   **内容:** 示意性地将树形结构的 token 序列表示为一串线性 token。
*   **核心发现/要点:** 辅助理解 Tree Attention Mask 的构建基础。

---

## 3. Figure 3: Illustration of token tree expansion

*   **目的:** 详细解释 Expansion-based token tree construction 的一个具体例子。
*   **内容:**
    *   展示一个深度为 3、宽度配置为 (2, 2, 1) 的 token tree。
    *   从 $t_0$ (root) 开始，第一步扩展 2 个分支 ($t_1, t_2$)，第二步每个分支再扩展 2 个 ($t_3, t_4$ 等)，第三步再扩展 1 个。
    *   最终生成 4 条完整的 token 序列（Sequence 1-4）。
*   **核心发现/要点:** 直观展示了如何通过参数化配置（如宽度 k）来生成多条猜测路径，增加了猜测的多样性。

---

## 4. Figure 4: Tree-based Parallel Decoding Comparison

*   **目的:** 深入对比现有序列解码和 SpecInfer 树形并行解码在 KV Cache 管理和注意力掩码 (Causal Mask) 上的差异。
*   **内容:**
    *   **Sequence-based Parallel Decoding (左侧):** 展示多条序列（例如 $t_2 \dots t_5, t_2 \dots t_9$），每条序列都有独立的 KV Cache 和因果掩码。强调了 KV Cache 冲突和重复计算问题。
    *   **Tree-based Parallel Decoding (右侧):** 展示 SpecInfer 的方法。所有 token（验证过的 $t_2$ 和推测的 $t_3 \dots t_9$）被视为一个整体进行处理。通过 **Topology-Aware Causal Mask**，实现共享 KV Cache 和并行计算。
*   **核心发现/要点:**
    *   直观揭示了现有并行解码方法在 KV Cache 管理上的低效性。
    *   突出 SpecInfer 的 **Topology-Aware Causal Mask** 如何有效地在树形结构中实现并行计算，同时维护因果依赖，减少内存访问和计算冗余。这是 Tree Attention 的核心。

---

## 5. Figure 5: Multi-step Speculative Sampling Illustration

*   **目的:** 解释 SpecInfer 的多步推测采样 (Multi-step Speculative Sampling, MSS) 机制在随机采样 (Stochastic Decoding) 下的工作原理。
*   **内容:**
    *   以三个 SSM (SSM 1, 2, 3) 的概率分布为例。
    *   展示每个 SSM 的猜测 token 及其概率。
    *   通过与 LLM 真实概率的比较 (`min(1, P_LLM / P_SSM)`) 来决定 token 是否被接受，以及如何生成 Normalized Residual Distribution 用于下一个 token 的采样。
    *   包含“Verified”和“Failed”的路径，说明验证和回退过程。
*   **核心发现/要点:** 详细展示了 SpecInfer 如何在 stochastic decoding 场景下，通过概率调整和多步采样，确保生成结果与原始 LLM 一致，同时最大化接受的 token 数量。

---

## 6. Figure 6: SpecInfer's Runtime Design Workflow

*   **目的:** 概览 SpecInfer 分布式运行系统的整个工作流程，包括请求调度、SSM 并行推理和 LLM 树形并行解码。
*   **内容:**
    *   **Request Scheduling:** 请求通过 Request Manager 分配给多个 SSM。
    *   **Distributing Requests & SSM-generated Tokens:** SSMs 并行生成 token，并将结果（Speculative Token Trees）返回。
    *   **Token Tree Merge:** Request Manager 合并 SSMs 生成的树。
    *   **LLM Tree-based Parallel Decoding:** LLM 并行验证 token tree，生成 LLM-generated Tokens。
    *   **Token Tree Verification:** 最终验证和接受。
    *   整个流程涉及多个 GPU 和 CPU。
*   **核心发现/要点:** 展示了 SpecInfer 在系统层面的分布式设计，强调了其如何利用数据并行和模型并行来高效管理多 SSM 和 LLM 的推理与验证。

---

## 7. Figure 7: End-to-End Inference Latency Comparison

*   **目的:** 对比 SpecInfer 与现有 LLM 服务系统在端到端推理延迟上的性能。
*   **内容:**
    *   三个子图，分别针对 LLaMA-7B, OPT-30B, LLaMA-65B 模型。
    *   X 轴是 Batch Size (BS=1 到 BS=16)。
    *   Y 轴是 Per-token latency (ms)。
    *   对比了 Baseline (vLLM, HuggingFace TGI, FasterTransformer, SpecInfer w/ Incremental Decoding) 和 SpecInfer 的不同模式 (Sequence-based Speculative Inference, Tree-based Speculative Inference)。
*   **核心发现/要点:**
    *   SpecInfer 在 Tree-based Speculative Inference 模式下，通常能显著降低 per-token latency，实现比现有系统更好的性能。
    *   性能提升随 Batch Size 增加而减小（因为大 Batch Size 已经有较高的 GPU 利用率，SpecInfer 可利用的“空闲”资源变少）。
    *   SpecInfer 的增量解码性能与现有系统持平，证明其实现效率。

---

## 8. Figure 8: Offloading-based Inference Latency Comparison

*   **目的:** 对比 SpecInfer 与 FlexGen 在 Offloading-based LLM Inference 模式下的性能。
*   **内容:**
    *   两个子图，针对 OPT-13B 和 OPT-30B 模型。
    *   X 轴是 Batch Size (BS=1 到 BS=16)。
    *   Y 轴是 Per-token latency (s)。
    *   对比 FlexGen 和 SpecInfer。
*   **核心发现/要点:** SpecInfer 在 Offloading 场景下也实现了显著的 per-token latency 降低，这表明其机制能有效减少 CPU DRAM 和 GPU HBM 之间的数据传输。

---

## 9. Figure 9: Speculative Performance with Different Token Tree Structures (CDF)

*   **目的:** 展示 SpecInfer 在不同 token tree 宽度配置下的推测性能。
*   **内容:**
    *   两个子图，针对 Greedy decoding 和 Stochastic decoding。
    *   X 轴是 CDF (Cumulative Distribution Function)，表示验证的 token 数量。
    *   Y 轴是平均每解码步验证的 token 数量。
    *   对比了不同 Tree Width (1 到 5) 的曲线。
*   **核心发现/要点:** 更大的 Tree Width（例如 3 到 5）可以显著提高每解码步平均验证的 token 数量，从而提升推测效率，尤其在随机解码场景下。

---

## 10. Figure 10: End-to-End Inference Latency with Different Tree Widths

*   **目的:** 展示不同 Tree Width 对 SpecInfer 端到端推理延迟的影响。
*   **内容:**
    *   X 轴是 Batch Size (BS=1 到 BS=16)。
    *   Y 轴是 Per-token latency (ms)。
    *   对比了不同 Tree Width (1 到 5) 的曲线。
*   **核心发现/要点:**
    *   对于小 Batch Size (BS=1, 2)，更大的 Tree Width 可以持续降低 per-token latency。
    *   对于大 Batch Size (BS ≥ 4)，存在一个最优的 Tree Width (例如 2 或 3)，过大的 Tree Width 反而会增加延迟，这表明在资源利用率高时，树越大开销越大。这可以为你们的**动态剪枝**提供论据。

---

## 11. Figure 11: Tree-based Parallel Decoding vs. Sequence-based Decoding (Per-token Latency)

*   **目的:** 进一步对比 SpecInfer 的树形并行解码与现有序列解码在 per-token latency 上的差异。
*   **内容:**
    *   X 轴是 Batch Size (BS=1 到 BS=16)。
    *   Y 轴是 Per-token latency (ms)。
    *   对比 Sequence-based Decoding 和 Tree-based Decoding。
*   **核心发现/要点:** Tree-based Decoding 在大 Batch Size 下比 Sequence-based Decoding 有显著优势（高达 1.8x），而在小 Batch Size 下性能接近。这归因于 Tree-based Decoding 能够消除共享前缀的重复计算和通过 Topology-Aware Causal Mask 融合 Attention 计算。

---

## 12. Table 1: Success Rate of Verifying a Token

*   **目的:** 展示在不同 top-k 候选 token 数量下，验证 token 的成功率。
*   **内容:**
    *   对比 Greedy decoding 和 Stochastic decoding。
    *   展示在五个不同数据集上，当 LLM 选定的 token 在 SSM 预测的 top-k 候选中时，验证成功的百分比。
    *   k 值从 1 到 5。
*   **核心发现/要点:** 即使 SSM 与 LLM 预测的 top-1 token 不一致，LLM 选定的 token 也很大概率在 SSM 的 top-k 候选中（例如 top-5 成功率可达 89%-97%）。这为 SpecInfer 通过多分支（例如 token tree）进行推测提供了理论依据和动机。

---

## 13. Table 2: Average Number of Tokens Verified by SpecInfer

*   **目的:** 展示 SpecInfer 在不同 Tree Width 下，平均每解码步验证的 token 数量。
*   **内容:**
    *   对比 Greedy decoding 和 Stochastic decoding。
    *   展示在五个不同数据集上，Tree Width 从 1 到 5 时，平均验证 token 数量。
*   **核心发现/要点:** 随着 Tree Width 的增加，平均每解码步验证的 token 数量显著增加，再次验证了树形结构在提高推测效率方面的有效性。

---

## 14. Table 3: Multi-Step Speculative Sampling Improvement

*   **目的:** 对比 SpecInfer 的 Multi-Step Speculative Sampling (MSS) 与 Naive Sampling (NS) 在随机采样下的性能。
*   **内容:**
    *   展示在五个不同数据集上，MSS 相对于 NS 在平均验证 token 数量上的改进倍数 (Improvement)。
*   **核心发现/要点:** MSS 能够持续地提高每解码步验证的 token 数量 (1.2-1.3x)，同时保证与 LLM 输出分布的一致性，证明了其在随机解码场景下的优势。

---