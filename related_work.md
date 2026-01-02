# Efficient Large Language Model Inference: A Survey of Speculative Decoding Techniques with Focus on Tree-Based Approaches and Dynamic Pruning

> Summary
> Autoregressive LLM inference faces fundamental bottlenecks from sequential token generation and memory bandwidth constraints, making efficient decoding critical for real-world applications. Speculative decoding addresses this by using a draft model to propose tokens for parallel verification by the target model, but linear approaches suffer from single-path inefficiency and computation waste on mismatches. Tree-based speculative decoding overcomes this by enabling multi-branch parallel token generation and verifying entire token trees in a single forward pass using tree attention mechanisms, while dynamic pruning strategies manage tree size to balance speculation depth/breadth with verification cost. Complementary methods like Medusa use multi-head prediction within a single model, and broader optimizations include KV cache management and attention algorithms. However, research gaps remain in achieving robust performance across diverse tasks and languages, developing quality-aware verification beyond strict model alignment, and creating holistic, hardware-aware systems for scalable and adaptive decoding.

## 1. Fundamental Bottlenecks in Autoregressive LLM Inference: Architectural Constraints and Efficiency Imperatives

The impressive capabilities of transformer-based Large Language Models (LLMs) are fundamentally constrained by their autoregressive inference architecture, which imposes severe limitations on real-time performance and hardware utilization. This section details the core architectural bottlenecks that necessitate advanced acceleration techniques, establishing the critical imperative for efficient inference in practical LLM deployments.

### 1.1 The Sequential Nature of Autoregressive Decoding: Inherent Latency Bottlenecks

Transformer inference operates through two distinct phases: the **prefill stage** and the **decode phase**. During prefill, the entire input sequence is processed in parallel, leveraging the GPU's computational parallelism through efficient matrix-matrix multiplications[^48]. However, the decode phase is fundamentally different and creates a persistent latency bottleneck.

The core limitation stems from the **strictly sequential token generation** inherent to autoregressive models. Each new token must be generated conditioned on all previously generated tokens, creating a step-by-step dependency where the *N*-th word cannot be predicted until the (*N*-1)-th word has been produced[^3]. This sequential decode phase forces the model to perform a full forward pass for each single token, underutilizing the GPU's parallel processing capabilities and creating a fundamental speed limit regardless of raw computational power[^3].

Unlike the prefill stage's efficient batch processing, the decode phase processes tokens one at a time, transforming operations from compute-bound matrix-matrix multiplications to less efficient, memory-bound matrix-vector multiplications[^48]. This sequential loop forces powerful GPU compute units to frequently sit idle while awaiting the next token generation cycle, resulting in poor overall hardware utilization.

### 1.2 Memory Bandwidth Constraints: The Data Movement Bottleneck

Contrary to assumptions about computational limits, LLM inference during the decode phase is predominantly **memory-bound rather than compute-bound**. Modern GPUs possess substantial floating-point operations per second (FLOPS), but the primary constraint is memory bandwidth—the speed at which data can be moved between the GPU's memory and its processing cores[^3].

For each token generated, the system must perform two massive data movements:

1. **Loading the entire model's weights**, which can be extremely large (e.g., 100+ GB for large models).
2. **Loading the entire conversation history** stored in the Key-Value (KV) Cache[^3].

This creates a severe imbalance. The operational intensity for autoregressive decoding in FP16 precision is approximately **1 FLOP/byte**, placing it far below the GPU's theoretical performance ridge point and leaving processing units idle while waiting for data transfers[^60]. The bottleneck indicates that the process is constrained by data movement speed, not raw computation[^3].

### 1.3 KV Cache Memory Explosion: Scaling Challenges with Context Length

The Key-Value (KV) cache is a critical optimization that stores intermediate states to avoid recomputing attention for previously processed tokens. However, it introduces substantial memory overhead that scales unfavorably with context length and batch size[^3].

The memory consumption of the KV cache can be formulated based on transformer architecture parameters:
$M_{\text{KV}} = B \times L \times N_{\text{layers}} \times N_{\text{heads}} \times d_{\text{head}} \times 2 \times \text{precision\_bytes}$

Where:

* $B$ = Batch size

* $L$ = Sequence length (context window)

* $N_{\text{layers}}$ = Number of transformer layers

* $N_{\text{heads}}$ = Number of attention heads per layer

* $d_{\text{head}}$ = Dimension per attention head

* $2$ = Factor for both Keys and Values

* $\text{precision\_bytes}$ = Bytes per element (e.g., 2 for FP16)

This formulation reveals explosive scaling. For instance, a KV cache representing a 128k token context for a single user with a Llama 3 70B model can consume about **40 GB of memory**[^65]. The problem is compounded by **memory fragmentation**, where systems pre-allocate large contiguous blocks for maximum possible response lengths, but most responses are shorter, leading to significant wasted memory[^3].

### 1.4 Efficiency Imperatives for Real-World LLM Applications

These architectural bottlenecks translate into critical operational challenges for deploying LLMs at scale. Efficient inference is a fundamental requirement for practical application viability across several dimensions:

**Cost Reduction and Economic Viability**: Inefficient inference directly increases infrastructure costs per token generated. High memory consumption from KV caches and poor GPU utilization due to memory bottlenecks undermine the economic sustainability of LLM services, making optimization techniques essential for cost control.

**Scalability and Throughput Constraints**: The sequential decode phase imposes a fundamental limit on request processing capacity. Systems face a difficult trade-off: processing requests individually (batch size of 1) minimizes latency for a single user but severely reduces total throughput and GPU utilization. Conversely, batching improves throughput but increases latency for individual users due to queuing delays[^3]. This creates a scalability barrier for high-demand applications.

**User Experience and Responsiveness**: In interactive applications like chatbots and real-time assistants, inference latency directly impacts user satisfaction. The sequential token generation bottleneck creates perceptible delays, especially for longer outputs. Accelerating token generation without compromising quality is essential for delivering responsive, natural-feeling interactions that meet modern user expectations.

The convergence of these architectural constraints establishes a clear imperative: advancing beyond the fundamental limitations of autoregressive decoding is essential for unlocking the full potential of LLMs in production environments. The subsequent sections explore how speculative decoding and related acceleration techniques address these bottlenecks by rethinking the sequential generation paradigm while maintaining output quality.

## 2. The Speculative Decoding Paradigm: From Core Principles to Linear Implementation

### 2.1 Foundational Concepts of Speculative Decoding

Speculative decoding is a leading optimization technique designed to overcome the fundamental bottleneck in large language model (LLM) inference: the inherently sequential nature of autoregressive generation [^3]. In standard decoding, each token requires a full forward pass of the model, leading to high latency and underutilized GPU compute capacity. The core innovation of speculative decoding is its two-model paradigm, which decouples the rapid proposal of candidate tokens from their rigorous verification [^45].

In this paradigm, a smaller, faster **draft model** rapidly proposes a sequence of candidate tokens (typically 3 to 12). This model is often a distilled or simplified version of the main model, optimized for speed. The larger, high-quality **target model** then acts as a verifier, processing the proposed tokens in parallel to ensure the final output quality is identical to what it would produce through standard autoregressive generation [^45].

The theoretical speedup of this approach is elegantly captured by the throughput equation for speculative decoding:

$$
V_{SD} = \beta_{T,D}(\ell_D + 1) \bigg/ \left(\frac{\ell_D}{\tilde{V}_D} + \frac{1}{\tilde{V}_T}\right)
$$

where $\beta_{T,D}$ is the token acceptance rate between the target and draft models, $\ell_D$ is the speculative length (number of draft tokens), and $\tilde{V}_D$ and $\tilde{V}_T$ are the processing speeds of the draft and target models, respectively [^6]. This equation highlights that the overall acceleration depends critically on achieving a high acceptance rate while minimizing the time cost of the draft generation phase.

### 2.2 Operational Mechanism of Linear Speculative Decoding

The linear, or classic, implementation of speculative decoding follows a deterministic three-phase workflow, often referred to as the draft-target approach [^45].

**1. Draft Generation Phase**
The draft model generates a sequence of $\ell_D$ candidate tokens autoregressively based on the current context. Although this step is sequential, it is significantly faster than if the target model performed it, due to the draft model's reduced size and complexity.

**2. Parallel Verification Phase**
The target model then processes the entire sequence—comprising the original input plus all $\ell_D$ draft tokens—in a single, batched forward pass. It computes probability distributions for each of the speculated token positions. This leverages the hardware efficiency of parallel computation, similar to the prefill stage in standard inference. The Key-Value (KV) cache is utilized to avoid recomputing states for the original prefix, so the computational cost is focused only on the new draft tokens [^45].

**3. Rejection Sampling and Acceptance Logic**
A rigorous, token-by-token validation is performed. For each position, the target model's probability for the draft token, $P_T(x_t)$, is compared against the draft model's proposed probability, $P_D(x_t)$. The acceptance logic is:

* If $P_T(x_t) \geq P_D(x_t)$, the token is accepted.

* If $P_T(x_t) < P_D(x_t)$, the token is rejected, and all subsequent tokens in the draft sequence are discarded [^45].

When a rejection occurs, the process reverts to standard autoregressive generation from the last accepted token to produce a corrected continuation. This strict validation guarantees that the final output distribution is mathematically identical to that of the target model alone, preserving quality [^45].

### 2.3 Performance Characteristics and the Acceptance Rate Trade-off

The key performance metric is the **Token Acceptance Rate (TAR)**, which represents the average number of draft tokens accepted by the target model per iteration. Empirical analysis reveals a fundamental trade-off: while a larger draft model typically achieves a higher TAR due to better alignment with the target model's distribution, its increased inference latency can offset the gains in parallelism [^6]  [^78].

Research indicates that a draft model's accuracy on standard language modeling tasks does not strongly correlate with its TAR in speculative decoding. This suggests that effective drafting relies on predicting common sequences well, rather than matching the target model's full generative capabilities [^78]. Furthermore, draft model latency is heavily bottlenecked by model depth; deeper models incur significantly higher latency, which can diminish or even reverse the expected throughput benefits [^78].

### 2.4 Critical Limitations of Linear Speculative Decoding

Despite its conceptual elegance, linear speculative decoding suffers from several inherent inefficiencies that motivate more advanced approaches.

**1. Single-Path Guessing Inefficiency**
The draft model proposes only a single linear sequence of tokens. If an early token in this sequence is rejected, the computational effort spent generating and verifying all subsequent tokens is entirely wasted, forcing a restart from the point of divergence. This single-path exploration is inefficient when the draft model's predictions frequently diverge from the target's preferred path [^78].

**2. Quadratic Computation Waste**
The rejection process leads to quadratic waste in the worst case: the draft model's sequential work to generate $\ell_D$ tokens and the target model's parallel verification pass are both discarded if the first token is rejected. This waste becomes pronounced when there is significant distributional divergence between the draft and target models [^78].

**3. Reliance on Separate Draft Models**
Maintaining a separate, trained draft model introduces practical deployment challenges, including additional memory footprint, potential distribution shifts, and the computational cost of pre-training or fine-tuning. This requirement complicates system integration and limits adaptability [^45].

**4. The Draft Model Bottleneck Paradox**
Analysis of model families like OPT reveals a paradox: although TAR consistently increases with draft model size, overall system throughput can eventually decrease. This is because the gains from higher acceptance are outweighed by the quadratic increase in draft generation latency, creating a complex optimization landscape for selecting the optimal draft model [^78].

These limitations underscore that while linear speculative decoding provides a foundational speedup mechanism, its efficiency is constrained by the draft model's single-path exploration and the delicate balance between acceptance rate and latency. This sets the stage for more sophisticated paradigms like tree-based speculative decoding, which aims to overcome these constraints by exploring multiple token possibilities in parallel.

## 3. Tree-Based Speculative Decoding: Multi-Branch Parallelism and Tree Attention Mechanisms

Tree-based speculative decoding represents a significant evolution beyond linear approaches by fundamentally addressing the inefficiency of single-path guessing. While linear speculative decoding suffers from quadratic computation waste on mismatches and limited exploration of the token space, tree-based methods enable multi-branch parallel token generation. This is achieved through sophisticated tree attention mechanisms that verify entire token forests in a single forward pass while rigorously preserving generation correctness, thereby unlocking substantial improvements in inference throughput and hardware utilization.

### 3.1 Overcoming Single-Path Limitations with Multi-Branch Exploration

Linear speculative decoding employs a draft model to propose a fixed, sequential chain of K tokens for verification by the target model. A core limitation of this approach is its **inefficient single-path guessing**: if any token in the proposed sequence fails verification, the entire subsequent draft computation is wasted. This leads to a quadratic relationship between draft length and potential computational waste[^3]. Furthermore, the linear chain cannot explore alternative token continuations when the initial prediction path diverges from what the target model would generate, limiting the potential acceptance length.

Tree-based speculative decoding transforms this paradigm by constructing a **draft token tree**. In this structure, each node represents a potential token, and branches represent alternative hypotheses about future token sequences[^44]. This multi-branch architecture allows for the parallel exploration of multiple potential futures from a given context. The key advantage is that the system is no longer constrained to a single prediction chain; if one branch proves incorrect, sibling branches can still be valid, allowing recovery from early errors without discarding the entire speculative effort. This parallel exploration dramatically increases the probability of finding and accepting longer sequences of correct tokens in each verification step.

### 3.2 Tree Attention: Efficient Parallel Verification of Hierarchical Structures

The enabling innovation for tree-based decoding is the **tree attention mechanism**. Standard transformer attention is designed for linear sequences, but tree attention extends this capability to handle hierarchical token structures through specialized, causality-preserving attention masks[^44].

The mechanism works by allowing the target model to process all tokens in the tree simultaneously in one forward pass. To maintain the strict causal dependencies of autoregressive generation, each token in the tree is configured to attend only to its ancestors along the unique path back to the root node. This is enforced by a carefully constructed attention mask that encodes the tree's topology, ensuring that the conditional probability relationships required for mathematically correct generation are preserved[^45]. The parallel verification of this entire structure, as opposed to verifying a single linear chain, is the source of significant efficiency gains.

Advanced implementations, such as the **Traversal Verification** algorithm, employ a **leaf-to-root traversal strategy** during verification[^82]. This approach evaluates entire candidate sequences from the leaves backward, using sequence-level acceptance criteria based on joint probability distributions. This method ensures that if a node is accepted, the entire sequence from that node to the root is accepted, while rejected nodes trigger the verification of sibling alternatives, thereby maximizing the utilization of the drafted tokens and improving overall acceptance rates[^82].

### 3.3 Adaptive Tree Construction and Dynamic Pruning

A critical advancement in tree-based decoding is the move from static tree shapes to dynamically constructed, adaptive trees optimized for each decoding step. Algorithms like **OPT-Tree** explicitly search for tree structures that maximize the mathematical expectation of the acceptance length at each step[^13]. These methods typically employ a greedy construction strategy, starting from the root and iteratively expanding nodes with the highest estimated path probabilities, continuing as long as the expected gain in accepted tokens outweighs the increased verification cost.

Dynamic pruning strategies are essential for managing computational overhead. Frameworks like **ProPD** integrate an **early pruning mechanism** to eliminate unpromising branches before they incur the full cost of verification[^15]. ProPD adds a lightweight prediction head after a few initial LLM layers to identify the top-k most probable successor tokens for each node. If a generated token falls outside this set for its parent, the entire sequence containing that token is pruned. This approach can reduce verification computation by over 2x without harming the final count of accepted tokens[^15]. The system dynamically balances the depth and breadth of the tree against verification time, optimizing for overall throughput using real-time performance estimates.

### 3.4 Performance Advantages and Implementation Considerations

Tree-based speculative decoding demonstrates clear empirical advantages. By exploring multiple branches in parallel, it achieves higher token acceptance rates and longer average accepted sequences compared to linear drafting, leading to greater overall speedup. For instance, optimized production-scale implementations like **EAGLE-3**, which utilizes a dynamic draft tree and parallel tree attention, have demonstrated inference latencies as low as **4 ms per token** for Llama models on modern GPU hardware, representing a significant improvement over previous methods[^81].

Successful integration into production systems requires addressing several practical challenges:

1. **Efficient Kernel Implementation**: Specialized GPU kernels are needed to compute tree attention efficiently, as standard attention implementations are optimized for linear sequences.
2. **Memory Management**: Storing multiple token hypotheses and their associated attention masks introduces additional memory overhead. Efficient implementations cache and reuse attention masks where possible[^15].
3. **Dynamic Batching**: Handling variable-sized trees across different requests in a batch requires sophisticated batching and padding strategies to maintain high GPU utilization.

### 3.5 Preservation of Correctness

A foundational requirement for any speculative decoding technique is the guarantee of **lossless inference**—the accelerated process must produce outputs identical in distribution to those generated by the target model's standard autoregressive decoding. Tree-based methods preserve this guarantee through their tree attention mechanism. By ensuring each token only attends to its correct ancestral path, the model computes the exact same conditional probabilities as it would in sequential generation. Formal proofs, such as those provided in the Traversal Verification work, establish this mathematical equivalence, ensuring no degradation in output quality for the sake of speed[^82].

In summary, tree-based speculative decoding represents a paradigm shift, transforming the exploration-verification trade-off in LLM inference. By enabling multi-branch parallel token generation through adaptive tree construction and efficient tree attention verification, these methods unlock substantial latency reductions and throughput improvements while maintaining the quality guarantees essential for real-world deployment.

## 4. Dynamic Pruning Strategies: Balancing Speculation Depth, Breadth, and Verification Cost

Dynamic pruning strategies are a critical advancement in tree-based speculative decoding, addressing the fundamental trade-off between exploring potential token sequences and managing the computational cost of verifying them. These mechanisms intelligently control the size and structure of draft token trees in real-time, optimizing the balance between speculation depth (the number of sequential tokens predicted), breadth (the number of parallel candidate tokens at each step), and the associated verification overhead. By pruning unpromising branches early and dynamically adjusting the tree's growth, these strategies maximize overall inference speedup and GPU utilization while preserving generation quality.

### 4.1 Early Pruning Mechanisms and Top-k Selection Criteria

Early pruning operates during the initial stages of tree construction to eliminate unpromising token sequences before they incur significant computational cost. The ProPD framework implements an effective early pruning mechanism that reduces computation by over 2× without harming the number of accepted tokens[^15]. Its core mechanism involves a Top-k-based selection criterion. After processing the input through a few initial layers of the model, an early prediction head generates a list of the Top-k most probable successor tokens for each position. If a proposed next token is not within this Top-k list, all sequences containing that token are deemed implausible and pruned from further expansion[^15]. To minimize latency, ProPD optimizes implementation by caching and subsampling the attention mask on the GPU instead of regenerating it for each pruning decision[^15].

A similar probability-based pruning approach is used in Retrieval-Augmented Speculative Decoding (RASD). It leverages the strong positive correlation between a draft model's confidence score and the eventual token acceptance rate. Specifically, RASD prunes retrieval results whose first token is not within the top-k of the draft model's initial output probability distribution, thereby constructing a more efficient retrieval tree[^19].

### 4.2 Dynamic Termination Conditions and Acceptance Probability Thresholds

Beyond early pruning, dynamic termination mechanisms decide when to stop expanding the tree based on acceptance probability thresholds. These strategies calculate the expected utility of generating additional tokens, pruning expansions that are likely to increase overall inference latency. The underlying principle is that the draft model's confidence scores are well-calibrated with the probability of a token being accepted by the target model. For a given drafting step, if the estimated acceptance probability ( p\_j ) falls below a threshold determined by the relative forward pass times of the draft and target models (( s\_d ) and ( s\_t )), then that step is pruned because its expected time gain is negative[^15]. This ensures computational resources are only spent on branches with a high likelihood of contributing to speedup.

### 4.3 Real-Time Efficiency Optimization with Performance Modeling

Advanced dynamic pruning systems employ real-time performance modeling to make optimal tree construction decisions. ProPD uses a weighted regression model to estimate verification overhead based on token tree size, leveraging the observed linear relationship between iteration time and the number of tokens to verify[^15]. The model continuously updates its estimate of the average iteration time ( T\_i^{perf} ) using an exponential moving average: ( T\_i^{perf} \leftarrow (1-\alpha)T\_i^{perf} + \alpha t\_i ), where ( t\_i ) is the latest observed time. Recent observations are prioritized through calculated weights ( W\_i = e^{-\lambda o\_i} ). The regression coefficients ( \hat{\beta}\_0, \hat{\beta}_1 ) are determined by minimizing the weighted error sum: ( \sum_{i=1}^S W\_i (T\_i^{perf} - (\beta\_0 + \beta\_1 i))^2 )[^15].

Concurrently, ProPD tracks the runtime accuracy of each prediction head to estimate the probable acceptance length of a candidate sequence. For a head ( i ), it updates the probability ( P\_k^i ) that the ground truth token is within its Top-k predictions. The probable accuracy of the k-th highest probability token is then ( p\_k^i = P\_k^i - P\_{k-1}^i ). For a sequence of tokens, the probable acceptance length ( l(\text{seq}) ) is the product of these accuracies: ( l(\text{seq}) = \prod\_{i=0}^n p\_k^i )[^15]. By calculating the estimated speed ( v = l(i) / T\_i^{est} ) for different tree sizes ( i ), the system can identify the optimal tree structure for maximum throughput.

### 4.4 Breadth and Depth Pruning in Cost-Aware Frameworks

The CAST (Cost-Aware Speculative Tree) framework provides a systematic approach that integrates the impact of batching and GPU hardware into dynamic tree construction[^83]. It formalizes pruning across two dimensions: breadth and depth.

**Breadth Pruning** determines how many candidate nodes to retain in each layer of the draft tree. It frames node selection as a utility maximization problem. For layer ( i ), it calculates the cumulative utility ( u\_k^{(i)} ) of selecting the top ( k ) nodes based on their confidence scores, normalized by the computational cost ( c\_k^{(i)} ) of verifying them[^83]. The algorithm retains nodes whose marginal utility exceeds a predefined threshold, effectively prioritizing tokens with high acceptance likelihood while accounting for their verification expense.

**Depth Pruning** decides whether to generate an additional layer in the tree. It proceeds to layer ( (i+1) ) only if the product of the average predictive quality ( \alpha\_i ) and the utility-to-cost ratio of the currently retained nodes exceeds another threshold ( C\_2 )[^83]. This ensures the tree does not grow deeper when the draft model's predictions become too uncertain to justify the extra verification cost.

### 4.5 Performance Characteristics and Strategic Comparisons

Different dynamic pruning strategies exhibit distinct performance characteristics based on their underlying mechanisms. ProPD, with its combination of early pruning and weighted regression modeling, is designed to balance computation and parallelism in real-time, maximizing efficiency across varying batch sizes, sequence lengths, and tasks[^15]. The CAST framework generalizes cost-aware decision-making by systematically considering batching and hardware factors often neglected in heuristic approaches, using pre-measured lookup tables for precise cost estimation[^83]. DySpec employs a greedy dynamic expansion strategy guided by the strong correlation between the draft model's output distribution and the acceptance rate, allowing it to adapt to diverse query distributions for higher throughput[^42].

These strategies collectively address the verification overhead bottleneck that limits traditional speculative decoding. By intelligently managing tree size and structure, they strike an optimal balance between exploration (speculation depth and breadth) and exploitation (verification cost), which is essential for maximizing inference acceleration while maintaining the quality of the generated text.

## 5. Alternative Parallel Decoding Approaches: Medusa and Multi-Token Prediction Methods

Beyond tree-based speculative decoding, a distinct class of parallel acceleration techniques has emerged that eliminates the need for a separate draft model entirely. These methods focus on enabling multi-token prediction within a single model architecture, thereby addressing the distribution shift problem inherent in the traditional draft-target paradigm. This section examines the core mechanisms, advantages, and limitations of these approaches, with a particular focus on Medusa and the "Your LLM Knows the Future" framework.

### 5.1 Medusa: Multiple Decoding Heads for Single-Model Acceleration

Medusa accelerates inference by augmenting a base LLM with multiple lightweight decoding heads, rather than relying on an external draft model[^24]. The core mechanism involves adding *K* extra decoding heads to the model's last hidden state $h_t$. Each head is tasked with predicting a future token at a specific offset: the *k*-th head predicts the token at position $(t + k + 1)$, while the original model head continues to predict the immediate next token at $(t + 1)$[^23]. The prediction of the *k*-th head is defined as a probability distribution over the vocabulary:

$$
p^{(k)}_t = \text{softmax} \left( W^{(k)}_2 \cdot \left( \text{SiLU} (W^{(k)}_1 \cdot h_t) + h_t \right) \right)
$$

where $W^{(k)}_1 \in \mathbb{R}^{d \times d}$ and $W^{(k)}_2 \in \mathbb{R}^{d \times V}$ are learnable weights, *d* is the hidden dimension, and *V* is the vocabulary size[^23]. This design allows multiple future tokens to be proposed in parallel from a single forward pass of the base model.

Medusa offers two primary training strategies to integrate these heads[^23]:

1. **MEDUSA-1 (Frozen Backbone)**: The parameters of the base LLM are kept frozen, and only the newly added Medusa heads are fine-tuned using a weighted cross-entropy loss.
2. **MEDUSA-2 (Joint Training)**: The Medusa heads are trained jointly with the backbone LLM, using a combined loss function and differential learning rates to preserve the model's original next-token prediction capability.

For verification, Medusa employs a **tree attention mechanism**. The top predictions from each head are combined via a Cartesian product to form multiple candidate continuation sequences. A specialized attention mask ensures that during a single verification pass by the target model, each token only attends to its correct predecessors within its candidate sequence, allowing all branches of the candidate tree to be verified in parallel[^23]. This process maintains output correctness identical to standard autoregressive generation.

### 5.2 The "Your LLM Knows the Future" Framework: Masked-Input Multi-Token Prediction

A more recent innovation challenges the approach of adding new prediction heads. The "Your LLM Knows the Future" framework proposes that a standard autoregressive LLM already possesses implicit knowledge about future tokens, which can be unlocked through targeted fine-tuning without substantial architectural changes[^97]. Its key innovations include:

1. **Masked-Input Formulation**: During training, unique mask tokens are appended to the input sequence, prompting the model to jointly predict multiple future tokens from a common prefix.
2. **Gated LoRA**: A gated Low-Rank Adaptation (LoRA) mechanism is used during fine-tuning. The LoRA parameters are activated only for the masked positions designated for multi-token prediction, ensuring the model's original next-token prediction behavior remains completely unchanged for non-masked tokens.
3. **Lightweight Sampler**: A small, learnable two-layer perceptron acts as a sampler head to generate coherent sequences from the model's parallel predictions, replacing more complex beam search.
4. **Quadratic Decoding Strategy**: Instead of a linear chain of speculated tokens, this framework interleaves mask tokens to create a quadratic expansion of candidate tokens. This design guarantees a consistent number of new speculative tokens are available at each step, even if some are rejected, maintaining steady decoding progress[^97].

This approach fundamentally differs from Medusa by aiming to "augment the input sequence to prompt the *pretrained model itself* to generate multiple token predictions," thereby parallelizing computation within the existing architecture with minimal added components[^97]. Reported speedups are significant, such as nearly 5× for code and math tasks and about 2.5× for general chat[^97].

### 5.3 Other Notable Parallel Decoding Strategies

Several other techniques explore parallel token generation from different angles:

* **Jacobi Decoding**: This is an algorithmic approach that attempts to break sequential dependencies by predicting multiple tokens simultaneously through parallel fixed-point iteration. A key limitation is that it often accurately predicts only one token per iteration, requiring multiple correction steps which can limit practical speedup[^102]. Enhanced versions that refine the LLM to converge faster to a fixed point have shown improved acceleration[^102].

* **Parallel Decoding via Hidden Transfer**: This method enables parallel generation of multiple successive tokens in one forward pass by using pseudo hidden states and a tree attention mechanism. It is reported to outperform Medusa and other single-model acceleration techniques in terms of inference speed while balancing semantic accuracy[^103].

### 5.4 Comparative Analysis

The following table summarizes the core characteristics of these alternative parallel decoding approaches in comparison to the traditional speculative decoding paradigm.

| **Approach**                         | **Core Mechanism**                                                                                  | **Architectural Changes**                          | **Training Complexity**                                        | **Key Advantages**                                                  | **Key Limitations**                                                                            |
| ------------------------------------ | --------------------------------------------------------------------------------------------------- | -------------------------------------------------- | -------------------------------------------------------------- | ------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------- |
| **Traditional Speculative Decoding** | Separate draft model proposes tokens for target model verification.                                 | Requires a separate draft model.                   | High (requires training/maintaining a draft model).            | Well-established; predictable behavior.                             | Distribution shift between models; draft model maintenance overhead.                           |
| **Medusa**                           | Multiple decoding heads added to the backbone LLM predict future tokens in parallel.                | Additional prediction heads attached to the model. | Moderate (fine-tuning of heads, with or without the backbone). | Eliminates separate draft model; avoids major distribution shift.   | Increased parameter count; training overhead for heads.                                        |
| **"Your LLM Knows the Future"**      | Masked-input prompting with gated LoRA fine-tuning unlocks model's inherent multi-token prediction. | Minimal (LoRA adapters + lightweight sampler).     | Low (parameter-efficient fine-tuning with LoRA).               | Leverages existing model capacity; minimal architectural intrusion. | Requires specific fine-tuning and prompt engineering.                                          |
| **Jacobi Decoding**                  | Parallel fixed-point iteration to predict multiple tokens simultaneously.                           | None (purely algorithmic).                         | None (if using a pre-trained model as-is).                     | No model modifications required.                                    | Poor token prediction per iteration often necessitates multiple corrections, limiting speedup. |

**Evolution of Design Philosophy**: The progression from Medusa's added prediction heads to the "Your LLM Knows the Future" framework's masked-input prompting represents a significant shift. The trend moves from augmenting the model's architecture to leveraging and refining the inherent parallel computation potential already present in the pre-trained transformer. This suggests a future direction toward more minimalistic, prompt-based, and parameter-efficient acceleration techniques that preserve the original model's integrity while achieving substantial inference speedups.

## 6. Broader LLM Inference Optimization Landscape: KV Cache Management and Attention Optimizations

While speculative decoding directly targets the sequential bottleneck of autoregressive token generation, efficient LLM inference requires a holistic approach that addresses several other critical architectural constraints. These complementary optimization techniques primarily focus on mitigating memory bandwidth limitations, managing the explosive growth of the Key-Value (KV) cache, and reducing the computational complexity of attention mechanisms. This section provides a brief overview of these broader categories to contextualize the field.

### 6.1 KV Cache Management for Long Contexts

The KV cache is a core optimization that stores intermediate key and value states from the attention mechanism for previously processed tokens, preventing their recomputation at each decoding step. However, its size grows linearly with both sequence length and batch size, leading to excessive memory consumption that can limit context windows and concurrent request handling[^3].

**StreamingLLM and Attention Sinks** offer a solution for infinite-length streaming applications. The framework is based on the observation that LLMs trained with finite attention windows allocate a significant portion of attention score to the initial tokens, regardless of their semantic relevance. These tokens act as "attention sinks," stabilizing the SoftMax computation. StreamingLLM maintains the KV states of these initial sink tokens (e.g., the first four tokens) alongside a rolling cache of the most recent tokens. This strategy prevents the performance collapse seen in simple sliding window approaches and enables models to handle sequences of millions of tokens without fine-tuning, achieving significant speedups in streaming settings[^34]  [^35].

**Paged Attention**, as implemented in systems like vLLM, tackles the problem of memory fragmentation within the KV cache. Traditional inference systems often pre-allocate large, contiguous memory blocks for each request, anticipating maximum response lengths. This leads to substantial wasted memory—estimates suggest 60-80% of GPU memory can be unused due to fragmentation[^3]. Paged Attention applies virtual memory concepts, dividing the KV cache into fixed-size blocks. This allows for non-contiguous, dynamic allocation per token, dramatically improving memory utilization, supporting larger batch sizes, and enabling more efficient sharing of cache across beams in beam search.

### 6.2 Optimized Attention Mechanisms

The standard self-attention mechanism has quadratic computational complexity with respect to sequence length, becoming a major bottleneck for long contexts. Algorithmic innovations have been developed to alleviate this.

**FlashAttention** is a seminal optimization that redesigns the attention algorithm to be memory-aware. Instead of writing the large intermediate attention matrix to slow GPU memory (HBM), FlashAttention performs the entire attention computation in faster on-chip SRAM by tiling the inputs and recomputing parts of the computation on-the-fly. This drastically reduces the number of memory reads/writes, which are often the true bottleneck, leading to substantial speedups and enabling longer context lengths during both training and inference[^110].

### 6.3 Memory Optimization and Quantization

Beyond specific cache and attention optimizations, general memory reduction techniques are critical for deploying large models.

**Quantization** reduces the numerical precision of model weights and activations, directly decreasing memory footprint and bandwidth requirements. Common approaches include post-training quantization (e.g., converting FP32 weights to INT8 for a 4x memory reduction) and more advanced quantization-aware training. Specialized KV cache quantization further targets the memory consumed by growing sequences[^3].

### 6.4 Synergy with Speculative Decoding

These broader optimizations are not isolated; they synergize powerfully with speculative decoding strategies:

* **Enhanced Verification**: Optimized attention kernels like FlashAttention accelerate the single forward pass required to verify an entire speculative token tree.

* **Increased Capacity**: Efficient KV cache management via Paging or StreamingLLM frees up GPU memory, allowing for larger batch sizes or more extensive speculative tree exploration without hitting memory limits.

* **Alleviated Bottlenecks**: Weight and activation quantization reduce the memory bandwidth pressure, which benefits all decoding phases, including the parallel verification step in speculative decoding.

In summary, achieving optimal LLM inference speed requires a co-designed stack addressing sequential decoding, memory bandwidth, KV cache growth, and attention computation. Speculative decoding is a key component within this broader ecosystem, and its effectiveness is often amplified when combined with the other optimization techniques outlined here.

## 7. Research Gaps and Future Directions: Toward More Robust, Parallel, and Adaptive Decoding

The landscape of LLM inference acceleration, particularly through speculative decoding, has made significant strides in addressing the fundamental bottleneck of autoregressive decoding. However, a critical examination reveals substantial limitations that constrain the practical deployment and scalability of current approaches. This section synthesizes the key research gaps across several critical dimensions, drawing from recent literature to outline the unmet needs for more robust, parallel, and adaptive decoding strategies.

### 7.1 Inconsistent Performance Across Diverse Inputs

A primary limitation of current speculative decoding methods is their inability to deliver consistent speedups across varying tasks, languages, and context lengths. Performance can degrade severely outside the narrow conditions of common benchmarks, which often assume short contexts (e.g., 2K tokens) and a batch size of one[^116]. For instance, the OWL paper identifies that methods like EAGLE3 can actually slow down generation by 0.81× when processing long-context inputs, achieving an acceptance length of only 1.28 tokens per verification step[^116]. This failure to generalize to real-world, long-context workloads—such as document analysis or extended multi-turn dialogues—is a critical barrier to practical application.

Furthermore, performance disparities extend to multilingual and multi-task settings. Research indicates that speed-up rates can vary significantly, with underrepresented languages and tasks experiencing markedly lower acceleration[^114]. This inconsistency stems from differences in how well the draft model aligns with the target model's distribution across different domains, leading to an inequitable and unreliable user experience.

### 7.2 Limitations in Verification Logic and Quality Assessment

The core verification mechanism in standard speculative decoding exhibits a fundamental flaw. As highlighted by the Judge Decoding paper, the strict requirement for **alignment** between the draft and target model's token predictions leads to the rejection of many high-quality draft tokens that represent objectively valid continuations[^117]. This quality-agnostic acceptance criterion, which only accepts a token if the draft's top prediction exactly matches the target's, severely limits the potential speedup. Early rejection becomes overwhelmingly likely, even when using powerful draft models or human-written text, because the system cannot recognize correct but non-aligned replies[^117].

While tree-based approaches mitigate the inefficiency of single-path guessing, they introduce new challenges related to verification overhead. The verification cost scales with the size of the token tree, creating a trade-off between speculation depth/breadth and computational waste. Frameworks like ProPD address this by introducing dynamic pruning mechanisms to eliminate unpromising token sequences early[^15]. However, the broader challenge remains: how to design verification processes that are both efficient and capable of assessing token quality beyond strict token-by-token matching.

### 7.3 Lack of Holistic System Optimization

Current speculative decoding approaches often suffer from fragmented optimization. The SPIN paper identifies three systemic limitations: (1) using homogeneous draft models for requests of varying difficulty, (2) inadequate support for efficient batch processing, and (3) isolated optimization of the speculation and verification phases without holistic system orchestration[^115]. This lack of integration prevents comprehensive throughput gains in real serving environments.

This issue is echoed in the broader inference acceleration field. Techniques for KV cache management (e.g., StreamingLLM[^34]) and optimized attention (e.g., FlashAttention) are often developed and evaluated in isolation[^114]. There is a notable absence of comprehensive evaluation for batched speculative decoding, making it difficult to assess real-world speedup across varying batch sizes[^114]. Future systems require co-designed architectures that seamlessly integrate dynamic drafting, adaptive verification, and efficient memory management.

### 7.4 Challenges in Scalability and Hardware Co-Design

Scaling speculative decoding to distributed systems and edge deployments presents unresolved challenges. In decentralized settings, network latency can dominate, and the behavior of speculative decoding under these conditions is not well characterized[^43]. While frameworks like Decentralized Speculative Decoding (DSD) show promise by turning communication latency into computation throughput, issues like cross-node synchronization for tree verification and heterogeneous hardware capabilities remain[^43].

Furthermore, most acceleration research operates within existing hardware constraints. Truly novel gains may require hardware-algorithm co-design. As noted in surveys, existing accelerators often miss fine-grained, collaborative optimization opportunities between computation and memory[^122]. Specialized hardware that natively supports dynamic tree structures, probabilistic pruning, and the parallel verification of token trees could unlock significant efficiency gains. Edge deployments add another layer of complexity, requiring solutions that carefully balance latency, energy consumption, and model accuracy[^124].

### 7.5 Synthesis: The Path Forward for Adaptive Decoding

The identified gaps collectively point toward the need for next-generation decoding frameworks built on principles of adaptability, robustness, and holistic optimization. Key future directions include:

* **Dynamic and Adaptive Mechanisms:** Decoding systems must adapt in real-time to input complexity, model uncertainty, and hardware state. This could involve confidence-modulated drafting, dynamic tree expansion based on acceptance probability correlations[^42], and adaptive layer-skipping[^119].

* **Quality-Aware Verification:** Moving beyond strict alignment to verification schemes that can judge the contextual correctness of a token sequence, perhaps through lightweight "judge" modules trained on the target model's embeddings[^117].

* **Cross-Domain Robustness:** Developing speculation strategies that ensure consistent acceleration across diverse tasks, languages, and context lengths, mitigating the performance disparities observed in current methods.

* **Unified System Co-Design:** Creating integrated serving systems that co-optimize all components—speculation, verification, KV cache management, and attention—with native support for batching and heterogeneous hardware[^115].

The research on tree-based speculative decoding with dynamic pruning is central to this evolution. By enabling multi-branch parallel token generation coupled with intelligent pruning and advanced verification, this paradigm directly addresses the core tension between parallelism and quality. It provides a foundational approach for building the robust, scalable, and efficient decoding engines required for the widespread deployment of next-generation LLMs.

## 8. Reference

[^1]: \[PDF] Hardware-Aware Parallel Prompt Decoding for Memory-Efficient ..., <a href="https://aclanthology.org/2025.findings-emnlp.120.pdf" target="_blank"><https://aclanthology.org/2025.findings-emnlp.120.pdf></a>

[^2]: \[Quick Review] Decoding Speculative Decoding - Liner, <a href="https://liner.com/review/decoding-speculative-decoding" target="_blank"><https://liner.com/review/decoding-speculative-decoding></a>

[^3]: Decoding Real-Time LLM Inference: A Guide to the Latency vs ..., <a href="https://medium.com/learnwithnk/decoding-real-time-llm-inference-a-guide-to-the-latency-vs-throughput-bottleneck-c1ad96442d50" target="_blank"><https://medium.com/learnwithnk/decoding-real-time-llm-inference-a-guide-to-the-latency-vs-throughput-bottleneck-c1ad96442d50></a>

[^4]: Hardware Acceleration of LLMs - Emergent Mind, <a href="https://www.emergentmind.com/topics/hardware-acceleration-of-llms" target="_blank"><https://www.emergentmind.com/topics/hardware-acceleration-of-llms></a>

[^5]: An Introduction to Speculative Decoding for Reducing Latency in AI ..., <a href="https://forums.developer.nvidia.com/t/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/345286" target="_blank"><https://forums.developer.nvidia.com/t/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/345286></a>

[^6]: 3-Model Speculative Decoding - arXiv, <a href="https://arxiv.org/html/2510.12966v1" target="_blank"><https://arxiv.org/html/2510.12966v1></a>

[^7]: Speculative Decoding: Speeding LLM Generation - Emergent Mind, <a href="https://www.emergentmind.com/topics/speculative-decoding-technique" target="_blank"><https://www.emergentmind.com/topics/speculative-decoding-technique></a>

[^8]: \[PDF] Faster In-Context Learning for LLMs via N-Gram Trie Speculative ..., <a href="https://aclanthology.org/2025.emnlp-main.911.pdf" target="_blank"><https://aclanthology.org/2025.emnlp-main.911.pdf></a>

[^9]: Training-Free Loosely Speculative Decoding - OpenReview, <a href="https://openreview.net/forum?id=JjoTg34YiU" target="_blank"><https://openreview.net/forum?id=JjoTg34YiU></a>

[^10]: GitHub - Geralt-Targaryen/Awesome-Speculative-Decoding, <a href="https://github.com/Geralt-Targaryen/Awesome-Speculative-Decoding" target="_blank"><https://github.com/Geralt-Targaryen/Awesome-Speculative-Decoding></a>

[^11]: Speculative decoding with CTC-based draft model for LLM inference ..., <a href="https://dl.acm.org/doi/10.5555/3737916.3740839" target="_blank"><https://dl.acm.org/doi/10.5555/3737916.3740839></a>

[^12]: Unveil Speculative Decoding's Potential for Accelerating Sparse MoE, <a href="https://arxiv.org/html/2505.19645v3" target="_blank"><https://arxiv.org/html/2505.19645v3></a>

[^13]: OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure, <a href="https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00735/128189/OPT-Tree-Speculative-Decoding-with-Adaptive-Draft" target="_blank"><https://direct.mit.edu/tacl/article/doi/10.1162/tacl_a_00735/128189/OPT-Tree-Speculative-Decoding-with-Adaptive-Draft></a>

[^14]: Looking back at speculative decoding - Google Research, <a href="https://research.google/blog/looking-back-at-speculative-decoding/" target="_blank"><https://research.google/blog/looking-back-at-speculative-decoding/></a>

[^15]: ProPD: Dynamic Token Tree Pruning and Generation for LLM ... - arXiv, <a href="https://arxiv.org/abs/2402.13485" target="_blank"><https://arxiv.org/abs/2402.13485></a>

[^16]: ProPD: Dynamic Token Tree Pruning and Generation for LLM ..., <a href="https://www.researchgate.net/publication/390632197_ProPD_Dynamic_Token_Tree_Pruning_and_Generation_for_LLM_Parallel_Decoding" target="_blank"><https://www.researchgate.net/publication/390632197_ProPD_Dynamic_Token_Tree_Pruning_and_Generation_for_LLM_Parallel_Decoding></a>

[^17]: Parallel Token Generation - Emergent Mind, <a href="https://www.emergentmind.com/topics/parallel-token-generation" target="_blank"><https://www.emergentmind.com/topics/parallel-token-generation></a>

[^18]: \[PDF] COST-AWARE DYNAMIC TREE CONSTRUCTION FOR EFFICIENT ..., <a href="https://openreview.net/pdf/b4859349ce05c687589dbdbf95088dc4826117c0.pdf" target="_blank"><https://openreview.net/pdf/b4859349ce05c687589dbdbf95088dc4826117c0.pdf></a>

[^19]: \[PDF] arXiv:2503.03434v1 \[cs.CL] 5 Mar 2025, <a href="https://arxiv.org/pdf/2503.03434" target="_blank"><https://arxiv.org/pdf/2503.03434></a>

[^20]: \[PDF] Efficient and Scalable Speculative Decoding with Multi-Stream ..., <a href="https://aclanthology.org/2025.emnlp-main.986.pdf" target="_blank"><https://aclanthology.org/2025.emnlp-main.986.pdf></a>

[^21]: \[PDF] DySpec: Faster speculative decoding with dynamic token tree structure, <a href="https://www.wict.pku.edu.cn/docs/20250717175203347541.pdf" target="_blank"><https://www.wict.pku.edu.cn/docs/20250717175203347541.pdf></a>

[^22]: Dynamic Token Tree Pruning and Generation for LLM Parallel ..., <a href="https://www.themoonlight.io/en/review/propd-dynamic-token-tree-pruning-and-generation-for-llm-parallel-decoding" target="_blank"><https://www.themoonlight.io/en/review/propd-dynamic-token-tree-pruning-and-generation-for-llm-parallel-decoding></a>

[^23]: Dynamic Draft Trees in Autoregressive Decoding - Emergent Mind, <a href="https://www.emergentmind.com/topics/dynamic-draft-trees" target="_blank"><https://www.emergentmind.com/topics/dynamic-draft-trees></a>

[^24]: Medusa: Simple LLM Inference Acceleration Framework with ... - arXiv, <a href="https://arxiv.org/abs/2401.10774" target="_blank"><https://arxiv.org/abs/2401.10774></a>

[^25]: Medusa: Simple Framework for Accelerating LLM Generation with ..., <a href="https://github.com/FasterDecoding/Medusa" target="_blank"><https://github.com/FasterDecoding/Medusa></a>

[^26]: Medusa: Simple LLM Inference Acceleration Framework with ... - Liner, <a href="https://liner.com/review/medusa-simple-llm-inference-acceleration-framework-with-multiple-decoding-heads" target="_blank"><https://liner.com/review/medusa-simple-llm-inference-acceleration-framework-with-multiple-decoding-heads></a>

[^27]: MEDUSA: Simple LLM inference acceleration framework with ..., <a href="https://dl.acm.org/doi/10.5555/3692070.3692273" target="_blank"><https://dl.acm.org/doi/10.5555/3692070.3692273></a>

[^28]: Achieve \~2x speed-up in LLM inference with Medusa-1 on Amazon ..., <a href="https://aws.amazon.com/blogs/machine-learning/achieve-2x-speed-up-in-llm-inference-with-medusa-1-on-amazon-sagemaker-ai/" target="_blank"><https://aws.amazon.com/blogs/machine-learning/achieve-2x-speed-up-in-llm-inference-with-medusa-1-on-amazon-sagemaker-ai/></a>

[^29]: MEDUSA: Simple LLM Inference Acceleration Framework with ..., <a href="https://experts.illinois.edu/en/publications/medusa-simple-llm-inference-acceleration-framework-with-multiple-/" target="_blank"><https://experts.illinois.edu/en/publications/medusa-simple-llm-inference-acceleration-framework-with-multiple-/></a>

[^30]: ArXiv Dives - Medusa - Oxen.ai, <a href="https://ghost.oxen.ai/arxiv-dives-medusa/" target="_blank"><https://ghost.oxen.ai/arxiv-dives-medusa/></a>

[^31]: \[PDF] Medusa: Simple LLM Inference Acceleration Framework with ..., <a href="https://www.semanticscholar.org/paper/Medusa%3A-Simple-LLM-Inference-Acceleration-Framework-Cai-Li/57e7af0b69325fafb371ef5d502e39ef9c90ef7e" target="_blank"><https://www.semanticscholar.org/paper/Medusa%3A-Simple-LLM-Inference-Acceleration-Framework-Cai-Li/57e7af0b69325fafb371ef5d502e39ef9c90ef7e></a>

[^32]: Medusa: Multiple Decoding Heads for Faster LLM Inference - Substack, <a href="https://substack.com/home/post/p-150122104?utm_campaign=post&utm_medium=web" target="_blank"><https://substack.com/home/post/p-150122104?utm_campaign=post&utm_medium=web></a>

[^33]: Medusa: Simple framework for accelerating LLM generation with ..., <a href="https://www.together.ai/blog/medusa" target="_blank"><https://www.together.ai/blog/medusa></a>

[^34]: Efficient Streaming Language Models with Attention Sinks - arXiv, <a href="https://arxiv.org/abs/2309.17453" target="_blank"><https://arxiv.org/abs/2309.17453></a>

[^35]: Efficient Streaming Language Models with Attention Sinks - GitHub, <a href="https://github.com/mit-han-lab/streaming-llm" target="_blank"><https://github.com/mit-han-lab/streaming-llm></a>

[^36]: \[PDF] Fine-Grained and Efficient KV Cache Retrieval for Long-context LLM ..., <a href="https://aclanthology.org/2025.findings-emnlp.515.pdf" target="_blank"><https://aclanthology.org/2025.findings-emnlp.515.pdf></a>

[^37]: \[PDF] © 2025 Akshat Sharma - IDEALS, <a href="https://www.ideals.illinois.edu/items/136217/bitstreams/445287/data.pdf" target="_blank"><https://www.ideals.illinois.edu/items/136217/bitstreams/445287/data.pdf></a>

[^38]: \[PDF] SINKQ: ACCURATE KV CACHE QUANTIZATION WITH DYNAMIC ..., <a href="https://openreview.net/pdf/3116bda2b92bc73967c37ed846d9e4b814c14e2c.pdf" target="_blank"><https://openreview.net/pdf/3116bda2b92bc73967c37ed846d9e4b814c14e2c.pdf></a>

[^39]: \[PDF] EFFICIENT AND CUSTOMIZABLE ATTENTION ENGINE FOR LLM ..., <a href="https://proceedings.mlsys.org/paper_files/paper/2025/file/dbf02b21d77409a2db30e56866a8ab3a-Paper-Conference.pdf" target="_blank"><https://proceedings.mlsys.org/paper_files/paper/2025/file/dbf02b21d77409a2db30e56866a8ab3a-Paper-Conference.pdf></a>

[^40]: RASD: Retrieval-Augmented Speculative Decoding, <a href="https://doi.org/10.48550/arxiv.2503.03434" target="_blank"><https://doi.org/10.48550/arxiv.2503.03434></a>

[^41]: DySpec: Faster speculative decoding with dynamic token tree structure, <a href="https://doi.org/10.1007/s11280-025-01344-0" target="_blank"><https://doi.org/10.1007/s11280-025-01344-0></a>

[^42]: DySpec: Faster Speculative Decoding with Dynamic Token Tree Structure, <a href="https://doi.org/10.48550/arxiv.2410.11744" target="_blank"><https://doi.org/10.48550/arxiv.2410.11744></a>

[^43]: \[PDF] arXiv:2503.21614v1 \[cs.CL] 27 Mar 2025, <a href="https://arxiv.org/pdf/2503.21614?" target="_blank"><https://arxiv.org/pdf/2503.21614>?</a>

[^44]: Speculative Decoding for Faster LLMs | by M, <a href="https://medium.com/foundation-models-deep-dive/speculative-decoding-for-faster-llms-55353785e4d7" target="_blank"><https://medium.com/foundation-models-deep-dive/speculative-decoding-for-faster-llms-55353785e4d7></a>

[^45]: An Introduction to Speculative Decoding for Reducing ..., <a href="https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/" target="_blank"><https://developer.nvidia.com/blog/an-introduction-to-speculative-decoding-for-reducing-latency-in-ai-inference/></a>

[^46]: Draft, Verify, & Improve Toward Training-Aware ..., <a href="https://arxiv.org/html/2510.05421" target="_blank"><https://arxiv.org/html/2510.05421></a>

[^47]: All About Transformer Inference | How To Scale Your Model, <a href="https://jax-ml.github.io/scaling-book/inference/" target="_blank"><https://jax-ml.github.io/scaling-book/inference/></a>

[^48]: Transformer Inference: Techniques for Faster AI Models, <a href="https://www.premai.io/blog/transformer-inference-techniques-for-faster-ai-models" target="_blank"><https://www.premai.io/blog/transformer-inference-techniques-for-faster-ai-models></a>

[^49]: A Deep Dive into LLM Inference Latencies - Hathora Blog, <a href="https://blog.hathora.dev/a-deep-dive-into-llm-inference-latencies/" target="_blank"><https://blog.hathora.dev/a-deep-dive-into-llm-inference-latencies/></a>

[^50]: Normal Inference Vs Kvcache Vs Lmcache - F22 Labs, <a href="https://www.f22labs.com/blogs/normal-inference-vs-kvcache-vs-lmcache/" target="_blank"><https://www.f22labs.com/blogs/normal-inference-vs-kvcache-vs-lmcache/></a>

[^51]: A Survey on Large Language Model Acceleration based on KV ..., <a href="https://arxiv.org/html/2412.19442v3" target="_blank"><https://arxiv.org/html/2412.19442v3></a>

[^52]: How LLM Inference Works - Arpit Bhayani, <a href="https://arpitbhayani.me/blogs/how-llm-inference-works/" target="_blank"><https://arpitbhayani.me/blogs/how-llm-inference-works/></a>

[^53]: Optimization and Tuning - vLLM, <a href="https://docs.vllm.ai/en/stable/configuration/optimization/" target="_blank"><https://docs.vllm.ai/en/stable/configuration/optimization/></a>

[^54]: LLM Inference Performance Engineering: Best Practices - Databricks, <a href="https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices" target="_blank"><https://www.databricks.com/blog/llm-inference-performance-engineering-best-practices></a>

[^55]: Breaking GPU and Host-Memory Bandwidth Bottlenecks in ..., <a href="https://arxiv.org/html/2512.16056" target="_blank"><https://arxiv.org/html/2512.16056></a>

[^56]: LLM Performance and AI Hardware: 2023–2025 ..., <a href="https://medium.com/@olku/llm-performance-and-ai-hardware-2023-2025-breakthroughs-fa3a1f8dc505" target="_blank"><https://medium.com/@olku/llm-performance-and-ai-hardware-2023-2025-breakthroughs-fa3a1f8dc505></a>

[^57]: vLLM: An Efficient Inference Engine for Large Language Models by ..., <a href="https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-192.pdf" target="_blank"><https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-192.pdf></a>

[^58]: Primers • On-device Transformers, <a href="https://aman.ai/primers/ai/on-device-transformers/" target="_blank"><https://aman.ai/primers/ai/on-device-transformers/></a>

[^59]: Autoregressive Decoding Bottlenecks, <a href="https://apxml.com/courses/how-to-build-a-large-language-model/chapter-28-efficient-inference-strategies/challenges-autoregressive-decoding" target="_blank"><https://apxml.com/courses/how-to-build-a-large-language-model/chapter-28-efficient-inference-strategies/challenges-autoregressive-decoding></a>

[^60]: Making LLMs Faster: My Deep Dive into Speculative Decoding, <a href="https://subhadipmitra.com/blog/2025/making-llm-faster/" target="_blank"><https://subhadipmitra.com/blog/2025/making-llm-faster/></a>

[^61]: FlashMLA: Revolutionizing Efficient Decoding in Large ..., <a href="https://theflyingbirds.in/blog/flashmla-revolutionizing-efficient-decoding-in-large-language-models-through-multi-latent-attention-and-hopper-gpu-optimization" target="_blank"><https://theflyingbirds.in/blog/flashmla-revolutionizing-efficient-decoding-in-large-language-models-through-multi-latent-attention-and-hopper-gpu-optimization></a>

[^62]: Parallel Decoding for LLM Inference, <a href="https://www.emergentmind.com/topics/parallel-decoding" target="_blank"><https://www.emergentmind.com/topics/parallel-decoding></a>

[^63]: How KV Cache Works & Why It Eats Memory | by M - Medium, <a href="https://medium.com/foundation-models-deep-dive/kv-cache-guide-part-2-of-5-under-the-hood-how-kv-cache-works-why-it-eats-memory-f37abf8ea13b" target="_blank"><https://medium.com/foundation-models-deep-dive/kv-cache-guide-part-2-of-5-under-the-hood-how-kv-cache-works-why-it-eats-memory-f37abf8ea13b></a>

[^64]: KV cache offloading | LLM Inference Handbook - BentoML, <a href="https://bentoml.com/llm/inference-optimization/kv-cache-offloading" target="_blank"><https://bentoml.com/llm/inference-optimization/kv-cache-offloading></a>

[^65]: Accelerate Large-Scale LLM Inference and KV Cache Offload with ..., <a href="https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/" target="_blank"><https://developer.nvidia.com/blog/accelerate-large-scale-llm-inference-and-kv-cache-offload-with-cpu-gpu-memory-sharing/></a>

[^66]: Understanding and Coding the KV Cache in LLMs from Scratch ..., <a href="https://www.facebook.com/groups/DeepNetGroup/posts/2514981112228089/" target="_blank"><https://www.facebook.com/groups/DeepNetGroup/posts/2514981112228089/></a>

[^67]: Understanding context length and memory usage : r/LocalLLaMA, <a href="https://www.reddit.com/r/LocalLLaMA/comments/1j7r1sm/understanding_context_length_and_memory_usage/" target="_blank"><https://www.reddit.com/r/LocalLLaMA/comments/1j7r1sm/understanding_context_length_and_memory_usage/></a>

[^68]: KV Cache: The Hidden Optimization Behind Real-Time AI Responses, <a href="https://www.linkedin.com/pulse/kv-cache-hidden-optimization-behind-real-time-ai-vinay-jayanna-cvfec" target="_blank"><https://www.linkedin.com/pulse/kv-cache-hidden-optimization-behind-real-time-ai-vinay-jayanna-cvfec></a>

[^69]: KV-Compress: Paged KV-Cache Compression with Variable ... - arXiv, <a href="https://arxiv.org/html/2410.00161v1" target="_blank"><https://arxiv.org/html/2410.00161v1></a>

[^70]: Architectural Imperatives for Million-Token Inference at Scale - Uplatz, <a href="https://uplatz.com/blog/the-context-window-explosion-architectural-imperatives-for-million-token-inference-at-scale/" target="_blank"><https://uplatz.com/blog/the-context-window-explosion-architectural-imperatives-for-million-token-inference-at-scale/></a>

[^71]: Transformer KV Cache: Methods & Limits - Emergent Mind, <a href="https://www.emergentmind.com/topics/transformer-kv-cache" target="_blank"><https://www.emergentmind.com/topics/transformer-kv-cache></a>

[^72]: Block Transformer: Global-to-Local Language Modeling for Fast Inference, <a href="https://doi.org/10.48550/arxiv.2406.02657" target="_blank"><https://doi.org/10.48550/arxiv.2406.02657></a>

[^73]: Large Language Model Partitioning for Low-Latency Inference at the Edge, <a href="https://doi.org/10.48550/arxiv.2505.02533" target="_blank"><https://doi.org/10.48550/arxiv.2505.02533></a>

[^74]: \[PDF] Decoding Speculative Decoding - ACL Anthology, <a href="https://aclanthology.org/2025.naacl-long.328.pdf" target="_blank"><https://aclanthology.org/2025.naacl-long.328.pdf></a>

[^75]: Speculative decoding | LLM Inference Handbook - BentoML, <a href="https://bentoml.com/llm/inference-optimization/speculative-decoding" target="_blank"><https://bentoml.com/llm/inference-optimization/speculative-decoding></a>

[^76]: Gumiho: A New Paradigm for Speculative Decoding - ROCm™ Blogs, <a href="https://rocm.blogs.amd.com/software-tools-optimization/gumiho/README.html" target="_blank"><https://rocm.blogs.amd.com/software-tools-optimization/gumiho/README.html></a>

[^77]: Speculative Sampling — TensorRT-LLM - GitHub Pages, <a href="https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html" target="_blank"><https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html></a>

[^78]: Decoding Speculative Decoding - arXiv, <a href="https://arxiv.org/html/2402.01528v3" target="_blank"><https://arxiv.org/html/2402.01528v3></a>

[^79]: SLED: A Speculative LLM Decoding Framework for Efficient Edge Serving, <a href="https://arxiv.org/pdf/2506.09397" target="_blank"><https://arxiv.org/pdf/2506.09397></a>

[^80]: \[Feature]: Tree-Attention Support for Speculative Decoding #18327, <a href="https://github.com/vllm-project/vllm/issues/18327" target="_blank"><https://github.com/vllm-project/vllm/issues/18327></a>

[^81]: Efficient Speculative Decoding for Llama at Scale: Challenges and Solutions, <a href="https://arxiv.org/pdf/2508.08192" target="_blank"><https://arxiv.org/pdf/2508.08192></a>

[^82]: Traversal Verification for Speculative Tree Decoding, <a href="https://doi.org/10.48550/arxiv.2505.12398" target="_blank"><https://doi.org/10.48550/arxiv.2505.12398></a>

[^83]: Inference-Cost-Aware Dynamic Tree Construction for Efficient ... - arXiv, <a href="https://arxiv.org/html/2510.26577v1" target="_blank"><https://arxiv.org/html/2510.26577v1></a>

[^84]: Accelerating Large Language Model Inference via Speculative..., <a href="https://openreview.net/forum?id=iW4lyuOQ0J" target="_blank"><https://openreview.net/forum?id=iW4lyuOQ0J></a>

[^85]: RADAR: ACCELERATING LARGE LANGUAGE MODEL ... - arXiv, <a href="https://arxiv.org/html/2512.14069v1" target="_blank"><https://arxiv.org/html/2512.14069v1></a>

[^86]: SpecInfer: Accelerating Large Language Model Serving with Tree ..., <a href="https://dl.acm.org/doi/10.1145/3620666.3651335" target="_blank"><https://dl.acm.org/doi/10.1145/3620666.3651335></a>

[^87]: \[PDF] Dynamic-Width Speculative Beam Decoding for LLM Inference, <a href="https://ojs.aaai.org/index.php/AAAI/article/view/34690/36845" target="_blank"><https://ojs.aaai.org/index.php/AAAI/article/view/34690/36845></a>

[^88]: PipeDec: Low-Latency Pipeline-based Inference with ..., <a href="https://arxiv.org/html/2504.04104v1" target="_blank"><https://arxiv.org/html/2504.04104v1></a>

[^89]: Training-Free Multi-Token Prediction via Probing, <a href="https://openreview.net/forum?id=Drfx9Gnqrv" target="_blank"><https://openreview.net/forum?id=Drfx9Gnqrv></a>

[^90]: Attention Optimization, <a href="https://www.aussieai.com/research/attention" target="_blank"><https://www.aussieai.com/research/attention></a>

[^91]: Not-a-Bandit: Provably No-Regret Drafter Selection in ..., <a href="https://arxiv.org/html/2510.20064v1" target="_blank"><https://arxiv.org/html/2510.20064v1></a>

[^92]: FOLD: Fast Correct Speculative Decoding, <a href="https://openreview.net/forum?id=zm35dmBdok" target="_blank"><https://openreview.net/forum?id=zm35dmBdok></a>

[^93]: ProPD: Dynamic Token Tree Pruning and Generation for LLM Parallel Decoding, <a href="https://doi.org/10.48550/arXiv.2402.13485" target="_blank"><https://doi.org/10.48550/arXiv.2402.13485></a>

[^94]: Medusa: Simple LLM Inference Acceleration Framework with ..., <a href="https://www.alphaxiv.org/zh/overview/2401.10774v1" target="_blank"><https://www.alphaxiv.org/zh/overview/2401.10774v1></a>

[^95]: \[PDF] Medusa: Simple LLM Inference Acceleration Framework ... - GitHub, <a href="https://raw.githubusercontent.com/mlresearch/v235/main/assets/cai24b/cai24b.pdf" target="_blank"><https://raw.githubusercontent.com/mlresearch/v235/main/assets/cai24b/cai24b.pdf></a>

[^96]: \[Literature Review] Medusa: Simple LLM Inference Acceleration ..., <a href="https://www.themoonlight.io/en/review/medusa-simple-llm-inference-acceleration-framework-with-multiple-decoding-heads" target="_blank"><https://www.themoonlight.io/en/review/medusa-simple-llm-inference-acceleration-framework-with-multiple-decoding-heads></a>

[^97]: Your LLM Knows the Future: Uncovering Its Multi-Token Prediction ..., <a href="https://arxiv.org/html/2507.11851v1" target="_blank"><https://arxiv.org/html/2507.11851v1></a>

[^98]: DeepSeek Explained 4: Multi-Token Prediction | by Shirley Li, <a href="https://medium.com/data-science-collective/deepseek-explained-4-multi-token-prediction-33f11fe2b868" target="_blank"><https://medium.com/data-science-collective/deepseek-explained-4-multi-token-prediction-33f11fe2b868></a>

[^99]: ParallelSpec: Parallel Drafter for Efficient Speculative Decoding, <a href="https://openreview.net/forum?id=SXvb8PS4Ud" target="_blank"><https://openreview.net/forum?id=SXvb8PS4Ud></a>

[^100]: PARD: Accelerating LLM Inference with Low-Cost PARallel Draft Model Adaptation, <a href="https://arxiv.org/pdf/2504.18583" target="_blank"><https://arxiv.org/pdf/2504.18583></a>

[^101]: When, What, and How: Rethinking Retrieval-Enhanced Speculative Decoding, <a href="https://arxiv.org/pdf/2511.01282" target="_blank"><https://arxiv.org/pdf/2511.01282></a>

[^102]: CLLMs: Consistency Large Language Models, <a href="https://doi.org/10.48550/arXiv.2403.00835" target="_blank"><https://doi.org/10.48550/arXiv.2403.00835></a>

[^103]: Parallel Decoding via Hidden Transfer for Lossless Large Language Model Acceleration, <a href="https://doi.org/10.48550/arxiv.2404.12022" target="_blank"><https://doi.org/10.48550/arxiv.2404.12022></a>

[^104]: Generation Meets Verification: Accelerating Large Language Model Inference with Smart Parallel Auto-Correct Decoding, <a href="https://doi.org/10.48550/arXiv.2402.11809" target="_blank"><https://doi.org/10.48550/arXiv.2402.11809></a>

[^105]: StreamingLLM: Efficient Streaming for LLMs - Emergent Mind, <a href="https://www.emergentmind.com/topics/streamingllm" target="_blank"><https://www.emergentmind.com/topics/streamingllm></a>

[^106]: Taming the Fragility of KV Cache Eviction in LLM Inference - arXiv, <a href="https://arxiv.org/html/2510.13334v1" target="_blank"><https://arxiv.org/html/2510.13334v1></a>

[^107]: The Secret Behind Fast LLM Inference: Unlocking the KV Cache, <a href="https://pub.towardsai.net/the-secret-behind-fast-llm-inference-unlocking-the-kv-cache-9c13140b632d" target="_blank"><https://pub.towardsai.net/the-secret-behind-fast-llm-inference-unlocking-the-kv-cache-9c13140b632d></a>

[^108]: Efficient Streaming Language Models with Attention Sinks, <a href="https://openreview.net/forum?id=NG7sS51zVF" target="_blank"><https://openreview.net/forum?id=NG7sS51zVF></a>

[^109]: KV Cache Optimization: Memory Efficiency for Production LLMs - Introl, <a href="https://introl.com/blog/kv-cache-optimization-memory-efficiency-production-llms-guide" target="_blank"><https://introl.com/blog/kv-cache-optimization-memory-efficiency-production-llms-guide></a>

[^110]: FlashAttention | LLM Inference Handbook - BentoML, <a href="https://bentoml.com/llm/inference-optimization/flashattention" target="_blank"><https://bentoml.com/llm/inference-optimization/flashattention></a>

[^111]: Hardware-Aware Parallel Prompt Decoding for Memory-Efficient ..., <a href="https://arxiv.org/html/2405.18628v3" target="_blank"><https://arxiv.org/html/2405.18628v3></a>

[^112]: Kangaroo: Lossless Self-Speculative Decoding for Accelerating ..., <a href="https://www.researchgate.net/publication/397198289_Kangaroo_Lossless_Self-Speculative_Decoding_for_Accelerating_LLMs_via_Double_Early_Exiting" target="_blank"><https://www.researchgate.net/publication/397198289_Kangaroo_Lossless_Self-Speculative_Decoding_for_Accelerating_LLMs_via_Double_Early_Exiting></a>

[^113]: SWIFT: On-the-Fly Self-Speculative Decoding for LLM Inference..., <a href="https://openreview.net/forum?id=EKJhH5D5wA" target="_blank"><https://openreview.net/forum?id=EKJhH5D5wA></a>

[^114]: Unlocking Efficiency in Large Language Model Inference: A Comprehensive Survey of Speculative Decoding, <a href="https://arxiv.org/pdf/2401.07851" target="_blank"><https://arxiv.org/pdf/2401.07851></a>

[^115]: SPIN: Accelerating Large Language Model Inference with Heterogeneous Speculative Models, <a href="https://arxiv.org/pdf/2503.15921" target="_blank"><https://arxiv.org/pdf/2503.15921></a>

[^116]: OWL: Overcoming Window Length-Dependence in Speculative Decoding for Long-Context Inputs, <a href="https://arxiv.org/pdf/2510.07535" target="_blank"><https://arxiv.org/pdf/2510.07535></a>

[^117]: Judge Decoding: Faster Speculative Sampling Requires Going Beyond Model Alignment, <a href="https://arxiv.org/pdf/2501.19309" target="_blank"><https://arxiv.org/pdf/2501.19309></a>

[^118]: A Comprehensive Survey of Accelerated Generation Techniques in Large Language Models, <a href="https://arxiv.org/pdf/2405.13019" target="_blank"><https://arxiv.org/pdf/2405.13019></a>

[^119]: An Adaptive Parallel Layer-Skipping Framework for Large Language Model Inference Speedup With Speculative Decoding, <a href="https://doi.org/10.23919/ics.2025.3575371" target="_blank"><https://doi.org/10.23919/ics.2025.3575371></a>

[^120]: Minions: Accelerating Large Language Model Inference with Adaptive and Collective Speculative Decoding, <a href="https://doi.org/10.48550/arxiv.2402.15678" target="_blank"><https://doi.org/10.48550/arxiv.2402.15678></a>

[^121]: Cerberus: Efficient Inference with Adaptive Parallel Decoding and Sequential Knowledge Enhancement, <a href="https://doi.org/10.48550/arxiv.2410.13344" target="_blank"><https://doi.org/10.48550/arxiv.2410.13344></a>

[^122]: New Solutions on LLM Acceleration, Optimization, and Application, <a href="https://arxiv.org/pdf/2406.10903" target="_blank"><https://arxiv.org/pdf/2406.10903></a>

[^123]: MCBP: A Memory-Compute Efficient LLM Inference Accelerator Leveraging Bit-Slice-enabled Sparsity and Repetitiveness, <a href="https://arxiv.org/pdf/2509.10372" target="_blank"><https://arxiv.org/pdf/2509.10372></a>

[^124]: CLONE: Customizing LLMs for Efficient Latency-Aware Inference at the Edge, <a href="https://arxiv.org/pdf/2506.02847" target="_blank"><https://arxiv.org/pdf/2506.02847></a>

[^125]: Hardware Design and Security Needs Attention: From Survey to Path Forward, <a href="https://arxiv.org/pdf/2504.08854" target="_blank"><https://arxiv.org/pdf/2504.08854></a>

[^126]: A Survey on Large Language Model Acceleration based on KV Cache Management, <a href="https://doi.org/10.48550/arXiv.2412.19442" target="_blank"><https://doi.org/10.48550/arXiv.2412.19442></a>

[^127]: ClusterKV: Manipulating LLM KV Cache in Semantic Space for Recallable Compression, <a href="https://doi.org/10.48550/arxiv.2412.03213" target="_blank"><https://doi.org/10.48550/arxiv.2412.03213></a>

[^128]: Efficient Long-Context LLM Inference via KV Cache Clustering, <a href="https://doi.org/10.48550/arxiv.2506.11418" target="_blank"><https://doi.org/10.48550/arxiv.2506.11418></a>

[^129]: SCBench: A KV Cache-Centric Analysis of Long-Context Methods, <a href="https://doi.org/10.48550/arxiv.2412.10319" target="_blank"><https://doi.org/10.48550/arxiv.2412.10319></a>

[^130]: Quest: Query-Aware Sparsity for Efficient Long-Context LLM Inference, <a href="https://doi.org/10.48550/arxiv.2406.10774" target="_blank"><https://doi.org/10.48550/arxiv.2406.10774></a>

