# From Rigid Trees to Adaptive Forests: A Critical Analysis of LLM Inference Acceleration Evolution from Linear to Confidence-Aware Dynamic Tree-Based Speculative Decoding

> Summary
> The report traces the evolution of LLM inference acceleration from linear speculative decoding, which suffers from limited path exploration, to tree-based methods like SpecInfer that enable parallel verification of multiple candidate sequences via static token trees. However, these static structures are criticized for their rigidity, creating an "Efficiency Gap": computational waste from verifying excessive branches in high-confidence scenarios and insufficient exploration of plausible paths in uncertain contexts. In contrast, confidence-aware adaptive approaches (e.g., CM-ASD, AdaEAGLE, OPT-Tree) dynamically adjust branching factors and tree depth based on real-time token probability distributions and model confidence. These methods optimize the resource trade-off by adaptively pruning low-probability branches, thereby better balancing acceptance rates against verification overhead compared to fixed-structure baselines.

## 1. Historical Evolution: From Linear to Tree-Based Speculative Decoding

The acceleration of Large Language Model (LLM) inference through speculative decoding represents a significant paradigm shift in computational efficiency, evolving from simple linear approaches to sophisticated tree-based architectures. This progression reflects the field's response to the fundamental bottleneck of autoregressive decoding, where each token generation depends sequentially on previous outputs, leading to underutilized hardware resources and high latency. The journey from linear to tree-based methods reveals a continuous effort to close the **"efficiency gap"**—the disparity between theoretical speedup and actual performance caused by computational waste and insufficient exploration of token possibilities.

### 1.1 The Dawn of Linear Speculative Decoding

The earliest approaches to speculative decoding emerged from the innovative concept of **Blockwise Decoding**, which introduced the foundational draft-then-verify paradigm[^20]. This linear approach operated on a straightforward principle: a smaller, faster "draft model" would generate a single sequence of candidate tokens, which the larger target LLM would then verify in parallel. The key insight was that if the draft model could accurately predict the target model's outputs, multiple tokens could be validated in a single forward pass, effectively reducing the number of sequential decoding steps.

Linear speculative decoding methods, including early implementations of **SpecDec**, shared several defining characteristics[^16]:

| Characteristic             | Linear Approach                        | Impact                                                             |
| -------------------------- | -------------------------------------- | ------------------------------------------------------------------ |
| **Candidate Generation**   | Single sequence from draft model       | Limited exploration of alternative paths                           |
| **Verification Mechanism** | Parallel validation of linear sequence | Simple implementation but wasted computation on incorrect prefixes |
| **Acceptance Strategy**    | Longest matching prefix                | Conservative approach that discards valid alternative branches     |

However, these linear approaches suffered from inherent limitations. By considering only a single speculative path, they created computational waste when the draft model's predictions diverged from the target LLM's preferences. This rigidity in candidate exploration directly contributed to the early manifestation of the efficiency gap, where potential speedup was lost due to insufficient exploration of alternative token sequences[^16].

### 1.2 The Paradigm Shift: Token Tree Verification

The breakthrough in speculative decoding arrived with the introduction of **token tree verification**, pioneered by **SpecInfer**[^1]. This approach represented a fundamental architectural shift from linear sequences to hierarchical tree structures, enabling parallel verification of *multiple* candidate token sequences simultaneously.

SpecInfer's core innovation involved organizing draft predictions from small speculative models (SSMs) into a **token tree** where candidate sequences shared common prefixes. This allowed the target LLM to verify the entire tree structure in a single parallel operation using a novel tree-based parallel decoding algorithm[^1]. The system employed two primary mechanisms for tree construction:

1. **Expansion-based mechanisms**: Creating a tree by deriving multiple tokens from a single SSM within a decoding step.
2. **Merge-based mechanisms**: Combining predictions from multiple SSMs to form a unified token tree.

For stochastic decoding, SpecInfer developed the **multi-step speculative sampling (MSS) algorithm**, which provably preserved the LLM's generative performance while maximizing verified tokens[^1]. This tree-based paradigm directly addressed a key limitation of linear methods by broadening the search space, allowing the verification process to explore several potential future token paths in one step.

### 1.3 Refinement and Specialization: Medusa and EAGLE

Following SpecInfer's foundational work, subsequent research refined tree-based speculative decoding through specialized approaches. **Medusa** introduced a simplified framework that eliminated the need for separate draft models by adding extra decoding heads directly to the backbone LLM[^28].

Medusa's key innovations included:

* **Multiple decoding heads** predicting subsequent tokens in parallel.

* A **tree-based attention mechanism** for constructing and verifying multiple candidate continuations simultaneously.

* A **typical acceptance scheme** as an alternative to rejection sampling, aiming to boost acceptance rates[^28].

Unlike linear speculative decoding which used a separate draft model to propose sequential candidates, Medusa's heads generated multiple candidate continuations simultaneously in a tree structure, allowing parallel processing of several potential paths at once[^28]. Concurrently, frameworks like **EAGLE** advanced tree-based approaches through optimized implementations of tree attention mechanisms[^7].

### 1.4 Computational Trade-offs and the Emergence of Static Rigidity

The evolution from linear to tree-based approaches revealed new trade-offs. While tree-based methods offered superior exploration, they introduced specific overheads[^1]:

| Computational Aspect       | Linear Approaches                   | Tree-Based Approaches                                       |
| -------------------------- | ----------------------------------- | ----------------------------------------------------------- |
| **Memory Overhead**        | Moderate (single sequence KV cache) | Higher (caching for all tokens in the token tree)           |
| **Computation Overhead**   | Draft model execution               | Verification of many speculated tokens that may be rejected |
| **Candidate Diversity**    | Limited to single path              | Multiple alternative branches                               |
| **Structural Flexibility** | N/A                                 | Fixed, preset tree shapes                                   |

Research indicated that computational waste in speculative decoding could arise from verifying entire token trees containing many tokens that do not match the LLM's actual output[^1]. More critically, the limitations of **static tree structures** became apparent. Methods like early versions of EAGLE and Medusa employed preset tree structures with fixed branching factors and depths[^26]. This rigidity meant the tree could not adapt to the context of the generated text, leading to two persistent inefficiencies that define the core efficiency gap:

1. **Computational waste in high-confidence scenarios**: When the next tokens are predictable, a static, wide tree may generate and verify many unnecessary candidate branches.
2. **Insufficient exploration in uncertain contexts**: When the token probability distribution is flat, a static tree may not explore enough alternative paths, leading to verification failures and rollbacks.

This inefficiency occurs because the "acceptance length of draft tokens strongly depends on the generated context," a dynamic that static structures fail to capture[^7].

### 1.5 The Adaptive Turn: Bridging the Efficiency Gap

The recognition of static tree limitations catalyzed the next evolutionary step: **adaptive tree construction methods**. Approaches like **C2T** introduced lightweight classifiers to dynamically generate and prune token trees, incorporating feature variables like entropy and depth beyond simple joint probability[^26]. Similarly, **Confidence-Modulated Adaptive Speculative Decoding (CM-ASD)** emerged to dynamically adjust drafting lengths and verification thresholds based on the drafter model's internal confidence in real-time[^2].

These adaptive mechanisms represent a direct response to the rigidity of early tree-based methods. They aim to dynamically balance the breadth of exploration with the cost of verification, aggressively drafting in high-confidence regions while proceeding cautiously in uncertain ones[^2]. This shift from static to adaptive structures marks the ongoing effort to close the efficiency gap by making speculative decoding context-aware and resource-optimized.

The historical progression from linear to tree-based speculative decoding thus illustrates a continuous optimization of the trade-off between exploration breadth and computational efficiency. Each evolutionary step has addressed limitations revealed by previous approaches, with the current frontier focusing on adaptive mechanisms to overcome the rigidity of static tree architectures.

## 2. SpecInfer and Static Tree Architectures: Innovations and Computational Inefficiencies

SpecInfer represents a paradigm shift in speculative decoding for large language model (LLM) acceleration by introducing a tree-based parallel verification framework[^34]. This approach fundamentally reimagines the draft-verify paradigm by organizing candidate token sequences into hierarchical tree structures rather than linear sequences, enabling the simultaneous verification of multiple speculative paths[^1]. While this architectural innovation promises substantial throughput improvements, it comes with inherent computational trade-offs that reveal critical limitations in static tree structures, particularly concerning memory overhead, verification inefficiencies, and structural rigidity.

### 2.1 Core Architectural Innovations: Token Tree Construction and Parallel Verification

SpecInfer's foundational contribution lies in its **token tree construction mechanisms** and **tree-based parallel decoding algorithm**, which collectively enable parallel verification of multiple speculative sequences[^34]. The system employs two complementary approaches for tree construction:

**Expansion-based token tree construction** leverages the observation that the target LLM's chosen token is often among a small speculative model's (SSM) top-k predictions. Instead of selecting only the top-1 token, SpecInfer expands multiple top-k tokens at each decoding step, creating branching paths that increase the probability of successful verification[^1]. This approach uses a static expansion strategy, such as a vector `<k₁, k₂, ..., kₘ>`, where `m` is the maximum speculative steps and `kᵢ` is the number of tokens expanded at step i.

**Merge-based token tree construction** aggregates predictions from multiple SSMs to enhance speculative diversity. SpecInfer can combine predictions from several SSMs to form a single, unified token tree, exploiting inter-model diversity[^1].

The **tree-based parallel decoding algorithm** is SpecInfer's most significant technical innovation. It enables the target LLM to verify the correctness of all candidate token sequences in a speculated token tree in a single, parallel inference execution\[\[1]\[34]]. This is achieved through a novel tree attention mechanism that generalizes Transformer attention from sequences to tree structures, allowing all tokens in the tree to be verified simultaneously rather than sequentially[^1].

For stochastic decoding, SpecInfer employs a **multi-step speculative sampling (MSS) algorithm** that provably preserves the LLM's exact output distribution while maximizing the number of verified tokens[^1].

### 2.2 Computational Inefficiencies in Static Tree Structures

Despite its architectural advances, SpecInfer's reliance on predetermined, static tree structures introduces significant computational inefficiencies that limit practical deployment efficiency. These inefficiencies manifest primarily as memory overhead and computation waste.

#### 2.2.1 Memory Overhead: SSM Parameters and Attention Caching

SpecInfer incurs memory overhead from two primary sources: storing SSM parameters and caching attention keys and values for the token tree[^1].

While individual SSMs are significantly smaller than the target LLM (typically 100-1000x smaller, representing less than a 1% memory increase per SSM), the aggregate footprint can become non-trivial when employing multiple SSMs[^1]. More critically, the attention mechanism requires caching keys and values for all tokens in the speculated token tree to compute attention outputs[^1]. For long-sequence generation, this caching introduces significant memory demands that can limit the number of requests served in parallel[^1].

| Memory Component   | Overhead Source              | Impact                                 |
| ------------------ | ---------------------------- | -------------------------------------- |
| SSM Parameters     | Multiple small models        | <1% per SSM, cumulative with scale[^1] |
| Key-Value Cache    | All tokens in the token tree | Significant for long sequences[^1]     |
| Tree Topology Data | Branching structure metadata | Proportional to tree size              |

#### 2.2.2 Computation Overhead: Unnecessary Token Verification

A major source of inefficiency is the computation wasted on verifying speculated tokens that do not match the LLM's actual output[^1]. In static tree-based decoding, an entire token tree—containing many candidate tokens and sequences—must be verified by the target LLM in a single parallel pass[^1]. This means computational resources are spent on attention calculations for many tokens that will ultimately be rejected. The paper notes that SpecInfer attempts to mitigate this by leveraging under-utilized GPU resources for the tree-based parallel decoding, claiming negligible runtime overhead compared to incremental decoding in such scenarios[^1].

This verification waste is exacerbated by the **structural rigidity** of static trees. Their branching factor (how many tokens are speculated at each step) and maximum depth are fixed parameters (e.g., `<2,2,1>`)[^1]. This rigidity creates a fundamental **efficiency gap**:

* **Computational waste in high-confidence scenarios:** When the LLM's next token is highly predictable, generating and verifying a large tree of alternative branches is unnecessary overhead.

* **Insufficient exploration in uncertain contexts:** When token probabilities are evenly distributed, a static tree with a fixed, small branching factor may fail to generate enough speculative diversity, missing potential parallel verification opportunities.

The static structure cannot adapt to the dynamic nature of language generation, where the optimal amount of speculation changes with every token based on the context and the model's internal confidence.

### 2.3 The Limitations of Static Draft Policy

The inflexibility of static tree architectures leads to a problem characterized as **draft policy misalignment**. Training objectives for draft models often optimize for generating a single, high-likelihood sequence (greedy path). However, tree-based decoding evaluates and may accept different branches within the tree[^16]. This misalignment means the draft model is not optimized for the actual verification strategy used during inference, potentially leading to suboptimal acceptance lengths and wasted computation[^16].

### 2.4 Comparative Analysis: Static vs. Adaptive Mechanisms

The limitations of static architectures become evident when contrasted with emerging adaptive approaches, which dynamically adjust their drafting strategy based on real-time confidence metrics.

| Mechanism                 | Static Tree (SpecInfer)           | Adaptive Approaches (e.g., CM-ASD, AdaEAGLE)                                  |
| ------------------------- | --------------------------------- | ----------------------------------------------------------------------------- |
| **Branching Factor**      | Fixed expansion vectors[^1]       | Dynamic adjustment based on token probability distributions[^2]               |
| **Drafting Length**       | Predefined maximum depth[^1]      | Context-aware prediction (e.g., via a Lightweight Draft Length Predictor)[^7] |
| **Verification Strategy** | Uniform across all tokens in tree | Confidence-modulated thresholds[^2]                                           |
| **Resource Allocation**   | Equal for all branches            | Prioritized allocation to high-confidence paths\[\[2]\[7]]                    |

Adaptive mechanisms like **Confidence-Modulated Adaptive Speculative Decoding (CM-ASD)** dynamically adjust the drafting window size and verification thresholds using a unified confidence score derived from the drafter model's probability distribution[^2]. This allows for aggressive drafting when confidence is high and conservative drafting when it is low, reducing rollbacks and improving resource utilization[^2].

Similarly, frameworks like **AdaEAGLE** employ a Lightweight Draft Length Predictor (LDLP) module to explicitly predict the optimal number of draft tokens for each decoding iteration based on the current context[^7]. This aims to minimize wasted draft computation and reduce the number of costly target model verification passes by generating a draft length close to what will actually be accepted[^7].

### 2.5 Verification Overhead and the Need for Adaptive Pruning

The verification phase in static tree architectures imposes overhead that scales with the size of the pre-built tree, not the utility of its branches. Adaptive pruning mechanisms demonstrate a path to better balance. For instance, the **STAR** framework uses a Neural Architecture Search (NAS) to jointly optimize draft model configuration (e.g., through attention-head pruning) and verification strategy[^17]. This hardware-aware optimization seeks to maximize overall speedup by explicitly balancing the draft model's speed against the token acceptance rate, thereby managing verification overhead[^17].

This highlights the core trade-off: while larger static trees may improve acceptance probabilities, they also amplify verification costs. Adaptive approaches seek to navigate this trade-off more efficiently by generating and verifying only the most promising speculative paths based on real-time signals.

## 3. The Efficiency Gap: Computational Waste versus Insufficient Exploration

Speculative decoding fundamentally operates on a delicate balance between two opposing inefficiencies: computational waste from over-speculation in high-confidence scenarios and insufficient exploration in uncertain contexts. This "efficiency gap" represents a critical bottleneck in achieving optimal inference acceleration. The evolution from linear to tree-based approaches, while innovative, has not resolved this core tension. In fact, the rigidity of static tree structures—with their fixed branching factors and predetermined depths—exacerbates the gap by failing to adapt to the dynamic probability distributions inherent in language generation.

### 3.1 The Dual Nature of Inefficiency

The efficiency gap manifests as two distinct but interrelated problems. **Computational waste** occurs when speculative methods generate excessive draft tokens that are ultimately rejected during verification, consuming computational resources without contributing to the final output. This is particularly problematic in high-confidence scenarios where the target model's probability distribution is concentrated around a few likely tokens, yet a static draft tree continues to generate a full, predetermined set of candidate branches[^1]. Conversely, **insufficient exploration** arises in uncertain contexts where the probability distribution is flatter, requiring broader exploration of potential continuations. Static methods, constrained by fixed structures, often fail to allocate sufficient computational resources to explore these diverse possibilities, leading to suboptimal paths and reduced acceptance rates.

### 3.2 How Static Tree Structures Amplify the Gap

Tree-based speculative decoding methods, pioneered by SpecInfer, introduced significant innovations by enabling the parallel verification of multiple draft sequences organized into a token tree[^1]. However, these static architectures introduce inherent limitations that directly worsen the efficiency gap:

| **Static Tree Limitation**      | **Impact on Efficiency Gap**                                                                                                                                  |
| :------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Fixed branching factors**     | In high-confidence scenarios, generates redundant, low-probability branches. In uncertain contexts, provides insufficient diversity for adequate exploration. |
| **Predetermined tree depth**    | Leads to over-speculation when acceptance probability decays rapidly, or under-speculation when longer viable sequences are possible.                         |
| **Uniform resource allocation** | Wastes computation on low-probability paths while under-investing in high-potential branches, as all nodes receive equal budget.                              |
| **Lack of context adaptation**  | Cannot adjust to changing probability distributions across different phases of text generation.                                                               |

These limitations are inherent to the design. For instance, SpecInfer's expansion-based method creates trees by deriving multiple tokens from a single small speculative model (SSM) based on fixed top-k predictions, regardless of the underlying confidence in those predictions[^1]. As noted in a comprehensive survey, encouraging drafters to prioritize early-position tokens suggests that a fixed tree cannot efficiently leverage shifting probability distributions and opportunities for early bifurcation[^16].

The computational overhead of static trees is twofold. First, there is **memory overhead** from storing SSM parameters and caching keys and values for all tokens in the token tree, though this is often considered negligible compared to the key-value cache for very long sequences[^1]. Second, and more critically, there is **computation overhead** from running SSMs to generate candidate tokens and from verifying an entire token tree by computing attention outputs for many speculated tokens that may ultimately be rejected[^1]. This verification of unnecessary tokens represents direct computational waste.

### 3.3 Adaptive Mechanisms for Bridging the Gap

Emerging research demonstrates that adaptive approaches can effectively bridge the efficiency gap by dynamically adjusting speculative structures based on real-time confidence estimates. These mechanisms move beyond rigid templates to optimize the trade-off between waste and exploration.

**Confidence-Modulated Adaptive Speculative Decoding (CM-ASD)** dynamically adjusts the drafting length and verification thresholds based on the drafter model's internal confidence[^2]. It employs a unified confidence score derived from entropy, logit margin, and softmax margin. The drafting length (k\_j) is then determined by the average confidence over the next positions: (k\_j = \min(k\_{max}, \max(k\_{min}, \lfloor \alpha \* \bar{C}_{j:k} \* k_{max} \rfloor))). Simultaneously, the verification threshold (\tau\_t) is adjusted as (\tau\_t = \tau\_{base} + \gamma \* (1 - C\_t)), allowing more lenient acceptance when confidence is high and stricter verification when it is low[^2]. This dual adaptation reduces rollback frequency and improves resource utilization.

**CAS-Spec (Cascade Adaptive Self-Speculative Decoding)** employs a **Dynamic Tree Cascade (DyTC)** algorithm that adaptively routes multi-level draft models and assigns draft lengths based on online heuristics of acceptance rates and latency prediction[^44]. The algorithm maintains Exponential Moving Average (EMA) estimates of acceptance rates for each draft configuration and uses a hardware-aware latency prediction model. By maximizing an Expected Walltime Improvement Factor, it dynamically determines where to start generating tokens in the tree and when to switch between draft models[^44].

**OPT-Tree** directly addresses static inefficiency by algorithmically searching for the optimal tree structure that maximizes the mathematical expectation of the acceptance length in each decoding step[^41]. This allows it to adapt to varying draft model capabilities and computational budgets, dynamically constructing scalable draft trees rather than relying on fixed heuristics[^41].

**AdaEAGLE** incorporates a **Lightweight Draft Length Predictor (LDLP)**, a small MLP that explicitly predicts the optimal number of draft tokens ((k\_r)) for each decoding iteration based on the context[^7]. By aiming to generate a number of tokens close to the optimal acceptance length, it minimizes wasted draft computation ((N\_{waste})) and reduces the number of costly target model forward passes ((N\_{target})), optimizing overall throughput[^7].

### 3.4 Comparative Performance: Static versus Adaptive

The quantitative benefits of adaptive mechanisms are evident in empirical results, which show significant improvements across key metrics related to the efficiency gap:

| **Aspect**                  | **Static Tree Structures**         | **Adaptive Approaches**                                   | **Improvement Evidence**                                                                                                                                                                                      |
| :-------------------------- | :--------------------------------- | :-------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Computational Waste**     | High in high-confidence scenarios. | Minimized via early stopping & optimal length prediction. | AdaEAGLE reduces wasted draft tokens significantly compared to fixed-length baselines[^7]. CAS-Spec's DyTC improves average speedup by **47%** over cascade-based and **48%** over tree-based baselines[^44]. |
| **Exploration Sufficiency** | Limited in uncertain contexts.     | Enhanced via confidence-aware expansion.                  | CM-ASD's dynamic drafting allows aggressive exploration when confidence permits and conservatism when needed[^2].                                                                                             |
| **Acceptance Rate**         | Suboptimal due to rigid structure. | Optimized through dynamic adjustment.                     | Adaptive methods achieve higher throughput by better aligning draft output with the target model.                                                                                                             |
| **Resource Utilization**    | Inflexible, uniform allocation.    | Context-aware, efficient allocation.                      | AdaEAGLE balances time spent on draft ((T\_{draft})) and target ((T\_{target})) models for better overall throughput[^7].                                                                                     |

### 3.5 Optimizing the Trade-off: Strategies and Trade-offs

Bridging the efficiency gap requires optimizing the inherent trade-off between acceptance rate, verification overhead, and computational waste. Successful adaptive strategies share several principles:

1. **Confidence-Aware Decision Making:** Utilizing real-time metrics like entropy[^2] or learned predictors[^7] to guide the speculation depth and breadth.
2. **Hardware-Aware Optimization:** Incorporating latency predictions tailored to specific hardware platforms to make cost-effective drafting decisions[^44].
3. **Multi-Objective Balancing:** Explicitly modeling the trade-off, as seen in AdaEAGLE's loss function which penalizes under-prediction of draft length to avoid costly target model passes[^7], or in STAR's Neural Architecture Search framework which jointly optimizes for draft speed and acceptance rate[^17].

In conclusion, the efficiency gap between computational waste and insufficient exploration is a fundamental challenge in speculative decoding. While static tree-based methods like SpecInfer provided a leap in parallelization potential, their rigidity locks in suboptimal resource allocation. Confidence-aware adaptive mechanisms represent the necessary evolution, dynamically tailoring the speculative process to the immediate generative context. By adjusting branching factors, depth, and verification strictness in real-time, these approaches directly optimize the core trade-off, leading to measured improvements in speedup, throughput, and resource efficiency\[\[2]\[7]\[44]].

## 4. Adaptive Mechanisms: Confidence-Aware Dynamic Tree Adjustment

The evolution from static tree-based speculative decoding to confidence-aware adaptive approaches represents a fundamental shift toward addressing the computational inefficiencies inherent in fixed tree structures. While methods like SpecInfer introduced parallel verification of multiple token sequences through token trees, they suffer from rigidity in handling varying contextual uncertainty[^1]. This section explores the emerging paradigm of adaptive mechanisms that dynamically reconfigure speculative decoding parameters based on real-time model confidence and token probability distributions, aiming to close the "efficiency gap."

### 4.1 The Computational Waste Efficiency Gap in Static Tree Structures

Static tree-based speculative decoding methods exhibit significant computational inefficiencies that manifest as an "efficiency gap" between high-confidence and uncertain generation scenarios. As analyzed in SpecInfer, these methods incur **memory overhead** from storing parameters of small speculative models (SSMs) and caching attention keys/values for all tokens in the token tree. They also incur **computation overhead** from verifying entire token trees containing many tokens that do not match the target LLM's output[^1].

This efficiency gap is pronounced in two opposing scenarios:

1. **High-confidence contexts**: Where token continuation is obvious, static trees generate excessive, unnecessary branching, resulting in wasted computation on low-probability token verifications.
2. **High-uncertainty contexts**: Where multiple plausible continuations exist, fixed branching factors may insufficiently explore the probability space, leading to frequent verification failures, token rollbacks, and costly re-speculation.

This rigidity underscores the need for adaptive mechanisms that can balance acceptance rates with computational efficiency in real-time.

### 4.2 AdaEAGLE: Explicit Modeling with a Lightweight Draft Length Predictor

AdaEAGLE introduces an explicit learning-based approach to adaptive draft structure optimization through its **Lightweight Draft Length Predictor (LDLP)** module[^7]. The LDLP directly predicts the optimal number of draft tokens ($k_r$) for each decoding iteration before generation, aiming to minimize wasted draft computation and reduce the number of costly target model forward passes.

**Technical Architecture and Operation:**
The LDLP is implemented as a lightweight **three-layer Multi-Layer Perceptron (MLP)**. For the *r*-th decoding iteration, it takes two context-aware inputs:

* The embedding ($e_{j+k_r^\circ}$) of the last token in the validated prefix.

* The last hidden state ($f_{j+k_r^\circ}$) from the target model's final layer normalization.

The predicted draft length for the next step is:
$\bar{k}_{r+1} = \operatorname{Round}(\operatorname{MLP}(e_{j+k_r^\circ}, f_{j+k_r^\circ}))$
The final draft length is then constrained by: $k_{r+1} = \min(k_{max}, \max(0, \bar{k}_{r+1}))$[^7].

**Training and Optimization:**
The LDLP is trained on collected data pairs of (embedding, hidden state) and the corresponding optimal draft length. It uses a penalized L1 loss function:

$$
L = \begin{cases}
\lambda \cdot |\bar{k} - k^\circ| & \text{if } \bar{k} < k^\circ \\
|\bar{k} - k^\circ| & \text{otherwise}
\end{cases}
$$

where $\lambda > 1$ is a penalty coefficient. This encourages the predictor to avoid under-predicting the draft length, as target model forward passes are more costly than generating a few extra draft tokens[^7]. By dynamically adjusting the draft length, AdaEAGLE reduces wasted tokens ($N_{waste}$) and improves overall throughput compared to fixed-length baselines[^7].

### 4.3 Confidence-Modulated Adaptive Speculative Decoding (CM-ASD)

CM-ASD provides a principled, information-theoretic framework for confidence-aware adaptation, dynamically adjusting both the drafting window size and the verification thresholds based on the drafter model's internal confidence[^2].

**Confidence Estimation Mechanisms:**
CM-ASD employs three complementary methods to estimate a scalar confidence score ($C_t \in [0,1]$) for each token prediction:

1. **Entropy-based Confidence**: $C_t^{(ent)} = -\sum_{y \in \nu} P_d(y|y^{<t}, x) \cdot \log P_d(y|y^{<t}, x)$, normalized by vocabulary size.
2. **Logit Margin Confidence**: $C_t^{(margin)} = z^{(1)} - z^{(2)}$, normalized via a sigmoid function.
3. **Softmax Margin Confidence**: $C_t^{(soft)} = P_d^{(1)} - P_d^{(2)}$.

These are combined into a unified score: $C_t = \lambda_1 \cdot \tilde{C}_t^{(ent)} + \lambda_2 \cdot \tilde{C}_t^{margin} + \lambda_3 \cdot \tilde{C}_t^{(soft)}$, with $\sum \lambda_i = 1$[^2].

**Dynamic Parameter Adjustment:**

* **Drafting Window**: The number of tokens to draft ($k_j$) is controlled by the average confidence over the next *k* positions, $\bar{C}_{j:k}$:
  $k_j = \min(k_{max}, \max(k_{min}, \lfloor \alpha \cdot \bar{C}_{j:k} \cdot k_{max} \rfloor))$
  Here, $\alpha$ is a global aggressiveness parameter. High confidence triggers aggressive drafting ($k_j$ approaches $\alpha \cdot k_{max}$), while low confidence leads to conservative drafting ($k_j$ reduces toward $k_{min}$)[^2].

* **Verification Threshold**: The likelihood margin for accepting a drafted token is dynamically adjusted:
  $\tau_t = \tau_{base} + \gamma \cdot (1 - C_t)$
  where $\tau_{base}$ is a fixed minimum tolerance and $\gamma$ is a tunable hyperparameter. This enables lenient acceptance when confidence is high and stricter verification when confidence is low, maintaining output fidelity while reducing rollbacks[^2]. This adaptive mechanism achieves significant speedups (up to 4-5x) over standard autoregressive decoding on benchmark tasks[^2].

### 4.4 Comparative Analysis and Resource Trade-offs

The following table summarizes and contrasts the core mechanisms of the adaptive approaches discussed:

| Approach            | Core Mechanism                  | Adjustment Scope                       | Confidence Estimation                           | Verification Adaptation                                    |
| :------------------ | :------------------------------ | :------------------------------------- | :---------------------------------------------- | :--------------------------------------------------------- |
| **AdaEAGLE (LDLP)** | Lightweight MLP predictor       | Draft length only                      | Implicit via context embeddings                 | None (fixed verification)                                  |
| **CM-ASD**          | Multi-metric confidence scoring | Draft length + verification thresholds | Explicit: entropy, logit margin, softmax margin | Dynamic thresholds: $\tau_t = \tau_{base} + \gamma(1-C_t)$ |

The implementation of these adaptive mechanisms involves critical resource trade-offs:

1. **Prediction Overhead vs. Computational Savings**: Lightweight predictors like LDLP add minimal overhead (a three-layer MLP) but yield substantial savings by reducing wasted draft tokens and target model passes[^7]. In contrast, CM-ASD's multi-metric confidence estimation provides finer-grained control but requires additional computation for entropy and margin calculations[^2].
2. **Balancing Acceptance Rates and Verification Cost**: The fundamental optimization is balancing the time spent on the draft model ($T_{draft}$) against the time spent on the target model for verification ($T_{target}$). Adaptive methods like AdaEAGLE aim to predict a draft length close to the optimal acceptance length, thereby minimizing both $N_{waste}$ and $N_{target}$[^7]. CM-ASD further optimizes this balance by modulating the verification strictness based on confidence, reducing rollback frequency[^2].
3. **Adaptive Pruning**: Advanced frameworks like STAR employ adaptive pruning within a Neural Architecture Search (NAS) to explicitly trade off draft model speed against token acceptance rate, searching for hardware-aware configurations that maximize overall speedup while preserving accuracy[^17].

### 4.5 Conclusion and Trajectory

The transition from static tree structures to confidence-aware adaptive approaches represents a significant advancement in speculative decoding. By dynamically adjusting parameters based on real-time model confidence and contextual uncertainty, these mechanisms directly target the computational inefficiency gap. They mitigate waste in high-confidence scenarios and enhance exploration in uncertain contexts. The trajectory points toward increasingly sophisticated, online, and hardware-aware adaptation, promising more robust and efficient LLM inference acceleration across diverse architectures and deployment scenarios.

## 5. Resource Trade-offs: Adaptive Pruning vs. Fixed-Structure Baselines

The fundamental challenge in optimizing speculative decoding lies in balancing the computational efficiency of draft generation against the verification overhead of the target model. Fixed-structure approaches, whether employing static tree architectures or predetermined draft lengths, often fail to adapt to the dynamic nature of language generation, leading to significant computational waste. This section analyzes the resource trade-offs between adaptive pruning mechanisms and their fixed-structure counterparts, quantifying improvements in throughput, latency reduction, and computational savings while maintaining output quality.

### 5.1 The Computational Waste Efficiency Gap in Static Structures

Static tree-based speculative decoding approaches, such as those pioneered by SpecInfer, introduce inherent computational inefficiencies that manifest as an "efficiency gap" between theoretical and practical performance[^1]. The primary sources of waste in fixed-structure systems include:

1. **Redundant Draft Computation**: When static draft lengths exceed the optimal acceptance length, significant computational resources are expended on generating tokens that will ultimately be rejected by the target model. As demonstrated in AdaEAGLE, static draft structures can lead to substantial wasted draft tokens ($N_{waste}$), with fixed-length approaches generating significantly more wasted tokens compared to adaptive methods[^7].
2. **Suboptimal Verification Frequency**: Static structures often trigger unnecessary verification steps by the target model. When draft lengths are too short, the target model must perform more frequent verification passes, increasing overall latency. Conversely, overly long drafts delay verification, potentially accumulating errors.
3. **Memory and Computation Overhead**: Static tree verification requires computing attention outputs for many speculated tokens that may not match the LLM's actual output, creating computation overhead[^1]. Furthermore, storing the parameters of multiple small speculative models (SSMs) and caching keys and values for all tokens in a static token tree introduces additional memory overhead, though this is often considered negligible compared to the key-value cache for very long sequences[^1].

### 5.2 Adaptive Pruning Mechanisms and Their Resource Optimization

#### 5.2.1 Neural Architecture Search for Hardware-Aware Draft Optimization

The STAR framework employs a Neural Architecture Search (NAS) approach for adaptive pruning that jointly optimizes multiple dimensions of draft model efficiency[^17]. Its NAS framework explores a search space that includes attention-head pruning to reduce computational complexity and visual token compression to alleviate memory bandwidth bottlenecks in Vision-Language Models (VLMs). This hardware-aware configuration search aims to maximize end-to-end speculative decoding speedup while maintaining task accuracy and accepted-token length[^17].

#### 5.2.2 Confidence-Modulated and Threshold-Based Adaptive Mechanisms

Adaptive Speculative Decoding (AdaSD) introduces two hyperparameter-free, dynamically adjusted thresholds to optimize resource use[^68]:

1. **Generation Threshold ($T_G$)**: Derived from token entropy, this threshold determines when to stop candidate token generation to prevent wasteful computation on uncertain predictions. It is updated as the mean entropy of all previously rejected tokens: $T_G = \text{avg}(e_r)$.
2. **Verification Threshold ($T_V$)**: Based on Jensen-Shannon (JS) distance between draft and target model distributions, this threshold adapts token acceptance criteria. It is updated as the midpoint between the mean JS distances of previously accepted and rejected tokens: $T_V = \frac{\text{avg}(d_a) + \text{avg}(d_r)}{2}$.
   These mechanisms allow AdaSD to achieve up to 49% speedup over standard speculative decoding while limiting accuracy degradation[^68].

Similarly, Confidence-Modulated Adaptive Speculative Decoding (CM-ASD) dynamically adjusts both the drafting window size and verification thresholds based on the drafter model's internal confidence score[^2]. The drafting length $k_j$ is determined by the average confidence over the next positions, allowing aggressive drafting when confidence is high and conservative drafting when it is low. The verification threshold $\tau_t$ is also confidence-modulated: $\tau_t = \tau_{base} + \gamma * (1 - C_t)$, leading to stricter verification in uncertain contexts[^2].

#### 5.2.3 Explicit Modeling of Adaptive Draft Lengths

AdaEAGLE addresses the inefficiencies of static structures through a Lightweight Draft Length Predictor (LDLP) module[^7]. This module explicitly predicts the optimal number of draft tokens ($k_r$) for each decoding iteration using a simple MLP that takes the embedding and last hidden state of the current token as context-aware input. The predicted length is given by $\bar{k}_{r+1} = \text{Round}(\text{MLP}(e_{j+k_r^∘}, f_{j+k_r^∘}))$ and constrained by a maximum $k_{max}$[^7]. This approach minimizes both wasted draft tokens ($N_{waste}$) and the number of costly target model forward passes ($N_{target}$), directly optimizing the trade-off between draft computation and verification overhead[^7].

### 5.3 Balancing Acceptance Rates and Verification Overhead

The core objective of adaptive pruning is to optimally balance the token acceptance rate with the computational cost of verification. STAR's NAS framework exemplifies this by searching for a draft model configuration that is fast enough for efficient drafting but accurate enough to ensure a high acceptance rate, thereby reducing the frequency of costly target model verifications[^17]. Its target-aware refinement mechanism uses Adaptive Intermediate Feature Distillation (AIFD) to guide the draft model training using the most informative intermediate layers from the target model, improving alignment and acceptance rates without degrading accuracy[^17].

AdaEAGLE balances this trade-off through the training objective of its LDLP module, which uses a penalized L1 loss. This loss function applies a penalty coefficient $\lambda > 1$ when the predicted draft length is less than the optimal ground truth length, encouraging predictions that avoid the high cost of additional target model forward passes[^7]. AdaEAGLE can also be integrated with threshold-based strategies like DDD for further refinement[^7].

### 5.4 Performance and Efficiency Gains

Adaptive pruning mechanisms demonstrate clear efficiency improvements over fixed-structure baselines:

* **Throughput and Speedup**: AdaSD achieves 23–49% higher throughput than vanilla speculative decoding on benchmark datasets[^68]. CM-ASD demonstrates speedups of 4.2x to 5.0x over autoregressive baselines on machine translation tasks[^2]. AdaEAGLE achieves a $1.62\times$ speedup over vanilla autoregressive decoding and outperforms fixed-length baselines[^7].

* **Reduced Waste**: AdaEAGLE significantly reduces the count of wasted draft tokens ($N_{waste}$) compared to fixed-length drafting[^7]. By avoiding over-aggressive drafting in low-confidence regions, CM-ASD reduces unnecessary rollbacks and wasted computation[^2].

* **Optimized Resource Utilization**: Adaptive methods like STAR's visual token compression directly reduce the memory bandwidth overhead associated with processing full visual tokens in the draft model, a major bottleneck in VLMs that static structures cannot mitigate[^17].

### 5.5 Conclusion on Resource Trade-offs

The resource trade-offs between adaptive pruning and fixed-structure baselines reveal a critical evolution in speculative decoding. While static approaches offer simplicity, they incur a significant efficiency gap due to redundant draft computation, suboptimal verification frequency, and rigid memory usage[^1]  [^7]. Adaptive mechanisms—whether through NAS-based hardware-aware optimization[^17], confidence-modulated parameter adjustment[^2], or explicit draft length prediction[^7]—dynamically balance acceptance rates with verification overhead. This leads to substantial improvements in throughput, reductions in computational waste, and more efficient resource utilization, all while preserving output quality, positioning adaptive pruning as a superior paradigm for efficient LLM inference.

## 6. Comparative Analysis: Static Tree-Based versus Confidence-Aware Adaptive Methods

The progression from linear speculative decoding to tree-based approaches, pioneered by SpecInfer, fundamentally altered LLM inference acceleration by enabling the parallel verification of multiple candidate token sequences[^1]. However, the computational inefficiencies inherent in static, predetermined tree structures have spurred the development of confidence-aware adaptive methods. This section provides a detailed comparative analysis across dimensions of computational efficiency, flexibility, implementation complexity, and scalability, synthesizing evidence into actionable insights.

### 6.1 Computational Efficiency: Bridging the Efficiency Gap

The core distinction between static and adaptive methods manifests in what can be termed the "Efficiency Gap": the imbalance between computational waste in high-confidence scenarios and insufficient speculative exploration in uncertain contexts. Static tree-based methods, such as early implementations of SpecInfer, employ fixed branching factors and depths regardless of the generative context\[\[1]\[16]]. This rigidity leads to two primary inefficiencies:

1. **High-Confidence Waste**: In contexts with low token entropy (high confidence), static trees generate numerous speculative branches that are unlikely to be accepted. This results in wasted computation from both the draft model and the verification process, as the target model computes attention outputs for many tokens that do not match its actual output[^1].
2. **Uncertainty Under-Exploration**: Conversely, in ambiguous, high-entropy contexts, a fixed tree structure may not explore enough plausible continuations, limiting the potential gains from parallel verification and leading to more frequent, costly rollbacks.

Adaptive approaches directly target this gap by dynamically adjusting parameters based on real-time confidence estimates. For instance, Confidence-Modulated Adaptive Speculative Decoding (CM-ASD) modulates both the drafting length and verification strictness. The number of speculatively generated tokens $k_j$ is determined by the formula $k_j = \min(k_{max}, \max(k_{min}, \lfloor \alpha * \bar{C}_{j:k} * k_{max} \rfloor))$, where $\bar{C}_{j:k}$ is the average confidence over the next *k* positions[^2]. This allows for aggressive drafting when confidence is high and conservative drafting when it is low. Simultaneously, the verification threshold $\tau_t$ is adjusted as $\tau_t = \tau_{base} + \gamma * (1 - C_t)$, making acceptance more lenient in high-confidence regions and stricter in low-confidence ones[^2].

The comparative impact on computational efficiency is summarized below:

| **Efficiency Dimension**         | **Static Tree Methods**                    | **Adaptive Approaches**                        | **Performance Impact**                                      |
| :------------------------------- | :----------------------------------------- | :--------------------------------------------- | :---------------------------------------------------------- |
| **High-Confidence Scenarios**    | Fixed branching generates redundant nodes  | Dynamic pruning reduces candidate tokens       | Reduces wasted draft computation ($N_{waste}$)[^7]          |
| **Uncertain Contexts**           | Limited exploration due to fixed structure | Expanded branching based on confidence/entropy | Increases acceptance length, reducing rollbacks[^2]         |
| **Verification Overhead**        | Constant per tree regardless of context    | Adaptive thresholds modulate verification cost | Optimizes balance between acceptance rate and overhead[^2]  |
| **Throughput-Latency Trade-off** | Suboptimal due to fixed draft latency      | Context-aware draft length prediction          | Achieves significant speedups (e.g., 1.62×–5.0×)\[\[2]\[7]] |

### 6.2 Flexibility and Context Adaptation

Language generation is inherently dynamic, with the optimal speculative path changing rapidly based on context. Static tree structures struggle with this variability[^16]. Adaptive methods introduce several mechanisms for context-aware tree construction:

* **Probability-Aware Tree Construction**: Algorithms like OPT-Tree dynamically construct draft trees by maximizing the expected acceptance length based on the draft model's token probability distributions, moving beyond fixed heuristics[^71].

* **Multi-Feature Confidence Estimation**: Rather than relying on a single metric, advanced methods like CM-ASD combine entropy-based, logit-margin, and softmax-margin confidence scores into a unified scalar $C_t$ for more robust uncertainty quantification[^2].

* **Explicit Length Prediction**: Frameworks such as AdaEAGLE employ a Lightweight Draft Length Predictor (LDLP), a small MLP that explicitly predicts the optimal number of draft tokens $k_r$ for the next iteration based on contextual embeddings and hidden states[^7].

These mechanisms enable a decision flow where the token probability distribution informs a confidence estimate, which in turn dynamically controls the drafting strategy (aggressive vs. conservative) and subsequent verification thresholds to optimize the final acceptance rate.

### 6.3 Implementation Complexity and Scalability

While adaptive methods offer superior efficiency and flexibility, they introduce additional implementation complexity that must be managed.

* **Training and Overhead**: Adaptive systems often require additional training or data collection. For example, AdaEAGLE's LDLP module must be trained on pairs of token states and optimal draft lengths using a specialized penalized L1 loss function[^7]. CM-ASD requires tuning hyperparameters like the aggressiveness factor $\alpha$ and threshold coefficient $\gamma$[^2]. Static tree methods, in contrast, have no such overhead beyond training the draft model itself.

* **Runtime Decision Cost**: The dynamic tree construction or parameter adjustment in adaptive methods incurs per-step computational overhead. However, research indicates this cost is typically offset by the reduction in wasted computation and improved acceptance rates[^7].

* **Scalability and Hardware Awareness**: Static methods have predictable computation patterns, easing distributed system scaling. Adaptive methods require careful engineering to maintain scalability. Some advanced systems, like STAR, integrate Neural Architecture Search (NAS) within a hardware-aware framework to jointly optimize draft model configuration (e.g., via attention-head pruning) for the best balance of draft speed and acceptance rate on target hardware[^17].

### 6.4 Performance Benchmarks

Experimental evaluations demonstrate the tangible benefits of adaptive methods:

* **Throughput Speedups**: On standard benchmarks, adaptive methods show substantial acceleration. AdaEAGLE reports a throughput of 66.35 tok/s and a 1.62× speedup over vanilla autoregressive decoding[^7]. CM-ASD achieves even greater gains of 4–5× speedup on WMT translation tasks while preserving output quality metrics like BLEU and ROUGE[^2].

* **Reduction in Waste**: A key efficiency metric is the reduction in wasted draft tokens ($N_{waste}$). AdaEAGLE significantly lowers this figure compared to fixed-length baselines, directly translating to better resource utilization[^7].

### 6.5 Practitioner Insights and Deployment Recommendations

The choice between static and adaptive speculative decoding depends on specific deployment constraints:

**Consider Static Tree Methods When:**

* Workloads are predictable with consistent context lengths and query patterns.

* Implementation simplicity is a higher priority than maximizing throughput.

* Computational resources for training and running adaptive components are limited.

* Integrating with legacy systems where adding dynamic decision logic is prohibitive.

**Prefer Adaptive Approaches When:**

* Facing variable workloads with diverse contexts and confidence patterns.

* Maximizing throughput and computational efficiency is critical.

* Maintaining high output fidelity while accelerating is paramount, especially in long-context or quality-sensitive applications.

**Implementation Strategy:**

1. **Profile First**: Instrument static decoding to measure the "efficiency gap" specific to your application—identify where waste or under-exploration occurs.
2. **Adopt Incrementally**: Start with hybrid approaches, such as adding adaptive pruning to a static tree, before implementing full dynamic tree construction.
3. **Evaluate Trade-offs**: Carefully weigh the expected performance gains of an adaptive method against its added training, complexity, and runtime decision costs.

**Remaining Challenges and Future Directions:**

* **Optimality-Speed Trade-off**: Developing faster algorithms for high-quality adaptive tree construction.

* **Generalization**: Creating adaptive mechanisms that transfer well across different LLM architectures and tasks.

* **Holistic Co-design**: Further integrating adaptive algorithms with hardware capabilities and memory hierarchies.

* **Uncertainty Estimation**: Improving confidence scoring methods, particularly for novel or out-of-distribution inputs.

In summary, the evolution from static to adaptive tree-based speculative decoding marks a shift toward more intelligent, context-aware acceleration. While static methods provide a foundation of simplicity and predictable performance, adaptive approaches offer a path to close the efficiency gap through dynamic, confidence-driven optimization. The optimal choice is application-dependent, with hybrid strategies serving as a practical intermediate step for many real-world deployments.

## 7. Future Directions and Emerging Paradigms

The evolution from linear to tree-based speculative decoding, exemplified by SpecInfer, has advanced LLM inference acceleration significantly [^1]. However, the computational inefficiencies and rigidity of static tree structures continue to motivate novel research. The emerging landscape points toward three transformative directions that promise to bridge the persistent efficiency gap: reinforcement learning-driven adaptive tree construction, hardware-aware optimization frameworks, and extensions to multi-modal models. These paradigms collectively aim to address the core tension between computational waste in high-confidence scenarios and insufficient exploration in uncertain contexts.

### 7.1 Reinforcement Learning for Adaptive Tree Construction

Static tree structures are fundamentally limited by their inability to adapt to the dynamic probability landscape of token generation [^16]. Reinforcement learning (RL) offers a principled framework for real-time optimization of the drafting and verification trade-off. A key innovation is **Learning to Draft (LTD)**, which directly optimizes throughput—accepted tokens per total time—by training two co-adaptive RL policies [^79]. A **depth policy** dynamically controls the draft tree's depth (influencing drafting cost), while a **size policy** manages the verification size (impacting verification cost). These policies are jointly optimized to maximize decoding efficiency, achieving speedup ratios of 2.24× to 4.32× and outperforming strong baselines by up to 36.4% [^79]. This RL approach proves robust even in high-temperature decoding scenarios where traditional methods degrade.

Another promising, training-free direction is **BanditSpec**, which formulates hyperparameter selection (e.g., drafting length) as a Multi-Armed Bandit problem [^82]. This online learning framework adaptively chooses configurations as text is generated, achieving throughput close to the oracle-best hyperparameter in diverse serving scenarios with provable optimal regret bounds [^82]. The integration of RL and bandit algorithms with tree-based decoding creates an intelligent feedback loop, enabling systems to learn optimal branching factors and depth based on real-time token probability distributions.

### 7.2 Hardware-Aware Optimization and Memory Hierarchy Exploitation

Computational inefficiency is fundamentally constrained by hardware limitations, particularly memory bandwidth and GPU architecture. Future systems must be co-designed with hardware in mind. Research like **Sequoia** introduces scalable, robust, and hardware-aware speculative decoding that leverages dynamic programming for optimal tree structures [^17]. Such systems optimize for specific memory hierarchies (e.g., balancing SRAM and HBM usage) and GPU pipelines to minimize verification overhead and execution latency.

For specialized hardware, **SpecMamba** demonstrates FPGA-based acceleration for State Space Models using speculative decoding, achieving significant speedups over GPU baselines through memory-aware hybrid backtracking and tiled verification [^17]. Hardware-aware optimization also involves adaptive pruning mechanisms, as seen in the **STAR** system, which uses a Neural Architecture Search (NAS) framework to jointly optimize draft model configuration (e.g., attention-head pruning, visual token compression) and verification strategies [^17]. This search maximizes overall speedup while maintaining task accuracy by finding the optimal balance between draft model speed and token acceptance rate.

### 7.3 Multi-Modal Speculative Decoding Extensions

Extending speculative decoding to Vision-Language Models (VLMs) introduces unique challenges, primarily due to the prefill stage being dominated by a large number of visual tokens, which inflates compute and memory requirements for the Key-Value (KV) cache [^81]. **SpecVLM** addresses this with an **elastic visual compressor** that adaptively selects among compression primitives—pruning, pooling, convolution, and resamplers—to balance computational resources (FLOPs/parameters) and accuracy per input [^81]. This adaptive selection is crucial for handling varying image resolutions and task difficulties.

Furthermore, SpecVLM employs an **online-logit distillation protocol** that trains the draft model using on-the-fly teacher logits and features, eliminating the need for costly offline distillation datasets [^81]. This protocol reveals a critical **training-time scaling effect**: longer online training monotonically increases the draft model's average accepted length, thereby directly improving speculative efficiency [^81]. This finding suggests multimodal speculative decoding benefits from extended, targeted training regimes. Future work must continue to optimize the simultaneous compression of visual tokens and alignment of cross-modal contexts to reduce KV cache traffic and memory bandwidth pressure.

### 7.4 Novel Architectural Paradigms and System Integration

Beyond algorithmic tweaks, holistic architectural innovations are emerging. **AdaEAGLE** explicitly models adaptive draft structures through a Lightweight Draft Length Predictor (LDLP), a small MLP that predicts the optimal number of draft tokens ($k_r$) for each decoding iteration [^7]. By minimizing the discrepancy between predicted and optimal acceptance lengths, it reduces wasted draft tokens and costly target model forward passes, achieving a 1.62× speedup over vanilla autoregressive decoding [^7].

The integration of **confidence-modulated mechanisms** represents another architectural breakthrough. **Confidence-Modulated Adaptive Speculative Decoding (CM-ASD)** dynamically adjusts the drafting window size ($k_j$) and verification thresholds ($\tau_t$) based on the drafter model's internal confidence score ($C_t$) [^2]. This allows for aggressive drafting in high-confidence regions and conservative, stricter verification in uncertain areas, reducing rollback frequency and improving resource utilization to achieve up to 4-5× speedup [^2].

Additionally, frameworks like **OPT-Tree** construct adaptive draft trees by greedily sampling tokens with the largest draft model probabilities to maximize the mathematical expectation of acceptance length ($E(A)$) in each step, offering a systematic alternative to static heuristics [^80].

### 7.5 Resource Trade-off Optimization and Adaptive Pruning

The core challenge remains optimizing the trade-off between acceptance rates and verification overhead. Adaptive methods demonstrate clear advantages over static baselines:

| Metric                            | Static Tree Methods          | Adaptive Approaches                  | Improvement                           |
| --------------------------------- | ---------------------------- | ------------------------------------ | ------------------------------------- |
| Mean Acceptance Length (MAL)      | Limited by fixed structure   | Dynamically maximized [^80]          | Up to 3.2× speedup [^80]              |
| Verification Overhead             | Constant per iteration       | Context-dependent reduction          | Not quantified                        |
| Computational Waste ($N_{waste}$) | High in mismatched scenarios | Minimized via length prediction [^7] | \~47% reduction vs. fixed-length [^7] |
| Throughput                        | Suboptimal balance           | Hardware-aware optimization [^17]    | Significant gains                     |

Adaptive pruning, as in STAR, balances acceptance rates and verification overhead by searching for configurations that minimize end-to-end latency while preserving accepted-token length and accuracy [^17]. Similarly, AdaEAGLE's LDLP is trained with a penalized L1 loss that more severely penalizes under-prediction of the draft length, encouraging longer drafts to reduce the frequency of expensive target model verifications [^7].

### 7.6 Concluding Vision: Toward Self-Optimizing Inference Systems

The convergence of RL-driven adaptation, hardware-aware co-design, and multi-modal extensions points toward a future of **self-optimizing inference systems**. These systems would autonomously and continuously adapt their speculative decoding architecture—tree depth, branching factors, pruning strategies—based on real-time performance metrics, hardware characteristics, and task-specific requirements. Breakthroughs will likely come from cross-disciplinary integration, combining insights from compiler optimization, hardware architecture, and online learning theory. As LLMs grow in scale and complexity, such adaptive and intelligent speculative decoding paradigms will be essential for enabling efficient, scalable, and practical inference across diverse real-world applications.

## 8. Reference

[^1]: \[PDF] Specinfer: Accelerating generative llm serving with speculative ..., <a href="https://arxiv.org/pdf/2305.09781" target="_blank"><https://arxiv.org/pdf/2305.09781></a>

[^2]: \[PDF] Confidence-Modulated Speculative Decoding for Large Language ..., <a href="https://arxiv.org/pdf/2508.15371" target="_blank"><https://arxiv.org/pdf/2508.15371></a>

[^3]: Adaptive Speculative Inference - Emergent Mind, <a href="https://www.emergentmind.com/topics/adaptive-speculative-inference" target="_blank"><https://www.emergentmind.com/topics/adaptive-speculative-inference></a>

[^4]: Speculative Decoding for Much Faster LLMs | by M - Towards AI, <a href="https://pub.towardsai.net/speculative-decoding-for-much-faster-llms-da5297863ebe" target="_blank"><https://pub.towardsai.net/speculative-decoding-for-much-faster-llms-da5297863ebe></a>

[^5]: Optimizing Speculative Decoding for Serving Large Language ..., <a href="https://arxiv.org/html/2406.14066v2" target="_blank"><https://arxiv.org/html/2406.14066v2></a>

[^6]: A Speculative LLM Decoding Framework for Efficient Edge Serving, <a href="https://dl.acm.org/doi/pdf/10.1145/3769102.3770608" target="_blank"><https://dl.acm.org/doi/pdf/10.1145/3769102.3770608></a>

[^7]: AdaEAGLE: Optimizing Speculative Decoding via Explicit Modeling of Adaptive Draft Structures, <a href="https://arxiv.org/pdf/2412.18910" target="_blank"><https://arxiv.org/pdf/2412.18910></a>

[^8]: Speculative Decoding for Multi-Sample Inference, <a href="https://doi.org/10.48550/arxiv.2503.05330" target="_blank"><https://doi.org/10.48550/arxiv.2503.05330></a>

[^9]: Closer Look at Efficient Inference Methods: A Survey of Speculative Decoding, <a href="https://doi.org/10.48550/arxiv.2411.13157" target="_blank"><https://doi.org/10.48550/arxiv.2411.13157></a>

[^10]: Tutorial Proposal: Speculative Decoding for Efficient LLM Inference, <a href="https://doi.org/10.48550/arxiv.2503.00491" target="_blank"><https://doi.org/10.48550/arxiv.2503.00491></a>

[^11]: Not-a-Bandit: Provably No-Regret Drafter Selection in Speculative ..., <a href="https://arxiv.org/html/2510.20064v1" target="_blank"><https://arxiv.org/html/2510.20064v1></a>

[^12]: \[PDF] Adaptive Draft-Verification for Efficient Large Language Model ..., <a href="https://ojs.aaai.org/index.php/AAAI/article/view/34647/36802" target="_blank"><https://ojs.aaai.org/index.php/AAAI/article/view/34647/36802></a>

[^13]: \[PDF] Boosting Speculative Decoding via Adaptive Candidate Lengths, <a href="https://openreview.net/pdf?id=ZwwY5UgNGh" target="_blank"><https://openreview.net/pdf?id=ZwwY5UgNGh></a>

[^14]: Speculative Decoding in Decentralized LLM Inference - arXiv, <a href="https://arxiv.org/html/2511.11733v1" target="_blank"><https://arxiv.org/html/2511.11733v1></a>

[^15]: SpecDec++: Boosting Speculative Decoding via Adaptive Candidate ..., <a href="https://www.researchgate.net/publication/381006280_SpecDec_Boosting_Speculative_Decoding_via_Adaptive_Candidate_Lengths" target="_blank"><https://www.researchgate.net/publication/381006280_SpecDec_Boosting_Speculative_Decoding_via_Adaptive_Candidate_Lengths></a>

[^16]: \[PDF] A Comprehensive Survey of Speculative Decoding - ACL Anthology, <a href="https://aclanthology.org/2024.findings-acl.456.pdf" target="_blank"><https://aclanthology.org/2024.findings-acl.456.pdf></a>

[^17]: STAR: Speculative Decoding with Searchable Drafting and..., <a href="https://openreview.net/forum?id=pMdKnxkFRw" target="_blank"><https://openreview.net/forum?id=pMdKnxkFRw></a>

[^18]: A Systematic Survey on Decoding Methods for Foundation ..., <a href="https://www.researchgate.net/profile/Haoran-Wang-96/publication/387703971_Make_Every_Token_Count_A_Systematic_Survey_on_Decoding_Methods_for_Foundation_Models/links/67784c8ce74ca64e1f49eb15/Make-Every-Token-Count-A-Systematic-Survey-on-Decoding-Methods-for-Foundation-Models.pdf" target="_blank"><https://www.researchgate.net/profile/Haoran-Wang-96/publication/387703971_Make_Every_Token_Count_A_Systematic_Survey_on_Decoding_Methods_for_Foundation_Models/links/67784c8ce74ca64e1f49eb15/Make-Every-Token-Count-A-Systematic-Survey-on-Decoding-Methods-for-Foundation-Models.pdf></a>

[^19]: Efficient LLM System with Speculative Decoding by Xiaoxuan Liu A ..., <a href="https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.pdf" target="_blank"><https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.pdf></a>

[^20]: Speculative Decoding with CTC-based Draft Model for LLM ..., <a href="https://arxiv.org/html/2412.00061v1" target="_blank"><https://arxiv.org/html/2412.00061v1></a>

[^21]: Speculative Decoding with CTC-based Draft Model for LLM ..., <a href="https://openreview.net/pdf?id=pGeAcYhnN5" target="_blank"><https://openreview.net/pdf?id=pGeAcYhnN5></a>

[^22]: ResDecode: Accelerating Large Language Models ..., <a href="https://www.sciopen.com/article/10.26599/BDMA.2024.9020074" target="_blank"><https://www.sciopen.com/article/10.26599/BDMA.2024.9020074></a>

[^23]: Speculative Decoding with CTC-based Draft Model for LLM ..., <a href="https://arxiv.org/pdf/2412.00061" target="_blank"><https://arxiv.org/pdf/2412.00061></a>

[^24]: arXiv:2405.13019v2 \[cs.CL] 24 May 2024 - Aman Chadha, <a href="https://www.amanchadha.com/research/2405.13019v2.pdf" target="_blank"><https://www.amanchadha.com/research/2405.13019v2.pdf></a>

[^25]: ResDecode: Accelerating Large Language Models ..., <a href="https://ieeexplore.ieee.org/iel8/8254253/11002434/11002449.pdf" target="_blank"><https://ieeexplore.ieee.org/iel8/8254253/11002434/11002449.pdf></a>

[^26]: C2T: A Classifier-Based Tree Construction Method in Speculative Decoding, <a href="https://arxiv.org/pdf/2502.13652" target="_blank"><https://arxiv.org/pdf/2502.13652></a>

[^27]: STree: Speculative Tree Decoding for Hybrid State-Space Models, <a href="https://arxiv.org/pdf/2505.14969" target="_blank"><https://arxiv.org/pdf/2505.14969></a>

[^28]: Medusa: Simple LLM Inference Acceleration Framework with ... - arXiv, <a href="https://arxiv.org/html/2401.10774v2" target="_blank"><https://arxiv.org/html/2401.10774v2></a>

[^29]: Accelerating Large Language Model Inference via Speculative..., <a href="https://openreview.net/forum?id=iW4lyuOQ0J" target="_blank"><https://openreview.net/forum?id=iW4lyuOQ0J></a>

[^30]: \[PDF] SpecInfer: Accelerating Large Language Model Serving with Tree ..., <a href="https://openreview.net/pdf/70a5314a7cafef983655e10723381820dea0c382.pdf" target="_blank"><https://openreview.net/pdf/70a5314a7cafef983655e10723381820dea0c382.pdf></a>

[^31]: \[Feature]: Tree-Attention Support for Speculative Decoding #18327, <a href="https://github.com/vllm-project/vllm/issues/18327" target="_blank"><https://github.com/vllm-project/vllm/issues/18327></a>

[^32]: \[PDF] due to their large volume of parameters, complex architec, <a href="https://par.nsf.gov/servlets/purl/10535279" target="_blank"><https://par.nsf.gov/servlets/purl/10535279></a>

[^33]: \[PDF] DySpec: Faster speculative decoding with dynamic token tree structure, <a href="https://www.wict.pku.edu.cn/docs/20250717175203347541.pdf" target="_blank"><https://www.wict.pku.edu.cn/docs/20250717175203347541.pdf></a>

[^34]: SpecInfer: Accelerating Large Language Model Serving with Tree ..., <a href="https://arxiv.org/html/2305.09781v4" target="_blank"><https://arxiv.org/html/2305.09781v4></a>

[^35]: \[PDF] Accelerating State Space Model Inference with Speculative Decoding, <a href="https://assets.amazon.science/45/72/4848937d41b0b152ab24f1ca7d41/snakes-and-ladders-accelerating-state-space-model-inference-with-speculative-decoding.pdf" target="_blank"><https://assets.amazon.science/45/72/4848937d41b0b152ab24f1ca7d41/snakes-and-ladders-accelerating-state-space-model-inference-with-speculative-decoding.pdf></a>

[^36]: SpecInfer: Accelerating Generative Large Language Model Serving ..., <a href="https://arxiv.org/abs/2305.09781" target="_blank"><https://arxiv.org/abs/2305.09781></a>

[^37]: SpecInfer: Accelerating Large Language Model Serving with Tree ..., <a href="https://www.researchgate.net/publication/380150985_SpecInfer_Accelerating_Large_Language_Model_Serving_with_Tree-based_Speculative_Inference_and_Verification" target="_blank"><https://www.researchgate.net/publication/380150985_SpecInfer_Accelerating_Large_Language_Model_Serving_with_Tree-based_Speculative_Inference_and_Verification></a>

[^38]: \[PDF] A Survey on Efficient Inference for Large Language Models, <a href="https://nicsefc.ee.tsinghua.edu.cn/nics_file/pdf/1c678c23-69df-405b-992d-130fc6d5a4f5.pdf" target="_blank"><https://nicsefc.ee.tsinghua.edu.cn/nics_file/pdf/1c678c23-69df-405b-992d-130fc6d5a4f5.pdf></a>

[^39]: Quick Overview on LLM Serving and Benchmarking - Jensen Low, <a href="https://www.jensenlwt.com/blog/quick-overview-on-llm-serving-and-benchmarking/" target="_blank"><https://www.jensenlwt.com/blog/quick-overview-on-llm-serving-and-benchmarking/></a>

[^40]: chenhongyu2048/LLM-inference-optimization-paper - GitHub, <a href="https://github.com/chenhongyu2048/LLM-inference-optimization-paper" target="_blank"><https://github.com/chenhongyu2048/LLM-inference-optimization-paper></a>

[^41]: OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure, <a href="https://transacl.org/index.php/tacl/article/view/6873" target="_blank"><https://transacl.org/index.php/tacl/article/view/6873></a>

[^42]: Faster Speculative Decoding via Effective Draft ... - ACL Anthology, <a href="https://aclanthology.org/2025.acl-long.486/" target="_blank"><https://aclanthology.org/2025.acl-long.486/></a>

[^43]: DeFT: Decoding with Flash Tree-attention for Efficient... - OpenReview, <a href="https://openreview.net/forum?id=2c7pfOqu9k" target="_blank"><https://openreview.net/forum?id=2c7pfOqu9k></a>

[^44]: CAS-Spec: Cascade Adaptive Self-Speculative Decoding for On-the-Fly Lossless Inference Acceleration of LLMs, <a href="https://arxiv.org/pdf/2510.26843" target="_blank"><https://arxiv.org/pdf/2510.26843></a>

[^45]: DySpec: Faster speculative decoding with dynamic token tree structure, <a href="https://doi.org/10.1007/s11280-025-01344-0" target="_blank"><https://doi.org/10.1007/s11280-025-01344-0></a>

[^46]: Accelerated Test-Time Scaling with Model-Free ..., <a href="https://aclanthology.org/2025.emnlp-main.1558.pdf" target="_blank"><https://aclanthology.org/2025.emnlp-main.1558.pdf></a>

[^47]: Accelerated Test-Time Scaling with Model-Free ..., <a href="https://openreview.net/pdf?id=2QThsA6lM8" target="_blank"><https://openreview.net/pdf?id=2QThsA6lM8></a>

[^48]: (PDF) Accelerated Test-Time Scaling with Model-Free ..., <a href="https://www.researchgate.net/publication/392466452_Accelerated_Test-Time_Scaling_with_Model-Free_Speculative_Sampling" target="_blank"><https://www.researchgate.net/publication/392466452_Accelerated_Test-Time_Scaling_with_Model-Free_Speculative_Sampling></a>

[^49]: Accelerated Test-Time Scaling with Model-Free ..., <a href="https://arxiv.org/pdf/2506.04708" target="_blank"><https://arxiv.org/pdf/2506.04708></a>

[^50]: Turning Trash into Treasure: Accelerating Inference of ..., <a href="https://aclanthology.org/2025.acl-long.338.pdf" target="_blank"><https://aclanthology.org/2025.acl-long.338.pdf></a>

[^51]: AdaEDL: Early Draft Stopping for Speculative Decoding of Large Language Models via an Entropy-based Lower Bound on Token Acceptance Probability, <a href="https://arxiv.org/pdf/2410.18351" target="_blank"><https://arxiv.org/pdf/2410.18351></a>

[^52]: AdaEAGLE: Optimizing Speculative Decoding via Explicit ..., <a href="https://arxiv.org/abs/2412.18910" target="_blank"><https://arxiv.org/abs/2412.18910></a>

[^53]: AdaEAGLE: Optimizing Speculative Decoding via Explicit ..., <a href="https://huggingface.co/papers/2412.18910" target="_blank"><https://huggingface.co/papers/2412.18910></a>

[^54]: Dynamic Draft Length Selection, <a href="https://www.emergentmind.com/topics/dynamic-draft-length-selection" target="_blank"><https://www.emergentmind.com/topics/dynamic-draft-length-selection></a>

[^55]: (PDF) AdaEAGLE: Optimizing Speculative Decoding via Explicit ..., <a href="https://www.researchgate.net/publication/387512253_AdaEAGLE_Optimizing_Speculative_Decoding_via_Explicit_Modeling_of_Adaptive_Draft_Structures" target="_blank"><https://www.researchgate.net/publication/387512253_AdaEAGLE_Optimizing_Speculative_Decoding_via_Explicit_Modeling_of_Adaptive_Draft_Structures></a>

[^56]: Optimizing Speculative Decoding via Explicit Modeling of ..., <a href="https://www.themoonlight.io/en/review/adaeagle-optimizing-speculative-decoding-via-explicit-modeling-of-adaptive-draft-structures" target="_blank"><https://www.themoonlight.io/en/review/adaeagle-optimizing-speculative-decoding-via-explicit-modeling-of-adaptive-draft-structures></a>

[^57]: Adaptive Speculative Decoding, <a href="https://www.emergentmind.com/topics/adaptive-speculative-decoding" target="_blank"><https://www.emergentmind.com/topics/adaptive-speculative-decoding></a>

[^58]: Confidence-Modulated Speculative Decoding for Large Language ..., <a href="https://arxiv.org/abs/2508.15371" target="_blank"><https://arxiv.org/abs/2508.15371></a>

[^59]: Adaptive Grouped Speculative Decoding - Emergent Mind, <a href="https://www.emergentmind.com/topics/adaptive-grouped-speculative-decoding" target="_blank"><https://www.emergentmind.com/topics/adaptive-grouped-speculative-decoding></a>

[^60]: Confidence-Modulated Speculative Decoding for Large Language ..., <a href="https://www.researchgate.net/publication/394484682_Confidence-Modulated_Speculative_Decoding_for_Large_Language_Models" target="_blank"><https://www.researchgate.net/publication/394484682_Confidence-Modulated_Speculative_Decoding_for_Large_Language_Models></a>

[^61]: Confidence-Modulated Speculative Decoding for Large Language ..., <a href="https://www.semanticscholar.org/paper/Confidence-Modulated-Speculative-Decoding-for-Large-Sen-Dasgupta/4969326024776f4ad7d6788aff8b0306b242440f" target="_blank"><https://www.semanticscholar.org/paper/Confidence-Modulated-Speculative-Decoding-for-Large-Sen-Dasgupta/4969326024776f4ad7d6788aff8b0306b242440f></a>

[^62]: Confidence-Modulated Speculative Decoding for Large Language ..., <a href="https://papers.cool/arxiv/2508.15371" target="_blank"><https://papers.cool/arxiv/2508.15371></a>

[^63]: I will be presenting our work “Confidence-Modulated Adaptive ..., <a href="https://www.facebook.com/jaydip.sen.35/posts/i-will-be-presenting-our-work-confidence-modulated-adaptive-speculative-decoding/10162073728266235/" target="_blank"><https://www.facebook.com/jaydip.sen.35/posts/i-will-be-presenting-our-work-confidence-modulated-adaptive-speculative-decoding/10162073728266235/></a>

[^64]: Speculative Decoding and Beyond: An In-Depth Survey of Techniques, <a href="https://arxiv.org/html/2502.19732v4" target="_blank"><https://arxiv.org/html/2502.19732v4></a>

[^65]: International Conference on Representation Learning 2025 (ICLR ..., <a href="https://proceedings.iclr.cc/paper_files/paper/2025" target="_blank"><https://proceedings.iclr.cc/paper_files/paper/2025></a>

[^66]: r/ollama - Speculative decoding via Arch (candidate release 0.4.0 ..., <a href="https://www.reddit.com/r/ollama/comments/1mqr4st/speculative_decoding_via_arch_candidate_release/" target="_blank"><https://www.reddit.com/r/ollama/comments/1mqr4st/speculative_decoding_via_arch_candidate_release/></a>

[^67]: LLM Inference Optimization Research, <a href="https://www.aussieai.com/research/inference-optimization" target="_blank"><https://www.aussieai.com/research/inference-optimization></a>

[^68]: AdaSD: Adaptive Speculative Decoding for Efficient Language Model Inference, <a href="https://arxiv.org/pdf/2512.11280" target="_blank"><https://arxiv.org/pdf/2512.11280></a>

[^69]: SpecServe: Efficient and SLO-Aware Large Language Model Serving with Adaptive Speculative Decoding, <a href="https://arxiv.org/pdf/2503.05096" target="_blank"><https://arxiv.org/pdf/2503.05096></a>

[^70]: Recursive Speculative Decoding: Accelerating LLM Inference via Sampling Without Replacement, <a href="https://doi.org/10.48550/arXiv.2402.14160" target="_blank"><https://doi.org/10.48550/arXiv.2402.14160></a>

[^71]: Speculative Decoding with Adaptive Draft Tree Structure, <a href="https://aclanthology.org/2025.tacl-1.8/" target="_blank"><https://aclanthology.org/2025.tacl-1.8/></a>

[^72]: DySpec: Faster Speculative Decoding with Dynamic Token ..., <a href="https://openreview.net/forum?id=orr5uPZY28" target="_blank"><https://openreview.net/forum?id=orr5uPZY28></a>

[^73]: Towards Efficient LLM Inference via Collective and ..., <a href="https://dl.acm.org/doi/10.1145/3712285.3759834" target="_blank"><https://dl.acm.org/doi/10.1145/3712285.3759834></a>

[^74]: Speculative Decoding with Adaptive Draft Tree Structure, <a href="https://arxiv.org/abs/2406.17276" target="_blank"><https://arxiv.org/abs/2406.17276></a>

[^75]: RASD: Retrieval-Augmented Speculative Decoding, <a href="https://doi.org/10.48550/arxiv.2503.03434" target="_blank"><https://doi.org/10.48550/arxiv.2503.03434></a>

[^76]: Improving Speculative Decoding with Dynamic Adjustment and Probability Smoothing, <a href="https://doi.org/10.1109/ICAACE65325.2025.11019425" target="_blank"><https://doi.org/10.1109/ICAACE65325.2025.11019425></a>

[^77]: SpecDec++: Boosting Speculative Decoding via Adaptive Candidate Lengths, <a href="https://doi.org/10.48550/arxiv.2405.19715" target="_blank"><https://doi.org/10.48550/arxiv.2405.19715></a>

[^78]: Speculative Decoding via Hybrid Drafting and Rollback-Aware Branch Parallelism, <a href="https://doi.org/10.48550/arxiv.2506.01979" target="_blank"><https://doi.org/10.48550/arxiv.2506.01979></a>

[^79]: Adaptive Speculative Decoding with Reinforcement Learning, <a href="https://openreview.net/forum?id=IK9cbzzXLt" target="_blank"><https://openreview.net/forum?id=IK9cbzzXLt></a>

[^80]: \[PDF] OPT-Tree: Speculative Decoding with Adaptive Draft Tree Structure, <a href="https://aclanthology.org/2025.tacl-1.8.pdf" target="_blank"><https://aclanthology.org/2025.tacl-1.8.pdf></a>

[^81]: SpecVLM: Fast Speculative Decoding in Vision-Language ..., <a href="https://arxiv.org/html/2509.11815v1" target="_blank"><https://arxiv.org/html/2509.11815v1></a>

[^82]: BanditSpec: Adaptive Speculative Decoding via Bandit Algorithms, <a href="https://arxiv.org/pdf/2505.15141" target="_blank"><https://arxiv.org/pdf/2505.15141></a>

