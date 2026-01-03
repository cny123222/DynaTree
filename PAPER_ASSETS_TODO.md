## Paper assets to add (placeholders already inserted / to be inserted)

This checklist is meant to be filled by you after we freeze the text. Items are ordered by importance.

### Figures (main paper)

- **Figure: Architecture overview (`Fig.~\\ref{fig:arch}`)**
  - **Where used**: Methodology (Overview of DynaTree).
  - **What to draw**: One decoding iteration as a pipeline:
    - Prefix + target KV cache
    - Draft tree expansion (top-\(B\), depth \(D\))
    - Dynamic pruning (threshold \(\tau\) + node budget \(N_{\\max}\))
    - BFS flattening
    - Tree attention mask + single target forward pass
    - Longest valid path selection + bonus token
    - Cache crop + rebuild on committed tokens
  - **Recommended format**: Vector (`.pdf`). Draw.io / Figma / Inkscape / TikZ are all fine.

- **Figure: Tree attention mask illustration (`Fig.~\\ref{fig:tree-attn}`)**
  - **Where used**: Methodology (Tree Attention for Parallel Verification).
  - **What to draw**:
    - A small example tree (depth 2--3, branching 2--3)
    - BFS flattening order (index each node)
    - The induced attention mask as a grid (or a schematic showing ``attend-to ancestors + prefix only'')
  - **Recommended format**: Vector (`.pdf`). If grid is too dense, show only the pattern (highlighted ancestor blocks).

### Figures (likely needed for Experiments section)

- **Figure: Parameter sweep / sensitivity (planned Figure 2)**
  - **Data source**: `results/tree_param_search_20251231_140952.json`
  - **What to include (suggested 6 subplots)**:
    - Speedup vs depth \(D\) (fix \(B\), \(\tau\))
    - Speedup vs branch factor \(B\) (fix \(D\), \(\tau\))
    - Speedup vs threshold \(\tau\) (fix \(D\), \(B\))
    - Speedup vs generation length
    - Average tree size (#nodes) heatmap over \((D,B)\) or \((D,\tau)\)
    - Acceptance / tokens-per-round distribution (optional)
  - **Output**: `papers/figures/param_sweep.pdf`
  - **Script hint**: there is `spec_decode/plot_paper_figures.py` (may be adapted for publication-quality plots).

### Tables (main paper)

- **Table: Experimental setup**
  - Target / draft model names and sizes
  - Hardware (GPU type), software stack, precision
  - Prompt set and generation lengths

- **Table: Main throughput / speedup results**
  - Baseline AR vs HF assisted vs Linear speculative vs DynaTree (Tree V2)
  - Metrics: throughput (tokens/s), speedup, acceptance (optional), tokens/round (optional)
  - Use numbers from `papers/Tree_Speculative_Decoding_实验报告.md` (Section 5.2).

- **Table: Ablation / component study**
  - Goal: isolate the value of tree attention / pruning / tree vs linear.
  - If you want a minimal ablation consistent with current code and report, use:
    - Linear speculative (best K)
    - Tree (no probability pruning; same \(D,B,N_{\\max}\))
    - Tree V2 (probability pruning; same \(D,B,N_{\\max},\\tau\))
  - Data source: either existing JSON sweep or a small rerun.

### Optional (appendix / supplementary)

- **Algorithm listing**: currently included as a portability-friendly pseudocode figure (`Fig.~\\ref{fig:algo}`). If space is tight, move it to appendix or supplement.
- **Case study visualization**: one prompt with the drafted tree, pruned nodes, and chosen path (good for intuition, optional).


