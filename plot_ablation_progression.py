#!/usr/bin/env python3
"""
Plot progressive ablation (Fixed Tree -> + Dynamic Breadth & Depth -> + History Adaptation)
using the paper's main benchmark JSON (WikiText-2, T=1500).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Academic paper style settings (consistent with other plotting scripts)
plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman", "DejaVu Serif"]
plt.rcParams["mathtext.fontset"] = "dejavuserif"
plt.rcParams["axes.edgecolor"] = "#999999"
plt.rcParams["axes.linewidth"] = 0.8
plt.rcParams["axes.labelcolor"] = "#333333"
plt.rcParams["xtick.color"] = "#333333"
plt.rcParams["ytick.color"] = "#333333"
plt.rcParams["font.size"] = 10


def main() -> None:
    src = Path("results/adaptive/main/paper_benchmark_main_1500tokens_2.json")
    data = json.loads(src.read_text())
    m = {r["method"]: r for r in data["all_results"]}

    fixed = m["Fixed Tree (D=5, B=2)"]
    phase2 = m["Phase 2: + Dynamic Depth"]  # corresponds to "+ Dynamic Breadth & Depth"
    phase3 = m["Phase 3: + History Adjust"]  # corresponds to "+ History Adaptation" (DynaTree)

    labels = ["Fixed Tree", "+ Breadth&Depth", "+ History"]
    throughput = [
        fixed["throughput_tps"],
        phase2["throughput_tps"],
        phase3["throughput_tps"],
    ]
    rounds = [
        fixed["total_rounds"],
        phase2["total_rounds"],
        phase3["total_rounds"],
    ]

    # Optional: tokens per iteration (already a field), plotted as secondary bars if needed later.
    tokens_per_iter = [
        fixed.get("tokens_per_round", 0.0),
        phase2.get("tokens_per_round", 0.0),
        phase3.get("tokens_per_round", 0.0),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))

    colors = ["#6FAF8A", "#8BACC6", "#D97757"]  # fixed, phase2, phase3 (highlight)
    edge = "#333333"

    # (a) Throughput
    ax = axes[0]
    x = np.arange(len(labels))
    bars = ax.bar(x, throughput, color=colors, edgecolor=edge, linewidth=0.6, alpha=0.92)
    ax.set_title("(a) Throughput", fontsize=11, pad=8)
    ax.set_ylabel("Tokens / second", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.35, linewidth=0.6)
    for b, v in zip(bars, throughput):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + max(throughput) * 0.01,
            f"{v:.1f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    # (b) Number of verification iterations
    ax = axes[1]
    bars = ax.bar(x, rounds, color=colors, edgecolor=edge, linewidth=0.6, alpha=0.92)
    ax.set_title("(b) Verification iterations", fontsize=11, pad=8)
    ax.set_ylabel("#Iter.", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.35, linewidth=0.6)
    for b, v in zip(bars, rounds):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + max(rounds) * 0.01,
            f"{int(v)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()

    out_pdf = Path("figures/ablation_progression.pdf")
    out_png = Path("figures/ablation_progression.png")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved: {out_pdf}")
    print(f"✓ Saved: {out_png}")
    print(f"Source: {src}")
    print("Values:")
    for name, thr, r, tpi in zip(labels, throughput, rounds, tokens_per_iter):
        print(f"  {name:16s} thr={thr:.2f}  iters={int(r)}  tokens/iter={tpi:.2f}")


if __name__ == "__main__":
    main()


