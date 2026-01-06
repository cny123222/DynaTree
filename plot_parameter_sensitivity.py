#!/usr/bin/env python3
"""
Parameter sensitivity visualization for DynaTree (WikiText-2).

Data source:
  results/adaptive/sensitivity/paper_benchmark_sensitivity_v2.json

We visualize the effect of:
  - confidence thresholds (tau_h, tau_l)
  - branch bounds (Bmin, Bmax)

All numbers are real and traceable to the JSON above.
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
    src = Path("results/adaptive/sensitivity/paper_benchmark_sensitivity_v2.json")
    data = json.loads(src.read_text())

    ar = next(r for r in data["all_results"] if r.get("method") == "Baseline (AR)")
    ar_thr = float(ar["throughput_tps"])

    phase3 = [r for r in data["all_results"] if r.get("method") == "Phase 3: + History Adjust"]
    # Normalize configs into a table
    rows = []
    for r in phase3:
        cfg = r.get("config") or {}
        rows.append(
            {
                "tau_h": float(cfg.get("high_conf_threshold")),
                "tau_l": float(cfg.get("low_conf_threshold")),
                "bmin": int(cfg.get("min_branch")),
                "bmax": int(cfg.get("max_branch")),
                "thr": float(r.get("throughput_tps", 0.0)),
                "thr_std": float(r.get("throughput_std", 0.0)),
                "acc": float(r.get("acceptance_rate", 0.0)) * 100.0,
            }
        )

    tau_pairs = sorted({(r["tau_h"], r["tau_l"]) for r in rows})
    bmax_vals = sorted({r["bmax"] for r in rows})

    # Build lookups (prefer Bmin=1 for the main sensitivity curves)
    def lookup(tau_h: float, tau_l: float, bmax: int, bmin: int = 1):
        for r in rows:
            if r["tau_h"] == tau_h and r["tau_l"] == tau_l and r["bmax"] == bmax and r["bmin"] == bmin:
                return r
        return None

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))

    # Panel (a): threshold impact (x=tau pair, lines=Bmax)
    ax = axes[0]
    x = np.arange(len(tau_pairs))
    xlabels = [f"({th:.1f},{tl:.1f})" for th, tl in tau_pairs]
    colors = {2: "#4A708B", 3: "#8BACC6", 4: "#D97757"}  # highlight bmax=4
    markers = {2: "o", 3: "s", 4: "D"}
    for bmax in bmax_vals:
        ys = []
        for th, tl in tau_pairs:
            r = lookup(th, tl, bmax, bmin=1)
            ys.append(r["thr"] if r else np.nan)
        ax.plot(
            x,
            ys,
            marker=markers.get(bmax, "o"),
            linewidth=2.0,
            markersize=6,
            color=colors.get(bmax, "#666666"),
            alpha=0.95,
            label=rf"$B_{{\max}}={bmax}$",
        )
    ax.axhline(y=ar_thr, color="#777777", linestyle="--", linewidth=0.9, alpha=0.6, label="AR")
    ax.set_title(r"(a) Impact of confidence thresholds", fontsize=11, pad=8)
    ax.set_xlabel(r"$(\tau_h,\tau_\ell)$", fontsize=11)
    ax.set_ylabel("Throughput (tokens/s)", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(xlabels, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.35, linewidth=0.6)
    ax.legend(loc="best", frameon=True, framealpha=0.95, edgecolor="#999999", fontsize=9)

    # Panel (b): branch-range impact (x=Bmax, lines=tau pair)
    ax = axes[1]
    x2 = np.arange(len(bmax_vals))
    x2labels = [f"{b}" for b in bmax_vals]
    tau_colors = {
        tau_pairs[0]: "#4A708B",
        tau_pairs[min(1, len(tau_pairs) - 1)]: "#8BACC6",
        tau_pairs[min(2, len(tau_pairs) - 1)]: "#D97757",
    }
    tau_markers = {tau_pairs[0]: "o", tau_pairs[min(1, len(tau_pairs) - 1)]: "s", tau_pairs[min(2, len(tau_pairs) - 1)]: "D"}
    for th, tl in tau_pairs:
        ys = []
        for bmax in bmax_vals:
            r = lookup(th, tl, bmax, bmin=1)
            ys.append(r["thr"] if r else np.nan)
        ax.plot(
            x2,
            ys,
            marker=tau_markers.get((th, tl), "o"),
            linewidth=2.0,
            markersize=6,
            color=tau_colors.get((th, tl), "#666666"),
            alpha=0.95,
            label=f"({th:.1f},{tl:.1f})",
        )
    ax.axhline(y=ar_thr, color="#777777", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_title(r"(b) Impact of branch upper bound", fontsize=11, pad=8)
    ax.set_xlabel(r"$B_{\max}$ (with $B_{\min}=1$)", fontsize=11)
    ax.set_ylabel("Throughput (tokens/s)", fontsize=11)
    ax.set_xticks(x2)
    ax.set_xticklabels(x2labels, fontsize=9)
    ax.grid(True, linestyle=":", alpha=0.35, linewidth=0.6)
    ax.legend(loc="best", frameon=True, framealpha=0.95, edgecolor="#999999", fontsize=9)

    # Panel (c): throughput vs acceptance (scatter)
    ax = axes[2]
    for r in rows:
        label = None
        if r["bmin"] == 2:
            label = r"$B_{\min}=2$ (stress case)"
        ax.scatter(
            r["acc"],
            r["thr"],
            s=90,
            marker="^" if r["bmin"] == 2 else "o",
            color=colors.get(r["bmax"], "#666666"),
            edgecolor="#333333",
            linewidth=0.6,
            alpha=0.9,
            label=label,
        )
    # Highlight best
    best = max(rows, key=lambda z: z["thr"])
    ax.scatter(best["acc"], best["thr"], s=140, marker="*", color="#D97757", edgecolor="#333333", linewidth=0.6, zorder=5)
    ax.axhline(y=ar_thr, color="#777777", linestyle="--", linewidth=0.9, alpha=0.6)
    ax.set_title(r"(c) Throughput–acceptance tradeoff", fontsize=11, pad=8)
    ax.set_xlabel("Accept. (%)", fontsize=11)
    ax.set_ylabel("Throughput (tokens/s)", fontsize=11)
    ax.grid(True, linestyle=":", alpha=0.35, linewidth=0.6)
    # De-duplicate legend entries
    handles, labels_ = ax.get_legend_handles_labels()
    uniq = {}
    for h, l in zip(handles, labels_):
        if l and l not in uniq:
            uniq[l] = h
    if uniq:
        ax.legend(list(uniq.values()), list(uniq.keys()), loc="lower right", frameon=True, framealpha=0.95, edgecolor="#999999", fontsize=9)

    plt.tight_layout()

    out_pdf = Path("figures/parameter_sensitivity.pdf")
    out_png = Path("figures/parameter_sensitivity.png")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved: {out_pdf}")
    print(f"✓ Saved: {out_png}")
    print(f"Source: {src}")
    print("tau pairs:", tau_pairs)
    print("Bmax values:", bmax_vals)


if __name__ == "__main__":
    main()


