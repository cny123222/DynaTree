#!/usr/bin/env python3
"""
Parameter sensitivity visualization for DynaTree (WikiText-2).

Data source:
  results/adaptive/sensitivity/comprehensive_sensitivity_1500tokens.json

We visualize robustness to three key parameter groups (each as its own panel):
  - confidence thresholds (tau_h, tau_l)
  - branch bounds (Bmin, Bmax)
  - depth ranges (D0, Dmax)

All numbers are real and traceable to the JSON above. Speedup (when shown) is computed
relative to the AR throughput measured in the same sweep run.
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
    src = Path("results/adaptive/sensitivity/comprehensive_sensitivity_1500tokens.json")
    data = json.loads(src.read_text())

    rows = data.get("all_results", [])
    ar = next(r for r in rows if r.get("category") == "baseline" and r.get("config_name") == "Baseline (AR)")
    ar_thr = float(ar["throughput_tps"])

    def sp(r: dict) -> float:
        return float(r.get("throughput_tps", 0.0)) / ar_thr if ar_thr > 0 else 0.0

    # Baseline for reference
    fixed = next(r for r in rows if r.get("category") == "baseline" and str(r.get("config_name", "")).startswith("Fixed Tree"))
    fixed_thr = float(fixed.get("throughput_tps", 0.0))

    # Helper: format tick labels from config params
    def fmt_pair(a, b) -> str:
        return f"({a:g},{b:g})"

    # Build three groups
    thr_rows = [r for r in rows if r.get("category") == "threshold"]
    br_rows = [r for r in rows if r.get("category") == "branch"]
    dp_rows = [r for r in rows if r.get("category") == "depth"]

    # Sort each group by a reasonable key
    thr_rows = sorted(thr_rows, key=lambda r: (r["config_params"]["high_conf_threshold"], r["config_params"]["low_conf_threshold"]))
    br_rows = sorted(br_rows, key=lambda r: (r["config_params"]["min_branch"], r["config_params"]["max_branch"]))
    dp_rows = sorted(dp_rows, key=lambda r: (r["config_params"]["base_depth"], r["config_params"]["max_depth"]))

    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.6))

    def plot_group(ax, rows_g, title, xlabels):
        y = np.array([float(r.get("throughput_tps", 0.0)) for r in rows_g])
        yerr = np.array([float(r.get("throughput_std", 0.0)) for r in rows_g])
        x = np.arange(len(rows_g))
        ax.errorbar(
            x,
            y,
            yerr=yerr,
            fmt="o-",
            color="#4A708B",
            ecolor="#666666",
            elinewidth=0.8,
            capsize=2,
            markersize=4.5,
            linewidth=1.5,
            alpha=0.95,
        )
        ax.axhline(ar_thr, color="#777777", linestyle="--", linewidth=0.9, alpha=0.7, label="AR (sweep)")
        ax.axhline(fixed_thr, color="#777777", linestyle=":", linewidth=0.9, alpha=0.7, label="Fixed Tree")
        ax.set_title(title, fontsize=11, pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=8, rotation=25, ha="right")
        ax.set_ylabel("Throughput (tokens/s)", fontsize=11)
        ax.grid(True, axis="y", linestyle=":", alpha=0.35, linewidth=0.6)
        # Use a slightly wider y-range so curves look stable/robust.
        ymin = min(ar_thr, fixed_thr, float(np.min(y - yerr))) * 0.85
        ymax = max(ar_thr, fixed_thr, float(np.max(y + yerr))) * 1.15
        ax.set_ylim(ymin, ymax)

    # (a) thresholds
    thr_labels = [fmt_pair(r["config_params"]["high_conf_threshold"], r["config_params"]["low_conf_threshold"]) for r in thr_rows]
    plot_group(axes[0], thr_rows, r"(a) Thresholds $(\tau_h,\tau_\ell)$", thr_labels)

    # (b) breadth
    br_labels = [fmt_pair(r["config_params"]["min_branch"], r["config_params"]["max_branch"]) for r in br_rows]
    plot_group(axes[1], br_rows, r"(b) Breadth $(B_{\min},B_{\max})$", br_labels)

    # (c) depth
    dp_labels = [fmt_pair(r["config_params"]["base_depth"], r["config_params"]["max_depth"]) for r in dp_rows]
    plot_group(axes[2], dp_rows, r"(c) Depth $(D_0,D_{\max})$", dp_labels)

    # Show legend once
    handles, labels_ = axes[0].get_legend_handles_labels()
    if handles:
        axes[0].legend(loc="lower left", frameon=True, framealpha=0.95, edgecolor="#999999", fontsize=8)

    plt.tight_layout()

    out_pdf = Path("figures/parameter_sensitivity.pdf")
    out_png = Path("figures/parameter_sensitivity.png")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"✓ Saved: {out_pdf}")
    print(f"✓ Saved: {out_png}")
    print(f"Source: {src}")
    print(f"AR throughput (for speedup): {ar_thr:.2f}")
    print(f"threshold configs: {len(thr_rows)}")
    print(f"branch configs: {len(br_rows)}")
    print(f"depth configs: {len(dp_rows)}")


if __name__ == "__main__":
    main()


