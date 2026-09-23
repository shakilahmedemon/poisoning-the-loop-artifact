#!/usr/bin/env python3
"""
Bar-chart view of the family sweep (RQ3): evasion depth and persistence
horizon per family, 5-seed mean +/- std, ordered by injection-window
volume. Reads results_family_sweep.csv (seed 0) and
results_family_sweep_seeds.csv (seeds 1-4, from run_volume_and_seeds.py
--part seeds) and combines them, same as analyze_all.py's section 2.

Usage:
    python plot_family_sweep.py [--figs-dir ./figs] [--out ./figs/fig_family_sweep_bars.pdf]
"""

import argparse
import os

import numpy as np
import pandas as pd

ORDER = ["simbot", "plite", "simda", "vflooder", "zbot", "sillyp2p"]
N_WINDOW = {"simbot": 27, "plite": 70, "simda": 80, "vflooder": 435,
            "zbot": 468, "sillyp2p": 1598}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs-dir", default="./figs")
    ap.add_argument("--out", default="./figs/fig_family_sweep_bars.pdf")
    args = ap.parse_args()

    s0 = pd.read_csv(os.path.join(args.figs_dir, "results_family_sweep.csv")
                      )[["family", "depth", "horizon"]].assign(seed=0)
    seeds = pd.read_csv(os.path.join(args.figs_dir, "results_family_sweep_seeds.csv")
                         )[["family", "depth", "horizon", "seed"]]
    d = pd.concat([s0, seeds])

    depth_mean = [d[d.family == f].depth.mean() for f in ORDER]
    depth_std = [d[d.family == f].depth.std(ddof=1) for f in ORDER]
    hor_mean = [d[d.family == f].horizon.mean() for f in ORDER]
    hor_std = [d[d.family == f].horizon.std(ddof=1) for f in ORDER]
    labels = [f"{f}\n(n={N_WINDOW[f]})" for f in ORDER]

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.linewidth": 0.6, "figure.dpi": 200})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.3))
    x = np.arange(len(ORDER))

    ax = axes[0]
    ax.bar(x, depth_mean, yerr=depth_std, capsize=2.5, color="#D55E00",
           edgecolor="black", linewidth=0.5, error_kw=dict(elinewidth=0.8, capthick=0.8))
    ax.set_ylabel("evasion depth")
    ax.set_ylim(0, 1.15)
    ax.grid(alpha=0.25, lw=0.4, axis="y")
    ax.set_title("evasion depth by family, ordered by volume", fontsize=7)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6, rotation=0)

    ax = axes[1]
    ax.bar(x, hor_mean, yerr=hor_std, capsize=2.5, color="#0072B2",
           edgecolor="black", linewidth=0.5, error_kw=dict(elinewidth=0.8, capthick=0.8))
    ax.set_ylabel("persistence horizon (months)")
    ax.set_ylim(0, 7.2)
    ax.grid(alpha=0.25, lw=0.4, axis="y")
    ax.set_title("persistence horizon by family", fontsize=7)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6, rotation=0)

    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print(f"[OK] wrote {args.out}")


if __name__ == "__main__":
    main()
