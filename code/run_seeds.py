#!/usr/bin/env python3
"""
Multi-seed driver for the constrained smart attack (attack_smart.py).

Loads BODMAS once and reuses it across all (strategy, family, seed) cells
instead of shelling out to attack_smart.py N times, which would reload the
250MB .npz and re-import shap/lightgbm on every call. Writes results
incrementally to results_seeds.csv so a crash partway through doesn't lose
completed runs, and prints a final mean +/- std summary table per cell.

Fixed config, matching the single-seed runs already reported: train_months=1,
label_rate=0.05, injection_rate=0.01, injection_months=4, trigger_size=8,
constrained=True (realizable trigger only).

Usage:
    python run_seeds.py --data-dir ./data --seeds 0 1 2
"""

import argparse
import csv
import os
import time

import numpy as np
import pandas as pd

from attack_smart import load_bodmas, build_trigger, run, aut


def evaluate(df_clean, df_attack, months, train_months, injection_months):
    injection_window = set(months[train_months:train_months + injection_months])
    post_injection = [i for i, m in enumerate(df_clean["month"])
                      if pd.Period(m, freq="M") not in injection_window]
    depth, horizon = 0.0, 0
    for i in post_injection:
        c = df_clean["family_recall_watermarked"].iloc[i]
        a = df_attack["family_recall_watermarked"].iloc[i]
        if pd.isna(c) or pd.isna(a):
            continue
        gap = c - a
        depth = max(depth, gap)
        if gap > 0.15:
            horizon += 1
        else:
            break
    f1_gap = abs(aut(df_clean["f1"].values) - aut(df_attack["f1"].values))
    real_fam_gap = abs(aut(df_clean["family_recall_clean"].values) -
                       aut(df_attack["family_recall_clean"].values))
    return depth, horizon, f1_gap, real_fam_gap


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--train-months", type=int, default=1)
    ap.add_argument("--label-rate", type=float, default=0.05)
    ap.add_argument("--injection-rate", type=float, default=0.01)
    ap.add_argument("--injection-months", type=int, default=4)
    ap.add_argument("--trigger-size", type=int, default=8)
    ap.add_argument("--out-csv", default="./figs/results_seeds.csv")
    args = ap.parse_args()

    import lightgbm as lgb

    print("[..] loading BODMAS once for the whole sweep")
    X, y, meta, period = load_bodmas(args.data_dir)
    months = list(period.unique())

    cells = [("random", "wacatac"), ("uncertainty", "wacatac"),
            ("random", "sfone"), ("uncertainty", "sfone")]

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fieldnames = ["strategy", "family", "seed", "depth", "horizon",
                 "f1_gap", "real_fam_gap", "wall_s"]
    write_header = not os.path.exists(args.out_csv)
    csv_f = open(args.out_csv, "a", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        csv_f.flush()

    total = len(cells) * len(args.seeds)
    done = 0
    t_start = time.time()

    for strategy, family in cells:
        for seed in args.seeds:
            done += 1
            t0 = time.time()
            print(f"\n[{done}/{total}] strategy={strategy} family={family} "
                  f"seed={seed}  (elapsed so far: {time.time()-t_start:.0f}s)")

            train_mask = period.isin(months[:args.train_months]).values
            train_idx = np.where(train_mask)[0]
            seed_clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63,
                                          learning_rate=0.08, n_jobs=-1,
                                          verbose=-1, random_state=seed)
            seed_clf.fit(X[train_idx], y[train_idx])
            trigger = build_trigger(seed_clf, X[train_idx], y[train_idx],
                                    args.trigger_size, constrained=True, seed=seed)

            df_clean, _ = run(strategy, False, X, y, meta, period, months,
                              args.train_months, args.label_rate,
                              args.injection_rate, args.injection_months,
                              family, trigger, seed=seed)
            df_attack, n_inj = run(strategy, True, X, y, meta, period, months,
                                   args.train_months, args.label_rate,
                                   args.injection_rate, args.injection_months,
                                   family, trigger, seed=seed)

            depth, horizon, f1_gap, real_fam_gap = evaluate(
                df_clean, df_attack, months, args.train_months, args.injection_months)
            wall = time.time() - t0
            print(f"[OK] depth={depth:.4f}  horizon={horizon}  "
                 f"f1_gap={f1_gap:.4f}  real_fam_gap={real_fam_gap:.4f}  "
                 f"n_injected={n_inj}  [{wall:.0f}s]")

            writer.writerow(dict(strategy=strategy, family=family, seed=seed,
                                 depth=depth, horizon=horizon, f1_gap=f1_gap,
                                 real_fam_gap=real_fam_gap, wall_s=wall))
            csv_f.flush()

    csv_f.close()
    print(f"\n[OK] all {total} runs done in {time.time()-t_start:.0f}s total")

    # ---- summary table: mean +/- std per cell ----
    df = pd.read_csv(args.out_csv)
    print("\n" + "=" * 92)
    print(f"{'strategy':>12} {'family':>8} | {'depth':>16} {'horizon':>14} "
         f"{'f1_gap':>16} {'real_fam_gap':>16}")
    print("-" * 92)
    for (strategy, family), g in df.groupby(["strategy", "family"], sort=False):
        def fmt(col):
            return f"{g[col].mean():.4f}+/-{g[col].std():.4f}"
        print(f"{strategy:>12} {family:>8} | {fmt('depth'):>16} "
             f"{g['horizon'].mean():.2f}+/-{g['horizon'].std():.2f}      "
             f"{fmt('f1_gap'):>16} {fmt('real_fam_gap'):>16}")
    print("=" * 92)


if __name__ == "__main__":
    main()
