#!/usr/bin/env python3
"""
Breadth check: does the constrained clean-label backdoor generalize across
target families, or does it only work on wacatac/sfone because they were
hand-picked?

Families below were chosen by RANK in the injection-window volume
distribution (roughly log-spaced from rank 1 to rank 86, the full set of
families with >=15 samples in Sep-Dec 2019), not by which ones were
expected to work -- see the conversation for the ranked list this was
drawn from. wacatac (rank 1) and sfone (rank 20) were already evaluated
in earlier experiments and are not rerun here.

Fixed config, matching every earlier constrained-attack run: train_months=1,
strategy=random, label_rate=0.05, injection_rate=0.01, injection_months=4,
trigger_size=8, constrained=True, seed=0 (single seed -- this is a breadth
check across conditions, not a per-family significance claim; the earlier
5-seed runs already established that variance at fixed family/strategy).

Usage:
    python run_family_sweep.py --data-dir ./data
"""

import argparse
import csv
import os
import time

import numpy as np
import pandas as pd

from attack_smart import load_bodmas, build_trigger, run, aut


FAMILIES = ["sillyp2p", "zbot", "vflooder", "simda", "plite", "simbot"]


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
    ap.add_argument("--train-months", type=int, default=1)
    ap.add_argument("--label-rate", type=float, default=0.05)
    ap.add_argument("--injection-rate", type=float, default=0.01)
    ap.add_argument("--injection-months", type=int, default=4)
    ap.add_argument("--trigger-size", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-csv", default="./figs/results_family_sweep.csv")
    args = ap.parse_args()

    import lightgbm as lgb

    print("[..] loading BODMAS once for the whole sweep")
    X, y, meta, period = load_bodmas(args.data_dir)
    months = list(period.unique())

    train_mask = period.isin(months[:args.train_months]).values
    train_idx = np.where(train_mask)[0]

    os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
    fieldnames = ["family", "n_in_window", "depth", "horizon", "f1_gap",
                 "real_fam_gap", "wall_s"]
    write_header = not os.path.exists(args.out_csv)
    csv_f = open(args.out_csv, "a", newline="")
    writer = csv.DictWriter(csv_f, fieldnames=fieldnames)
    if write_header:
        writer.writeheader()
        csv_f.flush()

    # Trigger construction doesn't depend on the target family (it's built
    # from the seed model + benign pool only) -- build it ONCE, reuse for
    # every family, saving 6x the SHAP computation.
    seed_clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63,
                                  learning_rate=0.08, n_jobs=-1, verbose=-1,
                                  random_state=args.seed)
    seed_clf.fit(X[train_idx], y[train_idx])
    trigger = build_trigger(seed_clf, X[train_idx], y[train_idx],
                            args.trigger_size, constrained=True, seed=args.seed)

    t_start = time.time()
    for i, family in enumerate(FAMILIES):
        t0 = time.time()
        n_window = int((meta.loc[(~period.isin(months[:args.train_months])).values
                                 & (y == 1), "family"] == family).sum())
        print(f"\n[{i+1}/{len(FAMILIES)}] family={family} (n_in_window~{n_window})  "
             f"elapsed={time.time()-t_start:.0f}s")

        df_clean, _ = run("random", False, X, y, meta, period, months,
                          args.train_months, args.label_rate,
                          args.injection_rate, args.injection_months,
                          family, trigger, seed=args.seed)
        df_attack, n_inj = run("random", True, X, y, meta, period, months,
                               args.train_months, args.label_rate,
                               args.injection_rate, args.injection_months,
                               family, trigger, seed=args.seed)

        depth, horizon, f1_gap, real_fam_gap = evaluate(
            df_clean, df_attack, months, args.train_months, args.injection_months)
        wall = time.time() - t0
        print(f"[OK] depth={depth:.4f}  horizon={horizon}  f1_gap={f1_gap:.4f}  "
             f"real_fam_gap={real_fam_gap:.4f}  n_injected={n_inj}  [{wall:.0f}s]")

        writer.writerow(dict(family=family, n_in_window=n_window, depth=depth,
                             horizon=horizon, f1_gap=f1_gap,
                             real_fam_gap=real_fam_gap, wall_s=wall))
        csv_f.flush()

    csv_f.close()
    print(f"\n[OK] sweep done in {time.time()-t_start:.0f}s total")

    df = pd.read_csv(args.out_csv)
    print("\n" + "=" * 80)
    print(f"{'family':>12} {'n_window':>9} {'depth':>8} {'horizon':>8} "
         f"{'f1_gap':>9} {'real_fam_gap':>13}")
    print("-" * 80)
    for _, row in df.sort_values("n_in_window", ascending=False).iterrows():
        print(f"{row['family']:>12} {row['n_in_window']:>9.0f} {row['depth']:>8.4f} "
             f"{row['horizon']:>8.0f} {row['f1_gap']:>9.4f} {row['real_fam_gap']:>13.4f}")
    print("=" * 80)


if __name__ == "__main__":
    main()
