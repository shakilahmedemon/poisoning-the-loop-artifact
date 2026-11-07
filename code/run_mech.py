#!/usr/bin/env python3
"""Mechanism measurement: log what each selection policy picks, per month.
Uses attack_smart_mech.py (copy of attack_smart.py + selection logging)."""
import argparse, os, time
import numpy as np, pandas as pd
from attack_smart_mech import load_bodmas, build_trigger, run

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="./data")
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--families", nargs="+", default=["wacatac"])
ap.add_argument("--strategies", nargs="+", default=["random", "uncertainty"])
ap.add_argument("--out-csv", default="./figs/mech_selection.csv")
a = ap.parse_args()

import lightgbm as lgb
X, y, meta, period = load_bodmas(a.data_dir)
months = list(period.unique())
tm = 1
out = []
for fam in a.families:
    for strat in a.strategies:
        for seed in a.seeds:
            t0 = time.time()
            idx = np.where(period.isin(months[:tm]).values)[0]
            clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08,
                                     n_jobs=-1, verbose=-1, random_state=seed)
            clf.fit(X[idx], y[idx])
            trig = build_trigger(clf, X[idx], y[idx], 8, constrained=True, seed=seed)
            for attack in (False, True):
                df, _ = run(strat, attack, X, y, meta, period, months, tm, 0.05, 0.01, 4,
                            fam, trig, seed=seed)
                df.insert(0, "attack", attack); df.insert(0, "seed", seed)
                df.insert(0, "strategy", strat); df.insert(0, "family", fam)
                out.append(df)
            print(f"[OK] {fam} {strat} seed={seed} [{time.time()-t0:.0f}s]", flush=True)
            pd.concat(out).to_csv(a.out_csv, index=False)
print("[DONE]")
