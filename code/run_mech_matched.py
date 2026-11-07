#!/usr/bin/env python3
"""Matched-count control: does forcing RANDOM selection to pick as many target-family
(or as many malware overall) samples as UNCERTAINTY does reproduce uncertainty's recovery?
Attacked runs only, seed given. Reads the uncertainty schedule from figs/mech_selection.csv."""
import argparse, time
import numpy as np, pandas as pd
from attack_smart_mech import load_bodmas, build_trigger, run

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="./data")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--family", default="wacatac")
ap.add_argument("--ref-csv", default="./figs/mech_selection.csv")
ap.add_argument("--out-csv", default="./figs/mech_matched.csv")
a = ap.parse_args()

ref = pd.read_csv(a.ref_csv)
ref = ref[(ref.family == a.family) & (ref.strategy == "uncertainty") & (ref.attack == True) & (ref.seed == a.seed)]
sched = {"random_matched_fam": dict(zip(ref.month.astype(str), ref.sel_fam_n)),
         "random_matched_mal": dict(zip(ref.month.astype(str), ref.sel_mal_n))}

import lightgbm as lgb
X, y, meta, period = load_bodmas(a.data_dir)
months = list(period.unique()); tm = 1
idx = np.where(period.isin(months[:tm]).values)[0]
clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, n_jobs=-1,
                         verbose=-1, random_state=a.seed)
clf.fit(X[idx], y[idx])
trig = build_trigger(clf, X[idx], y[idx], 8, constrained=True, seed=a.seed)
out = []
for strat in ("random_matched_fam", "random_matched_mal"):
    t0 = time.time()
    df, _ = run(strat, True, X, y, meta, period, months, tm, 0.05, 0.01, 4, a.family, trig,
                seed=a.seed, schedule=sched)
    df.insert(0, "strategy", strat); out.append(df)
    print(f"[OK] {strat} [{time.time()-t0:.0f}s]", flush=True)
    pd.concat(out).to_csv(a.out_csv, index=False)
print("[DONE]")
