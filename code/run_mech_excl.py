#!/usr/bin/env python3
"""Exclusion control: random selection that never picks benign samples carrying >=2 trigger values.
Attacked run only; compare against random / uncertainty / clean baselines from the stored runs."""
import argparse, sys, time, io, contextlib
import numpy as np, pandas as pd
import attack_smart_mech as M

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="./data")
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--family", default="wacatac")
ap.add_argument("--out-csv", default="./figs/mech_excl.csv")
a = ap.parse_args()

import lightgbm as lgb
X, y, meta, period = M.load_bodmas(a.data_dir)
months = list(period.unique()); tm = 1; N = len(months) - tm
TOTAL = len(a.seeds) * N
st = dict(done=0, t0=time.time(), label="")
def bar():
    f = st["done"] / TOTAL; el = time.time() - st["t0"]
    eta = (el / f - el) if f > 0.02 else float("nan")
    sys.stdout.write(f"\r[{'#'*int(30*f)}{'.'*(30-int(30*f))}] {100*f:5.1f}%  {st['label']:<18} elapsed {int(el//60):02d}:{int(el%60):02d}  ETA {'--:--' if eta!=eta else '%02d:%02d'%(eta//60,eta%60)}   ")
    sys.stdout.flush()
def tick(): st["done"] += 1; bar()
M.PROGRESS = tick

out = []
for seed in a.seeds:
    st["label"] = f"seed {seed}"; bar()
    idx = np.where(period.isin(months[:tm]).values)[0]
    clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, n_jobs=-1, verbose=-1, random_state=seed)
    clf.fit(X[idx], y[idx])
    with contextlib.redirect_stdout(io.StringIO()):
        trig = M.build_trigger(clf, X[idx], y[idx], 8, constrained=True, seed=seed)
    df, _ = M.run("random_excl_trig", True, X, y, meta, period, months, tm, 0.05, 0.01, 4, a.family, trig, seed=seed)
    df.insert(0, "arm", "random_excl_trig"); df.insert(0, "seed", seed)
    out.append(df); pd.concat(out).to_csv(a.out_csv, index=False)
print("\n[DONE] ->", a.out_csv, flush=True)
