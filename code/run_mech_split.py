#!/usr/bin/env python3
"""Split-uncertainty control: uncertainty on malware only vs benign only (other class random).
Malware count per month matches full uncertainty. Attacked runs; seed given."""
import argparse, sys, time, io, contextlib
import numpy as np, pandas as pd
import attack_smart_mech as M

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="./data")
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--family", default="wacatac")
ap.add_argument("--ref-csv", default="./figs/mech_selection.csv")
ap.add_argument("--out-csv", default="./figs/mech_split.csv")
a = ap.parse_args()

import lightgbm as lgb
ref_all = pd.read_csv(a.ref_csv)
X, y, meta, period = M.load_bodmas(a.data_dir)
months = list(period.unique()); tm = 1; N = len(months) - tm
ARMS = ("unc_mal_only", "unc_ben_only")
TOTAL = len(a.seeds) * len(ARMS) * N
st = dict(done=0, t0=time.time(), label="")
def bar():
    f = st["done"] / TOTAL; el = time.time() - st["t0"]
    eta = (el / f - el) if f > 0.02 else float("nan")
    sys.stdout.write(f"\r[{'#'*int(30*f)}{'.'*(30-int(30*f))}] {100*f:5.1f}%  {st['label']:<22} elapsed {int(el//60):02d}:{int(el%60):02d}  ETA {'--:--' if eta!=eta else '%02d:%02d'%(eta//60,eta%60)}   ")
    sys.stdout.flush()
def tick(): st["done"] += 1; bar()
M.PROGRESS = tick

out = []
for seed in a.seeds:
    if seed == 0:
        ref = ref_all[(ref_all.family == a.family) & (ref_all.strategy == "uncertainty") & (ref_all.attack == True) & (ref_all.seed == seed)]
    else:
        mm_ = pd.read_csv("./figs/mech_multi.csv")
        ref = mm_[(mm_.arm == "uncertainty") & (mm_.seed == seed)]
    sched = {arm: dict(zip(ref.month.astype(str), ref.sel_mal_n)) for arm in ARMS}
    idx = np.where(period.isin(months[:tm]).values)[0]
    clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, n_jobs=-1, verbose=-1, random_state=seed)
    clf.fit(X[idx], y[idx])
    with contextlib.redirect_stdout(io.StringIO()):
        trig = M.build_trigger(clf, X[idx], y[idx], 8, constrained=True, seed=seed)
    for arm in ARMS:
        st["label"] = f"seed {seed}: {arm}"; bar()
        df, _ = M.run(arm, True, X, y, meta, period, months, tm, 0.05, 0.01, 4, a.family, trig, seed=seed, schedule=sched)
        df.insert(0, "arm", arm); df.insert(0, "seed", seed)
        out.append(df); pd.concat(out).to_csv(a.out_csv, index=False)
print("\n[DONE] ->", a.out_csv, flush=True)
