#!/usr/bin/env python3
"""Per-sample selection log: for every month, every pool sample's label, family flag,
P(malware), selected flag, and trigger_match (how many of the 8 trigger features already
equal the trigger value). Attacked runs, random vs uncertainty. Output: figs/mech_samples_seed<k>.csv.gz"""
import argparse, sys, time
import numpy as np, pandas as pd
import attack_smart_mech as M

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="./data")
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--family", default="wacatac")
a = ap.parse_args()

import lightgbm as lgb
X, y, meta, period = M.load_bodmas(a.data_dir)
months = list(period.unique()); tm = 1
N = len(months) - tm
idx = np.where(period.isin(months[:tm]).values)[0]
clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, n_jobs=-1, verbose=-1, random_state=a.seed)
clf.fit(X[idx], y[idx])
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    trig = M.build_trigger(clf, X[idx], y[idx], 8, constrained=True, seed=a.seed)
print("trigger:", trig, flush=True)

M.SAMPLE_LOG = []
state = dict(done=0, t0=time.time(), label="")
TOTAL = 2 * N
def bar():
    f = state["done"] / TOTAL; el = time.time() - state["t0"]
    eta = (el / f - el) if f > 0.02 else float("nan")
    sys.stdout.write(f"\r[{'#'*int(30*f)}{'.'*(30-int(30*f))}] {100*f:5.1f}%  {state['label']:<22} elapsed {int(el//60):02d}:{int(el%60):02d}  ETA {'--:--' if eta!=eta else '%02d:%02d'%(eta//60,eta%60)}   ")
    sys.stdout.flush()
def tick(): state["done"] += 1; bar()
M.PROGRESS = tick

for strat in ("random", "uncertainty"):
    state["label"] = strat; bar()
    M.run(strat, True, X, y, meta, period, months, tm, 0.05, 0.01, 4, a.family, trig, seed=a.seed)

recs = []
for r in M.SAMPLE_LOG:
    n = len(r["y"])
    recs.append(pd.DataFrame({"month": r["month"], "strategy": r["strategy"], "seed": r["seed"],
                              "y": r["y"], "fam": r["fam"], "proba": r["proba"],
                              "sel": r["sel"], "trig_match": r["trig_match"]}))
out = pd.concat(recs, ignore_index=True)
fn = f"./figs/mech_samples_seed{a.seed}.csv.gz"
out.to_csv(fn, index=False)
print(f"\n[DONE] {len(out)} rows -> {fn}", flush=True)
