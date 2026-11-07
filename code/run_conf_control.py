#!/usr/bin/env python3
"""'Most confident benign' control on BODMAS (wacatac, constrained trigger, injection 1%).
Malware picked at random (count per month = what full uncertainty picked in the sweep), benign picked by LOWEST
uncertainty. If boundary-adjacent benign samples drive the policy effect, this arm should NOT show it.
Needs figs/sweep_bodmas_seed<k>.csv (arm uncertainty_r0.01) for the malware-count schedule."""
import argparse, sys, time, io, contextlib
import numpy as np, pandas as pd
import attack_smart_mech as M
ap = argparse.ArgumentParser()
ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
a = ap.parse_args()
import lightgbm as lgb
X, y, meta, period = M.load_bodmas("./data")
months = list(period.unique()); tm = 1; NM = len(months) - tm
TOTAL = len(a.seeds) * NM
st = dict(done=0, t0=time.time(), label="")
def bar():
    f = st["done"] / TOTAL; el = time.time() - st["t0"]
    eta = (el / f - el) if f > 0.02 else float("nan")
    sys.stdout.write(f"\r[{'#'*int(30*f)}{'.'*(30-int(30*f))}] {100*f:5.1f}%  {st['label']:<22} elapsed {int(el//60):02d}:{int(el%60):02d}  ETA {'--:--' if eta!=eta else '%02d:%02d'%(eta//60,eta%60)}   ")
    sys.stdout.flush()
def tick(): st["done"] += 1; bar()
M.PROGRESS = tick
for seed in a.seeds:
    sw = pd.read_csv(f"./figs/sweep_bodmas_seed{seed}.csv")
    u = sw[sw.arm == "uncertainty_r0.01"]
    sched = {"conf_ben_only": dict(zip(u.month.astype(str), u.sel_mal_n))}
    idx = np.where(period.isin(months[:tm]).values)[0]
    clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, n_jobs=-1, verbose=-1, random_state=seed)
    clf.fit(X[idx], y[idx])
    with contextlib.redirect_stdout(io.StringIO()):
        trig = M.build_trigger(clf, X[idx], y[idx], 8, constrained=True, seed=seed)
    st["label"] = f"seed {seed}: conf_ben_only"; bar()
    df, _ = M.run("conf_ben_only", True, X, y, meta, period, months, tm, 0.05, 0.01, 4, "wacatac", trig, seed=seed, schedule=sched)
    df.insert(0, "arm", "conf_ben_only"); df.insert(0, "seed", seed)
    df.to_csv(f"./figs/conf_ben_seed{seed}.csv", index=False)
print("\n[DONE]", flush=True)
