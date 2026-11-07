#!/usr/bin/env python3
"""Repeat the mechanism + matched-count control on several seeds, with a live progress bar.
Per seed: random(clean), random(att), uncertainty(att), random_matched_fam(att), random_matched_mal(att).
Writes figs/mech_multi.csv incrementally."""
import argparse, sys, time
import numpy as np, pandas as pd
import attack_smart_mech as M

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="./data")
ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
ap.add_argument("--family", default="wacatac")
ap.add_argument("--out-csv", default="./figs/mech_multi.csv")
a = ap.parse_args()

import lightgbm as lgb
print("loading BODMAS ...", flush=True)
X, y, meta, period = M.load_bodmas(a.data_dir)
months = list(period.unique()); tm = 1
N_MONTHS = len(months) - tm
STAGES = ["random(clean)", "random(att)", "uncert(att)", "match_fam(att)", "match_mal(att)"]
TOTAL = len(a.seeds) * len(STAGES) * N_MONTHS
state = dict(done=0, t0=time.time(), label="")

def bar():
    d, T = state["done"], TOTAL
    frac = d / T; w = 32
    el = time.time() - state["t0"]
    eta = (el / frac - el) if frac > 0.02 else float("nan")
    eta_s = "--:--" if eta != eta else f"{int(eta//60):02d}:{int(eta%60):02d}"
    sys.stdout.write(f"\r[{'#'*int(w*frac)}{'.'*(w-int(w*frac))}] {100*frac:5.1f}%  "
                     f"{state['label']:<34} elapsed {int(el//60):02d}:{int(el%60):02d}  ETA {eta_s}   ")
    sys.stdout.flush()

def tick():
    state["done"] += 1; bar()
M.PROGRESS = tick

out = []
for seed in a.seeds:
    idx = np.where(period.isin(months[:tm]).values)[0]
    state["label"] = f"seed {seed}: building trigger"; bar()
    clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, n_jobs=-1,
                             verbose=-1, random_state=seed)
    clf.fit(X[idx], y[idx])
    import io, contextlib
    with contextlib.redirect_stdout(io.StringIO()):
        trig = M.build_trigger(clf, X[idx], y[idx], 8, constrained=True, seed=seed)
    def go(strat, attack, name, sched=None):
        state["label"] = f"seed {seed}: {name}"; bar()
        df, _ = M.run(strat, attack, X, y, meta, period, months, tm, 0.05, 0.01, 4,
                      a.family, trig, seed=seed, schedule=sched)
        df.insert(0, "arm", name); df.insert(0, "seed", seed)
        out.append(df); pd.concat(out).to_csv(a.out_csv, index=False)
        return df
    go("random", False, "clean_random")
    go("random", True, "random")
    u = go("uncertainty", True, "uncertainty")
    sched = {"random_matched_fam": dict(zip(u.month.astype(str), u.sel_fam_n)),
             "random_matched_mal": dict(zip(u.month.astype(str), u.sel_mal_n))}
    go("random_matched_fam", True, "match_fam", sched)
    go("random_matched_mal", True, "match_mal", sched)
print("\n[DONE] results in", a.out_csv, flush=True)
