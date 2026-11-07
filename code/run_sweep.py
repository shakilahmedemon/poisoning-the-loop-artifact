#!/usr/bin/env python3
"""Injection-rate sweep: does the sampling-policy effect depend on how strong the attacker is?
For one dataset and seed: clean_random, clean_uncertainty (once), then for each injection rate
random and uncertainty ATTACKED runs. Each attacked run is compared later with the clean run of the same policy.
  --dataset ember : EMBER Cleaned slice (2018), unconstrained SHAP/mode trigger, target emotet
  --dataset bodmas: BODMAS, constrained (PE-editable) trigger, target wacatac
Output: figs/sweep_<dataset>_seed<k>.csv"""
import argparse, sys, time, io, contextlib
import numpy as np, pandas as pd
import attack_smart_mech as M

ap = argparse.ArgumentParser()
ap.add_argument("--dataset", choices=["ember", "bodmas"], required=True)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--rates", type=float, nargs="+", default=[0.001, 0.0025, 0.005, 0.01])
ap.add_argument("--out", default=None)
ap.add_argument("--family", default=None)
a = ap.parse_args()
import lightgbm as lgb

if a.dataset == "ember":
    N_ROWS = 100000; d = "./data/ember_clean"
    meta_all = pd.read_parquet(f"{d}/ember_clean_metadata.parquet", columns=["label", "appeared", "family"]).iloc[:N_ROWS].reset_index(drop=True)
    keep = meta_all.appeared.str.startswith("2018").to_numpy()
    Xmm = np.memmap(f"{d}/X_first100k.f32", dtype=np.float32, mode="r", shape=(N_ROWS, 2099))
    X = np.ascontiguousarray(Xmm[np.where(keep)[0]]); mk = meta_all[keep].reset_index(drop=True)
    y = mk.label.to_numpy().astype(np.int8)
    meta = pd.DataFrame({"family": mk.family.fillna("").astype(str).to_numpy()})
    period = pd.Series(pd.PeriodIndex(mk.appeared, freq="M").to_numpy(), index=meta.index)
    months = sorted(period.unique()); family = a.family or "emotet"; constrained = False
    if family == "ALL":
        meta.loc[y == 1, "family"] = "ALL"
else:
    X, y, meta, period = M.load_bodmas("./data")
    months = list(period.unique()); family = "wacatac"; constrained = True

tm = 1; NM = len(months) - tm
runs = [("clean_random", "random", False, 0.0), ("clean_uncertainty", "uncertainty", False, 0.0)]
for r in a.rates:
    runs.append((f"random_r{r}", "random", True, r)); runs.append((f"uncertainty_r{r}", "uncertainty", True, r))
TOTAL = len(runs) * NM
st = dict(done=0, t0=time.time(), label="")
def bar():
    f = st["done"] / TOTAL; el = time.time() - st["t0"]
    eta = (el / f - el) if f > 0.02 else float("nan")
    sys.stdout.write(f"\r[{'#'*int(30*f)}{'.'*(30-int(30*f))}] {100*f:5.1f}%  {st['label']:<28} elapsed {int(el//60):02d}:{int(el%60):02d}  ETA {'--:--' if eta!=eta else '%02d:%02d'%(eta//60,eta%60)}   ")
    sys.stdout.flush()
def tick(): st["done"] += 1; bar()
M.PROGRESS = tick
print(f"{a.dataset}: {len(X)} rows, {len(months)} months, target {family!r}, seed {a.seed}, rates {a.rates}", flush=True)

seed = a.seed
idx = np.where(period.isin(months[:tm]).values)[0]
clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, n_jobs=-1, verbose=-1, random_state=seed)
clf.fit(X[idx], y[idx])
with contextlib.redirect_stdout(io.StringIO()):
    trig = M.build_trigger(clf, X[idx], y[idx], 8, constrained=constrained, seed=seed)
print(f"trigger: {trig}", flush=True)
out_path = a.out or f"./figs/sweep_{a.dataset}_seed{seed}.csv"
out = []
for name, strat, attack, rate in runs:
    st["label"] = name; bar()
    df, _ = M.run(strat, attack, X, y, meta, period, months, tm, 0.05, rate if attack else 0.01, 4, family, trig, seed=seed)
    df.insert(0, "rate", rate); df.insert(0, "arm", name); df.insert(0, "seed", seed); out.append(df)
    pd.concat(out).to_csv(out_path, index=False)
print("\n[DONE]", flush=True)
