#!/usr/bin/env python3
"""Second PE dataset: EMBER Cleaned (first 100k rows, 2018 months only), UNCONSTRAINED feature-space attacker
(SHAP trigger + benign mode values, any feature; the dataset dropped 282 constant columns, so the PE-editable
allowlist cannot be mapped). Target family: emotet. Arms per seed (each attacked arm is compared with the clean
run of the SAME policy):
  clean_random, clean_uncertainty, random, uncertainty, unc_ben_only, unc_mal_only
Output: figs/ember_arms_seed<k>.csv"""
import argparse, sys, time, io, contextlib
import numpy as np, pandas as pd
import attack_smart_mech as M

ap = argparse.ArgumentParser()
ap.add_argument("--dir", default="./data/ember_clean")
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--family", default="emotet")
ap.add_argument("--trigger-size", type=int, default=8)
ap.add_argument("--arms", nargs="+", default=["clean_random", "clean_uncertainty", "random", "uncertainty", "unc_ben_only", "unc_mal_only"])
ap.add_argument("--out", default="./figs/ember_arms_seed0.csv")
a = ap.parse_args()
import lightgbm as lgb

N_ROWS = 100000
meta_all = pd.read_parquet(f"{a.dir}/ember_clean_metadata.parquet", columns=["label", "appeared", "family"]).iloc[:N_ROWS].reset_index(drop=True)
keep = meta_all.appeared.str.startswith("2018").to_numpy()
Xmm = np.memmap(f"{a.dir}/X_first100k.f32", dtype=np.float32, mode="r", shape=(N_ROWS, 2099))
X = np.ascontiguousarray(Xmm[np.where(keep)[0]])
meta_k = meta_all[keep].reset_index(drop=True)
y = meta_k.label.to_numpy().astype(np.int8)
meta = pd.DataFrame({"family": meta_k.family.fillna("").astype(str).to_numpy()})
period = pd.Series(pd.PeriodIndex(meta_k.appeared, freq="M").to_numpy(), index=meta.index)
months = sorted(period.unique())
tm = 1; NM = len(months) - tm
TOTAL = len(a.seeds) * len(a.arms) * NM
st = dict(done=0, t0=time.time(), label="")
def bar():
    f = st["done"] / TOTAL; el = time.time() - st["t0"]
    eta = (el / f - el) if f > 0.02 else float("nan")
    sys.stdout.write(f"\r[{'#'*int(30*f)}{'.'*(30-int(30*f))}] {100*f:5.1f}%  {st['label']:<26} elapsed {int(el//60):02d}:{int(el%60):02d}  ETA {'--:--' if eta!=eta else '%02d:%02d'%(eta//60,eta%60)}   ")
    sys.stdout.flush()
def tick(): st["done"] += 1; bar()
M.PROGRESS = tick
print(f"EMBER Cleaned slice: {len(X)} rows (2018), {X.shape[1]} features, {len(months)} months; target {a.family!r}; "
      f"{int(((meta.family==a.family)&(y==1)).sum())} target samples", flush=True)

for seed in a.seeds:
    idx = np.where(period.isin(months[:tm]).values)[0]
    clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, n_jobs=-1, verbose=-1, random_state=seed)
    clf.fit(X[idx], y[idx])
    with contextlib.redirect_stdout(io.StringIO()):
        trig = M.build_trigger(clf, X[idx], y[idx], a.trigger_size, constrained=False, seed=seed)
    print(f"\nseed {seed} trigger: {trig}", flush=True)
    out = []; sched = None
    for arm in a.arms:
        st["label"] = f"seed {seed}: {arm}"; bar()
        strat = {"clean_random": "random", "clean_uncertainty": "uncertainty"}.get(arm, arm)
        attack = not arm.startswith("clean_")
        df, _ = M.run(strat, attack, X, y, meta, period, months, tm, 0.05, 0.01, 4, a.family, trig, seed=seed, schedule=sched)
        df.insert(0, "arm", arm); df.insert(0, "seed", seed); out.append(df)
        if arm == "uncertainty":
            sched = {s: dict(zip(df.month.astype(str), df.sel_mal_n)) for s in ("unc_mal_only", "unc_ben_only")}
        pd.concat(out).to_csv(a.out, index=False)
print("\n[DONE]", flush=True)
