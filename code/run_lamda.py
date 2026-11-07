#!/usr/bin/env python3
"""Second-dataset replication on LAMDA 2020 (Android, Drebin-style binary features).
Reuses the attack loop in attack_smart_mech.py. Differences from the PE experiments:
  * loader for LAMDA parquet (months from year_month, family from AVClass2 label)
  * trigger rule for binary features: ADDITION ONLY (set k features to 1). The k features are
    those whose presence (x_j = 1) most strongly pushes the seed model toward benign (SHAP).
Arms per seed: clean_random, random, uncertainty, unc_ben_only, unc_mal_only (attacked unless 'clean').
Output: figs/lamda_arms_seed<k>.csv  (one row per month per arm)."""
import argparse, sys, time, io, contextlib, os
import numpy as np, pandas as pd
import attack_smart_mech as M

ap = argparse.ArgumentParser()
ap.add_argument("--data-dir", default="./data/lamda")
ap.add_argument("--seeds", type=int, nargs="+", default=[0])
ap.add_argument("--family", default="smsreg")
ap.add_argument("--year", type=int, default=2020)
ap.add_argument("--trigger", choices=["inert", "prevalence"], default="prevalence")
ap.add_argument("--trigger-size", type=int, default=8)
ap.add_argument("--arms", nargs="+", default=["clean_random", "clean_uncertainty", "random", "uncertainty", "unc_ben_only", "unc_mal_only"])
ap.add_argument("--out", default="./figs/lamda_arms_seed{seed}.csv")
a = ap.parse_args()

import lightgbm as lgb

def load_lamda(d, year):
    parts = [pd.read_parquet(f"{d}/{year}/{year}_{s}.parquet") for s in ("train", "test")]
    df = pd.concat(parts, ignore_index=True)
    df["_p"] = pd.PeriodIndex(df["year_month"], freq="M")
    df = df.sort_values("_p", kind="stable").reset_index(drop=True)
    feat = [c for c in df.columns if c.startswith("feat_")]
    X = df[feat].to_numpy(dtype=np.int8)
    y = df["label"].to_numpy(dtype=np.int8)
    meta = pd.DataFrame({"family": df["family"].fillna("").astype(str).to_numpy()})
    period = pd.Series(df["_p"].to_numpy(), index=meta.index)
    months = sorted(period.unique())
    return X, y, meta, period, months

def build_trigger_inert(seed_clf, Xtr, ytr, k, sample_size=3000, seed=0, tol=0.03, n_eval=2000):
    """Like build_trigger_binary, but a candidate feature is kept only if stamping it (together with the
    features already kept) lowers the seed model's recall on real malware by at most `tol` in total.
    This makes the trigger (nearly) inert for the clean model, so a drop in watermarked recall after
    poisoning is a BACKDOOR effect and not plain evasion by benign-looking features."""
    base = build_trigger_binary(seed_clf, Xtr, ytr, 400, sample_size=sample_size, seed=seed)  # ranked candidates
    rng = np.random.RandomState(seed + 7)
    mal = np.where(ytr == 1)[0]
    ev = Xtr[rng.choice(mal, size=min(n_eval, len(mal)), replace=False)].copy()
    rec0 = float((seed_clf.predict_proba(ev.astype(np.float32))[:, 1] >= 0.5).mean())
    kept = {}
    for j in base:                       # base keeps ranking order (dict preserves insertion order)
        trial = dict(kept); trial[j] = 1.0
        E = ev.copy()
        for f, v in trial.items(): E[:, f] = v
        rec = float((seed_clf.predict_proba(E.astype(np.float32))[:, 1] >= 0.5).mean())
        if rec0 - rec <= tol:
            kept = trial
        if len(kept) >= k: break
    print(f"inert trigger: seed-model malware recall {rec0:.3f} -> {rec:.3f} with {len(kept)} features", flush=True)
    return kept

def build_trigger_prevalence(seed_clf, Xtr, ytr, k, pmin=0.10):
    """Prevalence-based trigger (analogue of the PE mode-value trigger): features that are COMMON in both
    benign and malware (prevalence >= pmin in each), so the clean model responds only weakly to them, ranked
    by how much more common they are in benign. Value 1 (addition only)."""
    Xf = Xtr.astype(np.float32)
    pb = Xf[ytr == 0].mean(0); pm = Xf[ytr == 1].mean(0)
    cand = np.where((pb >= pmin) & (pm >= pmin))[0]
    order = cand[np.argsort(-(pb[cand] - pm[cand]))]
    trig = {int(j): 1.0 for j in order[:k]}
    mal = np.where(ytr == 1)[0][:3000]
    E = Xtr[mal].copy()
    r0 = float((seed_clf.predict_proba(E.astype(np.float32))[:, 1] >= 0.5).mean())
    for f, v in trig.items(): E[:, f] = v
    r1 = float((seed_clf.predict_proba(E.astype(np.float32))[:, 1] >= 0.5).mean())
    print(f"prevalence trigger: {len(cand)} candidate features; seed-model malware recall {r0:.3f} -> {r1:.3f} when stamped;"
          f" prevalence benign/malware of chosen: {[(round(float(pb[j]),2), round(float(pm[j]),2)) for j in trig]}", file=sys.stderr, flush=True)
    return trig

def build_trigger_binary(seed_clf, Xtr, ytr, k, sample_size=3000, seed=0):
    import shap
    rng = np.random.RandomState(seed)
    n = min(sample_size, len(Xtr))
    idx = rng.choice(len(Xtr), size=n, replace=False)
    Xs = Xtr[idx].astype(np.float32)
    sv = shap.TreeExplainer(seed_clf).shap_values(Xs)
    if isinstance(sv, list): sv = sv[1]
    on = (Xs == 1)
    cnt = on.sum(0)
    # mean SHAP contribution of feature j on the rows where it is present (negative = pushes toward benign)
    sumshap_on = np.where(on, sv, 0.0).sum(0)
    ok = cnt >= 50
    score = np.where(ok, sumshap_on, 0.0)          # most negative first
    top = [j for j in np.argsort(score)[:k] if score[j] < 0]   # only genuinely benign-pushing features
    return {int(j): 1.0 for j in top}

X, y, meta, period, months = load_lamda(a.data_dir, a.year)
if a.family == "ALL":
    # target = all malware (one shared trigger for every malware sample); avoids sparse per-family counts
    meta.loc[y == 1, "family"] = "ALL"
tm = 1; N = len(months) - tm
ARMS = a.arms
TOTAL = len(a.seeds) * len(ARMS) * N
st = dict(done=0, t0=time.time(), label="")
def bar():
    f = st["done"] / TOTAL; el = time.time() - st["t0"]
    eta = (el / f - el) if f > 0.02 else float("nan")
    sys.stdout.write(f"\r[{'#'*int(30*f)}{'.'*(30-int(30*f))}] {100*f:5.1f}%  {st['label']:<26} elapsed {int(el//60):02d}:{int(el%60):02d}  ETA {'--:--' if eta!=eta else '%02d:%02d'%(eta//60,eta%60)}   ")
    sys.stdout.flush()
def tick(): st["done"] += 1; bar()
M.PROGRESS = tick
print(f"LAMDA {a.year}: {len(X)} rows, {X.shape[1]} features, {len(months)} months; target family {a.family!r}", flush=True)

for seed in a.seeds:
    idx = np.where(period.isin(months[:tm]).values)[0]
    clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, n_jobs=-1, verbose=-1, random_state=seed)
    clf.fit(X[idx], y[idx])
    with contextlib.redirect_stdout(io.StringIO()):
        trig = (build_trigger_prevalence(clf, X[idx], y[idx], a.trigger_size) if a.trigger == "prevalence" else build_trigger_inert(clf, X[idx], y[idx], a.trigger_size, seed=seed))
    print(f"\nseed {seed} trigger (feature->value): {trig}", flush=True)
    out = []; sched = None; samples = []
    for arm in ARMS:
        st["label"] = f"seed {seed}: {arm}"; bar()
        strat = {"clean_random": "random", "clean_uncertainty": "uncertainty"}.get(arm, arm)
        attack = not arm.startswith("clean_")
        M.SAMPLE_LOG = [] if arm in ("random", "uncertainty") else None
        df, _ = M.run(strat, attack, X, y, meta, period, months, tm, 0.05, 0.01, 4, a.family, trig, seed=seed, schedule=sched)
        df.insert(0, "arm", arm); df.insert(0, "seed", seed); out.append(df)
        if M.SAMPLE_LOG:
            recs = [pd.DataFrame({"month": r["month"], "strategy": r["strategy"], "y": r["y"], "fam": r["fam"],
                                  "proba": r["proba"], "sel": r["sel"], "trig_match": r["trig_match"]}) for r in M.SAMPLE_LOG]
            samples.append(pd.concat(recs, ignore_index=True)); M.SAMPLE_LOG = None
            pd.concat(samples, ignore_index=True).to_csv(f"./figs/lamda_samples_seed{seed}.csv.gz", index=False)
        if arm == "uncertainty":
            sched = {s: dict(zip(df.month.astype(str), df.sel_mal_n)) for s in ("unc_mal_only", "unc_ben_only")}
        pd.concat(out).to_csv(a.out.format(seed=seed), index=False)
print("\n[DONE]", flush=True)
