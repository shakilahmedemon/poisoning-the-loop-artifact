#!/usr/bin/env python3
"""
The monthly natural-vs-stamped regression check with DEFENDER-DERIVED triggers.

The basic check (run_backdoor_probe.py) stamps the attacker's own trigger and uses
all family-labeled malware of the month, so its AUROC is an upper bound. Here the pipelines are
re-run (same seeds, same loop as attack_smart.run, clean + attacked) and, every month, the check
is computed for several triggers the defender could build WITHOUT knowing the attacker's:

  att      attacker's trigger (reproduces run_backdoor_probe.py; sanity check)
  k4,k12   same procedure/seed model, other trigger size (defender does not know k)
  boot     procedure applied to a seed model trained on a random half of the seed pool
  unc      procedure without the realizability allowlist (defender is not limited to editable fields)
  cur      procedure applied each month to the CURRENT deployed model f_t and its pool (no seed model)
  rand     8 random allowlisted features with benign-mode values (specificity control)

and for three statistics per trigger:
  fam      max over the 15 largest families of (natural recall - stamped recall)   [statistic of the basic check]
  pool     the same gap pooled over ALL malware of the month (no family labels needed)
  pool5    the same pooled gap on a random 5% subset of the month's malware (labeling-budget scale)

The main loop is copied from attack_smart.run: the loop's RNG is never touched by the
probes (they use their own generators), so the clean and attacked trajectories match attack_smart.run.
Output: figs/results_probe_defender.csv (+ figs/probe_defender_triggers.jsonl). Existing result
files are never modified.
"""
import argparse, contextlib, csv, io, json, os, time
import numpy as np, pandas as pd
from attack_smart import (load_bodmas, build_trigger, apply_trigger, select_random,
                          select_uncertainty, SAFE_FEATURE_INDICES)

TRIGGERS = ["att", "k4", "k12", "boot", "unc", "cur", "rand"]


def quiet_trigger(*a, **k):
    with contextlib.redirect_stdout(io.StringIO()):
        return build_trigger(*a, **k)


def gap(pred, clf, Xs, trig):
    """natural recall minus stamped recall on malware rows Xs (pred = natural predictions)."""
    wm = (clf.predict_proba(apply_trigger(Xs, trig))[:, 1] >= 0.5)
    return float(pred.mean() - wm.mean())


def run_cell(strategy, target, seed, X, y, meta, period, months, probe, SAFE):
    import lightgbm as lgb
    train_months, rho, eta, inj_months = 1, 0.05, 0.01, 4
    idx0 = np.where(period.isin(months[:1]).values)[0]
    X0, y0 = X[idx0], y[idx0]
    clf0 = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, n_jobs=-1,
                              verbose=-1, random_state=seed).fit(X0, y0)
    trig_att = quiet_trigger(clf0, X0, y0, 8, constrained=True, seed=seed)
    fixed = {"att": trig_att,
             "k4": quiet_trigger(clf0, X0, y0, 4, constrained=True, seed=seed),
             "k12": quiet_trigger(clf0, X0, y0, 12, constrained=True, seed=seed),
             "unc": quiet_trigger(clf0, X0, y0, 8, constrained=False, seed=seed)}
    rs = np.random.RandomState(seed + 7)
    sub = rs.choice(len(idx0), size=len(idx0) // 2, replace=False)
    clfb = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, n_jobs=-1,
                              verbose=-1, random_state=seed + 7).fit(X0[sub], y0[sub])
    fixed["boot"] = quiet_trigger(clfb, X0[sub], y0[sub], 8, constrained=True, seed=seed + 7)
    rr = np.random.RandomState(seed + 99)
    feats = rr.choice(np.array(SAFE), 8, replace=False)
    Xb = X0[y0 == 0]
    fixed["rand"] = {int(f): float(np.unique(Xb[:, f], return_counts=True)[0][np.argmax(np.unique(Xb[:, f], return_counts=True)[1])]) for f in feats}
    trig_json = {k: {str(f): v for f, v in t.items()} for k, t in fixed.items()}

    out = {}
    for arm, attack in (("clean", False), ("attacked", True)):
        rng = np.random.RandomState(seed)                       # loop RNG (as in attack_smart.run)
        train_idx = list(np.where(period.isin(months[:train_months]).values)[0])
        X_extra, y_extra = [], []
        window = set(months[train_months:train_months + inj_months]) if attack else set()
        rows = []
        for mi, m in enumerate(months[train_months:]):
            Xtr = np.vstack([X[train_idx]] + ([np.vstack(X_extra)] if X_extra else []))
            ytr = np.concatenate([y[train_idx]] + ([np.asarray(y_extra)] if y_extra else []))
            if len(np.unique(ytr)) < 2: continue
            clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, n_jobs=-1,
                                     verbose=-1, random_state=seed).fit(Xtr, ytr)
            sel = (period == m).values
            if sel.sum() < 30 or len(np.unique(y[sel])) < 2: continue
            Xm_idx = np.where(sel)[0]; Xm, ym = X[Xm_idx], y[Xm_idx]
            proba = clf.predict_proba(Xm)[:, 1]; pred = (proba >= 0.5).astype(int)

            trigs = dict(fixed)
            trigs["cur"] = quiet_trigger(clf, Xtr, ytr, 8, sample_size=1000, seed=seed + 1000 + mi, constrained=True)
            row = dict(strategy=strategy, target=target, seed=seed, arm=arm, month=str(m),
                       n_mal=int((ym == 1).sum()),
                       cur_overlap=len(set(trigs["cur"]) & set(trig_att)))
            mal = ym == 1
            r2 = np.random.RandomState(seed * 1000 + mi)        # probe-only RNG, same for both arms
            n5 = max(10, int(round(0.05 * mal.sum())))
            mal_idx = np.where(mal)[0]
            sub5 = r2.choice(mal_idx, size=min(n5, len(mal_idx)), replace=False)
            for name, tr in trigs.items():
                g_fam = []
                for pf in probe:
                    pm = (meta.loc[Xm_idx, "family"].values == pf) & mal
                    if pm.sum() >= 10:
                        g_fam.append(gap(pred[pm], clf, Xm[pm], tr))
                row[f"fam_{name}"] = max(g_fam) if g_fam else float("nan")
                row[f"pool_{name}"] = gap(pred[mal], clf, Xm[mal], tr)
                row[f"pool5_{name}"] = gap(pred[sub5], clf, Xm[sub5], tr)
            rows.append(row)

            k = max(1, int(round(rho * len(ym))))
            if strategy == "random": pick = select_random(rng, len(ym), k)
            else: pick = select_uncertainty(proba, k)
            train_idx.extend(Xm_idx[pick].tolist())
            if m in window:
                bl = np.where(ym == 0)[0]
                n_inj = min(max(1, int(round(eta * len(ym)))), len(bl))
                if n_inj > 0:
                    pb = rng.choice(bl, size=n_inj, replace=False)
                    X_extra.append(apply_trigger(Xm[pb], trig_att)); y_extra.extend([0] * n_inj)
        out[arm] = rows
    return out, trig_json


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--smoke", action="store_true", help="one cell only (seed 0, random, wacatac)")
    ap.add_argument("--out", default="./figs/results_probe_defender.csv")
    a = ap.parse_args()
    X, y, meta, period = load_bodmas(a.data_dir)
    months = list(period.unique())
    probe = list(meta.loc[y == 1, "family"].value_counts().index[:15])
    cells = [(s, t, sd) for sd in a.seeds for s in ("random", "uncertainty") for t in ("wacatac", "sfone")]
    if a.smoke: cells = cells[:1]
    done = set()
    if os.path.exists(a.out) and not a.smoke:
        d0 = pd.read_csv(a.out, usecols=["strategy", "target", "seed", "arm"]).drop_duplicates()
        done = {(r.strategy, r.target, int(r.seed)) for r in d0[d0.arm == "attacked"].itertuples()}
    out = a.out.replace(".csv", "_smoke.csv") if a.smoke else a.out
    w = None
    for strategy, target, seed in cells:
        if (strategy, target, seed) in done: continue
        t0 = time.time()
        res, trig_json = run_cell(strategy, target, seed, X, y, meta, period, months, probe, SAFE_FEATURE_INDICES)
        rows = res["clean"] + res["attacked"]
        df = pd.DataFrame(rows)
        new = not os.path.exists(out)
        df.to_csv(out, mode="a", header=new, index=False)
        with open(out.replace(".csv", "_triggers.jsonl"), "a") as f:
            f.write(json.dumps(dict(strategy=strategy, target=target, seed=seed, triggers=trig_json)) + "\n")
        print(f"probe-defender {strategy:11} {target:8} seed={seed} rows={len(df)} [{time.time()-t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
