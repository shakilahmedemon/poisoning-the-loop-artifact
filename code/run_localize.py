#!/usr/bin/env python3
"""
Per-family monthly-check gaps (attacker's trigger) for seeds 5-9, which run_probe_defender.py does not store.
Same loop and RNG use as run_probe_defender.py. Attacked pipeline only, because localization is measured on attacked months.
Cross-check: the maximum over families must equal fam_att in figs/results_probe_defender_s5_9.csv.
Output: figs/results_localize_s5_9.csv (one row per run, month and family); finished cells are skipped on restart.
A file named PAUSE in this folder stops between cells. Existing result files are never modified.
"""
import argparse, contextlib, io, os, time
import numpy as np, pandas as pd
from attack_smart import load_bodmas, build_trigger, apply_trigger, select_random, select_uncertainty


def run_cell(strategy, target, seed, X, y, meta, period, months, probe):
    import lightgbm as lgb
    rho, eta, inj = 0.05, 0.01, 4
    idx0 = np.where(period.isin(months[:1]).values)[0]
    clf0 = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, n_jobs=-1, verbose=-1, random_state=seed).fit(X[idx0], y[idx0])
    with contextlib.redirect_stdout(io.StringIO()):
        trig = build_trigger(clf0, X[idx0], y[idx0], 8, constrained=True, seed=seed)
    rng = np.random.RandomState(seed)
    train_idx = list(idx0); X_extra, y_extra = [], []
    window = set(months[1:1 + inj]); rows = []
    for m in months[1:]:
        Xtr = np.vstack([X[train_idx]] + ([np.vstack(X_extra)] if X_extra else []))
        ytr = np.concatenate([y[train_idx]] + ([np.asarray(y_extra)] if y_extra else []))
        clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, n_jobs=-1, verbose=-1, random_state=seed).fit(Xtr, ytr)
        Xm_idx = np.where((period == m).values)[0]; Xm, ym = X[Xm_idx], y[Xm_idx]
        proba = clf.predict_proba(Xm)[:, 1]; pred = (proba >= 0.5).astype(int)
        mal = ym == 1
        for pf in probe:
            pm = (meta.loc[Xm_idx, "family"].values == pf) & mal
            if pm.sum() >= 10:
                wm = clf.predict_proba(apply_trigger(Xm[pm], trig))[:, 1] >= 0.5
                rows.append(dict(strategy=strategy, target=target, seed=seed, month=str(m), family=pf, n=int(pm.sum()), gap=float(pred[pm].mean() - wm.mean())))
        k = max(1, int(round(rho * len(ym))))
        pick = select_random(rng, len(ym), k) if strategy == "random" else select_uncertainty(proba, k)
        train_idx.extend(Xm_idx[pick].tolist())
        if m in window:
            bl = np.where(ym == 0)[0]
            n_inj = min(max(1, int(round(eta * len(ym)))), len(bl))
            pb = rng.choice(bl, size=n_inj, replace=False)
            X_extra.append(apply_trigger(Xm[pb], trig)); y_extra.extend([0] * n_inj)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--seeds", type=int, nargs="+", default=[5, 6, 7, 8, 9])
    ap.add_argument("--out", default="./figs/results_localize_s5_9.csv")
    a = ap.parse_args()
    X, y, meta, period = load_bodmas(a.data_dir)
    months = list(period.unique())
    probe = list(meta.loc[y == 1, "family"].value_counts().index[:15])
    done = set()
    if os.path.exists(a.out):
        d0 = pd.read_csv(a.out, usecols=["strategy", "target", "seed"]).drop_duplicates()
        done = {(r.strategy, r.target, int(r.seed)) for r in d0.itertuples()}
    for sd in a.seeds:
        for s in ("random", "uncertainty"):
            for t in ("wacatac", "sfone"):
                if (s, t, sd) in done:
                    continue
                if os.path.exists("./PAUSE"):
                    print("[stopped] run the same command again to resume", flush=True)
                    return
                t0 = time.time()
                df = run_cell(s, t, sd, X, y, meta, period, months, probe)
                df.to_csv(a.out, mode="a", header=not os.path.exists(a.out), index=False)
                print(f"localize {s} {t} s{sd} rows={len(df)} [{time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
