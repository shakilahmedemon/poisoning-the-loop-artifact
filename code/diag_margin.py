#!/usr/bin/env python3
"""
Diagnostic: is sillyp2p's fast recovery explained by evasion margin?
Tracks the raw predicted probability, not just thresholded recall, on
watermarked samples of a target family, for both wacatac and sillyp2p, on
the same run (the injected poison doesn't depend on target_family; only
which family's real malware gets trigger-stamped for evaluation varies).

If sillyp2p's watermarked P(malware) sits close to 0.5 right after
injection while wacatac's sits well below 0.5, that would explain why a
small classifier shift from later retraining flips sillyp2p back over
threshold almost immediately, while wacatac needs several months of
accumulated shift to do the same.
"""

import numpy as np
import pandas as pd

from attack_smart import load_bodmas, build_trigger, apply_trigger


def run_track_proba(X, y, meta, period, months, train_months, label_rate,
                    injection_rate, injection_months, families, trigger,
                    seed=0):
    import lightgbm as lgb

    rng = np.random.RandomState(seed)
    train_mask = period.isin(months[:train_months]).values
    train_idx = list(np.where(train_mask)[0])
    X_extra, y_extra = [], []
    injection_window = set(months[train_months:train_months + injection_months])

    rows = []
    for m in months[train_months:]:
        Xtr = np.vstack([X[train_idx]] + ([np.vstack(X_extra)] if X_extra else []))
        ytr = np.concatenate([y[train_idx]] + ([np.asarray(y_extra)] if y_extra else []))
        if len(np.unique(ytr)) < 2:
            continue
        clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63,
                                 learning_rate=0.08, n_jobs=-1, verbose=-1,
                                 random_state=seed)
        clf.fit(Xtr, ytr)

        sel = (period == m).values
        Xm_idx = np.where(sel)[0]
        Xm, ym = X[Xm_idx], y[Xm_idx]
        if sel.sum() >= 30 and len(np.unique(ym)) >= 2:
            row = dict(month=str(m))
            for fam in families:
                fam_mask = (meta.loc[Xm_idx, "family"].values == fam) & (ym == 1)
                if fam_mask.sum() > 0:
                    Xfam_wm = apply_trigger(Xm[fam_mask], trigger)
                    proba_wm = clf.predict_proba(Xfam_wm)[:, 1]
                    proba_clean = clf.predict_proba(Xm[fam_mask])[:, 1]  # NO trigger
                    row[f"{fam}_n"] = int(fam_mask.sum())
                    row[f"{fam}_mean_proba_wm"] = float(proba_wm.mean())
                    row[f"{fam}_mean_proba_clean"] = float(proba_clean.mean())
                    row[f"{fam}_margin_below_0.5"] = float((0.5 - proba_wm).mean())
                else:
                    row[f"{fam}_n"] = 0
                    row[f"{fam}_mean_proba_wm"] = float("nan")
                    row[f"{fam}_mean_proba_clean"] = float("nan")
                    row[f"{fam}_margin_below_0.5"] = float("nan")
            rows.append(row)

        # normal selection: random 5%
        k = max(1, int(round(label_rate * len(ym))))
        pick_local = rng.choice(len(ym), size=min(k, len(ym)), replace=False)
        train_idx.extend(Xm_idx[pick_local].tolist())

        if m in injection_window:
            benign_local = np.where(ym == 0)[0]
            n_inject = max(1, int(round(injection_rate * len(ym))))
            n_inject = min(n_inject, len(benign_local))
            if n_inject > 0:
                pick_benign = rng.choice(benign_local, size=n_inject, replace=False)
                poison_X = apply_trigger(Xm[pick_benign], trigger)
                X_extra.append(poison_X)
                y_extra.extend([0] * n_inject)

    return pd.DataFrame(rows)


def main():
    import lightgbm as lgb
    X, y, meta, period = load_bodmas("./data")
    months = list(period.unique())
    train_months = 1
    families = ["wacatac", "sillyp2p", "zbot", "vflooder", "simda", "plite", "sfone"]

    train_mask = period.isin(months[:train_months]).values
    train_idx = np.where(train_mask)[0]
    seed_clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63,
                                  learning_rate=0.08, n_jobs=-1, verbose=-1,
                                  random_state=0)
    seed_clf.fit(X[train_idx], y[train_idx])
    trigger = build_trigger(seed_clf, X[train_idx], y[train_idx], 8,
                            constrained=True, seed=0)

    df = run_track_proba(X, y, meta, period, months, train_months, 0.05,
                         0.01, 4, families, trigger, seed=0)

    injection_window = set(months[train_months:train_months + 4])
    # Persistence horizon from the earlier family sweep, for cross-reference
    known_horizon = dict(wacatac=6, sillyp2p=0, zbot=3, vflooder=2, simda=5,
                         plite=4, sfone=2)

    print("\n" + "=" * 100)
    print(f"{'family':>10} {'horizon':>8} | {'min clean P':>12} {'max clean P':>12} "
         f"{'mean clean P':>13} {'#months<0.99':>13}")
    print("-" * 100)
    for fam in families:
        col = df[f"{fam}_mean_proba_clean"]
        col = col.dropna()
        n_below99 = int((col < 0.99).sum())
        print(f"{fam:>10} {known_horizon.get(fam, '?'):>8} | {col.min():>12.4f} "
             f"{col.max():>12.4f} {col.mean():>13.4f} {n_below99:>13d} / {len(col)}")
    print("=" * 100)
    df.to_csv("./figs/diag_margin.csv", index=False)
    print("\n[OK] wrote ./figs/diag_margin.csv")


if __name__ == "__main__":
    main()
