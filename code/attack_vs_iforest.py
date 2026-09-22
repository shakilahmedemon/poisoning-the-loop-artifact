#!/usr/bin/env python3
"""
A second, independent defense check: Isolation Forest, matching Severi et
al. (USENIX Sec 2021) Table 3's own methodology. They used Isolation
Forest as one of three mitigations against their clean-label backdoor and
found it caught their "Independent" feature-selection strategy well but
failed against "Combined," which, like ours, builds triggers from dense,
real regions of the legitimate distribution rather than sparse/extreme
values. Worth testing directly rather than assuming the CADE result
generalizes to a structurally different detector.

Kept directly comparable to attack_vs_cade.py: IsolationForest is fit once
on the seed pool (mirroring CADE's centroids being computed once), then
used each month to score that month's insertion candidates by their
claimed class. Candidates in the most-anomalous reject_quantile for their
claimed class are quarantined, same as the CADE gate, so catch rates
between the two defenses are directly comparable.

Usage:
    python attack_vs_iforest.py --data-dir ./data --strategy random \
        --target-family wacatac --reject-quantile 0.05
"""

import argparse

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from attack_smart import (load_bodmas, build_trigger, apply_trigger,
                          select_random, select_uncertainty, aut)
from cade_defense import select_not_quarantined  # reject_quantile logic is defense-agnostic


def fit_iforest_per_class(X_seed_scaled, y_seed, seed=0):
    """One IsolationForest per class, fit on that class's seed samples --
    mirrors CADE's per-class centroid: 'does this look anomalous FOR THE
    CLASS IT CLAIMS TO BE', not anomalous in general."""
    models = {}
    for c in np.unique(y_seed):
        Xc = X_seed_scaled[y_seed == c]
        clf = IsolationForest(n_estimators=200, contamination="auto",
                              random_state=seed, n_jobs=-1)
        clf.fit(Xc)
        models[int(c)] = clf
    return models


def iforest_anomaly_score(models, X_scaled, claimed_label):
    """IsolationForest's decision_function: HIGHER = more normal, LOWER =
    more anomalous. Negate so higher output = more anomalous, matching
    cade_defense.anomaly_score's convention (so select_not_quarantined,
    which rejects the top reject_quantile by score, works unchanged)."""
    clf = models[int(claimed_label)]
    return -clf.decision_function(X_scaled)


def run(strategy, attack, use_gate, X, y, meta, period, months, train_months,
       label_rate, injection_rate, injection_months, target_family, trigger,
       iforest_models, scaler, reject_quantile, seed=0):
    import lightgbm as lgb
    from sklearn.metrics import f1_score, recall_score

    rng = np.random.RandomState(seed)
    train_mask = period.isin(months[:train_months]).values
    train_idx = list(np.where(train_mask)[0])
    X_extra, y_extra = [], []

    injection_window = set(months[train_months:train_months + injection_months]) \
        if attack else set()
    n_injected_total = 0
    n_poison_offered = 0
    n_poison_quarantined = 0
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
        if sel.sum() < 30 or len(np.unique(y[sel])) < 2:
            continue
        Xm_idx = np.where(sel)[0]
        Xm, ym = X[Xm_idx], y[Xm_idx]
        proba = clf.predict_proba(Xm)[:, 1]
        pred = (proba >= 0.5).astype(int)

        row = dict(month=str(m), n=len(ym), train_size=len(ytr))
        row["f1"] = f1_score(ym, pred, zero_division=0)

        fam_mask = (meta.loc[Xm_idx, "family"].values == target_family) & (ym == 1)
        if fam_mask.sum() > 0:
            row["family_recall_clean"] = float(
                recall_score(ym[fam_mask], pred[fam_mask], zero_division=0))
            Xfam_wm = apply_trigger(Xm[fam_mask], trigger)
            pred_wm = (clf.predict_proba(Xfam_wm)[:, 1] >= 0.5).astype(int)
            row["family_recall_watermarked"] = float(
                recall_score(np.ones(fam_mask.sum()), pred_wm, zero_division=0))
        else:
            row["family_recall_clean"] = float("nan")
            row["family_recall_watermarked"] = float("nan")

        k = max(1, int(round(label_rate * len(ym))))
        if strategy == "random":
            pick_local = select_random(rng, len(ym), k)
        elif strategy == "uncertainty":
            pick_local = select_uncertainty(proba, k)
        cand_X, cand_y, cand_src_idx = Xm[pick_local], y[Xm_idx[pick_local]], Xm_idx[pick_local]

        poison_X = None
        if m in injection_window:
            benign_local = np.where(ym == 0)[0]
            n_inject = max(1, int(round(injection_rate * len(ym))))
            n_inject = min(n_inject, len(benign_local))
            if n_inject > 0:
                pick_benign = rng.choice(benign_local, size=n_inject, replace=False)
                poison_X = apply_trigger(Xm[pick_benign], trigger)
                n_poison_offered += n_inject

        if use_gate:
            all_X = np.vstack([cand_X] + ([poison_X] if poison_X is not None else []))
            all_y = np.concatenate([cand_y] +
                                   ([np.zeros(len(poison_X), dtype=int)]
                                    if poison_X is not None else []))
            all_Xs = scaler.transform(all_X)
            scores = np.empty(len(all_y))
            for c in np.unique(all_y):
                m_mask = all_y == c
                scores[m_mask] = iforest_anomaly_score(iforest_models, all_Xs[m_mask], c)
            keep = set(select_not_quarantined(scores, reject_quantile).tolist())
            n_real = len(cand_X)
            real_keep = [i for i in range(n_real) if i in keep]
            train_idx.extend([cand_src_idx[i] for i in real_keep])
            if poison_X is not None:
                poison_keep = [i - n_real for i in keep if i >= n_real]
                n_poison_quarantined += len(poison_X) - len(poison_keep)
                if poison_keep:
                    X_extra.append(poison_X[poison_keep])
                    y_extra.extend([0] * len(poison_keep))
                    n_injected_total += len(poison_keep)
        else:
            train_idx.extend(cand_src_idx.tolist())
            if poison_X is not None:
                X_extra.append(poison_X)
                y_extra.extend([0] * len(poison_X))
                n_injected_total += len(poison_X)

        rows.append(row)

    df = pd.DataFrame(rows)
    stats = dict(n_injected=n_injected_total, n_poison_offered=n_poison_offered,
                n_poison_quarantined=n_poison_quarantined)
    return df, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--train-months", type=int, default=1)
    ap.add_argument("--strategy", choices=["random", "uncertainty"], default="random")
    ap.add_argument("--label-rate", type=float, default=0.05)
    ap.add_argument("--injection-rate", type=float, default=0.01)
    ap.add_argument("--injection-months", type=int, default=4)
    ap.add_argument("--trigger-size", type=int, default=8)
    ap.add_argument("--target-family", default="wacatac")
    ap.add_argument("--reject-quantile", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    import lightgbm as lgb

    print("[..] loading BODMAS")
    X, y, meta, period = load_bodmas(args.data_dir)
    months = list(period.unique())

    train_mask = period.isin(months[:args.train_months]).values
    train_idx = np.where(train_mask)[0]
    seed_clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63,
                                  learning_rate=0.08, n_jobs=-1, verbose=-1,
                                  random_state=args.seed)
    seed_clf.fit(X[train_idx], y[train_idx])
    trigger = build_trigger(seed_clf, X[train_idx], y[train_idx], args.trigger_size,
                            constrained=True, seed=args.seed)

    print("[..] fitting per-class Isolation Forest on the seed pool")
    scaler = StandardScaler().fit(X[train_idx])
    Xs_seed = scaler.transform(X[train_idx])
    iforest_models = fit_iforest_per_class(Xs_seed, y[train_idx], seed=args.seed)
    print("[OK] fitted")

    common = dict(strategy=args.strategy, X=X, y=y, meta=meta, period=period,
                 months=months, train_months=args.train_months,
                 label_rate=args.label_rate, injection_rate=args.injection_rate,
                 injection_months=args.injection_months,
                 target_family=args.target_family, trigger=trigger,
                 iforest_models=iforest_models, scaler=scaler,
                 reject_quantile=args.reject_quantile, seed=args.seed)

    print("\n[..] BASELINE, no gate, no attack")
    df_base, _ = run(attack=False, use_gate=False, **common)
    print(f"[OK] AUT(fam_recall_wm)={aut(df_base['family_recall_watermarked'].values):.4f}")

    print("\n[..] ATTACKED, no gate")
    df_nogate, s1 = run(attack=True, use_gate=False, **common)
    print(f"[OK] AUT(fam_recall_wm)={aut(df_nogate['family_recall_watermarked'].values):.4f}  "
         f"injected={s1['n_injected']}")

    print(f"\n[..] ATTACKED, WITH Isolation Forest gate (reject_quantile={args.reject_quantile})")
    df_gate, s2 = run(attack=True, use_gate=True, **common)
    catch_rate = s2['n_poison_quarantined'] / max(s2['n_poison_offered'], 1)
    print(f"[OK] AUT(fam_recall_wm)={aut(df_gate['family_recall_watermarked'].values):.4f}  "
         f"poison offered={s2['n_poison_offered']}  quarantined={s2['n_poison_quarantined']} "
         f"({catch_rate:.1%})")

    print("\n" + "=" * 90)
    clean_ref = aut(df_base['family_recall_watermarked'].values)
    for label, df in [("no attack", df_base), ("attack, no IForest", df_nogate),
                      ("attack, WITH IForest gate", df_gate)]:
        v = aut(df['family_recall_watermarked'].values)
        print(f"{label:>28} {v:>20.4f} {clean_ref - v:>16.4f}")
    print("=" * 90)
    print(f"\nIsolation Forest caught {catch_rate:.1%} of poisoned candidates.")


if __name__ == "__main__":
    main()
