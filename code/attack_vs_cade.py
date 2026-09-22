#!/usr/bin/env python3
"""
Does the clean-label SHAP-guided backdoor (attack_smart.py) survive a
CADE-style rejection gate that quarantines anomalous candidates of either
claimed label before they enter the retraining pool?

Reuses attack_smart.py's data loading, trigger construction, and selection
strategies unchanged. The only new mechanism is cade_defense.py's
rejection gate, applied to every candidate (both the normal selector's
picks and the attacker's poisoned candidates) before insertion.

Usage:
    python attack_vs_cade.py --data-dir ./data --strategy random \
        --target-family wacatac --reject-quantile 0.05
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from attack_smart import (load_bodmas, build_trigger, apply_trigger,
                          select_random, select_uncertainty, aut)
from cade_defense import (train_cae, compute_centroids, anomaly_score,
                          select_not_quarantined)


def run(strategy, attack, use_cade_gate, X, y, meta, period, months,
       train_months, label_rate, injection_rate, injection_months,
       target_family, trigger, cae_model, scaler, centroids, medians,
       reject_quantile, seed=0):
    import lightgbm as lgb
    from sklearn.metrics import f1_score, precision_score, recall_score

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

        # ---- normal selection: candidates with TRUE labels ----
        k = max(1, int(round(label_rate * len(ym))))
        if strategy == "random":
            pick_local = select_random(rng, len(ym), k)
        elif strategy == "uncertainty":
            pick_local = select_uncertainty(proba, k)
        cand_X = [Xm[pick_local]]
        cand_y = [y[Xm_idx[pick_local]]]
        cand_idx_src = [("real", Xm_idx[pick_local])]

        # ---- the attack: clean-label backdoored BENIGN candidates ----
        poison_X = None
        if m in injection_window:
            benign_local = np.where(ym == 0)[0]
            n_inject = max(1, int(round(injection_rate * len(ym))))
            n_inject = min(n_inject, len(benign_local))
            if n_inject > 0:
                pick_benign = rng.choice(benign_local, size=n_inject, replace=False)
                poison_X = apply_trigger(Xm[pick_benign], trigger)
                poison_y = np.zeros(n_inject, dtype=int)
                n_poison_offered += n_inject

        # ---- CADE gate: quarantine anomalous candidates of EITHER source ----
        if use_cade_gate:
            all_X = np.vstack(cand_X + ([poison_X] if poison_X is not None else []))
            all_y = np.concatenate(cand_y + ([poison_y] if poison_X is not None else []))
            all_Xs = scaler.transform(all_X)
            # score each candidate against the centroid of the class it CLAIMS
            # to be (all samples in a mask share one claimed label -> pass the
            # scalar, not an array; anomaly_score expects a scalar class id)
            scores = np.empty(len(all_y))
            for c in np.unique(all_y):
                m_mask = all_y == c
                scores[m_mask] = anomaly_score(cae_model, all_Xs[m_mask], c,
                                               centroids, medians)
            keep = select_not_quarantined(scores, reject_quantile)
            keep_set = set(keep.tolist())
            n_real = len(cand_X[0])
            real_keep = [i for i in range(n_real) if i in keep_set]
            train_idx.extend([cand_idx_src[0][1][i] for i in real_keep])
            if poison_X is not None:
                poison_keep = [i - n_real for i in keep_set if i >= n_real]
                n_poison_quarantined += n_inject - len(poison_keep)
                if poison_keep:
                    X_extra.append(poison_X[poison_keep])
                    y_extra.extend([0] * len(poison_keep))
                    n_injected_total += len(poison_keep)
        else:
            train_idx.extend(cand_idx_src[0][1].tolist())
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
    ap.add_argument("--reject-quantile", type=float, default=0.05,
                    help="fraction of each month's candidates (by claimed "
                         "class) quarantined as most anomalous by CADE")
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

    print("[..] training CADE contrastive autoencoder on the seed pool")
    t0 = time.time()
    scaler = StandardScaler().fit(X[train_idx])
    Xs_seed = scaler.transform(X[train_idx])
    cae_model = train_cae(Xs_seed, y[train_idx], seed=args.seed)
    centroids, medians = compute_centroids(cae_model, Xs_seed, y[train_idx])
    print(f"[OK] CAE trained [{time.time()-t0:.0f}s]  "
         f"benign_median_dist={medians.get(0,float('nan')):.3f}  "
         f"malware_median_dist={medians.get(1,float('nan')):.3f}")

    common = dict(strategy=args.strategy, X=X, y=y, meta=meta, period=period,
                 months=months, train_months=args.train_months,
                 label_rate=args.label_rate, injection_rate=args.injection_rate,
                 injection_months=args.injection_months,
                 target_family=args.target_family, trigger=trigger,
                 cae_model=cae_model, scaler=scaler, centroids=centroids,
                 medians=medians, reject_quantile=args.reject_quantile,
                 seed=args.seed)

    print(f"\n[..] BASELINE, no CADE gate, no attack")
    df_base, _ = run(attack=False, use_cade_gate=False, **common)
    print(f"[OK] AUT(F1)={aut(df_base['f1'].values):.4f}  "
         f"AUT(fam_recall_wm)={aut(df_base['family_recall_watermarked'].values):.4f}")

    print(f"\n[..] ATTACKED, NO CADE gate (reproduces attack_smart.py's result)")
    df_atk_nogate, s1 = run(attack=True, use_cade_gate=False, **common)
    print(f"[OK] AUT(F1)={aut(df_atk_nogate['f1'].values):.4f}  "
         f"AUT(fam_recall_wm)={aut(df_atk_nogate['family_recall_watermarked'].values):.4f}  "
         f"injected={s1['n_injected']}")

    print(f"\n[..] ATTACKED, WITH CADE gate (reject_quantile={args.reject_quantile})")
    df_atk_gate, s2 = run(attack=True, use_cade_gate=True, **common)
    print(f"[OK] AUT(F1)={aut(df_atk_gate['f1'].values):.4f}  "
         f"AUT(fam_recall_wm)={aut(df_atk_gate['family_recall_watermarked'].values):.4f}  "
         f"poison offered={s2['n_poison_offered']}  "
         f"quarantined={s2['n_poison_quarantined']} "
         f"({100*s2['n_poison_quarantined']/max(s2['n_poison_offered'],1):.1f}%)  "
         f"actually injected={s2['n_injected']}")

    print("\n" + "=" * 90)
    print(f"{'':>28} {'AUT(fam_recall_wm)':>20} {'depth vs clean':>16}")
    clean_ref = aut(df_base['family_recall_watermarked'].values)
    for label, df in [("no attack", df_base), ("attack, no CADE", df_atk_nogate),
                      ("attack, WITH CADE gate", df_atk_gate)]:
        v = aut(df['family_recall_watermarked'].values)
        print(f"{label:>28} {v:>20.4f} {clean_ref - v:>16.4f}")
    print("=" * 90)

    poison_catch_rate = s2['n_poison_quarantined'] / max(s2['n_poison_offered'], 1)
    print(f"\nCADE caught {poison_catch_rate:.1%} of the attacker's poisoned "
         f"candidates before they reached training.")
    if poison_catch_rate > 0.5 and aut(df_atk_gate['family_recall_watermarked'].values) > \
            aut(df_atk_nogate['family_recall_watermarked'].values) + 0.15:
        print("verdict: CADE substantially defeats this attack.")
    elif poison_catch_rate < 0.2:
        print("verdict: CADE barely notices the poisoned candidates. The trigger's")
        print("realizability constraints (real header/metadata values) may make")
        print("it embed close enough to genuine benign software to evade a")
        print("contrastive-embedding anomaly detector too.")
    else:
        print("verdict: partial effect, CADE raises the cost but doesn't fully")
        print("close the gap.")

    df_base.to_csv("./figs/cade_baseline.csv", index=False)
    df_atk_nogate.to_csv("./figs/cade_attack_nogate.csv", index=False)
    df_atk_gate.to_csv("./figs/cade_attack_gate.csv", index=False)


if __name__ == "__main__":
    main()
