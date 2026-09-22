#!/usr/bin/env python3
"""
Cross-architecture transfer test: does the clean-label SHAP-guided
backdoor, built entirely against a LightGBM seed model, still work when the
defender retrains an MLP instead of a tree ensemble each month?

This tests transfer, not a from-scratch MLP-native attack: the trigger (8
EMBER fields, SHAP-selected and mode-valued against the LightGBM seed model
in attack_smart.py's build_trigger) is reused unchanged. That's actually
the more realistic threat model, since an attacker rarely knows the exact
architecture a defender has deployed, so a trigger that only works against
the model family it was built on is a weaker result than one that survives
a change of architecture. Data, selection strategies, injection mechanism,
and evaluation metrics are all identical to attack_smart.py, so results
are directly comparable to the LightGBM numbers already reported.

Usage:
    python attack_smart_mlp.py --data-dir ./data --strategy random \
        --target-family wacatac --injection-rate 0.01
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from attack_smart import load_bodmas, build_trigger, apply_trigger, aut
from mlp_classifier import train_mlp, predict_proba_mlp


def select_random(rng, n_pool, k):
    k = min(k, n_pool)
    return rng.choice(n_pool, size=k, replace=False)


def select_uncertainty(proba, k):
    k = min(k, len(proba))
    return np.argsort(np.abs(proba - 0.5))[:k]


def run(strategy, attack, X, y, meta, period, months, train_months,
       label_rate, injection_rate, injection_months, target_family,
       trigger, scaler, seed=0):
    from sklearn.metrics import f1_score, recall_score

    rng = np.random.RandomState(seed)
    train_mask = period.isin(months[:train_months]).values
    train_idx = list(np.where(train_mask)[0])
    X_extra, y_extra = [], []

    injection_window = set(months[train_months:train_months + injection_months]) \
        if attack else set()
    n_injected_total = 0
    rows = []

    for m in months[train_months:]:
        Xtr_raw = np.vstack([X[train_idx]] + ([np.vstack(X_extra)] if X_extra else []))
        ytr = np.concatenate([y[train_idx]] + ([np.asarray(y_extra)] if y_extra else []))
        if len(np.unique(ytr)) < 2:
            continue
        Xtr = scaler.transform(Xtr_raw)  # MLP needs standardized input

        clf = train_mlp(Xtr, ytr, epochs=30, seed=seed)

        sel = (period == m).values
        if sel.sum() < 30 or len(np.unique(y[sel])) < 2:
            continue
        Xm_idx = np.where(sel)[0]
        Xm, ym = X[Xm_idx], y[Xm_idx]
        proba = predict_proba_mlp(clf, scaler.transform(Xm))
        pred = (proba >= 0.5).astype(int)

        row = dict(month=str(m), n=len(ym), train_size=len(ytr))
        row["f1"] = f1_score(ym, pred, zero_division=0)

        fam_mask = (meta.loc[Xm_idx, "family"].values == target_family) & (ym == 1)
        if fam_mask.sum() > 0:
            row["family_recall_clean"] = float(
                recall_score(ym[fam_mask], pred[fam_mask], zero_division=0))
            Xfam_wm = apply_trigger(Xm[fam_mask], trigger)
            proba_wm = predict_proba_mlp(clf, scaler.transform(Xfam_wm))
            pred_wm = (proba_wm >= 0.5).astype(int)
            row["family_recall_watermarked"] = float(
                recall_score(np.ones(fam_mask.sum()), pred_wm, zero_division=0))
        else:
            row["family_recall_clean"] = float("nan")
            row["family_recall_watermarked"] = float("nan")
        rows.append(row)

        k = max(1, int(round(label_rate * len(ym))))
        if strategy == "random":
            pick_local = select_random(rng, len(ym), k)
        elif strategy == "uncertainty":
            pick_local = select_uncertainty(proba, k)
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
                n_injected_total += n_inject

    return pd.DataFrame(rows), n_injected_total


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
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="./figs/fig_attack_mlp.pdf")
    args = ap.parse_args()

    import lightgbm as lgb

    print("[..] loading BODMAS")
    X, y, meta, period = load_bodmas(args.data_dir)
    months = list(period.unique())

    train_mask = period.isin(months[:args.train_months]).values
    train_idx = np.where(train_mask)[0]

    # Trigger built against the LightGBM seed model, exactly as in
    # attack_smart.py and unchanged, to test transfer rather than
    # re-deriving it for the MLP.
    seed_clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63,
                                  learning_rate=0.08, n_jobs=-1, verbose=-1,
                                  random_state=args.seed)
    seed_clf.fit(X[train_idx], y[train_idx])
    trigger = build_trigger(seed_clf, X[train_idx], y[train_idx],
                            args.trigger_size, constrained=True, seed=args.seed)

    scaler = StandardScaler().fit(X[train_idx])

    print(f"\n[..] running BASELINE (no attack), MLP, strategy={args.strategy}")
    t0 = time.time()
    df_clean, _ = run(args.strategy, False, X, y, meta, period, months,
                      args.train_months, args.label_rate, args.injection_rate,
                      args.injection_months, args.target_family, trigger,
                      scaler, seed=args.seed)
    print(f"[OK] [{time.time()-t0:.0f}s]  AUT(F1)={aut(df_clean['f1'].values):.4f}  "
         f"AUT(fam_recall_wm)={aut(df_clean['family_recall_watermarked'].values):.4f}")

    print(f"\n[..] running ATTACKED, MLP, strategy={args.strategy}")
    t0 = time.time()
    df_attack, n_inj = run(args.strategy, True, X, y, meta, period, months,
                           args.train_months, args.label_rate, args.injection_rate,
                           args.injection_months, args.target_family, trigger,
                           scaler, seed=args.seed)
    print(f"[OK] [{time.time()-t0:.0f}s]  AUT(F1)={aut(df_attack['f1'].values):.4f}  "
         f"AUT(fam_recall_wm)={aut(df_attack['family_recall_watermarked'].values):.4f}  "
         f"injected={n_inj}")

    injection_window = set(months[args.train_months:args.train_months + args.injection_months])
    post_injection = [i for i, m in enumerate(df_clean["month"])
                      if pd.Period(m, freq="M") not in injection_window]
    depth, horizon = 0.0, 0
    for i in post_injection:
        c = df_clean["family_recall_watermarked"].iloc[i]
        a = df_attack["family_recall_watermarked"].iloc[i]
        if pd.isna(c) or pd.isna(a):
            continue
        gap = c - a
        depth = max(depth, gap)
        if gap > 0.15:
            horizon += 1
        else:
            break
    f1_gap = abs(aut(df_clean["f1"].values) - aut(df_attack["f1"].values))
    real_fam_gap = abs(aut(df_clean["family_recall_clean"].values) -
                       aut(df_attack["family_recall_clean"].values))

    print("\n" + "=" * 70)
    print(f"MLP classifier, strategy={args.strategy}, family={args.target_family}")
    print(f"  backdoor evasion depth:  {depth:.4f}")
    print(f"  persistence horizon:     {horizon} months")
    print(f"  overall F1 shift:        {f1_gap:.4f}")
    print(f"  real family recall gap:  {real_fam_gap:.4f}")
    print("=" * 70)

    df_clean.to_csv("./figs/attack_mlp_clean.csv", index=False)
    df_attack.to_csv("./figs/attack_mlp_attacked.csv", index=False)


if __name__ == "__main__":
    main()
