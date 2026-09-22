#!/usr/bin/env python3
"""
Native MLP attack. attack_smart_mlp.py reuses the LightGBM-derived trigger
against an MLP defender and finds it does not transfer; this builds the
trigger from scratch against the MLP itself, using shap.GradientExplainer
instead of shap.TreeExplainer. The question this answers that transfer
alone can't: is the vulnerability specific to tree ensembles, or does it
generalize to neural classifiers when the attacker has the right access to
build a trigger for one?

Stays in standardized feature space throughout (apply_trigger just
overwrites column indices with given values, so no new machinery is
needed). Realizability constraint (SAFE_FEATURE_INDICES) is imported
unchanged from attack_smart.py.

Usage:
    python attack_native_mlp.py --data-dir ./data --strategy random \
        --target-family wacatac --injection-rate 0.01
"""

import argparse
import time

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from attack_smart import load_bodmas, apply_trigger, aut, SAFE_FEATURE_INDICES
from mlp_classifier import train_mlp, predict_proba_mlp, build_trigger_mlp
from attack_smart_mlp import select_random, select_uncertainty


def run(strategy, attack, Xs, y, meta, period, months, train_months,
       label_rate, injection_rate, injection_months, target_family,
       trigger, seed=0):
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
        Xtr = np.vstack([Xs[train_idx]] + ([np.vstack(X_extra)] if X_extra else []))
        ytr = np.concatenate([y[train_idx]] + ([np.asarray(y_extra)] if y_extra else []))
        if len(np.unique(ytr)) < 2:
            continue
        clf = train_mlp(Xtr, ytr, epochs=30, seed=seed)

        sel = (period == m).values
        if sel.sum() < 30 or len(np.unique(y[sel])) < 2:
            continue
        Xm_idx = np.where(sel)[0]
        Xm, ym = Xs[Xm_idx], y[Xm_idx]
        proba = predict_proba_mlp(clf, Xm)
        pred = (proba >= 0.5).astype(int)

        row = dict(month=str(m), n=len(ym), train_size=len(ytr))
        row["f1"] = f1_score(ym, pred, zero_division=0)

        fam_mask = (meta.loc[Xm_idx, "family"].values == target_family) & (ym == 1)
        if fam_mask.sum() > 0:
            row["family_recall_clean"] = float(
                recall_score(ym[fam_mask], pred[fam_mask], zero_division=0))
            Xfam_wm = apply_trigger(Xm[fam_mask], trigger)
            proba_wm = predict_proba_mlp(clf, Xfam_wm)
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
    args = ap.parse_args()

    print("[..] loading BODMAS")
    X, y, meta, period = load_bodmas(args.data_dir)
    months = list(period.unique())

    train_mask = period.isin(months[:args.train_months]).values
    train_idx = np.where(train_mask)[0]

    scaler = StandardScaler().fit(X[train_idx])
    Xs = scaler.transform(X)  # whole dataset, standardized once

    print("[..] training seed MLP for native trigger construction")
    t0 = time.time()
    seed_model = train_mlp(Xs[train_idx], y[train_idx], epochs=30, seed=args.seed)
    print(f"[OK] seed MLP trained [{time.time()-t0:.0f}s]")

    trigger = build_trigger_mlp(seed_model, Xs[train_idx], y[train_idx],
                                args.trigger_size, SAFE_FEATURE_INDICES,
                                seed=args.seed)

    print(f"\n[..] running BASELINE (no attack), native-MLP trigger, "
         f"strategy={args.strategy}")
    t0 = time.time()
    df_clean, _ = run(args.strategy, False, Xs, y, meta, period, months,
                      args.train_months, args.label_rate, args.injection_rate,
                      args.injection_months, args.target_family, trigger,
                      seed=args.seed)
    print(f"[OK] [{time.time()-t0:.0f}s]  AUT(F1)={aut(df_clean['f1'].values):.4f}  "
         f"AUT(fam_recall_wm)={aut(df_clean['family_recall_watermarked'].values):.4f}")

    print(f"\n[..] running ATTACKED, native-MLP trigger, strategy={args.strategy}")
    t0 = time.time()
    df_attack, n_inj = run(args.strategy, True, Xs, y, meta, period, months,
                           args.train_months, args.label_rate, args.injection_rate,
                           args.injection_months, args.target_family, trigger,
                           seed=args.seed)
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
    print(f"NATIVE MLP trigger, strategy={args.strategy}, family={args.target_family}")
    print(f"  backdoor evasion depth:  {depth:.4f}")
    print(f"  persistence horizon:     {horizon} months")
    print(f"  overall F1 shift:        {f1_gap:.4f}")
    print(f"  real family recall gap:  {real_fam_gap:.4f}")
    print("=" * 70)

    df_clean.to_csv("./figs/attack_native_mlp_clean.csv", index=False)
    df_attack.to_csv("./figs/attack_native_mlp_attacked.csv", index=False)


if __name__ == "__main__":
    main()
