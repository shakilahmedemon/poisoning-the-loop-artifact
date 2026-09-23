#!/usr/bin/env python3
"""
Naive dirty-label attacker: baseline for whether poisoning a small slice
of each month's retraining pool creates a lasting blind spot for one
malware family, while the detector's overall F1 barely moves.

Simplest possible threat model (the smart, SHAP-guided version lives in
attack_smart.py): each month, before the defender retrains, the attacker
injects a small number of real malware samples from a target family into
the pool with their label flipped to benign. Features are untouched, only
the label lies, so a label audit catches every poisoned point instantly.
This is meant as a floor: if even this crude attack moves the needle, the
clean-label version should do more damage.

Injection happens only during the first --injection-months months, then
stops; what we care about is whether the blind spot persists afterward.

Metrics:
  - overall F1 each month: stealth check, should barely move
  - recall restricted to the target family: the attack's actual effect
  - persistence horizon: how many months after injection stops the family
    recall stays below a defender-alarming threshold

Usage:
    python attack_naive.py --data-dir ./data --train-months 1 \
        --strategy random --injection-rate 0.05 --injection-months 4
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd


# ----------------------------------------------------------------- loading

def load_bodmas(data_dir):
    with np.load(os.path.join(data_dir, "bodmas.npz"), allow_pickle=False) as z:
        X, y = z["X"].astype(np.float32), z["y"].astype(int)
    meta = pd.read_csv(os.path.join(data_dir, "bodmas_metadata.csv"))
    meta["_ts"] = pd.to_datetime(meta["timestamp"], errors="coerce")
    meta["family"] = meta["family"].fillna("")
    keep = meta["_ts"].notna().values
    X, y, meta = X[keep], y[keep], meta.loc[keep].reset_index(drop=True)
    order = np.argsort(meta["_ts"].values, kind="stable")
    X, y, meta = X[order], y[order], meta.iloc[order].reset_index(drop=True)
    malware_start = meta.loc[y == 1, "_ts"].min()
    trim = (meta["_ts"] >= malware_start).values
    X, y, meta = X[trim], y[trim], meta.loc[trim].reset_index(drop=True)
    period = meta["_ts"].dt.to_period("M")
    return X, y, meta, period


def pick_target_family(meta, y, period, months, train_months):
    """Pick the malware family with the most samples across the
    post-training test months, so there's enough signal to measure recall
    on every month without collapsing to 0/0."""
    test_mask = (~period.isin(months[:train_months])).values & (y == 1)
    counts = meta.loc[test_mask, "family"].value_counts()
    counts = counts[counts.index != ""]
    if counts.empty:
        sys.exit("[FAIL] no named malware families in the test period")
    target = counts.index[0]
    print(f"[..] target family: {target!r} ({counts.iloc[0]} samples across "
          f"the test period)")
    return target


# ----------------------------------------------------------------- selector
# only 'random' and 'uncertainty' here; nonconformity is left for later,
# this is just the floor check

def select_random(rng, n_pool, k):
    k = min(k, n_pool)
    return rng.choice(n_pool, size=k, replace=False)


def select_uncertainty(proba, k):
    k = min(k, len(proba))
    return np.argsort(np.abs(proba - 0.5))[:k]


# ----------------------------------------------------------------- the loop

def run(strategy, attack, X, y, meta, period, months, train_months,
       label_rate, injection_rate, injection_months, target_family, seed=0):
    import lightgbm as lgb
    from sklearn.metrics import f1_score, precision_score, recall_score

    rng = np.random.RandomState(seed)
    train_mask = period.isin(months[:train_months]).values
    train_idx = list(np.where(train_mask)[0])
    # parallel label array: normally == y[idx], but a poisoned entry is
    # stored with its flipped label. Keeps ground truth (y) untouched for
    # honest evaluation everywhere else.
    train_labels = list(y[train_idx])

    injection_window = set(months[train_months:train_months + injection_months]) \
        if attack else set()
    n_injected_total = 0
    rows = []

    for m_i, m in enumerate(months[train_months:]):
        Xtr = X[train_idx]
        ytr = np.asarray(train_labels)
        if len(np.unique(ytr)) < 2:
            continue

        clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63,
                                 learning_rate=0.08, n_jobs=-1, verbose=-1)
        clf.fit(Xtr, ytr)

        sel = (period == m).values
        if sel.sum() < 30 or len(np.unique(y[sel])) < 2:
            continue
        Xm_idx = np.where(sel)[0]
        Xm, ym = X[Xm_idx], y[Xm_idx]
        proba = clf.predict_proba(Xm)[:, 1]
        pred = (proba >= 0.5).astype(int)

        # ---- overall (stealth) metrics ----
        row = dict(month=str(m), n=len(ym), train_size=len(train_idx),
                   f1=f1_score(ym, pred, zero_division=0),
                   prec=precision_score(ym, pred, zero_division=0),
                   rec=recall_score(ym, pred, zero_division=0))

        # ---- target-family recall (blind spot depth) ----
        fam_mask = (meta.loc[Xm_idx, "family"].values == target_family) & (ym == 1)
        if fam_mask.sum() > 0:
            row["family_n"] = int(fam_mask.sum())
            row["family_recall"] = float(recall_score(ym[fam_mask], pred[fam_mask],
                                                       zero_division=0))
        else:
            row["family_n"] = 0
            row["family_recall"] = float("nan")
        rows.append(row)

        # ---- normal selection: pick k samples to label honestly ----
        k = max(1, int(round(label_rate * len(ym))))
        if strategy == "random":
            pick_local = select_random(rng, len(ym), k)
        elif strategy == "uncertainty":
            pick_local = select_uncertainty(proba, k)
        else:
            sys.exit(f"[FAIL] unknown strategy {strategy!r}")
        train_idx.extend(Xm_idx[pick_local].tolist())
        train_labels.extend(y[Xm_idx[pick_local]].tolist())

        # ---- the attack: inject flipped-label target-family malware ----
        if m in injection_window:
            fam_idx_local = np.where((meta.loc[Xm_idx, "family"].values == target_family)
                                     & (ym == 1))[0]
            n_inject = max(1, int(round(injection_rate * len(ym)))) \
                if len(fam_idx_local) else 0
            n_inject = min(n_inject, len(fam_idx_local))
            if n_inject > 0:
                inject_local = rng.choice(fam_idx_local, size=n_inject, replace=False)
                train_idx.extend(Xm_idx[inject_local].tolist())
                train_labels.extend([0] * n_inject)  # the lie: label as benign
                n_injected_total += n_inject

    df = pd.DataFrame(rows)
    return df, n_injected_total


def aut(scores):
    s = np.asarray([v for v in scores if not np.isnan(v)], dtype=float)
    if len(s) < 2:
        return float("nan")
    return float(np.mean((s[1:] + s[:-1]) / 2.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--train-months", type=int, default=1)
    ap.add_argument("--strategy", choices=["random", "uncertainty"], default="random")
    ap.add_argument("--label-rate", type=float, default=0.05)
    ap.add_argument("--injection-rate", type=float, default=0.05,
                    help="fraction of each injection month's samples "
                         "poisoned (as a fraction of that month's total "
                         "sample count, capped by family availability)")
    ap.add_argument("--injection-months", type=int, default=4,
                    help="how many months, starting right after the seed "
                         "training window, the attacker injects for")
    ap.add_argument("--out", default="./figs/fig_attack_naive.pdf")
    ap.add_argument("--target-family", default=None,
                    help="force a specific malware family instead of "
                         "auto-picking the most populous one overall. "
                         "Matters for a rate sweep, since the auto-pick "
                         "looks at total test-period volume, not volume "
                         "during the injection window specifically")
    args = ap.parse_args()

    print("[..] loading BODMAS")
    X, y, meta, period = load_bodmas(args.data_dir)
    months = list(period.unique())
    if args.target_family:
        target_family = args.target_family
        n_in_test = int((meta.loc[(~period.isin(months[:args.train_months])).values
                                  & (y == 1), "family"] == target_family).sum())
        print(f"[..] target family (forced): {target_family!r} "
              f"({n_in_test} samples across the test period)")
    else:
        target_family = pick_target_family(meta, y, period, months, args.train_months)

    print(f"\n[..] running BASELINE (no attack), strategy={args.strategy}")
    t0 = time.time()
    df_clean, _ = run(args.strategy, False, X, y, meta, period, months,
                      args.train_months, args.label_rate, args.injection_rate,
                      args.injection_months, target_family)
    print(f"[OK] baseline done [{time.time()-t0:.0f}s]  "
          f"AUT(F1)={aut(df_clean['f1'].values):.4f}  "
          f"AUT(family_recall)={aut(df_clean['family_recall'].values):.4f}")

    print(f"\n[..] running ATTACKED (inject {args.injection_rate:.0%}/month for "
          f"{args.injection_months} months), strategy={args.strategy}")
    t0 = time.time()
    df_attack, n_inj = run(args.strategy, True, X, y, meta, period, months,
                           args.train_months, args.label_rate, args.injection_rate,
                           args.injection_months, target_family)
    print(f"[OK] attacked done [{time.time()-t0:.0f}s]  "
          f"AUT(F1)={aut(df_attack['f1'].values):.4f}  "
          f"AUT(family_recall)={aut(df_attack['family_recall'].values):.4f}  "
          f"total poisoned points injected={n_inj}")

    # ---- the verdict, printed as a table ----
    print("\n" + "=" * 78)
    print(f"{'month':>9} | {'F1 clean':>9} {'F1 atk':>7} | "
          f"{'fam_rec clean':>13} {'fam_rec atk':>11} | injected-window?")
    print("-" * 78)
    injection_window = set(months[args.train_months:
                                  args.train_months + args.injection_months])
    for i in range(min(len(df_clean), len(df_attack))):
        m = df_clean["month"].iloc[i]
        in_window = "yes" if pd.Period(m, freq="M") in injection_window else ""
        print(f"{m:>9} | {df_clean['f1'].iloc[i]:9.4f} {df_attack['f1'].iloc[i]:7.4f} | "
              f"{df_clean['family_recall'].iloc[i]:13.4f} "
              f"{df_attack['family_recall'].iloc[i]:11.4f} | {in_window}")
    print("=" * 78)

    # ---- persistence horizon: months after injection stops where the
    #      attacked family recall stays below the baseline's ----
    post_injection = [i for i, m in enumerate(df_clean["month"])
                      if pd.Period(m, freq="M") not in injection_window]
    depth = 0.0
    horizon = 0
    for i in post_injection:
        c, a = df_clean["family_recall"].iloc[i], df_attack["family_recall"].iloc[i]
        if pd.isna(c) or pd.isna(a):
            continue
        gap = c - a
        depth = max(depth, gap)
        if gap > 0.15:  # attacked recall notably worse than clean
            horizon += 1
        else:
            break  # recovered, stop counting the contiguous horizon
    print(f"\nmax family-recall gap (clean - attacked) after injection stopped: "
          f"{depth:.4f}")
    print(f"persistence horizon (consecutive post-injection months with gap > 0.15): "
          f"{horizon}")
    f1_gap = abs(aut(df_clean["f1"].values) - aut(df_attack["f1"].values))
    print(f"overall AUT(F1) shift (stealth check, want this SMALL): {f1_gap:.4f}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.linewidth": 0.6, "figure.dpi": 200})
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.15))
    step = max(1, len(df_clean) // 6)

    ax = axes[0]
    ax.plot(range(len(df_clean)), df_clean["f1"], marker="o", ms=3, lw=1.1,
            color="#0072B2", label="clean")
    ax.plot(range(len(df_attack)), df_attack["f1"], marker="s", ms=3, lw=1.1,
            color="#D55E00", label="attacked")
    ax.axvspan(-0.5, args.injection_months - 0.5, color="#999999", alpha=0.15)
    ax.set_ylabel("overall F1"); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.set_title("stealth: overall F1 barely moves", fontsize=7)
    ax.set_xticks(range(0, len(df_clean), step))
    ax.set_xticklabels([df_clean["month"].iloc[i] for i in range(0, len(df_clean), step)],
                       rotation=45, ha="right", fontsize=6)
    ax.set_xlabel("test month")

    ax = axes[1]
    ax.plot(range(len(df_clean)), df_clean["family_recall"], marker="o", ms=3,
            lw=1.1, color="#0072B2", label="clean")
    ax.plot(range(len(df_attack)), df_attack["family_recall"], marker="s", ms=3,
            lw=1.1, color="#D55E00", label="attacked")
    ax.axvspan(-0.5, args.injection_months - 0.5, color="#999999", alpha=0.15)
    ax.set_ylabel(f"recall on {target_family}"); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_title("blind spot: target-family recall", fontsize=7)
    ax.set_xticks(range(0, len(df_clean), step))
    ax.set_xticklabels([df_clean["month"].iloc[i] for i in range(0, len(df_clean), step)],
                       rotation=45, ha="right", fontsize=6)
    ax.set_xlabel("test month")

    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print(f"\n[OK] wrote {args.out}")

    df_clean.to_csv("./figs/attack_naive_clean.csv", index=False)
    df_attack.to_csv("./figs/attack_naive_attacked.csv", index=False)

    print("\n" + "=" * 78)
    if depth > 0.15 and f1_gap < 0.02:
        print("SIGNAL: a naive dirty-label attack already carves a family-specific")
        print("blind spot while overall F1 stays flat. Proceed to Severi-style")
        print("feature-guided injection to see how much stronger this gets.")
    elif depth > 0.15:
        print("PARTIAL SIGNAL: the family recall drops, but overall F1 also moves")
        print("-- the attack is NOT stealthy yet at this rate. Try a lower")
        print("--injection-rate, or check if this family dominates the month.")
    else:
        print("NO SIGNAL at this rate/window. Try a higher --injection-rate, more")
        print("--injection-months, the 'uncertainty' strategy, or a rarer target")
        print("family (harder for the model to relearn from unpoisoned signal).")
    print("=" * 78)


if __name__ == "__main__":
    main()
