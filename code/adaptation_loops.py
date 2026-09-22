#!/usr/bin/env python3
"""
Three monthly retraining ("adaptation") strategies for the BODMAS stream,
mirroring BODMAS paper Fig. 1 / TESSERACT's delay-strategy comparison
(both cited in our related work).

Every strategy starts from the same seed classifier (trained on the first
--train-months months) and then, month by month:
  1. scores next month's unlabeled samples
  2. SELECTS a fraction `--label-rate` of them via the strategy's rule
  3. "labels" them (we cheat and use the ground truth -- in a real deployment
     this is the analyst-labeling step)
  4. adds them to the training pool and retrains

Strategies (selection rule sigma_t, see the project's formalisation notes):
  random        -- sigma_t(B) = uniform random subset of size label_rate*|B|
  uncertainty    -- sigma_t(B) = top-k by |0.5 - p(x)|  (closest to the
                    decision boundary; standard active-learning query)
  nonconformity  -- sigma_t(B) = top-k by distance from the training set's
                    per-class centroid in standardized feature space (a
                    cheap stand-in for BODMAS's own non-conformity score;
                    swap in a real conformal p-value later if you need the
                    paper-grade version)

This is the object your attacker (later) will poison: whichever sigma_t
you're using decides which samples get pulled into the next training round.

Usage:
    python adaptation_loops.py --data-dir ./data --train-months 6 --label-rate 0.05
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd


# ----------------------------------------------------------------- loading
# (same loader as bodmas_smoke_test.py -- kept standalone so this script has
#  no import dependency on it)

def load_bodmas(data_dir):
    with np.load(os.path.join(data_dir, "bodmas.npz"), allow_pickle=False) as z:
        X, y = z["X"].astype(np.float32), z["y"].astype(int)
    meta = pd.read_csv(os.path.join(data_dir, "bodmas_metadata.csv"))
    meta["_ts"] = pd.to_datetime(meta["timestamp"], errors="coerce")
    keep = meta["_ts"].notna().values
    X, y, meta = X[keep], y[keep], meta.loc[keep].reset_index(drop=True)
    order = np.argsort(meta["_ts"].values, kind="stable")
    X, y, meta = X[order], y[order], meta.iloc[order].reset_index(drop=True)
    # align to the malware collection window -- see bodmas_smoke_test.py's
    # January-spike investigation for why this matters
    malware_start = meta.loc[y == 1, "_ts"].min()
    trim = (meta["_ts"] >= malware_start).values
    X, y, meta = X[trim], y[trim], meta.loc[trim].reset_index(drop=True)
    period = meta["_ts"].dt.to_period("M")
    return X, y, meta, period


# ----------------------------------------------------------------- selectors

def select_random(rng, n_pool, k):
    k = min(k, n_pool)
    return rng.choice(n_pool, size=k, replace=False)


def select_uncertainty(proba, k):
    """Closest to 0.5 = most uncertain. proba is P(malware) for each point."""
    k = min(k, len(proba))
    uncertainty = np.abs(proba - 0.5)
    return np.argsort(uncertainty)[:k]


def select_nonconformity(Xpool, train_centroids, train_scale, k):
    """
    Cheap non-conformity proxy: standardized Euclidean distance from the
    pool point to its *nearest* per-class training centroid. Points far
    from both centroids are the most "alien" to what the model has seen --
    the same intuition as BODMAS's Fig. 1 non-conformity sampler.
    """
    Xs = (Xpool - train_scale["mean"]) / train_scale["std"]
    d0 = np.linalg.norm(Xs - train_centroids[0], axis=1)
    d1 = np.linalg.norm(Xs - train_centroids[1], axis=1)
    score = np.minimum(d0, d1)
    k = min(k, len(score))
    return np.argsort(-score)[:k]  # k largest distances = most "alien"


def compute_centroids(Xtr, ytr):
    mean, std = Xtr.mean(axis=0), Xtr.std(axis=0) + 1e-6
    Xs = (Xtr - mean) / std
    c0 = Xs[ytr == 0].mean(axis=0) if (ytr == 0).any() else np.zeros(Xtr.shape[1])
    c1 = Xs[ytr == 1].mean(axis=0) if (ytr == 1).any() else np.zeros(Xtr.shape[1])
    return (c0, c1), {"mean": mean, "std": std}


# ----------------------------------------------------------------- main loop

def run_strategy(strategy, X, y, period, months, train_months, label_rate, seed=0):
    import lightgbm as lgb
    from sklearn.metrics import f1_score, precision_score, recall_score

    rng = np.random.RandomState(seed)
    train_mask = period.isin(months[:train_months]).values
    train_idx = list(np.where(train_mask)[0])

    rows = []
    for m in months[train_months:]:
        Xtr, ytr = X[train_idx], y[train_idx]
        if len(np.unique(ytr)) < 2:
            continue  # can't train yet

        clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63,
                                 learning_rate=0.08, n_jobs=-1, verbose=-1)
        clf.fit(Xtr, ytr)

        # ---- evaluate on this month BEFORE it's added to the pool ----
        sel = (period == m).values
        if sel.sum() < 30 or len(np.unique(y[sel])) < 2:
            continue
        Xm_idx = np.where(sel)[0]
        Xm, ym = X[Xm_idx], y[Xm_idx]
        proba = clf.predict_proba(Xm)[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append(dict(month=str(m), n=len(ym), train_size=len(train_idx),
                         f1=f1_score(ym, pred, zero_division=0),
                         prec=precision_score(ym, pred, zero_division=0),
                         rec=recall_score(ym, pred, zero_division=0)))

        # ---- select a subset of this month to "label" and add to pool ----
        k = max(1, int(round(label_rate * len(ym))))
        if strategy == "random":
            pick_local = select_random(rng, len(ym), k)
        elif strategy == "uncertainty":
            pick_local = select_uncertainty(proba, k)
        elif strategy == "nonconformity":
            centroids, scale = compute_centroids(Xtr, ytr)
            pick_local = select_nonconformity(Xm, centroids, scale, k)
        else:
            sys.exit(f"[FAIL] unknown strategy {strategy!r}")

        train_idx.extend(Xm_idx[pick_local].tolist())

    return pd.DataFrame(rows)


def aut(scores):
    s = np.asarray([v for v in scores if not np.isnan(v)], dtype=float)
    if len(s) < 2:
        return float("nan")
    return float(np.mean((s[1:] + s[:-1]) / 2.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--train-months", type=int, default=6)
    ap.add_argument("--label-rate", type=float, default=0.05,
                    help="fraction of each month's samples added to the "
                         "training pool after retraining (e.g. 0.05 = 5%%)")
    ap.add_argument("--out", default="./figs/fig_adaptation_strategies.pdf")
    args = ap.parse_args()

    print("[..] loading BODMAS")
    X, y, meta, period = load_bodmas(args.data_dir)
    months = list(period.unique())
    print(f"[..] {len(months)} months available, training on first "
          f"{args.train_months}, adapting on the rest at "
          f"{args.label_rate:.0%}/month")

    results = {}
    for strategy in ("random", "uncertainty", "nonconformity"):
        t0 = time.time()
        df = run_strategy(strategy, X, y, period, months,
                          args.train_months, args.label_rate)
        results[strategy] = df
        a = aut(df["f1"].values)
        print(f"[OK] {strategy:14s} AUT(F1)={a:.4f}  "
              f"final_train_size={df['train_size'].iloc[-1] if len(df) else '-':>6}  "
              f"[{time.time()-t0:.0f}s]")

    # ---- figure: one line per strategy ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.linewidth": 0.6, "figure.dpi": 200})
    fig, ax = plt.subplots(figsize=(3.33, 2.2))
    colors = {"random": "#0072B2", "uncertainty": "#D55E00",
             "nonconformity": "#009E73"}  # Okabe-Ito
    markers = {"random": "o", "uncertainty": "s", "nonconformity": "^"}
    for strategy, df in results.items():
        if df.empty:
            continue
        ax.plot(range(len(df)), df["f1"], marker=markers[strategy], ms=3,
                lw=1.2, color=colors[strategy], label=strategy)
    step = max(1, len(next(iter(results.values()))) // 6)
    df0 = next(iter(results.values()))
    ax.set_xticks(range(0, len(df0), step))
    ax.set_xticklabels([df0["month"].iloc[i] for i in range(0, len(df0), step)],
                       rotation=45, ha="right", fontsize=6)
    ax.set_xlabel("test month"); ax.set_ylabel("F1")
    ax.set_ylim(0, 1.02); ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print(f"[OK] wrote {args.out}")

    for strategy, df in results.items():
        csv_path = f"./figs/adaptation_{strategy}.csv"
        df.to_csv(csv_path, index=False)
        print(f"[OK] wrote {csv_path}")


if __name__ == "__main__":
    main()
