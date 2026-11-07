#!/usr/bin/env python3
"""Feasibility check on BODMAS: load the data, train a LightGBM baseline on the
first --train-months months and measure the monthly decay on later months.

Prints a per-month table and the AUT, and writes figs/fig3_baseline.pdf.

Data: BODMAS feature vectors and metadata from
https://whyisyoung.github.io/BODMAS/, placed in ./data/.

Usage:
    python bodmas_smoke_test.py --data-dir ./data --train-months 6
"""

import argparse
import glob
import os
import sys
import time

import numpy as np
import pandas as pd


# loading

def find_one(data_dir, patterns, kind):
    for pat in patterns:
        hits = sorted(glob.glob(os.path.join(data_dir, pat)))
        if hits:
            return hits[0]
    sys.exit("[FAIL] No %s found in %r matching %s.\n"
             "       Download from https://whyisyoung.github.io/BODMAS/"
             % (kind, data_dir, patterns))


def load_features(npz_path):
    """BODMAS ships an .npz; key names have varied across releases, so sniff."""
    print("[..] loading features from %s" % os.path.basename(npz_path))
    t0 = time.time()
    # allow_pickle stays False on purpose: this .npz is a third-party download,
    # and pickle deserialisation is arbitrary code execution. BODMAS stores
    # plain numeric arrays, so this should load fine. If it raises, inspect the
    # archive before you consider relaxing this.
    with np.load(npz_path, allow_pickle=False) as z:
        keys = list(z.keys())
        print("     arrays present: %s" % keys)
        # Feature matrix = the only 2-D array. Labels = a 1-D array if present.
        X = y = None
        for k in keys:
            arr = z[k]
            if arr.ndim == 2 and X is None:
                X = arr
            elif arr.ndim == 1 and y is None:
                y = arr
        if X is None:
            sys.exit("[FAIL] No 2-D array in %s. Keys: %s" % (npz_path, keys))
        X = np.ascontiguousarray(X, dtype=np.float32)
        y = None if y is None else np.asarray(y).ravel()

    gb = X.nbytes / 1024 ** 3
    print("     X = %s float32  (%.2f GB)  [%.1fs]" % (X.shape, gb, time.time() - t0))
    if gb > 4.0:
        print("[!!] %.2f GB is large for a 16 GB machine. LightGBM needs roughly "
              "another %.1f GB to bin. If you OOM, use --subsample." % (gb, gb))
    return X, y


def load_metadata(csv_path, n_rows):
    print("[..] loading metadata from %s" % os.path.basename(csv_path))
    meta = pd.read_csv(csv_path)
    print("     columns: %s" % list(meta.columns))
    if len(meta) != n_rows:
        print("[!!] metadata rows (%d) != feature rows (%d). Check you paired "
              "the right two files." % (len(meta), n_rows))

    def pick(cands, what, required=True):
        for c in cands:                                   # exact match first
            for col in meta.columns:
                if col.strip().lower() == c:
                    return col
        for col in meta.columns:                          # then substring
            for c in cands:
                if c in col.strip().lower():
                    return col
        if required:
            sys.exit("[FAIL] No %s column found. Columns: %s"
                     % (what, list(meta.columns)))
        return None

    ts_col = pick(["timestamp", "time", "date", "first_seen"], "timestamp")
    label_col = pick(["label", "y", "malware", "is_malware"], "label", required=False)
    family_col = pick(["family", "category"], "family", required=False)

    meta["_ts"] = pd.to_datetime(meta[ts_col], errors="coerce")
    bad = int(meta["_ts"].isna().sum())
    if bad:
        print("[!!] %d unparseable timestamps (%.1f%%) -> dropped"
              % (bad, 100.0 * bad / len(meta)))
    print("     timestamp column : %r" % ts_col)
    print("     label column     : %r" % label_col)
    print("     family column    : %r" % family_col)
    return meta, label_col, family_col


# metrics

def aut(scores):
    """Area Under Time (Pendlebury et al., USENIX Security 2019): the
    trapezoidal mean of a metric over N periods, in [0, 1].

        AUT(f, N) = 1/(N-1) * sum_{k=1..N-1} (f_{k+1} + f_k) / 2
    """
    s = np.asarray([v for v in scores if not np.isnan(v)], dtype=float)
    if len(s) < 2:
        return float("nan")
    return float(np.mean((s[1:] + s[:-1]) / 2.0))


# main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--train-months", type=int, default=6,
                    help="months of history to train on; the rest is the test stream")
    ap.add_argument("--subsample", type=int, default=0,
                    help="cap training rows if you hit memory pressure")
    ap.add_argument("--out", default="./figs/fig3_baseline.pdf")
    ap.add_argument("--no-align-to-malware", action="store_true",
                    help="disable trimming the stream to the malware collection "
                         "window (BODMAS benign samples go back to 2007; malware "
                         "starts 08/2019 -- without trimming, early months are "
                         "single-class and the whole run is degenerate)")
    args = ap.parse_args()

    npz = find_one(args.data_dir, ["*.npz"], "feature .npz")
    csv = find_one(args.data_dir, ["*metadata*.csv", "*.csv"], "metadata .csv")

    X, y_npz = load_features(npz)
    meta, label_col, family_col = load_metadata(csv, X.shape[0])

    # Labels: prefer the npz, fall back to metadata.
    if y_npz is not None and len(y_npz) == X.shape[0]:
        y = y_npz.astype(int)
        print("[..] labels taken from the .npz")
    elif label_col is not None:
        y = pd.to_numeric(meta[label_col], errors="coerce").fillna(1).astype(int).values
        print("[..] labels taken from metadata column %r" % label_col)
    else:
        sys.exit("[FAIL] No labels anywhere. Check you have the full BODMAS release.")
    print("     class balance: %s  (malware rate %.3f)" % (np.bincount(y), y.mean()))

    # temporal ordering: never shuffle
    keep = meta["_ts"].notna().values
    X, y, meta = X[keep], y[keep], meta.loc[keep].reset_index(drop=True)
    order = np.argsort(meta["_ts"].values, kind="stable")
    X, y, meta = X[order], y[order], meta.iloc[order].reset_index(drop=True)

    # BODMAS benign samples are backdated to 2007-01-01; malware collection
    # only starts 2019-08-29 (see the paper, Sec. III). Left untrimmed, the
    # early "months" are single-class (benign only) and every downstream
    # metric degenerates to zero. Trim to the malware collection window,
    # matching the paper's own Table II (which starts at Test-10/19).
    if not args.no_align_to_malware and (y == 1).any():
        malware_start = meta.loc[y == 1, "_ts"].min()
        trim = (meta["_ts"] >= malware_start).values
        n_before = len(meta)
        X, y, meta = X[trim], y[trim], meta.loc[trim].reset_index(drop=True)
        print("[..] trimmed to malware collection window (>= %s): %d -> %d rows"
              % (malware_start.date(), n_before, len(meta)))

    period = meta["_ts"].dt.to_period("M")
    months = list(period.unique())
    print("[..] %d months, %s .. %s" % (len(months), months[0], months[-1]))
    if len(months) < args.train_months + 3:
        sys.exit("[FAIL] Only %d months. Need >= %d."
                 % (len(months), args.train_months + 3))

    train_mask = period.isin(months[:args.train_months]).values
    Xtr, ytr = X[train_mask], y[train_mask]
    if args.subsample and len(ytr) > args.subsample:
        idx = np.random.RandomState(0).choice(len(ytr), args.subsample, replace=False)
        Xtr, ytr = Xtr[idx], ytr[idx]
        print("[..] subsampled training set to %d rows" % args.subsample)
    print("[..] train: %d rows from %s..%s"
          % (Xtr.shape[0], months[0], months[args.train_months - 1]))

    try:
        import lightgbm as lgb
    except ImportError:
        sys.exit("[FAIL] pip install lightgbm")

    t0 = time.time()
    clf = lgb.LGBMClassifier(n_estimators=500, num_leaves=64, learning_rate=0.05,
                             n_jobs=-1, verbose=-1)
    clf.fit(Xtr, ytr)
    print("[OK] trained in %.1fs on CPU" % (time.time() - t0))

    # per-month decay on the held-out future
    from sklearn.metrics import f1_score, precision_score, recall_score

    rows = []
    for m in months[args.train_months:]:
        sel = (period == m).values
        if sel.sum() < 30 or len(np.unique(y[sel])) < 2:
            continue
        pred = clf.predict(X[sel])
        rows.append(dict(month=str(m), n=int(sel.sum()),
                         f1=f1_score(y[sel], pred, zero_division=0),
                         prec=precision_score(y[sel], pred, zero_division=0),
                         rec=recall_score(y[sel], pred, zero_division=0)))

    if not rows:
        sys.exit("[FAIL] No evaluable test months.")
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 100)
    print("\n" + df.to_string(index=False, float_format=lambda v: "%.4f" % v))

    a = aut(df["f1"].values)
    print("\n  AUT(F1, %d months) = %.4f" % (len(df), a))
    print("  first month F1 = %.4f   last month F1 = %.4f   drop = %+.4f"
          % (df["f1"].iloc[0], df["f1"].iloc[-1],
             df["f1"].iloc[0] - df["f1"].iloc[-1]))

    # Figure 3, clean condition
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.linewidth": 0.6, "figure.dpi": 200})
    fig, ax = plt.subplots(figsize=(3.33, 2.1))            # exact column width
    ax.plot(range(len(df)), df["f1"], marker="o", ms=3, lw=1.2,
            color="#0072B2", label="F1 (no attacker)")     # Okabe-Ito blue
    step = max(1, len(df) // 6)
    ax.set_xticks(range(0, len(df), step))
    ax.set_xticklabels([df["month"].iloc[i] for i in range(0, len(df), step)],
                       rotation=45, ha="right", fontsize=6)
    ax.set_xlabel("test month")
    ax.set_ylabel("F1")
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, fontsize=7)
    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print("  wrote %s" % args.out)

    # the verdict
    drop = df["f1"].iloc[0] - df["f1"].iloc[-1]
    print("\n" + "=" * 68)
    if drop > 0.05 or a < 0.92:
        print("GO. There is real temporal decay over this window, so the")
        print("retraining loop has drift to work with.")
    else:
        print("CAUTION. Little decay over this window. Try --train-months 3, or")
        print("restrict to malware families that appear late in the stream. If")
        print("drift stays flat, BODMAS's window is too short and a longer-horizon")
        print("corpus is needed (see LAMDA, arXiv 2505.18551).")
    print("=" * 68)


if __name__ == "__main__":
    main()
