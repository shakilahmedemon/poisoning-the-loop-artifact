#!/usr/bin/env python3
"""
Diagnose the January 2020 precision collapse seen when training on a single
month (Aug 2019) and testing month-by-month through Sep 2020.

Precision crashed to 0.47 while recall stayed at 0.998 -- that means the
classifier trained on Aug-2019 malware/benign started flagging a lot of
JANUARY BENIGN software as malicious. This script finds out why:

  1. Volume check: was there a burst of benign or malicious samples in
     Jan 2020 relative to neighboring months?
  2. Family check: did a new/rare malware family dominate the malicious
     side that month? (the BODMAS paper found "sfone" doing exactly this)
  3. Feature-drift check: which of the 2381 features moved the most
     between Aug-2019 benign and Jan-2020 benign (the training data the
     classifier trusted vs. the data it was tested on)?
"""

import numpy as np
import pandas as pd

DATA_DIR = "./data"

print("[..] loading data (same logic as the smoke test)")
with np.load(f"{DATA_DIR}/bodmas.npz", allow_pickle=False) as z:
    X, y = z["X"].astype(np.float32), z["y"].astype(int)

meta = pd.read_csv(f"{DATA_DIR}/bodmas_metadata.csv")
meta["_ts"] = pd.to_datetime(meta["timestamp"], errors="coerce")
keep = meta["_ts"].notna().values
X, y, meta = X[keep], y[keep], meta.loc[keep].reset_index(drop=True)
order = np.argsort(meta["_ts"].values, kind="stable")
X, y, meta = X[order], y[order], meta.iloc[order].reset_index(drop=True)

malware_start = meta.loc[y == 1, "_ts"].min()
trim = (meta["_ts"] >= malware_start).values
X, y, meta = X[trim], y[trim], meta.loc[trim].reset_index(drop=True)
period = meta["_ts"].dt.to_period("M")

# 1. volume
print("\n=== 1. Sample volume per month (benign vs malicious) ===")
vol = pd.DataFrame({"period": period, "y": y}).groupby(["period", "y"]).size().unstack(fill_value=0)
vol.columns = ["benign", "malicious"]
print(vol.to_string())

# 2. family
print("\n=== 2. Malware family composition, Aug-2019 (train) vs Jan-2020 (test) ===")
train_mask = (period == period.iloc[0]) & (y == 1)
jan_mask = (period == "2020-01") & (y == 1)
train_families = meta.loc[train_mask, "family"].value_counts()
jan_families = meta.loc[jan_mask, "family"].value_counts()
print(f"Aug-2019 malware: {train_mask.sum()} samples, {train_families.nunique()} families")
print("  top 5:", dict(train_families.head(5)))
print(f"Jan-2020 malware: {jan_mask.sum()} samples, {jan_families.nunique()} families")
print("  top 5:", dict(jan_families.head(5)))
new_families = set(jan_families.index) - set(train_families.index)
new_family_count = jan_families[jan_families.index.isin(new_families)].sum()
print(f"  families in Jan-2020 NEVER seen in Aug-2019 training: {len(new_families)} "
      f"families, {new_family_count} samples ({100*new_family_count/jan_mask.sum():.1f}% of Jan malware)")

# 3. feature drift on BENIGN side
# Precision collapsed => false positives => BENIGN samples misclassified.
# Compare Aug-2019 benign (what the model trusts) to Jan-2020 benign
# (what it was tested on) feature-by-feature.
print("\n=== 3. Feature drift: Aug-2019 benign vs Jan-2020 benign ===")
train_benign_mask = (period == period.iloc[0]) & (y == 0)
jan_benign_mask = (period == "2020-01") & (y == 0)
print(f"Aug-2019 benign: {train_benign_mask.sum()} samples")
print(f"Jan-2020 benign: {jan_benign_mask.sum()} samples")

Xtb = X[train_benign_mask.values]
Xjb = X[jan_benign_mask.values]
mean_tb, mean_jb = Xtb.mean(axis=0), Xjb.mean(axis=0)
std_tb = Xtb.std(axis=0) + 1e-6
# standardized shift per feature (like a z-score of the mean difference)
shift = np.abs(mean_jb - mean_tb) / std_tb
top_idx = np.argsort(-shift)[:15]
print("Top 15 features by standardized mean shift (index only -- BODMAS's .npz "
      "doesn't ship feature names, cross-reference against the EMBER feature "
      "spec if you need the human-readable name):")
for i in top_idx:
    print(f"  feat[{i:4d}]  aug19_mean={mean_tb[i]:12.2f}  jan20_mean={mean_jb[i]:12.2f}  "
          f"|shift|={shift[i]:8.3f}")

print("\n=== Verdict ===")
if new_family_count / max(jan_mask.sum(), 1) > 0.3:
    print("A large share of Jan-2020 malware came from families absent in training.")
    print("But that explains RECALL drops, not the PRECISION drop we actually saw.")
print("Precision crashed => false positives on the BENIGN side. Check section 3:")
print("if a handful of features shifted hard, Jan-2020's benign population looks")
print("statistically different from Aug-2019's -- i.e. genuine covariate drift in")
print("goodware, not a malware family burst. It shows whether the benign or the")
print("malware stream changed more.")
