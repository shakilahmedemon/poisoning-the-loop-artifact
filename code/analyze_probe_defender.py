#!/usr/bin/env python3
"""Analysis of figs/results_probe_defender.csv. Read-only.

For every (trigger, statistic) pair, over the post-window months (>= 2020-01):
  * month-level AUROC, attacked vs clean months, with a 95% cluster bootstrap over the
    (strategy, target, seed) run pairs (10,000 resamples);
  * run-level detection with a threshold that uses ONLY clean runs of OTHER seeds:
    theta = max over clean runs of the other seeds of (max post-window month statistic),
    so the false-positive rate on the held-out seed's clean runs is whatever it is
    (reported), instead of being tuned on the evaluation runs;
  * median and minimum of the attacked-run peak statistic, and the clean-run maximum.
"""
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score

import glob
d = pd.concat([pd.read_csv(f) for f in ["figs/results_probe_defender.csv"] + glob.glob("figs/results_probe_defender_s5_9.csv")],
              ignore_index=True)    # seeds 0-4 plus 5-9
d = d[d.month >= "2020-01"].copy()
TR = ["att", "k4", "k12", "boot", "unc", "cur", "rand"]
ST = ["fam", "pool", "pool5"]
d["run"] = d.strategy + "|" + d.target + "|" + d.seed.astype(str)
runs = sorted(d.run.unique())
rng = np.random.RandomState(0)
print("runs:", len(runs), " months/arm:", d.groupby("arm").size().to_dict())

def auc(df, col):
    x = df[col].values; y = (df.arm == "attacked").astype(int).values
    ok = ~np.isnan(x)
    return roc_auc_score(y[ok], x[ok]) if len(set(y[ok])) == 2 else np.nan

grp = {r: g for r, g in d.groupby("run")}
rows = []
for t in TR:
    for s in ST:
        col = f"{s}_{t}"
        a = auc(d, col)
        bs = []
        for _ in range(2000):                      # 2,000 resamples here; 10,000 for the paper figure
            pick = rng.choice(runs, size=len(runs), replace=True)
            bs.append(auc(pd.concat([grp[r] for r in pick]), col))
        lo, hi = np.nanpercentile(bs, [2.5, 97.5])
        peak = d.groupby(["run", "seed", "arm"])[col].max().reset_index()
        tp = fp = 0; n_a = n_c = 0
        for sd in sorted(d.seed.unique()):
            cal = peak[(peak.seed != sd) & (peak.arm == "clean")][col].max()
            te = peak[peak.seed == sd]
            tp += int((te[te.arm == "attacked"][col] > cal).sum()); n_a += int((te.arm == "attacked").sum())
            fp += int((te[te.arm == "clean"][col] > cal).sum()); n_c += int((te.arm == "clean").sum())
        pa = peak[peak.arm == "attacked"][col]; pc = peak[peak.arm == "clean"][col]
        rows.append(dict(trigger=t, stat=s, AUROC=round(a, 3), ci=f"[{lo:.3f},{hi:.3f}]",
                         run_TPR=f"{tp}/{n_a}", run_FPR=f"{fp}/{n_c}",
                         attacked_peak_median=round(pa.median(), 3), attacked_peak_min=round(pa.min(), 3),
                         clean_peak_max=round(pc.max(), 3)))
res = pd.DataFrame(rows)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)
print(res.to_string(index=False))
res.to_csv("figs/results_probe_defender_summary.csv", index=False)
ov = d[d.arm == "attacked"].cur_overlap
print("\n'cur' trigger overlap with attacker's 8 features (attacked runs, post-window): mean %.2f, max %d" % (ov.mean(), ov.max()))
