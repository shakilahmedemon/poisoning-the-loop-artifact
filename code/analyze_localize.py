#!/usr/bin/env python3
"""Localization of the monthly check over seeds 0-9 (attacker's trigger). Read-only; writes figs/results_localize_summary.csv.
In flagged attacked post-window months, how often does the family with the largest gap equal the true target?
Cross-check for seeds 5-9: the per-family maximum must equal fam_att stored by run_probe_defender.py."""
import numpy as np, pandas as pd

d = pd.read_csv("figs/results_localize_s5_9.csv")
p = d[d.month >= "2020-01"]
top = p.sort_values("gap", ascending=False).groupby(["strategy", "target", "seed", "month"], as_index=False).first()
ref = pd.read_csv("figs/results_probe_defender_s5_9.csv")
ref = ref[(ref.arm == "attacked") & (ref.month >= "2020-01")][["strategy", "target", "seed", "month", "fam_att"]]
m = top.merge(ref, on=["strategy", "target", "seed", "month"])
print("cross-check months", len(m), "max |gap - fam_att|", float(np.abs(m.gap - m.fam_att).max()))
rows = []
for th in (0.2, 0.3, 0.4, 0.5):
    f = top[top.gap > th]
    rows.append(dict(seeds="5-9", theta=th, flagged=len(f), target_is_top=int((f.family == f.target).sum())))
S = pd.DataFrame(rows)
S["share"] = S.target_is_top / S.flagged
S.to_csv("figs/results_localize_summary.csv", index=False)
print(S.round(3).to_string(index=False))
print("families", p.family.nunique(), "chance about", round(1 / p.family.nunique(), 3))
print("median rank of the true target among families, flagged months at theta=0.3:")
p2 = p.copy(); p2["rank"] = p2.groupby(["strategy", "target", "seed", "month"]).gap.rank(ascending=False, method="min")
fl = top[top.gap > 0.3][["strategy", "target", "seed", "month"]]
tr = p2[p2.family == p2.target].merge(fl, on=["strategy", "target", "seed", "month"])
print(len(tr), "months, median rank", tr["rank"].median(), "by target", tr.groupby("target")["rank"].median().to_dict())
