#!/usr/bin/env python3
"""Localization over all 10 seeds: seeds 0-4 from figs/results_backdoor_probe.csv, seeds 5-9 from figs/results_localize_s5_9.csv.
Same rule as the monthly check: families with at least 10 samples that month; post-window months only."""
import re
import pandas as pd

d = pd.read_csv("figs/results_backdoor_probe.csv")
d = d[(d.arm == "attacked") & (d.month >= "2020-01")]
fams = sorted({m.group(1) for c in d.columns for m in [re.match(r"probe_(.+)_wm$", c)] if m})
rows = []
for r in d.to_dict("records"):
    for f in fams:
        n = r[f"probe_{f}_n"]
        if n >= 10 and r[f"probe_{f}_rec"] == r[f"probe_{f}_rec"]:
            rows.append(dict(strategy=r["strategy"], target=r["target"], seed=r["seed"], month=r["month"], family=f, gap=r[f"probe_{f}_rec"] - r[f"probe_{f}_wm"]))
a = pd.DataFrame(rows)
b = pd.read_csv("figs/results_localize_s5_9.csv")
b = b[b.month >= "2020-01"]
G = ["strategy", "target", "seed", "month"]
out = []
for name, df in [("seeds 0-4", a), ("seeds 5-9", b), ("seeds 0-9", pd.concat([a, b[a.columns]]))]:
    df = df.assign(mx=df.groupby(G).gap.transform("max"))
    tied = df[df.gap >= df.mx - 1e-12]
    print(name, "attacked post-window months", tied.groupby(G).ngroups, "with a tied top family", int((tied.groupby(G).size() > 1).sum()))
    for th in (0.2, 0.3, 0.4, 0.5):
        f = tied[tied.mx > th]
        n = f.groupby(G).ngroups
        # A tie counts in the target's favor.
        hit = int((f.family == f.target).groupby([f[c] for c in G]).any().sum())
        out.append(dict(seeds=name, theta=th, flagged=n, target_among_largest=hit, share=hit / n))
        print(f"  theta={th}: flagged {n}, target among largest gap {hit} ({100 * hit / n:.1f}%)")
pd.DataFrame(out).to_csv("figs/results_localize_all_seeds.csv", index=False)
print("families per month, median:", int(pd.concat([a, b[a.columns]]).groupby(G).family.nunique().median()))
