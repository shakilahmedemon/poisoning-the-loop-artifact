#!/usr/bin/env python3
"""Per-seed mean post-window gap for the BODMAS seed-pool figure. Read-only."""
import pandas as pd

W = "family_recall_watermarked"
for name, tag in [("844", "bodmas_pool844_tm2"), ("3000", "bodmas_pool3000_tm2"), ("8368", "bodmas_pool0_tm2")]:
    for pol in ("random", "uncertainty"):
        vals = []
        for s in range(10):
            a = pd.read_csv(f"figs/seedpool/{tag}/wacatac__s{s}__{pol}_r0.01.csv").sort_values("month").set_index("month")[W]
            b = pd.read_csv(f"figs/seedpool/{tag}/wacatac__s{s}__clean_{pol}.csv").sort_values("month").set_index("month")[W]
            vals.append(round(float((b - a).dropna().iloc[4:].mean()), 3))
        print(name, pol, vals, round(sum(vals) / len(vals), 3))
