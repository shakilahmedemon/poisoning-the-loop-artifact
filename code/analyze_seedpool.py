#!/usr/bin/env python3
"""Seed-pool analysis. Reads the run files; writes figs/results_seedpool_summary.csv.
Compares the policy effect (random vs uncertainty) with swapped seed-pool sizes against the original datasets.
Gap, depth and horizon follow the paper: post-injection months only, each attacked run against the clean run of the same policy."""
import glob, os, re
import numpy as np, pandas as pd
from scipy import stats

WM = "family_recall_watermarked"


def metrics(run, ref, tm):
    a = run.sort_values("month").set_index("month")[WM]
    b = ref.sort_values("month").set_index("month")[WM]
    g = (b - a).dropna().iloc[4:].values   # months after the 4-month window; the clean run starts right after the seed months
    h = 0
    for x in g:
        if x > 0.15:
            h += 1
        else:
            break
    return dict(gap=float(g.mean()), depth=max([0.0] + list(g[: h + 1])), horizon=h, months=len(g))


rows = []
for setname, tm, pat in [("EMBER, pool 844 (swap)", 1, "figs/seedpool/ember_pool844_tm1/*.csv"),
                         ("BODMAS, 2-month pool (swap)", 2, "figs/seedpool/bodmas_pool0_tm2/*.csv"),
                         ("BODMAS, 2-month setup, pool 844", 2, "figs/seedpool/bodmas_pool844_tm2/*.csv"),
                         ("BODMAS, 2-month setup, pool 3000", 2, "figs/seedpool/bodmas_pool3000_tm2/*.csv")]:
    runs = {}
    for f in glob.glob(pat):
        m = re.match(r".*__s(\d+)__(.+)\.csv$", f.replace("\\", "/"))
        runs[(int(m.group(1)), m.group(2))] = pd.read_csv(f)
    for (sd, arm), v in runs.items():
        if arm.startswith("clean"):
            continue
        pol, r = arm.split("_r")
        if (sd, f"clean_{pol}") in runs:
            rows.append(dict(set=setname, seed=sd, rate=float(r), pol=pol, **metrics(v, runs[(sd, f"clean_{pol}")], tm)))
for setname, pat in [("EMBER, original pool", "figs/sweep_ember_seed?.csv"), ("BODMAS, original pool", "figs/sweep_bodmas_seed?.csv")]:
    for f in sorted(glob.glob(pat)):
        d = pd.read_csv(f); sd = int(d.seed.iloc[0])
        for r in sorted(x for x in d.rate.unique() if x > 0):
            for pol in ("random", "uncertainty"):
                rows.append(dict(set=setname, seed=sd, rate=float(r), pol=pol, **metrics(d[d.arm == f"{pol}_r{r}"], d[d.arm == f"clean_{pol}"], 1)))
R = pd.DataFrame(rows)
out = []
for (s, r), g in R.groupby(["set", "rate"]):
    a = g[g.pol == "random"].sort_values("seed"); b = g[g.pol == "uncertainty"].sort_values("seed")
    seeds = sorted(set(a.seed) & set(b.seed))
    if len(seeds) < 2:
        continue
    a = a[a.seed.isin(seeds)]; b = b[b.seed.isin(seeds)]
    out.append(dict(set=s, rate=r, n=len(seeds), months=int(a.months.iloc[0]), gap_random=a.gap.mean(), gap_uncert=b.gap.mean(),
                    ratio=a.gap.mean() / b.gap.mean() if abs(b.gap.mean()) > 1e-9 else np.nan,
                    depth_random=a.depth.mean(), depth_uncert=b.depth.mean(), horizon_random=a.horizon.mean(), horizon_uncert=b.horizon.mean(),
                    uncert_lower=f"{int((b.gap.values < a.gap.values).sum())}/{len(seeds)}", p=stats.ttest_rel(a.gap.values, b.gap.values).pvalue))
S = pd.DataFrame(out)
S.to_csv("figs/results_seedpool_summary.csv", index=False)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)
print(S.round(3).to_string(index=False))
print("\nClean-run selected-sample uncertainty, mean |p-0.5| (sel_mean_unc):")
for name, pat in [("EMBER pool 844", "figs/seedpool/ember_pool844_tm1/*clean_*.csv"), ("BODMAS 2-month", "figs/seedpool/bodmas_pool0_tm2/*clean_*.csv")]:
    fs = glob.glob(pat)
    if fs:
        d = pd.concat(pd.read_csv(f) for f in fs)
        print(name, d.groupby("arm").sel_mean_unc.mean().round(3).to_dict())
for name, f in [("EMBER pool 844", "figs/seedpool/ember_pool844_tm1/triggers.jsonl"), ("BODMAS 2-month", "figs/seedpool/bodmas_pool0_tm2/triggers.jsonl")]:
    if os.path.exists(f):
        t = pd.read_json(f, lines=True)
        print(name, "pool rows", t.pool_rows.unique().tolist(), "malware", t.pool_malware.tolist(), "distinct triggers", t.trigger.astype(str).nunique())
