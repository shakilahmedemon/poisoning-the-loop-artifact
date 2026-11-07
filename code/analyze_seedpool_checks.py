#!/usr/bin/env python3
"""Seed-pool runs: is the trigger inert on clean models, is the attack stealthy, how large is the poison share? Read-only."""
import glob, json, re
import numpy as np, pandas as pd


def aut(s):
    s = np.asarray([v for v in s if v == v], dtype=float)
    return float(np.mean((s[1:] + s[:-1]) / 2))


SETS = {"BODMAS orig pool 844": None, "BODMAS 2m pool 844": "bodmas_pool844_tm2", "BODMAS 2m pool 3000": "bodmas_pool3000_tm2",
        "BODMAS 2m pool 8368": "bodmas_pool0_tm2", "EMBER pool 844": "ember_pool844_tm1"}
rows = []
for name, tag in SETS.items():
    runs = {}
    if tag:
        for f in glob.glob(f"figs/seedpool/{tag}/*.csv"):
            m = re.match(r".*__s(\d+)__(.+)\.csv$", f.replace("\\", "/"))
            runs[(int(m.group(1)), m.group(2))] = pd.read_csv(f)
    else:
        for s in range(5):
            d = pd.read_csv(f"figs/sweep_bodmas_seed{s}.csv")
            for arm, g in d.groupby("arm"):
                runs[(s, arm)] = g
    for (sd, arm), v in runs.items():
        if arm.startswith("clean") or not arm.endswith("r0.01"):
            continue
        pol = arm.split("_r")[0]
        c = runs[(sd, f"clean_{pol}")].sort_values("month").reset_index(drop=True)
        v = v.sort_values("month").reset_index(drop=True)
        post = 4
        poison = v.train_size.iloc[post] - c.train_size.iloc[post]
        rows.append(dict(set=name, pol=pol, seed=sd,
                         inert=float((c.family_recall_clean - c.family_recall_watermarked).iloc[post:].mean()),
                         clean_f1=float(c.f1.mean()), f1_gap=abs(aut(v.f1) - aut(c.f1)),
                         natural_gap=abs(aut(v.family_recall_clean) - aut(c.family_recall_clean)),
                         pool_end=int(c.train_size.iloc[post]), poison=int(poison), share=poison / v.train_size.iloc[post]))
R = pd.DataFrame(rows)
pd.set_option("display.width", 220)
print(R.groupby(["set", "pol"])[["inert", "clean_f1", "f1_gap", "natural_gap", "pool_end", "poison", "share"]].mean().round(4).to_string())
print("max f1_gap / natural_gap by set:")
print(R.groupby("set")[["f1_gap", "natural_gap"]].max().round(4).to_string())
for tag in ("bodmas_pool844_tm2", "bodmas_pool3000_tm2", "bodmas_pool0_tm2", "ember_pool844_tm1"):
    T = [json.loads(l) for l in open(f"figs/seedpool/{tag}/triggers.jsonl")]
    fields = [set(t["trigger"]) for t in T]
    print(tag, "seeds", len(T), "pool malware share", np.round(np.mean([t["pool_malware"] / t["pool_rows"] for t in T]), 3), "distinct field sets", len({frozenset(x) for x in fields}), "fields common to all", sorted(set.intersection(*fields)))
