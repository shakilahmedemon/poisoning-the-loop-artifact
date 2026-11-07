#!/usr/bin/env python3
"""Unconstrained-trigger pilot analysis (read-only). BODMAS, target wacatac: unconstrained trigger (figs/unconstrained_pilot/unconstrained)
vs the paper's constrained trigger (figs/sweep_bodmas_seed*.csv), and EMBER (figs/sweep_ember_seed*.csv, unconstrained, emotet).
gap = clean stamped recall - attacked stamped recall of the SAME policy/seed/trigger, over post-window months (>= first month
after the 4-month injection window); depth/horizon as in analyze_baselines.py (episode = consecutive months with gap > 0.15 from the
start of the post-window). Paired t-test and sign count over seeds (uncertainty vs random)."""
import glob, numpy as np, pandas as pd
from scipy import stats
pd.set_option("display.width", 250)

def load_pilot():
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob("figs/unconstrained_pilot/unconstrained/*.csv"))], ignore_index=True)
def load_sweep(ds):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"figs/sweep_{ds}_seed*.csv"))], ignore_index=True)

def episode(g, thr=0.15):
    h = 0
    for x in g:
        if x > thr: h += 1
        else: break
    return h, max([0.0] + list(g[: h + 1]))

def stats_for(d, label):
    rows = []
    for pol in ("random", "uncertainty"):
        for rate in sorted(r for r in d.rate.unique() if r > 0):
            for sd in sorted(d.seed.unique()):
                b = d[(d.arm == f"clean_{pol}") & (d.seed == sd)].set_index("month").family_recall_watermarked
                a = d[(d.arm == f"{pol}_r{rate:g}") & (d.seed == sd)].set_index("month").family_recall_watermarked
                if len(a) == 0 or len(b) == 0: continue
                j = pd.concat([b.rename("b"), a.rename("a")], axis=1).dropna()
                months = sorted(j.index); post = months[4:]          # months after the 4-month window
                g = (j.loc[post, "b"] - j.loc[post, "a"]).values
                h, dep = episode(list(g))
                nat = d[(d.arm == f"{pol}_r{rate:g}") & (d.seed == sd)].set_index("month")
                rows.append(dict(set=label, policy=pol, rate=rate, seed=sd, gap=g.mean(), depth=dep, horizon=h))
    return pd.DataFrame(rows)

P = load_pilot()
print("pilot files:", P.groupby(["seed", "arm"]).ngroups, " seeds:", sorted(P.seed.unique()))
S = pd.concat([stats_for(P, "BODMAS unconstrained"), stats_for(load_sweep("bodmas"), "BODMAS constrained (paper)"),
               stats_for(load_sweep("ember"), "EMBER unconstrained")], ignore_index=True)
S.to_csv("figs/results_unconstrained_pilot_runs.csv", index=False)

out = []
for (st, rate), g in S.groupby(["set", "rate"]):
    r = g[g.policy == "random"].set_index("seed"); u = g[g.policy == "uncertainty"].set_index("seed")
    c = r.index.intersection(u.index)
    if len(c) < 2: continue
    d = (r.loc[c, "gap"] - u.loc[c, "gap"])
    out.append(dict(set=st, rate=rate, n=len(c), gap_random=r.loc[c, "gap"].mean(), gap_uncert=u.loc[c, "gap"].mean(),
                    ratio=r.loc[c, "gap"].mean() / max(u.loc[c, "gap"].mean(), 1e-9),
                    depth_random=r.loc[c, "depth"].mean(), depth_uncert=u.loc[c, "depth"].mean(),
                    horizon_random=r.loc[c, "horizon"].mean(), horizon_uncert=u.loc[c, "horizon"].mean(),
                    seeds_uncert_lower=f"{int((d > 0).sum())}/{len(c)}",
                    paired_t_p=stats.ttest_rel(r.loc[c, "gap"], u.loc[c, "gap"]).pvalue))
T = pd.DataFrame(out).sort_values(["set", "rate"])
print("\n=== policy effect (random vs uncertainty), mean over seeds ===")
print(T.round(3).to_string(index=False)); T.to_csv("figs/results_unconstrained_pilot_summary.csv", index=False)

# trigger inertness on the clean model with the unconstrained trigger (stamped vs natural recall)
c = P[P.arm.str.startswith("clean_")]
print("\n=== inertness of the unconstrained trigger on the clean model (natural - stamped recall, mean over months) ===")
print(c.assign(d=c.family_recall_clean - c.family_recall_watermarked).groupby("arm").d.agg(["mean", "min", "max"]).round(4).to_string())
# stealth: F1 difference attacked vs clean
print("\n=== F1 (mean over months): attacked vs clean, same policy ===")
for pol in ("random", "uncertainty"):
    b = P[P.arm == f"clean_{pol}"].f1.mean()
    for rate in sorted(r for r in P.rate.unique() if r > 0):
        print(pol, rate, round(P[P.arm == f"{pol}_r{rate:g}"].f1.mean() - b, 4))
# what trigger was used
import json
print("\n=== triggers (feature index: value) ===")
for l in open("figs/unconstrained_pilot/unconstrained/triggers.jsonl"):
    print(json.loads(l))
