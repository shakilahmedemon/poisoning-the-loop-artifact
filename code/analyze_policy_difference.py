#!/usr/bin/env python3
"""Where do BODMAS and EMBER differ? Computed from the stored sweep files; read-only.
Reads figs/sweep_bodmas_seed*.csv and figs/sweep_ember_seed*.csv (columns: arm, rate, month, n, train_size, sel_n, sel_mal_n,
sel_mean_unc, family_n, family_recall_clean, family_recall_watermarked, ...). Writes figs/results_policy_difference.csv. No model is fitted."""
import glob, numpy as np, pandas as pd
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)

def load(pat):
    return pd.concat([pd.read_csv(f) for f in sorted(glob.glob(pat))], ignore_index=True)

D = {"BODMAS": load("figs/sweep_bodmas_seed*.csv"), "EMBER": load("figs/sweep_ember_seed*.csv")}
for k, d in D.items():
    print(k, "rows", len(d), "seeds", sorted(d.seed.unique()), "arms", sorted(d.arm.unique()), "rates", sorted(d.rate.unique()))
rows = []
for k, d in D.items():
    d = d.copy()
    cl = d[d.arm.str.startswith("clean_")]
    for pol in ("random", "uncertainty"):
        c = cl[cl.arm == f"clean_{pol}"]
        g = c.groupby("month")
        r = dict(dataset=k, policy=pol,
                 month_n=c.n.mean(), month_mal_share=(c.month_mal_n / c.n).mean(),
                 budget_sel_n=c.sel_n.mean(),
                 sel_mal_share=(c.sel_mal_n / c.sel_n).mean(),
                 sel_mean_unc=c.sel_mean_unc.mean(),
                 train_first=c.groupby("seed").train_size.min().mean(), train_last=c.groupby("seed").train_size.max().mean(),
                 fam_n_per_month=c.family_n.mean(), fam_conf=c.fam_mean_p.mean(), fam_near_boundary=c.fam_near_boundary.mean(),
                 f1=c.f1.mean())
        rows.append(r)
R = pd.DataFrame(rows).round(3)
print("\n=== clean-run descriptives (no attack) ==="); print(R.T.to_string())
R.to_csv("figs/results_policy_difference.csv", index=False)

# backdoor strength by rate and policy: gap = clean stamped recall - attacked stamped recall (post injection months), same seed/policy
print("\n=== attack strength by rate and policy (mean over seeds of mean gap after the 4-month window; depth = max monthly gap) ===")
out = []
for k, d in D.items():
    for pol in ("random", "uncertainty"):
        base = d[(d.arm == f"clean_{pol}") & (d.rate == 0)].set_index(["seed", "month"]).family_recall_watermarked
        for rate in sorted(r for r in d.rate.unique() if r > 0):
            a = d[(d.arm == f"{pol}_r{rate:g}") & (d.rate == rate)].set_index(["seed", "month"]).family_recall_watermarked
            j = pd.concat([base.rename("b"), a.rename("a")], axis=1).dropna().reset_index()
            months = sorted(j.month.unique()); post = months[4:]
            jj = j[j.month.isin(post)]
            gs = jj.assign(g=jj.b - jj.a).groupby("seed").g
            out.append(dict(dataset=k, policy=pol, rate=rate, n_seeds=jj.seed.nunique(), gap=gs.mean().mean(), depth=gs.max().mean()))
O = pd.DataFrame(out).round(3); print(O.to_string(index=False)); O.to_csv("figs/results_policy_difference_strength.csv", index=False)

# poison share of the training set at the end of the injection window
print("\n=== train-set size at the first post-window month vs poison volume ===")
for k, d in D.items():
    a = d[(d.arm == "random_r0.01")]
    cl = d[(d.arm == "clean_random") & (d.rate == 0)]
    j = a.merge(cl[["seed", "month", "train_size"]], on=["seed", "month"], suffixes=("", "_clean"))
    j["extra"] = j.train_size - j.train_size_clean
    last = j[j.month == sorted(j.month.unique())[4]] if len(j.month.unique()) > 4 else j
    print(k, "poison samples in train at first post-window month: mean", round(last.extra.mean(), 1),
          " train_size", round(last.train_size.mean(), 1), " poison share", round((last.extra / last.train_size).mean(), 4))
