#!/usr/bin/env python3
"""Analysis of figs/baselines_eval/*.csv. Writes only the two summary CSVs.

Per (policy, family, seed) group, compared with the matching clean run (same seed, no attack):
  spread   paper attack (figs/defense_eval arm atk_nogate), reference = defense_eval arm base
  burst    same poison injected in one month; reference = baselines clean (wm_att)
  rand     random trigger, poison + evaluation with it; reference = baselines clean (wm_rnd)
  static_poison  one-shot model (no retraining) with the poison; reference = static_clean (wm_att)
gap = mean over months >= 2020-01 of (reference stamped recall - run stamped recall); horizon = first run of months
with gap > 0.15; depth = max{0, g_p1..g_p(h+1)}; AUT = area under time of stamped recall over the whole window.
Paired t-tests and Wilcoxon over the 20 groups, versus the spread (paper) attack.
"""
import glob, os, numpy as np, pandas as pd
from scipy import stats

B = pd.concat([pd.read_csv(f) for f in glob.glob("figs/baselines_eval/*.csv")], ignore_index=True)
D = pd.concat([pd.read_csv(f) for f in glob.glob("figs/defense_eval/*.csv")], ignore_index=True)
D["q"] = D["q"].round(2)
print("baseline files:", B.groupby(["strategy", "target", "seed", "kind"]).ngroups, " rows:", len(B))
G = ["strategy", "target", "seed"]


def aut(s):
    s = np.asarray([v for v in s if not np.isnan(v)], dtype=float)
    return float(np.mean((s[1:] + s[:-1]) / 2)) if len(s) >= 2 else np.nan


def episode(g):
    h = 0
    for x in g:
        if x > 0.15: h += 1
        else: break
    return h, max([0.0] + list(g[: h + 1]))


def series(df, col):
    return df.sort_values("month").set_index("month")[col]


def stats_of(ref, run):
    m = pd.concat([ref.rename("r"), run.rename("x")], axis=1).dropna()
    post = m[m.index >= "2020-01"]
    gap = (post.r - post.x).values
    h, peak = episode(list(gap))
    return dict(gap=float(gap.mean()) if len(gap) else np.nan, depth=peak, horizon=h,
                AUT=aut(m.x.values), AUT_ref=aut(m.r.values), minrecall=float(m.x.min()))


rows = []
for grp, g in B.groupby(G):
    k = {kind: v for kind, v in g.groupby("kind")}
    sd = grp
    dsel = D[(D.strategy == grp[0]) & (D.target == grp[1]) & (D.seed == grp[2])]
    base = dsel[dsel.arm == "base"]
    nog = dsel[dsel.arm == "atk_nogate"]
    base_s = series(base.drop_duplicates("month"), "family_recall_watermarked")
    # consistency of our clean run with the paper's base run
    cl_att = series(k["clean"], "wm_att")
    common = base_s.index.intersection(cl_att.index)
    mism = float(np.nanmax(np.abs(base_s[common] - cl_att[common]))) if len(common) else np.nan
    entries = {
        "spread": (base_s, series(nog.drop_duplicates("month"), "family_recall_watermarked")),
        "burst": (cl_att, series(k["burst"], "wm_att")),
        "rand": (series(k["clean"], "wm_rnd"), series(k["rand"], "wm_rnd")),
        "static_poison": (series(k["static_clean"], "wm_att"), series(k["static_poison"], "wm_att")),
    }
    for name, (ref, run) in entries.items():
        r = stats_of(ref, run)
        r.update(strategy=grp[0], target=grp[1], seed=grp[2], kind=name, clean_vs_paper_base_maxdiff=mism,
                 f1=float(k["rand" if name == "rand" else ("burst" if name == "burst" else
                          ("static_poison" if name == "static_poison" else "clean"))].f1.mean())
                 if name != "spread" else float(nog.f1.mean()))
        rows.append(r)
    # extra diagnostic: how inert is each trigger on the CLEAN model (stamped recall vs natural recall)
    rows.append(dict(strategy=grp[0], target=grp[1], seed=grp[2], kind="diag_clean_att_stamped_minus_natural",
                     gap=float((k["clean"].family_recall_clean - k["clean"].wm_att).mean()), clean_vs_paper_base_maxdiff=mism))
    rows.append(dict(strategy=grp[0], target=grp[1], seed=grp[2], kind="diag_clean_rnd_stamped_minus_natural",
                     gap=float((k["clean"].family_recall_clean - k["clean"].wm_rnd).mean()), clean_vs_paper_base_maxdiff=mism))
R = pd.DataFrame(rows)
R.to_csv("figs/results_baselines_runs.csv", index=False)
pd.set_option("display.width", 220)

print("\nclean run vs paper base run (stamped recall), max abs diff over groups:",
      f"{R.clean_vs_paper_base_maxdiff.max():.2e}")

ref = R[R.kind == "spread"].set_index(G)
out = []
for kind in ["spread", "burst", "rand", "static_poison"]:
    g = R[R.kind == kind].set_index(G)
    j = g.join(ref, rsuffix="_sp")
    row = dict(kind=kind, n=len(g), gap=f"{g.gap.mean():.3f} +- {g.gap.std()/np.sqrt(len(g)):.3f}",
               depth=f"{g.depth.mean():.3f}", horizon=f"{g.horizon.mean():.2f}", AUT=f"{g.AUT.mean():.4f}",
               min_recall=f"{g.minrecall.mean():.3f}", groups_gap_gt_0_15=int((g.gap > 0.15).sum()))
    if kind != "spread":
        d = (j.gap - j.gap_sp)
        row["gap_minus_spread"] = f"{d.mean():+.3f}"
        row["t_p"] = f"{stats.ttest_rel(j.gap, j.gap_sp).pvalue:.4f}"
        try: row["wilcoxon_p"] = f"{stats.wilcoxon(j.gap, j.gap_sp).pvalue:.4f}"
        except Exception: row["wilcoxon_p"] = "-"
        row["wins_vs_spread"] = f"{int((j.gap > j.gap_sp).sum())}/{len(j)} larger gap"
    out.append(row)
S = pd.DataFrame(out)
print("\n=== ALL 20 GROUPS ==="); print(S.to_string(index=False))
S.to_csv("figs/results_baselines_summary.csv", index=False)

for by in ("strategy", "target"):
    for v, gg in R[R.kind.isin(["spread", "burst", "rand", "static_poison"])].groupby(by):
        print(f"\n=== {by} = {v} (10 groups) ===")
        print(gg.groupby("kind")[["gap", "depth", "horizon", "AUT"]].mean().round(3).to_string())

print("\n=== trigger inertness on the CLEAN model: natural recall - stamped recall (mean over months, 20 groups) ===")
for kd in ["diag_clean_att_stamped_minus_natural", "diag_clean_rnd_stamped_minus_natural"]:
    x = R[R.kind == kd].gap
    print(f"  {kd}: mean {x.mean():+.4f}  min {x.min():+.4f}  max {x.max():+.4f}")

print("\n=== per-group gap table (policy, family, seed) ===")
P = R[R.kind.isin(["spread", "burst", "rand", "static_poison"])].pivot_table(index=G, columns="kind", values="gap").round(3)
print(P.to_string())
