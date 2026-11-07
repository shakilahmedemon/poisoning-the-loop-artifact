#!/usr/bin/env python3
"""Threshold sensitivity. Reads the run files; writes figs/results_threshold_*.csv.

Uses every seed present on disk.

A. Episode threshold delta (paper: 0.15). For each attack kind, mean horizon, share of groups whose
   stamped recall ever exceeds the clean run by more than delta (episode start), and the uncertainty-vs-random
   policy ratio of horizon. gap and AUT do not depend on delta.
B. Monthly-check alarm threshold. Run-level TPR/FPR when theta is (i) the max over clean runs of the OTHER
   seeds (paper), (ii) a lower quantile of those clean peaks, (iii) a fixed absolute value.
   AUROC (threshold-free) is repeated for reference.
C. Insertion-gate quantile q (0.05/0.10/0.20): poison catch rate, honest quarantine rate, gap reduction vs the
   ungated attack, paired over groups.
"""
import glob, numpy as np, pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

pd.set_option("display.width", 250); pd.set_option("display.max_columns", 40)
G = ["strategy", "target", "seed"]


def episode(g, delta):
    h = 0
    for x in g:
        if x > delta: h += 1
        else: break
    return h


def series(df, col):
    return df.drop_duplicates("month").sort_values("month").set_index("month")[col]


# A
B = pd.concat([pd.read_csv(f) for f in glob.glob("figs/baselines_eval/*.csv")], ignore_index=True)
D = pd.concat([pd.read_csv(f) for f in glob.glob("figs/defense_eval/*.csv")], ignore_index=True)
D["q"] = D["q"].round(2)
DELTAS = [0.05, 0.10, 0.15, 0.20, 0.25]
gaps = []   # per group, kind -> post-window gap series
for grp, g in B.groupby(G):
    k = {kd: v for kd, v in g.groupby("kind")}
    if not {"clean", "burst", "rand", "static_clean", "static_poison"} <= set(k): continue
    dsel = D[(D.strategy == grp[0]) & (D.target == grp[1]) & (D.seed == grp[2])]
    base = dsel[dsel.arm == "base"]; nog = dsel[dsel.arm == "atk_nogate"]
    if base.empty or nog.empty: continue
    pairs = {"spread": (series(base, "family_recall_watermarked"), series(nog, "family_recall_watermarked")),
             "burst": (series(k["clean"], "wm_att"), series(k["burst"], "wm_att")),
             "rand": (series(k["clean"], "wm_rnd"), series(k["rand"], "wm_rnd")),
             "static": (series(k["static_clean"], "wm_att"), series(k["static_poison"], "wm_att"))}
    for kind, (ref, run) in pairs.items():
        m = pd.concat([ref.rename("r"), run.rename("x")], axis=1).dropna()
        m = m[m.index >= "2020-01"]
        gaps.append(dict(strategy=grp[0], target=grp[1], seed=grp[2], kind=kind, gap=(m.r - m.x).values))
ng = len({(x["strategy"], x["target"], x["seed"]) for x in gaps})
print(f"A. episode threshold: {ng} complete groups, seeds {sorted({x['seed'] for x in gaps})}")
rows = []
for kind in ["spread", "burst", "rand", "static"]:
    for dl in DELTAS:
        hs, ever, by_pol = [], [], {"random": [], "uncertainty": []}
        for x in gaps:
            if x["kind"] != kind: continue
            h = episode(x["gap"], dl); hs.append(h); ever.append(int((x["gap"] > dl).any())); by_pol[x["strategy"]].append(h)
        r = np.mean(by_pol["random"]) if by_pol["random"] else np.nan
        u = np.mean(by_pol["uncertainty"]) if by_pol["uncertainty"] else np.nan
        rows.append(dict(kind=kind, delta=dl, n=len(hs), horizon_mean=round(np.mean(hs), 2),
                         groups_with_episode=f"{sum(ever)}/{len(ever)}",
                         horizon_random=round(r, 2), horizon_uncertainty=round(u, 2),
                         policy_ratio=round(r / u, 1) if u and u > 0 else np.inf))
A = pd.DataFrame(rows); print(A.to_string(index=False)); A.to_csv("figs/results_threshold_delta.csv", index=False)
# does burst > spread and rand ~ spread survive every delta? (paired t on horizon)
print("\nA2. paired comparison vs spread (horizon, paired over groups)")
rows = []
sp = {(x["strategy"], x["target"], x["seed"]): x["gap"] for x in gaps if x["kind"] == "spread"}
for kind in ["burst", "rand", "static"]:
    for dl in DELTAS:
        a, b = [], []
        for x in gaps:
            if x["kind"] != kind: continue
            key = (x["strategy"], x["target"], x["seed"])
            if key in sp: a.append(episode(x["gap"], dl)); b.append(episode(sp[key], dl))
        a, b = np.array(a), np.array(b)
        p = stats.ttest_rel(a, b).pvalue if len(a) > 1 and np.any(a != b) else np.nan
        rows.append(dict(kind=kind, delta=dl, mean_diff_vs_spread=round((a - b).mean(), 2), t_p=round(p, 4)))
A2 = pd.DataFrame(rows); print(A2.to_string(index=False)); A2.to_csv("figs/results_threshold_delta_paired.csv", index=False)

# B
P = pd.concat([pd.read_csv(f) for f in ["figs/results_probe_defender.csv"] + glob.glob("figs/results_probe_defender_s5_9.csv")],
              ignore_index=True)
P = P[P.month >= "2020-01"].copy()
P["run"] = P.strategy + "|" + P.target + "|" + P.seed.astype(str)
done = P.groupby("run").arm.nunique(); P = P[P.run.isin(done[done == 2].index)]
print(f"\nB. monthly-check alarm threshold: {P.run.nunique()} complete run pairs, seeds {sorted(P.seed.unique())}")
TR = ["att", "k12", "k4", "cur", "unc", "boot", "rand"]
rows = []
for t in TR:
    col = f"pool_{t}"
    y = (P.arm == "attacked").astype(int).values; x = P[col].values; ok = ~np.isnan(x)
    auc = roc_auc_score(y[ok], x[ok])
    peak = P.groupby(["run", "seed", "arm"])[col].max().reset_index()
    rules = [("clean max (paper)", lambda c: c.max()), ("clean q95", lambda c: c.quantile(0.95)),
             ("clean q90", lambda c: c.quantile(0.90)), ("clean q75", lambda c: c.quantile(0.75))]
    rules += [(f"fixed {v:.2f}", (lambda v: lambda c: v)(v)) for v in (0.02, 0.05, 0.10, 0.15)]
    for name, fn in rules:
        tp = fp = na = nc = 0
        for sd in sorted(P.seed.unique()):
            cal = fn(peak[(peak.seed != sd) & (peak.arm == "clean")][col])
            te = peak[peak.seed == sd]
            tp += int((te[te.arm == "attacked"][col] > cal).sum()); na += int((te.arm == "attacked").sum())
            fp += int((te[te.arm == "clean"][col] > cal).sum()); nc += int((te.arm == "clean").sum())
        rows.append(dict(trigger=t, rule=name, AUROC=round(auc, 3), TPR=f"{tp}/{na}", FPR=f"{fp}/{nc}",
                         TPR_pct=round(100 * tp / na), FPR_pct=round(100 * fp / nc)))
Bt = pd.DataFrame(rows)
print(Bt.pivot(index="trigger", columns="rule", values="TPR_pct").loc[TR].to_string())
print("\nFPR %:"); print(Bt.pivot(index="trigger", columns="rule", values="FPR_pct").loc[TR].to_string())
Bt.to_csv("figs/results_threshold_probe.csv", index=False)

# C
print("\nC. insertion-gate quantile")
rows = []
nog = {}
for grp, g in D[D.arm == "atk_nogate"].groupby(G):
    base = D[(D.strategy == grp[0]) & (D.target == grp[1]) & (D.seed == grp[2]) & (D.arm == "base")]
    if base.empty: continue
    nog[grp] = (series(base, "family_recall_watermarked"), series(g, "family_recall_watermarked"))
def gap_of(ref, run):
    m = pd.concat([ref.rename("r"), run.rename("x")], axis=1).dropna(); m = m[m.index >= "2020-01"]
    return float((m.r - m.x).mean())
for (d, q, mode), g in D[D.arm == "atk_gate"].groupby(["defense", "q", "mode"]):
    gl, cat, hq = [], [], []
    for grp, gg in g.groupby(G):
        if grp not in nog: continue
        ref = nog[grp][0]
        gl.append((gap_of(ref, nog[grp][1]), gap_of(ref, series(gg, "family_recall_watermarked"))))
        s = gg.drop_duplicates("month")
        cat.append(s.n_poison_quarantined.sum() / max(1, s.n_poison_offered.sum()))
        hq.append(s.n_honest_quarantined.sum() / max(1, s.n_honest_offered.sum()))
    if not gl: continue
    a = np.array(gl)
    rows.append(dict(defense=d, q=q, mode=mode, n=len(a), poison_caught_pct=round(100 * np.mean(cat), 1),
                     honest_quarantined_pct=round(100 * np.mean(hq), 1),
                     gap_ungated=round(a[:, 0].mean(), 3), gap_gated=round(a[:, 1].mean(), 3),
                     reduction=round((a[:, 0] - a[:, 1]).mean(), 3),
                     paired_p=round(stats.ttest_rel(a[:, 0], a[:, 1]).pvalue, 4) if len(a) > 1 else np.nan))
C = pd.DataFrame(rows).sort_values(["defense", "mode", "q"]); print(C.to_string(index=False))
C.to_csv("figs/results_threshold_gates.csv", index=False)
