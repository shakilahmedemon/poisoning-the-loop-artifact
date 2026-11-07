#!/usr/bin/env python3
"""Analysis of figs/defense_eval/*.csv. Read-only.

Per (defense, mode, quantile) over the 20 groups (2 policies x 2 families x 5 seeds):
  catch      share of the offered poison that the gate quarantined
  honest     share of honestly selected candidates that the gate quarantined (cost of the gate)
  AUT_*      area under time of recall on trigger-stamped target malware (whole test window)
  recover    (AUT_gate - AUT_nogate) / (AUT_base - AUT_nogate): share of the lost AUT the gate wins back
  gap/depth/horizon   paper definitions, post-window months (>= 2020-01), vs the clean no-gate run of the same
                      policy/family/seed (gap = base - gated); horizon = first run of months with gap > 0.15;
                      depth = max{0, g_p1..g_p(h+1)}
Also paired t-test (gap of gated run vs gap of ungated attacked run, over the 20 groups).
"""
import glob, os, numpy as np, pandas as pd
from scipy import stats

D = "figs/defense_eval"
frames = [pd.read_csv(f) for f in glob.glob(os.path.join(D, "*.csv"))]
d = pd.concat(frames, ignore_index=True)
d["q"] = d["q"].round(2)
print("files:", len(frames), " rows:", len(d))

def aut(s):
    s = np.asarray([v for v in s if not np.isnan(v)], dtype=float)
    return float(np.mean((s[1:] + s[:-1]) / 2)) if len(s) >= 2 else np.nan

def episode(g):
    h = 0
    for x in g:
        if x > 0.15: h += 1
        else: break
    peak = max([0.0] + list(g[: h + 1]))
    return h, peak

G = ["strategy", "target", "seed"]
base = {k: v.sort_values("month") for k, v in d[d.arm == "base"].groupby(G)}
nog = {k: v.sort_values("month") for k, v in d[d.arm == "atk_nogate"].groupby(G)}

def metrics(run, grp):
    run = run.sort_values("month"); b = base[grp]; n = nog[grp]
    m = run.merge(b[["month", "family_recall_watermarked"]].rename(columns={"family_recall_watermarked": "b"}), on="month")
    m = m[m.month >= "2020-01"]
    gap = (m["b"] - m["family_recall_watermarked"]).values
    ok = ~np.isnan(gap); gap = gap[ok]
    h, peak = episode(list(gap))
    a_run, a_b, a_n = aut(run.family_recall_watermarked), aut(b.family_recall_watermarked), aut(n.family_recall_watermarked)
    return dict(gap=float(np.mean(gap)) if len(gap) else np.nan, depth=peak, horizon=h,
                AUT=a_run, rec=(a_run - a_n) / (a_b - a_n) if abs(a_b - a_n) > 1e-9 else np.nan,
                f1=float(run.f1.mean()))

rows = []
for grp, g in d[d.arm.isin(["atk_nogate", "atk_gate"])].groupby(G + ["arm", "defense", "q", "mode"]):
    r = g.iloc[0]
    m = metrics(g, tuple(grp[:3]))
    m.update(strategy=grp[0], target=grp[1], seed=grp[2], arm=grp[3], defense=grp[4], q=grp[5], mode=grp[6],
             catch=r.n_poison_quarantined / max(r.n_poison_offered, 1) if grp[3] == "atk_gate" else 0.0,
             honest=r.n_honest_quarantined / max(r.n_honest_offered, 1) if grp[3] == "atk_gate" else 0.0,
             poison=int(r.n_poison_offered))
    rows.append(m)
R = pd.DataFrame(rows)
R.to_csv("figs/results_defense_eval_runs.csv", index=False)
ref = R[R.arm == "atk_nogate"].set_index(G)["gap"]
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 30)

def summarize(df, label_cols):
    out = []
    for key, g in df.groupby(label_cols):
        key = key if isinstance(key, tuple) else (key,)
        g = g.copy()
        g["ref"] = [ref[(s, t, sd)] for s, t, sd in zip(g.strategy, g.target, g.seed)]
        tt = stats.ttest_rel(g.gap, g.ref) if len(g) > 2 else None
        out.append(dict(zip(label_cols, key), n=len(g), catch=f"{100*g.catch.mean():.1f}%", honest_lost=f"{100*g.honest.mean():.1f}%",
                        gap=f"{g.gap.mean():.3f}", depth=f"{g.depth.mean():.3f}", horizon=f"{g.horizon.mean():.2f}",
                        AUT=f"{g.AUT.mean():.4f}", recover=f"{100*g.rec.mean():.1f}%",
                        gap_vs_nogate=(f"{(g.gap-g.ref).mean():+.3f} (p={tt.pvalue:.3f})" if tt is not None else "-")))
    return pd.DataFrame(out)

print("\n=== ALL 20 GROUPS (2 policies x 2 families x 5 seeds) ===")
S = summarize(R, ["defense", "mode", "q"]); print(S.to_string(index=False)); S.to_csv("figs/results_defense_eval_summary.csv", index=False)
for strat in ("random", "uncertainty"):
    print(f"\n=== policy = {strat} (10 groups) ===")
    print(summarize(R[R.strategy == strat], ["defense", "mode", "q"]).to_string(index=False))
for tgt in ("wacatac", "sfone"):
    print(f"\n=== family = {tgt} (10 groups) ===")
    print(summarize(R[R.target == tgt], ["defense", "mode", "q"]).to_string(index=False))
# single cell (seed 0, wacatac) in the setting of attack_vs_cade.py
print("\n=== single cell (seed 0, wacatac, q=0.05, global) ===")
x = R[(R.seed == 0) & (R.target == "wacatac") & (R.q.isin([0.0, 0.05])) & (R["mode"].isin(["-", "global"]))]
print(x[["strategy", "defense", "mode", "catch", "poison", "AUT"]].round(4).to_string(index=False))
