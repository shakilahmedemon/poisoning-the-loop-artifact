#!/usr/bin/env python3
"""Aggregate every new experiment into paper-ready numbers. Safe to run on partial data."""
import os, numpy as np, pandas as pd
from scipy import stats
F = "figs/"
def have(n): return os.path.exists(F + n)
def ms(x): return f"{np.mean(x):.3f}±{np.std(x, ddof=1) if len(x)>1 else 0:.3f}"

print("=== 1. malware-trigger ablation (eval-time malware trigger variants) ===")
if have("results_ablation_malware_trigger.csv"):
    d = pd.read_csv(F + "results_ablation_malware_trigger.csv")
    for (s, fam), g in d.groupby(["strategy", "family"], sort=False):
        row = []
        for v in ("full8", "six", "five"):
            h = g[g.variant == v]
            row.append(f"{v}: depth {ms(h.depth)} hor {ms(h.horizon)} (n={len(h)})")
        p = ""
        a = g[g.variant == "full8"].sort_values("seed").depth.values; b = g[g.variant == "six"].sort_values("seed").depth.values
        if len(a) == len(b) and len(a) > 2: p = f" paired-t p(full8 vs six)={stats.ttest_rel(a, b).pvalue:.4f}"
        print(f"{s:11} {fam:8} | " + " | ".join(row) + p)

print("\n=== 2. multi-seed family sweep (seeds 0-4) ===")
if have("results_family_sweep_seeds.csv") and have("results_family_sweep.csv"):
    s0 = pd.read_csv(F + "results_family_sweep.csv")[["family", "depth", "horizon"]].assign(seed=0)
    d = pd.concat([s0, pd.read_csv(F + "results_family_sweep_seeds.csv")[["family", "depth", "horizon", "seed"]]])
    for fam, g in d.groupby("family", sort=False):
        print(f"{fam:9} seeds={len(g)} depth {ms(g.depth)} horizon {g.horizon.mean():.2f}±{g.horizon.std(ddof=1):.2f}")

print("\n=== 3. volume intervention ===")
if have("results_volume_intervention.csv"):
    d = pd.read_csv(F + "results_volume_intervention.csv")
    base = None
    if have("results_family_sweep_seeds.csv") and have("results_family_sweep.csv"):
        b0 = pd.read_csv(F + "results_family_sweep.csv").assign(seed=0)
        base = pd.concat([b0, pd.read_csv(F + "results_family_sweep_seeds.csv")])
    for fam, g in d.groupby("family", sort=False):
        if base is not None:
            bb = base[(base.family == fam) & (base.seed.isin(sorted(g.seed.unique())))]
            print(f"{fam:9} scale=1.0  depth {ms(bb.depth)} horizon {bb.horizon.mean():.2f} (n={len(bb)})")
        for sc, h in g.groupby("scale"):
            print(f"{fam:9} scale={sc:<5} n_fam~{int(h.n_family_after.mean()):6d} depth {ms(h.depth)} horizon {h.horizon.mean():.2f}±{h.horizon.std(ddof=1):.2f} (n={len(h)})")

print("\n=== 4. mixed policy (defense) ===")
if have("results_mixed_policy.csv") and have("results_seeds.csv"):
    m = pd.read_csv(F + "results_mixed_policy.csv"); r = pd.read_csv(F + "results_seeds.csv")
    for fam in ("wacatac", "sfone"):
        for pol, dd in (("random", r[(r.strategy == "random") & (r.family == fam)]),
                        ("uncertainty", r[(r.strategy == "uncertainty") & (r.family == fam)]),
                        ("mixed", m[m.family == fam])):
            print(f"{fam:8} {pol:11} depth {ms(dd.depth)} horizon {dd.horizon.mean():.2f}±{dd.horizon.std(ddof=1):.2f} (n={len(dd)})")
        g = m[m.family == fam]
        print(f"{fam:8} clean AUT-F1  random {ms(g.aut_f1_random)}  uncertainty {ms(g.aut_f1_uncertainty)}  mixed {ms(g.aut_f1_mixed)}")

print("\n=== 5. backdoor probe (defender-side regression test) ===")
if have("results_backdoor_probe.csv"):
    d = pd.read_csv(F + "results_backdoor_probe.csv")
    fams = sorted({c[6:-4] for c in d.columns if c.startswith("probe_") and c.endswith("_rec")})
    rows = []
    for (s, t, sd, arm), g in d.groupby(["strategy", "target", "seed", "arm"]):
        inj = g.iloc[:4]   # first 4 evaluated months after seed = injection window
        for i, r_ in g.iterrows():
            gaps = {f: r_[f"probe_{f}_rec"] - r_[f"probe_{f}_wm"] for f in fams if not np.isnan(r_[f"probe_{f}_rec"])}
            if not gaps: continue
            top = max(gaps, key=gaps.get)
            rows.append(dict(strategy=s, target=t, seed=sd, arm=arm, month=r_["month"], stat=gaps[top], top=top,
                             tgt_gap=gaps.get(t, np.nan), post=int(list(g.index).index(i) >= 4)))
    R = pd.DataFrame(rows); P = R[R.post == 1]
    if len(P):
        clean = P[P.arm == "clean"]; att = P[P.arm == "attacked"]
        for thr in (0.2, 0.3, 0.4, 0.5):
            far = (clean.groupby(["strategy", "target", "seed"]).stat.max() > thr).mean()
            det = (att.groupby(["strategy", "target", "seed"]).stat.max() > thr).mean()
            hit = (att[att.stat > thr].top == att[att.stat > thr].target).mean() if (att.stat > thr).any() else float("nan")
            print(f"threshold {thr:.1f}: false-alarm runs {far:.2f} | detected attacked runs {det:.2f} | argmax==target {hit:.2f}")
        from sklearn.metrics import roc_auc_score
        y = np.r_[np.zeros(len(clean)), np.ones(len(att))]; sc = np.r_[clean.stat.values, att.stat.values]
        print(f"AUROC (post-window month-level, max-family gap): {roc_auc_score(y, sc):.3f}  (n_clean={len(clean)}, n_att={len(att)})")
