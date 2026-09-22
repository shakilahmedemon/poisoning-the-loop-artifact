#!/usr/bin/env python3
"""
Two experiments that turn the paper's open questions into evidence.

--part seeds : family sweep re-run for seeds 1-4 (seed 0 already in
               results_family_sweep.csv) -> per-family mean +/- std, makes the
               cross-family heterogeneity (RQ3) statistically meaningful.
--part volume: INTERVENTION on family volume. The volume hypothesis says a family
               with more real files per month dilutes the backdoor faster (ordinary
               selection sees more honest exposure). We scale the target family's
               malware rows in the stream (scale<1: random subsample; scale>1: each
               row replicated floor(scale)x plus a random remainder) and measure
               depth/horizon at random selection.
                 sillyp2p (largest, zero persistence): scale 0.5, 0.25, 0.1
                 simda    (small, long persistence)  : scale 5, 10, 20
Resume-safe: every completed cell is appended to the CSV and skipped on restart.
"""
import argparse, csv, os, time
import numpy as np, pandas as pd
from attack_smart import load_bodmas, build_trigger, run, aut
from run_seeds import evaluate

FAMILIES = ["sillyp2p", "zbot", "vflooder", "simda", "plite", "simbot"]
VOLUME = {"sillyp2p": [0.5, 0.25, 0.1], "simda": [5.0, 10.0, 20.0]}


def rescale(X, y, meta, period, family, scale, seed):
    rng = np.random.RandomState(1000 + seed)
    is_t = ((meta["family"].values == family) & (y == 1))
    idx_t = np.where(is_t)[0]
    keep_other = np.where(~is_t)[0]
    whole = int(np.floor(scale)); frac = scale - whole
    parts = [np.tile(idx_t, whole)] if whole >= 1 else []
    if frac > 0:
        parts.append(rng.choice(idx_t, size=int(round(frac * len(idx_t))), replace=False))
    sel = np.concatenate([keep_other] + parts) if parts else keep_other
    sel = np.sort(sel)
    return (X[sel], y[sel], meta.iloc[sel].reset_index(drop=True),
            period.iloc[sel].reset_index(drop=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["seeds", "volume", "mixed"], required=True)
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    a = ap.parse_args()
    import lightgbm as lgb
    X, y, meta, period = load_bodmas(a.data_dir)
    months = list(period.unique())
    out = {"seeds": "family_sweep_seeds", "volume": "volume_intervention", "mixed": "mixed_policy"}[a.part]
    out = f"./figs/results_{out}.csv"
    fields = ["family", "scale", "seed", "n_family_after", "depth", "horizon", "f1_gap", "real_fam_gap", "wall_s"]
    if a.part == "mixed":
        fields += ["aut_f1_random", "aut_f1_uncertainty", "aut_f1_mixed"]
    new = not os.path.exists(out)
    f = open(out, "a", newline=""); w = csv.DictWriter(f, fieldnames=fields)
    if new: w.writeheader(); f.flush()
    done = set() if new else {(r.family, float(r.scale), int(r.seed)) for r in pd.read_csv(out).itertuples()}

    cells = []
    if a.part == "seeds":
        for s in (a.seeds or [1, 2, 3, 4]):
            cells += [(fam, 1.0, s) for fam in FAMILIES]
    elif a.part == "mixed":
        for s in (a.seeds or [0, 1, 2, 3, 4]):
            cells += [(fam, 1.0, s) for fam in ("wacatac", "sfone")]
    else:
        for s in (a.seeds or [0, 1, 2]):
            for fam, scales in VOLUME.items():
                cells += [(fam, sc, s) for sc in scales]

    for fam, scale, seed in cells:
        if (fam, scale, seed) in done: continue
        t0 = time.time()
        X2, y2, m2, p2 = (X, y, meta, period) if scale == 1.0 else rescale(X, y, meta, period, fam, scale, seed)
        mo = list(p2.unique())
        idx = np.where(p2.isin(mo[:1]).values)[0]
        clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, n_jobs=-1,
                                 verbose=-1, random_state=seed).fit(X2[idx], y2[idx])
        trig = build_trigger(clf, X2[idx], y2[idx], 8, constrained=True, seed=seed)
        pol = "mixed" if a.part == "mixed" else "random"
        dc, _ = run(pol, False, X2, y2, m2, p2, mo, 1, 0.05, 0.01, 4, fam, trig, seed=seed)
        da, _ = run(pol, True, X2, y2, m2, p2, mo, 1, 0.05, 0.01, 4, fam, trig, seed=seed)
        depth, hor, f1g, rfg = evaluate(dc, da, mo, 1, 4)
        extra = {}
        if a.part == "mixed":
            dr, _ = run("random", False, X2, y2, m2, p2, mo, 1, 0.05, 0.01, 4, fam, trig, seed=seed)
            du, _ = run("uncertainty", False, X2, y2, m2, p2, mo, 1, 0.05, 0.01, 4, fam, trig, seed=seed)
            extra = dict(aut_f1_random=aut(dr["f1"].values), aut_f1_uncertainty=aut(du["f1"].values),
                         aut_f1_mixed=aut(dc["f1"].values))
        n_fam = int(((m2["family"].values == fam) & (y2 == 1)).sum())
        w.writerow(dict(family=fam, scale=scale, seed=seed, n_family_after=n_fam, depth=depth, horizon=hor,
                        f1_gap=f1g, real_fam_gap=rfg, wall_s=time.time() - t0, **extra)); f.flush()
        print(f"{a.part} {fam:8} scale={scale:<5} seed={seed} n_fam={n_fam:6d} depth={depth:.3f} horizon={hor}", flush=True)
    f.close()

if __name__ == "__main__":
    main()
