#!/usr/bin/env python3
"""
Defender-side backdoor regression test for a continual-retraining pipeline.

After every retraining the defender stamps natural malware of each family
with a SHAP-derived benign-direction trigger, built with the same procedure
the attacker uses, here from the same seed model (an upper bound on what
the defender can know), and compares recall on stamped vs. natural
samples. A family whose stamped recall falls far below its natural recall
gets flagged. Runs clean and attacked pipelines and records, per month and
per family, natural/stamped recall so detection power and false-alarm rate
can be computed offline (analyze_all.py).
"""
import argparse, csv, os, time
import numpy as np, pandas as pd
from attack_smart import load_bodmas, build_trigger, run

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-probe", type=int, default=15)
    a = ap.parse_args()
    import lightgbm as lgb
    X, y, meta, period = load_bodmas(a.data_dir)
    months = list(period.unique())
    fam_counts = meta.loc[y == 1, "family"].value_counts()
    probe = [f for f in fam_counts.index[: a.n_probe]]
    out = "./figs/results_backdoor_probe.csv"
    new = not os.path.exists(out)
    done = set() if new else {(r.strategy, r.target, int(r.seed), r.arm) for r in pd.read_csv(out, usecols=["strategy","target","seed","arm"]).drop_duplicates().itertuples()}
    f = None; w = None
    cells = [(s, t, sd) for sd in a.seeds for s in ("random", "uncertainty") for t in ("wacatac", "sfone")]
    for strategy, target, seed in cells:
        if (strategy, target, seed, "attacked") in done: continue
        t0 = time.time()
        idx = np.where(period.isin(months[:1]).values)[0]
        clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, n_jobs=-1,
                                 verbose=-1, random_state=seed).fit(X[idx], y[idx])
        trig = build_trigger(clf, X[idx], y[idx], 8, constrained=True, seed=seed)
        for arm, attack in (("clean", False), ("attacked", True)):
            df, _ = run(strategy, attack, X, y, meta, period, months, 1, 0.05, 0.01, 4, target, trig,
                        seed=seed, probe_families=probe)
            df.insert(0, "arm", arm); df.insert(0, "seed", seed); df.insert(0, "target", target); df.insert(0, "strategy", strategy)
            if f is None:
                f = open(out, "a", newline=""); 
                cols = list(df.columns)
                if not new:
                    cols = list(pd.read_csv(out, nrows=0).columns)
                w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                if new: w.writeheader(); new = False
            w.writerows(df.to_dict("records")); f.flush()
        print(f"probe {strategy:11} {target:8} seed={seed} [{time.time()-t0:.0f}s]", flush=True)

if __name__ == "__main__":
    main()
