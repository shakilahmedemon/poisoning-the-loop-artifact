#!/usr/bin/env python3
"""Mechanism controls for additional seeds (wacatac, constrained trigger, 1% injection).

Runs the same arms and settings as run_mech_multi.py, run_mech_split.py and
run_conf_control.py, but saves every run to its own file as soon as it finishes:

    <out-dir>/<arm>__s<seed>.csv     (written to a temp file, then renamed)

Finished runs are skipped, so an interrupted queue resumes where it stopped.
A file named PAUSE in the working directory stops the runner between runs.
The controls take their monthly malware counts from the uncertainty run of the
same seed (identical to the sweep files used by run_conf_control.py).
"""
import argparse, contextlib, io, os, time
import numpy as np, pandas as pd
import attack_smart_mech as M

ARMS = ["clean_random", "random", "uncertainty", "match_fam", "match_mal",
        "unc_mal_only", "unc_ben_only", "conf_ben_only"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--seeds", type=int, nargs="+", default=[5, 6, 7, 8, 9])
    ap.add_argument("--arms", nargs="+", default=ARMS, choices=ARMS)
    ap.add_argument("--family", default="wacatac")
    ap.add_argument("--out-dir", default="./figs/mech10")
    a = ap.parse_args()

    path = lambda arm, s: os.path.join(a.out_dir, f"{arm}__s{s}.csv")
    todo = [(s, arm) for s in a.seeds for arm in a.arms if not os.path.exists(path(arm, s))]
    print(f"{len(a.seeds) * len(a.arms) - len(todo)}/{len(a.seeds) * len(a.arms)} runs done", flush=True)
    if not todo:
        return
    os.makedirs(a.out_dir, exist_ok=True)

    import lightgbm as lgb
    X, y, meta, period = M.load_bodmas(a.data_dir)
    months = list(period.unique()); tm = 1
    idx = np.where(period.isin(months[:tm]).values)[0]
    trig_cache = {}

    def trigger(seed):
        if seed not in trig_cache:
            clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, n_jobs=-1,
                                     verbose=-1, random_state=seed).fit(X[idx], y[idx])
            with contextlib.redirect_stdout(io.StringIO()):
                trig_cache[seed] = M.build_trigger(clf, X[idx], y[idx], 8, constrained=True, seed=seed)
        return trig_cache[seed]

    def schedule(arm, seed):
        """Same schedule dicts as the original scripts, built from this seed's uncertainty run."""
        if arm in ("clean_random", "random", "uncertainty"):
            return None
        u = pd.read_csv(path("uncertainty", seed))
        fam = dict(zip(u.month.astype(str), u.sel_fam_n))
        mal = dict(zip(u.month.astype(str), u.sel_mal_n))
        if arm in ("match_fam", "match_mal"):
            return {"random_matched_fam": fam, "random_matched_mal": mal}
        if arm in ("unc_mal_only", "unc_ben_only"):
            return {"unc_mal_only": mal, "unc_ben_only": mal}
        return {"conf_ben_only": mal}

    strategy = {"clean_random": ("random", False), "random": ("random", True), "uncertainty": ("uncertainty", True),
                "match_fam": ("random_matched_fam", True), "match_mal": ("random_matched_mal", True),
                "unc_mal_only": ("unc_mal_only", True), "unc_ben_only": ("unc_ben_only", True),
                "conf_ben_only": ("conf_ben_only", True)}

    for seed, arm in todo:
        if os.path.exists("./PAUSE"):
            print("[paused] remove PAUSE and run the same command to resume", flush=True)
            return
        if arm not in ("clean_random", "random", "uncertainty") and not os.path.exists(path("uncertainty", seed)):
            raise SystemExit(f"uncertainty run for seed {seed} is missing; include it in --arms")
        t0 = time.time()
        strat, attack = strategy[arm]
        df, _ = M.run(strat, attack, X, y, meta, period, months, tm, 0.05, 0.01, 4,
                      a.family, trigger(seed), seed=seed, schedule=schedule(arm, seed))
        df.insert(0, "arm", arm); df.insert(0, "seed", seed)
        tmp = path(arm, seed) + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, path(arm, seed))
        print(f"mech s{seed} {arm} [{time.time() - t0:.0f}s]", flush=True)
    print("all requested runs done", flush=True)


if __name__ == "__main__":
    main()
