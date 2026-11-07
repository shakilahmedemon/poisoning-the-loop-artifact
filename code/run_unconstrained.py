#!/usr/bin/env python3
"""
Pilot: does the BODMAS policy effect (uncertainty << random) survive an UNCONSTRAINED trigger?
EMBER uses an unconstrained trigger (any feature) and shows no policy effect; BODMAS uses the editable-field
allowlist. Same BODMAS data, same loop (run_sweep.py), target wacatac, only the trigger changes:
  --trigger unconstrained   SHAP trigger over ALL features (default)
  --trigger constrained     control: the paper's allowlisted trigger (should reproduce figs/sweep_bodmas_seed*.csv)

Per seed: clean_random, clean_uncertainty, and {random,uncertainty}_r<rate> for each rate. Every run is saved on
completion to figs/unconstrained_pilot/<trigger>/<key>.csv (atomic), so nothing is lost on a stop and a restart skips
finished runs.

Pause / resume (same as run_baselines.py): Ctrl+C once = finish the current run then stop; twice = abort now (finished
runs are safe); a file named PAUSE in this folder = stop between runs. Run the same command again to resume.
"""
import argparse, contextlib, io, os, signal, sys, time, json
import numpy as np, pandas as pd
import attack_smart_mech as M

PAUSE_FILE = "./PAUSE"
STOP = {"flag": False, "count": 0}


def on_sigint(sig, frm):
    STOP["count"] += 1
    if STOP["count"] >= 2:
        print("\n[abort] second Ctrl+C: aborting now (finished runs are safe)", flush=True); raise KeyboardInterrupt
    STOP["flag"] = True
    print("\n[pause] Ctrl+C: finishing the current run, then stopping (Ctrl+C again aborts now)", flush=True)


def plan(seeds, rates):
    items = []
    for sd in seeds:
        items += [dict(seed=sd, arm="clean_random", strategy="random", attack=False, rate=0.0),
                  dict(seed=sd, arm="clean_uncertainty", strategy="uncertainty", attack=False, rate=0.0)]
        for r in rates:
            items += [dict(seed=sd, arm=f"random_r{r}", strategy="random", attack=True, rate=r),
                      dict(seed=sd, arm=f"uncertainty_r{r}", strategy="uncertainty", attack=True, rate=r)]
    return items


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--rates", type=float, nargs="+", default=[0.0025, 0.01])
    ap.add_argument("--trigger", choices=["unconstrained", "constrained"], default="unconstrained")
    ap.add_argument("--family", default="wacatac")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--max-runs", type=int, default=0, help="stop after this many runs this session (0 = no limit)")
    a = ap.parse_args()
    outdir = f"./figs/unconstrained_pilot/{a.trigger}"
    items = plan(a.seeds, a.rates)
    key = lambda it: f"{a.family}__s{it['seed']}__{it['arm']}"
    path = lambda it: os.path.join(outdir, key(it) + ".csv")
    for f in os.listdir(outdir) if os.path.isdir(outdir) else []:
        if f.endswith(".tmp"): os.remove(os.path.join(outdir, f))
    done0 = sum(os.path.exists(path(i)) for i in items)
    print(f"{done0}/{len(items)} runs done ({100*done0/len(items):.1f}%)  [trigger={a.trigger}, family={a.family}]")
    if a.status or done0 == len(items): return
    if os.path.exists(PAUSE_FILE):
        print("[note] delete the PAUSE file to run:", os.path.abspath(PAUSE_FILE)); return
    signal.signal(signal.SIGINT, on_sigint)
    os.makedirs(outdir, exist_ok=True)
    import lightgbm as lgb
    from tqdm import tqdm
    print("[..] loading BODMAS", flush=True)
    X, y, meta, period = M.load_bodmas(a.data_dir)
    months = list(period.unique()); tm = 1; n_months = len(months) - tm
    cache = {}
    outer = tqdm(total=len(items), initial=done0, desc="runs", unit="run", ascii=True, position=0,
                 bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")
    inner = tqdm(total=n_months, desc="  months", leave=False, ascii=True, position=1)
    M.PROGRESS = lambda: inner.update(1)
    session = 0
    try:
        for it in items:
            if os.path.exists(path(it)): continue
            if STOP["flag"] or os.path.exists(PAUSE_FILE) or (a.max_runs and session >= a.max_runs):
                outer.write("[stopped] progress is saved; run the same command again to resume."); break
            sd = it["seed"]
            if sd not in cache:
                cache.clear()
                idx = np.where(period.isin(months[:tm]).values)[0]
                clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, n_jobs=-1, verbose=-1,
                                         random_state=sd).fit(X[idx], y[idx])
                with contextlib.redirect_stdout(io.StringIO()):
                    cache[sd] = M.build_trigger(clf, X[idx], y[idx], 8, constrained=(a.trigger == "constrained"), seed=sd)
                with open(os.path.join(outdir, "triggers.jsonl"), "a") as f:
                    f.write(json.dumps(dict(seed=sd, trigger={int(k): float(v) for k, v in cache[sd].items()})) + "\n")
            outer.set_postfix_str(f"s{sd} {it['arm']}")
            inner.reset(total=n_months)
            df, _ = M.run(it["strategy"], it["attack"], X, y, meta, period, months, tm, 0.05,
                          it["rate"] if it["attack"] else 0.01, 4, a.family, cache[sd], seed=sd)
            df.insert(0, "rate", it["rate"]); df.insert(0, "arm", it["arm"]); df.insert(0, "seed", sd)
            df["trigger"] = a.trigger
            tmp = path(it) + ".tmp"
            df.to_csv(tmp, index=False)
            with open(tmp, "r+b") as f:
                f.flush(); os.fsync(f.fileno())
            os.replace(tmp, path(it))
            session += 1
            outer.update(1)
        else:
            outer.write("[finished] all runs done.")
    except KeyboardInterrupt:
        outer.write("[aborted] finished runs are saved; run the same command again to resume.")
    finally:
        inner.close(); outer.close()
        print(f"{sum(os.path.exists(path(i)) for i in items)}/{len(items)} runs done")


if __name__ == "__main__":
    main()
