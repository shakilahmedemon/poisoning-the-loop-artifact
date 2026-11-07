#!/usr/bin/env python3
"""
Is seed-pool size why the policy effect appears on BODMAS and not on EMBER?
Seed pools differ 9x (BODMAS 844, EMBER 7,692). Two controlled swaps, same loop as run_sweep.py:
  --dataset ember  --pool-size 844   EMBER with its first-month pool randomly subsampled to 844 rows
  --dataset bodmas --train-months 2  BODMAS seed pool = first two periods (8,368 rows, close to EMBER's)
Without either option the run reproduces the stored sweep (a check of the environment).

Per seed: clean_random, clean_uncertainty, {random,uncertainty}_r<rate>. The trigger is derived on the swapped pool.
Each run is saved on completion to figs/seedpool/<tag>/<key>.csv (atomic); a restart skips finished runs.
Pause: a file named PAUSE in this folder stops between runs. Existing result files are never modified.
"""
import argparse, contextlib, io, json, os, time
import numpy as np, pandas as pd
import attack_smart_mech as M

PAUSE_FILE = "./PAUSE"


def load_ember():
    d = "./data/ember_clean"; n = 100000
    meta_all = pd.read_parquet(f"{d}/ember_clean_metadata.parquet", columns=["label", "appeared", "family"]).iloc[:n].reset_index(drop=True)
    keep = meta_all.appeared.str.startswith("2018").to_numpy()
    Xmm = np.memmap(f"{d}/X_first100k.f32", dtype=np.float32, mode="r", shape=(n, 2099))
    X = np.ascontiguousarray(Xmm[np.where(keep)[0]]); mk = meta_all[keep].reset_index(drop=True)
    y = mk.label.to_numpy().astype(np.int8)
    meta = pd.DataFrame({"family": mk.family.fillna("").astype(str).to_numpy()})
    period = pd.Series(pd.PeriodIndex(mk.appeared, freq="M").to_numpy(), index=meta.index)
    return X, y, meta, period, sorted(period.unique())


def subsample_pool(X, y, meta, period, months, tm, size, seed):
    """Keep `size` random rows of the seed pool and every later row. Later months are untouched."""
    pool = np.where(period.isin(months[:tm]).values)[0]
    if size <= 0 or size >= len(pool):
        return X, y, meta, period
    rs = np.random.RandomState(seed + 31)
    keep = np.sort(np.concatenate([rs.choice(pool, size=size, replace=False), np.where(~period.isin(months[:tm]).values)[0]]))
    return X[keep], y[keep], meta.iloc[keep].reset_index(drop=True), period.iloc[keep].reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=["ember", "bodmas"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--rates", type=float, nargs="+", default=[0.0025, 0.01])
    ap.add_argument("--pool-size", type=int, default=0)
    ap.add_argument("--train-months", type=int, default=1)
    ap.add_argument("--max-runs", type=int, default=0)
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args()
    tag = f"{a.dataset}_pool{a.pool_size}_tm{a.train_months}"
    outdir = f"./figs/seedpool/{tag}"
    family = "emotet" if a.dataset == "ember" else "wacatac"
    items = []
    for sd in a.seeds:
        items += [dict(seed=sd, arm="clean_random", strategy="random", attack=False, rate=0.0),
                  dict(seed=sd, arm="clean_uncertainty", strategy="uncertainty", attack=False, rate=0.0)]
        for r in a.rates:
            items += [dict(seed=sd, arm=f"random_r{r}", strategy="random", attack=True, rate=r),
                      dict(seed=sd, arm=f"uncertainty_r{r}", strategy="uncertainty", attack=True, rate=r)]
    path = lambda it: os.path.join(outdir, f"{family}__s{it['seed']}__{it['arm']}.csv")
    done0 = sum(os.path.exists(path(i)) for i in items)
    print(f"{done0}/{len(items)} runs done [{tag}]", flush=True)
    if a.status or done0 == len(items):
        return
    os.makedirs(outdir, exist_ok=True)
    import lightgbm as lgb
    if a.dataset == "ember":
        X0, y0, meta0, period0, months = load_ember(); constrained = False
    else:
        X0, y0, meta0, period0 = M.load_bodmas("./data"); months = list(period0.unique()); constrained = True
    tm = a.train_months
    cache = {}
    session = 0
    for it in items:
        if os.path.exists(path(it)):
            continue
        if os.path.exists(PAUSE_FILE) or (a.max_runs and session >= a.max_runs):
            print("[stopped] run the same command again to resume", flush=True)
            break
        sd = it["seed"]; t0 = time.time()
        if sd not in cache:
            cache.clear()
            X, y, meta, period = subsample_pool(X0, y0, meta0, period0, months, tm, a.pool_size, sd)
            idx = np.where(period.isin(months[:tm]).values)[0]
            clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, n_jobs=-1, verbose=-1, random_state=sd).fit(X[idx], y[idx])
            with contextlib.redirect_stdout(io.StringIO()):
                trig = M.build_trigger(clf, X[idx], y[idx], 8, constrained=constrained, seed=sd)
            cache[sd] = (X, y, meta, period, trig)
            with open(os.path.join(outdir, "triggers.jsonl"), "a") as f:
                f.write(json.dumps(dict(seed=sd, pool_rows=len(idx), pool_malware=int((y[idx] == 1).sum()), trigger={int(k): float(v) for k, v in trig.items()})) + "\n")
        X, y, meta, period, trig = cache[sd]
        df, _ = M.run(it["strategy"], it["attack"], X, y, meta, period, months, tm, 0.05, it["rate"] if it["attack"] else 0.01, 4, family, trig, seed=sd)
        df.insert(0, "rate", it["rate"]); df.insert(0, "arm", it["arm"]); df.insert(0, "seed", sd)
        tmp = path(it) + ".tmp"
        df.to_csv(tmp, index=False)
        os.replace(tmp, path(it))
        session += 1
        print(f"{tag} s{sd} {it['arm']} [{time.time() - t0:.0f}s]", flush=True)
    print(f"{sum(os.path.exists(path(i)) for i in items)}/{len(items)} runs done [{tag}]", flush=True)


if __name__ == "__main__":
    main()
