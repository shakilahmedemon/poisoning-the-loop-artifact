#!/usr/bin/env python3
"""
Insertion-time defenses against the clean-label backdoor.

For every (policy, target family, seed) group:
    base           clean pipeline, no attack, no gate                  (reference for the gap)
    atk_nogate     attacked pipeline, no gate                          (reproduces attack_smart.py)
    <defense> x quantile x mode   attacked pipeline with an insertion-time gate
        defense  in cade | iforest | spectral
        quantile in 0.05 | 0.10 | 0.20      (share of candidates quarantined: threshold sweep)
        mode     perclass = quantile applied separately within each claimed class (what the paper text says)
                 global   = quantile over all candidates jointly (what attack_vs_cade.py / attack_vs_iforest.py
                            actually did); run only for cade/iforest at q=0.05 to document the difference.
    spectral = adapted spectral-signature defense (Tran et al., NeurIPS 2018): per month and claimed class,
               top singular vector of the centered, standardized candidate batch; outlier score = squared
               projection. Adapted because the original needs a trained network's representation.

Every run is independent and is saved the moment it finishes (figs/defense_eval/<key>.csv, written to a
temp file then renamed). A run exists on disk <=> it is done. So you can stop at any time and restart:
finished runs are skipped, nothing starts from the beginning.

Pause / stop:
    * Ctrl+C once  -> finishes the current run (<~2 min), saves it, exits cleanly.
    * Ctrl+C twice -> aborts immediately (only the run in progress is lost).
    * create a file named PAUSE in this folder -> same as one Ctrl+C, checked between runs.
Progress bars: outer = runs done / total with elapsed + ETA, inner = months of the current run.

Usage:  python run_defense_eval.py                # everything (seeds 0-4)
        python run_defense_eval.py --seeds 0      # only seed 0
        python run_defense_eval.py --status       # show how much is done, run nothing
"""
import argparse, contextlib, io, os, signal, sys, time, json
import numpy as np, pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler

from attack_smart import load_bodmas, build_trigger, apply_trigger, select_random, select_uncertainty
from cade_defense import train_cae, compute_centroids, anomaly_score
from attack_vs_iforest import fit_iforest_per_class, iforest_anomaly_score

OUTDIR = "./figs/defense_eval"
PAUSE_FILE = "./PAUSE"
NJOBS = -1
QS = [0.05, 0.10, 0.20]
DEFENSES = ["cade", "iforest", "spectral"]
STOP = {"flag": False, "count": 0}


def on_sigint(sig, frm):
    STOP["count"] += 1
    if STOP["count"] >= 2:
        print("\n[abort] second Ctrl+C: aborting now (the run in progress is lost; finished runs are safe)", flush=True)
        raise KeyboardInterrupt
    STOP["flag"] = True
    print("\n[pause] Ctrl+C received: finishing the current run, saving it, then stopping. "
          "(press Ctrl+C again to abort immediately)", flush=True)


def plan(seeds):
    """Ordered list of run keys (seed-major so partial results are already a complete picture)."""
    items = []
    for seed in seeds:
        for strategy in ("random", "uncertainty"):
            for target in ("wacatac", "sfone"):
                g = dict(strategy=strategy, target=target, seed=seed)
                items.append(dict(g, arm="base", defense="none", q=0.0, mode="-"))
                items.append(dict(g, arm="atk_nogate", defense="none", q=0.0, mode="-"))
                for d in DEFENSES:
                    for q in QS:
                        items.append(dict(g, arm="atk_gate", defense=d, q=q, mode="perclass"))
                for d in ("cade", "iforest"):
                    items.append(dict(g, arm="atk_gate", defense=d, q=0.05, mode="global"))
    return items


def key_of(it):
    return f"{it['strategy']}__{it['target']}__s{it['seed']}__{it['arm']}__{it['defense']}__q{it['q']:.2f}__{it['mode']}"


def is_done(it):
    return os.path.exists(os.path.join(OUTDIR, key_of(it) + ".csv"))


def spectral_scores(Z):
    n = len(Z)
    if n < 3:
        return np.zeros(n)
    Zc = Z - Z.mean(axis=0)
    _, _, vt = np.linalg.svd(Zc, full_matrices=False)
    return (Zc @ vt[0]) ** 2


def keep_mask(scores, claimed, q, mode):
    keep = np.ones(len(scores), dtype=bool)
    if mode == "global":
        keep = scores <= np.quantile(scores, 1.0 - q)
    else:
        for c in np.unique(claimed):
            m = claimed == c
            keep[m] = scores[m] <= np.quantile(scores[m], 1.0 - q)
    return keep


def make_gate(defense, seed, X0, y0, cache):
    """Return scorer(Xs_scaled, claimed_label)->anomaly scores, plus the scaler. Cached per (defense, seed)."""
    ck = (defense, seed)
    if ck in cache:
        return cache[ck]
    scaler = StandardScaler().fit(X0)
    Xs = scaler.transform(X0)
    if defense == "cade":
        with contextlib.redirect_stdout(io.StringIO()):
            cae = train_cae(Xs, y0, seed=seed)
            cen, med = compute_centroids(cae, Xs, y0)
        scorer = lambda Z, c: anomaly_score(cae, Z, c, cen, med)
    elif defense == "iforest":
        models = fit_iforest_per_class(Xs, y0, seed=seed)
        scorer = lambda Z, c: iforest_anomaly_score(models, Z, c)
    else:
        scorer = lambda Z, c: spectral_scores(Z)
    cache.clear() if len(cache) > 6 else None
    cache[ck] = (scorer, scaler)
    return cache[ck]


def run_one(it, ctx, trigger, gate, bar):
    import lightgbm as lgb
    from sklearn.metrics import f1_score, recall_score
    X, y, meta, period, months = ctx["X"], ctx["y"], ctx["meta"], ctx["period"], ctx["months"]
    strategy, target, seed = it["strategy"], it["target"], it["seed"]
    attack = it["arm"] != "base"
    rng = np.random.RandomState(seed)
    train_idx = list(np.where(period.isin(months[:1]).values)[0])
    X_extra, y_extra = [], []
    window = set(months[1:1 + 4]) if attack else set()
    st = dict(n_poison_offered=0, n_poison_quarantined=0, n_honest_offered=0, n_honest_quarantined=0)
    rows = []
    for m in months[1:]:
        Xtr = np.vstack([X[train_idx]] + ([np.vstack(X_extra)] if X_extra else []))
        ytr = np.concatenate([y[train_idx]] + ([np.asarray(y_extra)] if y_extra else []))
        if len(np.unique(ytr)) < 2:
            bar.update(1); continue
        clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, n_jobs=NJOBS,
                                 verbose=-1, random_state=seed).fit(Xtr, ytr)
        sel = (period == m).values
        if sel.sum() < 30 or len(np.unique(y[sel])) < 2:
            bar.update(1); continue
        Xm_idx = np.where(sel)[0]; Xm, ym = X[Xm_idx], y[Xm_idx]
        proba = clf.predict_proba(Xm)[:, 1]; pred = (proba >= 0.5).astype(int)
        row = dict(month=str(m), n=len(ym), train_size=len(ytr), f1=f1_score(ym, pred, zero_division=0))
        fm = (meta.loc[Xm_idx, "family"].values == target) & (ym == 1)
        if fm.sum() > 0:
            row["family_recall_clean"] = float(recall_score(ym[fm], pred[fm], zero_division=0))
            pw = (clf.predict_proba(apply_trigger(Xm[fm], trigger))[:, 1] >= 0.5).astype(int)
            row["family_recall_watermarked"] = float(pw.mean())
        else:
            row["family_recall_clean"] = row["family_recall_watermarked"] = float("nan")
        rows.append(row)

        k = max(1, int(round(0.05 * len(ym))))
        pick = select_random(rng, len(ym), k) if strategy == "random" else select_uncertainty(proba, k)
        cand_X, cand_y, cand_src = Xm[pick], y[Xm_idx[pick]], Xm_idx[pick]
        poison_X = None
        if m in window:
            bl = np.where(ym == 0)[0]
            n_inj = min(max(1, int(round(0.01 * len(ym)))), len(bl))
            if n_inj > 0:
                pb = rng.choice(bl, size=n_inj, replace=False)
                poison_X = apply_trigger(Xm[pb], trigger)
                st["n_poison_offered"] += n_inj
        if it["arm"] == "atk_gate":
            scorer, scaler = gate
            allX = np.vstack([cand_X] + ([poison_X] if poison_X is not None else []))
            allY = np.concatenate([cand_y] + ([np.zeros(len(poison_X), dtype=int)] if poison_X is not None else []))
            Z = scaler.transform(allX)
            scores = np.empty(len(allY))
            for c in np.unique(allY):
                mm = allY == c
                scores[mm] = scorer(Z[mm], c)
            keep = keep_mask(scores, allY, it["q"], it["mode"])
            n_real = len(cand_X)
            train_idx.extend(cand_src[np.where(keep[:n_real])[0]].tolist())
            st["n_honest_offered"] += n_real
            st["n_honest_quarantined"] += int((~keep[:n_real]).sum())
            if poison_X is not None:
                pk = keep[n_real:]
                st["n_poison_quarantined"] += int((~pk).sum())
                if pk.any():
                    X_extra.append(poison_X[pk]); y_extra.extend([0] * int(pk.sum()))
        else:
            train_idx.extend(cand_src.tolist())
            if poison_X is not None:
                X_extra.append(poison_X); y_extra.extend([0] * len(poison_X))
        bar.update(1)
    df = pd.DataFrame(rows)
    for k_, v_ in {**it, **st}.items():
        df[k_] = v_
    return df


def save_run(it, df):
    os.makedirs(OUTDIR, exist_ok=True)
    final = os.path.join(OUTDIR, key_of(it) + ".csv")
    tmp = final + ".tmp"
    df.to_csv(tmp, index=False)
    with open(tmp, "r+b") as f:
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, final)          # atomic: the file exists only when the run is complete


def status(items):
    done = sum(is_done(i) for i in items)
    print(f"{done}/{len(items)} runs done ({100*done/len(items):.1f}%)")
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--max-runs", type=int, default=0, help="stop after this many runs this session (0 = no limit)")
    ap.add_argument("--shard", default="0/1", help="k/N: this process takes runs with index %% N == k (open N terminals to run N processes in parallel; all share the same figs/defense_eval folder)")
    ap.add_argument("--n-jobs", type=int, default=-1, help="LightGBM threads per process (use 16//N when sharding)")
    a = ap.parse_args()
    global NJOBS
    NJOBS = a.n_jobs
    k_, n_ = (int(v) for v in a.shard.split("/"))
    items = [it for i, it in enumerate(plan(a.seeds)) if i % n_ == k_]
    for f in os.listdir(OUTDIR) if os.path.isdir(OUTDIR) else []:   # clean half-written temp files from a hard kill
        if f.endswith(".tmp"): os.remove(os.path.join(OUTDIR, f))
    done0 = status(items)
    if a.status or done0 == len(items):
        return
    if os.path.exists(PAUSE_FILE):
        print(f"[note] a file named PAUSE exists in this folder; delete it to run.  ({os.path.abspath(PAUSE_FILE)})"); return
    signal.signal(signal.SIGINT, on_sigint)
    print("[..] loading BODMAS", flush=True)
    X, y, meta, period = load_bodmas(a.data_dir)
    months = list(period.unique())
    ctx = dict(X=X, y=y, meta=meta, period=period, months=months)
    n_months = len(months) - 1
    import lightgbm as lgb
    cache, seedcache = {}, {}
    outer = tqdm(total=len(items), initial=done0, desc="runs", unit="run", ascii=True, position=0,
                 bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")
    session = 0
    try:
        for it in items:
            if is_done(it):
                continue
            if STOP["flag"] or os.path.exists(PAUSE_FILE) or (a.max_runs and session >= a.max_runs):
                outer.write("[stopped] progress is saved; run the same command again to resume.")
                break
            sd = it["seed"]
            if sd not in seedcache:
                seedcache.clear()
                idx0 = np.where(period.isin(months[:1]).values)[0]
                clf0 = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, n_jobs=NJOBS,
                                          verbose=-1, random_state=sd).fit(X[idx0], y[idx0])
                with contextlib.redirect_stdout(io.StringIO()):
                    trig = build_trigger(clf0, X[idx0], y[idx0], 8, constrained=True, seed=sd)
                seedcache[sd] = (idx0, trig)
            idx0, trig = seedcache[sd]
            gate = make_gate(it["defense"], sd, X[idx0], y[idx0], cache) if it["arm"] == "atk_gate" else None
            outer.set_postfix_str(f"{it['strategy'][:4]}/{it['target']}/s{sd} {it['arm']} {it['defense']} q{it['q']:.2f} {it['mode']}")
            inner = tqdm(total=n_months, desc="  months", leave=False, ascii=True, position=1)
            t0 = time.time()
            df = run_one(it, ctx, trig, gate, inner)
            inner.close()
            save_run(it, df)
            session += 1
            outer.update(1)
        else:
            outer.write("[finished] all runs done.")
    except KeyboardInterrupt:
        outer.write("[aborted] finished runs are saved; run the same command again to resume.")
    finally:
        outer.close()
        status(items)


if __name__ == "__main__":
    main()
