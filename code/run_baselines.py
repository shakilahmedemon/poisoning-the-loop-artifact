#!/usr/bin/env python3
"""
Baselines for the clean-label backdoor in the retraining loop.

Per (policy, target family, seed) group, five independent runs (each saved on completion):
  clean         no attack. Stamped-recall evaluated with BOTH the attacker's trigger (att) and a random trigger (rnd).
  rand          attack identical to the paper's (4-month window, eta=1%) but the poison AND the evaluation use a
                RANDOM trigger (8 random allowlisted fields, benign-mode values)  -> does SHAP guidance matter?
  burst         attack with the attacker's trigger and the SAME total poison (324-ish) injected all in the first
                month instead of spread over four months                          -> does injection timing matter?
  static_clean  ONE-SHOT, no retraining loop: a model trained once on the seed pool (never updated), no poison
  static_poison ONE-SHOT: the same model trained once on seed pool + the same total poison (att trigger), never updated
                -> what the retraining loop changes about persistence (Severi et al.'s setting vs the loop)
The paper's own attacked run (att trigger, spread) already exists in figs/defense_eval/*atk_nogate* and is reused.

Pause / resume (same as run_defense_eval.py): Ctrl+C once = finish current run then stop; twice = abort now;
file named PAUSE in this folder = stop between runs. Finished runs live in figs/baselines_eval/<key>.csv
(atomic write), so a restart skips them.
"""
import argparse, contextlib, io, os, signal, time
import numpy as np, pandas as pd
from tqdm import tqdm
from attack_smart import (load_bodmas, build_trigger, apply_trigger, select_random, select_uncertainty,
                          SAFE_FEATURE_INDICES)

OUTDIR = "./figs/baselines_eval"
PAUSE_FILE = "./PAUSE"
NJOBS = -1
KINDS = ["clean", "rand", "burst", "static_clean", "static_poison"]
STOP = {"flag": False, "count": 0}


def on_sigint(sig, frm):
    STOP["count"] += 1
    if STOP["count"] >= 2:
        print("\n[abort] second Ctrl+C: aborting now (finished runs are safe)", flush=True); raise KeyboardInterrupt
    STOP["flag"] = True
    print("\n[pause] Ctrl+C: finishing the current run, then stopping (Ctrl+C again aborts now)", flush=True)


def plan(seeds):
    return [dict(strategy=s, target=t, seed=sd, kind=k)
            for sd in seeds for s in ("random", "uncertainty") for t in ("wacatac", "sfone") for k in KINDS]


def key_of(it):
    return f"{it['strategy']}__{it['target']}__s{it['seed']}__{it['kind']}"


def is_done(it):
    return os.path.exists(os.path.join(OUTDIR, key_of(it) + ".csv"))


def save_run(it, df):
    os.makedirs(OUTDIR, exist_ok=True)
    final = os.path.join(OUTDIR, key_of(it) + ".csv"); tmp = final + ".tmp"
    df.to_csv(tmp, index=False)
    with open(tmp, "r+b") as f:
        f.flush(); os.fsync(f.fileno())
    os.replace(tmp, final)


def mode_trigger(Xb, feats):
    out = {}
    for f in feats:
        u, c = np.unique(Xb[:, f], return_counts=True)
        out[int(f)] = float(u[np.argmax(c)])
    return out


def poison_counts(ctx):
    """Per-month poison counts of the paper's attack (eta=1% of month volume, capped by benign available)."""
    X, y, period, months = ctx["X"], ctx["y"], ctx["period"], ctx["months"]
    counts = []
    for m in months[1:5]:
        ym = y[(period == m).values]
        counts.append(min(max(1, int(round(0.01 * len(ym)))), int((ym == 0).sum())))
    return counts


def fam_recall(clf, Xm, mal_fam_mask, trig):
    if mal_fam_mask.sum() == 0:
        return float("nan")
    return float((clf.predict_proba(apply_trigger(Xm[mal_fam_mask], trig))[:, 1] >= 0.5).mean())


def run_loop(it, ctx, trig_att, trig_rnd, bar):
    import lightgbm as lgb
    from sklearn.metrics import f1_score, recall_score
    X, y, meta, period, months = ctx["X"], ctx["y"], ctx["meta"], ctx["period"], ctx["months"]
    strategy, target, seed, kind = it["strategy"], it["target"], it["seed"], it["kind"]
    poison_trig = {"clean": None, "rand": trig_rnd, "burst": trig_att}[kind]
    rng = np.random.RandomState(seed)
    train_idx = list(np.where(period.isin(months[:1]).values)[0])
    X_extra, y_extra = [], []
    pc = poison_counts(ctx)
    schedule = {}
    if kind == "rand":
        schedule = {m: n for m, n in zip(months[1:5], pc)}
    elif kind == "burst":
        schedule = {months[1]: sum(pc)}
    rows = []
    for m in months[1:]:
        Xtr = np.vstack([X[train_idx]] + ([np.vstack(X_extra)] if X_extra else []))
        ytr = np.concatenate([y[train_idx]] + ([np.asarray(y_extra)] if y_extra else []))
        if len(np.unique(ytr)) < 2:
            bar.update(1); continue
        clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, n_jobs=NJOBS, verbose=-1,
                                 random_state=seed).fit(Xtr, ytr)
        sel = (period == m).values
        if sel.sum() < 30 or len(np.unique(y[sel])) < 2:
            bar.update(1); continue
        Xm_idx = np.where(sel)[0]; Xm, ym = X[Xm_idx], y[Xm_idx]
        proba = clf.predict_proba(Xm)[:, 1]; pred = (proba >= 0.5).astype(int)
        fm = (meta.loc[Xm_idx, "family"].values == target) & (ym == 1)
        rows.append(dict(month=str(m), n=len(ym), f1=f1_score(ym, pred, zero_division=0),
                         family_recall_clean=float(recall_score(ym[fm], pred[fm], zero_division=0)) if fm.sum() else float("nan"),
                         wm_att=fam_recall(clf, Xm, fm, trig_att), wm_rnd=fam_recall(clf, Xm, fm, trig_rnd)))
        k = max(1, int(round(0.05 * len(ym))))
        pick = select_random(rng, len(ym), k) if strategy == "random" else select_uncertainty(proba, k)
        train_idx.extend(Xm_idx[pick].tolist())
        n_inj = schedule.get(m, 0)
        if n_inj > 0:
            bl = np.where(ym == 0)[0]
            pb = rng.choice(bl, size=min(n_inj, len(bl)), replace=False)
            X_extra.append(apply_trigger(Xm[pb], poison_trig)); y_extra.extend([0] * len(pb))
        bar.update(1)
    return pd.DataFrame(rows)


def run_static(it, ctx, trig_att, trig_rnd, bar):
    import lightgbm as lgb
    from sklearn.metrics import f1_score, recall_score
    X, y, meta, period, months = ctx["X"], ctx["y"], ctx["meta"], ctx["period"], ctx["months"]
    target, seed = it["target"], it["seed"]
    idx0 = np.where(period.isin(months[:1]).values)[0]
    Xtr, ytr = X[idx0], y[idx0]
    if it["kind"] == "static_poison":
        rs = np.random.RandomState(seed + 5)
        px = []
        for m, n in zip(months[1:5], poison_counts(ctx)):
            ym = y[(period == m).values]; Xmm = X[(period == m).values]
            bl = np.where(ym == 0)[0]
            px.append(apply_trigger(Xmm[rs.choice(bl, size=n, replace=False)], trig_att))
        Xp = np.vstack(px)
        Xtr = np.vstack([Xtr, Xp]); ytr = np.concatenate([ytr, np.zeros(len(Xp), dtype=int)])
    clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, n_jobs=NJOBS, verbose=-1,
                             random_state=seed).fit(Xtr, ytr)
    rows = []
    for m in months[1:]:
        sel = (period == m).values
        if sel.sum() < 30 or len(np.unique(y[sel])) < 2:
            bar.update(1); continue
        Xm_idx = np.where(sel)[0]; Xm, ym = X[Xm_idx], y[Xm_idx]
        pred = (clf.predict_proba(Xm)[:, 1] >= 0.5).astype(int)
        fm = (meta.loc[Xm_idx, "family"].values == target) & (ym == 1)
        rows.append(dict(month=str(m), n=len(ym), f1=f1_score(ym, pred, zero_division=0),
                         family_recall_clean=float(recall_score(ym[fm], pred[fm], zero_division=0)) if fm.sum() else float("nan"),
                         wm_att=fam_recall(clf, Xm, fm, trig_att), wm_rnd=fam_recall(clf, Xm, fm, trig_rnd)))
        bar.update(1)
    return pd.DataFrame(rows)


def main():
    global NJOBS
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--max-runs", type=int, default=0)
    a = ap.parse_args(); NJOBS = a.n_jobs
    items = plan(a.seeds)
    for f in os.listdir(OUTDIR) if os.path.isdir(OUTDIR) else []:
        if f.endswith(".tmp"): os.remove(os.path.join(OUTDIR, f))
    done0 = sum(is_done(i) for i in items)
    print(f"{done0}/{len(items)} runs done ({100*done0/len(items):.1f}%)")
    if a.status or done0 == len(items): return
    if os.path.exists(PAUSE_FILE):
        print("[note] delete the PAUSE file to run:", os.path.abspath(PAUSE_FILE)); return
    signal.signal(signal.SIGINT, on_sigint)
    print("[..] loading BODMAS", flush=True)
    X, y, meta, period = load_bodmas(a.data_dir)
    months = list(period.unique())
    ctx = dict(X=X, y=y, meta=meta, period=period, months=months)
    import lightgbm as lgb
    n_months = len(months) - 1
    seedcache = {}
    outer = tqdm(total=len(items), initial=done0, desc="runs", unit="run", ascii=True, position=0,
                 bar_format="{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}] {postfix}")
    session = 0
    try:
        for it in items:
            if is_done(it): continue
            if STOP["flag"] or os.path.exists(PAUSE_FILE) or (a.max_runs and session >= a.max_runs):
                outer.write("[stopped] progress is saved; run the same command again to resume."); break
            sd = it["seed"]
            if sd not in seedcache:
                seedcache.clear()
                idx0 = np.where(period.isin(months[:1]).values)[0]
                clf0 = lgb.LGBMClassifier(n_estimators=300, num_leaves=63, learning_rate=0.08, n_jobs=NJOBS,
                                          verbose=-1, random_state=sd).fit(X[idx0], y[idx0])
                with contextlib.redirect_stdout(io.StringIO()):
                    trig = build_trigger(clf0, X[idx0], y[idx0], 8, constrained=True, seed=sd)
                rr = np.random.RandomState(sd + 99)
                feats = rr.choice(np.array(SAFE_FEATURE_INDICES), 8, replace=False)
                trn = mode_trigger(X[idx0][y[idx0] == 0], feats)
                seedcache[sd] = (trig, trn)
            trig_att, trig_rnd = seedcache[sd]
            outer.set_postfix_str(f"{it['strategy'][:4]}/{it['target']}/s{sd} {it['kind']}")
            inner = tqdm(total=n_months, desc="  months", leave=False, ascii=True, position=1)
            df = (run_static if it["kind"].startswith("static") else run_loop)(it, ctx, trig_att, trig_rnd, inner)
            inner.close()
            for k_, v_ in it.items(): df[k_] = v_
            save_run(it, df)
            session += 1
            outer.update(1)
        else:
            outer.write("[finished] all runs done.")
    except KeyboardInterrupt:
        outer.write("[aborted] finished runs are saved; run the same command again to resume.")
    finally:
        outer.close()
        print(f"{sum(is_done(i) for i in items)}/{len(items)} runs done")


if __name__ == "__main__":
    main()
