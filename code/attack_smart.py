#!/usr/bin/env python3
"""
Clean-label SHAP-guided backdoor, adapted from Severi et al., "Explanation-
Guided Backdoor Poisoning Attacks Against Malware Classifiers" (USENIX Sec
2021), for the continual retraining setting instead of their one-shot split.

Difference from attack_naive.py: that script mislabels real target-family
malware as benign, so a label audit catches it immediately. This script
never touches a label. It edits a handful of feature values on real benign
samples to a trigger pattern the model already associates with goodware and
adds them to training with their true label. At evaluation time (not
training) the same trigger is stamped onto the target family's real malware
to check whether it now evades detection.

Trigger construction follows Severi et al. Sec. 4's "Independent" strategy:
run SHAP TreeExplainer on the seed model over a sample of its training pool,
sum raw SHAP per feature, take the --trigger-size features with the most
negative sums (strongest pull toward benign), and set each to its mean
value among benign pool samples. This is a simplified stand-in for the
paper's CountSHAP value selection (which also weighs per-value frequency);
worth revisiting if the effect looks marginal.

Feature-space only for now, no problem-space constraints (see
problem_space_validation.py; realizability requirements follow Pierazzi et
al., S&P 2020).

Usage:
    python attack_smart.py --data-dir ./data --train-months 1 \
        --strategy random --injection-rate 0.01 --injection-months 4 \
        --target-family wacatac --trigger-size 8
"""

import argparse
import os
import sys
import time

import numpy as np
import pandas as pd


# ----------------------------------------------------------------- loading

def load_bodmas(data_dir):
    with np.load(os.path.join(data_dir, "bodmas.npz"), allow_pickle=False) as z:
        X, y = z["X"].astype(np.float32), z["y"].astype(int)
    meta = pd.read_csv(os.path.join(data_dir, "bodmas_metadata.csv"))
    meta["_ts"] = pd.to_datetime(meta["timestamp"], errors="coerce")
    meta["family"] = meta["family"].fillna("")
    keep = meta["_ts"].notna().values
    X, y, meta = X[keep], y[keep], meta.loc[keep].reset_index(drop=True)
    order = np.argsort(meta["_ts"].values, kind="stable")
    X, y, meta = X[order], y[order], meta.iloc[order].reset_index(drop=True)
    malware_start = meta.loc[y == 1, "_ts"].min()
    trim = (meta["_ts"] >= malware_start).values
    X, y, meta = X[trim], y[trim], meta.loc[trim].reset_index(drop=True)
    period = meta["_ts"].dt.to_period("M")
    return X, y, meta, period


# ----------------------------------------------------------------- problem-space allowlist
#
# EMBER feature layout (BODMAS uses the same format, feature_version=2,
# 2381 dims total, verified against the official extractor source,
# elastic/ember/ember/features.py):
#   [0,256)      ByteHistogram: normalized (sums to 1), so moving one bin
#                moves all 255 others. Not independently editable.
#   [256,512)    ByteEntropyHistogram: same problem.
#   [512,616)    StringExtractor: offsets 0/1/2 are numstrings/avlength/
#                printables, scalar counts, editable via padding but coarse
#                and visible in file size. Offsets 3-98 are printabledist, a
#                96-bin normalized histogram, not independently editable for
#                the same reason as above. Offsets 99-103 are entropy/paths/
#                urls/registry/MZ, scalar counts.
#   [616,626)    GeneralFileInfo: size is a safe scalar; the rest
#                (has_debug/exports/imports/has_relocations/has_resources/
#                has_signature/has_tls/symbols/vsize) are tied to real code
#                structure, so flipping has_signature without a valid
#                signature, or exports/imports counts without matching
#                functions, breaks the binary or is trivially inconsistent.
#   [626,688)    HeaderFileInfo: offset 0 is timestamp, safe and a single
#                field. Offsets 1-50 are FeatureHasher output over
#                machine/characteristics/subsystem/dll_characteristics/magic,
#                hashed, so there's no way to solve backward for a value
#                that produces a target hash. Offsets 51-61 are 11 scalar
#                version/size fields, safe.
#   [688,943)    SectionInfo: offsets 0-4 are num_sections/num_zero_size/
#                num_empty_name/num_RX/num_W, achievable by adding a section
#                via a PE-editing utility (same approach Severi et al.
#                built). Offsets 5-254 are five FeatureHashers over
#                per-section name/size/entropy/vsize/characteristics, hashed.
#   [943,2223)   ImportsInfo: entirely hashed.
#   [2223,2351)  ExportsInfo: entirely hashed.
#   [2351,2381)  DataDirectories: size+RVA pairs per directory; setting
#                these without a real, valid structure at that address/size
#                corrupts the file.
#
# This mirrors Severi et al.'s own finding (USENIX Sec 2021, Sec. 6.1): after
# dropping hashed features and features with cross-dependencies, they were
# left with 17 of EMBER's ~2351 features actually safe to edit independently.
# The allowlist below is built the same way, from first principles against
# the source above, not copied from their paper since we don't have their
# exact index list. SAFE_FEATURE_INDICES below is the pool the constrained
# attacker's SHAP search is restricted to.

SAFE_FEATURE_INDICES = (
    [512, 513, 514, 611, 612, 613, 614, 615] +     # string scalar counts
    [616] +                                          # size
    [626] +                                          # timestamp
    list(range(677, 688)) +                          # 11 header version/size scalars
    list(range(688, 693))                            # 5 section-count scalars
)


# ----------------------------------------------------------------- trigger

def build_trigger(seed_clf, Xtr, ytr, trigger_size, sample_size=3000, seed=0,
                  constrained=False):
    """SHAP-guided trigger: trigger_size (feature_idx -> value) pairs that
    are strongly and consistently goodware-oriented in the seed model."""
    import shap

    rng = np.random.RandomState(seed)
    n = min(sample_size, len(Xtr))
    idx = rng.choice(len(Xtr), size=n, replace=False)
    Xs, ys = Xtr[idx], ytr[idx]

    explainer = shap.TreeExplainer(seed_clf)
    sv = explainer.shap_values(Xs)
    if isinstance(sv, list):          # older shap API: [class0, class1]
        sv = sv[1]
    assert sv.ndim == 2 and sv.shape == Xs.shape

    feature_sum = sv.sum(axis=0)      # per-feature: sum of SHAP contributions
    # most negative = most strongly, consistently goodware-pushing
    ranked = np.argsort(feature_sum)
    if constrained:
        allowed = set(SAFE_FEATURE_INDICES)
        ranked = np.array([f for f in ranked if f in allowed])
        print(f"[..] CONSTRAINED search: restricted to {len(allowed)} "
              f"independently-editable EMBER fields (see SAFE_FEATURE_INDICES)")
    top_features = ranked[:trigger_size]

    benign_mask = ys == 0
    if benign_mask.sum() < 5:
        sys.exit("[FAIL] too few benign samples in the seed pool to build a trigger")

    # Value selection: use the mode of each feature among real benign
    # samples, not the mean. The mean of an integer-valued field (e.g. linker
    # version, section count) is a value that no real PE could ever carry
    # (a "linker version 11.54" doesn't exist), which would silently break
    # problem-space realizability even for an otherwise-editable feature.
    # The mode is guaranteed to be a value some real benign file actually
    # has. This is the fast approximation of Severi et al.'s CountSHAP
    # value selector, which explicitly restricts to observed values for
    # exactly this reason.
    def mode_value(col):
        uniq, counts = np.unique(col, return_counts=True)
        return float(uniq[np.argmax(counts)])

    Xb = Xs[benign_mask]
    trigger_values = [mode_value(Xb[:, f]) for f in top_features]

    trigger = dict(zip(top_features.tolist(), trigger_values))
    print(f"[..] trigger ({trigger_size} features), by SHAP goodware-push "
          f"(most negative first):")
    for f in top_features:
        print(f"      feat[{f:4d}]  sum_shap={feature_sum[f]:12.3f}  "
              f"trigger_value={trigger[f]:12.4f}")
    return trigger


def apply_trigger(X, trigger):
    """Return a COPY of X with the trigger features overwritten."""
    Xw = X.copy()
    for f, v in trigger.items():
        Xw[:, f] = v
    return Xw


# ----------------------------------------------------------------- selector

def select_random(rng, n_pool, k):
    k = min(k, n_pool)
    return rng.choice(n_pool, size=k, replace=False)


def select_uncertainty(proba, k):
    k = min(k, len(proba))
    return np.argsort(np.abs(proba - 0.5))[:k]


# ----------------------------------------------------------------- the loop

def run(strategy, attack, X, y, meta, period, months, train_months,
       label_rate, injection_rate, injection_months, target_family,
       trigger, seed=0, eval_triggers=None):
    # eval_triggers: optional {name: trigger_dict}, each applied only to
    # evaluation-time malware (never to training) and logged as an extra
    # column family_recall_wm_<name>. Used by run_ablation_malware_trigger.py.
    import lightgbm as lgb
    from sklearn.metrics import f1_score, precision_score, recall_score

    rng = np.random.RandomState(seed)
    train_mask = period.isin(months[:train_months]).values
    train_idx = list(np.where(train_mask)[0])
    # X_extra / y_extra hold the clean-label backdoored copies, synthetic
    # rows that don't exist in the original dataset, so they can't be tracked by
    # index alone. Everything here carries its true label; nothing is lied
    # about, which is the whole point of a clean-label attack.
    X_extra, y_extra = [], []

    injection_window = set(months[train_months:train_months + injection_months]) \
        if attack else set()
    n_injected_total = 0
    rows = []

    for m in months[train_months:]:
        Xtr = np.vstack([X[train_idx]] + ([np.vstack(X_extra)] if X_extra else []))
        ytr = np.concatenate([y[train_idx]] + ([np.asarray(y_extra)] if y_extra else []))
        if len(np.unique(ytr)) < 2:
            continue

        clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63,
                                 learning_rate=0.08, n_jobs=-1, verbose=-1,
                                 random_state=seed)
        clf.fit(Xtr, ytr)

        sel = (period == m).values
        if sel.sum() < 30 or len(np.unique(y[sel])) < 2:
            continue
        Xm_idx = np.where(sel)[0]
        Xm, ym = X[Xm_idx], y[Xm_idx]
        proba = clf.predict_proba(Xm)[:, 1]
        pred = (proba >= 0.5).astype(int)

        row = dict(month=str(m), n=len(ym), train_size=len(ytr),
                   f1=f1_score(ym, pred, zero_division=0),
                   prec=precision_score(ym, pred, zero_division=0),
                   rec=recall_score(ym, pred, zero_division=0))

        # family recall on the untouched real samples (sanity check;
        # should track the clean baseline closely, since we never
        #      poison malware labels in this attack) ----
        fam_mask = (meta.loc[Xm_idx, "family"].values == target_family) & (ym == 1)
        if fam_mask.sum() > 0:
            row["family_n"] = int(fam_mask.sum())
            row["family_recall_clean"] = float(
                recall_score(ym[fam_mask], pred[fam_mask], zero_division=0))

            # the actual attack metric: does the trigger let this
            # month's real target-family malware evade detection?
            # watermarked copies are evaluated only, never trained on.
            Xfam_wm = apply_trigger(Xm[fam_mask], trigger)
            pred_wm = (clf.predict_proba(Xfam_wm)[:, 1] >= 0.5).astype(int)
            row["family_recall_watermarked"] = float(
                recall_score(np.ones(fam_mask.sum()), pred_wm, zero_division=0))
            for name, trig in (eval_triggers or {}).items():
                pw = (clf.predict_proba(apply_trigger(Xm[fam_mask], trig))[:, 1] >= 0.5).astype(int)
                row[f"family_recall_wm_{name}"] = float(
                    recall_score(np.ones(fam_mask.sum()), pw, zero_division=0))
        else:
            for name in (eval_triggers or {}):
                row[f"family_recall_wm_{name}"] = float("nan")
            row["family_n"] = 0
            row["family_recall_clean"] = float("nan")
            row["family_recall_watermarked"] = float("nan")
        rows.append(row)

        # ---- normal selection: pick k samples to label honestly ----
        k = max(1, int(round(label_rate * len(ym))))
        if strategy == "random":
            pick_local = select_random(rng, len(ym), k)
        elif strategy == "uncertainty":
            pick_local = select_uncertainty(proba, k)
        elif strategy == "mixed":
            # hybrid policy: half uncertainty, half uniform random over the rest
            pu = select_uncertainty(proba, k // 2)
            rest = np.setdiff1d(np.arange(len(ym)), pu)
            pr = rng.choice(rest, size=min(k - len(pu), len(rest)), replace=False)
            pick_local = np.concatenate([pu, pr])
        else:
            sys.exit(f"[FAIL] unknown strategy {strategy!r}")
        train_idx.extend(Xm_idx[pick_local].tolist())

        # the attack: clean-label backdoored benign copies
        if m in injection_window:
            benign_local = np.where(ym == 0)[0]
            n_inject = max(1, int(round(injection_rate * len(ym))))
            n_inject = min(n_inject, len(benign_local))
            if n_inject > 0:
                pick_benign = rng.choice(benign_local, size=n_inject, replace=False)
                backdoored = apply_trigger(Xm[pick_benign], trigger)
                X_extra.append(backdoored)
                y_extra.extend([0] * n_inject)  # true label, no lie
                n_injected_total += n_inject

    df = pd.DataFrame(rows)
    return df, n_injected_total


def aut(scores):
    s = np.asarray([v for v in scores if not np.isnan(v)], dtype=float)
    if len(s) < 2:
        return float("nan")
    return float(np.mean((s[1:] + s[:-1]) / 2.0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="./data")
    ap.add_argument("--train-months", type=int, default=1)
    ap.add_argument("--strategy", choices=["random", "uncertainty"], default="random")
    ap.add_argument("--label-rate", type=float, default=0.05)
    ap.add_argument("--injection-rate", type=float, default=0.01,
                    help="fraction of each injection month's BENIGN samples "
                         "backdoored and added (with their TRUE label)")
    ap.add_argument("--injection-months", type=int, default=4)
    ap.add_argument("--trigger-size", type=int, default=8,
                    help="number of features in the SHAP-guided trigger")
    ap.add_argument("--constrained", action="store_true",
                    help="restrict the SHAP trigger search to EMBER fields "
                         "that are independently editable in a real PE file "
                         "(SAFE_FEATURE_INDICES), the problem-space-aware "
                         "attacker, vs. the default unrestricted feature-"
                         "space attacker")
    ap.add_argument("--target-family", default="wacatac")
    ap.add_argument("--seed", type=int, default=0,
                    help="controls model training, SHAP background sampling, "
                         "and which points get poisoned each month; vary "
                         "this across runs to get real variance, not just "
                         "a point estimate")
    ap.add_argument("--out", default="./figs/fig_attack_smart.pdf")
    args = ap.parse_args()

    print("[..] loading BODMAS")
    X, y, meta, period = load_bodmas(args.data_dir)
    months = list(period.unique())
    n_in_test = int((meta.loc[(~period.isin(months[:args.train_months])).values
                              & (y == 1), "family"] == args.target_family).sum())
    print(f"[..] target family: {args.target_family!r} ({n_in_test} samples "
          f"across the test period)")

    # ---- seed model + SHAP trigger, computed once up front ----
    import lightgbm as lgb
    train_mask = period.isin(months[:args.train_months]).values
    train_idx = np.where(train_mask)[0]
    print(f"\n[..] training seed model on {len(train_idx)} samples "
          f"({months[0]}..{months[args.train_months-1]}) to derive the trigger")
    seed_clf = lgb.LGBMClassifier(n_estimators=300, num_leaves=63,
                                  learning_rate=0.08, n_jobs=-1, verbose=-1,
                                  random_state=args.seed)
    seed_clf.fit(X[train_idx], y[train_idx])
    trigger = build_trigger(seed_clf, X[train_idx], y[train_idx], args.trigger_size,
                            constrained=args.constrained, seed=args.seed)

    print(f"\n[..] running BASELINE (no attack), strategy={args.strategy}")
    t0 = time.time()
    df_clean, _ = run(args.strategy, False, X, y, meta, period, months,
                      args.train_months, args.label_rate, args.injection_rate,
                      args.injection_months, args.target_family, trigger,
                      seed=args.seed)
    print(f"[OK] baseline done [{time.time()-t0:.0f}s]  "
          f"AUT(F1)={aut(df_clean['f1'].values):.4f}  "
          f"AUT(family_recall_watermarked)="
          f"{aut(df_clean['family_recall_watermarked'].values):.4f}  "
          f"<- backdoor evasion baseline: an UNTRAINED trigger should barely "
          f"move recall from family_recall_clean")

    print(f"\n[..] running ATTACKED (clean-label inject {args.injection_rate:.0%}/month "
          f"for {args.injection_months} months), strategy={args.strategy}")
    t0 = time.time()
    df_attack, n_inj = run(args.strategy, True, X, y, meta, period, months,
                           args.train_months, args.label_rate, args.injection_rate,
                           args.injection_months, args.target_family, trigger,
                           seed=args.seed)
    print(f"[OK] attacked done [{time.time()-t0:.0f}s]  "
          f"AUT(F1)={aut(df_attack['f1'].values):.4f}  "
          f"AUT(family_recall_watermarked)="
          f"{aut(df_attack['family_recall_watermarked'].values):.4f}  "
          f"total clean-label points injected={n_inj}")

    # ---- table ----
    print("\n" + "=" * 96)
    print(f"{'month':>9} | {'F1 clean':>9} {'F1 atk':>7} | "
          f"{'fam_rec(real)':>13} | {'fam_rec(wm) clean':>17} {'fam_rec(wm) atk':>15} | window?")
    print("-" * 96)
    injection_window = set(months[args.train_months:
                                  args.train_months + args.injection_months])
    for i in range(min(len(df_clean), len(df_attack))):
        m = df_clean["month"].iloc[i]
        in_window = "yes" if pd.Period(m, freq="M") in injection_window else ""
        print(f"{m:>9} | {df_clean['f1'].iloc[i]:9.4f} {df_attack['f1'].iloc[i]:7.4f} | "
              f"{df_clean['family_recall_clean'].iloc[i]:13.4f} | "
              f"{df_clean['family_recall_watermarked'].iloc[i]:17.4f} "
              f"{df_attack['family_recall_watermarked'].iloc[i]:15.4f} | {in_window}")
    print("=" * 96)

    # ---- verdict ----
    post_injection = [i for i, m in enumerate(df_clean["month"])
                      if pd.Period(m, freq="M") not in injection_window]
    depth, horizon = 0.0, 0
    for i in post_injection:
        c = df_clean["family_recall_watermarked"].iloc[i]
        a = df_attack["family_recall_watermarked"].iloc[i]
        if pd.isna(c) or pd.isna(a):
            continue
        gap = c - a
        depth = max(depth, gap)
        if gap > 0.15:
            horizon += 1
        else:
            break
    f1_gap = abs(aut(df_clean["f1"].values) - aut(df_attack["f1"].values))
    real_fam_gap = abs(aut(df_clean["family_recall_clean"].values) -
                       aut(df_attack["family_recall_clean"].values))

    print(f"\nbackdoor evasion depth (max watermarked-recall gap, post-injection): "
          f"{depth:.4f}")
    print(f"persistence horizon (months gap>0.15 after injection stops): {horizon}")
    print(f"overall AUT(F1) shift (stealth check, want small): {f1_gap:.4f}")
    print(f"real (unwatermarked) family recall shift (should be ~0, confirms "
          f"the attack is surgical, not a blanket family attack): {real_fam_gap:.4f}")

    # ---- figure ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8,
                         "axes.linewidth": 0.6, "figure.dpi": 200})
    fig, axes = plt.subplots(3, 1, figsize=(3.33, 5.0), sharex=True)

    ax = axes[0]
    ax.plot(range(len(df_clean)), df_clean["f1"], marker="o", ms=3, lw=1.1,
            color="#0072B2", label="clean")
    ax.plot(range(len(df_attack)), df_attack["f1"], marker="s", ms=3, lw=1.1,
            color="#D55E00", label="attacked")
    ax.axvspan(-0.5, args.injection_months - 0.5, color="#999999", alpha=0.15)
    ax.set_ylabel("overall F1"); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.4)
    ax.legend(frameon=False, fontsize=7, loc="lower left")
    ax.set_title("stealth: overall F1", fontsize=7)

    ax = axes[1]
    ax.plot(range(len(df_clean)), df_clean["family_recall_clean"], marker="o", ms=3,
            lw=1.1, color="#0072B2", label="clean")
    ax.plot(range(len(df_attack)), df_attack["family_recall_clean"], marker="s", ms=3,
            lw=1.1, color="#D55E00", label="attacked")
    ax.axvspan(-0.5, args.injection_months - 0.5, color="#999999", alpha=0.15)
    ax.set_ylabel("real recall"); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_title("surgical check: UNWATERMARKED family recall (should overlap)", fontsize=7)

    ax = axes[2]
    ax.plot(range(len(df_clean)), df_clean["family_recall_watermarked"], marker="o",
            ms=3, lw=1.1, color="#0072B2", label="clean")
    ax.plot(range(len(df_attack)), df_attack["family_recall_watermarked"], marker="s",
            ms=3, lw=1.1, color="#D55E00", label="attacked")
    ax.axvspan(-0.5, args.injection_months - 0.5, color="#999999", alpha=0.15)
    ax.set_ylabel("watermarked recall"); ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.25, lw=0.4)
    ax.set_title("the attack: recall on TRIGGER-bearing malware", fontsize=7)
    step = max(1, len(df_clean) // 6)
    ax.set_xticks(range(0, len(df_clean), step))
    ax.set_xticklabels([df_clean["month"].iloc[i] for i in range(0, len(df_clean), step)],
                       rotation=45, ha="right", fontsize=6)
    ax.set_xlabel("test month")

    fig.tight_layout(pad=0.3)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out)
    print(f"\n[OK] wrote {args.out}")

    df_clean.to_csv("./figs/attack_smart_clean.csv", index=False)
    df_attack.to_csv("./figs/attack_smart_attacked.csv", index=False)

    print("\n" + "=" * 96)
    if depth > 0.15 and f1_gap < 0.02 and real_fam_gap < 0.05:
        print("SIGNAL: clean-label backdoor works. Watermarked malware evades")
        print("detection, overall F1 and REAL family recall barely move --")
        print("this attack passes both the stealth check AND the label-audit")
        print("check that would have caught the naive dirty-label attack.")
    elif depth > 0.15:
        print("partial signal: the backdoor works but leaks into either overall")
        print("F1 or real family recall, check the numbers above.")
    else:
        print("NO SIGNAL: try a larger --trigger-size, higher --injection-rate,")
        print("or note that clean-label attacks are known to need MORE poison")
        print("than dirty-label ones (Severi et al. report needing ~1% of the")
        print("full training set even in the unrestricted setting).")
    print("=" * 96)


if __name__ == "__main__":
    main()
