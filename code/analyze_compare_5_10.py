"""5-seed vs 10-seed comparison for the seed-pool and mechanism controls. Read-only."""
import glob, re, sys
import numpy as np, pandas as pd
from scipy import stats

sys.stdout.reconfigure(encoding="utf-8")
F = "figs/"
WM = "family_recall_watermarked"


def gaps(run, ref):
    a = run.sort_values("month").set_index("month")[WM]
    b = ref.sort_values("month").set_index("month")[WM]
    return (b - a).dropna().iloc[4:].values


def summ(g):
    h = 0
    for x in g:
        if x > 0.15: h += 1
        else: break
    return dict(gap=float(g.mean()), depth=max([0.0] + list(g[: h + 1])), horizon=h, maxgap=float(g.max()))


# ---------------- seed pool
print("=== SEED POOL (1% unless noted)")
SETS = [("BODMAS 2-month, pool 844", "bodmas_pool844_tm2", 0.01), ("BODMAS 2-month, pool 3000", "bodmas_pool3000_tm2", 0.01),
        ("BODMAS 2-month, pool 8368", "bodmas_pool0_tm2", 0.01), ("BODMAS 2-month, pool 8368, 0.5%", "bodmas_pool0_tm2", 0.005),
        ("EMBER pool 844", "ember_pool844_tm1", 0.01), ("EMBER pool 844, 0.25%", "ember_pool844_tm1", 0.0025),
        ("EMBER pool 844, 0.1%", "ember_pool844_tm1", 0.001)]
rows = []
for name, tag, rate in SETS:
    fam = "emotet" if tag.startswith("ember") else "wacatac"
    for s in range(10):
        rec = {}
        for pol in ("random", "uncertainty"):
            run = pd.read_csv(f"{F}seedpool/{tag}/{fam}__s{s}__{pol}_r{rate}.csv")
            ref = pd.read_csv(f"{F}seedpool/{tag}/{fam}__s{s}__clean_{pol}.csv")
            g = gaps(run, ref)
            rec.update({f"{pol}_{k}": v for k, v in summ(g).items()})
            rec[f"{pol}_poison"] = float((run.train_size - ref.train_size).iloc[4] / run.train_size.iloc[4])
        rows.append(dict(set=name, seed=s, **rec))
P = pd.DataFrame(rows)
for name, g in P.groupby("set", sort=False):
    for n in (5, 10):
        x = g[g.seed < n]
        p = stats.ttest_rel(x.random_gap, x.uncertainty_gap).pvalue
        print(f"{name:34s} n={n:2d}  random {x.random_gap.mean():.3f}  uncert {x.uncertainty_gap.mean():.3f}  "
              f"h_rand {x.random_horizon.mean():.1f}  lower {int((x.uncertainty_gap < x.random_gap).sum())}/{n}  p={p:.3g}  "
              f"poison {100 * x.random_poison.mean():.1f}%")
print("per-seed random gaps (fig:pool dots):")
for name in ("BODMAS 2-month, pool 844", "BODMAS 2-month, pool 3000", "BODMAS 2-month, pool 8368"):
    g = P[P.set == name]
    print(f"  {name}: random {g.random_gap.round(3).tolist()}  uncert {g.uncertainty_gap.round(3).tolist()}")

# ---------------- mechanism
print("\n=== MECHANISM (wacatac, 1%, gap against the clean random run of the same seed)")
runs = {}
sel = pd.read_csv(F + "mech_selection.csv")
runs[(0, "clean_random")] = sel[(sel.strategy == "random") & (~sel.attack)]
runs[(0, "random")] = sel[(sel.strategy == "random") & (sel.attack)]
runs[(0, "uncertainty")] = sel[(sel.strategy == "uncertainty") & (sel.attack)]
mt = pd.read_csv(F + "mech_matched.csv")
runs[(0, "match_fam")] = mt[mt.strategy == "random_matched_fam"]
runs[(0, "match_mal")] = mt[mt.strategy == "random_matched_mal"]
mm = pd.read_csv(F + "mech_multi.csv")
for (s, arm), d in mm.groupby(["seed", "arm"]):
    runs[(s, arm)] = d
for s in range(5):
    sp = pd.read_csv(F + ("mech_split.csv" if s == 0 else f"mech_split_seed{s}.csv"))
    for arm, d in sp.groupby("arm"):
        runs[(s, arm)] = d
    runs[(s, "conf_ben_only")] = pd.read_csv(F + f"conf_ben_seed{s}.csv")
for f in glob.glob(F + "mech10/*.csv"):
    m = re.match(r".*[\\/](.+)__s(\d+)\.csv$", f)
    runs[(int(m.group(2)), m.group(1))] = pd.read_csv(f)

ARMS = ["conf_ben_only", "random", "unc_ben_only", "uncertainty", "match_mal", "match_fam", "unc_mal_only"]
M = []
for s in range(10):
    ref = runs[(s, "clean_random")]
    for arm in ARMS:
        M.append(dict(seed=s, arm=arm, **summ(gaps(runs[(s, arm)], ref))))
M = pd.DataFrame(M)
piv = M.pivot(index="seed", columns="arm", values="gap")
for arm in ARMS:
    for n in (5, 10):
        x = M[(M.arm == arm) & (M.seed < n)]
        line = f"{arm:14s} n={n:2d}  gap {x.gap.mean():.3f}  depth {x.depth.mean():.2f}  horizon {x.horizon.mean():.1f}  maxgap {x.maxgap.mean():.2f}"
        if arm != "random":
            r = piv.loc[piv.index < n]
            d = r[arm] - r["random"]
            line += f"  vs random: lower in {int((d < 0).sum())}/{n}, higher in {int((d > 0).sum())}/{n}, p={stats.ttest_rel(r[arm], r['random']).pvalue:.3g}"
        print(line)
    print(f"   per seed: {piv[arm].round(3).tolist()}")
print("conf_ben_only horizon per seed:", M[M.arm == "conf_ben_only"].horizon.tolist())
