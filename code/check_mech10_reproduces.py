#!/usr/bin/env python3
"""Check that run_mech10.py reproduces stored seed-1 runs of run_mech_multi.py exactly."""
import sys
import pandas as pd

old = pd.read_csv("./figs/mech_multi.csv")
bad = 0
for arm in ("clean_random", "uncertainty"):
    new = pd.read_csv(f"./figs/mech10_verify/{arm}__s1.csv")
    ref = old[(old.seed == 1) & (old.arm == arm)].reset_index(drop=True)
    cols = [c for c in ref.columns if c in new.columns and c not in ("seed", "arm")]
    num = [c for c in cols if pd.api.types.is_numeric_dtype(ref[c])]
    if len(new) != len(ref):
        print(arm, "row count differs", len(new), len(ref)); bad += 1; continue
    diff = (new[num] - ref[num]).abs().max().max()
    same_text = all((new[c].astype(str) == ref[c].astype(str)).all() for c in cols if c not in num)
    print(f"{arm}: {len(num)} numeric columns, max abs difference {diff}, text columns equal: {same_text}")
    bad += int(diff != 0 or not same_text)
print("REPRODUCES EXACTLY" if not bad else "MISMATCH")
sys.exit(1 if bad else 0)
