# Replication package (anonymized for double-blind review)

Code and raw per-run results for the submission "Poisoning the Loop". No
data or malware binary is included here.

## Data
BODMAS feature vectors + metadata are a public dataset (Yang et al., DLS
2021) - grab them yourself and drop `bodmas.npz` and `bodmas_metadata.csv`
into `./data/`. The BODMAS malware binaries are only available from the
dataset maintainers on request, and they can't be redistributed;
`malware_static_validation.py` reads them straight from the maintainers'
zip into memory and never writes anything to disk or executes it.

## Layout
- `code/attack_smart.py` - the continual-retraining loop, SHAP trigger,
  clean-label attack. Everything else builds on this.
- `code/run_seeds.py` - RQ2 multi-seed runs.
- `code/run_family_sweep.py`, `code/run_volume_and_seeds.py` - RQ3 family
  sweep, multi-seed re-run, volume intervention, mixed policy.
- `code/plot_family_sweep.py` - bar-chart view of the family sweep, reads
  the two CSVs above and needs no BODMAS data or retraining.
- `code/attack_native_mlp.py`, `code/attack_smart_mlp.py` - RQ4
  cross-architecture.
- `code/attack_vs_cade.py`, `code/attack_vs_iforest.py`, `code/cade_defense.py`
  - drift-gate defenses.
- `code/problem_space_validation.py` (benign carrier) and
  `code/malware_static_validation.py` (real malware, in memory), plus
  `code/run_ablation_malware_trigger.py` for the malware-side ablation.
- `results/` - every CSV behind the paper's tables and figures.

## Reproducing

```
pip install lightgbm shap scikit-learn pandas numpy pefile torch
```

then run whichever script from the repo root with `--data-dir ./data`.
Everything is CPU only; the full suite runs under 48 hours on a normal
workstation, nothing here needs a GPU or a cluster.
