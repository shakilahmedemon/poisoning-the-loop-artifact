# Replication package (anonymized for double-blind review)

Code and raw per-run results for the submission "Poisoning the Loop".
No data or malware binary is included.

## Data (obtain separately)
* BODMAS feature vectors + metadata: public dataset (Yang et al., DLS 2021). Place
  `bodmas.npz` and `bodmas_metadata.csv` in `./data/`.
* BODMAS malware binaries are available from the dataset maintainers on request and
  MUST NOT be redistributed; `malware_static_validation.py` reads them from the
  maintainers' zip directly into memory (nothing is written to disk or executed).

## Layout
* `code/attack_smart.py` - continual-retraining loop, SHAP trigger, clean-label attack.
* `code/run_seeds.py` - RQ2 multi-seed runs. `code/run_family_sweep.py`,
  `code/run_volume_and_seeds.py` - RQ3 sweep, multi-seed, volume intervention, mixed policy.
* `code/attack_native_mlp.py`, `attack_smart_mlp.py` - RQ4 cross-architecture.
* `code/attack_vs_cade.py`, `attack_vs_iforest.py`, `cade_defense.py` - drift-gate defenses.
* `code/problem_space_validation.py` (benign carrier), `malware_static_validation.py`
  (real malware, in memory), `run_ablation_malware_trigger.py` (malware-side ablation).
* `results/` - every CSV behind the tables and figures.

## Reproduce
`pip install lightgbm shap scikit-learn pandas numpy pefile torch`; then run the scripts
above from the repository root with `--data-dir ./data`. CPU only; the full suite takes
under 48 hours on a consumer workstation.
