# Poisoning the Loop: Replication Package

Code and raw per-run results for the submission "Poisoning the Loop: Clean-Label
Backdoors in Continually Retrained Malware Detectors".

The package contains every runner and analysis script behind the paper's results,
the per-run result files they produced, and the pinned environment used to verify
them. No dataset and no malware binary is included.

## Repository layout

```
code/                    runners, analyses and shared modules
results/                 per-run result files (CSV, CSV.GZ, JSONL)
  _out10/, _out_final/   printed output of the analysis scripts
  baselines_eval/        baseline runs
  defense_eval/          insertion-time gate runs
  seedpool/              seed-pool control runs
  unconstrained_pilot/   unconstrained-trigger runs
requirements.lock.txt    exact package versions used for verification
```

## Setup

Python 3.13. Everything runs on CPU.

```
pip install -r requirements.lock.txt
```

With these versions, three clean and attacked BODMAS runs and three EMBER runs
were regenerated from the raw data and matched the stored results exactly.

## Data

Place the datasets under `./data/`:

- **BODMAS** (Yang et al., DLS 2021): `bodmas.npz` and `bodmas_metadata.csv`.
- **EMBER Cleaned** (public Hugging Face release): the metadata parquet and the
  first 100,000 feature rows as `ember_clean/X_first100k.f32`.

The BODMAS malware binaries are available only from the dataset maintainers and
cannot be redistributed. `malware_static_validation.py` reads the disarmed
binaries from the maintainers' zip into memory only. It never writes them to disk
and never executes them.

## Running

Run the scripts from the repository root with `code/` on the import path. They
read and write `./figs`, so copy the stored results there first:

```
cp -r results figs
export PYTHONPATH=code        # PowerShell: $env:PYTHONPATH = "code"
python code/analyze_seedpool.py
```

The `analyze_*.py` scripts need no dataset: they recompute the summaries from the
stored per-run files, and their printed output is kept in `results/_out10/` and
`results/_out_final/` for comparison.

Runners save each run as soon as it finishes and skip finished runs on restart. A
file named `PAUSE` in the working directory stops a runner between runs.

## Paper results and the scripts that produce them

| Paper result | Runner | Analysis | Stored results |
|---|---|---|---|
| Dirty-label baseline (RQ1) | `attack_naive.py` | | `attack_naive_*.csv` |
| Main result, 10 seeds (RQ2) | `run_seeds.py` | `analyze_all.py` | `results_seeds.csv` |
| Which selected samples matter (RQ2) | `run_mech*.py`, `run_conf_control.py` | | `mech_*.csv`, `conf_ben_seed*.csv` |
| Injection-rate sweep and second dataset (RQ2) | `run_sweep.py`, `run_ember.py` | `analyze_policy_difference.py` | `sweep_*.csv`, `ember_arms_seed*.csv` |
| Unconstrained-trigger control (RQ2) | `run_unconstrained.py` | `analyze_unconstrained_pilot.py` | `unconstrained_pilot/` |
| Seed-pool controls (RQ2) | `run_seedpool.py` | `analyze_seedpool.py`, `analyze_seedpool_checks.py`, `analyze_seedpool_seeds.py` | `seedpool/`, `results_seedpool_summary.csv` |
| Mixed selection policy (RQ2) | `run_volume_and_seeds.py` | | `results_mixed_policy.csv` |
| Baselines and episode threshold (RQ3) | `run_baselines.py` | `analyze_baselines.py`, `analyze_threshold_sensitivity.py` | `baselines_eval/`, `results_baselines_*.csv`, `results_threshold_*.csv` |
| Family sweep and volume intervention (RQ4) | `run_family_sweep.py`, `run_volume_and_seeds.py` | `plot_family_sweep.py` | `results_family_sweep*.csv`, `results_volume_intervention.csv` |
| Cross-architecture (RQ5) | `attack_smart_mlp.py`, `attack_native_mlp.py` | | `attack_mlp_*.csv`, `attack_native_mlp_*.csv` |
| Insertion-time gates | `run_defense_eval.py` | `analyze_defense_eval.py` | `defense_eval/`, `results_defense_eval_*.csv` |
| Monthly natural-vs-stamped check | `run_backdoor_probe.py`, `run_probe_defender.py` | `analyze_probe_defender.py` | `results_backdoor_probe.csv`, `results_probe_defender*.csv` |
| Localization, 10 seeds | `run_localize.py` | `analyze_localize.py`, `analyze_localize_all.py` | `results_localize_*.csv` |
| Realizability | `problem_space_validation.py`, `malware_static_validation.py`, `run_ablation_malware_trigger.py` | | `malware_static_validation_results.csv`, `results_ablation_malware_trigger.csv` |
| Android attempt (not used in the paper) | `run_lamda.py` | | `lamda_*` |

## Shared modules

| Module | Purpose |
|---|---|
| `attack_smart.py` | Data loading, SHAP-guided trigger, realizable-field allowlist and the retraining loop |
| `attack_smart_mech.py` | `attack_smart.py` with selection logging and the control policies |
| `cade_defense.py` | CADE-style contrastive autoencoder gate |
| `attack_vs_cade.py`, `attack_vs_iforest.py` | Single-configuration gate runs, superseded by `run_defense_eval.py` |
| `mlp_classifier.py` | MLP classifier and native-MLP trigger |
| `ember_features_reference.py` | EMBER feature extractor, copied from github.com/elastic/ember under that project's license |

## Exploratory scripts

These were used to check the data and the setup before the main experiments and
do not produce numbers reported in the paper: `bodmas_smoke_test.py`,
`adaptation_loops.py`, `investigate_jan_spike.py` and `diag_margin.py`.

## Notes

- The per-sample malware table holds SHA-256 hashes and numbers only.
- Random seeds are fixed in every runner.
