# Verification status

Results are separated into three categories:

- **Independently checked:** re-derived from the saved output files with
  separate code written for that purpose. These checks read existing
  artifacts only; they do not re-run feature extraction or refit models.
- **Developer-reported:** pass/fail checks built into the pipeline scripts and
  recorded in their own output files when the analysis was run.
- **Awaiting independent verification:** not yet confirmed outside the original
  run.

## Independently checked

Re-derived during repository preparation (2026-11-11) from the saved
out-of-fold prediction file, the per-subject metric table and the pair audit:

| Quantity | Reported | Re-derived |
|---|---|---|
| Out-of-fold epochs; row index exactly 0…N−1 | 161,028 | 161,028; yes |
| Subjects / recordings | 78 / 153 | 78 / 153 |
| Subjects with two recordings | 75 | 75 |
| Mean subject macro-F1, LR (true-support stages) | 0.544824 | 0.544824 (from raw predictions) |
| Mean subject macro-F1, Dummy | 0.119750 | 0.119750 (from raw predictions) |
| Median \|Δ\| of LR mean log-loss, 75 pairs | 0.112113 | 0.112113 (from raw probabilities) |
| Pairs with different scorers | 49 | 49 |
| Date-discordant pairs | 07, 17, 27, 32, 37, 60, 62, 72 | same 8 subjects |

SHA-256 values of all saved inputs and outputs that are referenced inside the
result files were recomputed against the original files and match
(configuration, feature matrix, QA report, LOSO runner, out-of-fold
predictions, subject metrics, LOSO summary, pair audit, window audit,
paired-analysis script and its three outputs). The four result JSON files in
`results/` are byte-identical copies of the originals.

### Code checks in the published layout (2026-11-11)

Lightweight checks only. No EEG was processed and no model was fitted.

- All files in `src/` and `tests/` compile (`python -m py_compile`).
- Each of the nine scripts in `src/` starts from the repository root and
  prints its `--help`, so the imports between sibling modules resolve in the
  `src/` layout.
- Every command-line flag used in the README reproduction commands exists in
  the corresponding script's argument parser.
- Synthetic test suite: `python -m pytest tests` → **3 passed**
  (`test_resume_and_complete_153_records`,
  `test_corrupt_saved_features_blocks_resume`,
  `test_changed_yaml_blocks_preflight`). Run with Python 3.13.12 and the
  package versions listed in `environment.yml`, plus pytest 9.2.0.
  These tests cover the orchestration of feature extraction (resume,
  integrity guards, configuration-hash guard) with synthetic fixtures. They do
  not test spectral feature values, the LOSO runner or the paired analysis.

## Developer-reported (recorded in pipeline outputs)

- Dataset inventory: 78 subjects, 153 recordings, 75 complete pairs, all 306
  EDF SHA-256 values matched `SHA256SUMS.txt` (`results/audit_summary.json`).
- Annotation alignment for subject 00: no uncovered or multiply covered
  epochs, no off-grid boundaries (`PASS`).
- Window audit for all 153 records (`results/window_audit_summary.json`, `PASS`).
- Feature extraction: 153 records, 161,028 rows, per-record provenance and
  pilot reproduction (`PASS`).
- Pre-model QA: schema, cohort counts, stage counts, LOSO schedule with
  exactly-once coverage (`PASS`).
- LOSO: 78 fits per model, subject-disjoint folds, all LR fits converged at
  `max_iter = 2000`, out-of-fold coverage exactly one (`PASS`,
  `results/loso_summary.json`).
- Paired analysis: 75 pairs, 26/49 scorer split, 67/8 date split
  (`PASS`, `results/night_pair_summary.json`).
- Synthetic-data tests for the feature-extraction orchestration
  (`tests/test_extract_sleep_edf_all_v1_2.py`): no record of an earlier run is
  kept, but they were re-run in the published layout (see above). The
  developer's notes also report synthetic-data checks
  of the LOSO runner and a synthetic end-to-end test of the paired-analysis
  script (78 subjects, 153 records, 75 pairs, 161,028 epochs; class order,
  clipping, deterministic bootstrap, overwrite guard); those test scripts are
  not included in `tests/`.

The docstring of `analyze_sleep_edf_night_pairs_v1_2.py` refers to the LOSO
predictions as "independently validated"; this wording predates the checks
above and should be read as "verified by the pipeline's own QA gates".

## Awaiting independent verification

- **The computational environment has not been independently reproduced.**
  `environment.yml` lists the versions recorded at run time, but it has not
  been solved or installed from scratch. The original runs and the checks above
  used a local virtual environment on a Miniconda Python 3.13.12 (macOS ARM64).
- A complete re-execution of the pipeline from the raw dataset in a fresh
  environment (feature extraction and LOSO have been run once).
- Per-subject and per-record tables have not been reviewed for publication and
  are not included.

## Provenance hashes

The result files record SHA-256 hashes of their inputs. For publication, the
configuration file and three scripts received comment/docstring-only edits that
remove machine-specific paths and project-internal names. These edits do not
change behaviour, but they do change the file hashes:

| File | Hash recorded in results | Hash of published file |
|---|---|---|
| `config/study_v1_2.yaml` | `d97625be…dd8c93` | `791c2788…f4029c` |
| `src/extract_sleep_edf_all_v1_2.py` | `80abff80…7c93c8` | `114a85e3…31f9ec` |
| `src/extract_sleep_edf_pilot_v1_2_fixed.py` | `3b5fe241…43d43a` | `97438094…c22ced` |
| `src/validate_sleep_edf_features_v1_2.py` | not recorded | `3e018090…efdb57` |
| `src/run_sleep_edf_loso_v1_2.py` | `c55e7f71…ca859d` | **unchanged** |
| `src/analyze_sleep_edf_night_pairs_v1_2.py` | `235c0d77…296061` | **unchanged** |

The parsed YAML of the published configuration is identical to the original
(`yaml.safe_load` comparison). Because the stages check each other's hashes, a
reproduction must be run end to end with the published configuration; the
published configuration cannot be combined with intermediate files produced by
the original run.

## Changes made during repository preparation

"Moved only" means the published file is byte-identical to the original
(SHA-256 compared). Every other entry was modified as described, so its hash
differs from the original.

| Original location | Published location | Change |
|---|---|---|
| `build_sleep_edf_manifest.py` | `src/` | moved only |
| `sleep_edf_two_night_smoke.py` | `src/` | moved only |
| `audit_sleep_edf_epoch_alignment.py` | `src/` | moved only |
| `audit_sleep_edf_windows_v1_2.py` | `src/` | moved only |
| `extract_sleep_edf_pilot_v1_2_fixed.py` | `src/` | one comment reworded |
| `extract_sleep_edf_all_v1_2.py` | `src/` | usage docstring paths; one error-message wording |
| `validate_sleep_edf_features_v1_2.py` | `src/` | usage docstring |
| `run_sleep_edf_loso_v1_2.py` | `src/` | moved only (byte-identical) |
| `analyze_sleep_edf_night_pairs_v1_2.py` | `src/` | moved only (byte-identical) |
| `test_extract_sleep_edf_all_v1_2.py` | `tests/` | configuration path; `tests/conftest.py` adds `src/` to the import path |
| study configuration YAML | `config/study_v1_2.yaml` | renamed; header comments only |
| `results/loso_v1_2/published/loso_summary.json` | `results/loso_summary.json` | moved only |
| `results/night_pairs_v1_2/night_pair_summary.json` | `results/night_pair_summary.json` | moved only |
| `results/night_pairs_v1_2/figures/*.png` | `results/figures/` | moved only |
| `results/audit_summary.json`, `results/window_audit_summary.json` | `results/` | unchanged |

Not published: `extract_sleep_edf_pilot_v1_2.py` (superseded by the `_fixed`
version, which the pipeline imports), internal protocol drafts, and all
per-subject, per-record and per-epoch outputs.
