# Verification status

Results are separated into three categories:

- **Independently checked:** recomputed from saved artifacts with separate
  code written for that purpose. These checks read existing artifacts only;
  they do not re-run feature extraction or refit models.
- **Developer-reported:** pass/fail checks built into the pipeline scripts and
  recorded in their own output files when the analysis was run.
- **Not independently verified:** not confirmed outside the original run.

## Independent numerical audit (AUDIT-01, 2026-11-16)

**Scope.** Saved out-of-fold (OOF) predictions → all reported paired-recording
and LOSO statistics. The audit started from the saved artifacts of the
historical run: the OOF prediction file (161,028 epochs), the per-subject,
per-record, per-pair and per-pair-stage tables, the pair audit, the window
audit, and both summary JSON files. It used separate NumPy/pandas code. The
original analysis scripts were read to establish conventions but were neither
imported nor executed. No EEG was processed, no features were extracted and no
model was fitted.

**Result.** 296 comparison groups comprising **9,190 numerical values**, all
within tolerance. The tolerance was an absolute error ≤ 1e-10 for floating-point
values, with exact equality for counts, booleans, identifiers and missing-value
patterns. The largest discrepancy was **5.11 × 10⁻¹⁵** (LOSO summary); for the
paired-recording tables and summary it was ≤ 4.4 × 10⁻¹⁶. No numerical
correction was required.

Also confirmed:
- All recorded SHA-256 hashes corresponding to artifacts supplied for AUDIT-01
  were independently verified. The original feature matrix was not supplied,
  so its recorded SHA-256 could not be independently checked;
- OOF coverage is exactly once for row indices 0–161,027. Each recording's
  epoch count equals its main-window epochs minus label-excluded epochs
  (153/153), and the stage counts match the feature audit;
- LR and Dummy probabilities are finite, lie in [0, 1], sum to 1 and agree
  with the predicted class;
- the epoch-loss convention (clip to [1e-15, 1], renormalise, −log p(true));
  record, pair and stage-balanced weighting; and the shared-stage eligibility
  rule (≥ 10 epochs in both recordings);
- pairing (75 pairs; subjects 13, 36 and 52 unpaired), the 26/49 scorer split
  and the 67/8 date-concordance split, computed from the saved metadata;
- all nine paired bootstrap intervals and both LOSO bootstrap intervals,
  reproduced with the same resampling unit, seeds and percentile method;
- the Bland–Altman bias and limits of agreement, reconstructed from the 67
  date-concordant pairs;
- for every held-out subject, the saved Dummy probabilities equal the stage
  proportions of the other 77 subjects (maximum error 1.9 × 10⁻¹⁶). This
  corroborates the subject-level fold construction for the Dummy model.

**What the audit does not cover.** It verifies results computed *from the
saved OOF predictions*. It does **not** verify the historical execution from
raw EDF recordings through annotation alignment, feature extraction and model
fitting. The feature matrix, raw recordings and per-fold checkpoints were not
part of the audit, so the feature-matrix hash recorded in the LOSO summary
could not be checked. Correct LR fitting within folds and absence of upstream
leakage are supported by the pipeline's own checks and the saved predictions,
but are **not independently proven**. The audit does not reproduce training
variability. Reproducing the environment from `environment.yml` was outside
its scope.

**Findings and their resolution.**

| Finding | Resolution |
|---|---|
| F1: the schema-1.0 block `all_nominal_pairs_sign_invariant` mixed orientation-invariant statistics with orientation-dependent signed diagnostics | Corrected: the paired summary was relabelled to schema 2.0 with every value unchanged; see [`RESULTS_SCHEMA.md`](RESULTS_SCHEMA.md). The configuration comments, `METHODS.md` and `README.md` were updated to match. |
| F2: a pre-execution validation statement in the internal documentation was out of date | Corrected by a separate erratum; sealed original documentation remains unchanged. No numerical results were affected. |
| F3: raw-data-to-model execution cannot be assessed from saved predictions | Stated as a limitation here and in the README. |

The audit checked the schema-1.0 summary
(`60a77cf8…06fe67e`). The published schema-2.0 file carries the same values.
Running `src/restructure_night_pair_summary.py` on the audited file reproduces
it byte for byte, and a separate check confirmed that no value was lost or
altered.

## Checks during repository preparation (2026-11-11)

Recomputed from the saved OOF prediction file, the per-subject metric table and
the pair audit (superseded in scope by AUDIT-01, kept for the record):

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

The input and output SHA-256 values recorded inside the result files were
recomputed and matched for the available original artifacts only. This
excludes the feature matrix, which is not available for independent
verification: it was not supplied for AUDIT-01 and is not published, so its
recorded SHA-256 has not been independently checked. Of the four result
JSON files in `results/`, three are byte-identical copies of the originals
(`loso_summary.json`, `audit_summary.json`, `window_audit_summary.json`). The
fourth, `night_pair_summary.json`, was relabelled to schema 2.0 (see above).

### Code checks in the published layout

Lightweight checks only. No EEG was processed and no model was fitted.

- All files in `src/` and `tests/` compile (`python -m py_compile`).
- Each of the ten scripts in `src/` starts from the repository root and prints
  its `--help`, so the imports between sibling modules resolve in the `src/`
  layout.
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
- The developer's notes report synthetic-data checks of the LOSO runner and a
  synthetic end-to-end test of the paired-analysis script (78 subjects,
  153 records, 75 pairs, 161,028 epochs; class order, clipping, deterministic
  bootstrap, overwrite guard). Those test scripts are not included in
  `tests/`.

The docstring of `analyze_sleep_edf_night_pairs_v1_2.py` refers to the LOSO
predictions as "independently validated". This wording predates both the checks
above and AUDIT-01; read it as "verified by the pipeline's own QA gates". The
docstring is left unchanged so that the script keeps its recorded hash.

## Not independently verified

- **Raw-data-to-model execution.** No complete re-execution from the raw
  dataset through feature extraction and LOSO fitting has been carried out
  outside the original run. Feature extraction and LOSO have been run once.
- **The computational environment has not been independently reproduced.**
  `environment.yml` lists the versions recorded at run time, but it has not
  been solved or installed from scratch. The original runs and the checks above
  used a local virtual environment on a Miniconda Python 3.13.12 (macOS ARM64).
- Per-subject and per-record tables have not been reviewed for publication and
  are not included.

## Provenance hashes

The result files record SHA-256 hashes of their inputs. For publication, the
configuration file and three scripts received comment/docstring-only edits that
remove machine-specific paths and project-internal names. After AUDIT-01, the
configuration received further comment-only edits (F1). These edits do not
change behaviour, but they do change the file hashes:

| File | Hash recorded in results (historical run) | Hash of published file |
|---|---|---|
| `config/study_v1_2.yaml` | `d97625be…dd8c93` | `811d3297…afc361` |
| `src/extract_sleep_edf_all_v1_2.py` | `80abff80…7c93c8` | `114a85e3…31f9ec` |
| `src/extract_sleep_edf_pilot_v1_2_fixed.py` | `3b5fe241…43d43a` | `97438094…c22ced` |
| `src/validate_sleep_edf_features_v1_2.py` | not recorded | `3e018090…efdb57` |
| `src/run_sleep_edf_loso_v1_2.py` | `c55e7f71…ca859d` | **unchanged** |
| `src/analyze_sleep_edf_night_pairs_v1_2.py` | `235c0d77…296061` | **unchanged** (audited version) |
| `src/restructure_night_pair_summary.py` | new (schema relabelling) | `cc3c2c4a…15e05e7` |
| `results/night_pair_summary.json` | schema 1.0: `60a77cf8…06fe67e` | schema 2.0: `660acc23…a36eaa9` |

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
| `analyze_sleep_edf_night_pairs_v1_2.py` | `src/` | moved only (byte-identical; the version audited in AUDIT-01) |
| — | `src/restructure_night_pair_summary.py` | new after AUDIT-01: relabels the schema-1.0 summary to schema 2.0 without arithmetic |
| `test_extract_sleep_edf_all_v1_2.py` | `tests/` | configuration path; `tests/conftest.py` adds `src/` to the import path |
| study configuration YAML | `config/study_v1_2.yaml` | renamed; header comments; after AUDIT-01, comments on the paired summaries' orientation |
| `results/loso_v1_2/published/loso_summary.json` | `results/loso_summary.json` | moved only |
| `results/night_pairs_v1_2/night_pair_summary.json` | `results/night_pair_summary.json` | relabelled to schema 2.0 after AUDIT-01; values unchanged |
| `results/night_pairs_v1_2/figures/*.png` | `results/figures/` | moved only |
| `results/audit_summary.json`, `results/window_audit_summary.json` | `results/` | unchanged |

Not published: `extract_sleep_edf_pilot_v1_2.py` (superseded by the `_fixed`
version, which the pipeline imports), internal protocol drafts, the private
audit evidence package, and all per-subject, per-record and per-epoch outputs.
