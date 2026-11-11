# Methods

This document describes the analysis as implemented in `src/` with the
parameters in `config/study_v1_2.yaml` (protocol version 1.2). Where the text
and the configuration differ, the configuration is authoritative.

## 1. Data source and cohort

- **Dataset:** PhysioNet Sleep-EDF Database Expanded v1.0.0,
  <https://doi.org/10.13026/C2X676>. Sleep-Cassette (SC) study only.
- **Population:** healthy volunteers recorded at home during normal activity
  (EDF headers indicate 1989–1991), aged 25–101 years.
- **Inventory:** 153 PSG recordings with matching hypnograms from 78 subjects.
  75 subjects have two recordings; subject 13 has only night 1, subjects 36 and
  52 only night 2.
- **Access:** the dataset ZIP is read in place. Each PSG/hypnogram pair is
  streamed to a temporary directory, processed and deleted. The full archive is
  never extracted.
- **Integrity:** every EDF member is checked by streaming SHA-256 against the
  `SHA256SUMS.txt` file shipped in the archive (all 306 entries matched).
- **Units of analysis:** independent unit = subject; repeated unit = recording;
  prediction unit = 30-s epoch.

## 2. Manifest and recording metadata (`build_sleep_edf_manifest.py`)

- File names are parsed as `SC4<subject:2><night:1><tape><scorer>-PSG|Hypnogram.edf`.
- **Night index** = the night digit in the file name (PhysioNet convention).
  EDF header start dates and times are recorded but do not determine order.
- **Date concordance:** a pair is date-concordant if
  `date(night 2) − date(night 1) = 1 day`. 67 pairs are concordant, 8 are
  discordant (subjects 07, 17, 27, 32, 37, 60, 62, 72), including reversed,
  identical and multi-day differences.
- **Scorer code** = the eighth character of the hypnogram file name, which
  PhysioNet documents as the scorer identifier. 11 distinct codes occur;
  26 pairs share a scorer and 49 do not. Assignment is not random (code `C`
  scored 61 of 153 recordings).
- Outputs: `manifest_inventory.csv`, `pair_audit.csv`, `audit_summary.json`.
  The aggregate audit is published in `results/audit_summary.json`.

## 3. Annotation alignment and analysis window

### 3.1 Epochs and labels
- Channel `EEG Fpz-Cz`, verified sampling rate 100 Hz, read with MNE
  `read_raw_edf(preload=False)`; hypnograms read with `mne.read_annotations`.
- Non-overlapping 30-s epochs aligned to the PSG start; each epoch takes the
  label of the annotation covering it. Alignment checks: no uncovered or
  multiply covered epochs, annotation boundaries on the 30-s grid
  (`audit_sleep_edf_epoch_alignment.py`, subject 00). During feature
  extraction every record is checked for unknown labels, invalid or
  overlapping annotation ranges, and epochs not covered exactly once; any
  failure stops the run.
- Label map: `Sleep stage W`→W, `Sleep stage 1`→N1, `Sleep stage 2`→N2,
  `Sleep stage 3` and `Sleep stage 4`→N3, `Sleep stage R`→REM.
  `Sleep stage ?` and `Movement time` are excluded. Any other label stops the run.

### 3.2 Main-sleep window (`audit_sleep_edf_windows_v1_2.py`)
Defined from the reference labels, so it is an **evaluation universe**, not a
real-time pipeline:
1. sleep epoch = N1, N2, N3 or REM;
2. sleep episodes = runs of sleep epochs, merged when separated by < 60 min;
3. main episode = the episode with the most sleep epochs;
4. window = main episode ± 30 min, clipped to the recording.

The sensitivity window "first to last sleep epoch ± 30 min" is counted for every
record but not used for modelling. 49 of 153 records contain more than one
sleep episode; 4,140 sleep epochs fall outside the main episode
(`results/window_audit_summary.json`).

## 4. Spectral features (`extract_sleep_edf_pilot_v1_2_fixed.py`, `extract_sleep_edf_all_v1_2.py`)

- Signal converted to µV.
- Welch PSD per epoch: Hann window, 400 samples (4 s), 200-sample overlap,
  `nfft = 400` (0.25 Hz resolution), constant detrend, density scaling, mean
  averaging.
- Band power = sum of PSD bins in the band × 0.25 Hz. Bands (Hz): delta
  [0.5, 4), theta [4, 8), alpha [8, 12), sigma [12, 16), beta [16, 30].
- Total power = sum of the five bands, so the five relative powers sum to 1.
- Features (10): `log10(band power + 1e-12)` for the five bands and the five
  relative powers.
- Epoch rejection is label-independent only: non-finite values or exactly zero
  variance.
- A one-subject pilot (two recordings of subject 00, 1,968 epochs) is extracted
  first; the full extraction checks numerically that it reproduces the pilot
  rows.
- Each record is written atomically with a provenance JSON (source EDF hashes,
  configuration/manifest/window hashes, extractor hash, library versions). The
  combined matrix has 161,028 rows (W 35,640; N1 20,238; N2 67,056;
  N3 12,632; REM 25,462).

## 5. Pre-model QA (`validate_sleep_edf_features_v1_2.py`)

Read-only checks before any model is fitted: input hashes, column schema,
finite values, relative powers summing to 1, cohort counts (78 / 153 / 75),
stage counts, and a LOSO fold schedule with each subject tested exactly once.
Six subjects have no N3 epochs in the analysis window. The runner refuses to
start unless this report has status `PASS`.

## 6. Models and validation (`run_sleep_edf_loso_v1_2.py`)

- **Floor:** `DummyClassifier(strategy="prior")`.
- **Primary model:** `Pipeline([StandardScaler(), LogisticRegression(penalty="l2",
  C=1.0, solver="lbfgs", tol=1e-4, max_iter=2000, class_weight=None)])`.
  No hyperparameter search. The scaler is fitted on training subjects only.
- **Convergence policy:** if any fit fails to converge at `max_iter=2000`, all 78
  fits are repeated once with `max_iter=10000`. In the reported run all fits
  converged at 2000 iterations.
- **Validation:** `LeaveOneGroupOut` with `groups = subject_id` (78 folds). Both
  recordings of the held-out subject are predicted by the same model. Subject
  disjointness and exactly-once out-of-fold coverage are asserted.
- Inputs never include age, sex, subject identity, night index, scorer or
  hypnogram-derived variables.

## 7. Performance metrics

For each subject *i*, with the out-of-subject confusion matrix over all of that
subject's analysed epochs and for each stage *c*:

`F1_ic = 2·TP_ic / (2·TP_ic + FP_ic + FN_ic)`

- **Primary:** `S_i` = stages with at least one reference epoch in subject *i*;
  `macroF1_i = mean_{c ∈ S_i} F1_ic`; reported value = unweighted mean over
  the 78 subjects. `S_i` depends only on the reference labels and is identical
  for both models.
- **Sensitivity definitions:** five fixed classes with `zero_division = 0`;
  union of true and predicted classes.
- **Descriptive pooled values:** macro-F1 of the pooled confusion matrix with
  each epoch weighted by `1/n_epochs_i`, and unweighted.
- **Uncertainty:** 2,000 bootstrap resamples of subjects (seed `20261111`),
  2.5th–97.5th percentiles. Conditional on the trained models; descriptive only.

## 8. Paired-recording analysis (`analyze_sleep_edf_night_pairs_v1_2.py`)

Uses only the saved LOSO out-of-fold probabilities; no model is refitted.
Restricted to the 75 subjects with two recordings.

- **Epoch loss:** the five LR probabilities (order W, N1, N2, N3, REM), clipped
  to [1e-15, 1], renormalised; loss = −log p(true stage).
- **Record outcomes:** mean epoch loss (central), median epoch loss, and a
  stage-balanced loss (mean of per-stage mean losses over the stages with
  ≥ 10 epochs in **both** recordings of the subject).
- **Paired difference:** `Δ_i = L_i,night2 − L_i,night1` (file-name night index).
- **Sign-invariant summaries over all 75 pairs:** mean and median of |Δ|, IQR of
  |Δ|, SD of Δ, within-pair scale `sqrt(mean(Δ²)/2)` as specified in the
  protocol (includes systematic bias) and centred `SD(Δ)/√2`, the corresponding
  repeatability coefficients (× 2.77), SD of the pair means between subjects,
  and Spearman correlation between the two records.
- **Signed summaries:** mean and median Δ only for the 67 date-concordant pairs.
- **Descriptive strata:** same vs different scorer (26 / 49), concordant vs
  discordant dates (67 / 8). Not causal contrasts.
- **Bootstrap:** 2,000 resamples of subject pairs, seed `20261111`
  (+1 and +2 for the |Δ| mean and median intervals).
- **Figures:** Bland–Altman plot of mean loss (pair mean vs Δ; markers by
  scorer agreement, discordant-date pairs circled; dashed line = mean Δ over
  concordant pairs, dotted lines = ± 1.96 SD), and a histogram of |Δ|.

## 9. Interpretation boundaries

Permitted: out-of-subject performance of this fixed rule in SC and its spread
across people; the size of between-recording loss differences relative to
between-person spread, with the scorer and ordering caveats; the documented
data-quality observations (header-date discordance, scorer changes).

Not supported by this design: claims of novelty, of state-of-the-art or
clinically valid performance, of intrinsic sleep stability, of causal night,
age or scorer effects, or attribution of within-subject variance to
physiology rather than annotation.
