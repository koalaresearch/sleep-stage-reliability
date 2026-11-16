# Paired-recording results: schema 2.0

`results/night_pair_summary.json` uses **schema 2.0**. Version 2.0 changes the
labels and grouping of the paired-recording statistics. Every numerical value
is identical to the schema-1.0 output of the audited analysis run.

## Why the schema changed

In schema 1.0 the block `all_nominal_pairs_sign_invariant` contained both
statistics that do not depend on the orientation of a pair and signed
statistics that do. The pair orientation is the nominal file-name night order
(`Δ = record 2 − record 1`), and it is not verified chronology for the 8 pairs
with discordant EDF header dates. The label "sign invariant" was therefore
wrong for part of the block. Schema 2.0 separates three groups:

| Schema-2.0 block | Contents | Pairs |
|---|---|---|
| `orientation_invariant_all_pairs` | Unchanged if the two recordings of any pair are swapped | 75 |
| `nominal_orientation_diagnostics_all_pairs` | Depend on the nominal night order; diagnostics only, not chronological effects | 75 |
| `chronological_signed_date_concordant_pairs` | Signed summaries where nominal order agrees with consecutive header dates | 67 |

Each stratum (`same_scorer`, `different_scorer`, `concordant_dates`,
`discordant_dates`) is split the same way into `orientation_invariant` and
`nominal_orientation_diagnostics`.

## Field mapping (applies to each outcome `<k>` = `mean`, `median`, `balanced`)

| Schema 1.0 path (`outcomes.<k>.…`) | Schema 2.0 path (`outcomes.<k>.…`) |
|---|---|
| `outcome` | `outcome` |
| `all_nominal_pairs_sign_invariant.n` | `orientation_invariant_all_pairs.n` and `nominal_orientation_diagnostics_all_pairs.n` |
| `all_nominal_pairs_sign_invariant.available` | `orientation_invariant_all_pairs.available` |
| `all_nominal_pairs_sign_invariant.mean_abs_delta` | `orientation_invariant_all_pairs.mean_abs_delta` |
| `all_nominal_pairs_sign_invariant.median_abs_delta` | `orientation_invariant_all_pairs.median_abs_delta` |
| `all_nominal_pairs_sign_invariant.abs_delta_q25_q75` | `orientation_invariant_all_pairs.abs_delta_q25_q75` |
| `all_nominal_pairs_sign_invariant.sw_rms_protocol_v1_2` | `orientation_invariant_all_pairs.sw_rms_protocol_v1_2` |
| `all_nominal_pairs_sign_invariant.repeatability_coefficient_rms_protocol_v1_2` | `orientation_invariant_all_pairs.repeatability_coefficient_rms_protocol_v1_2` |
| `mean_abs_delta_bootstrap_ci_conditional` | `orientation_invariant_all_pairs.mean_abs_delta_bootstrap_ci_conditional` |
| `median_abs_delta_bootstrap_ci_conditional` | `orientation_invariant_all_pairs.median_abs_delta_bootstrap_ci_conditional` |
| `sd_between_subject_pair_means` | `orientation_invariant_all_pairs.sd_between_subject_pair_means` |
| `all_nominal_pairs_sign_invariant.mean_signed_nominal_delta` | `nominal_orientation_diagnostics_all_pairs.mean_signed_nominal_delta` |
| `all_nominal_pairs_sign_invariant.median_signed_nominal_delta` | `nominal_orientation_diagnostics_all_pairs.median_signed_nominal_delta` |
| `all_nominal_pairs_sign_invariant.sd_signed_delta` | `nominal_orientation_diagnostics_all_pairs.sd_signed_delta` |
| `all_nominal_pairs_sign_invariant.sw_centered_bias_adjusted` | `nominal_orientation_diagnostics_all_pairs.sw_centered_bias_adjusted` |
| `all_nominal_pairs_sign_invariant.repeatability_coefficient_centered_descriptive` | `nominal_orientation_diagnostics_all_pairs.repeatability_coefficient_centered_descriptive` |
| `spearman_between_two_records` | `nominal_orientation_diagnostics_all_pairs.spearman_between_two_records` |
| `signed_date_concordant_only.{n, mean, median, mean_bootstrap_ci_conditional}` | `chronological_signed_date_concordant_pairs.{n, mean, median, mean_bootstrap_ci_conditional}` |
| `strata.<s>.{n, available}` | `strata.<s>.orientation_invariant.{n, available}`; `n` also in `strata.<s>.nominal_orientation_diagnostics.n` |
| `strata.<s>.{mean_abs_delta, median_abs_delta, abs_delta_q25_q75, sw_rms_protocol_v1_2, repeatability_coefficient_rms_protocol_v1_2}` | `strata.<s>.orientation_invariant.{same names}` |
| `strata.<s>.{mean_signed_nominal_delta, median_signed_nominal_delta, sd_signed_delta, sw_centered_bias_adjusted, repeatability_coefficient_centered_descriptive}` | `strata.<s>.nominal_orientation_diagnostics.{same names}` |

All top-level fields of schema 1.0 are kept unchanged: `note`,
`bootstrap_unit`, `bootstrap_replicates`, `bootstrap_seed`,
`signed_effect_scope`, `protocol_formula_warning`, `status`, `protocol`,
`models_retrained`, `oof_epochs`, `paired_subjects`, `same_scorer_pairs`,
`different_scorer_pairs`, `date_concordant_pairs`, `date_discordant_pairs`,
`input_sha256` and `output_sha256`. The last two describe the historical
execution of the analysis script.

New top-level fields in schema 2.0: `schema_version`, `schema_provenance`,
`orientation_definitions` and `reporting_deviation_note`.

Individual metric names are unchanged; only their containing blocks were
renamed. A metric's orientation behaviour is given by its block, not its name.
For example, `sw_centered_bias_adjusted` removes the mean bias but is
orientation-dependent.

## Provenance

| Item | SHA-256 |
|---|---|
| Schema-1.0 summary (audited historical output; repository commit `1c43723`) | `60a77cf8363eb8f6073cf2ccce67a5381b15f63034aa2c4d07b97445c06fe67e` |
| Schema-2.0 summary (`results/night_pair_summary.json`) | `660acc23722519fad1e57c2c68314f5c98dab77f197beb07860d8c959a36eaa9` |
| Relabelling script (`src/restructure_night_pair_summary.py`) | `cc3c2c4a95a8020a805b6ad096547b5f9ba09e22e9428aa438ee18d9615e05e7` |
| Analysis script that produced the schema-1.0 output (unchanged) | `235c0d775a7c793fae1e63f69f3dca2d153690bf598d86fdd100a05503296061` |

Running `src/restructure_night_pair_summary.py` on the schema-1.0 file
reproduces the schema-2.0 file byte for byte. The schema-2.0 file records the
schema-1.0 hash in `schema_provenance.derived_from_sha256`. The script checks
that every input value appears unchanged in the output, and it stops on any
unexpected input layout. The schema-1.0 file is preserved unchanged in the
repository history and in the private audit evidence.
