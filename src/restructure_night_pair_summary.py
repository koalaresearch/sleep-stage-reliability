#!/usr/bin/env python3
"""Relabel a schema-1.0 paired-recording summary into the schema-2.0 layout.

The paired analysis (``analyze_sleep_edf_night_pairs_v1_2.py``) writes
``night_pair_summary.json`` in schema 1.0. In that layout the block
``all_nominal_pairs_sign_invariant`` mixes statistics that do not depend on the
orientation of a pair (record 2 - record 1 vs record 1 - record 2) with signed
statistics that do. Schema 2.0 separates them into:

* ``orientation_invariant_all_pairs``: unchanged if any pair is reversed;
* ``nominal_orientation_diagnostics_all_pairs``: depend on the nominal
  file-name night order, which is not verified chronology for 8 pairs;
* ``chronological_signed_date_concordant_pairs``: signed summaries restricted
  to the 67 pairs whose EDF header dates are consecutive.

This script only moves and renames fields. It performs no arithmetic, and it
stops if any input value would be lost or if the input layout is unexpected.
It never overwrites an existing output file.

Usage (from the repository root):
  python src/restructure_night_pair_summary.py \
    --in work/night_pairs_v1_2/night_pair_summary.json \
    --out work/night_pair_summary.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

SCHEMA_IN, SCHEMA_OUT = '1.0', '2.0'
OUTCOMES = ('mean', 'median', 'balanced')
STRATA = ('same_scorer', 'different_scorer', 'concordant_dates', 'discordant_dates')

# Fields of the schema-1.0 per-group statistics block, by orientation behaviour.
INVARIANT_STATS = ('mean_abs_delta', 'median_abs_delta', 'abs_delta_q25_q75',
                   'sw_rms_protocol_v1_2', 'repeatability_coefficient_rms_protocol_v1_2')
NOMINAL_STATS = ('mean_signed_nominal_delta', 'median_signed_nominal_delta', 'sd_signed_delta',
                 'sw_centered_bias_adjusted', 'repeatability_coefficient_centered_descriptive')
GROUP_KEYS = {'n', 'available', *INVARIANT_STATS, *NOMINAL_STATS}
OUTCOME_KEYS = {'outcome', 'all_nominal_pairs_sign_invariant', 'signed_date_concordant_only',
                'mean_abs_delta_bootstrap_ci_conditional', 'median_abs_delta_bootstrap_ci_conditional',
                'sd_between_subject_pair_means', 'spearman_between_two_records', 'strata'}
TOP_KEYS = {'outcomes', 'note', 'bootstrap_unit', 'bootstrap_replicates', 'bootstrap_seed',
            'signed_effect_scope', 'protocol_formula_warning', 'status', 'protocol',
            'models_retrained', 'oof_epochs', 'paired_subjects', 'same_scorer_pairs',
            'different_scorer_pairs', 'date_concordant_pairs', 'date_discordant_pairs',
            'input_sha256', 'output_sha256'}


def require(test: bool, message: str) -> None:
    if not test:
        raise ValueError(message)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def field_mapping(v1: dict) -> list[tuple[tuple, tuple]]:
    """Return (schema-1.0 path, schema-2.0 path) pairs for every leaf value."""
    m: list[tuple[tuple, tuple]] = []
    for k in OUTCOMES:
        o1, o2 = ('outcomes', k), ('outcomes', k)
        m.append((o1 + ('outcome',), o2 + ('outcome',)))
        g1 = o1 + ('all_nominal_pairs_sign_invariant',)
        inv = o2 + ('orientation_invariant_all_pairs',)
        nom = o2 + ('nominal_orientation_diagnostics_all_pairs',)
        chron = o2 + ('chronological_signed_date_concordant_pairs',)
        m += [(g1 + ('n',), inv + ('n',)), (g1 + ('available',), inv + ('available',))]
        m += [(g1 + (s,), inv + (s,)) for s in INVARIANT_STATS]
        m += [(o1 + (s,), inv + (s,)) for s in ('mean_abs_delta_bootstrap_ci_conditional',
                                                 'median_abs_delta_bootstrap_ci_conditional',
                                                 'sd_between_subject_pair_means')]
        m.append((g1 + ('n',), nom + ('n',)))
        m += [(g1 + (s,), nom + (s,)) for s in NOMINAL_STATS]
        m.append((o1 + ('spearman_between_two_records',), nom + ('spearman_between_two_records',)))
        m += [(o1 + ('signed_date_concordant_only', s), chron + (s,))
              for s in ('n', 'mean', 'median', 'mean_bootstrap_ci_conditional')]
        for st in STRATA:
            s1, s2 = o1 + ('strata', st), o2 + ('strata', st)
            m += [(s1 + ('n',), s2 + ('orientation_invariant', 'n')),
                  (s1 + ('available',), s2 + ('orientation_invariant', 'available'))]
            m += [(s1 + (s,), s2 + ('orientation_invariant', s)) for s in INVARIANT_STATS]
            m.append((s1 + ('n',), s2 + ('nominal_orientation_diagnostics', 'n')))
            m += [(s1 + (s,), s2 + ('nominal_orientation_diagnostics', s)) for s in NOMINAL_STATS]
    for key in sorted(TOP_KEYS - {'outcomes'}):
        m.append(((key,), (key,)))
    return m


def leaves(obj, prefix=()):
    """Yield (path, value) for every leaf; lists of numbers count as one leaf."""
    if isinstance(obj, dict) and obj and prefix[:1] not in (('input_sha256',), ('output_sha256',)):
        for k, v in obj.items():
            yield from leaves(v, prefix + (k,))
    else:
        yield prefix, obj


def get(obj, path):
    for p in path:
        obj = obj[p]
    return obj


def put(obj, path, value):
    for p in path[:-1]:
        obj = obj.setdefault(p, {})
    obj[path[-1]] = value


def validate_v1(v1: dict) -> None:
    require('schema_version' not in v1, 'Input already carries a schema_version; expected schema 1.0 output')
    require(set(v1) == TOP_KEYS, f'Unexpected top-level keys: {sorted(set(v1) ^ TOP_KEYS)}')
    require(v1['status'] == 'PASS' and v1['models_retrained'] is False, 'Input is not a PASS, no-refit summary')
    require(set(v1['outcomes']) == set(OUTCOMES), 'Unexpected outcome set')
    for k in OUTCOMES:
        o = v1['outcomes'][k]
        require(set(o) == OUTCOME_KEYS, f'Unexpected keys in outcome {k}')
        require(set(o['all_nominal_pairs_sign_invariant']) == GROUP_KEYS, f'Unexpected group keys in {k}')
        require(set(o['strata']) == set(STRATA), f'Unexpected strata in {k}')
        for st in STRATA:
            require(set(o['strata'][st]) == GROUP_KEYS, f'Unexpected keys in {k}/{st}')


def restructure(v1: dict, source_sha256: str, script_sha256: str) -> tuple[dict, list]:
    validate_v1(v1)
    mapping = field_mapping(v1)
    mapped_sources = {src for src, _ in mapping}
    all_sources = {p for p, _ in leaves(v1)}
    require(all_sources == mapped_sources,
            f'Unmapped or missing input fields: {sorted(all_sources ^ mapped_sources)}')
    v2: dict = {'schema_version': SCHEMA_OUT}
    v2['schema_provenance'] = {
        'derived_from_schema_version': SCHEMA_IN,
        'derived_from_sha256': source_sha256,
        'transform_script': 'src/restructure_night_pair_summary.py',
        'transform_script_sha256': script_sha256,
        'transform': 'fields moved and renamed only; no value recomputed',
        'historical_execution_provenance': ('input_sha256 and output_sha256 are copied unchanged from the '
                                            'schema-1.0 output and describe that historical execution of '
                                            'analyze_sleep_edf_night_pairs_v1_2.py'),
    }
    v2['orientation_definitions'] = {
        'orientation_invariant': ('unchanged if the two recordings of any pair are swapped: absolute '
                                  'differences, RMS-based within-pair scale and repeatability coefficient, '
                                  'SD of pair means, and their bootstrap intervals'),
        'nominal_orientation_diagnostic': ('depends on the nominal night order from the file name '
                                           '(delta = record 2 - record 1), which is not verified chronology '
                                           'for the 8 date-discordant pairs: signed mean, signed median, SD of '
                                           'signed differences, centred within-pair scale and repeatability '
                                           'coefficient, and the Spearman correlation between record columns'),
        'chronological_signed': ('signed summaries restricted to the 67 pairs whose EDF header dates are '
                                 'consecutive, where the nominal order agrees with the recorded dates'),
        'strata_note': ('in the concordant_dates stratum the nominal order agrees with the recorded dates; in '
                        'all other strata the signed values are nominal-orientation diagnostics only'),
    }
    v2['reporting_deviation_note'] = ('The configuration reports signed deltas on date-concordant pairs only. '
                                      'Signed values over all 75 pairs and within strata are retained as '
                                      'labelled nominal-orientation diagnostics; they are not chronological '
                                      'effects.')
    for src, dst in mapping:
        put(v2, dst, get(v1, src))
    # Order the outcome blocks for readability.
    for k in OUTCOMES:
        o = v2['outcomes'][k]
        v2['outcomes'][k] = {key: o[key] for key in (
            'outcome', 'orientation_invariant_all_pairs', 'nominal_orientation_diagnostics_all_pairs',
            'chronological_signed_date_concordant_pairs', 'strata')}
    # Final check: every mapped value is present and identical.
    for src, dst in mapping:
        require(get(v1, src) == get(v2, dst) and type(get(v1, src)) is type(get(v2, dst)),
                f'Value changed: {src} -> {dst}')
    return v2, mapping


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--in', dest='src', required=True, type=Path, help='schema-1.0 night_pair_summary.json')
    ap.add_argument('--out', required=True, type=Path, help='new schema-2.0 JSON file (must not exist)')
    args = ap.parse_args()
    require(args.src.is_file(), f'Missing input: {args.src}')
    require(args.out.parent.is_dir(), f'Output folder does not exist: {args.out.parent}')
    require(not args.out.exists(), f'Output exists: {args.out}. No overwrite.')
    v1 = json.loads(args.src.read_text(encoding='utf-8'))
    v2, mapping = restructure(v1, sha256(args.src), sha256(Path(__file__)))
    fd, tmp = tempfile.mkstemp(dir=args.out.parent, prefix='.schema2-', suffix='.part')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as fh:
            fh.write(json.dumps(v2, indent=2, allow_nan=False) + '\n')
        os.rename(tmp, args.out)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    print(f'Schema {SCHEMA_IN} -> {SCHEMA_OUT}: {len(mapping)} fields relabelled, values unchanged.')
    print('Output:', args.out)


if __name__ == '__main__':
    main()
