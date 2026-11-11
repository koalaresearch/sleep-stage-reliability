#!/usr/bin/env python3
"""Sequential, resumable Sleep-EDF SC feature extraction, protocol v1.2.

Input: externally stored, SHA-256-verified Sleep-EDF Expanded ZIP, verified
manifest and window audit, frozen v1.2 YAML. Uses the previously approved
extract_sleep_edf_pilot_v1_2_fixed.py functions without changing the Welch,
label alignment, main-sleep-window or feature definitions.

Output (local): one atomically committed CSV + provenance JSON per record, and,
once all 153 pass, a combined CSV with a run audit. Never trains a model,
modifies the ZIP, or extracts the complete dataset.

Run from the repository root with the environment active:

python src/extract_sleep_edf_all_v1_2.py \
  --zip '/path/to/sleep-edfx-1.0.0.zip' \
  --manifest './results/manifest_inventory.csv' \
  --window-audit './results/window_audit_all_153.csv' \
  --window-summary './results/window_audit_summary.json' \
  --config './config/study_v1_2.yaml' \
  --out './results/features_v1_2' \
  --pilot-csv './results/pilot_features_subject00.csv'

Re-run this same command after interruption to skip already audited records.
Use --limit 2 for a two-record shakeout; no combined file is produced then.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import mne
import numpy as np
import scipy
import yaml

from extract_sleep_edf_pilot_v1_2_fixed import (
    CLASSES,
    FEATURE_NAMES,
    json_numpy_default,
    process_record,
)

COLUMNS = [
    'subject_id', 'record_id', 'night_index', 'epoch_index',
    'epoch_start_sec', 'stage', *FEATURE_NAMES,
]
EXPECTED_RECORDS = 153
EXPECTED_SUBJECTS = 78
EXPECTED_PAIRS = 75


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open('r', newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def as_int(value: object, label: str) -> int:
    try:
        return int(value)
    except (TypeError, ValueError) as e:
        raise ValueError(f'{label} must be integer, got {value!r}') from e


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path: Path, obj: dict) -> None:
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False,
                               default=json_numpy_default, allow_nan=False)
                    + '\n', encoding='utf-8')


def validate_study_config(cfg: dict) -> None:
    if cfg['study']['protocol'] != 'v1.2' or cfg['data']['expected_recordings'] != 153:
        raise ValueError('Expected the 153-record study configuration v1.2')
    p = cfg['preprocessing']
    w = p['analysis_window']
    expected = {
        'main_channel': 'EEG Fpz-Cz',
        'sample_rate_hz': 100,
        'epoch_length_seconds': 30,
        'stage_order': list(CLASSES),
        'unknown_label_policy': 'raise_error',
        'band_power_integration': 'rectangle_sum',
        'total_power': 'sum_of_five_bands',
        'frequency_intervals': 'left_closed_right_open_except_last',
    }
    for key, value in expected.items():
        if p.get(key) != value:
            raise ValueError(f'Frozen preprocessing parameter differs: {key}')
    win_expected = {
        'is_reference_defined': True,
        'episode_merge_max_wake_gap_minutes': 60,
        'episode_choice': 'most_sleep_epochs',
        'wake_context_minutes_each_side': 30,
        'clip_to_recording': True,
        'sensitivity_window': 'first_to_last_sleep_epoch_plus_30min',
    }
    for key, value in win_expected.items():
        if w.get(key) != value:
            raise ValueError(f'Frozen analysis-window policy differs: {key}')
    if p['epoch_rejection'] != {'non_finite': True, 'zero_variance': True}:
        raise ValueError('Unexpected epoch rejection policy')
    if list(p['stage_map'].items()) != [
        ('Sleep stage W', 'W'), ('Sleep stage 1', 'N1'),
        ('Sleep stage 2', 'N2'), ('Sleep stage 3', 'N3'),
        ('Sleep stage 4', 'N3'), ('Sleep stage R', 'REM')
    ]:
        raise ValueError('Unexpected stage mapping')
    if p['exclude_labels'] != ['Sleep stage ?', 'Movement time']:
        raise ValueError('Unexpected exclusions')
    if list(p['bands_hz']) != ['delta', 'theta', 'alpha', 'sigma', 'beta']:
        raise ValueError('Unexpected band definitions')
    if tuple(tuple(float(v) for v in p['bands_hz'][key]) for key in p['bands_hz']) != (
        (0.5, 4), (4, 8), (8, 12), (12, 16), (16, 30)
    ):
        raise ValueError('Unexpected frequency bands')


def validate_inputs(args: argparse.Namespace) -> tuple[list[dict], dict[str, dict], dict, dict]:
    input_paths = [args.zip, args.manifest, args.window_audit, args.window_summary, args.config]
    if not all(path.is_file() for path in input_paths):
        raise FileNotFoundError('ZIP, verified manifest, window CSV/summary or YAML not found')
    with args.config.open(encoding='utf-8') as f:
        cfg = yaml.safe_load(f)
    validate_study_config(cfg)

    manifest_hash = sha256(args.manifest)
    config_hash = sha256(args.config)
    windows_hash = sha256(args.window_audit)
    summary = read_json(args.window_summary)
    if summary.get('status') != 'PASS':
        raise ValueError('Window audit is not PASS')
    if summary['config_sha256'] != config_hash:
        raise ValueError('YAML changed after window audit; refuse mixed run')
    if summary['verified_manifest_sha256'] != manifest_hash:
        raise ValueError('Manifest changed after window audit; refuse mixed run')
    if summary['window_csv_sha256'] != windows_hash:
        raise ValueError('Window CSV changed after audit; refuse mixed run')
    if summary.get('records') != EXPECTED_RECORDS or summary.get('subjects') != EXPECTED_SUBJECTS:
        raise ValueError('Unexpected window-audit cohort size')

    manifest = read_csv(args.manifest)
    windows = read_csv(args.window_audit)
    if len(manifest) != EXPECTED_RECORDS or len(windows) != EXPECTED_RECORDS:
        raise ValueError('Expected 153 manifest and 153 window-audit rows')
    unique = {r['record_id'] for r in manifest}
    if len(unique) != EXPECTED_RECORDS or len(set(r['record_id'] for r in windows)) != EXPECTED_RECORDS:
        raise ValueError('Duplicate record IDs')
    if any(r['sha256_match'] != 'matched' for r in manifest):
        raise ValueError('One or more EDF hashes have not been verified')
    if len({r['subject_id'] for r in manifest}) != EXPECTED_SUBJECTS:
        raise ValueError('Expected 78 subject IDs')
    groups = Counter(r['subject_id'] for r in manifest)
    if sum(n == 2 for n in groups.values()) != EXPECTED_PAIRS:
        raise ValueError('Expected 75 complete pairs')
    windows_by_id = {r['record_id']: r for r in windows}
    if unique != set(windows_by_id):
        raise ValueError('Window audit and manifest have different records')
    for r in manifest:
        w = windows_by_id[r['record_id']]
        if (r['subject_id'], str(r['night_index'])) != (w['subject_id'], str(w['night_index'])):
            raise ValueError(f'Metadata mismatch in window CSV: {r["record_id"]}')
        if w['scorer_code'] != r['scorer_code']:
            raise ValueError(f'Scorer mismatch: {r["record_id"]}')
    manifest.sort(key=lambda r: (r['subject_id'], int(r['night_index'])))

    with ZipFile(args.zip) as archive:
        for r in manifest:
            for kind, expected_member in [('psg', 'psg_member'), ('hyp', 'hypnogram_member')]:
                member = r[expected_member]
                info = archive.getinfo(member)
                if info.file_size != int(r['psg_uncompressed_bytes' if kind == 'psg' else 'hyp_uncompressed_bytes']):
                    raise ValueError(f'Manifest size differs from ZIP: {member}')
    stamps = {
        'config_sha256': config_hash,
        'manifest_sha256': manifest_hash,
        'window_csv_sha256': windows_hash,
        'window_summary_sha256': sha256(args.window_summary),
        'extractor_sha256': sha256(Path(__file__)),
        'pilot_functions_sha256': sha256(Path(sys.modules[process_record.__module__].__file__)),
    }
    return manifest, windows_by_id, cfg, stamps


def ensure_same_sources(stamps: dict, args: argparse.Namespace) -> None:
    for label, path in [('config_sha256', args.config), ('manifest_sha256', args.manifest),
                        ('window_csv_sha256', args.window_audit),
                        ('window_summary_sha256', args.window_summary)]:
        if sha256(path) != stamps[label]:
            raise RuntimeError(f'Input was modified during run: {path}')


def verify_pilot(actual: list[dict], pilot: list[dict], record_id: str) -> None:
    expected = [r for r in pilot if r['record_id'] == record_id]
    if not expected:
        return
    if len(expected) != len(actual):
        raise AssertionError(f'Pilot row count differs for {record_id}: {len(expected)} vs {len(actual)}')
    for idx, (one, two) in enumerate(zip(actual, expected)):
        for key in COLUMNS:
            if key in FEATURE_NAMES:
                a, b = float(one[key]), float(two[key])
                if not math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12):
                    raise AssertionError(f'Pilot comparison mismatch {record_id}, row {idx}, {key}')
            else:
                if str(one[key]) != str(two[key]):
                    raise AssertionError(f'Pilot comparison mismatch {record_id}, row {idx}, {key}')


def validate_features(rows: list[dict], meta: dict, window: dict, rec_id: str) -> None:
    if meta['record_id'] != rec_id:
        raise AssertionError('Record ID mismatch')
    fields = [
        ('main_sleep_window_start_epoch', 'main_start_epoch'),
        ('main_sleep_window_end_epoch_exclusive', 'main_end_epoch_exclusive'),
        ('main_sleep_window_length_epochs', 'main_window_epochs'),
        ('all_recording_epochs', 'all_recording_epochs'),
        ('episode_count', 'number_of_sleep_episodes'),
        ('sensitivity_window_start_epoch', 'first_last_start_epoch'),
        ('sensitivity_window_end_epoch_exclusive', 'first_last_end_epoch_exclusive'),
    ]
    for a, b in fields:
        if as_int(meta[a], a) != as_int(window[b], b):
            raise AssertionError(f'{rec_id}: window audit differs for {a} vs {b}')
    if meta['excluded_by_label'] != int(window['main_excluded_epochs']):
        raise AssertionError(f'{rec_id}: excluded annotations differ from window audit')
    expected_stage_counts = json.loads(window['stage_counts_main'])
    for cls in [*CLASSES, 'None']:
        if int(meta['window_reference_stage_counts'].get(cls, 0)) != int(expected_stage_counts.get(cls, 0)):
            raise AssertionError(f'{rec_id}: reference-stage counts differ: {cls}')
    if len(rows) != int(meta['feature_rows']):
        raise AssertionError('Feature row count differs from extraction report')
    if len(rows) + meta['rejected_nonfinite'] + meta['rejected_zero_variance'] + meta['excluded_by_label'] != int(window['main_window_epochs']):
        raise AssertionError(f'{rec_id}: window accounting mismatch')
    unique_epochs = set()
    observed = Counter()
    for r in rows:
        idx = int(r['epoch_index'])
        if idx in unique_epochs:
            raise AssertionError(f'{rec_id}: duplicate epoch index {idx}')
        unique_epochs.add(idx)
        if str(r['record_id']) != rec_id:
            raise AssertionError(f'{rec_id}: unexpected record id in feature row')
        if str(r['stage']) not in CLASSES:
            raise AssertionError(f'{rec_id}: unknown stage {r["stage"]}')
        if idx < int(window['main_start_epoch']) or idx >= int(window['main_end_epoch_exclusive']):
            raise AssertionError(f'{rec_id}: epoch outside main window')
        if int(r['epoch_start_sec']) != idx * 30:
            raise AssertionError(f'{rec_id}: misaligned epoch start')
        values = [float(r[k]) for k in FEATURE_NAMES]
        if not all(math.isfinite(v) for v in values):
            raise AssertionError(f'{rec_id}: non-finite features')
        rel = values[5:]
        if any(v < -1e-12 or v > 1+1e-12 for v in rel) or not math.isclose(sum(rel), 1.0, abs_tol=1e-9):
            raise AssertionError(f'{rec_id}: relative powers do not sum to one')
        observed[r['stage']] += 1
    if sum(observed.values()) != len(rows):
        raise AssertionError('Stage counting error')
    if any(observed[cls] > int(expected_stage_counts.get(cls, 0)) for cls in CLASSES):
        raise AssertionError(f'{rec_id}: stage counts exceed window reference')


def current_versions() -> dict[str, str]:
    return {
        'python': sys.version.split()[0],
        'mne': mne.__version__,
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'yaml': yaml.__version__,
    }


def verify_existing(output: Path, record: dict, stamps: dict, versions: dict, window: dict) -> dict:
    audit = output / 'audit.json'
    features = output / 'features.csv'
    if not audit.is_file() or not features.is_file():
        raise RuntimeError(f'Incomplete saved record {output}; do not overwrite it automatically')
    report = read_json(audit)
    for key, value in stamps.items():
        if report['input_hashes'].get(key) != value:
            raise RuntimeError(f'Resumption blocked: saved {record["record_id"]} used a different {key}')
    if report.get('versions') != versions:
        raise RuntimeError(f'Resumption blocked: library versions changed for {record["record_id"]}')
    if report.get('source_psg_sha256') != record['sha256_actual_psg'] or report.get('source_hyp_sha256') != record['sha256_actual_hyp']:
        raise RuntimeError(f'Resumption blocked: source hash changed for {record["record_id"]}')
    if report.get('features_csv_sha256') != sha256(features):
        raise RuntimeError(f'Saved {record["record_id"]} features CSV SHA-256 differs from audit')
    with features.open(newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            raise RuntimeError(f'Column mismatch in saved {features}')
        existing_rows = list(reader)
    validate_features(existing_rows, report['extraction'], window, record['record_id'])
    return report


def save_record_atomic(out_root: Path, record: dict, rows: list[dict], meta: dict,
                       stamps: dict, versions: dict, batch_epochs: int) -> dict:
    rec_dir = out_root / 'records'
    rec_id = record['record_id']
    final = rec_dir / rec_id
    if final.exists():
        raise FileExistsError(f'{final} already exists; no automatic overwrites')
    stage_dir = Path(tempfile.mkdtemp(prefix=f'.stage-{rec_id}-', dir=str(rec_dir)))
    try:
        features_file = stage_dir / 'features.csv'
        with features_file.open('w', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
        report = {
            'record_id': rec_id,
            'source_psg_sha256': record['sha256_actual_psg'],
            'source_hyp_sha256': record['sha256_actual_hyp'],
            'input_hashes': stamps,
            'versions': versions,
            'batch_epochs': batch_epochs,
            'features_csv_sha256': sha256(features_file),
            'extraction': meta,
            'purpose': 'Frozen v1.2 spectral features; no model training',
        }
        write_json(stage_dir / 'audit.json', report)
        # Directory rename commits both files as one verified record.
        os.rename(stage_dir, final)
        return report
    finally:
        if stage_dir.exists():
            shutil.rmtree(stage_dir)


def combine_atomic(out_root: Path, records: list[dict], stamps: dict, versions: dict,
                   windows: dict[str, dict]) -> tuple[Path, dict]:
    out = out_root / 'features_all_153.csv'
    audit_file = out_root / 'features_all_153_audit.json'
    reports = []
    for rec in records:
        reports.append(verify_existing(out_root/'records'/rec['record_id'], rec, stamps, versions,
                                       windows[rec['record_id']]))
    if out.exists() != audit_file.exists():
        raise RuntimeError('Incomplete aggregate output: one of CSV / JSON is missing')
    if out.exists():
        audit = read_json(audit_file)
        if audit.get('input_hashes') != stamps or audit.get('versions') != versions or audit.get('aggregate_sha256') != sha256(out):
            raise RuntimeError('Existing aggregate audit differs; refusing to overwrite')
        print('Previously completed aggregate verified; no overwrite', flush=True)
        return out, audit

    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8',
                                         dir=str(out_root), prefix='.features_all-',
                                         suffix='.part', delete=False) as f:
            tmp = Path(f.name)
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            all_rows = 0
            stage_counts = Counter()
            for rec in records:
                path = out_root / 'records' / rec['record_id'] / 'features.csv'
                with path.open('r', encoding='utf-8', newline='') as src:
                    reader = csv.DictReader(src)
                    if reader.fieldnames != COLUMNS:
                        raise ValueError(f'Invalid columns in {path}')
                    for row in reader:
                        writer.writerow(row)
                        all_rows += 1
                        stage_counts[row['stage']] += 1
        aggregate_hash = sha256(tmp)
        audit = {
            'status': 'PASS',
            'purpose': 'EEG-derived features, no model fit or performance analysis',
            'records': len(records),
            'subjects': len({r['subject_id'] for r in records}),
            'feature_rows': all_rows,
            'columns': COLUMNS,
            'stage_counts': dict(stage_counts),
            'input_hashes': stamps,
            'versions': versions,
            'aggregate_sha256': aggregate_hash,
            'per_record_hashes': {r['record_id']: rep['features_csv_sha256'] for r, rep in zip(records, reports)},
        }
        # Guard audit in staged form, then publish CSV and audit atomically one by one;
        # if interrupted between renames, the next run flags the pair as incomplete.
        with tempfile.NamedTemporaryFile(mode='w', newline='', encoding='utf-8',
                                         dir=str(out_root), prefix='.features_audit-',
                                         suffix='.part', delete=False) as f:
            tmp_audit = Path(f.name)
        write_json(tmp_audit, audit)
        os.rename(tmp, out)
        tmp = None
        os.rename(tmp_audit, audit_file)
        return out, audit
    finally:
        if tmp is not None and tmp.exists():
            tmp.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--zip', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--window-audit', type=Path, required=True)
    parser.add_argument('--window-summary', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--tmp', type=Path, default=None,
                        help='Existing scratch directory for ONE transient EDF (default macOS temp)')
    parser.add_argument('--batch-epochs', type=int, default=64)
    parser.add_argument('--limit', type=int, default=None,
                        help='For preflight: process at most N sorted records; no aggregate unless all 153 done')
    parser.add_argument('--pilot-csv', type=Path, default=None,
                        help='Optional validated 2-record pilot CSV; compare overlap numerically')
    args = parser.parse_args()
    if args.batch_epochs < 1 or args.batch_epochs > 128:
        raise ValueError('--batch-epochs must be from 1 to 128 to bound RAM use')
    if args.limit is not None and not (1 <= args.limit <= EXPECTED_RECORDS):
        raise ValueError('--limit must be between 1 and 153')
    if not args.out.parent.is_dir():
        raise FileNotFoundError(f'Result parent directory does not exist: {args.out.parent}')

    records, windows, cfg, stamps = validate_inputs(args)
    versions = current_versions()
    temp = args.tmp or Path(tempfile.gettempdir())
    if not temp.is_dir():
        raise FileNotFoundError(f'Scratch directory not found: {temp}')
    if args.pilot_csv:
        if not args.pilot_csv.is_file():
            raise FileNotFoundError(f'Pilot CSV not found: {args.pilot_csv}')
        pilot = read_csv(args.pilot_csv)
        if len(pilot) != 1968 or Counter(r['record_id'] for r in pilot) != Counter({'SC4001E':841, 'SC4002E':1127}):
            raise ValueError('Expected the previously approved 1,968-row subject00 pilot CSV')
        if set(pilot[0]) != set(COLUMNS):
            raise ValueError('Unexpected pilot CSV columns')
        print('Verified pilot reference: 1968 rows', flush=True)
    else:
        pilot = []

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'records').mkdir(exist_ok=True)
    selected = records[:args.limit] if args.limit else records
    completed = reused = 0
    with ZipFile(args.zip) as archive:
        for ordinal, rec in enumerate(selected, 1):
            rec_id = rec['record_id']
            target = args.out/'records'/rec_id
            if target.exists():
                report = verify_existing(target, rec, stamps, versions, windows[rec_id])
                # If this record has pilot reference, ensure skipped rows still match it.
                if pilot and any(r['record_id'] == rec_id for r in pilot):
                    saved_rows = read_csv(target/'features.csv')
                    verify_pilot(saved_rows, pilot, rec_id)
                reused += 1
                print(f'[{ordinal}/{len(selected)}] SKIP verified {rec_id} | '
                      f'{report["extraction"]["feature_rows"]} epochs', flush=True)
                continue
            max_size = (archive.getinfo(rec['psg_member']).file_size +
                        archive.getinfo(rec['hypnogram_member']).file_size)
            reserve = 512 * 1024**2
            if shutil.disk_usage(temp).free < max_size + reserve:
                raise RuntimeError(f'Not enough scratch space for {rec_id}; '
                                   f'need {max_size/1024**2:.0f} MiB + 512 MiB reserve')
            if shutil.disk_usage(args.out).free < 512 * 1024**2:
                raise RuntimeError('Less than 512 MiB free on result volume; stop safely')
            rows, meta = process_record(archive, rec, cfg, temp, batch_epochs=args.batch_epochs)
            ensure_same_sources(stamps, args)
            validate_features(rows, meta, windows[rec_id], rec_id)
            verify_pilot(rows, pilot, rec_id)
            save_record_atomic(args.out, rec, rows, meta, stamps, versions, args.batch_epochs)
            completed += 1
            print(f'[{ordinal}/{len(selected)}] PASS {rec_id} | '
                  f'window={meta["main_sleep_window_length_epochs"]}, '
                  f'features={len(rows)}, excluded={meta["excluded_by_label"]}, '
                  f'rejected={meta["rejected_nonfinite"]+meta["rejected_zero_variance"]}', flush=True)

    if args.limit is not None and args.limit < EXPECTED_RECORDS:
        print(f'PARTIAL — {completed} new + {reused} previously verified; '
              f'{EXPECTED_RECORDS-len(selected)} remaining; no aggregate or ML', flush=True)
        print('Run the same command WITHOUT --limit to complete and resume.', flush=True)
        return

    all_dir = args.out/'records'
    if sum((all_dir/r['record_id']).is_dir() for r in records) != EXPECTED_RECORDS:
        raise AssertionError('Not all 153 record directories exist; cannot aggregate')
    combined_path, agg = combine_atomic(args.out, records, stamps, versions, windows)
    ensure_same_sources(stamps, args)
    if agg['records'] != EXPECTED_RECORDS or agg['subjects'] != EXPECTED_SUBJECTS:
        raise AssertionError('Final aggregate cohort counts differ')
    print('\nPASS — 153 records extracted and independently audited; no ML trained', flush=True)
    print('Subject count:', agg['subjects'], 'Feature rows:', agg['feature_rows'], flush=True)
    print('CSV:', combined_path, flush=True)
    print('Audit:', args.out/'features_all_153_audit.json', flush=True)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nINTERRUPTED — completed records are preserved; rerun the same command to resume', file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f'FAILED — {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
