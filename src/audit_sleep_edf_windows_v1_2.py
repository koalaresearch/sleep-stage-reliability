#!/usr/bin/env python3
"""Label-only, read-only audit of v1.2 main-sleep windows across all Sleep-EDF SC records.

Uses the existing verified manifest/ZIP, reads PSG *headers* and small hypnograms,
but never reads PSG signal samples, extracts whole datasets, or trains models.
Outputs one CSV and a compact JSON audit; does not touch the v1.2 protocol.
Requires the previously validated local files audit_sleep_edf_epoch_alignment.py
and extract_sleep_edf_pilot_v1_2_fixed.py in the same directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import sys
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import mne
import numpy as np
import yaml

from audit_sleep_edf_epoch_alignment import read_psg_header_duration
from extract_sleep_edf_pilot_v1_2_fixed import epoch_labels, main_sleep_window

SLEEP = frozenset(('N1', 'N2', 'N3', 'REM'))
COLUMNS = (
    'subject_id', 'record_id', 'night_index', 'scorer_code',
    'all_recording_epochs', 'main_start_epoch', 'main_end_epoch_exclusive',
    'main_window_epochs', 'main_sleep_epochs', 'main_wake_epochs',
    'main_excluded_epochs', 'number_of_sleep_episodes',
    'all_record_sleep_epochs', 'sleep_epochs_outside_main',
    'first_last_start_epoch', 'first_last_end_epoch_exclusive',
    'first_last_window_epochs', 'windows_identical',
    'fraction_main_window_sleep', 'stage_counts_main',
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def audit_one(z: ZipFile, row: dict, cfg: dict, tmp: Path) -> dict:
    p = cfg['preprocessing']
    T = read_psg_header_duration(z, row['psg_member'])
    if abs(T/30.0 - round(T/30.0)) > 1e-8:
        raise ValueError(f"{row['record_id']}: PSG duration is not an integral number of epochs")
    n_epochs = int(round(T/30.0))
    with TemporaryDirectory(prefix='sleepedf-window-', dir=str(tmp)) as d:
        hyp = Path(d)/'record-Hypnogram.edf'
        with z.open(row['hypnogram_member']) as src, hyp.open('wb') as out:
            shutil.copyfileobj(src, out, length=1024*1024)
        ann = mne.read_annotations(str(hyp))
    stages = epoch_labels(ann, n_epochs, p['stage_map'], p['exclude_labels'])
    win = p['analysis_window']
    lo, hi, info = main_sleep_window(
        stages,
        gap_min=float(win['episode_merge_max_wake_gap_minutes']),
        context_min=float(win['wake_context_minutes_each_side']),
    )
    sleep_idxs = np.flatnonzero([s in SLEEP for s in stages])
    if len(sleep_idxs) == 0:
        raise ValueError(f"{row['record_id']}: no sleep epochs")
    pad = int(round(float(win['wake_context_minutes_each_side']) * 60 / 30))
    sens_lo = max(0, int(sleep_idxs[0])-pad)
    sens_hi = min(n_epochs, int(sleep_idxs[-1])+1+pad)
    counter = Counter(str(s) for s in stages[lo:hi])
    n_sleep_main = sum(counter[k] for k in SLEEP)
    n_excluded = counter.get('None', 0)
    n_wake = counter.get('W', 0)
    if n_sleep_main+n_wake+n_excluded != hi-lo:
        raise AssertionError(f"{row['record_id']}: inconsistent window label totals")
    return {
        'subject_id': row['subject_id'], 'record_id': row['record_id'],
        'night_index': int(row['night_index']), 'scorer_code': row['scorer_code'],
        'all_recording_epochs': n_epochs,
        'main_start_epoch': int(lo), 'main_end_epoch_exclusive': int(hi),
        'main_window_epochs': int(hi-lo),
        'main_sleep_epochs': int(n_sleep_main), 'main_wake_epochs': int(n_wake),
        'main_excluded_epochs': int(n_excluded),
        'number_of_sleep_episodes': int(info['episode_count']),
        'all_record_sleep_epochs': int(len(sleep_idxs)),
        'sleep_epochs_outside_main': int(len(sleep_idxs)-n_sleep_main),
        'first_last_start_epoch': int(sens_lo),
        'first_last_end_epoch_exclusive': int(sens_hi),
        'first_last_window_epochs': int(sens_hi-sens_lo),
        'windows_identical': bool((lo,hi)==(sens_lo,sens_hi)),
        'fraction_main_window_sleep': round(n_sleep_main/(hi-lo), 6),
        'stage_counts_main': json.dumps({k:counter.get(k,0) for k in ['W','N1','N2','N3','REM','None']}, sort_keys=True),
    }


def main() -> int:
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--zip', required=True, type=Path)
    ap.add_argument('--manifest', required=True, type=Path)
    ap.add_argument('--config', required=True, type=Path)
    ap.add_argument('--out', default=Path('results'), type=Path)
    ap.add_argument('--tmp', default=None, type=Path, help='Existing scratch directory; default OS temp folder')
    a=ap.parse_args()
    if not all(x.is_file() for x in (a.zip, a.manifest, a.config)):
        raise FileNotFoundError('ZIP, manifest or config missing')
    with a.manifest.open(newline='', encoding='utf-8') as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 153 or len(set(r['record_id'] for r in rows))!=153:
        raise AssertionError('Expected exactly 153 distinct SC records')
    if any(r['sha256_match']!='matched' for r in rows):
        raise AssertionError('Manifest does not contain 153 verified SHA-256 matches')
    with a.config.open(encoding='utf-8') as f:
        cfg=yaml.safe_load(f)
    if cfg['study']['protocol']!='v1.2':
        raise ValueError('Expected v1.2 study config')
    temp = a.tmp or Path(__import__('tempfile').gettempdir())
    if not temp.is_dir():
        raise FileNotFoundError(f'Scratch directory does not exist: {temp}')
    rows.sort(key=lambda r:(r['subject_id'], int(r['night_index'])))
    res=[]
    with ZipFile(a.zip) as z:
        for idx, r in enumerate(rows,1):
            item=audit_one(z, r, cfg, temp)
            res.append(item)
            if idx % 25 == 0 or idx==153:
                print(f'Checked {idx}/153 hypnograms; no PSG samples read', flush=True)
    multiple=[r['record_id'] for r in res if r['number_of_sleep_episodes']>1]
    dropped=[r['record_id'] for r in res if r['sleep_epochs_outside_main']>0]
    differing=[r['record_id'] for r in res if not r['windows_identical']]
    a.out.mkdir(parents=True, exist_ok=True)
    csv_dest=a.out/'window_audit_all_153.csv'
    json_dest=a.out/'window_audit_summary.json'
    with csv_dest.open('w', newline='', encoding='utf-8') as f:
        writer=csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader(); writer.writerows(res)
    summary={
        'status':'PASS', 'purpose':'label-only, pre-model window audit, not a performance result',
        'records':len(res), 'subjects':len({r['subject_id'] for r in res}),
        'records_with_multiple_episodes':len(multiple),
        'multiple_episode_record_ids':multiple,
        'records_with_sleep_outside_main':len(dropped),
        'sleep_outside_main_record_ids':dropped,
        'records_differing_from_first_last_window':len(differing),
        'differing_record_ids':differing,
        'total_sleep_epochs_excluded_by_main_window':sum(r['sleep_epochs_outside_main'] for r in res),
        'total_main_window_epochs':sum(r['main_window_epochs'] for r in res),
        'total_first_last_window_epochs':sum(r['first_last_window_epochs'] for r in res),
        'config_sha256':sha256(a.config), 'verified_manifest_sha256':sha256(a.manifest),
        'window_csv_sha256':sha256(csv_dest),
        'libraries':{'python':sys.version.split()[0], 'mne':mne.__version__},
        'note':'A full EDF signal alignment check and feature extraction still remain.',
    }
    json_dest.write_text(json.dumps(summary,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
    print('\nPASS — 153 hypnograms inspected; no ML run')
    print('More than one episode:',len(multiple))
    print('Main vs first-last window differs:',len(differing))
    print('Sleep epochs excluded by main-window selection:',summary['total_sleep_epochs_excluded_by_main_window'])
    print('Reports:',csv_dest, 'and',json_dest)
    return 0

if __name__=='__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f'FAILED — {type(exc).__name__}: {exc}',file=sys.stderr)
        raise SystemExit(1)
