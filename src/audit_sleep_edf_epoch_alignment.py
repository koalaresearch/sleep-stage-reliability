#!/usr/bin/env python3
"""Audit 30-second Sleep-EDF PSG-to-hypnogram alignment for one SC subject.

Reads EDF fixed PSG headers and the (small) hypnogram EDF members directly
from the existing ZIP. Does NOT decompress/copy PSG signal data, load EEG,
change an existing manifest, or fit a model.

Dependencies: MNE, NumPy (project .venv).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

import mne
import numpy as np

ALLOWED = {
    'Sleep stage W', 'Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3',
    'Sleep stage 4', 'Sleep stage R', 'Sleep stage ?', 'Movement time',
}
SLEEP = {'Sleep stage 1', 'Sleep stage 2', 'Sleep stage 3', 'Sleep stage 4', 'Sleep stage R'}


def read_psg_header_duration(z: ZipFile, member: str) -> float:
    with z.open(member) as f:
        header = f.read(256)
    if len(header) != 256:
        raise ValueError(f'PSG header incomplete: {member}')
    try:
        count = int(header[236:244].decode('ascii').strip())
        duration = float(header[244:252].decode('ascii').strip())
    except (ValueError, UnicodeError) as exc:
        raise ValueError(f'Invalid EDF record-count/duration header: {member}') from exc
    if count <= 0 or not math.isfinite(duration) or duration <= 0:
        raise ValueError(f'Invalid duration in PSG EDF header: {member}')
    return count * duration


def audit_record(z: ZipFile, row: dict) -> dict:
    T = read_psg_header_duration(z, row['psg_member'])
    # Original PSG is never extracted; only hypnogram EDF is placed in a temp directory.
    with TemporaryDirectory(prefix='sleepedf-hyp-audit-') as temp:
        path = Path(temp) / 'record-Hypnogram.edf'
        with z.open(row['hypnogram_member']) as source, path.open('wb') as sink:
            for block in iter(lambda: source.read(1024 * 1024), b''):
                sink.write(block)
        ann = mne.read_annotations(str(path))

    if ann.orig_time is not None:
        raise ValueError(
            f"{row['record_id']}: annotations have orig_time={ann.orig_time}; "
            'manual timestamp alignment check required before applying onset values'
        )
    n_epochs = int(math.floor(T / 30.0 + 1e-9))
    remainder = T - 30 * n_epochs
    centers = (np.arange(n_epochs, dtype=float) + 0.5) * 30.0
    covered_count = np.zeros(n_epochs, dtype=np.uint16)
    labels = np.empty(n_epochs, dtype=object)
    labels[:] = None
    past_end = []
    before_start = []
    off_grid_boundaries = 0
    nonfinite = []
    for onset, dur, description in zip(ann.onset, ann.duration, ann.description):
        onset, dur, description = float(onset), float(dur), str(description)
        end = onset + dur
        if not all(map(math.isfinite, (onset, dur, end))) or dur <= 0:
            nonfinite.append([onset, dur, description])
            continue
        if onset < -1e-6:
            before_start.append({'label': description, 'start_s': round(onset, 3), 'end_s': round(end, 3)})
        if end > T + 1e-6:
            past_end.append({'label': description, 'start_s': round(onset, 3),
                             'end_s': round(end, 3), 'overhang_s': round(end - T, 3)})
        for x in (onset, end):
            if -1e-6 <= x <= T + 1e-6 and abs(x / 30.0 - round(x / 30.0)) > 1e-5:
                off_grid_boundaries += 1
        mask = (centers >= onset) & (centers < end)
        covered_count[mask] += 1
        labels[mask] = description

    label_counts = dict(sorted(Counter(str(x) for x in labels if x is not None).items()))
    unknown_labels = sorted(set(str(x) for x in ann.description) - ALLOWED)
    gaps = int(np.sum(covered_count == 0))
    multiple = int(np.sum(covered_count > 1))
    scorable = sum(label_counts.get(x, 0) for x in SLEEP | {'Sleep stage W'})
    excluded = sum(label_counts.get(x, 0) for x in {'Sleep stage ?', 'Movement time'})
    status = ('PASS' if not any([
        gaps, multiple, unknown_labels, off_grid_boundaries, nonfinite,
        before_start, abs(remainder) > 1e-5, scorable + excluded != n_epochs,
    ]) else 'REVIEW')
    return {
        'subject_id': row['subject_id'],
        'record_id': row['record_id'],
        'night_index': int(row['night_index']),
        'scorer_code': row['scorer_code'],
        'psg_duration_s': T,
        'total_complete_30s_epochs': n_epochs,
        'partial_epoch_remainder_s': round(remainder, 6),
        'hypnogram_annotation_count': len(ann),
        'annotation_orig_time': None,
        'epoch_label_counts': label_counts,
        'scorable_epoch_count': scorable,
        'excluded_epoch_count': excluded,
        'uncovered_epoch_count': gaps,
        'multiply_covered_epoch_count': multiple,
        'off_30s_grid_boundary_count': off_grid_boundaries,
        'unknown_labels': unknown_labels,
        'bad_duration_annotations': nonfinite,
        'annotations_before_start': before_start,
        'annotations_extending_beyond_psg': past_end,
        'alignment_status': status,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--subject', default='00')
    parser.add_argument('--out', type=Path, default=Path('results'))
    args = parser.parse_args()
    subject = str(args.subject).zfill(2)
    with args.manifest.open(newline='', encoding='utf-8') as f:
        rows = [row for row in csv.DictReader(f) if row['subject_id'] == subject]
    rows.sort(key=lambda row: int(row['night_index']))
    if [int(r['night_index']) for r in rows] != [1, 2]:
        raise ValueError(f'Subject {subject} does not have two nights in the manifest')
    if any(r['sha256_match'] != 'matched' for r in rows):
        raise ValueError('Manifest requires prior full SHA-256 PASS')
    with ZipFile(args.zip) as z:
        results = [audit_record(z, row) for row in rows]
    args.out.mkdir(parents=True, exist_ok=True)
    report_path = args.out / f'annotation_alignment_subject{subject}.json'
    report_path.write_text(json.dumps({'subject_id':subject, 'records':results},indent=2,
                                      ensure_ascii=False)+'\n',encoding='utf-8')
    for rec in results:
        print(f"{rec['record_id']}: {rec['alignment_status']} | "
              f"{rec['total_complete_30s_epochs']} epochs | "
              f"scorable={rec['scorable_epoch_count']} excluded={rec['excluded_epoch_count']} | "
              f"gaps={rec['uncovered_epoch_count']} overlaps={rec['multiply_covered_epoch_count']}")
        print('  Labels by 30-s epoch:', rec['epoch_label_counts'])
        print('  Annotations extending beyond PSG:', rec['annotations_extending_beyond_psg'])
        print('  Off-grid annotation boundaries:', rec['off_30s_grid_boundary_count'])
    print('Report:', report_path)
    overall = 'PASS' if all(x['alignment_status']=='PASS' for x in results) else 'REVIEW'
    print('Alignment audit:', overall)
    return 0 if overall == 'PASS' else 1


if __name__ == '__main__':
    raise SystemExit(main())
