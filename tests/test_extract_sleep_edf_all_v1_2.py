"""Synthetic orchestration tests: do not use real Sleep-EDF or run EEG ML."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest
import yaml

import extract_sleep_edf_all_v1_2 as pipeline


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_fixture(tmp_path: Path) -> dict:
    config = tmp_path / 'study_v1_2.yaml'
    config.write_bytes((Path(__file__).resolve().parents[1] / 'config' / 'study_v1_2.yaml').read_bytes())
    folder = tmp_path / 'results'
    folder.mkdir()
    archive = tmp_path / 'dataset.zip'
    manifest = folder / 'manifest_inventory.csv'
    window_csv = folder / 'window_audit_all_153.csv'
    window_summary = folder / 'window_audit_summary.json'
    manifest_rows = []
    audit_rows = []
    with ZipFile(archive, 'w') as z:
        for n in range(78):
            sid = f'{n:02}'
            for night in [1,2]:
                if (n==13 and night==2) or (n in (36,52) and night==1):
                    continue
                rid = f'SC4{sid}{night}E'
                psg_member = f'archive/sleep-cassette/{rid}0-PSG.edf'
                hyp_member = f'archive/sleep-cassette/{rid}C-Hypnogram.edf'
                psg_data = f'synthetic PSG {rid}'.encode()
                hyp_data = f'synthetic Hyp {rid}'.encode()
                z.writestr(psg_member, psg_data)
                z.writestr(hyp_member, hyp_data)
                manifest_rows.append(dict(subject_id=sid, record_id=rid, night_index=night, scorer_code='C',
                                          sha256_match='matched', psg_member=psg_member,
                                          hypnogram_member=hyp_member,
                                          psg_uncompressed_bytes=len(psg_data),
                                          hyp_uncompressed_bytes=len(hyp_data),
                                          sha256_actual_psg=digest(psg_data),
                                          sha256_actual_hyp=digest(hyp_data)))
                counts = dict(W=0,N1=0,N2=2,N3=0,REM=0,None_=0)
                # JSON key None is a string.
                counts['None'] = counts.pop('None_')
                audit_rows.append(dict(subject_id=sid,record_id=rid,night_index=night,scorer_code='C',
                                       main_start_epoch=0, main_end_epoch_exclusive=2,
                                       main_window_epochs=2, all_recording_epochs=2,
                                       number_of_sleep_episodes=1,first_last_start_epoch=0,
                                       first_last_end_epoch_exclusive=2,main_excluded_epochs=0,
                                       stage_counts_main=json.dumps(counts)))

    with manifest.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=manifest_rows[0].keys());writer.writeheader();writer.writerows(manifest_rows)
    with window_csv.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=audit_rows[0].keys());writer.writeheader();writer.writerows(audit_rows)
    summary=dict(status='PASS', config_sha256=pipeline.sha256(config),
                 verified_manifest_sha256=pipeline.sha256(manifest),
                 window_csv_sha256=pipeline.sha256(window_csv), records=153, subjects=78)
    window_summary.write_text(json.dumps(summary))
    return dict(zip=archive, manifest=manifest, windows=window_csv, summary=window_summary,
                config=config,out=folder/'features_v1_2')


def fake_record(_archive, row, _cfg, _temp, batch_epochs=64):
    rid = row['record_id']
    n = int(row['night_index'])
    feats={f: 0.2 for f in pipeline.FEATURE_NAMES}
    rows = [dict(subject_id=row['subject_id'],record_id=rid,night_index=n,
                 epoch_index=i,epoch_start_sec=30*i,stage='N2',**feats) for i in range(2)]
    meta = dict(record_id=rid,night_index=n,all_recording_epochs=2,
                main_sleep_window_start_epoch=0,main_sleep_window_end_epoch_exclusive=2,
                main_sleep_window_length_epochs=2,episode_count=1,
                sensitivity_window_start_epoch=0,sensitivity_window_end_epoch_exclusive=2,
                excluded_by_label=0,window_reference_stage_counts={'N2':2},
                feature_rows=2,rejected_nonfinite=0,rejected_zero_variance=0)
    return rows, meta


def call_main(monkeypatch, sources:dict, *flags):
    argv=['extract_sleep_edf_all_v1_2.py','--zip',str(sources['zip']),
          '--manifest',str(sources['manifest']),
          '--window-audit',str(sources['windows']),
          '--window-summary',str(sources['summary']),
          '--config',str(sources['config']),
          '--out',str(sources['out']),*flags]
    monkeypatch.setattr(sys,'argv',argv)
    pipeline.main()


def test_resume_and_complete_153_records(tmp_path,monkeypatch):
    sources=build_fixture(tmp_path)
    monkeypatch.setattr(pipeline,'process_record',fake_record)
    # Count calls to verify resume truly skips completed records.
    counts={'new':0}
    def fake_counted(*args,**kwargs):
        counts['new']+=1
        return fake_record(*args,**kwargs)
    monkeypatch.setattr(pipeline,'process_record',fake_counted)
    call_main(monkeypatch,sources,'--limit','2')
    assert counts['new']==2
    call_main(monkeypatch,sources,'--limit','2')
    assert counts['new']==2
    call_main(monkeypatch,sources)
    assert counts['new']==153
    combined=sources['out']/'features_all_153.csv'
    report=json.loads((sources['out']/'features_all_153_audit.json').read_text())
    assert report['status']=='PASS'
    assert report['subjects']==78
    assert report['records']==153
    assert report['feature_rows']==306
    assert report['aggregate_sha256']==pipeline.sha256(combined)
    call_main(monkeypatch,sources)
    assert counts['new']==153


def test_corrupt_saved_features_blocks_resume(tmp_path,monkeypatch):
    sources=build_fixture(tmp_path)
    monkeypatch.setattr(pipeline,'process_record',fake_record)
    call_main(monkeypatch,sources,'--limit','1')
    target=sources['out']/'records'/'SC4001E'/'features.csv'
    target.write_text(target.read_text()+'CORRUPTED\n')
    with pytest.raises(RuntimeError,match='SHA-256 differs'):
        call_main(monkeypatch,sources,'--limit','1')


def test_changed_yaml_blocks_preflight(tmp_path,monkeypatch):
    sources=build_fixture(tmp_path)
    monkeypatch.setattr(pipeline,'process_record',fake_record)
    with sources['config'].open('a') as f:
        f.write('\n# changed after window audit\n')
    with pytest.raises(ValueError,match='YAML changed'):
        call_main(monkeypatch,sources,'--limit','1')
