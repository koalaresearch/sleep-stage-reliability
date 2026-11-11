#!/usr/bin/env python3
"""Frozen Core evaluation for Sleep-EDF SC v1.2: LOSO LR vs Dummy prior.

Requires validated features and a PASS pre-model QA report. Models are trained
only after explicit --run. No EDF read, no ZIP, no hyperparameter search.

Each of the 78 held-out subject predictions is saved atomically as a resumable
checkpoint with data/config/code SHA-256. Training-only StandardScaler via
sklearn Pipeline. Two-phase convergence policy: first try all 78 models with
max_iter=2000. If ANY logistic regression fails to converge, re-run ALL 78
from scratch with max_iter=10000, retain and report any remaining warnings.

Results are published only after full 78-subject OOF coverage and audit.
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
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

STAGES = ('W', 'N1', 'N2', 'N3', 'REM')
BANDS = ('delta', 'theta', 'alpha', 'sigma', 'beta')
FEATURES = [f'log10_{b}_uv2' for b in BANDS] + [f'rel_{b}' for b in BANDS]
METADATA = ('row_index', 'subject_id', 'record_id', 'night_index',
            'epoch_index', 'epoch_start_sec', 'stage')
PROBA_COLS = [f'{model}_p_{stage}' for model in ('lr', 'dummy') for stage in STAGES]
PRED_COLS = [*METADATA, 'lr_pred', 'dummy_pred', *PROBA_COLS]


def must(ok: bool, why: str) -> None:
    if not ok:
        raise AssertionError(why)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda: f.read(1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()


def read_json(p: Path) -> dict:
    return json.loads(p.read_text(encoding='utf8'))


def write_json(p: Path, value: dict) -> None:
    p.write_text(json.dumps(value, indent=2, ensure_ascii=False,
                            allow_nan=False) + '\n', encoding='utf8')


def validate_params(cfg: dict) -> dict:
    must(cfg['study']['protocol'] == 'v1.2', 'Wrong protocol')
    must(cfg['validation']['split'] == 'LeaveOneGroupOut', 'Wrong split')
    must(cfg['validation']['groups'] == 'subject_id', 'Wrong grouping')
    must(cfg['validation']['primary_f1_class_set'] == 'classes_with_true_support_in_subject',
         'Wrong primary metric')
    must(cfg['models']['dummy']['strategy'] == 'prior', 'Dummy must use prior')
    must(cfg['models']['logistic_regression']['scaler'] == 'StandardScaler',
         'Scaler differs')
    model = cfg['models']['logistic_regression']
    must(model['penalty'] == 'l2' and model['solver'] == 'lbfgs', 'Model not frozen')
    must(float(model['C']) == 1.0 and float(model['tol']) == 1e-4,
         'LR hyperparameters differ')
    must(model['class_weight'] is None and int(model['max_iter']) == 2000,
         'LR class weight or iteration count differs')
    must(model['convergence_failure_policy'] == 'rerun_once_max_iter_10000_then_keep_and_report',
         'Convergence policy differs')
    must(cfg['preprocessing']['stage_order'] == list(STAGES),
         'Unexpected stage order')
    must(cfg['validation']['bootstrap_unit'] == 'subject_id',
         'Unexpected bootstrap unit')
    must(cfg['validation']['bootstrap_subject_replicates'] == 2000,
         'Unexpected bootstrap count')
    return model


def load_inputs(args: argparse.Namespace):
    for path in (args.features, args.qa, args.config):
        must(path.is_file(), f'Missing file {path}')
    must(args.out.parent.is_dir(), 'Output parent folder missing')
    cfg = yaml.safe_load(args.config.read_text(encoding='utf8'))
    m = validate_params(cfg)
    qa = read_json(args.qa)
    must(qa.get('status') == 'PASS' and qa.get('test_coverage_exactly_once') is True,
         'Pre-model QA is not PASS')
    must(qa.get('loso_folds') == 78, 'Pre-model QA LOSO count differs')
    input_hashes = {
        'features_sha256': sha256(args.features),
        'config_sha256': sha256(args.config),
        'qa_sha256': sha256(args.qa),
        'runner_sha256': sha256(Path(__file__)),
    }
    original_hashes = set(qa.get('input_sha256', {}).values())
    must(input_hashes['features_sha256'] in original_hashes,
         'QA does not cover this features.csv')
    must(input_hashes['config_sha256'] in original_hashes,
         'QA does not cover this study YAML')
    df = pd.read_csv(args.features, dtype={'subject_id': str, 'record_id': str},
                     keep_default_na=False)
    must(len(df) == int(qa['feature_rows']) and df['subject_id'].nunique() == 78,
         'Feature dimensions changed since preflight')
    must(all(c in df.columns for c in FEATURES), 'Missing feature columns')
    must(df['stage'].isin(STAGES).all(), 'Unknown class label')
    x = df[FEATURES].to_numpy(dtype=np.float64)
    y = df['stage'].to_numpy(dtype=str)
    subject = df['subject_id'].to_numpy(dtype=str)
    must(np.isfinite(x).all(), 'Invalid feature matrix')
    must(df['record_id'].nunique() == 153, 'Wrong record count')
    must(set(y) == set(STAGES), 'Missing global stage')
    must(len(x) == len(y) == len(subject), 'Input dimensions differ')
    return cfg, m, qa, input_hashes, df, x, y, subject


def model_for_fold(settings: dict, max_iter: int) -> Pipeline:
    return Pipeline([
        ('scale', StandardScaler()),
        ('lr', LogisticRegression(
            penalty=settings['penalty'], C=float(settings['C']),
            solver=settings['solver'], tol=float(settings['tol']),
            max_iter=max_iter, class_weight=settings['class_weight'],
        )),
    ])


def align_probas(model, x_test: np.ndarray) -> np.ndarray:
    p = np.asarray(model.predict_proba(x_test), dtype=float)
    labels = [str(c) for c in model.classes_]
    must(set(labels) == set(STAGES), 'Training fold lacks expected classes')
    p = p[:, [labels.index(cls) for cls in STAGES]]
    must(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), 'Invalid probabilities')
    must(np.allclose(p.sum(axis=1), 1.0, atol=1e-8, rtol=0),
         'Probabilities do not sum to one')
    return p


def checkpoint_dir(root: Path, subject_id: str) -> Path:
    return root / f'subject_{subject_id}'


def load_checkpoint(root: Path, subject_id: str, n: int,
                    hashes: dict, iteration_count: int) -> dict | None:
    dest = checkpoint_dir(root, subject_id)
    if not dest.exists():
        return None
    must(dest.is_dir(), f'Invalid saved checkpoint: {dest}')
    pred_path = dest / 'predictions.csv'
    meta_path = dest / 'audit.json'
    must(pred_path.is_file() and meta_path.is_file(),
         f'Incomplete checkpoint: {dest}; no overwrite')
    meta = read_json(meta_path)
    must(meta['input_hashes'] == hashes and meta['subject_id'] == subject_id,
         f'Checkpoint source differs: {subject_id}')
    must(meta['max_iter'] == iteration_count and meta['n_rows'] == n,
         f'Checkpoint model spec differs: {subject_id}')
    must(meta['prediction_sha256'] == sha256(pred_path),
         f'Checkpoint file hash differs: {subject_id}')
    return meta


def train_save_fold(root: Path, subject_id: str, train_idx: np.ndarray,
                    test_idx: np.ndarray, x: np.ndarray, y: np.ndarray,
                    df: pd.DataFrame, settings: dict, input_hashes: dict,
                    max_iter: int) -> dict:
    expected = checkpoint_dir(root, subject_id)
    saved = load_checkpoint(root, subject_id, len(test_idx), input_hashes, max_iter)
    if saved is not None:
        print(f'SKIP verified {subject_id} ({len(test_idx)} epochs)', flush=True)
        return saved
    must(not expected.exists(), f'Destination occupied: {expected}')
    must(subject_id not in set(df['subject_id'].to_numpy()[train_idx]),
         'Held-out subject in training data')
    must(set(df['subject_id'].to_numpy()[test_idx]) == {subject_id},
         'Test set includes another subject')
    must(set(y[train_idx]) == set(STAGES), 'Training fold lacks one or more classes')
    lr = model_for_fold(settings, max_iter)
    dummy = DummyClassifier(strategy='prior')
    with threadpool_limits(limits=1):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            lr.fit(x[train_idx], y[train_idx])
        dummy.fit(x[train_idx], y[train_idx])
        p_lr = align_probas(lr, x[test_idx])
        p_dummy = align_probas(dummy, x[test_idx])
        pred_lr = lr.predict(x[test_idx])
        pred_dummy = dummy.predict(x[test_idx])
    convergence = [str(w.message) for w in caught
                   if issubclass(w.category, ConvergenceWarning)]
    unexpected_warn = [str(w.message) for w in caught
                       if not issubclass(w.category, ConvergenceWarning)]
    raw_meta = df.iloc[test_idx][['subject_id','record_id','night_index',
                                  'epoch_index','epoch_start_sec','stage']].copy()
    raw_meta.insert(0, 'row_index', test_idx.astype(np.int64))
    raw_meta['lr_pred'] = pred_lr
    raw_meta['dummy_pred'] = pred_dummy
    for j, cls in enumerate(STAGES):
        raw_meta[f'lr_p_{cls}'] = p_lr[:, j]
        raw_meta[f'dummy_p_{cls}'] = p_dummy[:, j]
    raw_meta = raw_meta[PRED_COLS]
    must(list(raw_meta.columns) == PRED_COLS, 'Unexpected OOF columns')
    root.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(dir=root, prefix=f'.{subject_id}-'))
    try:
        output = staging / 'predictions.csv'
        raw_meta.to_csv(output, index=False, float_format='%.17g')
        report = {
            'subject_id': subject_id, 'n_train_rows': int(len(train_idx)),
            'n_train_subjects': 77, 'n_rows': int(len(test_idx)),
            'n_test_records': int(df.iloc[test_idx]['record_id'].nunique()),
            'max_iter': max_iter, 'n_iter': int(lr.named_steps['lr'].n_iter_[0]),
            'convergence_warnings': convergence,
            'other_warnings': unexpected_warn,
            'prediction_sha256': sha256(output),
            'input_hashes': input_hashes,
            'versions': {'sklearn': sklearn.__version__,
                         'numpy': np.__version__, 'pandas': pd.__version__},
        }
        write_json(staging / 'audit.json', report)
        os.rename(staging, expected)
        print(f'PASS subject {subject_id} | test_epochs={len(test_idx)} | '
              f'LR_iter={report["n_iter"]} | convergence_warnings={len(convergence)}', flush=True)
        return report
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def classwise_f1(y_true: np.ndarray, y_pred: np.ndarray):
    tp = np.array([np.sum((y_true == c) & (y_pred == c)) for c in STAGES], dtype=float)
    real = np.array([np.sum(y_true == c) for c in STAGES], dtype=float)
    predicted = np.array([np.sum(y_pred == c) for c in STAGES], dtype=float)
    denom = real + predicted
    f1 = np.divide(2 * tp, denom, out=np.zeros(len(STAGES), dtype=float), where=denom > 0)
    return f1, real > 0, (real + predicted) > 0


def summary_scores(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    f1, present, union = classwise_f1(y_true, y_pred)
    must(present.any(), 'Empty held-out subject')
    return {
        'true_support': float(f1[present].mean()),
        'fixed_5': float(f1.mean()),
        'union': float(f1[union].mean()),
        'n_true_classes': int(present.sum()),
    }


def weighted_pooled_f1(y_true, y_pred, weights) -> float:
    f = []
    for cls in STAGES:
        is_true, is_pred = y_true == cls, y_pred == cls
        tp = float(weights[is_true & is_pred].sum())
        fp = float(weights[~is_true & is_pred].sum())
        fn = float(weights[is_true & ~is_pred].sum())
        denom = 2 * tp + fp + fn
        f.append((2 * tp / denom) if denom > 0 else 0.0)
    return float(np.mean(f))


def publish(out: Path, checkpoint_root: Path, input_hashes: dict,
            cfg: dict, df: pd.DataFrame, y: np.ndarray,
            subjects: np.ndarray, selected_max_iter: int) -> None:
    pub = out / 'published'
    if pub.exists():
        summary = pub / 'loso_summary.json'
        must(summary.is_file() and read_json(summary).get('input_hashes') == input_hashes,
             'Published output exists but does not match this run')
        print('Previously published results detected; left unchanged:', pub)
        return
    chunks = []
    warnings_all = []
    for s in sorted(set(subjects)):
        fold = checkpoint_dir(checkpoint_root, s)
        must(fold.is_dir(), f'Missing fold {s}')
        cp = load_checkpoint(checkpoint_root, s, int(np.sum(subjects == s)),
                             input_hashes, selected_max_iter)
        assert cp is not None
        warnings_all.extend([{'subject': s, 'message': m} for m in cp['convergence_warnings']])
        chk = pd.read_csv(fold / 'predictions.csv', dtype={'subject_id':str,'record_id':str},
                          keep_default_na=False)
        must(list(chk.columns) == PRED_COLS, f'Incorrect columns for {s}')
        chunks.append(chk)
    merged = pd.concat(chunks, ignore_index=True).sort_values('row_index').reset_index(drop=True)
    must(len(merged) == len(df) and merged['row_index'].is_unique,
         'OOF row coverage / uniqueness failure')
    must(np.array_equal(merged['row_index'].to_numpy(dtype=np.int64), np.arange(len(df))),
         'OOF row index not exactly 0..N-1')
    for col in ('subject_id', 'record_id', 'night_index', 'epoch_index', 'epoch_start_sec', 'stage'):
        must(np.array_equal(merged[col].astype(str).to_numpy(), df[col].astype(str).to_numpy()),
             f'OOF metadata drift: {col}')
    for model in ('lr', 'dummy'):
        pred = merged[f'{model}_pred'].to_numpy(dtype=str)
        must(set(pred).issubset(STAGES), f'{model} unexpected labels')
        p = merged[[f'{model}_p_{s}' for s in STAGES]].to_numpy(dtype=np.float64)
        must(np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all(),
             f'{model} invalid probabilities')
        must(np.allclose(p.sum(axis=1), 1.0, atol=1e-8, rtol=0),
             f'{model} probabilities not normalized')
        predicted_index = np.array([STAGES.index(s) for s in pred], dtype=np.int64)
        must(np.all(np.isclose(p[np.arange(len(p)), predicted_index], p.max(axis=1), atol=1e-12)),
             f'{model} prediction is not a maximum-probability class')
    subject_rows = []
    for s, subset in merged.groupby('subject_id', sort=True):
        truth = subset['stage'].to_numpy(dtype=str)
        r = {'subject_id':str(s), 'n_epochs':int(len(subset)),
             'n_records':int(subset['record_id'].nunique())}
        for model in ('lr', 'dummy'):
            score = summary_scores(truth, subset[f'{model}_pred'].to_numpy(dtype=str))
            r.update({f'{model}_f1_{k}':v for k,v in score.items()})
        subject_rows.append(r)
    sub = pd.DataFrame(subject_rows)
    must(len(sub)==78, 'Subject score count differs')
    rng = np.random.default_rng(int(cfg['validation']['bootstrap_random_state']))
    reps = int(cfg['validation']['bootstrap_subject_replicates'])
    idx = rng.integers(0, len(sub), size=(reps, len(sub)))
    summary = {'status':'PASS', 'scope':'LOSO Core only: no night claims',
               'n_subjects':78, 'n_recordings':153, 'n_epochs':int(len(merged)),
               'n_fits_per_model':78, 'convergence_pass_max_iter':selected_max_iter,
               'remaining_convergence_warnings':warnings_all,
               'bootstrap_is_conditional_descriptive':True,
               'input_hashes':input_hashes,
               'models':{},
               'versions':{'sklearn':sklearn.__version__, 'numpy':np.__version__,
                           'pandas':pd.__version__}}
    subject_n = merged.groupby('subject_id').size()
    w = merged['subject_id'].map((1.0/subject_n).to_dict()).to_numpy(dtype=float)
    for model in ('lr','dummy'):
        scores = sub[f'{model}_f1_true_support'].to_numpy(dtype=float)
        means = scores[idx].mean(axis=1)
        pred = merged[f'{model}_pred'].to_numpy(dtype=str)
        summary['models'][model] = {
            'mean_subject_macro_f1':float(scores.mean()),
            'median_subject_macro_f1':float(np.median(scores)),
            'iqr_subject_macro_f1':[float(v) for v in np.quantile(scores,[0.25,0.75])],
            'sd_subject_macro_f1':float(scores.std(ddof=1)),
            'bootstrap_interval_subject_resample_conditional': [float(v) for v in np.quantile(means,[.025,.975])],
            'mean_subject_fixed_five_f1':float(sub[f'{model}_f1_fixed_5'].mean()),
            'mean_subject_union_f1':float(sub[f'{model}_f1_union'].mean()),
            'equal_subject_weighted_epoch_confusion_macro_f1':weighted_pooled_f1(y,pred,w),
            'raw_pooled_macro_f1':weighted_pooled_f1(y,pred,np.ones(len(y))),
        }
    staged = Path(tempfile.mkdtemp(dir=out, prefix='.publish-'))
    try:
        merged.to_csv(staged/'oof_predictions.csv', index=False, float_format='%.17g')
        sub.to_csv(staged/'subject_metrics.csv', index=False, float_format='%.17g')
        summary['artifact_sha256'] = {
            'oof_predictions_csv':sha256(staged/'oof_predictions.csv'),
            'subject_metrics_csv':sha256(staged/'subject_metrics.csv'),
        }
        write_json(staged/'loso_summary.json', summary)
        # Atomic directory promotion. Re-runs do not silently overwrite.
        os.rename(staged,pub)
        print(f'\nPASS — 78 LOSO folds, all {len(merged):,} epochs OOF, no subject overlap')
        print('LR mean subject macro-F1:',round(summary['models']['lr']['mean_subject_macro_f1'],4))
        print('Dummy mean subject macro-F1:',round(summary['models']['dummy']['mean_subject_macro_f1'],4))
        print('Results:', pub)
    finally:
        if staged.exists(): shutil.rmtree(staged)


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    for arg in ('features','qa','config','out'):
        p.add_argument('--'+arg, type=Path, required=True)
    p.add_argument('--run',action='store_true',help='Explicit authorization to fit 78 LR and 78 Dummy models')
    args=p.parse_args()
    cfg, settings, qa, hashes, df, x, y, subjects=load_inputs(args)
    must(len(x)==int(qa['feature_rows']),'Feature rows do not match independent QA')
    print(f'Input verified: {len(x):,} epochs | 78 subjects | 153 recordings')
    print('Source SHA-256:',hashes['features_sha256'])
    if not args.run:
        print('PREFLIGHT ONLY — no models trained. Add --run to fit.')
        return
    args.out.mkdir(parents=True,exist_ok=True)
    logo=LeaveOneGroupOut()
    split_list=[]
    coverage=np.zeros(len(y),dtype=np.uint8)
    for train,test in logo.split(x,y,groups=subjects):
        sset=set(subjects[test]);must(len(sset)==1,'Wrong subject test size')
        must(set(subjects[train]).isdisjoint(sset),'Subject overlap')
        coverage[test]+=1
        split_list.append((next(iter(sset)),train,test))
    must(len(split_list)==78 and np.all(coverage==1),'LOSO coverage failed')
    for limit in (2000,10000):
        saved_root=args.out/f'iteration_{limit}'
        saved_root.mkdir(parents=True,exist_ok=True)
        convergence=[]
        for no,(sid,train,test) in enumerate(split_list,1):
            print(f'[{no}/78 | max_iter={limit}] ',end='',flush=True)
            meta=train_save_fold(saved_root,sid,train,test,x,y,df,settings,hashes,limit)
            if meta['convergence_warnings']:
                convergence.append(sid)
        if convergence and limit==2000:
            print(f'First full pass finished with convergence warnings for {len(convergence)} subjects.')
            print('Protocol policy: repeating all 78 fits with max_iter=10000.')
            continue
        if convergence:
            print(f'WARNING: {len(convergence)} folds still have convergence warnings after retry.')
        publish(args.out,saved_root,hashes,cfg,df,y,subjects,limit)
        break


if __name__=='__main__':
    try: main()
    except KeyboardInterrupt:
        print('\nINTERRUPTED: checkpoints preserved. Re-run same command.',file=sys.stderr)
        sys.exit(130)
    except Exception as e:
        print(f'FAILED — {type(e).__name__}: {e}',file=sys.stderr)
        sys.exit(1)
