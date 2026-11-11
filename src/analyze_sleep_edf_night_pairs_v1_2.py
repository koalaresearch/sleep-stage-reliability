#!/usr/bin/env python3
"""Sleep-EDF SC v1.2: read-only, no-refit paired-recording analysis.

Consumes the published, independently validated 78-subject LOSO OOF
probabilities, the original pair audit, and the pre-model window audit.
No EDF, ZIP, MNE, training or modification of source results.

Produces transparent, pre-specified, descriptive estimates for the 75 pairs.
The YAML's sqrt(sum(delta**2)/(2n)) is retained *as specified*, but named
RMS-derived within-pair scale, not a bias-adjusted within-person SD.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

STAGES = ('W', 'N1', 'N2', 'N3', 'REM')
EPS_EXPECTED = 161028
N_SUBJECTS, N_RECORDS, N_PAIRS = 78, 153, 75


def require(test: bool, message: str) -> None:
    if not test:
        raise ValueError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def parse_bool(value: Any) -> bool:
    value = str(value).strip().lower()
    if value in ('true', '1'):
        return True
    if value in ('false', '0'):
        return False
    raise ValueError('Expected explicit True/False, received: ' + repr(value))


def json_compatible(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(v) for v in value]
    if isinstance(value, np.ndarray):
        return [json_compatible(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return json_compatible(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def extract_epoch_log_loss(df: pd.DataFrame, epsilon: float) -> np.ndarray:
    probs = df[[f'lr_p_{stage}' for stage in STAGES]].to_numpy(dtype=np.float64)
    require(np.isfinite(probs).all(), 'Non-finite model probabilities')
    require(((probs >= 0) & (probs <= 1)).all(), 'Invalid probabilities')
    require(np.allclose(probs.sum(axis=1), 1.0, rtol=0, atol=1e-8),
            'Probabilities do not sum to one')
    clipped = np.clip(probs, epsilon, 1.0)
    clipped /= clipped.sum(axis=1, keepdims=True)
    order = {s: i for i, s in enumerate(STAGES)}
    true_idx = df['stage'].map(order).to_numpy()
    require(pd.notna(true_idx).all(), 'Invalid stage label')
    return -np.log(clipped[np.arange(len(df)), true_idx.astype(int)])


def aggregate_record(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Both frames are compact; no per-epoch predictions exported from this stage.
    records = []
    by_stage = []
    for (sid, rec, night), block in df.groupby(['subject_id','record_id','night_index'], sort=True):
        losses = block['lr_epoch_log_loss'].to_numpy(dtype=float)
        row = {
            'subject_id': str(sid), 'record_id': str(rec), 'night_index': int(night),
            'n_epochs': int(len(block)),
            'lr_mean_log_loss': float(losses.mean()),
            'lr_median_log_loss': float(np.median(losses)),
        }
        for stage in STAGES:
            chunk = block.loc[block['stage'] == stage, 'lr_epoch_log_loss']
            count = int(len(chunk))
            row[f'n_{stage}'] = count
            row[f'fraction_{stage}'] = count / len(block)
            by_stage.append({'subject_id': str(sid), 'record_id': str(rec),
                             'night_index': int(night), 'stage': stage,
                             'n_epochs': count,
                             'stage_mean_loss': float(chunk.mean()) if count else np.nan})
        records.append(row)
    return pd.DataFrame(records), pd.DataFrame(by_stage)


def pair_stats(deltas: np.ndarray) -> dict:
    a = np.asarray(deltas, dtype=float)
    if len(a) < 2 or not np.isfinite(a).all():
        return {'n':int(len(a)), 'available':False}
    # Protocol v1.2's specified quantity includes bias. The classical
    # s_w = SD(differences)/sqrt(2) removes mean bias; report both.
    rms_sw = float(np.sqrt(np.mean(np.square(a)) / 2))
    centered_sw = float(np.std(a, ddof=1) / np.sqrt(2))
    return {
        'n':int(len(a)), 'available':True,
        'mean_signed_nominal_delta':float(np.mean(a)),
        'median_signed_nominal_delta':float(np.median(a)),
        'mean_abs_delta':float(np.mean(np.abs(a))),
        'median_abs_delta':float(np.median(np.abs(a))),
        'abs_delta_q25_q75':[float(z) for z in np.quantile(np.abs(a),[.25,.75])],
        'sd_signed_delta':float(np.std(a,ddof=1)),
        'sw_rms_protocol_v1_2':rms_sw,
        'repeatability_coefficient_rms_protocol_v1_2':2.77*rms_sw,
        'sw_centered_bias_adjusted':centered_sw,
        'repeatability_coefficient_centered_descriptive':2.77*centered_sw,
    }


def bootstrap_interval(values: np.ndarray, method, n_reps: int, seed: int) -> list[float] | None:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if len(vals) < 2: return None
    rng = np.random.default_rng(seed)
    result = [float(method(vals[rng.integers(0,len(vals),size=len(vals))])) for _ in range(n_reps)]
    return [float(z) for z in np.quantile(result,[.025,.975])]


def pair_records(records: pd.DataFrame, stages: pd.DataFrame, pairs: pd.DataFrame,
                 windows: pd.DataFrame, min_per_class: int) -> tuple[pd.DataFrame,pd.DataFrame]:
    require(len(records) == N_RECORDS, 'Record-level result count not 153')
    require(records['record_id'].is_unique, 'Duplicate record-level metrics')
    require(pairs['subject_id'].nunique() == N_SUBJECTS, 'Pair audit subject count differs')
    require(len(windows) == N_RECORDS and windows['record_id'].is_unique, 'Window audit invalid')
    wcols = ['record_id','all_record_sleep_epochs','sleep_epochs_outside_main',
             'main_window_epochs','main_sleep_epochs','number_of_sleep_episodes']
    records = records.merge(windows[wcols],on='record_id',how='left',validate='one_to_one',indicator=True)
    require((records['_merge']=='both').all(), 'Window audit missing records')
    records=records.drop(columns='_merge')
    records['fraction_sleep_excluded'] = records['sleep_epochs_outside_main']/records['all_record_sleep_epochs']
    require(np.isfinite(records['fraction_sleep_excluded']).all(), 'Invalid excluded sleep fractions')
    require(((records['fraction_sleep_excluded']>=0)&(records['fraction_sleep_excluded']<=1)).all(), 'Invalid excluded fraction')
    pair_rows = []
    stage_rows = []
    for _, p in pairs.iterrows():
        if int(p['n_records']) != 2: continue
        sid = str(p['subject_id'])
        a = records.loc[(records.subject_id==sid)&(records.night_index==1)]
        b = records.loc[(records.subject_id==sid)&(records.night_index==2)]
        require(len(a)==1 and len(b)==1, f'Invalid pair: {sid}')
        a,b=a.iloc[0],b.iloc[0]
        require(a.record_id==p['record_1'] and b.record_id==p['record_2'],f'Pair audit mismatched for {sid}')
        row = {
            'subject_id':sid, 'record_1':str(a.record_id),'record_2':str(b.record_id),
            'same_scorer':parse_bool(p['same_scorer']),
            'date_concordant':parse_bool(p['date_concordant']),
            'scorer_1':str(p['scorer_1']),'scorer_2':str(p['scorer_2']),
            'n_epochs_1':int(a.n_epochs), 'n_epochs_2':int(b.n_epochs),
            'sleep_excluded_fraction_1':float(a.fraction_sleep_excluded),
            'sleep_excluded_fraction_2':float(b.fraction_sleep_excluded),
            'delta_sleep_excluded_fraction_nominal':float(b.fraction_sleep_excluded-a.fraction_sleep_excluded),
            'n_reference_stages_shared_ge_min':0,
        }
        for stage in STAGES:
            row[f'fraction_{stage}_1']=float(a[f'fraction_{stage}'])
            row[f'fraction_{stage}_2']=float(b[f'fraction_{stage}'])
            row[f'delta_fraction_{stage}']=float(b[f'fraction_{stage}']-a[f'fraction_{stage}'])
        for metric, col in [('mean','lr_mean_log_loss'),('median','lr_median_log_loss')]:
            row[f'{metric}_1']=float(a[col]);row[f'{metric}_2']=float(b[col])
            row[f'delta_{metric}_nominal']=float(b[col]-a[col]);row[f'abs_delta_{metric}']=abs(row[f'delta_{metric}_nominal'])
            row[f'pair_average_{metric}']=float((a[col]+b[col])/2)
        # Class-balanced on the identical true-stage support across the two
        # records, using only stage means for classes with >= min epochs both.
        supported=[]
        for stage in STAGES:
            n1=int(a[f'n_{stage}']); n2=int(b[f'n_{stage}'])
            c1=stages.loc[(stages.record_id==a.record_id)&(stages.stage==stage)]
            c2=stages.loc[(stages.record_id==b.record_id)&(stages.stage==stage)]
            require(len(c1)==len(c2)==1, f'Missing per-stage record {sid}/{stage}')
            loss1 = float(c1.iloc[0].stage_mean_loss)
            loss2 = float(c2.iloc[0].stage_mean_loss)
            eligible = n1>=min_per_class and n2>=min_per_class
            stage_rows.append({'subject_id':sid, 'record_1':a.record_id,'record_2':b.record_id,
                               'stage':stage,'n_epochs_1':n1,'n_epochs_2':n2,
                               'meets_min_in_both':bool(eligible),
                               'lr_stage_mean_loss_1':loss1 if n1 else np.nan,
                               'lr_stage_mean_loss_2':loss2 if n2 else np.nan,
                               'delta_stage_mean_loss_nominal':loss2-loss1 if eligible else np.nan})
            if eligible:
                supported.append((loss1,loss2))
        row['n_reference_stages_shared_ge_min']=len(supported)
        row['shared_stage_set']=','.join(s['stage'] for s in stage_rows[-len(STAGES):] if s['meets_min_in_both'])
        if supported:
            x=np.asarray(supported)
            row['balanced_1']=float(x[:,0].mean());row['balanced_2']=float(x[:,1].mean())
            row['delta_balanced_nominal']=float(x[:,1].mean()-x[:,0].mean())
            row['abs_delta_balanced']=abs(row['delta_balanced_nominal'])
            row['pair_average_balanced']=float(x.mean())
        else:
            for key in ('balanced_1','balanced_2','delta_balanced_nominal','abs_delta_balanced','pair_average_balanced'):
                row[key]=np.nan
        pair_rows.append(row)
    all_pairs=pd.DataFrame(pair_rows)
    stage_pairs=pd.DataFrame(stage_rows)
    require(len(all_pairs)==N_PAIRS and all_pairs.subject_id.is_unique,'Expected 75 distinct subject pairs')
    require(sum(all_pairs.same_scorer)==26 and sum(~all_pairs.same_scorer)==49, 'Scorer pair counts differ')
    require(sum(all_pairs.date_concordant)==67 and sum(~all_pairs.date_concordant)==8, 'Date concordance counts differ')
    return all_pairs, stage_pairs, records


def make_summary(pairs: pd.DataFrame, cfg: dict) -> dict:
    n_rep = int(cfg['validation']['bootstrap_subject_replicates'])
    seed=int(cfg['validation']['bootstrap_random_state'])
    outcome_labels={'mean':'lr_mean_epoch_log_loss', 'median':'lr_median_epoch_log_loss',
                    'balanced':'lr_shared_stage_balanced_mean_log_loss'}
    results={}
    for k,label in outcome_labels.items():
        valid=pairs.dropna(subset=[f'delta_{k}_nominal'])
        require(len(valid)==N_PAIRS if k!='balanced' else len(valid)>0,
                f'Unavailable paired metric {k}')
        d=valid[f'delta_{k}_nominal'].to_numpy(dtype=float)
        u=valid.loc[valid.date_concordant]
        signed=u[f'delta_{k}_nominal'].to_numpy(dtype=float)
        result={
            'outcome':label,
            'all_nominal_pairs_sign_invariant':pair_stats(d),
            'signed_date_concordant_only':{
                'n':len(signed),
                'mean':float(signed.mean()) if len(signed) else None,
                'median':float(np.median(signed)) if len(signed) else None,
                'mean_bootstrap_ci_conditional':bootstrap_interval(signed,np.mean,n_rep,seed),
            },
            'mean_abs_delta_bootstrap_ci_conditional':bootstrap_interval(np.abs(d),np.mean,n_rep,seed+1),
            'median_abs_delta_bootstrap_ci_conditional':bootstrap_interval(np.abs(d),np.median,n_rep,seed+2),
            'sd_between_subject_pair_means':float(valid[f'pair_average_{k}'].std(ddof=1)),
            'spearman_between_two_records':None,
            'strata':{}
        }
        rho,pvalue=spearmanr(valid[f'{k}_1'],valid[f'{k}_2'])
        result['spearman_between_two_records']=float(rho) if np.isfinite(rho) else None
        # No p-values: descriptive pre-specification. Explicitly drop test output.
        for name,subset in [('same_scorer',valid.loc[valid.same_scorer]),
                            ('different_scorer',valid.loc[~valid.same_scorer]),
                            ('concordant_dates',valid.loc[valid.date_concordant]),
                            ('discordant_dates',valid.loc[~valid.date_concordant])]:
            result['strata'][name]=pair_stats(subset[f'delta_{k}_nominal'].to_numpy(dtype=float))
        results[k]=result
    return {'outcomes':results,
            'note':'All estimates are descriptive and conditional on the saved LOSO OOF predictions. No causal night, age, scorer or clinical reliability claims.',
            'bootstrap_unit':'subject_pair','bootstrap_replicates':n_rep,'bootstrap_seed':seed,
            'signed_effect_scope':'67 date-concordant pairs only; nominal file index is NOT verified chronology for all subjects',
            'protocol_formula_warning':'sqrt(mean(delta^2)/2) includes systematic session bias; separate centered SD(delta)/sqrt(2) supplied.',
            }


def write_plots(pairs: pd.DataFrame, dest: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest.mkdir(parents=True,exist_ok=True)
    a=pairs
    valid=a.loc[a.date_concordant]
    center=valid['delta_mean_nominal'].mean()
    sd=valid['delta_mean_nominal'].std(ddof=1)
    fig,ax=plt.subplots(figsize=(8,5.2),dpi=160)
    for match,color,marker,name in [(True,'tab:blue','o','Same scorer'),
                                     (False,'tab:orange','s','Different scorer')]:
        b=a.loc[a.same_scorer==match]
        ax.scatter(b['pair_average_mean'],b['delta_mean_nominal'],
                   c=color,s=32,marker=marker,label=f'{name} (n={len(b)})', alpha=.72)
    dd=a.loc[~a.date_concordant]
    ax.scatter(dd['pair_average_mean'],dd['delta_mean_nominal'],
               facecolors='none',edgecolors='black',s=125,linewidths=1.4,
               label=f'Header dates discordant (n={len(dd)})')
    ax.axhline(0,color='gray',lw=.9)
    ax.axhline(center,color='black',ls='--',lw=1,label='Mean nominal difference, concordant')
    for bound in [center-1.96*sd,center+1.96*sd]:
        ax.axhline(bound,color='gray',ls=':',lw=1)
    ax.set(title='Bland–Altman: out-of-subject LR loss, two nominal records',
           xlabel='Mean loss across records',ylabel='Record 2 − record 1 loss')
    ax.legend(fontsize=7,loc='best');fig.tight_layout()
    fig.savefig(dest/'bland_altman_lr_mean_loss.png');plt.close(fig)
    fig,ax=plt.subplots(figsize=(7.5,4.5),dpi=160)
    ax.hist(a['abs_delta_mean'],bins=18,edgecolor='white',alpha=.85)
    ax.set(title='Absolute between-record difference in LR mean log-loss (75 pairs)',
           xlabel='|record 2 − record 1| log-loss',ylabel='Number of subjects')
    fig.tight_layout();fig.savefig(dest/'abs_delta_lr_mean_loss.png');plt.close(fig)


def run(args) -> None:
    for p in [args.oof,args.loso_summary,args.pair_audit,args.window_audit,args.config]:
        require(p.is_file(),f'Missing input: {p}')
    require(args.out.parent.is_dir(),f'Output parent does not exist: {args.out.parent}')
    require(not args.out.exists(),f'Output directory already exists; no overwrite: {args.out}')
    cfg=yaml.safe_load(args.config.read_text())
    require(cfg['study']['protocol']=='v1.2','Config protocol differs')
    nr=cfg['night_reliability']
    require(nr['outcome_central']=='night_mean_log_loss','Central outcome differs')
    require(nr['pair_unit']=='subject_id' and nr['night_index_source']=='filename_digit',
            'Pair analysis policy differs')
    require(nr['mixed_model']=='none', 'Mixed model is not part of protocol')
    require(nr['signed_delta_reported_on']=='date_concordant_pairs_only',
            'Signed delta scope differs')
    epsilon=float(cfg['validation']['logloss_clipping_epsilon'])
    require(epsilon==1e-15,'Clipping policy differs')
    min_per_class=int(nr['class_balanced_min_epochs_per_class_per_night'])
    require(min_per_class==10,'Class-balanced minimum changed')
    original=json.loads(args.loso_summary.read_text())
    require(original['status']=='PASS' and original['n_epochs']==EPS_EXPECTED and
            original['n_subjects']==N_SUBJECTS and original['n_recordings']==N_RECORDS,
            'Expected validated LOSO summary not found')
    require(original['artifact_sha256']['oof_predictions_csv']==sha256(args.oof),
            'OOF predictions hash does not match published LOSO summary')
    require(original['input_hashes']['config_sha256']==sha256(args.config),
            'Current config differs from the frozen LOSO config')
    oof=pd.read_csv(args.oof, dtype={'subject_id':str,'record_id':str},keep_default_na=False)
    require(len(oof)==EPS_EXPECTED and oof['row_index'].is_unique, 'OOF coverage invalid')
    require(np.array_equal(np.sort(oof['row_index'].to_numpy()),np.arange(EPS_EXPECTED)),
            'OOF indices not exactly 0..N-1')
    require(oof.subject_id.nunique()==N_SUBJECTS and oof.record_id.nunique()==N_RECORDS,
            'OOF scope differs')
    require(oof['night_index'].isin([1,2]).all(),'Invalid night_index')
    require(oof['stage'].isin(STAGES).all(),'Unknown stage in OOF')
    require(not oof.duplicated(['subject_id','record_id','epoch_index']).any(),
            'Duplicate epoch IDs')
    oof['lr_epoch_log_loss']=extract_epoch_log_loss(oof,epsilon)
    records,stages=aggregate_record(oof)
    original_pair=pd.read_csv(args.pair_audit,dtype={'subject_id':str,'record_1':str,'record_2':str},keep_default_na=False)
    win=pd.read_csv(args.window_audit,dtype={'subject_id':str,'record_id':str},keep_default_na=False)
    pairs,stage_pairs,records=pair_records(records,stages,original_pair,win,min_per_class)
    result=make_summary(pairs,cfg)
    result.update({'status':'PASS','protocol':'v1.2','models_retrained':False,
                   'oof_epochs':len(oof),'paired_subjects':len(pairs),
                   'same_scorer_pairs':int(pairs.same_scorer.sum()),
                   'different_scorer_pairs':int((~pairs.same_scorer).sum()),
                   'date_concordant_pairs':int(pairs.date_concordant.sum()),
                   'date_discordant_pairs':int((~pairs.date_concordant).sum()),
                   'input_sha256':{
                       'oof_predictions_csv':sha256(args.oof),
                       'loso_summary_json':sha256(args.loso_summary),
                       'pair_audit_csv':sha256(args.pair_audit),
                       'window_audit_csv':sha256(args.window_audit),
                       'config_yaml':sha256(args.config),
                       'analysis_script':sha256(Path(__file__)),
                   }})
    args.out.mkdir(parents=False)
    records.to_csv(args.out/'record_metrics.csv',index=False,float_format='%.17g')
    pairs.to_csv(args.out/'paired_metrics.csv',index=False,float_format='%.17g')
    stage_pairs.to_csv(args.out/'paired_stage_metrics.csv',index=False,float_format='%.17g')
    write_plots(pairs,args.out/'figures')
    result['output_sha256']={name:sha256(args.out/name) for name in
        ('record_metrics.csv','paired_metrics.csv','paired_stage_metrics.csv')}
    (args.out/'night_pair_summary.json').write_text(json.dumps(json_compatible(result),indent=2,allow_nan=False)+'\n')
    m=result['outcomes']['mean']['all_nominal_pairs_sign_invariant']
    s=result['outcomes']['mean']['signed_date_concordant_only']
    print('PASS — 75 paired subjects analyzed from validated OOF predictions; no refit')
    print('Scorers: same=26 different=49 | Header dates: concordant=67 discordant=8')
    print(f"Median |Δ| LR mean log-loss: {m['median_abs_delta']:.5f}")
    print(f"RMS-based within-pair scale (v1.2): {m['sw_rms_protocol_v1_2']:.5f}")
    print(f"Centered within-pair SD (descriptive): {m['sw_centered_bias_adjusted']:.5f}")
    print(f"Mean signed Δ among 67 concordant pairs: {s['mean']:.5f}")
    print('Outputs:',args.out)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oof',required=True,type=Path)
    parser.add_argument('--loso-summary',required=True,type=Path)
    parser.add_argument('--pair-audit',required=True,type=Path)
    parser.add_argument('--window-audit',required=True,type=Path)
    parser.add_argument('--config',required=True,type=Path)
    parser.add_argument('--out',required=True,type=Path)
    args=parser.parse_args()
    try:run(args)
    except Exception as e:
        print(f'FAILED — {type(e).__name__}: {e}',file=__import__('sys').stderr)
        raise SystemExit(1)

if __name__=='__main__':main()
