#!/usr/bin/env python3
"""Numerical-reference and leakage checks using the real prepared cohort."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from compare_models import nested_fit, test_ridge_math, bootstrap, dump


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--workspace',type=Path,required=True)
    args=ap.parse_args();ws=args.workspace
    out=Path(__file__).resolve().parent
    inp=ws/'outputs/erastin_40gene_comparison'
    data=np.load(inp/'comparison_inputs.npz',allow_pickle=False)
    X=data['X_panel_log2tpm1'];y=data['y_sensitivity'];ids=data['sample_ids'];groups=data['group_ids']
    index={v:i for i,v in enumerate(ids)}
    spec=json.load(open(inp/'splits.json'))
    folds=[]
    for f in spec['outer_folds']:
        folds.append({'label':str(f['fold']), 'train':np.array([index[v] for v in f['train_ModelIDs']]),
            'test':np.array([index[v] for v in f['test_ModelIDs']]),
            'inner':[{'train':np.array([index[v] for v in g['train_ModelIDs']]),
                      'test':np.array([index[v] for v in g['validation_ModelIDs']])} for g in f['inner_folds']]})
    alphas=np.array(json.load(open(inp/'comparison_config.json'))['ridge_alphas'])
    saved=pd.read_csv(out/'baseline_checkpoint/oof_predictions.tsv',sep='\t')
    fit=json.load(open(out/'baseline_checkpoint/fitted_outer_models.json'))
    errors=[]
    for f in folds:
        model=next(m['fitted_model'] for m in fit if m['model']=='panel_expression' and m['fold']==f['label'])
        tr,te=f['train'],f['test']
        scaler=StandardScaler().fit(np.asarray(X[tr],dtype=float))
        ref=Ridge(alpha=model['alpha'],solver='svd').fit(scaler.transform(np.asarray(X[tr],dtype=float)),y[tr])
        pred=ref.predict(scaler.transform(np.asarray(X[te],dtype=float)))
        errors.append(float(np.max(np.abs(pred-saved.panel_expression.to_numpy()[te]))))
    assert max(errors)<1e-9
    f=folds[0]
    p,fr,_,m=nested_fit('panel_expression',X,y,[f],alphas,len(y),groups)
    changed=y.copy();changed[f['test']]+=np.linspace(10,100,len(f['test']))
    p2,fr2,_,m2=nested_fit('panel_expression',X,changed,[f],alphas,len(y),groups)
    assert np.array_equal(p[f['test']],p2[f['test']]) and fr[0]['alpha']==fr2[0]['alpha']
    shifted=X.astype(float).copy();shifted[f['test']]+=1000
    _,fr3,_,m3=nested_fit('panel_expression',shifted,y,[f],alphas,len(y),groups)
    assert fr3[0]['alpha']==fr[0]['alpha']
    for key in ['mean','scale','coef']:
        assert np.array_equal(m[0]['fitted_model'][key],m3[0]['fitted_model'][key]),key
    # Rank-deficient p>n features exercise the SVD path used by embedding arms.
    wide=np.tile(X.astype(float),(1,17))[:,:643]
    wide[:,640:]=0
    wide_error=test_ridge_math(wide,y,f,1e-4)
    a=saved.panel_expression.to_numpy()
    boot=bootstrap(y,{'a':a,'identical_copy':a.copy()},groups,count=30)
    for metric in ['spearman','rmse','r2']:
        assert np.array_equal(boot['a'][metric],boot['identical_copy'][metric])
    report={'passed':True,'all_five_panel_folds_match_independent_sklearn_fits':True,
        'max_abs_prediction_errors_by_fold':errors,
        'outer_test_target_perturbation_leaves_predictions_and_alpha_unchanged':True,
        'outer_test_feature_shift_leaves_scaler_coefficients_and_alpha_unchanged':True,
        'rank_deficient_643_feature_ridge_max_abs_error':wide_error,
        'paired_group_bootstrap_identical_predictions_have_exactly_zero_difference':True}
    progress=json.load(open(ws/'work/bulkformer/extraction/progress.json'))
    completed=progress['completed']
    if completed>=8:
        emb=np.load(ws/'work/bulkformer/extraction/panel.npy',mmap_mode='r')[:completed].copy()
        assert np.isfinite(emb).all()
        small={'train':np.arange(completed-2),'test':np.arange(completed-2,completed)}
        report['partial_real_embedding_ridge_max_abs_error']=test_ridge_math(emb,y[:completed],small,1.0)
        report['partial_embedding_rows_checked']=completed
    dump(out/'runner_validation.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
