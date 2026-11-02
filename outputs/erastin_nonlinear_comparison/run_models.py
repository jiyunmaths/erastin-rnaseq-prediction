#!/usr/bin/env python3
"""Nested nonlinear-head comparison on the fixed corrected lineage holdouts."""
import concurrent.futures as futures
import hashlib
import importlib.util
import json
import multiprocessing as mp
import os
from pathlib import Path
import platform
import sys
import time
import warnings

import joblib
import numpy as np
import pandas as pd
import sklearn
from scipy.stats import spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
WS=OUT.parent.parent
REPS=['panel_expression','panel_contextual_embedding','all_gene_contextual_embedding']
FAMILIES=['ridge','random_forest','neural_network']
SEEDS=[20261104,20261105,20261106]
RF_GRID=[{'min_samples_leaf':leaf,'max_features':mf} for leaf in [5,15] for mf in ['sqrt',0.3]]
NN_GRID=[{'width':w,'alpha':a} for w in [16,32] for a in [10.0,100.0]]
STATE={}


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.bool_):return bool(x)
    return x


def dump(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)


def metrics(y,p):
    mse=float(np.mean((y-p)**2));den=float(np.sum((y-y.mean())**2))
    return {'n':len(y),'spearman':float(spearmanr(y,p).statistic) if np.ptp(y)>0 and np.ptp(p)>0 else np.nan,
        'rmse':np.sqrt(mse),'r2':1-len(y)*mse/den if den else np.nan}


def load_state():
    d=np.load(WS/'outputs/erastin_40gene_comparison/comparison_inputs.npz',allow_pickle=False)
    e=np.load(WS/'outputs/erastin_model_comparison/bulkformer_embeddings.npz',allow_pickle=False)
    s=np.load(WS/'outputs/erastin_lineage_corrected/shared_splits.npz',allow_pickle=False)
    assert np.array_equal(d['sample_ids'],e['sample_ids']) and np.array_equal(d['sample_ids'],s['sample_ids'])
    STATE.update(ids=d['sample_ids'],y=d['y_sensitivity'].astype(float),groups=d['group_ids'],lineage=d['lineage'],
        features={REPS[0]:d['X_panel_log2tpm1'].astype(float),REPS[1]:e['panel_embedding'].astype(float),REPS[2]:e['all_gene_embedding'].astype(float)},
        labels=list(s['holdout_labels']),inner=s['inner_fold_by_holdout'][0])


def fitted_prediction(family,params,x,y,tr,te,retain=False):
    begin=time.perf_counter();diagnostics=[]
    if family=='random_forest':
        model=RandomForestRegressor(n_estimators=128,min_samples_leaf=params['min_samples_leaf'],
            max_features=params['max_features'],bootstrap=True,n_jobs=1,random_state=SEEDS[0])
        model.fit(x[tr],y[tr]);p=model.predict(x[te])
        saved={'family':family,'model':model} if retain else None
        diagnostics=[{'seed':SEEDS[0],'warnings':[],'n_trees':128}]
        member_predictions=p[None,:]
    else:
        # Both scalers see only the current training subset. No random early-
        # stopping validation split is introduced; L-BFGS uses training loss.
        sx=StandardScaler().fit(x[tr]);sy=StandardScaler().fit(y[tr,None])
        a=sx.transform(x[tr]);b=sx.transform(x[te]);target=sy.transform(y[tr,None]).ravel()
        members=[];member_predictions=[]
        for seed in SEEDS:
            model=MLPRegressor(hidden_layer_sizes=(params['width'],),activation='tanh',solver='lbfgs',
                alpha=params['alpha'],max_iter=2000,max_fun=40000,tol=1e-3,early_stopping=False,random_state=seed)
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');model.fit(a,target)
            prediction=sy.inverse_transform(model.predict(b)[:,None]).ravel()
            assert np.isfinite(prediction).all()
            member_predictions.append(prediction)
            diagnostics.append({'seed':seed,'iterations':model.n_iter_,'training_loss':model.loss_,
                'warnings':[{'category':w.category.__name__,'message':str(w.message)} for w in caught]})
            if retain:members.append(model)
        member_predictions=np.array(member_predictions);p=member_predictions.mean(axis=0)
        saved={'family':family,'x_scaler':sx,'y_scaler':sy,'members':members} if retain else None
    assert np.isfinite(p).all()
    return p,saved,{'seconds':time.perf_counter()-begin,'members':diagnostics},member_predictions


def one_holdout(family,rep,index,protocol_hash):
    token=f'{family}__{rep}__{index:02d}'
    folder=OUT/'folds';meta_path=folder/(token+'.json');model_path=folder/(token+'.joblib');pred_path=folder/(token+'.npz')
    if meta_path.exists():
        old=json.loads(meta_path.read_text());assert old['protocol_sha256']==protocol_hash
        assert digest(model_path)==old['model_sha256'] and digest(pred_path)==old['predictions_sha256']
        return old
    x=STATE['features'][rep];y=STATE['y'];g=STATE['groups'];label=STATE['labels'][index]
    inner=STATE['inner'][index];tr=np.flatnonzero(inner>=0);te=np.flatnonzero(STATE['lineage']==label)
    assert set(g[tr]).isdisjoint(g[te]) and not np.any(STATE['lineage'][tr]==label)
    grid=RF_GRID if family=='random_forest' else NN_GRID
    val=np.full((len(grid),len(y)),np.nan);tuning=[];fit_diagnostics=[]
    for candidate,params in enumerate(grid):
        losses=[]
        for k in range(5):
            a=tr[inner[tr]!=k];b=tr[inner[tr]==k]
            assert set(g[a]).isdisjoint(g[b]) and set(a)|set(b)==set(tr)
            assert set(a).isdisjoint(te) and set(b).isdisjoint(te)
            p,_,diag,_=fitted_prediction(family,params,x,y,a,b)
            assert np.isnan(val[candidate,b]).all();val[candidate,b]=p
            losses.append(float(np.sqrt(np.mean((y[b]-p)**2))))
            fit_diagnostics.append({'candidate':candidate,'inner_fold':k,'train_n':len(a),'validation_n':len(b),**diag})
        assert np.isfinite(val[candidate,tr]).all() and np.isnan(val[candidate,te]).all()
        tuning.append({'candidate':candidate,'parameters':params,'mean_inner_rmse':float(np.mean(losses)),'fold_rmse':losses})
    chosen=int(np.argmin([r['mean_inner_rmse'] for r in tuning]))
    p,saved,diag,members=fitted_prediction(family,grid[chosen],x,y,tr,te,retain=True)
    fit_diagnostics.append({'candidate':chosen,'inner_fold':None,'stage':'outer_refit','train_n':len(tr),'test_n':len(te),**diag})
    saved.update(train_rows=tr,test_rows=te,parameters=grid[chosen],representation=rep,held_out_lineage=label,
        protocol_sha256=protocol_hash)
    tmp=model_path.with_suffix('.joblib.tmp');joblib.dump(saved,tmp,compress=3);tmp.replace(model_path)
    np.savez_compressed(pred_path,train_rows=tr,test_rows=te,inner_validation_predictions=val,
        test_prediction=p,member_test_predictions=members)
    meta={'arm':family+'__'+rep,'family':family,'representation':rep,'holdout_index':index,'lineage':label,
        'train_n':len(tr),'test_n':len(te),'selected_candidate':chosen,'selected_parameters':grid[chosen],
        'mean_inner_rmse':tuning[chosen]['mean_inner_rmse'],'metrics':metrics(y[te],p),'tuning':tuning,
        'fit_diagnostics':fit_diagnostics,'protocol_sha256':protocol_hash,
        'model_sha256':digest(model_path),'predictions_sha256':digest(pred_path)}
    dump(meta_path,meta)
    return meta


def main():
    started=time.time();load_state();(OUT/'folds').mkdir(exist_ok=True)
    sources=[]
    source_groups=[('erastin_40gene_comparison','artifact_manifest.json',['comparison_inputs.npz']),
        ('erastin_model_comparison','result_artifact_manifest.json',['bulkformer_embeddings.npz']),
        ('erastin_lineage_corrected','artifact_manifest.json',['shared_splits.npz','per_run_predictions.npz','per_lineage_metrics.tsv','pooled_summary.tsv','run_comparison.py']),
        ('erastin_repeated_cv','artifact_manifest.json',['repeat_cv.py'])]
    for directory,manifest,names in source_groups:
        root=WS/'outputs'/directory;known={r['path']:r['sha256'] for r in json.loads((root/manifest).read_text())['files']}
        for name in names:
            p=root/name;h=digest(p);assert h==known[name]
            sources.append({'path':str(p.relative_to(WS)),'sha256':h})
    protocol={'created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),'status':'frozen_before_any_heldout_scoring',
        'script_sha256':digest(Path(__file__)),'samples':745,'identity_groups':734,'heldout_lineages':24,
        'split_source':'corrected leave-lineage-out shared_splits.npz, original seed 20261031, five inner folds',
        'representations':REPS,'families':FAMILIES,'random_forest_grid':RF_GRID,'neural_network_grid':NN_GRID,
        'rf_fixed':{'n_estimators':128,'bootstrap':True,'random_state':SEEDS[0],'n_jobs':1},
        'nn_fixed':{'activation':'tanh','solver':'lbfgs','max_iter':2000,'max_fun':40000,'tol':1e-3,
            'early_stopping':False,'initialization_seeds':SEEDS,'prediction':'average of three independently initialized networks in inner and outer fits',
            'scaling':'StandardScaler for X and y fit only on the current training subset'},
        'tuning':'minimum equal-weight mean five-fold inner RMSE in original response units; deterministic first-candidate ties',
        'ridge_reference':'reuse saved original-seed corrected predictions and inner tuning losses; never compare new models to ridge averaged across other split seeds',
        'nested_family_representation_selector':'for each outer lineage choose the arm with smallest inner RMSE from all nine already tuned arms; tie order ridge/RF/NN then expression/panel/all',
        'pooled_metrics':['spearman','rmse','r2'],'macro_metrics':'equal-weight within-lineage mean among 19 lineages with n>=10',
        'uncertainty':'5000 paired whole-lineage resamples conditional on fixed predictions; 24 clusters pooled, 19 macro; no refitting',
        'bootstrap_seeds':{'pooled':2026110401,'macro':2026110402},
        'comparison_scope':'exploratory new-head comparison on one fixed inner split scheme; not a 20-split repeat and not external validation',
        'training_only_timing_pilot':'used only runtime and convergence on an inner-training subset; no validation or outer-test predictions inspected',
        'warnings_policy':'record all optimizer warnings and finite predictions; no performance-based reruns or grid expansion',
        'prior_ridge_results_already_seen':True,'workers':6,
        'environment':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__}}
    p=OUT/'protocol.json'
    if p.exists():
        old=json.loads(p.read_text());protocol['created_utc']=old['created_utc'];assert protocol==old
    else:dump(p,protocol)
    ph=digest(p);dump(OUT/'source_manifest.json',{'files':sources})
    dump(OUT/'progress.json',{'phase':'fitting','completed_outer_jobs':0,'total_outer_jobs':144})
    print(json.dumps({'event':'protocol_frozen','sha256':ph,'outer_jobs':144}),flush=True)
    done=[]
    with futures.ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn'),initializer=load_state) as pool:
        pending={pool.submit(one_holdout,family,rep,i,ph):(family,rep,i) for i in range(24)
            for family in ['random_forest','neural_network'] for rep in REPS}
        for f in futures.as_completed(pending):
            r=f.result();done.append(r)
            dump(OUT/'progress.json',{'phase':'fitting','completed_outer_jobs':len(done),'total_outer_jobs':144,
                'elapsed_seconds':time.time()-started,'latest_lineage':r['lineage'],'latest_arm':r['arm']})
            print(json.dumps({'event':'outer_job_complete','completed':len(done),'total':144,'lineage':r['lineage'],
                'arm':r['arm'],'elapsed_seconds':time.time()-started}),flush=True)
    assert digest(Path(__file__))==protocol['script_sha256'] and digest(p)==ph
    for source in sources:assert digest(WS/source['path'])==source['sha256']
    dump(OUT/'fit_completion.json',{'status':'complete','outer_jobs':len(done),'elapsed_seconds':time.time()-started,
        'source_and_script_hashes_unchanged':True,'protocol_sha256':ph})
    dump(OUT/'progress.json',{'phase':'fits_complete','completed_outer_jobs':144,'total_outer_jobs':144,'elapsed_seconds':time.time()-started})
    print(json.dumps({'event':'all_fits_complete','elapsed_seconds':time.time()-started}),flush=True)


if __name__=='__main__':main()
