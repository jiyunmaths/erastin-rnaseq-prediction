#!/usr/bin/env python3
"""Nested Elastic Net feature discovery with the core64 forced into ridge."""
import concurrent.futures as futures
import json
import multiprocessing as mp
import time

import joblib
import numpy as np

import common as c
import models
from make_splits import inner_rows


def fit_repeat(assay,r,ph):
    complete=c.OUT/assay/'folds'/f'repeat_{r+1:02d}_complete.json'
    if complete.exists():
        m=json.load(open(complete));assert m['protocol_sha256']==ph
        for row in m['files']:assert c.sha(c.OUT/row['path'])==row['sha256']
        return m
    start=time.monotonic();s=c.STATE[assay]
    tr,te=np.flatnonzero(~s['test'][r]),np.flatnonzero(s['test'][r])
    core,candidates=c.INDEX['core_indices'],c.INDEX['candidate_indices']
    val=np.full((len(models.GRID),len(s['y'])),np.nan)
    loss=np.empty((len(models.GRID),5))
    selected_inner=np.zeros((5,len(models.SELECTORS),len(candidates)),dtype=bool)
    position={j:k for k,j in enumerate(candidates)}
    diagnostics=[]
    for fold in range(5):
        a,b=inner_rows(s['inner'][r],fold)
        assert set(a)|set(b)==set(tr) and set(s['groups'][a]).isdisjoint(s['groups'][b])
        x,z=c.features(assay,a),c.features(assay,b)
        paths,diag=models.select_paths(x,s['y'][a],s['lineage'][a],core,candidates)
        diagnostics.append(dict(fold=fold,**diag))
        pp,_=models.ridge_path(x[:,core],s['y'][a],z[:,core],models.ALPHAS)
        val[:17,b]=pp
        for j,fit in enumerate(paths):
            extra=fit['feature_indices'];indices=np.r_[core,extra]
            selected_inner[fold,j,[position[v] for v in extra]]=True
            if len(extra): pp,_=models.ridge_path(x[:,indices],s['y'][a],z[:,indices],models.ALPHAS)
            else: pp=val[:17,b].copy()
            val[17+j*17:17+(j+1)*17,b]=pp
        loss[:,fold]=np.sqrt(np.mean((val[:,b]-s['y'][b][None,:])**2,axis=1))
    valid=s['inner'][r]>=0
    assert np.isfinite(val[:,valid]).all() and np.isnan(val[:,~valid]).all()
    baseline=np.load(c.SOURCE/assay/'folds'/f'screening__r{r+1:02d}.npz')
    np.testing.assert_allclose(val[:17,valid],baseline['validation_predictions'][:17,valid],atol=1e-10,rtol=0)
    chosen=int(np.argmin(loss.mean(axis=1)));params=models.GRID[chosen]
    train,test=c.features(assay,tr),c.features(assay,te)
    paths,diag=models.select_paths(train,s['y'][tr],s['lineage'][tr],core,candidates)
    sel=params['selector']
    extra=paths[sel]['feature_indices'] if sel>=0 else np.array([],dtype=int)
    indices=np.r_[core,extra]
    pred,model=models.ridge_path(train[:,indices],s['y'][tr],test[:,indices],[params['ridge_alpha']],retain=True)
    model.update(feature_indices=indices,gene_columns=c.INDEX['gene_columns'][indices])
    root=c.OUT/assay/'folds'/f'enet_selected_ridge__r{r+1:02d}'
    joblib.dump(model,root.with_suffix('.joblib'),compress=3)
    outer_selection=np.zeros((len(models.SELECTORS),len(candidates)),dtype=bool)
    outer_coef=np.zeros(outer_selection.shape)
    for j,fit in enumerate(paths):
        pos=[position[v] for v in fit['feature_indices']]
        outer_selection[j,pos]=True;outer_coef[j,pos]=fit['coefficients']
    np.savez_compressed(root.with_suffix('.npz'),train_rows=tr,test_rows=te,test_prediction=pred[0],feature_indices=indices,
        inner_validation_predictions=val,inner_fold_rmse=loss,inner_selection_masks=selected_inner,
        outer_selection_masks=outer_selection,outer_selector_coefficients=outer_coef)
    m=dict(assay=assay,repeat=r+1,protocol_sha256=ph,selected_candidate=chosen,selected_parameters=params,
        selector_parameters=models.SELECTORS[sel] if sel>=0 else None,selected_genes=len(extra),gene_count=len(indices),
        train_n=len(tr),test_n=len(te),mean_inner_rmse=float(loss[chosen].mean()),
        inner_diagnostics=diagnostics,outer_diagnostics=diag,seconds=time.monotonic()-start,
        npz_sha256=c.sha(root.with_suffix('.npz')),joblib_sha256=c.sha(root.with_suffix('.joblib')))
    c.dump(root.with_suffix('.json'),m)
    files=[root.with_suffix('.'+ext) for ext in ['npz','json','joblib']]
    done=dict(assay=assay,repeat=r+1,protocol_sha256=ph,seconds=m['seconds'],selected_genes=len(extra),
        files=[dict(path=str(p.relative_to(c.OUT)),sha256=c.sha(p)) for p in files])
    c.dump(complete,done)
    return done


def main():
    begin=time.monotonic();ph=c.prepare();c.verify()
    pf=json.load(open(c.OUT/'preflight_validation.json'));assert pf['passed'] and pf['protocol_sha256']==ph
    with futures.ProcessPoolExecutor(max_workers=c.WORKERS,mp_context=mp.get_context('spawn'),initializer=c.initialize) as pool:
        jobs=[pool.submit(fit_repeat,a,r,ph) for r in range(20) for a in c.ASSAYS]
        for done,f in enumerate(futures.as_completed(jobs),1):
            m=f.result();elapsed=time.monotonic()-begin
            status=dict(phase='nested_elastic_net_selection',completed=done,total=40,elapsed_seconds=elapsed,
                estimated_remaining_seconds=elapsed/done*(40-done),last_assay=m['assay'],last_repeat=m['repeat'],last_added_genes=m['selected_genes'])
            c.dump(c.OUT/'progress.json',status);print(json.dumps(status),flush=True)
    c.verify();c.dump(c.OUT/'completion.json',dict(complete=True,outer_models=40,seconds=time.monotonic()-begin))
    print('NESTED_ELASTIC_NET_COMPLETE',flush=True)


if __name__=='__main__':main()
