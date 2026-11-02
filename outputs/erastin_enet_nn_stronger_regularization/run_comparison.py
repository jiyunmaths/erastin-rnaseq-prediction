#!/usr/bin/env python3
"""Reuse old NN candidates; fit stronger penalties with fold checkpoints."""
import concurrent.futures as futures
import json
import multiprocessing as mp
import time

import joblib
import numpy as np

import common as c
from model_helpers import fitted_neural
from make_splits import inner_rows


def check_solver(d):
    assert not any(w['category']=='ConvergenceWarning' for m in d['members'] for w in m['warnings']),d


def fit_fold(assay,r,fold,ph,cache):
    root=c.OUT/assay/'folds'/f'inner__r{r+1:02d}__f{fold}'
    if root.with_suffix('.json').exists():
        m=json.load(open(root.with_suffix('.json')));assert m['protocol_sha256']==ph and c.sha(root.with_suffix('.npz'))==m['npz_sha256'];return root
    start=time.monotonic();s=c.STATE[assay];tr,va=inner_rows(s['inner'][r],fold)
    grid=c.grid_for(assay);n,no=len(s['nn_grid']),len(s['old_nn_grid'])
    old=c.OLD/assay/'folds'/f'inner__r{r+1:02d}__f{fold}';a=np.load(old.with_suffix('.npz'))
    np.testing.assert_array_equal(a['training_rows'],tr);np.testing.assert_array_equal(a['validation_rows'],va)
    reused=c.old_columns(assay);new=np.setdiff1d(np.arange(len(grid)),reused)
    val=np.full((len(grid),len(va)),np.nan);members=np.full((len(grid),3,len(va)),np.nan)
    val[reused]=a['validation_predictions'];members[reused]=a['member_validation_predictions']
    diagnostics=[]
    for sel in range(-1,20):
        extra=c.INDEX['candidate_indices'][cache['inner_selection_masks'][fold,sel]] if sel>=0 else np.array([],dtype=int)
        indices=np.r_[c.INDEX['core_indices'],extra];x=c.features(assay,indices)
        for j in range(no,n):
            candidate=(sel+1)*n+j;params=s['nn_grid'][j]
            pp,_,d,mm=fitted_neural(params,x,s['y'],tr,va,s['nn_seeds'][r],np.ones(len(tr)),retain=False)
            check_solver(d);val[candidate]=pp;members[candidate]=mm
            diagnostics.append(dict(candidate=candidate,selector=sel,params=params,**d))
    assert np.isfinite(val).all() and np.isfinite(members[new]).all()
    loss=np.sqrt(np.mean((val-s['y'][va][None,:])**2,axis=1))
    np.savez_compressed(root.with_suffix('.npz'),training_rows=tr,validation_rows=va,validation_predictions=val,
        member_validation_predictions=members,fold_rmse=loss,reused_columns=reused,new_columns=new)
    c.dump(root.with_suffix('.json'),dict(assay=assay,repeat=r+1,fold=fold,protocol_sha256=ph,seeds=s['nn_seeds'][r],
        source_npz_sha256=c.sha(old.with_suffix('.npz')),train_n=len(tr),validation_n=len(va),diagnostics=diagnostics,
        new_ensembles=len(new),seconds=time.monotonic()-start,npz_sha256=c.sha(root.with_suffix('.npz'))))
    return root


def old_fit(assay,r,arm,params):
    if arm=='core_extended':root=c.BASE[assay]/'folds'/f'neural_network__p64__r{r+1:02d}'
    elif arm=='optional_extended':root=c.OLD/assay/'folds'/f'enet_selected_nn__r{r+1:02d}'
    else:root=c.OLD/'forced_additions'/assay/'folds'/f'forced_nn__r{r+1:02d}'
    m=json.load(open(root.with_suffix('.json')));a=np.load(root.with_suffix('.npz'))
    expected={k:params[k] for k in ['width','alpha']} if arm=='core_extended' else params
    assert m['selected_parameters']==expected,(arm,m['selected_parameters'],expected)
    return joblib.load(root.with_suffix('.joblib')),a['test_prediction'],a['member_test_predictions'],dict(reused_old=True,source=str(root.relative_to(c.WS)))


def fit_repeat(assay,r,ph):
    complete=c.OUT/assay/'folds'/f'repeat_{r+1:02d}_complete.json'
    if complete.exists():
        m=json.load(open(complete));assert m['protocol_sha256']==ph
        for row in m['files']:assert c.sha(c.OUT/row['path'])==row['sha256']
        return m
    start=time.monotonic();s=c.STATE[assay];grid=c.grid_for(assay);cache=c.cache(assay,r);n=len(s['nn_grid'])
    tr,te=np.flatnonzero(~s['test'][r]),np.flatnonzero(s['test'][r])
    val=np.full((len(grid),len(s['y'])),np.nan);loss=np.empty((len(grid),5));files=[]
    for fold in range(5):
        root=fit_fold(assay,r,fold,ph,cache);a=np.load(root.with_suffix('.npz'));val[:,a['validation_rows']]=a['validation_predictions'];loss[:,fold]=a['fold_rmse']
        files += [root.with_suffix('.'+e) for e in ['npz','json']]
    valid=s['inner'][r]>=0;assert np.isfinite(val[:,valid]).all() and np.isnan(val[:,~valid]).all()
    old=np.load(c.OLD/assay/'folds'/f'enet_selected_nn__r{r+1:02d}.npz')
    np.testing.assert_array_equal(val[c.old_columns(assay)],old['inner_validation_predictions'])
    choices=dict(core_extended=int(np.argmin(loss[:n].mean(axis=1))),optional_extended=int(np.argmin(loss.mean(axis=1))),
        forced_extended=n+int(np.argmin(loss[n:].mean(axis=1))))
    np.savez_compressed(c.OUT/assay/'folds'/f'tuning__r{r+1:02d}.npz',inner_validation_predictions=val,inner_fold_rmse=loss)
    files.append(c.OUT/assay/'folds'/f'tuning__r{r+1:02d}.npz')
    reused_current={};results=[]
    for arm in c.ARMS:
        chosen=choices[arm];params=grid[chosen];sel=params['selector'];key=(sel,params['width'],params['alpha'])
        extra=c.INDEX['candidate_indices'][cache['outer_selection_masks'][sel]] if sel>=0 else np.array([],dtype=int)
        indices=np.r_[c.INDEX['core_indices'],extra]
        if key in reused_current:
            model,pred,members,first=reused_current[key];diag=dict(reused_current=True,source_arm=first)
        elif {k:params[k] for k in ['width','alpha']} in s['old_nn_grid']:
            model,pred,members,diag=old_fit(assay,r,arm,params)
        else:
            pred,model,diag,members=fitted_neural(params,c.features(assay,indices),s['y'],tr,te,s['nn_seeds'][r],np.ones(len(tr)),retain=True);check_solver(diag)
        model=dict(model,feature_indices=indices,gene_columns=c.INDEX['gene_columns'][indices])
        reused_current.setdefault(key,(model,pred,members,arm))
        root=c.OUT/assay/'folds'/f'{arm}__r{r+1:02d}'
        joblib.dump(model,root.with_suffix('.joblib'),compress=3)
        np.savez_compressed(root.with_suffix('.npz'),train_rows=tr,test_rows=te,test_prediction=pred,member_test_predictions=members,feature_indices=indices)
        c.dump(root.with_suffix('.json'),dict(assay=assay,arm=arm,repeat=r+1,protocol_sha256=ph,seeds=s['nn_seeds'][r],
            selected_candidate=chosen,selected_parameters=params,added_genes=len(extra),total_genes=len(indices),mean_inner_rmse=float(loss[chosen].mean()),fit_diagnostics=diag,
            npz_sha256=c.sha(root.with_suffix('.npz')),joblib_sha256=c.sha(root.with_suffix('.joblib'))))
        files += [root.with_suffix('.'+e) for e in ['npz','json','joblib']]
        results.append(dict(arm=arm,params=params,added_genes=len(extra)))
    m=dict(assay=assay,repeat=r+1,protocol_sha256=ph,seconds=time.monotonic()-start,results=results,
        files=[dict(path=str(f.relative_to(c.OUT)),sha256=c.sha(f)) for f in files])
    c.dump(complete,m);return m


def main():
    begin=time.monotonic();ph=c.prepare();c.verify();pf=json.load(open(c.OUT/'preflight_validation.json'));assert pf['passed'] and pf['protocol_sha256']==ph
    with futures.ProcessPoolExecutor(max_workers=c.WORKERS,mp_context=mp.get_context('spawn'),initializer=c.initialize) as pool:
        jobs=[pool.submit(fit_repeat,a,r,ph) for r in range(20) for a in c.ASSAYS]
        for done,f in enumerate(futures.as_completed(jobs),1):
            m=f.result();elapsed=time.monotonic()-begin
            status=dict(phase='stronger_regularization',completed=done,total=40,elapsed_seconds=elapsed,
                estimated_remaining_seconds=elapsed/done*(40-done),last_assay=m['assay'],last_repeat=m['repeat'])
            c.dump(c.OUT/'progress.json',status);print(json.dumps(status),flush=True)
    c.verify();c.dump(c.OUT/'completion.json',dict(complete=True,logical_outer_ensembles=120,seconds=time.monotonic()-begin));print('ALL_FITS_COMPLETE',flush=True)


if __name__=='__main__':main()
