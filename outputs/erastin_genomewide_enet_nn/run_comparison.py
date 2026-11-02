#!/usr/bin/env python3
"""Nested EN setting / NN hyperparameter tuning, with fold checkpoints."""
import concurrent.futures as futures
import json
import multiprocessing as mp
import time

import joblib
import numpy as np

import common as c
from model_helpers import fitted_neural
from make_splits import inner_rows


def check_solver(diag):
    warnings=[v for d in diag['members'] for v in d['warnings']]
    assert not any(v['category']=='ConvergenceWarning' for v in warnings),warnings


def fit_fold(assay,r,fold,ph,cache):
    root=c.OUT/assay/'folds'/f'inner__r{r+1:02d}__f{fold}'
    if root.with_suffix('.json').exists():
        m=json.load(open(root.with_suffix('.json')));assert m['protocol_sha256']==ph
        assert c.sha(root.with_suffix('.npz'))==m['npz_sha256']
        return root
    start=time.monotonic();s=c.STATE[assay];tr,va=inner_rows(s['inner'][r],fold)
    grid=c.grid_for(assay);n=len(s['nn_grid'])
    old=np.load(c.BASE[assay]/'folds'/f'neural_network__p64__r{r+1:02d}.npz')
    val=np.empty((len(grid),len(va)));members=np.full((len(grid),3,len(va)),np.nan)
    val[:n]=old['inner_validation_predictions'][:,va]
    diagnostics=[];counts=[]
    for sel in range(20):
        indices=np.r_[c.INDEX['core_indices'],c.INDEX['candidate_indices'][cache['inner_selection_masks'][fold,sel]]]
        assert len(indices)==len(np.unique(indices))
        counts.append(len(indices));x=c.features(assay,indices)
        for j,params in enumerate(s['nn_grid']):
            candidate=(sel+1)*n+j
            pred,_,diag,pp=fitted_neural(params,x,s['y'],tr,va,s['nn_seeds'][r],np.ones(len(tr)),retain=False)
            check_solver(diag);val[candidate]=pred;members[candidate]=pp
            diagnostics.append(dict(candidate=candidate,selector=sel,params=params,**diag))
    assert np.isfinite(val).all() and np.isfinite(members[n:]).all()
    loss=np.sqrt(np.mean((val-s['y'][va][None,:])**2,axis=1))
    np.savez_compressed(root.with_suffix('.npz'),training_rows=tr,validation_rows=va,
        validation_predictions=val,member_validation_predictions=members,fold_rmse=loss,selected_gene_counts=np.array(counts))
    c.dump(root.with_suffix('.json'),dict(assay=assay,repeat=r+1,fold=fold,protocol_sha256=ph,
        train_n=len(tr),validation_n=len(va),seeds=s['nn_seeds'][r],diagnostics=diagnostics,seconds=time.monotonic()-start,
        npz_sha256=c.sha(root.with_suffix('.npz'))))
    return root


def fit_repeat(assay,r,ph):
    complete=c.OUT/assay/'folds'/f'repeat_{r+1:02d}_complete.json'
    if complete.exists():
        m=json.load(open(complete));assert m['protocol_sha256']==ph
        for row in m['files']:assert c.sha(c.OUT/row['path'])==row['sha256']
        return m
    start=time.monotonic();s=c.STATE[assay];grid=c.grid_for(assay);cache=c.cache(assay,r)
    tr,te=np.flatnonzero(~s['test'][r]),np.flatnonzero(s['test'][r])
    assert set(s['groups'][tr]).isdisjoint(s['groups'][te])
    val=np.full((len(grid),len(s['y'])),np.nan);loss=np.empty((len(grid),5));files=[]
    for fold in range(5):
        root=fit_fold(assay,r,fold,ph,cache);a=np.load(root.with_suffix('.npz'));va=a['validation_rows']
        val[:,va]=a['validation_predictions'];loss[:,fold]=a['fold_rmse']
        files.extend(root.with_suffix('.'+ext) for ext in ['npz','json'])
    valid=s['inner'][r]>=0
    assert np.isfinite(val[:,valid]).all() and np.isnan(val[:,~valid]).all()
    chosen=int(np.argmin(loss.mean(axis=1)));params=grid[chosen];sel=params['selector']
    extra=c.INDEX['candidate_indices'][cache['outer_selection_masks'][sel]] if sel>=0 else np.array([],dtype=int)
    indices=np.r_[c.INDEX['core_indices'],extra]
    if sel<0:
        source=c.BASE[assay]/'folds'/f'neural_network__p64__r{r+1:02d}'
        meta=json.load(open(source.with_suffix('.json')));original=np.load(source.with_suffix('.npz'))
        assert meta['selected_parameters']=={k:params[k] for k in ['width','alpha']}
        model=joblib.load(source.with_suffix('.joblib'));pred=original['test_prediction'];members=original['member_test_predictions']
        diag=dict(reused=True,source=str(source.relative_to(c.WS)))
    else:
        pred,model,diag,members=fitted_neural(params,c.features(assay,indices),s['y'],tr,te,s['nn_seeds'][r],np.ones(len(tr)),retain=True)
        check_solver(diag)
    model=dict(model,feature_indices=indices,gene_columns=c.INDEX['gene_columns'][indices])
    root=c.OUT/assay/'folds'/f'enet_selected_nn__r{r+1:02d}'
    joblib.dump(model,root.with_suffix('.joblib'),compress=3)
    np.savez_compressed(root.with_suffix('.npz'),train_rows=tr,test_rows=te,test_prediction=pred,
        member_test_predictions=members,feature_indices=indices,inner_validation_predictions=val,inner_fold_rmse=loss)
    c.dump(root.with_suffix('.json'),dict(assay=assay,repeat=r+1,protocol_sha256=ph,selected_candidate=chosen,
        selected_parameters=params,selected_genes=len(extra),gene_count=len(indices),seeds=s['nn_seeds'][r],
        train_n=len(tr),test_n=len(te),mean_inner_rmse=float(loss[chosen].mean()),fit_diagnostics=diag,
        reused_core64=bool(sel<0),seconds=time.monotonic()-start,npz_sha256=c.sha(root.with_suffix('.npz')),joblib_sha256=c.sha(root.with_suffix('.joblib'))))
    files.extend(root.with_suffix('.'+ext) for ext in ['npz','json','joblib'])
    done=dict(assay=assay,repeat=r+1,protocol_sha256=ph,seconds=time.monotonic()-start,selected_genes=len(extra),
        files=[dict(path=str(p.relative_to(c.OUT)),sha256=c.sha(p)) for p in files])
    c.dump(complete,done);return done


def main():
    begin=time.monotonic();ph=c.prepare();c.verify()
    pf=json.load(open(c.OUT/'preflight_validation.json'));assert pf['passed'] and pf['protocol_sha256']==ph
    with futures.ProcessPoolExecutor(max_workers=c.WORKERS,mp_context=mp.get_context('spawn'),initializer=c.initialize) as pool:
        jobs=[pool.submit(fit_repeat,a,r,ph) for r in range(20) for a in c.ASSAYS]
        for done,f in enumerate(futures.as_completed(jobs),1):
            m=f.result();elapsed=time.monotonic()-begin
            status=dict(phase='nested_enet_nn',completed=done,total=40,elapsed_seconds=elapsed,
                estimated_remaining_seconds=elapsed/done*(40-done),last_assay=m['assay'],last_repeat=m['repeat'],last_added_genes=m['selected_genes'])
            c.dump(c.OUT/'progress.json',status);print(json.dumps(status),flush=True)
    c.verify();c.dump(c.OUT/'completion.json',dict(complete=True,outer_ensembles=40,seconds=time.monotonic()-begin))
    print('NEURAL_FITS_COMPLETE',flush=True)


if __name__=='__main__':main()
