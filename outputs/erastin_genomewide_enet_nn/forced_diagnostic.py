#!/usr/bin/env python3
"""Best nonempty EN+NN diagnostic, registered before new held-out score review."""
import concurrent.futures as futures
import json
import multiprocessing as mp
import time

import joblib
import numpy as np
import pandas as pd

import common as c
from analysis_helpers import evaluate
from model_helpers import fitted_neural
from run_comparison import check_solver

DEST=c.OUT/'forced_additions'


def prepare():
    p=dict(status='registered_before_inspection_of_new_nn_heldout_scores',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        rationale='Early training-fold choices favored zero additions. A separately registered diagnostic evaluates the best nonempty panel while preserving the original primary zero-fallback analysis.',
        information_available='Parent protocol, preflight, original baseline results, completion counts and selected-gene counts from training-validation choices. New NN outer prediction errors have not been reviewed.',
        selection='Use the same saved inner losses. Exclude only the core-only candidate family; argmin mean five-fold RMSE among the unchanged 20 EN settings and original NN grid. No new inner fits, no top-K cap, no new tuning parameters or test-based selection.',
        outer_fit='Mandatory core64 plus all outer-training EN-selected genes at that setting. Same three-member NN, seeds, scaling and solver. Reuse parent outer fit if identical; otherwise refit on outer training only.',
        parent_protocol_sha256=c.sha(c.OUT/'protocol.json'),script_sha256=c.sha(c.OUT/'forced_diagnostic.py'),
        primary_comparisons=[['enet_forced_nn','nn64']],secondary_comparisons=[['enet_forced_nn','enet_selected_nn'],['enet_forced_nn','enet_selected_ridge']],
        uncertainty=json.load(open(c.OUT/'protocol.json'))['uncertainty'],
        scope='Secondary exploratory diagnostic, not a replacement for the frozen primary procedure. Every original seed/cohort retained.')
    if (DEST/'protocol_initial.json').exists():
        p.update(supersedes_initial_protocol_sha256=c.sha(DEST/'protocol_initial.json'),
            initial_source_sha256=c.sha(DEST/'initial_diagnostic_source.py'),
            bookkeeping_amendment='Correct diagnostic bootstrap output location before any diagnostic fits or new held-out score inspection; original registration/source archived. Fitting, selection and comparisons unchanged.')
    path=DEST/'protocol.json'
    if path.exists():
        old=json.load(open(path));p['created_utc']=old['created_utc'];assert old==p
    else:c.dump(path,p)
    return c.sha(path)


def fit(assay,r,ph):
    (DEST/assay/'folds').mkdir(parents=True,exist_ok=True)
    root=DEST/assay/'folds'/f'forced_nn__r{r+1:02d}'
    if root.with_suffix('.json').exists():
        m=json.load(open(root.with_suffix('.json')));assert m['protocol_sha256']==ph
        for ext in ['npz','joblib']:assert c.sha(root.with_suffix('.'+ext))==m[ext+'_sha256']
        return m
    s=c.STATE[assay];tr,te=np.flatnonzero(~s['test'][r]),np.flatnonzero(s['test'][r])
    parent=c.OUT/assay/'folds'/f'enet_selected_nn__r{r+1:02d}'
    a=np.load(parent.with_suffix('.npz'));m=json.load(open(parent.with_suffix('.json')))
    n=len(s['nn_grid']);chosen=n+int(np.argmin(a['inner_fold_rmse'][n:].mean(axis=1)));params=c.grid_for(assay)[chosen]
    sel=params['selector'];cache=c.cache(assay,r)
    extra=c.INDEX['candidate_indices'][cache['outer_selection_masks'][sel]];assert len(extra)>0
    indices=np.r_[c.INDEX['core_indices'],extra]
    reused=chosen==m['selected_candidate']
    if reused:
        model=joblib.load(parent.with_suffix('.joblib'));pred=a['test_prediction'];members=a['member_test_predictions'];diag=dict(reused_parent=True)
    else:
        pred,model,diag,members=fitted_neural(params,c.features(assay,indices),s['y'],tr,te,s['nn_seeds'][r],np.ones(len(tr)),retain=True);check_solver(diag)
        model=dict(model,feature_indices=indices,gene_columns=c.INDEX['gene_columns'][indices])
    joblib.dump(model,root.with_suffix('.joblib'),compress=3)
    np.savez_compressed(root.with_suffix('.npz'),train_rows=tr,test_rows=te,test_prediction=pred,member_test_predictions=members,feature_indices=indices)
    result=dict(assay=assay,repeat=r+1,protocol_sha256=ph,parent_npz_sha256=c.sha(parent.with_suffix('.npz')),
        selected_candidate=chosen,selected_parameters=params,added_genes=len(extra),reused_parent=reused,
        fit_diagnostics=diag,npz_sha256=c.sha(root.with_suffix('.npz')),joblib_sha256=c.sha(root.with_suffix('.joblib')))
    c.dump(root.with_suffix('.json'),result);return result


def main():
    ph=prepare();c.initialize();c.verify()
    print('FORCED_DIAGNOSTIC_REGISTERED',ph,flush=True)
    while not (c.OUT/'completion.json').exists():time.sleep(5)
    with futures.ProcessPoolExecutor(max_workers=8,mp_context=mp.get_context('spawn'),initializer=c.initialize) as pool:
        jobs=[pool.submit(fit,a,r,ph) for a in c.ASSAYS for r in range(20)]
        for j,f in enumerate(futures.as_completed(jobs),1):f.result();print('FORCED_FIT_COMPLETE',j,flush=True)
    # Parent prediction arrays are created after its numerical audits complete.
    while not (c.OUT/'validation.json').exists():time.sleep(5)
    protocol=json.load(open(DEST/'protocol.json'));allframes=[];params=[];objective_error=prediction_error=0.
    for assay,s in c.STATE.items():
        base=np.load(c.OUT/assay/'predictions.npz');preds={k:base[k].copy() for k in ['nn64','enet_selected_nn','enet_selected_ridge']}
        preds['enet_forced_nn']=np.full(s['test'].shape,np.nan)
        for r,mask in enumerate(s['test']):
            root=DEST/assay/'folds'/f'forced_nn__r{r+1:02d}'
            a=np.load(root.with_suffix('.npz'));m=json.load(open(root.with_suffix('.json')));model=joblib.load(root.with_suffix('.joblib'))
            parent=np.load(c.OUT/assay/'folds'/f'enet_selected_nn__r{r+1:02d}.npz');n=len(s['nn_grid'])
            assert m['selected_candidate']==n+int(np.argmin(parent['inner_fold_rmse'][n:].mean(axis=1)))
            assert c.sha(c.OUT/assay/'folds'/f'enet_selected_nn__r{r+1:02d}.npz')==m['parent_npz_sha256']
            tr,te=np.flatnonzero(~mask),np.flatnonzero(mask);sel=m['selected_parameters']['selector']
            indices=np.r_[c.INDEX['core_indices'],c.INDEX['candidate_indices'][c.cache(assay,r)['outer_selection_masks'][sel]]]
            np.testing.assert_array_equal(a['feature_indices'],indices);np.testing.assert_array_equal(model['feature_indices'],indices)
            np.testing.assert_array_equal(a['train_rows'],tr);np.testing.assert_array_equal(a['test_rows'],te)
            x=c.features(assay,indices);sx,sy=model['x_scaler'],model['y_scaler'];members=[]
            from sklearn.preprocessing import StandardScaler
            np.testing.assert_allclose(sx.mean_,StandardScaler().fit(x[tr]).mean_,atol=1e-12,rtol=0)
            target=sy.transform(s['y'][tr,None]).ravel()
            for j,v in enumerate(model['members']):
                assert v.random_state==int(s['nn_seeds'][r,j]);np.testing.assert_array_equal(v.training_sample_weight_,np.ones(len(tr)))
                members.append(sy.inverse_transform(v.predict(sx.transform(x[te]))[:,None]).ravel())
                obj=.5*np.mean((v.predict(sx.transform(x[tr]))-target)**2)+.5*v.alpha*sum(np.sum(t*t) for t in v.coefs_)/len(tr)
                objective_error=max(objective_error,abs(obj-v.loss_));assert abs(obj-v.loss_)<1e-10
            pred=np.mean(members,axis=0);np.testing.assert_allclose(members,a['member_test_predictions'],atol=1e-12,rtol=0)
            error=float(np.max(np.abs(pred-a['test_prediction'])));prediction_error=max(prediction_error,error);assert error<1e-12
            preds['enet_forced_nn'][r,te]=pred
            params.append(dict(assay=assay,repeat=r+1,added_genes=m['added_genes'],reused_parent=m['reused_parent'],**m['selected_parameters']))
        np.savez_compressed(DEST/assay/'predictions.npz',sample_ids=s['ids'],group_ids=s['groups'],lineage=s['lineage'],y=s['y'],outer_test=s['test'],**preds)
        original_out=c.OUT
        try:
            c.OUT=DEST
            allframes.append(evaluate(assay,preds,protocol))
        finally:
            c.OUT=original_out
    names=['summary.tsv','paired_comparisons.tsv','lineage_paired_effects.tsv','lineage_win_counts.tsv','per_repeat_pooled_metrics.tsv','per_lineage_metrics.tsv','per_repeat_macro_metrics.tsv']
    for j,n in enumerate(names):pd.concat([d[j] for d in allframes],ignore_index=True).to_csv(DEST/n,sep='\t',index=False,float_format='%.12g')
    pd.DataFrame(params).to_csv(DEST/'selected_parameters.tsv',sep='\t',index=False)
    assert c.sha(c.OUT/'protocol.json')==protocol['parent_protocol_sha256'] and c.sha(c.OUT/'forced_diagnostic.py')==protocol['script_sha256']
    c.dump(DEST/'validation.json',dict(passed=True,outer_ensembles=40,parent_tuning_choices_recomputed=True,prediction_max_error=prediction_error,outer_objectives_checked=120,training_objective_max_error=objective_error))
    summary=pd.read_csv(DEST/'summary.tsv',sep='\t');paired=pd.read_csv(DEST/'paired_comparisons.tsv',sep='\t')
    lines=['# Forced-addition neural-network diagnostic','',
        'This secondary diagnostic was registered after early training-validation choices favored the core-only model, before inspection of the new NN held-out scores. It uses the best nonempty setting from the unchanged inner NN grid. All original 64 genes remain; no new inner fitting, tuning parameters or test-driven gene choices are added. The primary zero-fallback comparison remains unchanged.','',
        '| Assay | Core64 NN RMSE | Forced additions NN RMSE | RMSE reduction | Conditional 95% interval | Favorable repeats /20 |','|---|---:|---:|---:|---:|---:|']
    for assay in c.ASSAYS:
        new=summary[(summary.assay==assay)&(summary.model=='enet_forced_nn')].iloc[0];base=summary[(summary.assay==assay)&(summary.model=='nn64')].iloc[0]
        d=paired[(paired.assay==assay)&(paired.reference_model=='nn64')&(paired.metric=='pooled_rmse')].iloc[0]
        lines.append(f'| {assay} | {base.pooled_rmse_mean:.6f} | {new.pooled_rmse_mean:.6f} | {d.improvement:+.6f} | [{d.conditional_ci_low:+.6f}, {d.conditional_ci_high:+.6f}] | {d.favorable_repeats} |')
    lines += ['','Positive RMSE reductions favor additions. Intervals are exploratory paired whole-lineage bootstrap intervals conditional on fits; overlapping seeds are not independent experiments. Models, predictions, selected parameters, lineage effects and validation are saved in this directory.','']
    (DEST/'REVIEW.md').write_text('\n'.join(lines));print('FORCED_ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':main()
