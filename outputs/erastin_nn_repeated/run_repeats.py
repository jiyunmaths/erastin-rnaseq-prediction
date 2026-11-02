#!/usr/bin/env python3
"""Repeat the unchanged expression-panel neural pipeline on matched lineage splits."""
import concurrent.futures as futures
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import joblib
import numpy as np

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
WS=OUT.parent.parent


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


prior=module('prior_nonlinear',WS/'outputs/erastin_nonlinear_comparison/run_models.py')
STATE={}
MASTER=20261107


def load_state():
    d=np.load(WS/'outputs/erastin_40gene_comparison/comparison_inputs.npz',allow_pickle=False)
    s=np.load(WS/'outputs/erastin_lineage_corrected/shared_splits.npz',allow_pickle=False)
    old=json.loads((WS/'outputs/erastin_nonlinear_comparison/protocol.json').read_text())
    assert np.array_equal(d['sample_ids'],s['sample_ids'])
    triplets=np.random.SeedSequence(MASTER).generate_state(60).reshape(20,3)
    assert len(set(triplets.ravel()))==60 and set(triplets.ravel()).isdisjoint(old['nn_fixed']['initialization_seeds'])
    STATE.update(ids=d['sample_ids'],x=d['X_panel_log2tpm1'].astype(float),y=d['y_sensitivity'].astype(float),
        groups=d['group_ids'],lineage=d['lineage'],labels=list(s['holdout_labels']),inner=s['inner_fold_by_holdout'],
        split_seeds=s['seeds'],nn_seeds=np.vstack([old['nn_fixed']['initialization_seeds'],triplets]).astype(np.uint32),old_protocol=old)
    assert STATE['x'].shape==(745,40) and np.isfinite(STATE['x']).all()
    assert len(s['seeds'])==21 and int(s['seeds'][0])==20261031
    assert prior.NN_GRID==old['neural_network_grid']


def fit_holdout(repeat,index,ph):
    folder=OUT/('replay' if repeat==0 else 'folds');folder.mkdir(exist_ok=True)
    token=f'repeat_{repeat:02d}__lineage_{index:02d}'
    mpth=folder/(token+'.json');jpth=folder/(token+'.joblib');npth=folder/(token+'.npz')
    if mpth.exists():
        old=json.loads(mpth.read_text());assert old['protocol_sha256']==ph
        assert prior.digest(jpth)==old['model_sha256'] and prior.digest(npth)==old['predictions_sha256']
        return old
    x=STATE['x'];y=STATE['y'];g=STATE['groups'];label=STATE['labels'][index]
    inner=STATE['inner'][repeat,index];tr=np.flatnonzero(inner>=0);te=np.flatnonzero(STATE['lineage']==label)
    assert set(g[tr]).isdisjoint(g[te]) and label not in set(STATE['lineage'][tr])
    seeds=[int(s) for s in STATE['nn_seeds'][repeat]]
    # Each worker executes one job at a time; this process-local assignment
    # reuses the exact prior fit function with the registered initialization trio.
    prior.SEEDS=seeds
    val=np.full((4,len(y)),np.nan);tuning=[];diagnostics=[]
    for c,params in enumerate(prior.NN_GRID):
        losses=[]
        for k in range(5):
            a=tr[inner[tr]!=k];b=tr[inner[tr]==k]
            assert set(a)|set(b)==set(tr) and set(g[a]).isdisjoint(g[b])
            assert set(a).isdisjoint(te) and set(b).isdisjoint(te)
            p,_,diag,_=prior.fitted_prediction('neural_network',params,x,y,a,b)
            assert np.isnan(val[c,b]).all();val[c,b]=p
            losses.append(float(np.sqrt(np.mean((y[b]-p)**2))))
            diagnostics.append({'candidate':c,'inner_fold':k,'train_n':len(a),'validation_n':len(b),**diag})
        assert np.isfinite(val[c,tr]).all() and np.isnan(val[c,te]).all()
        tuning.append({'candidate':c,'parameters':params,'fold_rmse':losses,'mean_inner_rmse':float(np.mean(losses))})
    chosen=int(np.argmin([r['mean_inner_rmse'] for r in tuning]))
    p,saved,diag,members=prior.fitted_prediction('neural_network',prior.NN_GRID[chosen],x,y,tr,te,retain=True)
    diagnostics.append({'candidate':chosen,'stage':'outer_refit','train_n':len(tr),'test_n':len(te),**diag})
    saved.update(train_rows=tr,test_rows=te,parameters=prior.NN_GRID[chosen],representation='panel_expression',
        held_out_lineage=label,repeat=repeat,initialization_seeds=seeds,protocol_sha256=ph)
    tmp=jpth.with_suffix('.joblib.tmp');joblib.dump(saved,tmp,compress=3);tmp.replace(jpth)
    np.savez_compressed(npth,train_rows=tr,test_rows=te,inner_validation_predictions=val,test_prediction=p,member_test_predictions=members)
    result={'repeat':repeat,'inner_split_seed':int(STATE['split_seeds'][repeat]),'initialization_seeds':seeds,
        'lineage':label,'holdout_index':index,'train_n':len(tr),'test_n':len(te),'selected_candidate':chosen,
        'selected_parameters':prior.NN_GRID[chosen],'mean_inner_rmse':tuning[chosen]['mean_inner_rmse'],
        'metrics':prior.metrics(y[te],p),'tuning':tuning,'fit_diagnostics':diagnostics,'protocol_sha256':ph,
        'model_sha256':prior.digest(jpth),'predictions_sha256':prior.digest(npth)}
    prior.dump(mpth,result);return result


def main():
    started=time.time();load_state()
    sources=[]
    source_sets=[('erastin_40gene_comparison','artifact_manifest.json',['comparison_inputs.npz']),
        ('erastin_lineage_corrected','artifact_manifest.json',['shared_splits.npz','per_run_predictions.npz','per_lineage_metrics.tsv','protocol.json']),
        ('erastin_nonlinear_comparison','artifact_manifest.json',['run_models.py','protocol.json','predictions.npz']),
        ('erastin_repeated_cv','artifact_manifest.json',['repeat_cv.py'])]
    replay_indices=[STATE['labels'].index(label) for label in ['Lung','Thyroid']]
    source_sets[2][2].extend([f'folds/neural_network__panel_expression__{i:02d}.json' for i in replay_indices])
    for directory,manifest,names in source_sets:
        root=WS/'outputs'/directory;known={r['path']:r['sha256'] for r in json.loads((root/manifest).read_text())['files']}
        for name in names:
            path=root/name;h=prior.digest(path);assert h==known[name]
            sources.append({'path':str(path.relative_to(WS)),'sha256':h})
    protocol={'status':'frozen_before_followup_fitting','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'script_sha256':prior.digest(Path(__file__)),'samples':745,'identity_groups':734,'outer_lineages':24,
        'features':'same 40-gene log2(TPM+1) expression panel only','outer_holdouts':'unchanged from corrected leave-lineage-out evaluation',
        'repeats':20,'inner_split_seeds':[int(v) for v in STATE['split_seeds'][1:]],
        'initialization_master_seed':MASTER,'initialization_generator':'numpy.random.SeedSequence(master).generate_state(60).reshape(20,3)',
        'initialization_seed_triplets':STATE['nn_seeds'][1:].tolist(),
        'variation':'inner fold assignment and neural initialization trio change jointly; outer lineage tests stay fixed',
        'nn_fixed_settings':STATE['old_protocol']['nn_fixed'],
        'initialization_override':'use this protocol seed triplet for each repeat; other neural settings unchanged',
        'nn_grid':prior.NN_GRID,'ensemble':'three-member mean in both inner validation and outer prediction',
        'tuning':'four candidates, minimum equal-weight mean five-fold inner RMSE; X and target scalers fit on current training subset only',
        'ridge':'reuse saved ridge panel-expression predictions from the identical 20 inner split seeds; no averaging before pairing',
        'original_seed':'excluded from followup means; replay Lung and Thyroid to verify the unchanged prior neural pipeline',
        'metrics':['spearman','rmse','r2'],'macro':'equal-weight within-lineage metrics for the same 19 lineages with n>=10',
        'bootstrap':{'count':5000,'pooled_seed':2026110701,'macro_seed':2026110702,
            'unit':'whole held-out lineage, all models/repeats paired','estimand':'difference of mean per-repeat metrics',
            'scope':'conditional on fitted predictions; no bootstrap refitting; repeats are not independent datasets'},
        'reporting':'all 20 repeats, mean/SD/range, paired win count, per-lineage means; no adaptive seed selection',
        'previous_single_split_results_seen':True,'warnings':'record all warnings; no test-score-based retry or grid expansion',
        'workers':6}
    path=OUT/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text());protocol['created_utc']=old['created_utc'];assert old==protocol
    else:prior.dump(path,protocol)
    ph=prior.digest(path);prior.dump(OUT/'source_manifest.json',{'files':sources})
    np.savez_compressed(OUT/'shared_splits.npz',sample_ids=STATE['ids'],group_ids=STATE['groups'],lineage=STATE['lineage'],
        holdout_labels=np.array(STATE['labels']),inner_split_seeds=STATE['split_seeds'][1:],
        neural_initialization_seeds=STATE['nn_seeds'][1:],inner_fold_by_holdout=STATE['inner'][1:])
    print(json.dumps({'event':'protocol_and_all_seeds_frozen','protocol_sha256':ph}),flush=True)
    old_p=np.load(WS/'outputs/erastin_nonlinear_comparison/predictions.npz',allow_pickle=False)
    assert np.array_equal(old_p['sample_ids'],STATE['ids'])
    replay=[]
    with futures.ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn'),initializer=load_state) as pool:
        jobs=[pool.submit(fit_holdout,0,i,ph) for i in replay_indices]
        for job in jobs:
            r=job.result();i=r['holdout_index'];p=np.load(OUT/'replay'/f'repeat_00__lineage_{i:02d}.npz',allow_pickle=False)
            old_meta=json.loads((WS/'outputs/erastin_nonlinear_comparison/folds'/f'neural_network__panel_expression__{i:02d}.json').read_text())
            error=float(np.max(np.abs(p['test_prediction']-old_p['neural_network__panel_expression'][p['test_rows']])))
            assert error<1e-10 and r['selected_parameters']==old_meta['selected_parameters']
            assert np.allclose([v['mean_inner_rmse'] for v in r['tuning']],[v['mean_inner_rmse'] for v in old_meta['tuning']],rtol=0,atol=1e-12)
            replay.append({'lineage':r['lineage'],'max_abs_prediction_error':error,'selected_parameters_reproduced':True,'all_candidate_losses_reproduced':True})
        prior.dump(OUT/'original_replay_validation.json',{'passed':True,'checks':replay})
        print(json.dumps({'event':'original_neural_pipeline_reproduced','checks':replay}),flush=True)
        pending={pool.submit(fit_holdout,r,i,ph):(r,i) for r in range(1,21) for i in range(24)}
        completed=0;counts={r:0 for r in range(1,21)}
        for future in futures.as_completed(pending):
            result=future.result();completed+=1;counts[result['repeat']]+=1
            progress={'phase':'fitting','completed_outer_jobs':completed,'total_outer_jobs':480,
                'complete_repeats':sum(v==24 for v in counts.values()),'total_repeats':20,'elapsed_seconds':time.time()-started}
            prior.dump(OUT/'progress.json',progress)
            if completed%12==0 or completed==480:print(json.dumps({'event':'fit_progress',**progress}),flush=True)
    assert prior.digest(path)==ph and prior.digest(Path(__file__))==protocol['script_sha256']
    for s in sources:assert prior.digest(WS/s['path'])==s['sha256']
    prior.dump(OUT/'fit_completion.json',{'status':'complete','outer_jobs':480,'repeats':20,
        'elapsed_seconds':time.time()-started,'source_script_and_protocol_hashes_unchanged':True,'protocol_sha256':ph})
    prior.dump(OUT/'progress.json',{'phase':'fits_complete','completed_outer_jobs':480,'total_outer_jobs':480,'complete_repeats':20})
    print(json.dumps({'event':'all_fits_complete','elapsed_seconds':time.time()-started}),flush=True)


if __name__=='__main__':main()
