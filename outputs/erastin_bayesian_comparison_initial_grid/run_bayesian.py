#!/usr/bin/env python3
"""Nested Gaussian BN benchmark on the saved 21 corrected lineage split sets."""
import concurrent.futures as futures
import hashlib
import importlib.util
import json
import multiprocessing as mp
from pathlib import Path
import platform
import sys
import time

import numpy as np
import sklearn

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
WS=OUT.parent.parent
sys.path.insert(0,str(OUT))
import gaussian_network as gn
FAMILIES=['gaussian_naive','gaussian_tan']
GRID=[0.,.01,.03,.1,.2,.4,.6,.8]
STATE={}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def clean(x):
    if isinstance(x,dict):return {k:clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,(np.integer,)):return int(x)
    if isinstance(x,(np.floating,float)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.bool_):return bool(x)
    return x


def dump(path,value):
    temp=path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n');temp.replace(path)


def initialize():
    d=np.load(WS/'outputs/erastin_40gene_comparison/comparison_inputs.npz',allow_pickle=False)
    s=np.load(WS/'outputs/erastin_lineage_corrected/shared_splits.npz',allow_pickle=False)
    assert np.array_equal(d['sample_ids'],s['sample_ids'])
    STATE.update(x=d['X_panel_log2tpm1'].astype(float),y=d['y_sensitivity'].astype(float),ids=d['sample_ids'],
        lineage=d['lineage'],groups=d['group_ids'],genes=d['panel_symbols'],inner=s['inner_fold_by_holdout'],
        labels=s['holdout_labels'],seeds=s['seeds'])


def holdout(repeat,index,ph):
    token=f'repeat_{repeat:02d}__lineage_{index:02d}'
    jp=OUT/'folds'/(token+'.json');npth=OUT/'folds'/(token+'.npz')
    if jp.exists():
        meta=json.loads(jp.read_text());assert meta['protocol_sha256']==ph and sha(npth)==meta['array_sha256'];return meta
    x,y=STATE['x'],STATE['y'];label=STATE['labels'][index];inner=STATE['inner'][repeat,index]
    tr=np.flatnonzero(inner>=0);te=np.flatnonzero(STATE['lineage']==label)
    assert set(STATE['groups'][tr]).isdisjoint(STATE['groups'][te]) and label not in set(STATE['lineage'][tr])
    val=np.full((2,len(GRID),len(y)),np.nan);tuning=np.empty((2,len(GRID),5))
    # Nothing outside each inner training set enters its moments or tree.
    for k in range(5):
        a=tr[inner[tr]!=k];b=tr[inner[tr]==k]
        assert set(a)|set(b)==set(tr) and set(STATE['groups'][a]).isdisjoint(STATE['groups'][b])
        for m,family in enumerate(FAMILIES):
            for c,shrink in enumerate(GRID):
                model=gn.fit_network(x[a],y[a],family,shrink)
                p,_=gn.predict(model,x[b]);assert np.isfinite(p).all()
                val[m,c,b]=p;tuning[m,c,k]=np.sqrt(np.mean((y[b]-p)**2))
    assert np.isfinite(val[:,:,tr]).all() and np.isnan(val[:,:,te]).all()
    chosen=np.argmin(tuning.mean(axis=2),axis=1)
    saved=[];pred=[];variance=[]
    for m,family in enumerate(FAMILIES):
        model=gn.fit_network(x[tr],y[tr],family,GRID[chosen[m]])
        p,var=gn.predict(model,x[te]);saved.append(model);pred.append(p);variance.append(var)
    np.savez_compressed(npth,train_rows=tr,test_rows=te,inner_validation_predictions=val,
        inner_fold_rmse=tuning,test_predictions=np.array(pred),test_variances=np.array(variance),selected_candidates=chosen)
    meta={'repeat':repeat,'inner_seed':int(STATE['seeds'][repeat]),'lineage_index':index,'lineage':str(label),
        'train_n':len(tr),'test_n':len(te),'selected_models':saved,'selected_candidates':chosen,
        'selected_mean_inner_rmse':[tuning[m,chosen[m]].mean() for m in range(2)],
        'protocol_sha256':ph,'array_sha256':sha(npth),'outer_and_inner_groups_disjoint':True}
    dump(jp,meta);return meta


def main():
    start=time.time();initialize();(OUT/'folds').mkdir(parents=True,exist_ok=True)
    sources=[]
    for directory,manifest,names in [
        ('erastin_40gene_comparison','artifact_manifest.json',['comparison_inputs.npz']),
        ('erastin_lineage_corrected','artifact_manifest.json',['shared_splits.npz','per_run_predictions.npz','per_lineage_metrics.tsv']),
        ('erastin_nn_repeated','artifact_manifest.json',['predictions.npz','protocol.json']),
        ('erastin_nonlinear_comparison','artifact_manifest.json',['predictions.npz','summary.tsv','protocol.json']),
        ('erastin_repeated_cv','artifact_manifest.json',['repeat_cv.py'])]:
        root=WS/'outputs'/directory;known={v['path']:v['sha256'] for v in json.loads((root/manifest).read_text())['files']}
        for name in names:
            path=root/name;h=sha(path);assert h==known[name]
            sources.append({'path':str(path.relative_to(WS)),'sha256':h})
    protocol={'status':'frozen_before_BN_heldout_results','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'scripts':{name:sha(OUT/name) for name in ['gaussian_network.py','run_bayesian.py']},
        'data':'unchanged 745 cancer lines, 734 related-identity groups, 40 log2(TPM+1) panel genes',
        'response':'negative reported CTRP erastin AUC, continuous, no discretization or prediction clipping',
        'families':FAMILIES,
        'gaussian_naive':'Y is the root, all 40 genes have Y as their sole parent; conditionally independent genes given continuous Y',
        'gaussian_tan':'Y is root and parent of all genes; add a maximum spanning tree of Gaussian conditional mutual information between genes given Y; orient gene tree from fixed panel index 0',
        'interpretation':'arrows express statistical factorization, not a causal or mechanistic gene network; conditional response mean is linear in expression',
        'parameter_estimation':'point estimates from training-only standardized joint covariance; not full Bayesian averaging over graph and parameter uncertainty',
        'regularization':'S=(1-lambda)*C + lambda*I on jointly standardized Y and X; both graph and local Gaussian CPDs use S',
        'shrinkage_grid':GRID,'graph_edges':'39 gene-gene tree edges plus 40 response-gene edges for TAN; 40 response-gene edges for naive',
        'inference':'exact Gaussian conditional mean and variance from BN precision; ensemble not required because fitting is deterministic',
        'outer_splits':'all 24 fixed corrected lineage holdouts unchanged; original inner seed plus saved 20 followup seeds',
        'inner_seeds':[int(v) for v in STATE['seeds']],
        'tuning':'minimum equal-weight mean 5-fold inner RMSE; each inner fit learns its own scalers, moments, graph, CPDs; first candidate wins ties',
        'primary_comparisons':'20-repeat TAN vs matched expression ridge, expression neural network and naive BN; paired seed-level metrics',
        'secondary_comparisons':'original seed 20261031 BN vs all nine saved ridge/RF/NN x expression/panel-BulkFormer/all-BulkFormer arms; RF not available over 20 repeats',
        'additional_repeat_references':'saved panel-focused and all-gene BulkFormer ridge across the same 20 seeds',
        'metrics':['spearman','rmse','r2'],'macro':'equal weight over same 19 lineages with n>=10; all 745 lines in pooled scores',
        'uncertainty':{'count':5000,'repeat_pooled_seed':2026110901,'repeat_macro_seed':2026110902,
            'original_pooled_seed':2026110903,'original_macro_seed':2026110904,
            'unit':'whole held-out lineage; all arms and repeats paired; conditional on predictions with no refitting; no independent-seed significance tests'},
        'predictive_uncertainty':'held-out Gaussian negative log likelihood, 90% conditional predictive interval coverage and width; no post-hoc calibration',
        'graph_reporting':'undirected gene-edge stability across 480 selected repeat TAN fits; do not interpret arbitrary tree direction as causality',
        'previous_models_and_gene_importance_results_seen':True,'exploratory_followup':True,
        'workers':6,'environment':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},
        'references':[
            {'url':'https://dang.cs.technion.ac.il/journal_papers/friedman1997Bayesian.pdf','use':'TAN factorization and maximum spanning tree; original paper is classification, implemented here with continuous Gaussian CPDs'},
            {'url':'https://pgmpy.org/api/generated/models/pgmpy.models.LinearGaussianBayesianNetwork.html','use':'linear Gaussian BN to joint Gaussian representation and conditioning'},
            {'url':'https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ledoit_wolf.html','use':'diagonal covariance shrinkage form only; lambda selected by inner CV, not Ledoit-Wolf estimate'}]}
    path=OUT/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text());protocol['created_utc']=old['created_utc'];assert old==protocol
    else:dump(path,protocol)
    ph=sha(path);dump(OUT/'source_manifest.json',{'files':sources})
    dump(OUT/'mathematical_validation.json',gn.verify_math())
    np.savez_compressed(OUT/'shared_splits.npz',sample_ids=STATE['ids'],groups=STATE['groups'],lineage=STATE['lineage'],
        genes=STATE['genes'],labels=STATE['labels'],seeds=STATE['seeds'],inner_fold_by_holdout=STATE['inner'])
    print(json.dumps({'event':'protocol_frozen','sha256':ph,'model_holdouts':1008}),flush=True)
    done=0
    with futures.ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        pending=[pool.submit(holdout,r,l,ph) for r in range(21) for l in range(24)]
        for future in futures.as_completed(pending):
            future.result();done+=1
            if done%24==0:
                progress={'phase':'fitting','completed_lineage_jobs':done,'total_lineage_jobs':504,'elapsed_seconds':time.time()-start}
                dump(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
    for source in sources:assert sha(WS/source['path'])==source['sha256']
    for name,h in protocol['scripts'].items():assert sha(OUT/name)==h
    assert sha(path)==ph
    dump(OUT/'fit_completion.json',{'status':'complete','model_holdouts':1008,'inner_model_fits':21*24*2*8*5,
        'elapsed_seconds':time.time()-start,'source_and_script_hashes_unchanged':True})
    dump(OUT/'progress.json',{'phase':'fits_complete','completed_lineage_jobs':504,'total_lineage_jobs':504})
    print(json.dumps({'event':'fits_complete','elapsed_seconds':time.time()-start}),flush=True)


if __name__=='__main__':main()
