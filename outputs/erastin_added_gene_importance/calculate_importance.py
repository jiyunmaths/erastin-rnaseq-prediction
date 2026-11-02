#!/usr/bin/env python3
"""Exact held-out within-lineage permutation analysis of the 24 added genes."""
import concurrent.futures as futures
import hashlib
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;WS=OUT.parent.parent
SOURCE=WS/'outputs/erastin_panel_comparison';STATE={}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    if isinstance(v,np.bool_):return bool(v)
    return v
def dump(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n');temp.replace(path)
def initialize():
    d=np.load(OUT/'analysis_inputs.npz');STATE.update({k:d[k] for k in d.files})
def native(model,x):
    z=model['x_scaler'].transform(x)
    return np.mean([model['y_scaler'].inverse_transform(m.predict(z)[:,None]).ravel() for m in model['members']],axis=0)

def one(r,l,ph):
    root=OUT/'folds'/f'r{r+1:02d}__l{l:02d}'
    if root.with_suffix('.json').exists():
        meta=json.loads(root.with_suffix('.json').read_text());assert meta['protocol_sha256']==ph and meta['array_sha256']==sha(root.with_suffix('.npz'));return meta
    source=SOURCE/'folds'/f'neural_network__p64__r{r+1:02d}__l{l:02d}'
    meta=json.loads(source.with_suffix('.json').read_text());a=np.load(source.with_suffix('.npz'));model=joblib.load(source.with_suffix('.joblib'))
    tr,te=a['train_rows'],a['test_rows'];x=STATE['x'][te];y=STATE['y'][te];indices=STATE['added_indices'];n=len(te);p=len(indices)
    np.testing.assert_array_equal(te,np.flatnonzero(STATE['lineage']==STATE['labels'][l]));np.testing.assert_array_equal(tr,np.flatnonzero(STATE['inner'][r,l]>=0))
    assert set(STATE['groups'][tr]).isdisjoint(STATE['groups'][te]) and STATE['labels'][l] not in set(STATE['lineage'][tr])
    sx=StandardScaler().fit(STATE['x'][tr]);sy=StandardScaler().fit(STATE['y'][tr,None])
    for actual,expected in [(model['x_scaler'].mean_,sx.mean_),(model['x_scaler'].scale_,sx.scale_),
                            (model['y_scaler'].mean_,sy.mean_),(model['y_scaler'].scale_,sy.scale_)]:np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
    assert [m.random_state for m in model['members']]==meta['neural_seeds']==STATE['neural_seeds'][r].tolist()
    z=sx.transform(x);prediction=native(model,x);base=(prediction-y)**2
    error=float(np.max(np.abs(prediction-STATE['predictions'][r,te])));assert error<1e-11
    np.testing.assert_allclose(prediction,a['test_prediction'],rtol=0,atol=1e-11)
    members=[];gradient=np.zeros((n,p))
    for m in model['members']:
        assert m.activation=='tanh' and len(m.coefs_)==2 and m.out_activation_=='identity'
        w1=m.coefs_[0];w2=m.coefs_[1].ravel();h=z@w1+m.intercepts_[0]
        members.append((h,w1,w2,float(m.intercepts_[1][0])))
        gradient+=(((1-np.tanh(h)**2)*w2)@w1[indices].T)*sy.scale_[0]/len(model['members'])
    # One native receiver/donor check for every added gene plus the block.
    direct=np.repeat(x[[0]],p+1,axis=0)
    direct[np.arange(p),indices]=x[-1,indices];direct[-1,indices]=x[-1,indices]
    reference=native(model,direct)
    eps=1e-5;plus=np.repeat(x[[0]],p,axis=0);minus=plus.copy()
    plus[np.arange(p),indices]+=eps*sx.scale_[indices];minus[np.arange(p),indices]-=eps*sx.scale_[indices]
    finite=(native(model,plus)-native(model,minus))/(2*eps)
    gradient_error=float(np.max(np.abs(finite-gradient[0])));assert gradient_error<1e-8
    cell_loss=np.empty((n,p+1));cell_shift=np.empty_like(cell_loss)
    identity_error=swap_error=exhaustive_error=0.
    for j in range(p+1):
        shuffled=np.zeros((n,n))
        for h,w1,w2,bias in members:
            if j<p:
                difference=z[None,:,indices[j]]-z[:,None,indices[j]]
                latent_shift=difference[:,:,None]*w1[indices[j]]
            else:
                contribution=z[:,indices]@w1[indices]
                latent_shift=contribution[None,:,:]-contribution[:,None,:]
            pp=(np.tanh(h[:,None,:]+latent_shift)@w2+bias)*sy.scale_[0]+sy.mean_[0]
            shuffled+=pp/len(members)
        identity_error=max(identity_error,float(np.max(np.abs(np.diag(shuffled)-prediction))))
        swap_error=max(swap_error,float(abs(shuffled[0,-1]-reference[j])))
        loss=(shuffled-y[:,None])**2
        cell_loss[:,j]=loss.mean(axis=1)-base
        cell_shift[:,j]=((shuffled-prediction[:,None])**2).mean(axis=1)
        if n==2 and r==0:
            losses=[];shifts=[]
            for perm in itertools.permutations(range(n)):
                xx=x.copy()
                if j<p:xx[:,indices[j]]=x[np.array(perm),indices[j]]
                else:xx[:,indices]=x[np.array(perm)][:,indices]
                pp=native(model,xx);losses.append(np.mean((pp-y)**2));shifts.append(np.mean((pp-prediction)**2))
            exhaustive_error=max(exhaustive_error,float(abs(np.mean(losses)-loss.mean())),float(abs(np.mean(shifts)-cell_shift[:,j].mean())))
    assert max(identity_error,swap_error,exhaustive_error)<1e-11
    if n==1:assert np.max(np.abs(cell_loss))<1e-12 and np.max(cell_shift)<1e-20
    np.savez_compressed(root.with_suffix('.npz'),test_rows=te,baseline_prediction=prediction,baseline_mse=base.mean(),
                        per_cell_delta_mse=cell_loss,per_cell_mean_squared_prediction_shift=cell_shift,
                        importance_mse=cell_loss.mean(axis=0),mean_squared_prediction_shift=cell_shift.mean(axis=0),standardized_gradient=gradient)
    result=dict(repeat=r+1,lineage_index=l,lineage=str(STATE['labels'][l]),n=n,protocol_sha256=ph,
        baseline_prediction_error=error,native_swap_error=swap_error,identity_swap_error=identity_error,
        gradient_finite_difference_error=gradient_error,exhaustive_two_cell_error=exhaustive_error,
        array_sha256=sha(root.with_suffix('.npz')))
    dump(root.with_suffix('.json'),result);return result

def main():
    start=time.monotonic();(OUT/'folds').mkdir(exist_ok=True)
    known={r['path']:r['sha256'] for r in json.loads((SOURCE/'artifact_manifest.json').read_text())['files']}
    names=['panel_inputs.npz','shared_splits.npz','predictions.npz','panel_definitions.json','validation.json']
    names.extend(str(p.relative_to(SOURCE)) for p in sorted((SOURCE/'folds').glob('neural_network__p64__*')))
    sources=[]
    for filename in names:
        path=SOURCE/filename;assert sha(path)==known[filename];sources.append(dict(path=str(path.relative_to(WS)),sha256=known[filename]))
    d=np.load(SOURCE/'panel_inputs.npz');s=np.load(SOURCE/'shared_splits.npz');pred=np.load(SOURCE/'predictions.npz')
    definition=json.loads((SOURCE/'panel_definitions.json').read_text());genes=d['genes_64'];added=np.array(definition['first_additions_24'])
    indices=np.array([list(genes).index(g) for g in added]);assert len(indices)==24 and np.array_equal(indices,np.arange(40,64))
    for a in [s,pred]:np.testing.assert_array_equal(a['sample_ids'],d['sample_ids'])
    protocol=dict(status='frozen_before_added_gene_importance',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        script_sha256=sha(Path(__file__)),model='Saved three-member 64-gene neural-network ensembles; no retraining.',
        genes=added,baseline_genes=genes[:40],model_holdouts=480,repeats=20,
        primary_metric='Exact expected held-out MSE increase after a within-held-out-lineage permutation of one added gene.',
        influence='RMS change in predicted negative erastin AUC under the same permutation; computed separately from error benefit.',
        exact_expectation='Enumerate n*n receiver/donor pairs including self replacements; additive per-cell losses make this the expectation over all random permutations.',
        block='Jointly permute all 24 additions as one vector, preserving their internal dependence. Breaks their association with the original 40 genes and response.',
        gradients='Analytical local derivative per training SD of log2(TPM+1), averaged over ensemble members before aggregation; associative direction only.',
        primary_scope='Sample-weighted 19 lineages with n>=10 (721 cells); equal-lineage weighting as sensitivity analysis.',
        small_lineages='n=2..9 descriptive only; singleton has no within-lineage permutation information.',
        uncertainty=dict(draws=5000,seed=2026110301,unit='Whole held-out lineage among 19 eligible lineages, paired genes/repeats.',
                         scope='Pointwise conditional bootstrap, no refitting or multiplicity correction; seeds reuse the same cohort.'),
        consistency='Per-seed ranks/top-five frequency; per-lineage signs/top-five ranks; pairwise rank correlation across lineages.',
        limits=['Permutation can disrupt correlated genes and generate atypical expression combinations.',
                'Importance in the 64-gene model does not isolate the gain from retraining a 40-gene model; individual effects are not additive.',
                'Prediction direction is not a causal gene-perturbation effect; low importance does not imply biological irrelevance.'],workers=6)
    path=OUT/'protocol.json'
    if path.exists():
        prior=json.loads(path.read_text());protocol['created_utc']=prior['created_utc'];assert prior==clean(protocol)
    else:dump(path,protocol)
    ph=sha(path);dump(OUT/'source_manifest.json',dict(files=sources))
    np.savez_compressed(OUT/'analysis_inputs.npz',x=d['x_64'],y=d['y'],sample_ids=d['sample_ids'],groups=d['group_ids'],
                        lineage=d['lineage'],genes=genes,added_genes=added,added_indices=indices,labels=s['holdout_labels'],
                        inner=s['inner_fold_by_holdout'],neural_seeds=s['neural_initialization_seeds'],predictions=pred['neural_network__64'])
    print(json.dumps(dict(event='protocol_frozen',sha256=ph)),flush=True);initialize();results=[]
    with futures.ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        jobs=[pool.submit(one,r,l,ph) for r in range(20) for l in range(24)]
        for future in futures.as_completed(jobs):
            results.append(future.result())
            if len(results)%48==0:
                progress=dict(phase='calculating',complete=len(results),total=480,elapsed_seconds=time.monotonic()-start)
                dump(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
    for source in sources:assert sha(WS/source['path'])==source['sha256']
    assert sha(path)==ph and sha(Path(__file__))==protocol['script_sha256']
    dump(OUT/'calculation_validation.json',dict(passed=True,saved_models=480,added_genes=24,joint_block=True,no_refitting=True,
         checks=results,elapsed_seconds=time.monotonic()-start))
    dump(OUT/'progress.json',dict(phase='calculation_complete',complete=480,total=480))
    print(json.dumps(dict(event='calculation_complete',seconds=time.monotonic()-start)),flush=True)

if __name__=='__main__':main()
