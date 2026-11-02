#!/usr/bin/env python3
"""Frozen 64-gene NN comparison with inverse-square-root and capped weights."""
import concurrent.futures as futures
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import shutil
import sys
import time
import warnings
import joblib
import numpy as np
import scipy
import sklearn
from sklearn.preprocessing import StandardScaler
from weighted_mlp import WeightedMLPRegressor
from lineage_weights import SCHEMES,weights_for,weight_audit,CAP

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;WS=OUT.parent.parent;WORKERS=10;STATE={}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return v.item()
    return v

def dump(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)

def initialize():
    a=np.load(OUT/'inputs.npz');s=np.load(OUT/'shared_splits.npz');p=json.loads((OUT/'protocol.json').read_text())
    STATE.update(x=a['x'],y=a['y'],genes=a['genes'],ids=a['sample_ids'],groups=a['group_ids'],lineage=a['lineage'],
        labels=s['holdout_labels'],inner=s['inner_fold_by_holdout'],seeds=s['inner_split_seeds'],nn_seeds=s['neural_initialization_seeds'],grid=p['neural_grid'])

def fitted_neural(params,x,y,tr,te,seeds,weights,retain=False):
    begin=time.perf_counter();sx=StandardScaler().fit(x[tr]);sy=StandardScaler().fit(y[tr,None])
    a=sx.transform(x[tr]);b=sx.transform(x[te]);target=sy.transform(y[tr,None]).ravel()
    members=[];predictions=[];diagnostics=[]
    for seed in seeds:
        model=WeightedMLPRegressor(hidden_layer_sizes=(params['width'],),activation='tanh',solver='lbfgs',
            alpha=params['alpha'],max_iter=2000,max_fun=40000,tol=.001,early_stopping=False,random_state=int(seed))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(a,target,sample_weight=weights)
        p=sy.inverse_transform(model.predict(b)[:,None]).ravel();assert np.isfinite(p).all();predictions.append(p)
        diagnostics.append(dict(seed=int(seed),iterations=model.n_iter_,training_loss=model.loss_,
            warnings=[dict(category=v.category.__name__,message=str(v.message)) for v in caught]))
        if retain:members.append(model)
    pp=np.array(predictions);saved=dict(family='neural_network',x_scaler=sx,y_scaler=sy,members=members) if retain else None
    return pp.mean(axis=0),saved,dict(seconds=time.perf_counter()-begin,members=diagnostics),pp

def fit_job(scheme,r,l,ph):
    root=OUT/'folds'/f'{scheme}__r{r+1:02d}__l{l:02d}'
    if root.with_suffix('.json').exists():
        m=json.loads(root.with_suffix('.json').read_text());assert m['protocol_sha256']==ph
        for ext in ['npz','joblib']:assert sha(root.with_suffix('.'+ext))==m[ext+'_sha256']
        return m
    begin=time.monotonic();s=STATE;x=s['x'];y=s['y'];g=s['groups'];lineage=s['lineage'];inner=s['inner'][r,l];heldout=s['labels'][l]
    tr=np.flatnonzero(inner>=0);te=np.flatnonzero(lineage==heldout)
    assert set(g[tr]).isdisjoint(g[te]) and heldout not in set(lineage[tr])
    grid=s['grid'];val=np.full((len(grid),len(y)),np.nan);loss=np.empty((len(grid),5));inner_weights=np.full((5,len(y)),np.nan)
    diagnostics=[];scaling=[];seeds=s['nn_seeds'][r]
    for k in range(5):
        a=tr[inner[tr]!=k];b=tr[inner[tr]==k];assert set(a)|set(b)==set(tr) and set(g[a]).isdisjoint(g[b])
        w=weights_for(lineage[a],scheme);inner_weights[k,a]=w
        scaling.append(dict(inner_fold=k,y_mean=float(y[a].mean()),y_scale=float(y[a].std()),weights=weight_audit(lineage[a],scheme)))
        for c,params in enumerate(grid):
            p,_,diag,_=fitted_neural(params,x,y,a,b,seeds,w)
            val[c,b]=p;loss[c,k]=np.sqrt(np.mean((y[b]-p)**2));diagnostics.append(dict(stage='inner',candidate=c,inner_fold=k,**diag))
    assert np.isfinite(val[:,tr]).all() and np.isnan(val[:,inner<0]).all()
    c=int(np.argmin(loss.mean(axis=1)));params=grid[c];w=weights_for(lineage[tr],scheme)
    p,saved,diag,mpred=fitted_neural(params,x,y,tr,te,seeds,w,retain=True);diagnostics.append(dict(stage='outer',candidate=c,**diag))
    joblib.dump(saved,root.with_suffix('.joblib'),compress=3)
    np.savez_compressed(root.with_suffix('.npz'),train_rows=tr,test_rows=te,inner_validation_predictions=val,inner_fold_rmse=loss,
        test_prediction=p,member_test_predictions=mpred,outer_training_weights=w,inner_training_weights=inner_weights)
    m=dict(scheme=scheme,panel=64,family='neural_network',repeat=r+1,lineage_index=l,lineage=str(heldout),inner_split_seed=int(s['seeds'][r]),
        neural_seeds=seeds,train_n=len(tr),test_n=len(te),selected_candidate=c,selected_parameters=params,mean_inner_rmse=float(loss[c].mean()),
        inner_scaling=scaling,outer_weights=weight_audit(lineage[tr],scheme),fit_diagnostics=diagnostics,protocol_sha256=ph,
        npz_sha256=sha(root.with_suffix('.npz')),joblib_sha256=sha(root.with_suffix('.joblib')),seconds=time.monotonic()-begin)
    dump(root.with_suffix('.json'),m);return m

def prepare():
    OUT.mkdir(exist_ok=True);(OUT/'folds').mkdir(exist_ok=True);sources=[]
    for directory,names in [('erastin_panel_comparison',['panel_inputs.npz','shared_splits.npz','protocol.json','predictions.npz','per_repeat_metrics.tsv']),
                            ('erastin_lineage_weighted',['predictions.npz','protocol.json','weighted_mlp.py','preflight_validation.json','validation.json'])]:
        root=WS/'outputs'/directory;known={v['path']:v['sha256'] for v in json.loads((root/'artifact_manifest.json').read_text())['files']}
        for name in names:
            assert sha(root/name)==known[name],name;sources.append(dict(path=str((root/name).relative_to(WS)),sha256=known[name]))
    for row in json.loads((OUT/'helper_sources.json').read_text())['sources']:
        assert sha(WS/row['path'])==row['sha256'];sources.append(dict(path=row['path'],sha256=row['sha256']))
    base=WS/'outputs/erastin_panel_comparison';a=np.load(base/'panel_inputs.npz');s=np.load(base/'shared_splits.npz');prior=json.loads((base/'protocol.json').read_text())
    assert sha(OUT/'weighted_mlp.py')==sha(WS/'outputs/erastin_lineage_weighted/weighted_mlp.py')
    versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__)
    assert sklearn.__version__=='1.5.2' and versions['numpy']==prior['versions']['numpy']
    p=dict(status='frozen_before_moderate_weight_fits',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        genes=list(a['genes_64']),samples=745,identity_groups=734,repeats=20,lineage_holdouts=24,inner_folds=5,
        split_seeds=s['inner_split_seeds'],neural_seed_triplets=s['neural_initialization_seeds'],neural_grid=prior['neural_grid'],neural_fixed=prior['neural_fixed'],
        schemes=dict(inverse_sqrt='w_i=(1/sqrt(n_lineage))/(training-subset mean of 1/sqrt(n_lineage)); mean one.',
            capped_inverse_5='w_i=min(c/n_lineage,5), c solved so sum(w)=N. The FINAL normalized weights have a hard cap of five; uncapped lineages retain inverse-frequency ratios.'),
        cap=CAP,cap_selection='Fixed at five before fitting or inspecting the new test predictions; no cap search or test-based selection.',
        weight_scope='Each current fitting subset independently; training memberships only. All represented lineages included, including n<10.',
        tuning='UNCHANGED minimum equal-weight mean of five UNWEIGHTED inner-fold RMSEs. First candidate breaks ties. Each arm tuned separately.',
        scaling='UNCHANGED unweighted StandardScaler for X/y, fit only on current training rows; mean-one weights retain the prior L2 scale.',
        controls='Exact saved sample order, targets, groups, outer/inner splits, grid, three initialization seeds per repeat, architecture, L-BFGS settings and weighted-loss adapter.',
        baseline='Reuse verified unweighted 64-gene NN predictions. Previous full inverse-frequency 64-gene NN included as a reused contextual reference.',
        primary_comparisons=[['inverse_sqrt','unweighted'],['capped_inverse_5','unweighted']],
        secondary_comparisons=[['inverse_sqrt','inverse_frequency'],['capped_inverse_5','inverse_frequency'],['inverse_sqrt','capped_inverse_5']],
        reporting='Mean of per-repeat pooled 745-cell metrics and macro metrics over the same 19 lineages with n>=10. All 20 repeats retained.',
        uncertainty=dict(count=5000,pooled_seed=2026110501,macro_seed=2026110502,unit='Paired whole held-out lineage across all arms/repeats, conditional on fitted predictions; no refitting or multiplicity adjustment.'),
        limitations='Exploratory reuse of the existing cohort, motivated by earlier inverse-frequency results; no independent validation. Inner folds mix training lineages, outer tests hold out entire lineages.',
        new_outer_models=960,workers=WORKERS,versions=versions,
        scripts={name:sha(OUT/name) for name in ['run_moderate.py','lineage_weights.py','weighted_mlp.py','evaluation.py','helper_sources.json','validate_preflight.py']})
    pp=OUT/'protocol.json'
    if pp.exists():
        old=json.loads(pp.read_text());p['created_utc']=old['created_utc'];assert old==clean(p)
    else:dump(pp,p)
    dump(OUT/'source_manifest.json',dict(files=sources))
    np.savez_compressed(OUT/'inputs.npz',x=a['x_64'],genes=a['genes_64'],y=a['y'],sample_ids=a['sample_ids'],group_ids=a['group_ids'],lineage=a['lineage'])
    shutil.copyfile(base/'shared_splits.npz',OUT/'shared_splits.npz')
    return sha(pp)

def main():
    start=time.monotonic();ph=prepare();initialize();v=json.loads((OUT/'preflight_validation.json').read_text());assert v['passed'] and v['protocol_sha256']==ph
    print(json.dumps(dict(event='protocol_frozen',sha256=ph,new_outer_models=960)),flush=True)
    with futures.ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        jobs=[(scheme,r,l) for r in range(20) for l in range(24) for scheme in SCHEMES]
        pending=[pool.submit(fit_job,*args,ph) for args in jobs];n=0;counts={}
        for future in futures.as_completed(pending):
            m=future.result();n+=1;counts[m['scheme']]=counts.get(m['scheme'],0)+1;elapsed=time.monotonic()-start
            progress=dict(phase='fitting',completed=n,total=960,by_arm=counts,elapsed_seconds=elapsed,estimated_remaining_seconds=elapsed/n*(960-n))
            dump(OUT/'progress.json',progress)
            if n%48==0 or n==960:print(json.dumps(dict(event='fit_progress',**progress)),flush=True)
    p=json.loads((OUT/'protocol.json').read_text());assert sha(OUT/'protocol.json')==ph
    for name,h in p['scripts'].items():assert sha(OUT/name)==h
    for v in json.loads((OUT/'source_manifest.json').read_text())['files']:assert sha(WS/v['path'])==v['sha256']
    dump(OUT/'fit_completion.json',dict(complete=True,new_outer_models=960,elapsed_seconds=time.monotonic()-start,sources_unchanged=True))
    dump(OUT/'progress.json',dict(phase='fits_complete',completed=960,total=960));print(json.dumps(dict(event='all_fits_complete',seconds=time.monotonic()-start)),flush=True)
if __name__=='__main__':main()
