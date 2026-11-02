#!/usr/bin/env python3
"""Exploratory fixed five-gene versus 64-gene ridge/NN comparison on shared splits."""
import concurrent.futures as futures
import json,multiprocessing as mp,platform,shutil,sys,time,warnings
from pathlib import Path
import joblib,numpy as np,scipy,sklearn
from model_helpers import sha,clean,dump,fitted_neural
from ridge_helpers import ridge_path
from make_splits import inner_rows
OUT=Path(__file__).resolve().parent;WS=OUT.parent.parent;FAMILIES=['ridge','neural_network'];PANELS=[5,64];WORKERS=10;STATE={}
FIVE_GENES=["SLC7A11","GPX4","ABCC1","ACSL4","AIFM2"]
sys.dont_write_bytecode=True

def arm_name(f,p):return f+'__'+str(p)
def initialize():
    a=np.load(OUT/'panel_inputs.npz');s=np.load(OUT/'shared_splits.npz');p=json.loads((OUT/'protocol.json').read_text())
    STATE.update(x={n:a['x_'+str(n)] for n in PANELS},genes={n:a['genes_'+str(n)] for n in PANELS},y=a['y'],ids=a['sample_ids'],groups=a['group_ids'],lineage=a['lineage'],
        test=s['outer_test'],inner=s['inner_fold'],seeds=s['outer_split_seeds'],inner_seeds=s['inner_split_seeds'],nn_seeds=s['neural_initialization_seeds'],nn_grid=p['neural_grid'],alphas=np.array(p['ridge_alphas']))
    STATE['model_seeds']=np.array([int(v)%(2**31-1) for v in STATE['seeds']])

def grid_for(f):return [dict(alpha=float(v)) for v in STATE['alphas']] if f=='ridge' else STATE['nn_grid']

def fit_job(panel,family,r,ph):
    root=OUT/'folds'/f'{family}__p{panel}__r{r+1:02d}'
    if root.with_suffix('.json').exists():
        m=json.loads(root.with_suffix('.json').read_text());assert m['protocol_sha256']==ph
        for ext in ['npz','joblib']:assert sha(root.with_suffix('.'+ext))==m[ext+'_sha256']
        return m
    start=time.monotonic();s=STATE;x=s['x'][panel];y=s['y'];genes=s['genes'][panel];g=s['groups'];lin=s['lineage'];inner=s['inner'][r]
    tr=np.flatnonzero(~s['test'][r]);te=np.flatnonzero(s['test'][r]);assert set(g[tr]).isdisjoint(g[te]) and set(lin[tr])==set(lin[te])==set(lin)
    grid=grid_for(family);val=np.full((len(grid),len(y)),np.nan);loss=np.empty((len(grid),5));scaling=[];diagnostics=[];seed=int(s['model_seeds'][r])
    for k in range(5):
        a,b=inner_rows(inner,k);assert set(a)|set(b)==set(tr) and set(g[a]).isdisjoint(g[b]) and set(lin[a])==set(lin)
        mu,sd=float(y[a].mean()),float(y[a].std());scaling.append(dict(fold=k,y_mean=mu,y_scale=sd))
        if family=='ridge':
            pp,_=ridge_path(x,y,a,b,s['alphas']);val[:,b]=pp.T;loss[:,k]=np.sqrt(np.mean((pp-y[b,None])**2,axis=0))
        elif family=='neural_network':
            for c,params in enumerate(grid):
                p,_,diag,_=fitted_neural(params,x,y,a,b,s['nn_seeds'][r],np.ones(len(a)));val[c,b]=p;loss[c,k]=np.sqrt(np.mean((y[b]-p)**2));diagnostics.append(dict(stage='inner',candidate=c,fold=k,**diag))
    assert np.isfinite(val[:,inner>=0]).all() and np.isnan(val[:,inner<0]).all();c=int(np.argmin(loss.mean(axis=1)));params=grid[c];members=None
    if family=='ridge':p,saved=ridge_path(x,y,tr,te,[params['alpha']],retain=True);p=p[:,0]
    elif family=='neural_network':
        p,saved,diag,members=fitted_neural(params,x,y,tr,te,s['nn_seeds'][r],np.ones(len(tr)),retain=True);diagnostics.append(dict(stage='outer',candidate=c,**diag))
    assert np.isfinite(p).all();joblib.dump(saved,root.with_suffix('.joblib'),compress=3)
    arrays=dict(train_rows=tr,test_rows=te,test_prediction=p,inner_validation_predictions=val,inner_fold_rmse=loss)
    if members is not None:arrays['member_test_predictions']=members
    np.savez_compressed(root.with_suffix('.npz'),**arrays)
    m=dict(family=family,panel=panel,repeat=r+1,outer_split_seed=int(s['seeds'][r]),inner_split_seed=int(s['inner_seeds'][r]),neural_seeds=s['nn_seeds'][r],model_seed=seed,
        train_n=len(tr),test_n=len(te),train_lineages=23,test_lineages=23,selected_candidate=c,selected_parameters=params,mean_inner_rmse=float(loss[c].mean()),inner_scaling=scaling,fit_diagnostics=diagnostics,
        reused=False,protocol_sha256=ph,seconds=time.monotonic()-start,npz_sha256=sha(root.with_suffix('.npz')))
    m['joblib_sha256']=sha(root.with_suffix('.joblib'))
    dump(root.with_suffix('.json'),m);return m

def prepare():
    OUT.mkdir(exist_ok=True);(OUT/'folds').mkdir(exist_ok=True);(OUT/'source_metadata').mkdir(exist_ok=True);sources=[];reuse=[]
    source=WS/'outputs/erastin_stratified_panel_comparison';known={v['path']:v['sha256'] for v in json.loads((source/'artifact_manifest.json').read_text())['files']}
    names=['panel_inputs.npz','shared_splits.npz','protocol.json','predictions.npz','summary.tsv','validation.json','split_lineage_counts.tsv','inner_split_lineage_counts.tsv','sample_test_coverage.tsv','model_helpers.py','weighted_mlp.py','ridge_helpers.py','metric_helpers.py','evaluate.py','make_splits.py']
    for name in names:
        assert sha(source/name)==known[name];sources.append(dict(path=str((source/name).relative_to(WS)),sha256=known[name]))
    selection=WS/'outputs/erastin_stratified_gene_importance';sknown={v['path']:v['sha256'] for v in json.loads((selection/'artifact_manifest.json').read_text())['files']}
    for name in ['protocol.json','summary.json','gene_importance_summary.tsv']:
        assert sha(selection/name)==sknown[name];sources.append(dict(path=str((selection/name).relative_to(WS)),sha256=sknown[name]))
    oldp=json.loads((source/'protocol.json').read_text());a=np.load(source/'panel_inputs.npz');spl=np.load(source/'shared_splits.npz');genes=list(a['genes_64']);indices=np.array([genes.index(g) for g in FIVE_GENES]);assert len(set(indices))==5
    arrays=dict(x_5=a['x_64'][:,indices],x_64=a['x_64'],genes_5=np.array(FIVE_GENES),genes_64=a['genes_64'],five_gene_indices_in_64=indices)
    for key in ['y','sample_ids','group_ids','lineage']:arrays[key]=a[key]
    path=OUT/'panel_inputs.npz'
    if path.exists():
        z=np.load(path)
        for key,value in arrays.items():np.testing.assert_array_equal(z[key],value)
    else:np.savez_compressed(path,**arrays)
    for name in ['shared_splits.npz','split_lineage_counts.tsv','inner_split_lineage_counts.tsv','sample_test_coverage.tsv']:
        dest=OUT/name
        if dest.exists():assert sha(dest)==sha(source/name)
        else:shutil.copyfile(source/name,dest)
    for name in ['model_helpers.py','weighted_mlp.py','ridge_helpers.py','metric_helpers.py','evaluate.py','make_splits.py']:assert sha(OUT/name)==sha(source/name)
    for family in FAMILIES:
        for r in range(20):
            for ext in ['npz','joblib','json']:
                rel=f'folds/{family}__p64__r{r+1:02d}.{ext}';assert sha(source/rel)==known[rel];sources.append(dict(path=str((source/rel).relative_to(WS)),sha256=known[rel]))
                dest=OUT/('source_metadata' if ext=='json' else 'folds')/f'{family}__p64__r{r+1:02d}.{ext}'
                if dest.exists():assert sha(dest)==known[rel]
                else:shutil.copyfile(source/rel,dest)
                reuse.append(dict(source=str((source/rel).relative_to(WS)),local=str(dest.relative_to(OUT)),sha256=known[rel]))
    p=dict(status='frozen_before_fixed_five_gene_fits',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),samples=744,identity_groups=733,lineages=23,excluded_samples=oldp['excluded_samples'],
        panels={'5':FIVE_GENES,'64':genes},five_gene_indices_in_64=indices,split_design=oldp['split_design'],inner_design=oldp['inner_design'],ampulla_handling=oldp['ampulla_handling'],
        outer_train_sizes=oldp['outer_train_sizes'],outer_test_sizes=oldp['outer_test_sizes'],split_seeds=spl['outer_split_seeds'],inner_split_seeds=spl['inner_split_seeds'],neural_seed_triplets=spl['neural_initialization_seeds'],
        ridge_alphas=oldp['ridge_alphas'],neural_grid=oldp['neural_grid'],neural_fixed=oldp['neural_fixed'],weighting=oldp['weighting'],tuning=oldp['tuning'],
        scaling='Training-only unweighted StandardScaler for ridge/NN features, target centering for ridge and training-only y standardization for NN.',
        controls='Byte-identical shared outer/inner split arrays and seed triplets. All other model settings and grids unchanged. Both panels separately tuned using inner training/validation only.',
        baseline='Reuse 20 verified 64-gene ridge and 20 verified 64-gene NN models. Fit both five-gene models from scratch on each outer training set, with separate inner hyperparameter selection; 40 new + 40 reused outer models.',
        feature_selection='The five genes were explicitly fixed by the user after prior held-out importance analysis of the same 20 splits. They are NOT selected independently within each outer training set.',
        selection_limitation='Earlier test responses influenced the choice of this five-gene panel. Reusing the same outer splits gives an exploratory post-selection comparison, not independent validation of feature selection. Inner hyperparameter tuning does not remove this selection bias; a fresh response cohort or prospectively reserved test set is required for confirmatory validation.',
        primary_comparisons=[['ridge__5','ridge__64'],['neural_network__5','neural_network__64']],
        secondary_comparisons=[['neural_network__5','ridge__5'],['neural_network__64','ridge__64'],['neural_network__5','ridge__64'],['ridge__5','neural_network__64']],
        primary_metrics=oldp['primary_metrics'],secondary_metrics=oldp['secondary_metrics'],metric_notes=oldp['metric_notes'],
        uncertainty=dict(count=5000,seed=2026110901,unit=oldp['uncertainty']['unit']+' These conditional intervals also do not adjust for the prior five-gene selection.'),
        equivalence='No equivalence/noninferiority margin or confirmatory equivalence test is defined. An interval crossing zero does not establish equivalence.',
        limitations=oldp['limitations'],new_outer_models=40,reused_outer_models=40,total_outer_models=80,workers=WORKERS,
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__),
        data_sha256={name:sha(OUT/name) for name in ['panel_inputs.npz','shared_splits.npz']},
        scripts={name:sha(OUT/name) for name in ['run_comparison.py','model_helpers.py','weighted_mlp.py','ridge_helpers.py','metric_helpers.py','evaluate.py','make_splits.py','validate_preflight.py']})
    path=OUT/'protocol.json'
    if path.exists():
        existing=json.loads(path.read_text());p['created_utc']=existing['created_utc'];assert existing==clean(p)
    else:dump(path,p)
    ph=sha(path);dump(OUT/'source_manifest.json',dict(files=sources));dump(OUT/'reuse_manifest.json',dict(files=reuse,source_protocol_sha256=sha(source/'protocol.json')))
    for family in FAMILIES:
        for r in range(20):
            original=json.loads((OUT/'source_metadata'/f'{family}__p64__r{r+1:02d}.json').read_text());assert original['family']==family and original['panel']==64 and original['protocol_sha256']==sha(source/'protocol.json')
            m=dict(original,reused=True,protocol_sha256=ph,reuse_source_protocol_sha256=original['protocol_sha256'],original_metadata_sha256=sha(OUT/'source_metadata'/f'{family}__p64__r{r+1:02d}.json'))
            dest=OUT/'folds'/f'{family}__p64__r{r+1:02d}.json'
            if dest.exists():assert json.loads(dest.read_text())==clean(m)
            else:dump(dest,m)
    return ph

def main():
    start=time.monotonic();ph=prepare();initialize();v=json.loads((OUT/'preflight_validation.json').read_text());assert v['passed'] and v['protocol_sha256']==ph
    with futures.ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        jobs=[(p,f,r) for r in range(20) for p in PANELS for f in FAMILIES if p==5];pending=[pool.submit(fit_job,*args,ph) for args in jobs];n=0;counts={}
        for future in futures.as_completed(pending):
            m=future.result();n+=1;arm=arm_name(m['family'],m['panel']);counts[arm]=counts.get(arm,0)+1;elapsed=time.monotonic()-start
            v=dict(phase='fitting',completed=n,total=40,reused=40,by_arm=counts,elapsed_seconds=elapsed,estimated_remaining_seconds=elapsed/n*(40-n));dump(OUT/'progress.json',v)
            if n%8==0:print(json.dumps(v),flush=True)
    p=json.loads((OUT/'protocol.json').read_text());assert sha(OUT/'protocol.json')==ph
    for k,h in p['scripts'].items():assert sha(OUT/k)==h
    for k,h in p['data_sha256'].items():assert sha(OUT/k)==h
    for a in json.loads((OUT/'source_manifest.json').read_text())['files']:assert sha(WS/a['path'])==a['sha256']
    dump(OUT/'fit_completion.json',dict(complete=True,new_outer_models=40,reused_outer_models=40,total_outer_models=80,elapsed_seconds=time.monotonic()-start));dump(OUT/'progress.json',dict(phase='fits_complete',completed=40,total=40,reused=40));print('ALL_FITS_COMPLETE',flush=True)
if __name__=='__main__':main()
