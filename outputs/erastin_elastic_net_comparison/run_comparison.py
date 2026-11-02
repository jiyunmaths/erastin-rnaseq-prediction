#!/usr/bin/env python3
"""Elastic Net on both fixed panels, compared with the saved ridge/NN arms."""
import concurrent.futures as futures
import json,multiprocessing as mp,platform,shutil,sys,time
from pathlib import Path
import joblib,numpy as np,scipy,sklearn
from model_helpers import sha,clean,dump
from enet_helpers import GRID,L1_RATIOS,ALPHAS,MAX_ITER,TOL,fit_path,fit_selected
from make_splits import inner_rows
OUT=Path(__file__).resolve().parent;WS=OUT.parent.parent;PANELS=[5,64];FAMILIES=['ridge','elastic_net','neural_network'];BASELINES=['ridge','neural_network'];WORKERS=8;STATE={}
sys.dont_write_bytecode=True

def arm_name(f,p):return f+'__'+str(p)
def initialize():
    a=np.load(OUT/'panel_inputs.npz');s=np.load(OUT/'shared_splits.npz');p=json.loads((OUT/'protocol.json').read_text())
    STATE.update(x={n:a['x_'+str(n)] for n in PANELS},genes={n:a['genes_'+str(n)] for n in PANELS},y=a['y'],ids=a['sample_ids'],groups=a['group_ids'],lineage=a['lineage'],
        test=s['outer_test'],inner=s['inner_fold'],seeds=s['outer_split_seeds'],inner_seeds=s['inner_split_seeds'],nn_seeds=s['neural_initialization_seeds'],alphas=np.array(p['ridge_alphas']),nn_grid=p['neural_grid'])
    STATE['model_seeds']=np.array([int(v)%(2**31-1) for v in STATE['seeds']])

def grid_for(f):return GRID if f=='elastic_net' else [dict(alpha=float(v)) for v in STATE['alphas']] if f=='ridge' else STATE['nn_grid']

def fit_job(panel,r,ph):
    root=OUT/'folds'/f'elastic_net__p{panel}__r{r+1:02d}'
    if root.with_suffix('.json').exists():
        m=json.load(open(root.with_suffix('.json')));assert m['protocol_sha256']==ph
        for ext in ['npz','joblib']:assert sha(root.with_suffix('.'+ext))==m[ext+'_sha256']
        return m
    begin=time.monotonic();s=STATE;x=s['x'][panel];y=s['y'];g=s['groups'];lin=s['lineage'];inner=s['inner'][r];tr=np.flatnonzero(~s['test'][r]);te=np.flatnonzero(s['test'][r])
    assert set(g[tr]).isdisjoint(g[te]) and set(lin[tr])==set(lin[te])==set(lin)
    val=np.full((len(GRID),len(y)),np.nan);loss=np.empty((len(GRID),5));scaling=[];diagnostics=[]
    for k in range(5):
        a,b=inner_rows(inner,k);assert set(a)|set(b)==set(tr) and set(g[a]).isdisjoint(g[b]) and set(lin[a])==set(lin)
        pred,coef,diag,scale=fit_path(x,y,a,b);val[:,b]=pred;loss[:,k]=np.sqrt(np.mean((pred-y[b][None,:])**2,axis=1));scaling.append(dict(fold=k,**scale));diagnostics.extend(dict(stage='inner',fold=k,**v) for v in diag)
    assert np.isfinite(val[:,inner>=0]).all() and np.isnan(val[:,inner<0]).all()
    c=int(np.argmin(loss.mean(axis=1)));params=GRID[c];pred,saved,diag=fit_selected(x,y,tr,te,params);diagnostics.append(dict(stage='outer',**diag));assert np.isfinite(pred).all()
    joblib.dump(saved,root.with_suffix('.joblib'),compress=3);np.savez_compressed(root.with_suffix('.npz'),train_rows=tr,test_rows=te,test_prediction=pred,inner_validation_predictions=val,inner_fold_rmse=loss)
    m=dict(family='elastic_net',panel=panel,repeat=r+1,outer_split_seed=int(s['seeds'][r]),inner_split_seed=int(s['inner_seeds'][r]),neural_seeds=s['nn_seeds'][r],model_seed=int(s['model_seeds'][r]),
        train_n=len(tr),test_n=len(te),train_lineages=23,test_lineages=23,selected_candidate=c,selected_parameters=params,mean_inner_rmse=float(loss[c].mean()),inner_scaling=scaling,fit_diagnostics=diagnostics,
        reused=False,nonzero_genes=diag['nonzero_genes'],protocol_sha256=ph,seconds=time.monotonic()-begin,npz_sha256=sha(root.with_suffix('.npz')),joblib_sha256=sha(root.with_suffix('.joblib')))
    dump(root.with_suffix('.json'),m);return m

def prepare():
    OUT.mkdir(exist_ok=True);(OUT/'folds').mkdir(exist_ok=True);(OUT/'source_metadata').mkdir(exist_ok=True);source=WS/'outputs/erastin_five_gene_comparison';known={v['path']:v['sha256'] for v in json.load(open(source/'artifact_manifest.json'))['files']};sources=[];reuse=[]
    for name in ['panel_inputs.npz','shared_splits.npz','protocol.json','predictions.npz','summary.tsv','validation.json','split_lineage_counts.tsv','inner_split_lineage_counts.tsv','sample_test_coverage.tsv','model_helpers.py','weighted_mlp.py','metric_helpers.py','evaluate.py','make_splits.py']:
        assert sha(source/name)==known[name];sources.append(dict(path=str((source/name).relative_to(WS)),sha256=known[name]))
    for name in ['panel_inputs.npz','shared_splits.npz','split_lineage_counts.tsv','inner_split_lineage_counts.tsv','sample_test_coverage.tsv']:
        dest=OUT/name
        if dest.exists():assert sha(dest)==known[name]
        else:shutil.copyfile(source/name,dest)
    for name in ['model_helpers.py','weighted_mlp.py','metric_helpers.py','evaluate.py','make_splits.py']:assert sha(OUT/name)==known[name]
    for f in BASELINES:
        for panel in PANELS:
            for r in range(20):
                for ext in ['json','npz','joblib']:
                    rel=f'folds/{f}__p{panel}__r{r+1:02d}.{ext}';assert sha(source/rel)==known[rel];sources.append(dict(path=str((source/rel).relative_to(WS)),sha256=known[rel]));dest=OUT/('source_metadata' if ext=='json' else 'folds')/f'{f}__p{panel}__r{r+1:02d}.{ext}'
                    if dest.exists():assert sha(dest)==known[rel]
                    else:shutil.copyfile(source/rel,dest)
                    reuse.append(dict(source=str((source/rel).relative_to(WS)),local=str(dest.relative_to(OUT)),sha256=known[rel]))
    old=json.load(open(source/'protocol.json'));spl=np.load(OUT/'shared_splits.npz');a=np.load(OUT/'panel_inputs.npz')
    p=dict(status='frozen_before_elastic_net_fits',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),samples=744,identity_groups=733,lineages=23,excluded_samples=old['excluded_samples'],panels=old['panels'],
        split_design=old['split_design'],inner_design=old['inner_design'],ampulla_handling=old['ampulla_handling'],outer_train_sizes=old['outer_train_sizes'],outer_test_sizes=old['outer_test_sizes'],
        split_seeds=spl['outer_split_seeds'],inner_split_seeds=spl['inner_split_seeds'],neural_seed_triplets=spl['neural_initialization_seeds'],ridge_alphas=old['ridge_alphas'],neural_grid=old['neural_grid'],neural_fixed=old['neural_fixed'],
        elastic_net_grid=GRID,elastic_net_l1_ratios=L1_RATIOS,elastic_net_alphas=ALPHAS,elastic_net_fixed=dict(max_iter=MAX_ITER,tol=TOL,selection='cyclic',positive=False,fit_intercept=True),
        objective='On training-standardized X/y: squared error/(2*n) + alpha*l1_ratio*sum(abs(beta)) + alpha*(1-l1_ratio)*sum(beta^2)/2. Native unpenalized intercept in outer fit. Inner paths use exactly centered X/y.',
        elastic_net_tuning='328 pre-fixed candidates (41 explicit alphas from 10 to 1e-4, descending, times 8 L1 ratios). Mean of five unweighted inner-fold RMSEs in original target units; first grid candidate breaks ties. All preprocessing fit on each current inner training subset. No ElasticNetCV outside a scaling pipeline.',
        paths='Exact warm-start coordinate-descent paths per L1 ratio for inner candidates. Explicit training-only Gram/Xy matrices. Selected outer models refit with native ElasticNet. Path/native predictions and augmented-Lasso equivalence checked independently.',
        weighting='Unweighted for all six arms.',scaling='X/y StandardScaler fitted only on current training subset for Elastic Net. Baseline ridge/NN preprocessing unchanged. Alpha values are not numerically comparable to Ridge alpha because objective normalization and y scaling differ.',
        sparsity='Nonzero coefficient count is the count of literal nonzero standardized-target coefficients. Report per-split counts, selected genes and coefficient stability; selection frequency is not a causal or significance measure.',
        baseline='Reuse all 80 verified ridge/NN five- and 64-gene outer models/predictions; fit 40 new Elastic Net outer models under byte-identical splits.',
        primary_comparisons=[['elastic_net__5','ridge__5'],['elastic_net__64','ridge__64'],['elastic_net__5','neural_network__5'],['elastic_net__64','neural_network__64'],['elastic_net__5','elastic_net__64']],
        secondary_comparisons=[['elastic_net__5','ridge__64'],['elastic_net__5','neural_network__64'],['elastic_net__64','ridge__5'],['elastic_net__64','neural_network__5']],
        primary_metrics=old['primary_metrics'],secondary_metrics=old['secondary_metrics'],metric_notes=old['metric_notes'],feature_selection=old['feature_selection'],selection_limitation=old['selection_limitation'],
        validation_scope='Elastic Net variable selection uses only training data and inner tuning. The fixed five-gene arm still inherits earlier test-informed panel selection; the existing cohort has informed multiple model choices. Results remain exploratory.',
        uncertainty=dict(count=5000,seed=2026111001,unit=old['uncertainty']['unit']),limitations=old['limitations'],
        new_outer_models=40,reused_outer_models=80,total_outer_models=120,workers=WORKERS,
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__),
        data_sha256={name:sha(OUT/name) for name in ['panel_inputs.npz','shared_splits.npz']},scripts={name:sha(OUT/name) for name in ['run_comparison.py','enet_helpers.py','model_helpers.py','weighted_mlp.py','metric_helpers.py','evaluate.py','make_splits.py','validate_preflight.py']})
    pp=OUT/'protocol.json'
    if pp.exists():previous=json.load(open(pp));p['created_utc']=previous['created_utc'];assert previous==clean(p)
    else:dump(pp,p)
    ph=sha(pp);dump(OUT/'source_manifest.json',dict(files=sources));dump(OUT/'reuse_manifest.json',dict(files=reuse,source_protocol_sha256=sha(source/'protocol.json')))
    for f in BASELINES:
        for panel in PANELS:
            for r in range(20):
                src=OUT/'source_metadata'/f'{f}__p{panel}__r{r+1:02d}.json';original=json.load(open(src));assert original['protocol_sha256']==sha(source/'protocol.json')
                m=dict(original,reused=True,protocol_sha256=ph,reuse_source_protocol_sha256=original['protocol_sha256'],original_metadata_sha256=sha(src));dest=OUT/'folds'/f'{f}__p{panel}__r{r+1:02d}.json'
                if dest.exists():assert json.load(open(dest))==clean(m)
                else:dump(dest,m)
    return ph

def main():
    start=time.monotonic();ph=prepare();initialize();v=json.load(open(OUT/'preflight_validation.json'));assert v['passed'] and v['protocol_sha256']==ph
    with futures.ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        jobs=[pool.submit(fit_job,p,r,ph) for r in range(20) for p in PANELS];n=0
        for future in futures.as_completed(jobs):
            m=future.result();n+=1;elapsed=time.monotonic()-start;v=dict(phase='fitting',completed=n,total=40,reused=80,elapsed_seconds=elapsed,estimated_remaining_seconds=elapsed/n*(40-n));dump(OUT/'progress.json',v)
            if n%4==0:print(json.dumps(v),flush=True)
    p=json.load(open(OUT/'protocol.json'));assert sha(OUT/'protocol.json')==ph
    for name,h in p['scripts'].items():assert sha(OUT/name)==h
    for name,h in p['data_sha256'].items():assert sha(OUT/name)==h
    for a in json.load(open(OUT/'source_manifest.json'))['files']:assert sha(WS/a['path'])==a['sha256']
    dump(OUT/'fit_completion.json',dict(complete=True,new_outer_models=40,reused_outer_models=80,total_outer_models=120,elapsed_seconds=time.monotonic()-start));dump(OUT/'progress.json',dict(phase='fits_complete',completed=40,total=40,reused=80));print('ALL_FITS_COMPLETE',flush=True)
if __name__=='__main__':main()
