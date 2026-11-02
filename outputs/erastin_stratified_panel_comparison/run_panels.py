#!/usr/bin/env python3
"""Unweighted 40/64/80-gene ridge, NN and XGBoost on shared stratified splits."""
import concurrent.futures as futures
import json,multiprocessing as mp,platform,shutil,sys,time,warnings
from pathlib import Path
import joblib,numpy as np,scipy,sklearn,xgboost as xgb
from model_helpers import sha,clean,dump,fitted_neural
from ridge_helpers import ridge_path
from make_splits import inner_rows
import xgb_helpers as xp
OUT=Path(__file__).resolve().parent;WS=OUT.parent.parent;FAMILIES=['ridge','neural_network','xgboost'];PANELS=[40,64,80];WORKERS=10;STATE={}
sys.dont_write_bytecode=True

def arm_name(f,p):return f+'__'+str(p)
def initialize():
    a=np.load(OUT/'panel_inputs.npz');s=np.load(OUT/'shared_splits.npz');p=json.loads((OUT/'protocol.json').read_text())
    STATE.update(x={n:a['x_'+str(n)] for n in PANELS},genes={n:a['genes_'+str(n)] for n in PANELS},y=a['y'],ids=a['sample_ids'],groups=a['group_ids'],lineage=a['lineage'],
        test=s['outer_test'],inner=s['inner_fold'],seeds=s['outer_split_seeds'],inner_seeds=s['inner_split_seeds'],nn_seeds=s['neural_initialization_seeds'],nn_grid=p['neural_grid'],alphas=np.array(p['ridge_alphas']))
    STATE['model_seeds']=np.array([int(v)%(2**31-1) for v in STATE['seeds']])

def matrix(x,genes,y=None):return xgb.DMatrix(x,label=y,feature_names=list(genes),nthread=1)
def grid_for(f):return [dict(alpha=float(v)) for v in STATE['alphas']] if f=='ridge' else STATE['nn_grid'] if f=='neural_network' else xp.GRID

def fit_job(panel,family,r,ph):
    root=OUT/'folds'/f'{family}__p{panel}__r{r+1:02d}'
    if root.with_suffix('.json').exists():
        m=json.loads(root.with_suffix('.json').read_text());assert m['protocol_sha256']==ph
        for ext in ['npz','ubj' if family=='xgboost' else 'joblib']:assert sha(root.with_suffix('.'+ext))==m[ext+'_sha256']
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
        else:
            da=matrix(x[a],genes,(y[a]-mu)/sd);db=matrix(x[b],genes)
            for j,structure in enumerate(xp.STRUCTURES):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');model=xgb.train(xp.parameters(structure,seed),da,num_boost_round=max(xp.ROUNDS))
                diagnostics.extend(dict(stage='inner',fold=k,structure=j,warning_category=v.category.__name__,warning=str(v.message)) for v in caught)
                for q,n in enumerate(xp.ROUNDS):
                    c=j*len(xp.ROUNDS)+q;p=model.predict(db,iteration_range=(0,n)).astype(float)*sd+mu;val[c,b]=p;loss[c,k]=np.sqrt(np.mean((y[b]-p)**2))
    assert np.isfinite(val[:,inner>=0]).all() and np.isnan(val[:,inner<0]).all();c=int(np.argmin(loss.mean(axis=1)));params=grid[c];members=None
    if family=='ridge':p,saved=ridge_path(x,y,tr,te,[params['alpha']],retain=True);p=p[:,0]
    elif family=='neural_network':
        p,saved,diag,members=fitted_neural(params,x,y,tr,te,s['nn_seeds'][r],np.ones(len(tr)),retain=True);diagnostics.append(dict(stage='outer',candidate=c,**diag))
    else:
        mu,sd=float(y[tr].mean()),float(y[tr].std())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model=xgb.train(xp.parameters(params,seed),matrix(x[tr],genes,(y[tr]-mu)/sd),num_boost_round=params['num_boost_round'])
        diagnostics.extend(dict(stage='outer',warning_category=v.category.__name__,warning=str(v.message)) for v in caught)
        p=model.predict(matrix(x[te],genes)).astype(float)*sd+mu;saved=dict(y_mean=mu,y_scale=sd)
    assert np.isfinite(p).all();ext='ubj' if family=='xgboost' else 'joblib'
    if family=='xgboost':model.save_model(root.with_suffix('.ubj'))
    else:joblib.dump(saved,root.with_suffix('.joblib'),compress=3)
    arrays=dict(train_rows=tr,test_rows=te,test_prediction=p,inner_validation_predictions=val,inner_fold_rmse=loss)
    if members is not None:arrays['member_test_predictions']=members
    np.savez_compressed(root.with_suffix('.npz'),**arrays)
    m=dict(family=family,panel=panel,repeat=r+1,outer_split_seed=int(s['seeds'][r]),inner_split_seed=int(s['inner_seeds'][r]),neural_seeds=s['nn_seeds'][r],model_seed=seed,
        train_n=len(tr),test_n=len(te),train_lineages=23,test_lineages=23,selected_candidate=c,selected_parameters=params,mean_inner_rmse=float(loss[c].mean()),inner_scaling=scaling,fit_diagnostics=diagnostics,
        reused=False,protocol_sha256=ph,seconds=time.monotonic()-start,npz_sha256=sha(root.with_suffix('.npz')))
    m[ext+'_sha256']=sha(root.with_suffix('.'+ext))
    if family=='xgboost':m.update(saved)
    dump(root.with_suffix('.json'),m);return m

def prepare():
    OUT.mkdir(exist_ok=True);(OUT/'folds').mkdir(exist_ok=True);(OUT/'source_metadata').mkdir(exist_ok=True);sources=[];reuse=[]
    for directory,names in [('erastin_panel_comparison',['panel_inputs.npz','protocol.json']),
                            ('erastin_lineage_stratified_nn',['inputs.npz','shared_splits.npz','protocol.json','predictions.npz','summary.tsv','split_lineage_counts.tsv','inner_split_lineage_counts.tsv','sample_test_coverage.tsv','model_helpers.py','metric_helpers.py','evaluate.py','make_splits.py','weighted_mlp.py']),
                            ('erastin_model_comparison',['compare_models.py']),('erastin_xgboost_comparison',['run_comparison.py'])]:
        root=WS/'outputs'/directory;manifest='result_artifact_manifest.json' if directory=='erastin_model_comparison' else 'artifact_manifest.json';known={v['path']:v['sha256'] for v in json.loads((root/manifest).read_text())['files']}
        for name in names:
            assert sha(root/name)==known[name];sources.append(dict(path=str((root/name).relative_to(WS)),sha256=known[name]))
    source=WS/'outputs/erastin_lineage_stratified_nn';prior=json.loads((source/'protocol.json').read_text());base=WS/'outputs/erastin_panel_comparison';oldp=json.loads((base/'protocol.json').read_text());a=np.load(base/'panel_inputs.npz');d=np.load(source/'inputs.npz');spl=np.load(source/'shared_splits.npz');keep=a['lineage']!='Cervix'
    for k in ['y','sample_ids','group_ids','lineage']:np.testing.assert_array_equal(a[k][keep],d[k]);np.testing.assert_array_equal(spl[k] if k in spl.files else d[k],d[k])
    arrays={k:a[k][keep] if k.startswith('x_') else a[k] for k in ['x_40','x_64','x_80','genes_40','genes_64','genes_80']}
    for k in ['y','sample_ids','group_ids','lineage']:arrays[k]=d[k]
    np.testing.assert_array_equal(arrays['x_64'],d['x']);np.testing.assert_array_equal(arrays['x_64'][:,:40],arrays['x_40']);np.testing.assert_array_equal(arrays['x_80'][:,:64],arrays['x_64'])
    path=OUT/'panel_inputs.npz'
    if path.exists():
        z=np.load(path)
        for k,v in arrays.items():np.testing.assert_array_equal(z[k],v)
    else:np.savez_compressed(path,**arrays)
    for name in ['shared_splits.npz','split_lineage_counts.tsv','inner_split_lineage_counts.tsv','sample_test_coverage.tsv']:
        path=OUT/name
        if path.exists():assert sha(path)==sha(source/name)
        else:shutil.copyfile(source/name,path)
    for name in ['model_helpers.py','metric_helpers.py','evaluate.py','make_splits.py','weighted_mlp.py']:assert sha(OUT/name)==sha(source/name)
    assert xp.GRID==oldp['xgboost_grid'] and xp.FIXED==oldp['xgboost_fixed']
    known={v['path']:v['sha256'] for v in json.loads((source/'artifact_manifest.json').read_text())['files']}
    for r in range(20):
        for ext in ['npz','joblib','json']:
            rel=f'folds/unweighted__r{r+1:02d}.{ext}';assert sha(source/rel)==known[rel];sources.append(dict(path=str((source/rel).relative_to(WS)),sha256=known[rel]))
            dest=OUT/('source_metadata' if ext=='json' else 'folds')/f'neural_network__p64__r{r+1:02d}.{ext}'
            if dest.exists():assert sha(dest)==known[rel]
            else:shutil.copyfile(source/rel,dest)
            reuse.append(dict(source=str((source/rel).relative_to(WS)),local=str(dest.relative_to(OUT)),sha256=known[rel]))
    neural_fixed={k:v for k,v in prior['neural_fixed'].items() if k!='initialization_seeds'}
    p=dict(status='frozen_before_stratified_panel_fits',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),samples=744,identity_groups=733,lineages=23,excluded_samples=prior['excluded_samples'],
        panels={str(n):arrays['genes_'+str(n)] for n in PANELS},split_design=prior['outer_design'],inner_design=prior['inner_design'],ampulla_handling=prior['ampulla_handling'],
        outer_train_sizes=prior['outer_train_sizes'],outer_test_sizes=prior['outer_test_sizes'],split_seeds=spl['outer_split_seeds'],inner_split_seeds=spl['inner_split_seeds'],neural_seed_triplets=spl['neural_initialization_seeds'],
        xgboost_seeds=[int(v)%(2**31-1) for v in spl['outer_split_seeds']],ridge_alphas=oldp['ridge_alphas'],neural_grid=prior['neural_grid'],neural_fixed=neural_fixed,xgboost_grid=xp.GRID,xgboost_fixed=xp.FIXED,
        weighting='Unweighted training in every model and every fold. NN adapter receives an exact all-ones sample-weight vector and is checked against native MLPRegressor.',
        tuning='Original model-specific grids; minimum unweighted equal-weight mean of five inner-fold RMSEs; original candidate order breaks ties. Each arm tuned independently.',
        scaling='Training-only unweighted StandardScaler for ridge/NN features; training-only y standardization for NN/XGBoost, target centering for ridge. Unchanged raw log2(TPM+1) features for XGBoost.',
        controls='Byte-identical saved 20 outer/inner split arrays across all nine arms, identical samples/targets/genes within a panel, fixed family grids/settings and matched initialization/sampling seeds.',
        baseline='Reuse all 20 verified unweighted 64-gene NN models with exact input/split equality and native-MLP nested replay checks in repeats 1/20. Refit eight other arms, 160 new outer models; nine arms total 180.',
        primary_comparisons='Within each family: 64 vs 40, 80 vs 40, 80 vs 64.',secondary_comparisons='Within each panel: NN vs ridge, XGBoost vs ridge, XGBoost vs NN. Expanded ridge/XGBoost vs 40-gene NN.',
        primary_metrics=['pooled_spearman','pooled_rmse','pooled_r2','macro23_rmse'],secondary_metrics=['macro19_rmse'],
        metric_notes=prior['metric_notes'],uncertainty=dict(count=5000,seed=2026110701,unit=prior['uncertainty']['unit']),
        limitations=prior['limitations']+' Panel/model comparisons reuse this cohort after earlier exploration; no multiplicity adjustment or independent external validation.',
        new_outer_models=160,reused_outer_models=20,total_outer_models=180,workers=WORKERS,
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__,xgboost=xgb.__version__),
        data_sha256={name:sha(OUT/name) for name in ['panel_inputs.npz','shared_splits.npz']},
        scripts={name:sha(OUT/name) for name in ['run_panels.py','ridge_helpers.py','xgb_helpers.py','model_helpers.py','weighted_mlp.py','metric_helpers.py','evaluate.py','make_splits.py','validate_preflight.py']})
    path=OUT/'protocol.json'
    if path.exists():
        existing=json.loads(path.read_text());p['created_utc']=existing['created_utc'];assert existing==clean(p)
    else:dump(path,p)
    ph=sha(path);dump(OUT/'source_manifest.json',dict(files=sources));dump(OUT/'reuse_manifest.json',dict(files=reuse,original_protocol_sha256=sha(source/'protocol.json')))
    for r in range(20):
        original=json.loads((OUT/'source_metadata'/f'neural_network__p64__r{r+1:02d}.json').read_text());assert original['arm']=='unweighted' and original['protocol_sha256']==sha(source/'protocol.json')
        m=dict(original,family='neural_network',panel=64,reused=True,model_seed=int(spl['outer_split_seeds'][r])%(2**31-1),protocol_sha256=ph,original_protocol_sha256=original['protocol_sha256'],original_metadata_sha256=sha(OUT/'source_metadata'/f'neural_network__p64__r{r+1:02d}.json'))
        dest=OUT/'folds'/f'neural_network__p64__r{r+1:02d}.json'
        if dest.exists():assert json.loads(dest.read_text())==clean(m)
        else:dump(dest,m)
    return ph

def main():
    start=time.monotonic();ph=prepare();initialize();v=json.loads((OUT/'preflight_validation.json').read_text());assert v['passed'] and v['protocol_sha256']==ph
    with futures.ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        jobs=[(p,f,r) for r in range(20) for p in PANELS for f in FAMILIES if not(p==64 and f=='neural_network')];pending=[pool.submit(fit_job,*args,ph) for args in jobs];n=0;counts={}
        for future in futures.as_completed(pending):
            m=future.result();n+=1;arm=arm_name(m['family'],m['panel']);counts[arm]=counts.get(arm,0)+1;elapsed=time.monotonic()-start
            v=dict(phase='fitting',completed=n,total=160,reused=20,by_arm=counts,elapsed_seconds=elapsed,estimated_remaining_seconds=elapsed/n*(160-n));dump(OUT/'progress.json',v)
            if n%8==0:print(json.dumps(v),flush=True)
    p=json.loads((OUT/'protocol.json').read_text());assert sha(OUT/'protocol.json')==ph
    for k,h in p['scripts'].items():assert sha(OUT/k)==h
    for k,h in p['data_sha256'].items():assert sha(OUT/k)==h
    for a in json.loads((OUT/'source_manifest.json').read_text())['files']:assert sha(WS/a['path'])==a['sha256']
    dump(OUT/'fit_completion.json',dict(complete=True,new_outer_models=160,reused_outer_models=20,total_outer_models=180,elapsed_seconds=time.monotonic()-start));dump(OUT/'progress.json',dict(phase='fits_complete',completed=160,total=160,reused=20));print('ALL_FITS_COMPLETE',flush=True)
if __name__=='__main__':main()
