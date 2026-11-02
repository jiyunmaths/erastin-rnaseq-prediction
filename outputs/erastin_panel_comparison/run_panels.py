#!/usr/bin/env python3
"""Frozen 40/64/80-gene comparison using the exact prior nested splits."""
import concurrent.futures as futures
import importlib.util
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
import sklearn
import xgboost as xgb

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;WS=OUT.parent.parent

def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

xp=module('prior_xgboost',WS/'outputs/erastin_xgboost_comparison/run_comparison.py')
nn=module('prior_neural',WS/'outputs/erastin_nonlinear_comparison/run_models.py')
cm=module('prior_ridge',WS/'outputs/erastin_model_comparison/compare_models.py')
sha=xp.sha;dump=xp.dump
STATE={};FAMILIES=['ridge','neural_network','xgboost'];WORKERS=10

def initialize():
    a=np.load(OUT/'panel_inputs.npz');s=np.load(OUT/'shared_splits.npz')
    STATE.update(x={p:a['x_'+str(p)].astype(float) for p in [40,64,80]},
        genes={p:a['genes_'+str(p)] for p in [40,64,80]},y=a['y'],ids=a['sample_ids'],groups=a['group_ids'],lineage=a['lineage'],
        labels=s['holdout_labels'],inner=s['inner_fold_by_holdout'],seeds=s['inner_split_seeds'],nn_seeds=s['neural_initialization_seeds'])
    STATE['model_seeds']=np.array([int(v)%(2**31-1) for v in STATE['seeds']])
    STATE['alphas']=np.array(json.loads((OUT/'protocol.json').read_text())['ridge_alphas'])

def matrix(x,genes,y=None):return xgb.DMatrix(x,label=y,feature_names=list(genes),nthread=1)

def fit_job(panel,family,r,l,ph,replay=False):
    folder=OUT/('replay' if replay else 'folds');folder.mkdir(exist_ok=True)
    token=f'{family}__p{panel}__r{r+1:02d}__l{l:02d}';root=folder/token
    if root.with_suffix('.json').exists():
        m=json.loads(root.with_suffix('.json').read_text());assert m['protocol_sha256']==ph
        for ext in ['npz','ubj' if family=='xgboost' else 'joblib']:assert sha(root.with_suffix('.'+ext))==m[ext+'_sha256']
        return m
    begin=time.monotonic();x=STATE['x'][panel];genes=STATE['genes'][panel];y=STATE['y'];g=STATE['groups']
    inner=STATE['inner'][r,l];label=STATE['labels'][l];tr=np.flatnonzero(inner>=0);te=np.flatnonzero(STATE['lineage']==label)
    assert set(g[tr]).isdisjoint(g[te]) and label not in set(STATE['lineage'][tr])
    grid=([dict(alpha=float(a)) for a in STATE['alphas']] if family=='ridge' else nn.NN_GRID if family=='neural_network' else xp.GRID)
    val=np.full((len(grid),745),np.nan);loss=np.empty((len(grid),5));diagnostics=[]
    model_seed=int(STATE['model_seeds'][r]);nn.SEEDS=[int(v) for v in STATE['nn_seeds'][r]]
    scaling=[]
    for k in range(5):
        a=tr[inner[tr]!=k];b=tr[inner[tr]==k]
        assert set(a)|set(b)==set(tr) and set(g[a]).isdisjoint(g[b])
        scaling.append(dict(inner_fold=k,y_mean=float(y[a].mean()),y_scale=float(y[a].std())))
        if family=='ridge':
            pp,_=cm.ridge_path(x,y,a,b,STATE['alphas']);val[:,b]=pp.T
            loss[:,k]=np.sqrt(np.mean((pp-y[b,None])**2,axis=0))
        elif family=='neural_network':
            for c,params in enumerate(grid):
                p,_,diag,_=nn.fitted_prediction(family,params,x,y,a,b)
                val[c,b]=p;loss[c,k]=np.sqrt(np.mean((y[b]-p)**2))
                diagnostics.append(dict(stage='inner',candidate=c,inner_fold=k,**diag))
        else:
            mu,sd=scaling[-1]['y_mean'],scaling[-1]['y_scale']
            da=matrix(x[a],genes,(y[a]-mu)/sd);db=matrix(x[b],genes)
            for j,structure in enumerate(xp.STRUCTURES):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always');model=xgb.train(xp.parameters(structure,model_seed),da,num_boost_round=600)
                diagnostics.extend(dict(stage='inner',inner_fold=k,structure=j,warning_category=w.category.__name__,warning=str(w.message)) for w in caught)
                for q,n in enumerate(xp.ROUNDS):
                    c=j*len(xp.ROUNDS)+q;p=model.predict(db,iteration_range=(0,n)).astype(float)*sd+mu
                    val[c,b]=p;loss[c,k]=np.sqrt(np.mean((y[b]-p)**2))
    assert np.isfinite(val[:,tr]).all() and np.isnan(val[:,inner<0]).all()
    c=int(np.argmin(loss.mean(axis=1)));params=grid[c];member_predictions=None
    if family=='ridge':
        pp,saved=cm.ridge_path(x,y,tr,te,[params['alpha']],retain=True);p=pp[:,0]
    elif family=='neural_network':
        p,saved,diag,member_predictions=nn.fitted_prediction(family,params,x,y,tr,te,retain=True)
        diagnostics.append(dict(stage='outer',candidate=c,**diag))
    else:
        mu,sd=float(y[tr].mean()),float(y[tr].std())
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model=xgb.train(xp.parameters(params,model_seed),matrix(x[tr],genes,(y[tr]-mu)/sd),num_boost_round=params['num_boost_round'])
        diagnostics.extend(dict(stage='outer',warning_category=w.category.__name__,warning=str(w.message)) for w in caught)
        p=model.predict(matrix(x[te],genes)).astype(float)*sd+mu;saved=dict(y_mean=mu,y_scale=sd)
    assert np.isfinite(p).all()
    ext='ubj' if family=='xgboost' else 'joblib'
    if family=='xgboost':model.save_model(root.with_suffix('.ubj'))
    else:joblib.dump(saved,root.with_suffix('.joblib'),compress=3)
    arrays=dict(train_rows=tr,test_rows=te,inner_validation_predictions=val,inner_fold_rmse=loss,test_prediction=p)
    if member_predictions is not None:arrays['member_test_predictions']=member_predictions
    np.savez_compressed(root.with_suffix('.npz'),**arrays)
    meta=dict(panel=panel,family=family,repeat=r+1,lineage_index=l,lineage=str(label),inner_split_seed=int(STATE['seeds'][r]),
        model_seed=model_seed,neural_seeds=nn.SEEDS,train_n=len(tr),test_n=len(te),selected_candidate=c,selected_parameters=params,
        mean_inner_rmse=float(loss[c].mean()),inner_scaling=scaling,fit_diagnostics=diagnostics,
        protocol_sha256=ph,npz_sha256=sha(root.with_suffix('.npz')),seconds=time.monotonic()-begin)
    meta[ext+'_sha256']=sha(root.with_suffix('.'+ext))
    if family=='xgboost':meta.update(saved)
    dump(root.with_suffix('.json'),meta);return meta

def prepare():
    OUT.mkdir(exist_ok=True);sources=[]
    source_sets=[('erastin_panel_expansion',['panel_definitions.json','candidate_gene_evidence.tsv']),
        ('erastin_40gene_comparison',['comparison_inputs.npz','matched_depmap_expression.npz','comparison_config.json']),
        ('erastin_nn_repeated',['shared_splits.npz','predictions.npz','protocol.json','per_repeat_metrics.tsv']),
        ('erastin_xgboost_comparison',['run_comparison.py','predictions.npz','protocol.json','per_repeat_metrics.tsv']),
        ('erastin_nonlinear_comparison',['run_models.py']),('erastin_model_comparison',['compare_models.py']),
        ('erastin_bayesian_comparison',['review_bayesian.py']),('erastin_repeated_cv',['repeat_cv.py'])]
    for directory,names in source_sets:
        root=WS/'outputs'/directory
        manifest='result_artifact_manifest.json' if directory=='erastin_model_comparison' else 'artifact_manifest.json'
        known={r['path']:r['sha256'] for r in json.loads((root/manifest).read_text())['files']}
        for name in names:
            assert sha(root/name)==known[name],str(root/name)
            sources.append(dict(path=str((root/name).relative_to(WS)),sha256=known[name]))
    d=np.load(WS/'outputs/erastin_40gene_comparison/comparison_inputs.npz')
    exp=np.load(WS/'outputs/erastin_40gene_comparison/matched_depmap_expression.npz')
    s=np.load(WS/'outputs/erastin_nn_repeated/shared_splits.npz')
    definition=json.loads((WS/'outputs/erastin_panel_expansion/panel_definitions.json').read_text())
    for a in [exp,s]:np.testing.assert_array_equal(a['sample_ids'],d['sample_ids'])
    cols={c.split(' (')[0]:i for i,c in enumerate(exp['gene_columns'])};arrays={}
    for size,key in [(40,'original_40'),(64,'proposed_64'),(80,'proposed_80')]:
        genes=definition[key];assert len(genes)==len(set(genes))==size
        x=exp['X_log2tpm1'][:,[cols[g] for g in genes]].astype(float);assert np.isfinite(x).all()
        arrays['x_'+str(size)]=x;arrays['genes_'+str(size)]=np.array(genes)
    np.testing.assert_array_equal(arrays['x_40'],d['X_panel_log2tpm1'])
    np.testing.assert_array_equal(arrays['x_64'][:,:40],arrays['x_40']);np.testing.assert_array_equal(arrays['x_80'][:,:64],arrays['x_64'])
    nnprotocol=json.loads((WS/'outputs/erastin_nn_repeated/protocol.json').read_text())
    xprotocol=json.loads((WS/'outputs/erastin_xgboost_comparison/protocol.json').read_text())
    alphas=json.loads((WS/'outputs/erastin_40gene_comparison/comparison_config.json').read_text())['ridge_alphas']
    assert nn.NN_GRID==nnprotocol['nn_grid'] and xp.GRID==xprotocol['grid'] and xp.FIXED==xprotocol['fixed_parameters']
    protocol=dict(status='frozen_before_expanded_panel_fits',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        script_sha256=sha(Path(__file__)),panels={str(k):arrays['genes_'+str(k)] for k in [40,64,80]},samples=745,identity_groups=734,
        repeats=20,lineage_holdouts=24,inner_folds=5,split_seeds=s['inner_split_seeds'],neural_seed_triplets=s['neural_initialization_seeds'],
        xgboost_seeds=xprotocol['model_seeds'],ridge_alphas=alphas,neural_grid=nn.NN_GRID,neural_fixed=nnprotocol['nn_fixed_settings'],
        xgboost_grid=xp.GRID,xgboost_fixed=xp.FIXED,
        tuning='Minimum equal-weight mean five inner-fold RMSE; first grid member breaks ties. Each panel tuned independently.',
        controls='Identical samples, targets, nested folds, per-repeat initialization/sampling seeds, model settings and search grids across panels.',
        scaling='Ridge: train-only X scaler and target centering. NN: train-only X/y StandardScaler. XGBoost: unchanged features and train-only target mean/SD.',
        baseline='Reuse verified 40-gene predictions; six full nested replays check prior results for Lung/Thyroid in seed 1.',
        comparisons='Within each family: 64 vs 40, 80 vs 40, and 80 vs 64. Also cross-family contrasts at each size and expanded models vs the 40-gene NN.',
        uncertainty=dict(count=5000,pooled_seed=2026110201,macro_seed=2026110202,unit='Whole held-out lineage, paired across all nine arms and all repeats; no refitting.'),
        reporting='Pooled 745-line scores and equal-weight macro scores over 19 lineages with n>=10. All seeds retained. No test-based gene/grid changes.',
        limitations='Exploratory reuse of this cohort after earlier model comparisons. Some mechanistic literature also uses CTRP/CCLE. No independent validation or multiplicity adjustment.',
        new_outer_models=2880,baseline_replay_models=6,workers=WORKERS,
        versions=dict(python=platform.python_version(),numpy=np.__version__,sklearn=sklearn.__version__,xgboost=xgb.__version__))
    pp=OUT/'protocol.json'
    if pp.exists():
        old=json.loads(pp.read_text());protocol['created_utc']=old['created_utc'];assert old==xp.clean(protocol)
    else:dump(pp,protocol)
    dump(OUT/'source_manifest.json',dict(files=sources))
    shutil.copyfile(WS/'outputs/erastin_panel_expansion/panel_definitions.json',OUT/'panel_definitions.json')
    shutil.copyfile(WS/'outputs/erastin_nn_repeated/shared_splits.npz',OUT/'shared_splits.npz')
    np.savez_compressed(OUT/'panel_inputs.npz',**arrays,y=d['y_sensitivity'],sample_ids=d['sample_ids'],group_ids=d['group_ids'],lineage=d['lineage'])
    return sha(pp)

def main():
    start=time.monotonic();ph=prepare();initialize()
    print(json.dumps(dict(event='protocol_frozen',sha256=ph,new_outer_models=2880)),flush=True)
    baseline=np.load(WS/'outputs/erastin_xgboost_comparison/predictions.npz')
    replay_checks=[]
    with futures.ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        replay=[pool.submit(fit_job,40,f,0,list(STATE['labels']).index(l),ph,True) for f in FAMILIES for l in ['Lung','Thyroid']]
        for future in futures.as_completed(replay):
            m=future.result();token=f'{m["family"]}__p40__r01__l{m["lineage_index"]:02d}'
            a=np.load(OUT/'replay'/(token+'.npz'));e=float(np.max(np.abs(a['test_prediction']-baseline[m['family']][0,a['test_rows']])));assert e<1e-10
            replay_checks.append(dict(family=m['family'],lineage=m['lineage'],prediction_error=e))
        dump(OUT/'baseline_replay_validation.json',dict(passed=True,checks=replay_checks))
        print(json.dumps(dict(event='baseline_replays_passed',checks=replay_checks)),flush=True)
        jobs=[(p,f,r,l) for r in range(20) for l in range(24) for p in [64,80] for f in FAMILIES]
        pending={pool.submit(fit_job,*args,ph):args for args in jobs};completed=0;counts={}
        for future in futures.as_completed(pending):
            m=future.result();completed+=1;key=f'{m["family"]}__{m["panel"]}';counts[key]=counts.get(key,0)+1
            elapsed=time.monotonic()-start
            progress=dict(phase='fitting',completed=completed,total=2880,by_arm=counts,elapsed_seconds=elapsed,
                estimated_remaining_seconds=elapsed/completed*(2880-completed))
            dump(OUT/'progress.json',progress)
            if completed%48==0 or completed==2880:print(json.dumps(dict(event='fit_progress',**progress)),flush=True)
    assert sha(Path(__file__))==json.loads((OUT/'protocol.json').read_text())['script_sha256']
    assert sha(OUT/'protocol.json')==ph
    for source in json.loads((OUT/'source_manifest.json').read_text())['files']:assert sha(WS/source['path'])==source['sha256']
    dump(OUT/'fit_completion.json',dict(complete=True,new_outer_models=2880,baseline_replays=6,elapsed_seconds=time.monotonic()-start,sources_unchanged=True))
    dump(OUT/'progress.json',dict(phase='fits_complete',completed=2880,total=2880))
    print(json.dumps(dict(event='all_fits_complete',seconds=time.monotonic()-start)),flush=True)

if __name__=='__main__':main()
