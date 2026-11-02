#!/usr/bin/env python3
"""64-gene NN weighting comparison with every lineage in train and test."""
import concurrent.futures as futures
import json,multiprocessing as mp,platform,sys,time
from pathlib import Path
import joblib,numpy as np,scipy,sklearn
from model_helpers import sha,clean,dump,fitted_neural
from lineage_weights import weights_for,weight_audit
from make_splits import outer_split,inner_folds,inner_rows
OUT=Path(__file__).resolve().parent;WS=OUT.parent.parent;STATE={};ARMS=['unweighted','inverse_sqrt','capped_inverse_5','inverse_frequency'];WORKERS=10
sys.dont_write_bytecode=True

def initialize():
    a=np.load(OUT/'inputs.npz');s=np.load(OUT/'shared_splits.npz');p=json.loads((OUT/'protocol.json').read_text())
    STATE.update(x=a['x'],y=a['y'],genes=a['genes'],ids=a['sample_ids'],groups=a['group_ids'],lineage=a['lineage'],
        test=s['outer_test'],inner=s['inner_fold'],seeds=s['outer_split_seeds'],inner_seeds=s['inner_split_seeds'],nn_seeds=s['neural_initialization_seeds'],grid=p['neural_grid'])

def fit_job(arm,r,ph):
    root=OUT/'folds'/f'{arm}__r{r+1:02d}'
    if root.with_suffix('.json').exists():
        m=json.loads(root.with_suffix('.json').read_text());assert m['protocol_sha256']==ph
        for ext in ['npz','joblib']:assert sha(root.with_suffix('.'+ext))==m[ext+'_sha256']
        return m
    begin=time.monotonic();s=STATE;x=s['x'];y=s['y'];g=s['groups'];lineage=s['lineage'];inner=s['inner'][r]
    tr=np.flatnonzero(~s['test'][r]);te=np.flatnonzero(s['test'][r]);assert set(g[tr]).isdisjoint(g[te]) and set(lineage[tr])==set(lineage[te])==set(lineage)
    grid=s['grid'];val=np.full((len(grid),len(y)),np.nan);loss=np.empty((len(grid),5));inner_w=np.full((5,len(y)),np.nan);scaling=[];diagnostics=[]
    for k in range(5):
        a,b=inner_rows(inner,k);assert set(a)|set(b)==set(tr) and set(g[a]).isdisjoint(g[b]) and set(lineage[a])==set(lineage)
        w=weights_for(lineage[a],arm);inner_w[k,a]=w;scaling.append(dict(fold=k,y_mean=float(y[a].mean()),y_scale=float(y[a].std()),weights=weight_audit(lineage[a],arm)))
        for c,params in enumerate(grid):
            p,_,diag,_=fitted_neural(params,x,y,a,b,s['nn_seeds'][r],w);val[c,b]=p;loss[c,k]=np.sqrt(np.mean((y[b]-p)**2));diagnostics.append(dict(stage='inner',candidate=c,fold=k,**diag))
    assert np.isfinite(val[:,inner>=0]).all() and np.isnan(val[:,inner<0]).all()
    c=int(np.argmin(loss.mean(axis=1)));w=weights_for(lineage[tr],arm);p,saved,diag,member=fitted_neural(grid[c],x,y,tr,te,s['nn_seeds'][r],w,retain=True);diagnostics.append(dict(stage='outer',candidate=c,**diag))
    joblib.dump(saved,root.with_suffix('.joblib'),compress=3)
    np.savez_compressed(root.with_suffix('.npz'),train_rows=tr,test_rows=te,test_prediction=p,member_test_predictions=member,inner_validation_predictions=val,inner_fold_rmse=loss,outer_training_weights=w,inner_training_weights=inner_w)
    m=dict(arm=arm,repeat=r+1,panel=64,outer_split_seed=int(s['seeds'][r]),inner_split_seed=int(s['inner_seeds'][r]),neural_seeds=s['nn_seeds'][r],
        train_n=len(tr),test_n=len(te),train_lineages=len(np.unique(lineage[tr])),test_lineages=len(np.unique(lineage[te])),selected_candidate=c,selected_parameters=grid[c],mean_inner_rmse=float(loss[c].mean()),
        inner_scaling=scaling,outer_weights=weight_audit(lineage[tr],arm),fit_diagnostics=diagnostics,seconds=time.monotonic()-begin,protocol_sha256=ph,npz_sha256=sha(root.with_suffix('.npz')),joblib_sha256=sha(root.with_suffix('.joblib')))
    dump(root.with_suffix('.json'),m);return m

def prepare():
    OUT.mkdir(exist_ok=True);(OUT/'folds').mkdir(exist_ok=True);source=WS/'outputs/erastin_nn_moderate_weights';known={v['path']:v['sha256'] for v in json.loads((source/'artifact_manifest.json').read_text())['files']};sources=[]
    for name in ['inputs.npz','shared_splits.npz','protocol.json','run_moderate.py','weighted_mlp.py','lineage_weights.py','evaluation.py','preflight_validation.json']:
        assert sha(source/name)==known[name];sources.append(dict(path=str((source/name).relative_to(WS)),sha256=known[name]))
    a=np.load(source/'inputs.npz');old=np.load(source/'shared_splits.npz');prior=json.loads((source/'protocol.json').read_text());keep=a['lineage']!='Cervix';lineage=a['lineage'][keep];groups=a['group_ids'][keep]
    assert len(lineage)==744 and len(np.unique(groups))==733 and len(np.unique(lineage))==23
    outer_seeds=old['inner_split_seeds'];inner_seeds=np.array([np.random.SeedSequence([int(v),20261106,1]).generate_state(1)[0] for v in outer_seeds],dtype=np.uint32)
    tests=np.array([outer_split(lineage,groups,seed) for seed in outer_seeds]);inner=np.array([inner_folds(lineage,groups,mask,seed) for mask,seed in zip(tests,inner_seeds)])
    inp=dict(x=a['x'][keep],y=a['y'][keep],genes=a['genes'],sample_ids=a['sample_ids'][keep],group_ids=groups,lineage=lineage,original_rows=np.flatnonzero(keep))
    splits=dict(sample_ids=inp['sample_ids'],group_ids=groups,lineage=lineage,outer_test=tests,inner_fold=inner,outer_split_seeds=outer_seeds,inner_split_seeds=inner_seeds,neural_initialization_seeds=old['neural_initialization_seeds'])
    for name,data in [('inputs.npz',inp),('shared_splits.npz',splits)]:
        path=OUT/name
        if path.exists():
            z=np.load(path);assert set(z.files)==set(data)
            for k,v in data.items():np.testing.assert_array_equal(z[k],v)
        else:np.savez_compressed(path,**data)
    rows=[];inner_counts=[]
    for r,mask in enumerate(tests):
        for label in np.unique(lineage):
            ix=lineage==label;rows.append(dict(repeat=r+1,lineage=str(label),cohort_n=int(ix.sum()),train_n=int(np.sum(ix&~mask)),test_n=int(np.sum(ix&mask)),train_identity_groups=len(np.unique(groups[ix&~mask])),test_identity_groups=len(np.unique(groups[ix&mask]))))
        for k in range(5):
            aa,bb=inner_rows(inner[r],k)
            for label in np.unique(lineage):inner_counts.append(dict(repeat=r+1,inner_fold=k,lineage=str(label),train_n=int(np.sum(lineage[aa]==label)),validation_n=int(np.sum(lineage[bb]==label))))
    import pandas as pd
    pd.DataFrame(rows).to_csv(OUT/'split_lineage_counts.tsv',sep='\t',index=False);pd.DataFrame(inner_counts).to_csv(OUT/'inner_split_lineage_counts.tsv',sep='\t',index=False)
    coverage=np.sum(tests,axis=0);pd.DataFrame(dict(sample_id=inp['sample_ids'],group_id=groups,lineage=lineage,test_repeats=coverage)).to_csv(OUT/'sample_test_coverage.tsv',sep='\t',index=False)
    p=dict(status='frozen_before_lineage_stratified_fits',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        samples=744,identity_groups=733,lineages=23,excluded_samples=[dict(sample_id=str(v),lineage='Cervix',reason='Singleton lineage cannot occur in both disjoint sets.') for v in a['sample_ids'][~keep]],
        outer_design='20 response-blind lineage-stratified identity-group holdouts. Approximately 80/20 per lineage, nearest feasible rounded sample count, at least one identity group on each side. Uniform random group subsets conditional on each sample-count target.',
        split_seeds=outer_seeds,inner_split_seeds=inner_seeds,neural_seed_triplets=old['neural_initialization_seeds'],
        inner_design='Five identity-group-stratified validation folds within outer training. A lineage represented by one training identity group is always inner training (-2) and never inner validation. Thus all 23 lineages remain in every inner training set; validation has 22 lineages because Ampulla has only one outer-training line.',
        ampulla_handling='Two total lines: one outer train and one outer test in every repeat. Outer-training Ampulla is present in every inner fit but not inner validation; outer test is never used for tuning.',
        cohort_lineage_counts={str(v):int(np.sum(lineage==v)) for v in np.unique(lineage)},outer_train_sizes=[int((~v).sum()) for v in tests],outer_test_sizes=[int(v.sum()) for v in tests],
        genes=list(a['genes']),neural_grid=prior['neural_grid'],neural_fixed=prior['neural_fixed'],arms=ARMS,
        weights='Unweighted; inverse square root; capped inverse frequency with FINAL maximum 5 and mean one; full inverse frequency. All weights recomputed from current fitting rows and normalized to mean one.',
        tuning='Minimum unweighted equal-mean five inner-fold RMSE, original four candidates and tie order; each arm tuned separately.',
        scaling='Unweighted StandardScaler for X/y fit only on current training rows, identical NN settings and initialization seed triplets across arms.',
        baseline='All four arms newly fitted under the same new splits. Previous leave-lineage-out scores/predictions are not reused or directly pooled.',
        primary_metrics=['pooled_spearman','pooled_rmse','pooled_r2','macro23_rmse'],secondary_metrics=['macro19_rmse'],
        metric_notes='Mean of per-repeat outer-test scores. Macro23 RMSE equally averages all 23 held-in lineages; macro19 RMSE uses the 19 cohort lineages with n>=10. Per-lineage Spearman is descriptive and undefined for one-point/constant test subsets; no macro Spearman is used because tiny held-out subsets make it unstable.',
        primary_comparisons=[[v,'unweighted'] for v in ARMS if v!='unweighted'],secondary_comparisons=[['inverse_sqrt','inverse_frequency'],['capped_inverse_5','inverse_frequency'],['inverse_sqrt','capped_inverse_5']],
        uncertainty=dict(count=5000,seed=2026110601,unit='Paired whole-lineage bootstrap shared across models and all 20 split-specific test sets; conditional on predictions, no refitting, no multiplicity adjustment. Repeated test sets overlap and are not independent biological replicates.'),
        limitations='Exploratory reuse of the cell-line cohort; evaluates new cell lines from represented lineages, not unseen-lineage transfer. No external validation. Rare-lineage test subsets are tiny; one Ampulla training line cannot be an inner validation point.',
        new_outer_models=80,workers=WORKERS,versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__),
        data_sha256={name:sha(OUT/name) for name in ['inputs.npz','shared_splits.npz']},scripts={name:sha(OUT/name) for name in ['run_stratified.py','make_splits.py','model_helpers.py','metric_helpers.py','lineage_weights.py','weighted_mlp.py','evaluate.py','validate_preflight.py']})
    path=OUT/'protocol.json'
    if path.exists():
        existing=json.loads(path.read_text());p['created_utc']=existing['created_utc'];assert existing==clean(p)
    else:dump(path,p)
    dump(OUT/'source_manifest.json',dict(files=sources));return sha(path)

def main():
    start=time.monotonic();ph=prepare();initialize();a=json.loads((OUT/'preflight_validation.json').read_text());assert a['passed'] and a['protocol_sha256']==ph
    with futures.ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        pending=[pool.submit(fit_job,arm,r,ph) for r in range(20) for arm in ARMS];n=0
        for future in futures.as_completed(pending):
            a=future.result();n+=1;elapsed=time.monotonic()-start;v=dict(phase='fitting',completed=n,total=80,elapsed_seconds=elapsed,estimated_remaining_seconds=elapsed/n*(80-n));dump(OUT/'progress.json',v)
            if n%4==0:print(json.dumps(v),flush=True)
    p=json.loads((OUT/'protocol.json').read_text());assert sha(OUT/'protocol.json')==ph
    for k,h in p['scripts'].items():assert sha(OUT/k)==h
    for k,h in p['data_sha256'].items():assert sha(OUT/k)==h
    for a in json.loads((OUT/'source_manifest.json').read_text())['files']:assert sha(WS/a['path'])==a['sha256']
    dump(OUT/'fit_completion.json',dict(complete=True,outer_models=80,elapsed_seconds=time.monotonic()-start));dump(OUT/'progress.json',dict(phase='fits_complete',completed=80,total=80));print('ALL_FITS_COMPLETE',flush=True)
if __name__=='__main__':main()
