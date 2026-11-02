#!/usr/bin/env python3
"""Train all six unweighted PRISM arms using shared grouped lineage splits."""
import concurrent.futures as futures
import json,multiprocessing as mp,platform,shutil,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd,scipy,sklearn
from model_helpers import sha,clean,dump,fitted_neural
from ridge_helpers import ridge_path
from enet_helpers import GRID,L1_RATIOS,ALPHAS,MAX_ITER,TOL,fit_path,fit_selected
from make_splits import outer_split,inner_folds,inner_rows
OUT=Path(__file__).resolve().parent;WS=OUT.parent.parent;FAMILIES=['ridge','elastic_net','neural_network'];PANELS=[5,64];WORKERS=8;STATE={}
sys.dont_write_bytecode=True

def arm_name(f,p):return f+'__'+str(p)
def initialize():
    a=np.load(OUT/'panel_inputs.npz');s=np.load(OUT/'shared_splits.npz');p=json.load(open(OUT/'protocol.json'))
    STATE.update(x={n:a['x_'+str(n)] for n in PANELS},genes={n:a['genes_'+str(n)] for n in PANELS},y=a['y'],ids=a['sample_ids'],groups=a['group_ids'],lineage=a['lineage'],test=s['outer_test'],inner=s['inner_fold'],seeds=s['outer_split_seeds'],inner_seeds=s['inner_split_seeds'],nn_seeds=s['neural_initialization_seeds'],alphas=np.array(p['ridge_alphas']),nn_grid=p['neural_grid'])
    STATE['model_seeds']=np.array([int(v)%(2**31-1) for v in STATE['seeds']])
def grid_for(f):return GRID if f=='elastic_net' else [dict(alpha=float(v)) for v in STATE['alphas']] if f=='ridge' else STATE['nn_grid']

def fit_job(panel,family,r,ph):
    root=OUT/'folds'/f'{family}__p{panel}__r{r+1:02d}'
    if root.with_suffix('.json').exists():
        m=json.load(open(root.with_suffix('.json')));assert m['protocol_sha256']==ph
        for ext in ['npz','joblib']:assert sha(root.with_suffix('.'+ext))==m[ext+'_sha256']
        return m
    start=time.monotonic();s=STATE;x=s['x'][panel];y=s['y'];g=s['groups'];lin=s['lineage'];inner=s['inner'][r];tr=np.flatnonzero(~s['test'][r]);te=np.flatnonzero(s['test'][r]);assert set(g[tr]).isdisjoint(g[te]) and set(lin[tr])==set(lin[te])==set(lin)
    grid=grid_for(family);val=np.full((len(grid),len(y)),np.nan);loss=np.empty((len(grid),5));scaling=[];diagnostics=[];seed=int(s['model_seeds'][r])
    for k in range(5):
        a,b=inner_rows(inner,k);assert set(a)|set(b)==set(tr) and set(g[a]).isdisjoint(g[b]) and set(lin[a])==set(lin)
        scaling.append(dict(fold=k,y_mean=float(y[a].mean()),y_scale=float(y[a].std())))
        if family=='ridge':
            pp,_=ridge_path(x,y,a,b,s['alphas']);val[:,b]=pp.T;loss[:,k]=np.sqrt(np.mean((pp-y[b,None])**2,axis=0))
        elif family=='elastic_net':
            pp,_,diag,_=fit_path(x,y,a,b);val[:,b]=pp;loss[:,k]=np.sqrt(np.mean((pp-y[b][None,:])**2,axis=1));diagnostics.extend(dict(stage='inner',fold=k,**d) for d in diag)
        else:
            for c,params in enumerate(grid):
                pred,_,diag,_=fitted_neural(params,x,y,a,b,s['nn_seeds'][r],np.ones(len(a)));val[c,b]=pred;loss[c,k]=np.sqrt(np.mean((pred-y[b])**2));diagnostics.append(dict(stage='inner',candidate=c,fold=k,**diag))
    assert np.isfinite(val[:,inner>=0]).all() and np.isnan(val[:,inner<0]).all();c=int(np.argmin(loss.mean(axis=1)));params=grid[c];members=None
    if family=='ridge':pred,saved=ridge_path(x,y,tr,te,[params['alpha']],retain=True);pred=pred[:,0]
    elif family=='elastic_net':pred,saved,diag=fit_selected(x,y,tr,te,params);diagnostics.append(dict(stage='outer',**diag))
    else:
        pred,saved,diag,members=fitted_neural(params,x,y,tr,te,s['nn_seeds'][r],np.ones(len(tr)),retain=True);diagnostics.append(dict(stage='outer',candidate=c,**diag))
    assert np.isfinite(pred).all();joblib.dump(saved,root.with_suffix('.joblib'),compress=3);arrays=dict(train_rows=tr,test_rows=te,test_prediction=pred,inner_validation_predictions=val,inner_fold_rmse=loss)
    if members is not None:arrays['member_test_predictions']=members
    np.savez_compressed(root.with_suffix('.npz'),**arrays)
    m=dict(family=family,panel=panel,repeat=r+1,outer_split_seed=int(s['seeds'][r]),inner_split_seed=int(s['inner_seeds'][r]),neural_seeds=s['nn_seeds'][r],model_seed=seed,train_n=len(tr),test_n=len(te),train_lineages=len(set(lin[tr])),test_lineages=len(set(lin[te])),selected_candidate=c,selected_parameters=params,mean_inner_rmse=float(loss[c].mean()),inner_scaling=scaling,fit_diagnostics=diagnostics,reused=False,protocol_sha256=ph,seconds=time.monotonic()-start,npz_sha256=sha(root.with_suffix('.npz')),joblib_sha256=sha(root.with_suffix('.joblib')))
    if family=='elastic_net':m['nonzero_genes']=int(np.count_nonzero(saved['model'].coef_))
    dump(root.with_suffix('.json'),m);return m

def prepare():
    OUT.mkdir(exist_ok=True);(OUT/'folds').mkdir(exist_ok=True);source=WS/'outputs/erastin_prism_validation';known={r['path']:r['sha256'] for r in json.load(open(source/'artifact_manifest.json'))['files']};records=[]
    for name in ['matched_prism_inputs.npz','matched_cohort.tsv','cohort_summary.json','cohort_audit.tsv','prism_erastin_all_rows.tsv','erastin_treatment_metadata.tsv','gene_mapping.tsv','source_manifest.json','validation.json','protocol.json','figshare_20564034.json','prism-repurposing-20q2-secondary-screen-readme']:
        assert sha(source/name)==known[name];records.append(dict(path=str((source/name).relative_to(WS)),sha256=known[name]))
        if name in ['matched_cohort.tsv','cohort_audit.tsv','prism_erastin_all_rows.tsv','erastin_treatment_metadata.tsv','gene_mapping.tsv','figshare_20564034.json','prism-repurposing-20q2-secondary-screen-readme']:shutil.copyfile(source/name,OUT/name)
    for row in json.load(open(source/'source_manifest.json'))['files']:assert sha(WS/row['path'])==row['sha256']
    seedsource=WS/'outputs/erastin_elastic_net_comparison';ss=json.load(open(seedsource/'artifact_manifest.json'));ss={v['path']:v['sha256'] for v in ss['files']}
    for name in ['shared_splits.npz','protocol.json','model_helpers.py','weighted_mlp.py','metric_helpers.py','make_splits.py','enet_helpers.py']:
        assert sha(seedsource/name)==ss[name];records.append(dict(path=str((seedsource/name).relative_to(WS)),sha256=ss[name]))
        if name.endswith('.py'):assert sha(OUT/name)==ss[name]
    ridgesource=WS/'outputs/erastin_five_gene_comparison/ridge_helpers.py';assert sha(OUT/'ridge_helpers.py')==sha(ridgesource);records.append(dict(path=str(ridgesource.relative_to(WS)),sha256=sha(ridgesource)))
    a=np.load(source/'matched_prism_inputs.npz');seed=np.load(seedsource/'shared_splits.npz');old=json.load(open(seedsource/'protocol.json'));arrays={k:a[k] for k in ['x_5','x_64','genes_5','genes_64','y','sample_ids','group_ids','lineage']};arrays['prism_auc']=a['prism_auc'];arrays['five_gene_indices_in_64']=np.array([list(a['genes_64']).index(g) for g in a['genes_5']]);assert arrays['x_64'].shape==(386,64) and len(set(arrays['group_ids']))==379
    np.testing.assert_array_equal(arrays['x_5'],arrays['x_64'][:,arrays['five_gene_indices_in_64']]);np.testing.assert_array_equal(arrays['y'],-arrays['prism_auc'])
    path=OUT/'panel_inputs.npz'
    if path.exists():
        for k,v in arrays.items():np.testing.assert_array_equal(np.load(path)[k],v)
    else:np.savez_compressed(path,**arrays)
    lin=arrays['lineage'];groups=arrays['group_ids'];ids=arrays['sample_ids'];labels=np.unique(lin);assert len(labels)==20 and 'Cervix' not in labels
    for label in labels:assert len(set(groups[lin==label]))>=2
    tests=np.stack([outer_split(lin,groups,v) for v in seed['outer_split_seeds']]);inner=np.stack([inner_folds(lin,groups,tests[r],v) for r,v in enumerate(seed['inner_split_seeds'])]);splits=dict(sample_ids=ids,group_ids=groups,lineage=lin,outer_test=tests,inner_fold=inner,**{k:seed[k] for k in ['outer_split_seeds','inner_split_seeds','neural_initialization_seeds']})
    pp=OUT/'shared_splits.npz'
    if pp.exists():
        for k,v in splits.items():np.testing.assert_array_equal(np.load(pp)[k],v)
    else:np.savez_compressed(pp,**splits)
    outerrows=[];innerrows=[]
    for r,mask in enumerate(tests):
        for l in labels:
            lm=lin==l;outerrows.append(dict(repeat=r+1,lineage=str(l),cohort_n=int(lm.sum()),cohort_groups=len(set(groups[lm])),train_n=int((lm&~mask).sum()),test_n=int((lm&mask).sum()),train_groups=len(set(groups[lm&~mask])),test_groups=len(set(groups[lm&mask]))))
        for k in range(5):
            tr,va=inner_rows(inner[r],k)
            for l in labels:innerrows.append(dict(repeat=r+1,inner_fold=k,lineage=str(l),train_n=int(np.sum(lin[tr]==l)),validation_n=int(np.sum(lin[va]==l)),protected_n=int(np.sum((inner[r]==-2)&(lin==l)))))
    pd.DataFrame(outerrows).to_csv(OUT/'split_lineage_counts.tsv',sep='\t',index=False);pd.DataFrame(innerrows).to_csv(OUT/'inner_split_lineage_counts.tsv',sep='\t',index=False);pd.DataFrame(dict(ModelID=ids,lineage=lin,group_id=groups,test_appearances=tests.sum(axis=0))).to_csv(OUT/'sample_test_coverage.tsv',sep='\t',index=False)
    large=[str(l) for l in labels if np.sum(lin==l)>=10];nn_grid=[dict(width=w,alpha=float(v)) for w in [16,32] for v in [1,10,100,1000]]
    p=dict(status='fixed_before_direct_PRISM_fits',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),response_source='Official Broad DepMap PRISM Repurposing 20Q2 v1, DOI 10.6084/m9.figshare.20564034.v1; erastin BRD-A25004090-001-08-4, HTS002 secondary dose-response AUC.',target='Negative reported PRISM AUC, higher=more sensitive. Preserve values above 1 and finite processed curves. No clipping, R2-based response filtering, response imputation or calibration to CTRP.',
        samples=len(ids),identity_groups=len(set(groups)),lineages=len(labels),lineage_labels=labels,large_lineage_count=len(large),large_lineage_labels=large,panels={str(n):arrays['genes_'+str(n)] for n in PANELS},five_gene_indices_in_64=arrays['five_gene_indices_in_64'],
        cohort='All 386 matched PRISM cancer lines and 379 verified identity groups, including lines previously excluded from frozen-model transfer scoring because they were never held out under the old CTRP splits. No additional lineage exclusions are needed.',
        split_design='20 response-blind lineage-stratified grouped holdouts. Within each lineage, sample intact identity groups uniformly among subsets with nearest feasible sample count to round(0.2*n), requiring at least one group in both training/testing. New masks generated on PRISM cohort, shared by all six arms.',
        inner_design='Same five response-blind group-stratified inner folds per outer split across all six arms. Every inner training fit retains all 20 lineages; rare lineages cannot occur in every inner validation fold. If an outer-training lineage had one group it would remain protected in every inner training fit.',
        split_seeds=seed['outer_split_seeds'],inner_split_seeds=seed['inner_split_seeds'],neural_seed_triplets=seed['neural_initialization_seeds'],outer_train_sizes=(~tests).sum(axis=1),outer_test_sizes=tests.sum(axis=1),
        ridge_alphas=old['ridge_alphas'],neural_grid=nn_grid,neural_fixed=old['neural_fixed'],neural_grid_change='Expand previous widths 16/32 x alpha 10/100 to widths 16/32 x alpha 1/10/100/1000, fixed before any direct PRISM fitting. Three-member tanh L-BFGS ensemble for each candidate and outer refit.',
        elastic_net_grid=GRID,elastic_net_l1_ratios=L1_RATIOS,elastic_net_alphas=ALPHAS,elastic_net_fixed=dict(max_iter=MAX_ITER,tol=TOL,selection='cyclic',positive=False),
        weighting='Unweighted training and validation in all six arms.',scaling='All feature scalers use only current training subset. Ridge centers y in current training; Elastic Net/NN standardize y in current training. Each inner candidate is predicted using preprocessing fitted on that inner-training subset.',
        tuning='Minimize equal-mean five-fold unweighted RMSE in original negative-PRISM-AUC units. First candidate in fixed grid wins a tie. Independently tune each model/panel/outer split, then refit on outer training rows only.',
        controls='All 120 models newly trained on PRISM, with no reused CTRP-fitted parameters. Original numerical fit helpers/architecture reused and verified. Add a training-mean prediction control using only each outer-training y.',
        primary_metrics=['pooled_rmse','macro_all_rmse'],secondary_metrics=['pooled_spearman','pooled_r2','pooled_mae','pooled_pearson','macro_large_rmse'],
        metrics='Compute each score on the 78-line test set per split, then average across 20 repeats. Equal-lineage macro RMSE uses all 20 lineages; secondary larger-lineage macro uses cohort n>=10. Constant predictions have undefined correlations, reported as NA with defined-repeat counts; never substitute zero or discard RMSE scores.',
        primary_comparisons=[[arm_name(f,5),arm_name(f,64)] for f in FAMILIES],secondary_comparisons=[[arm_name('elastic_net',n),arm_name('ridge',n)] for n in PANELS]+[[arm_name('neural_network',n),arm_name('ridge',n)] for n in PANELS]+[[arm_name('neural_network',n),arm_name('elastic_net',n)] for n in PANELS],
        uncertainty=dict(count=5000,seed=2026111201,unit='Paired whole-lineage bootstrap shared across arms and all 20 split-specific test sets. Fixed fits, no refitting or correction for model/panel comparisons. Pairwise correlation differences use only repeats where both correlations are defined.'),
        panel_selection='The five genes were fixed from previous CTRP importance analysis before PRISM training. No PRISM-driven gene reselection is performed. Earlier PRISM transfer results and this cohort have already been inspected; this is an exploratory internal benchmark, not an untouched external validation.',
        equivalence='No equivalence/noninferiority margin is defined; intervals containing zero do not establish equivalence.',limitations=['20 repeated holdouts overlap; they are not independent biological replications.','Lineages are represented in both sets. This design measures new cell-line prediction within represented lineages, not transfer to absent lineages.','Public processed AUC and STR checks do not establish independent raw-well or replicate-level QC.','This evaluates the predefined grids, not globally optimal neural architectures.'],
        new_outer_models=120,reused_outer_models=0,total_outer_models=120,workers=WORKERS,versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__),data_sha256={name:sha(OUT/name) for name in ['panel_inputs.npz','shared_splits.npz']},scripts={name:sha(OUT/name) for name in ['run_comparison.py','model_helpers.py','weighted_mlp.py','metric_helpers.py','ridge_helpers.py','enet_helpers.py','make_splits.py','evaluate.py','validate_preflight.py']})
    path=OUT/'protocol.json'
    if path.exists():previous=json.load(open(path));p['created_utc']=previous['created_utc'];assert previous==clean(p)
    else:dump(path,p)
    dump(OUT/'source_manifest.json',dict(files=records,prism_source_manifest_sha256=sha(source/'source_manifest.json')));return sha(path)

def main():
    start=time.monotonic();ph=prepare();initialize();v=json.load(open(OUT/'preflight_validation.json'));assert v['passed'] and v['protocol_sha256']==ph
    with futures.ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        jobs=[pool.submit(fit_job,p,f,r,ph) for r in range(20) for p in PANELS for f in FAMILIES];n=0;counts={}
        for future in futures.as_completed(jobs):
            m=future.result();n+=1;arm=arm_name(m['family'],m['panel']);counts[arm]=counts.get(arm,0)+1;elapsed=time.monotonic()-start;v=dict(phase='fitting',completed=n,total=120,by_arm=counts,elapsed_seconds=elapsed,estimated_remaining_seconds=elapsed/n*(120-n));dump(OUT/'progress.json',v)
            if n%12==0:print(json.dumps(v),flush=True)
    p=json.load(open(OUT/'protocol.json'));assert sha(OUT/'protocol.json')==ph
    for name,h in p['scripts'].items():assert sha(OUT/name)==h
    for name,h in p['data_sha256'].items():assert sha(OUT/name)==h
    for a in json.load(open(OUT/'source_manifest.json'))['files']:assert sha(WS/a['path'])==a['sha256']
    dump(OUT/'fit_completion.json',dict(complete=True,new_outer_models=120,reused_outer_models=0,total_outer_models=120,elapsed_seconds=time.monotonic()-start));dump(OUT/'progress.json',dict(phase='fits_complete',completed=120,total=120));print('ALL_FITS_COMPLETE',flush=True)
if __name__=='__main__':main()
