#!/usr/bin/env python3
"""CTRP gene-only Hallmark additions under fixed shared nested holdouts."""
import concurrent.futures as futures,json,multiprocessing as mp,platform,shutil,sys,time
from pathlib import Path
import joblib,numpy as np,scipy,sklearn
from model_helpers import sha,clean,dump,fitted_neural
from ridge_helpers import ridge_path
from enet_helpers import GRID,L1_RATIOS,ALPHAS,MAX_ITER,TOL,fit_path,fit_selected
from make_splits import inner_rows
OUT=Path(__file__).resolve().parent;WS=OUT.parent.parent;SOURCE=WS/'outputs/erastin_elastic_net_comparison';PREP=WS/'outputs/erastin_emt_ddr_panels'
FAMILIES=['ridge','elastic_net','neural_network'];PANELS=['ferro5','ferro64','ferro64_emt','ferro64_ddr','ferro64_emt_ddr'];NEW=PANELS[2:];WORKERS=8;STATE={};sys.dont_write_bytecode=True
def arm_name(f,p):return f+'__'+p
def root_for(f,p,r):return OUT/'folds'/f'{f}__{p}__r{r+1:02d}'
def initialize():
    a=np.load(OUT/'panel_inputs.npz');sp=np.load(OUT/'shared_splits.npz');p=json.load(open(OUT/'protocol.json'));STATE.update(x={n:a['x_'+n] for n in PANELS},genes={n:a['genes_'+n] for n in PANELS},y=a['y'],ids=a['sample_ids'],groups=a['group_ids'],lineage=a['lineage'],test=sp['outer_test'],inner=sp['inner_fold'],seeds=sp['outer_split_seeds'],inner_seeds=sp['inner_split_seeds'],nn_seeds=sp['neural_initialization_seeds'],alphas=np.array(p['ridge_alphas']),nn_grid=p['neural_grid'])
def grid_for(f):return GRID if f=='elastic_net' else [dict(alpha=float(a)) for a in STATE['alphas']] if f=='ridge' else STATE['nn_grid']
def fit_job(panel,family,r,ph):
    root=root_for(family,panel,r)
    if root.with_suffix('.json').exists():
        m=json.load(open(root.with_suffix('.json')));assert m['protocol_sha256']==ph
        for ext in ['npz','joblib']:assert sha(root.with_suffix('.'+ext))==m[ext+'_sha256']
        return m
    start=time.monotonic();s=STATE;x=s['x'][panel];y=s['y'];inner=s['inner'][r];tr=np.flatnonzero(~s['test'][r]);te=np.flatnonzero(s['test'][r]);grid=grid_for(family);val=np.full((len(grid),len(y)),np.nan);loss=np.empty((len(grid),5));scaling=[];diagnostics=[]
    assert set(s['groups'][tr]).isdisjoint(s['groups'][te]) and set(s['lineage'][tr])==set(s['lineage'][te])==set(s['lineage'])
    for k in range(5):
        a,b=inner_rows(inner,k);assert set(a)|set(b)==set(tr) and set(s['groups'][a]).isdisjoint(s['groups'][b]);scaling.append(dict(fold=k,y_mean=float(y[a].mean()),y_scale=float(y[a].std())))
        if family=='ridge':p,_=ridge_path(x,y,a,b,s['alphas']);val[:,b]=p.T;loss[:,k]=np.sqrt(np.mean((p-y[b,None])**2,axis=0))
        elif family=='elastic_net':p,_,diag,_=fit_path(x,y,a,b);val[:,b]=p;loss[:,k]=np.sqrt(np.mean((p-y[b][None,:])**2,axis=1));diagnostics.extend(dict(stage='inner',fold=k,**d) for d in diag)
        else:
            for c,params in enumerate(grid):
                p,_,d,_=fitted_neural(params,x,y,a,b,s['nn_seeds'][r],np.ones(len(a)));val[c,b]=p;loss[c,k]=np.sqrt(np.mean((p-y[b])**2));diagnostics.append(dict(stage='inner',fold=k,candidate=c,**d))
    assert np.isfinite(val[:,inner>=0]).all() and np.isnan(val[:,inner<0]).all();c=int(np.argmin(loss.mean(axis=1)));params=grid[c];members=None
    if family=='ridge':p,model=ridge_path(x,y,tr,te,[params['alpha']],retain=True);pred=p[:,0]
    elif family=='elastic_net':pred,model,d=fit_selected(x,y,tr,te,params);diagnostics.append(dict(stage='outer',**d))
    else:pred,model,d,members=fitted_neural(params,x,y,tr,te,s['nn_seeds'][r],np.ones(len(tr)),retain=True);diagnostics.append(dict(stage='outer',**d))
    joblib.dump(model,root.with_suffix('.joblib'),compress=3);arrays=dict(train_rows=tr,test_rows=te,test_prediction=pred,inner_validation_predictions=val,inner_fold_rmse=loss)
    if members is not None:arrays['member_test_predictions']=members
    np.savez_compressed(root.with_suffix('.npz'),**arrays);m=dict(family=family,panel=panel,repeat=r+1,gene_count=x.shape[1],outer_split_seed=int(s['seeds'][r]),inner_split_seed=int(s['inner_seeds'][r]),neural_seeds=s['nn_seeds'][r],train_n=len(tr),test_n=len(te),train_lineages=23,test_lineages=23,selected_candidate=c,selected_parameters=params,mean_inner_rmse=float(loss[c].mean()),inner_scaling=scaling,fit_diagnostics=diagnostics,reused=False,protocol_sha256=ph,seconds=time.monotonic()-start,npz_sha256=sha(root.with_suffix('.npz')),joblib_sha256=sha(root.with_suffix('.joblib')));dump(root.with_suffix('.json'),m);return m
def prepare():
    OUT.mkdir(exist_ok=True);(OUT/'folds').mkdir(exist_ok=True);(OUT/'source_metadata').mkdir(exist_ok=True);records=[];known={r['path']:r['sha256'] for r in json.load(open(SOURCE/'artifact_manifest.json'))['files']}
    assert json.load(open(PREP/'preparation_validation.json'))['passed']
    for name in ['panel_inputs.npz','shared_splits.npz','panel_definitions.json','gene_mapping.tsv','coverage_summary.json','source_manifest.json','preparation_validation.json','prepare_panels.py']:
        records.append(dict(path=str((PREP/name).relative_to(WS)),sha256=sha(PREP/name)))
        if name in ['panel_inputs.npz','shared_splits.npz','panel_definitions.json','gene_mapping.tsv','coverage_summary.json']:
            dest=OUT/name
            if dest.exists():assert sha(dest)==sha(PREP/name)
            else:shutil.copyfile(PREP/name,dest)
    pv=json.load(open(PREP/'preparation_validation.json'));assert sha(OUT/'panel_inputs.npz')==pv['panel_input_sha256'] and sha(OUT/'shared_splits.npz')==pv['shared_split_sha256']
    for row in json.load(open(PREP/'source_manifest.json'))['files']:assert sha(WS/row['path'])==row['sha256'];records.append(row)
    for name in ['protocol.json','summary.tsv','predictions.npz','validation.json','shared_splits.npz','panel_inputs.npz','split_lineage_counts.tsv','inner_split_lineage_counts.tsv','sample_test_coverage.tsv','model_helpers.py','weighted_mlp.py','metric_helpers.py','make_splits.py','enet_helpers.py']:
        assert sha(SOURCE/name)==known[name];records.append(dict(path=str((SOURCE/name).relative_to(WS)),sha256=known[name]))
    for name in ['model_helpers.py','weighted_mlp.py','metric_helpers.py','make_splits.py','enet_helpers.py']:assert sha(OUT/name)==known[name]
    for source,name in [(WS/'outputs/erastin_five_gene_comparison','ridge_helpers.py'),(WS/'outputs/erastin_prism_direct_comparison','evaluate.py')]:
        kk={r['path']:r['sha256'] for r in json.load(open(source/'artifact_manifest.json'))['files']};assert sha(source/name)==sha(OUT/name)==kk[name];records.append(dict(path=str((source/name).relative_to(WS)),sha256=kk[name]))
    for name in ['split_lineage_counts.tsv','inner_split_lineage_counts.tsv','sample_test_coverage.tsv']:shutil.copyfile(SOURCE/name,OUT/name)
    for f in FAMILIES:
        for p in [5,64]:
            for r in range(20):
                for ext in ['json','npz','joblib']:
                    name=f'folds/{f}__p{p}__r{r+1:02d}.{ext}';assert sha(SOURCE/name)==known[name];records.append(dict(path=str((SOURCE/name).relative_to(WS)),sha256=known[name]));dest=OUT/'source_metadata'/Path(name).name if ext=='json' else root_for(f,'ferro'+str(p),r).with_suffix('.'+ext)
                    if dest.exists():assert sha(dest)==known[name]
                    else:shutil.copyfile(SOURCE/name,dest)
    old=json.load(open(SOURCE/'protocol.json'));definitions=json.load(open(OUT/'panel_definitions.json'));p=dict(status='fixed_before_Hallmark_addition_fits',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),samples=744,identity_groups=733,lineages=23,excluded_samples=old['excluded_samples'],target='Unchanged negative harmonized CTRP erastin AUC. Higher means more sensitive. No culture predictors.',panels=definitions,split_design=old['split_design'],inner_design=old['inner_design'],ampulla_handling=old['ampulla_handling'],outer_train_sizes=old['outer_train_sizes'],outer_test_sizes=old['outer_test_sizes'],split_seeds=old['split_seeds'],inner_split_seeds=old['inner_split_seeds'],neural_seed_triplets=old['neural_seed_triplets'],ridge_alphas=old['ridge_alphas'],neural_grid=old['neural_grid'],neural_fixed=old['neural_fixed'],elastic_net_grid=GRID,elastic_net_fixed=dict(max_iter=MAX_ITER,tol=TOL,selection='cyclic'),weighting='Unweighted; equal-mean five-fold RMSE in original response units; first fixed grid candidate breaks ties.',preprocessing='Every StandardScaler fits only the current inner training set, or outer training at selected refit. No feature filtering based on response or all-cohort variance; no culture/lineage/identity predictors.',selection='Full Hallmark EMT (200) and DNA repair (150) sets selected by user before these fits. Include all mapped genes; no data-driven gene selection outside Elastic Net training. Preserve 64-gene order, append additions, deduplicate stable Entrez IDs. GPX4 is already in the 64-gene panel. BOLA2/BOLA2B header collision resolved by exact symbol and release HGNC, retaining BOLA2 only.',source_sets='Official MSigDB snapshots downloaded 2026-11-02; JSON membership and HTML stable NCBI identifiers independently agree. Sets are expression programs, not exhaustive functional catalogues. BRCA1/BRCA2/ZEB1/ZEB2 not added as unselected extras.',primary_comparisons=[[arm_name(f,p),arm_name(f,'ferro64')] for f in FAMILIES for p in NEW],secondary_comparisons=[[arm_name(f,'ferro64_emt_ddr'),arm_name(f,p)] for f in FAMILIES for p in ['ferro64_emt','ferro64_ddr']]+[[arm_name(f,'ferro64'),arm_name(f,'ferro5')] for f in FAMILIES],primary_metrics=['pooled_spearman','pooled_rmse','pooled_r2','macro_all_rmse'],secondary_metrics=['macro_large_rmse','pooled_pearson','pooled_mae'],uncertainty=dict(count=5000,seed=2026111601,unit=old['uncertainty']['unit']),limitations=[old['limitations'],old['selection_limitation'],'Hallmark membership fixed independently of these fits, but this cell-line cohort has informed earlier modelling choices; overlapping holdouts are exploratory, not independent external validation.','Expression of repair/EMT genes does not establish functional repair deficiency or an EMT state; EMT interpretation in hematologic lineages is especially limited.','Erastin system-xc inhibition and GPX4-inhibitor sensitivity are different targets; mechanistic literature is a rationale, not a guarantee of predictive gain.','The exact upstream harmonized CTRP AUC generation recipe remains unverified.'],new_outer_models=180,reused_outer_models=120,total_outer_models=300,workers=WORKERS,versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__),data_sha256={name:sha(OUT/name) for name in ['panel_inputs.npz','shared_splits.npz','panel_definitions.json','gene_mapping.tsv']},scripts={name:sha(OUT/name) for name in ['run_comparison.py','validate_preflight.py','enet_helpers.py','ridge_helpers.py','model_helpers.py','weighted_mlp.py','metric_helpers.py','evaluate.py','make_splits.py']})
    path=OUT/'protocol.json'
    if path.exists():prev=json.load(open(path));p['created_utc']=prev['created_utc'];assert clean(p)==prev
    else:dump(path,p)
    ph=sha(path)
    for f in FAMILIES:
        for n in [5,64]:
            for r in range(20):
                original=OUT/'source_metadata'/f'{f}__p{n}__r{r+1:02d}.json';m=json.load(open(original));m.update(panel='ferro'+str(n),gene_count=n,reused=True,protocol_sha256=ph,original_metadata_sha256=sha(original),source_protocol_sha256=sha(SOURCE/'protocol.json'));dest=root_for(f,'ferro'+str(n),r).with_suffix('.json')
                if dest.exists():assert json.load(open(dest))==clean(m)
                else:dump(dest,m)
    dump(OUT/'source_manifest.json',dict(files=records));return ph
def main():
    start=time.monotonic();ph=prepare();initialize();pf=json.load(open(OUT/'preflight_validation.json'));assert pf['passed'] and pf['protocol_sha256']==ph
    with futures.ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        jobs=[pool.submit(fit_job,p,f,r,ph) for r in range(20) for p in NEW for f in FAMILIES];done=0
        for future in futures.as_completed(jobs):
            m=future.result();done+=1;elapsed=time.monotonic()-start;progress=dict(phase='fitting',completed=done,total=180,reused=120,elapsed_seconds=elapsed,estimated_remaining_seconds=elapsed/done*(180-done));dump(OUT/'progress.json',progress)
            if done%18==0:print(json.dumps(progress),flush=True)
    pr=json.load(open(OUT/'protocol.json'));assert sha(OUT/'protocol.json')==ph
    for name,h in pr['scripts'].items():assert sha(OUT/name)==h
    for name,h in pr['data_sha256'].items():assert sha(OUT/name)==h
    for row in json.load(open(OUT/'source_manifest.json'))['files']:assert sha(WS/row['path'])==row['sha256']
    dump(OUT/'fit_completion.json',dict(complete=True,new_outer_models=180,reused_outer_models=120,total_outer_models=300,seconds=time.monotonic()-start));dump(OUT/'progress.json',dict(phase='fits_complete',completed=180,total=180,reused=120));print('ALL_FITS_COMPLETE',flush=True)
if __name__=='__main__':main()
