#!/usr/bin/env python3
"""Train culture-augmented and culture-only PRISM models on unchanged splits."""
import concurrent.futures as futures
import json,multiprocessing as mp,platform,shutil,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd,scipy,sklearn
from model_helpers import sha,clean,dump
from culture_features import CulturePreprocessor,extract_metadata,CAT_FIELDS,MIN_COUNTS
from fit_helpers import GRID,ALPHAS,L1_RATIOS,MAX_ITER,TOL,ridge_path,elastic_path,fit_neural,fit_outer
from make_splits import inner_rows
OUT=Path(__file__).resolve().parent;WS=OUT.parent.parent;SOURCE=WS/'outputs/erastin_prism_direct_comparison';FAMILIES=['ridge','elastic_net','neural_network'];PANELS=[0,5,64];WORKERS=8;STATE={}
sys.dont_write_bytecode=True
def arm_name(f,p,view):return f+'__'+str(p)+'__'+view
def root_for(f,p,view,r):return OUT/'folds'/f'{f}__p{p}__{view}__r{r+1:02d}'
def initialize():
    a=np.load(OUT/'panel_inputs.npz');sp=np.load(OUT/'shared_splits.npz');pr=json.load(open(OUT/'protocol.json'));meta=pd.read_csv(OUT/'culture_metadata.tsv',sep='\t',keep_default_na=False);meta['serum_percent']=pd.to_numeric(meta.serum_percent,errors='coerce');meta['serum_missing']=meta.serum_missing.astype(str).eq('True');np.testing.assert_array_equal(meta.ModelID,a['sample_ids'])
    STATE.update(x={0:np.empty((len(a['y']),0)),5:a['x_5'],64:a['x_64']},y=a['y'],ids=a['sample_ids'],groups=a['group_ids'],lineage=a['lineage'],test=sp['outer_test'],inner=sp['inner_fold'],seeds=sp['outer_split_seeds'],inner_seeds=sp['inner_split_seeds'],nn_seeds=sp['neural_initialization_seeds'],alphas=np.array(pr['ridge_alphas']),nn_grid=pr['neural_grid'],meta=meta)
def grid_for(f):return GRID if f=='elastic_net' else [dict(alpha=float(v)) for v in STATE['alphas']] if f=='ridge' else STATE['nn_grid']
def preprocess(panel,a,b):
    s=STATE;x=s['x'][panel];prep=CulturePreprocessor().fit(x[a],s['meta'].iloc[a]);z=prep.transform(x[a],s['meta'].iloc[a]);zz=prep.transform(x[b],s['meta'].iloc[b]);assert np.max(np.abs(z.mean(axis=0)))<1e-12
    return z,zz,prep

def fit_job(panel,family,r,ph):
    root=root_for(family,panel,'culture',r)
    if root.with_suffix('.json').exists():
        m=json.load(open(root.with_suffix('.json')));assert m['protocol_sha256']==ph
        for ext in ['npz','joblib']:assert sha(root.with_suffix('.'+ext))==m[ext+'_sha256']
        return m
    begin=time.monotonic();s=STATE;y=s['y'];g=s['groups'];lin=s['lineage'];assignment=s['inner'][r];tr=np.flatnonzero(~s['test'][r]);te=np.flatnonzero(s['test'][r]);grid=grid_for(family);val=np.full((len(grid),len(y)),np.nan);loss=np.empty((len(grid),5));diags=[];preprocessing=[]
    assert set(g[tr]).isdisjoint(g[te]) and set(lin[tr])==set(lin[te])==set(lin)
    for k in range(5):
        a,b=inner_rows(assignment,k);assert set(a)|set(b)==set(tr) and set(g[a]).isdisjoint(g[b]);z,zz,prep=preprocess(panel,a,b);preprocessing.append(dict(fold=k,**prep.diagnostic(s['meta'].iloc[b])))
        if family=='ridge':pp,_=ridge_path(z,y[a],zz,s['alphas']);val[:,b]=pp;loss[:,k]=np.sqrt(np.mean((pp-y[b][None,:])**2,axis=1))
        elif family=='elastic_net':
            pp,diagnostics=elastic_path(z,y[a],zz);val[:,b]=pp;loss[:,k]=np.sqrt(np.mean((pp-y[b][None,:])**2,axis=1));diags.extend(dict(stage='inner',fold=k,**d) for d in diagnostics)
        else:
            for c,params in enumerate(grid):
                p,_,diag,_=fit_neural(z,y[a],zz,params,s['nn_seeds'][r]);val[c,b]=p;loss[c,k]=np.sqrt(np.mean((p-y[b])**2));diags.append(dict(stage='inner',fold=k,candidate=c,**diag))
    assert np.isfinite(val[:,assignment>=0]).all() and np.isnan(val[:,assignment<0]).all();chosen=int(np.argmin(loss.mean(axis=1)));params=grid[chosen];z,zz,prep=preprocess(panel,tr,te);pred,model,diag,members=fit_outer(family,z,y[tr],zz,params,s['nn_seeds'][r]);diags.append(dict(stage='outer',**diag));model.update(family=family,panel=panel,view='culture',preprocessor=prep);joblib.dump(model,root.with_suffix('.joblib'),compress=3);arrays=dict(train_rows=tr,test_rows=te,test_prediction=pred,inner_validation_predictions=val,inner_fold_rmse=loss)
    if members is not None:arrays['member_test_predictions']=members
    np.savez_compressed(root.with_suffix('.npz'),**arrays);m=dict(family=family,panel=panel,view='culture',repeat=r+1,outer_split_seed=int(s['seeds'][r]),inner_split_seed=int(s['inner_seeds'][r]),neural_seeds=s['nn_seeds'][r],train_n=len(tr),test_n=len(te),train_lineages=20,test_lineages=20,selected_candidate=chosen,selected_parameters=params,mean_inner_rmse=float(loss[chosen].mean()),preprocessing=preprocessing,outer_preprocessing=prep.diagnostic(s['meta'].iloc[te]),fit_diagnostics=diags,reused=False,protocol_sha256=ph,seconds=time.monotonic()-begin,npz_sha256=sha(root.with_suffix('.npz')),joblib_sha256=sha(root.with_suffix('.joblib')));dump(root.with_suffix('.json'),m);return m

def prepare():
    OUT.mkdir(exist_ok=True);(OUT/'folds').mkdir(exist_ok=True);(OUT/'source_metadata').mkdir(exist_ok=True);known={v['path']:v['sha256'] for v in json.load(open(SOURCE/'artifact_manifest.json'))['files']};records=[]
    names=['panel_inputs.npz','shared_splits.npz','protocol.json','summary.tsv','predictions.npz','validation.json','weighted_mlp.py','model_helpers.py','metric_helpers.py','evaluate.py','make_splits.py','split_lineage_counts.tsv','inner_split_lineage_counts.tsv','sample_test_coverage.tsv']
    for name in names:assert sha(SOURCE/name)==known[name];records.append(dict(path=str((SOURCE/name).relative_to(WS)),sha256=known[name]))
    for name in ['panel_inputs.npz','shared_splits.npz','split_lineage_counts.tsv','inner_split_lineage_counts.tsv','sample_test_coverage.tsv']:
        dest=OUT/name
        if dest.exists():assert sha(dest)==known[name]
        else:shutil.copyfile(SOURCE/name,dest)
    for name in ['weighted_mlp.py','model_helpers.py','metric_helpers.py','evaluate.py','make_splits.py']:assert sha(OUT/name)==known[name]
    raw=WS/'work/erastin/raw';manifest=json.load(open(WS/'outputs/erastin_40gene_comparison/source_manifest.json'))
    for name in ['Model.csv','README.txt']:
        entry=next(v for v in manifest['files'] if v['local_name']==name);assert sha(raw/name)==entry['sha256'];records.append(dict(path=str((raw/name).relative_to(WS)),sha256=entry['sha256']))
    a=np.load(OUT/'panel_inputs.npz');models=pd.read_csv(raw/'Model.csv',dtype=str,keep_default_na=False).set_index('ModelID');meta=extract_metadata(models,a['sample_ids']);meta.to_csv(OUT/'culture_metadata.tsv',sep='\t',index=False,float_format='%.17g');coverage=[]
    for field in ['GrowthPattern','OnboardedMedia','FormulationID','SerumFreeMedia','PlateCoating','ModelDerivationMaterial','ModelTreatment','CulturedResistanceDrug']:
        s=models.loc[a['sample_ids'],field];coverage.append(dict(field=field,nonempty=int(s.ne('').sum()),missing=int(s.eq('').sum()),unique_nonmissing=int(s[s.ne('')].nunique()),values=s.value_counts().to_dict()))
    dump(OUT/'metadata_coverage.json',dict(fields=coverage,code_recipe_bijection=True,column_semantics='Observed OnboardedMedia values are MF codes and observed FormulationID values are recipes; source README labels are reversed relative to this release CSV. Raw names/values retained.',serum_percent_counts=meta.serum_percent.value_counts().to_dict(),missing_serum=int(meta.serum_percent.isna().sum()),limitation='Onboarding annotations are covariates, not verified PRISM assay growth conditions.'))
    meta.groupby(['medium_code','raw_FormulationID'],dropna=False).size().reset_index(name='cell_lines').to_csv(OUT/'medium_dictionary.tsv',sep='\t',index=False)
    for family in FAMILIES:
        for panel in [5,64]:
            for r in range(20):
                for ext in ['json','npz','joblib']:
                    name=f'folds/{family}__p{panel}__r{r+1:02d}.{ext}';assert sha(SOURCE/name)==known[name];records.append(dict(path=str((SOURCE/name).relative_to(WS)),sha256=known[name]));dest=OUT/'source_metadata'/Path(name).name if ext=='json' else root_for(family,panel,'expression',r).with_suffix('.'+ext)
                    if dest.exists():assert sha(dest)==known[name]
                    else:shutil.copyfile(SOURCE/name,dest)
    old=json.load(open(SOURCE/'protocol.json'));p=dict(status='fixed_before_culture_augmented_fits',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),samples=386,identity_groups=379,lineages=20,source_target=old['response_source'],target=old['target'],panels=old['panels'],split_design=old['split_design'],inner_design=old['inner_design'],outer_train_sizes=old['outer_train_sizes'],outer_test_sizes=old['outer_test_sizes'],split_seeds=old['split_seeds'],inner_split_seeds=old['inner_split_seeds'],neural_seed_triplets=old['neural_seed_triplets'],ridge_alphas=old['ridge_alphas'],neural_grid=old['neural_grid'],neural_fixed=old['neural_fixed'],elastic_net_grid=GRID,elastic_net_l1_ratios=L1_RATIOS,elastic_net_alphas=ALPHAS,elastic_net_fixed=dict(max_iter=MAX_ITER,tol=TOL,selection='cyclic'),
        primary_features='Fixed 5/64 gene expression + growth pattern + medium code + basal-medium recipe prefix + recorded FBS percentage and its missingness. No response/lineage/identity/source-vendor fields as model inputs.',
        condition_fields='GrowthPattern; observed MF code from OnboardedMedia; recipe from FormulationID. One-to-one code/recipe mapping verified. Serum % extracted by explicit % FBS regex; do not infer unrecorded supplements or basal nutrient concentrations.',
        excluded_fields='PlateCoating is constant None in all 386; SerumFreeMedia has only False among 367 known and 19 missing, redundant with medium missingness. ModelDerivationMaterial, ModelTreatment and CulturedResistanceDrug are entirely empty. Exact full recipe not encoded separately from its one-to-one medium code.',
        metadata_limit='These Model.csv fields describe onboarding, not verified PRISM-screen conditions. Associations may reflect lineage, source or other cell-line properties rather than causal culture effects. The release README description and observed medium columns are reversed; preserve and audit raw field values.',
        preprocessing='Fit separately inside every inner training fold and on outer training for refit. Median-impute serum % using current training values; standardize genes and serum only. Encode growth with training-observed categories; medium code/basal family need >=5 current-training rows to retain separate categories. Other/unseen categories share reserved OTHER; missing is explicit MISSING. One-hot and missingness indicators are centered but never variance-standardized, avoiding rare-category amplification. Training column means center the joint matrix.',
        categorical_min_counts=MIN_COUNTS,weighting='Unweighted across all arms; tuning equal-mean five-fold RMSE in original PRISM target units. No outer-test selection of categories, imputation values, scalers or hyperparameters.',
        controls='Reuse 120 verified gene-only models, newly fit 120 gene+culture models and 60 culture-only models. Same source targets, RNA values, outer/inner masks and initialization seed triplets. Native unweighted MLP replaces the unit-weight adapter; numerical parity checked independently.',
        primary_comparisons=[[arm_name(f,p,'culture'),arm_name(f,p,'expression')] for f in FAMILIES for p in [5,64]],secondary_comparisons=[[arm_name(f,p,'culture'),arm_name(f,0,'culture')] for f in FAMILIES for p in [5,64]]+[[arm_name(f,5,'culture'),arm_name(f,64,'culture')] for f in FAMILIES],
        primary_metrics=old['primary_metrics'],secondary_metrics=old['secondary_metrics'],uncertainty=dict(count=5000,seed=2026111401,unit=old['uncertainty']['unit']),new_outer_models=180,reused_outer_models=120,total_outer_models=300,workers=WORKERS,
        limitations=old['limitations']+['Adding culture covariates measures prediction gain under these shared lineage-stratified splits; it does not isolate causal growth-condition effects or demonstrate transfer to unseen media.','This experiment is exploratory; prior response/model results have informed the ongoing investigation.'],
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__),data_sha256={name:sha(OUT/name) for name in ['panel_inputs.npz','shared_splits.npz','culture_metadata.tsv']},scripts={name:sha(OUT/name) for name in ['run_comparison.py','culture_features.py','fit_helpers.py','validate_preflight.py','evaluate.py','metric_helpers.py','model_helpers.py','weighted_mlp.py','make_splits.py']})
    pp=OUT/'protocol.json'
    if pp.exists():prev=json.load(open(pp));p['created_utc']=prev['created_utc'];assert prev==clean(p)
    else:dump(pp,p)
    ph=sha(pp)
    for f in FAMILIES:
        for panel in [5,64]:
            for r in range(20):
                original=OUT/'source_metadata'/f'{f}__p{panel}__r{r+1:02d}.json';m=json.load(open(original));m.update(view='expression',reused=True,protocol_sha256=ph,original_metadata_sha256=sha(original),source_protocol_sha256=sha(SOURCE/'protocol.json'));dest=root_for(f,panel,'expression',r).with_suffix('.json')
                if dest.exists():assert json.load(open(dest))==clean(m)
                else:dump(dest,m)
    dump(OUT/'source_manifest.json',dict(files=records));return ph

def main():
    begin=time.monotonic();ph=prepare();initialize();v=json.load(open(OUT/'preflight_validation.json'));assert v['passed'] and v['protocol_sha256']==ph
    with futures.ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        jobs=[pool.submit(fit_job,p,f,r,ph) for r in range(20) for p in PANELS for f in FAMILIES];done=0
        for future in futures.as_completed(jobs):
            m=future.result();done+=1;elapsed=time.monotonic()-begin;v=dict(phase='fitting',completed=done,total=180,reused=120,elapsed_seconds=elapsed,estimated_remaining_seconds=elapsed/done*(180-done));dump(OUT/'progress.json',v)
            if done%18==0:print(json.dumps(v),flush=True)
    p=json.load(open(OUT/'protocol.json'));assert sha(OUT/'protocol.json')==ph
    for name,h in p['scripts'].items():assert sha(OUT/name)==h
    for name,h in p['data_sha256'].items():assert sha(OUT/name)==h
    for r in json.load(open(OUT/'source_manifest.json'))['files']:assert sha(WS/r['path'])==r['sha256']
    dump(OUT/'fit_completion.json',dict(complete=True,new_outer_models=180,reused_outer_models=120,total_outer_models=300,elapsed_seconds=time.monotonic()-begin));dump(OUT/'progress.json',dict(phase='fits_complete',completed=180,total=180,reused=120));print('ALL_FITS_COMPLETE',flush=True)
if __name__=='__main__':main()
