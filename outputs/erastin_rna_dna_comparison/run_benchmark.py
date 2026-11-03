#!/usr/bin/env python3
"""Four omics arms, three retained model families, 20 shared nested holdouts."""
import argparse
import concurrent.futures as cf
import hashlib
import json
import multiprocessing as mp
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn
import modeling as m
from make_splits import inner_rows

OUT=Path(__file__).resolve().parent;WS=OUT.parent.parent
BASE={'CTRP':WS/'outputs/erastin_elastic_net_comparison','PRISM':WS/'outputs/erastin_prism_direct_comparison'}
FAMILIES=['ridge','elastic_net','neural_network'];WORKERS=8;STATE={}


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v


def dump(path,v):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(clean(v),indent=2,allow_nan=False)+'\n');tmp.replace(path)


def initialize():
    cov=json.load(open(OUT/'coverage_summary.json'))
    for ds,root in BASE.items():
        a=np.load(OUT/ds/'inputs.npz');sp=np.load(OUT/ds/'shared_splits.npz')
        STATE[ds]={k:a[k] for k in a.files}
        STATE[ds].update(test=sp['outer_test'],inner=sp['inner_fold'],nn_seeds=sp['neural_initialization_seeds'],
            source=json.load(open(root/'protocol.json')),reuse_rna=cov['assays'][ds]['identical_original_cohort'])
        for k in ['sample_ids','group_ids','lineage']:np.testing.assert_array_equal(a[k],sp[k])


def arm_name(family,arm):return family+'__'+arm


def prepare():
    initialize();assert json.load(open(OUT/'preparation_validation.json'))['passed']
    manifest=json.load(open(OUT/'source_manifest.json'))
    for row in manifest['files']:assert sha(WS/row['path'])==row['sha256']
    inputs={}
    for ds in BASE:
        (OUT/ds/'folds').mkdir(exist_ok=True)
        inputs[ds]={n:sha(OUT/ds/n) for n in ['inputs.npz','shared_splits.npz']}
    scripts=['run_benchmark.py','modeling.py','prepare_data.py','download_data.py','weighted_mlp.py','enet_reference.py','make_splits.py','evaluate.py','metric_helpers.py']
    p=dict(status='frozen_before_four_arm_multimodal_fits',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        arms=m.ARMS,families=FAMILIES,assays=list(BASE),repeats=20,workers=WORKERS,
        release='DepMap 24Q4 Public v1, same pinned release as RNA',
        cohorts=json.load(open(OUT/'coverage_summary.json'))['assays'],
        splits='Project original 20 outer test masks and five inner assignments onto each common RNA/DNA cohort. All arms use identical cells and folds. No inner assignments needed repair; every evaluated lineage occurs in both outer train and test. CTRP has 739 lines; PRISM 386.',
        rna='Retain all original 64 genes, in canonical order, with current-training StandardScaler. Unweighted RNA-only baseline, no culture or lineage prediction covariates.',
        mutation='Six prespecified annotation features: KEAP1/BAP1/TP53 LikelyLoF and NFE2L2/TP53/KRAS hotspot. Official genotyped 0 stays 0 and 1/2 becomes 1. Missing genotype never becomes wild type. Value 2 is summed AF >0.95, not a biallelic-loss call. LikelyLoF/hotspot are annotations, not experimentally verified functional status.',
        mutation_preprocessing='Within current fit training only, require >=5 carriers and >=5 non-carriers. Center binary variables without dividing by their SD. Tune multiplication by 1 or 2 inside inner CV. TP53 damaging/hotspot categories can overlap; no universal direction imposed.',
        cnv='66 available genes: 63 of the RNA genes plus KEAP1, TP53, KRAS. CBS is absent from the official CNV header and remains RNA-only. Transform relative copy ratio as log2(x+1), matching the portal transform; no amplification/deep-deletion or biallelic labels inferred.',
        cnv_preprocessing='Within current fit training only, require >=80% finite values per CN feature, median-impute from training, drop training constants, fit StandardScaler. Tune CN block multiplication by 0.25 or 1 inside inner CV. Primary cohort requires >=80% of the available 66 CN features per sample, using no outcomes.',
        tuning='Separate original assay-specific grids and mean of five inner-fold RMSEs in original target units. Cartesian product with prespecified block scales; first grid index resolves ties. Complete preprocessing refit in every inner training fold and outer training set.',
        grids={ds:{arm:{f:m.grid(f,arm,s['source']) for f in FAMILIES} for arm in m.ARMS} for ds,s in STATE.items()},
        neural_fixed=dict(activation='tanh',solver='lbfgs',max_iter=2000,max_fun=40000,tol=.001,members=3,seeds='Original three initialization seeds for each repeat, reused across arms'),
        elastic_net_fixed=dict(path_iteration_limits=m.ENET_PATH_LIMITS,native_fit_iteration_batch=m.ENET_MAX_ITER,native_fit_max_batches=10,tol=m.enet.TOL,coordinate_descent='cyclic',kkt_tolerance=1e-6,
            preflight_amendment='After training-only preflight convergence warnings at 50000 and a cold native-fit warning at 500000, use fixed convergence-only retries up to 5 million iterations. Same objective/grid/tolerance. No production fits or outer outcomes inspected.'),
        reuse='PRISM RNA-only source models/predictions reused exactly. CTRP RNA-only controls refit on the same 739 lines as DNA arms; never compare a DNA subset directly against a larger RNA cohort.',
        numerical_policy='Abort on solver convergence/numerical warnings; allow only the known SciPy L-BFGS deprecation warning. All resumed checkpoints require this protocol hash and verified checksums.',
        primary_comparisons=[[arm_name(f,a),arm_name(f,'rna')] for f in FAMILIES for a in m.ARMS[1:]],
        secondary_comparisons=[[arm_name(f,a),arm_name('neural_network','rna')] for f in FAMILIES if f!='neural_network' for a in m.ARMS[1:]],
        metrics=['pooled RMSE (primary)','Spearman','Pearson','MAE','R2','equal-lineage RMSE','equal-large-lineage RMSE'],
        uncertainty=dict(count=5000,seed=2026110301,unit='Paired whole-lineage bootstrap; preserve all arms and overlapping repeats together. Conditional pointwise intervals; no retraining or multiple-comparison correction.'),
        limitations=['Previously inspected cohorts, so exploratory despite correct nesting.','Rare mutation classes have limited power, especially BAP1; training-only frequency filter can omit them.',
            'A null result applies to this compact feature set, encoding and finite tuning grid, not to all genomics or multi-omics approaches.',
            'DNA and RNA are release-matched but can originate from different profiles and collection conditions.','CTRP endpoint-provenance limitation remains; assays are separate and overlap in cell lines.'],
        retained_registry_unchanged=True,input_hashes=inputs,source_manifest_sha256=sha(OUT/'source_manifest.json'),
        scripts={n:sha(OUT/n) for n in scripts},versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__))
    if (OUT/'protocol.json').exists():
        old=json.load(open(OUT/'protocol.json'));p['created_utc']=old['created_utc'];assert old==clean(p)
    else:dump(OUT/'protocol.json',p)
    return sha(OUT/'protocol.json')


def verify():
    p=json.load(open(OUT/'protocol.json'))
    for n,h in p['scripts'].items():assert sha(OUT/n)==h,n
    assert sha(OUT/'source_manifest.json')==p['source_manifest_sha256']
    for ds,files in p['input_hashes'].items():
        for n,h in files.items():assert sha(OUT/ds/n)==h


def fit_one(ds,r,family,arm,ph):
    s=dict(STATE[ds]);s['current_seeds']=s['nn_seeds'][r];root=OUT/ds/'folds'/f'{family}__{arm}__r{r+1:02d}'
    if root.with_suffix('.json').exists():
        md=json.load(open(root.with_suffix('.json')));assert md['protocol_sha256']==ph
        assert sha(root.with_suffix('.npz'))==md['npz_sha256']
        if not md['reused']:assert sha(root.with_suffix('.joblib'))==md['joblib_sha256']
        return md
    start=time.monotonic();grid=m.grid(family,arm,s['source']);N=len(s['y']);val=np.full((len(grid),N),np.nan);loss=np.empty((len(grid),5));files=[]
    reuse=bool(arm=='rna' and s['reuse_rna'])
    source=BASE[ds]/'folds'/f'{family}__p64__r{r+1:02d}'
    old=np.load(source.with_suffix('.npz')) if reuse else None
    for fold in range(5):
        inner=root.parent/(root.name+f'__inner{fold}');tr,va=inner_rows(s['inner'][r],fold)
        if inner.with_suffix('.json').exists():
            md=json.load(open(inner.with_suffix('.json')));assert md['protocol_sha256']==ph
            assert sha(inner.with_suffix('.npz'))==md['npz_sha256']
            a=np.load(inner.with_suffix('.npz'));np.testing.assert_array_equal(a['training_rows'],tr);np.testing.assert_array_equal(a['validation_rows'],va)
            v=a['validation_predictions'];d=md['preprocessing']
        else:
            if reuse:
                v=old['inner_validation_predictions'][:,va]
                pp=m.OmicsPreprocessor(arm).fit(s['rna'][tr],s['mutation'][tr],s['cnv'][tr])
                d=dict(reused=True,training_n=len(tr),mutation_keep=pp.mutation_keep,cnv_keep=pp.cnv_keep,feature_names=pp.names(s))
            else:v,d=m.candidates(family,arm,s,tr,va)
            np.savez_compressed(inner.with_suffix('.npz'),training_rows=tr,validation_rows=va,validation_predictions=v)
            dump(inner.with_suffix('.json'),dict(dataset=ds,repeat=r+1,model_family=family,arm=arm,fold=fold,protocol_sha256=ph,preprocessing=d,npz_sha256=sha(inner.with_suffix('.npz'))))
        assert v.shape==(len(grid),len(va));val[:,va]=v;loss[:,fold]=np.sqrt(np.mean((v-s['y'][va][None,:])**2,axis=1))
        files.extend(inner.with_suffix('.'+ext) for ext in ['npz','json'])
    valid=s['inner'][r]>=0;assert np.isfinite(val[:,valid]).all() and np.isnan(val[:,~valid]).all()
    chosen=int(np.argmin(loss.mean(axis=1)));params=grid[chosen];tr,te=np.flatnonzero(~s['test'][r]),np.flatnonzero(s['test'][r])
    if reuse:
        original=json.load(open(source.with_suffix('.json')))
        assert {k:v for k,v in params.items() if not k.endswith('_scale')}==original['selected_parameters']
        np.testing.assert_array_equal(old['train_rows'],tr);np.testing.assert_array_equal(old['test_rows'],te)
        pred=old['test_prediction'];diag=dict(reused=True,features=64,mutations=0,copy_numbers=0);feature_names=s['genes']
    else:
        pred,model,diag=m.fit_selected(family,arm,s,tr,te,params);feature_names=model['feature_names'];joblib.dump(model,root.with_suffix('.joblib'),compress=3)
        files.append(root.with_suffix('.joblib'))
    np.savez_compressed(root.with_suffix('.npz'),train_rows=tr,test_rows=te,test_prediction=pred,inner_fold_rmse=loss,
        inner_validation_predictions=val,feature_names=feature_names)
    md=dict(dataset=ds,repeat=r+1,family=family,arm=arm,protocol_sha256=ph,reused=reuse,source_model=str(source.relative_to(WS)) if reuse else None,
        selected_candidate=chosen,selected_parameters=params,mean_inner_rmse=float(loss[chosen].mean()),feature_names=feature_names,
        fit_diagnostics=diag,seconds=time.monotonic()-start,npz_sha256=sha(root.with_suffix('.npz')),
        joblib_sha256=sha(root.with_suffix('.joblib')) if not reuse else None)
    dump(root.with_suffix('.json'),md);files.extend(root.with_suffix('.'+ext) for ext in ['npz','json'])
    return md


def fit_repeat(ds,r,ph):
    root=OUT/ds/'folds';complete=root/f'repeat__r{r+1:02d}_complete.json'
    if complete.exists():
        md=json.load(open(complete));assert md['protocol_sha256']==ph
        for f in md['files']:assert sha(OUT/f['path'])==f['sha256']
        return md
    start=time.monotonic();results=[]
    for family in FAMILIES:
        for arm in m.ARMS:
            md=fit_one(ds,r,family,arm,ph);results.append(md)
            dump(OUT/ds/f'progress_r{r+1:02d}.json',dict(dataset=ds,repeat=r+1,outer_models_finished=len(results),total=12,last_model=arm_name(family,arm),elapsed_seconds=time.monotonic()-start))
    files=sorted(root.glob(f'*__r{r+1:02d}.*'))+sorted(root.glob(f'*__r{r+1:02d}__inner*.*'))
    # Complete manifest excludes itself and only covers immutable fit artifacts.
    md=dict(dataset=ds,repeat=r+1,protocol_sha256=ph,models=12,seconds=time.monotonic()-start,
        files=[dict(path=str(p.relative_to(OUT)),sha256=sha(p)) for p in files])
    dump(complete,md);return md


def preflight(ph):
    checks=[]
    for ds,s0 in STATE.items():
        s=dict(s0);s['current_seeds']=s['nn_seeds'][0];tr,va=inner_rows(s['inner'][0],0)
        # Entire preprocessing is invariant to arbitrary excluded-cohort changes.
        pp=m.OmicsPreprocessor('rna_mutation_cnv').fit(s['rna'][tr],s['mutation'][tr],s['cnv'][tr])
        altered={k:s[k].copy() for k in ['rna','mutation','cnv']};outside=np.ones(len(s['y']),bool);outside[tr]=False
        for k in altered:altered[k][outside]=1. if k=='mutation' else 999.
        qq=m.OmicsPreprocessor('rna_mutation_cnv').fit(*(altered[k][tr] for k in ['rna','mutation','cnv']))
        assert joblib.hash(pp)==joblib.hash(qq)
        np.testing.assert_array_equal(pp.mutation_keep, np.flatnonzero((s['mutation'][tr].sum(axis=0)>=5)&((len(tr)-s['mutation'][tr].sum(axis=0))>=5)))
        errors=[]
        for family in FAMILIES:
            # Check complete mixed-arm candidate paths and selected native refit.
            pred,_=m.candidates(family,'rna_mutation_cnv',s,tr,va)
            grid=m.grid(family,'rna_mutation_cnv',s['source']);indices=[0,len(grid)-1] if family!='neural_network' else [0]
            for j in indices:
                p,model,_=m.fit_selected(family,'rna_mutation_cnv',s,tr,va,grid[j])
                error=float(np.max(np.abs(p-pred[j])));limit=1e-7 if family=='elastic_net' else 1e-10
                assert error<limit,(ds,family,error);errors.append(dict(family=family,max_native_path_error=error))
        if s['reuse_rna']:
            for family in FAMILIES:
                source=BASE[ds]/'folds'/f'{family}__p64__r01';a=np.load(source.with_suffix('.npz'));md=json.load(open(source.with_suffix('.json')))
                p,_=m.candidates(family,'rna',s,tr,va)
                error=float(np.max(np.abs(p-a['inner_validation_predictions'][:,va])));assert error<1e-10,(family,error)
                params=dict(mutation_scale=0.,cnv_scale=0.,**md['selected_parameters'])
                test=np.flatnonzero(s['test'][0]);train=np.flatnonzero(~s['test'][0]);values,_,_=m.fit_selected(family,'rna',s,train,test,params)
                err=float(np.max(np.abs(values-a['test_prediction'])));assert err<1e-10,(family,err)
                errors.append(dict(family=family,baseline_inner_error=error,baseline_outer_error=err))
        checks.append(dict(dataset=ds,preprocessing_exclusion_invariance=True,training_only_carrier_filter_verified=True,checks=errors))
    verify();dump(OUT/'preflight_validation.json',dict(passed=True,protocol_sha256=ph,checks=checks));print(json.dumps(clean(dict(preflight_passed=True,checks=checks))),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    ph=prepare();verify()
    if not (OUT/'preflight_validation.json').exists():preflight(ph)
    assert json.load(open(OUT/'preflight_validation.json'))['protocol_sha256']==ph
    if args.preflight_only:return
    start=time.monotonic()
    with cf.ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        jobs=[pool.submit(fit_repeat,ds,r,ph) for r in range(20) for ds in BASE]
        for done,f in enumerate(cf.as_completed(jobs),1):
            md=f.result();elapsed=time.monotonic()-start
            p=dict(phase='four_arm_RNA_DNA',completed_repeats=done,total_repeats=40,elapsed_seconds=elapsed,
                estimated_remaining_seconds=elapsed/done*(40-done),last_dataset=md['dataset'],last_repeat=md['repeat'])
            dump(OUT/'progress.json',p);print(json.dumps(p),flush=True)
    verify();dump(OUT/'completion.json',dict(complete=True,repeat_comparisons=40,outer_models=480,protocol_sha256=ph,seconds=time.monotonic()-start))
    print('RNA_DNA_FITS_COMPLETE',flush=True)


if __name__=='__main__':main()
