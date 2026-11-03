#!/usr/bin/env python3
"""Training-local NN importance selection and nested top-10/top-20 comparison."""
import argparse
import concurrent.futures as futures
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
from sklearn.preprocessing import StandardScaler

from make_splits import inner_folds, inner_rows
from model_helpers import fitted_neural

OUT = Path(__file__).resolve().parent
WS = OUT.parent.parent
REGISTRY = WS/'outputs/retained_64_gene_models/model_registry.json'
BASE = {'CTRP':WS/'outputs/erastin_elastic_net_comparison',
        'PRISM':WS/'outputs/erastin_prism_direct_comparison'}
TOPK = [10, 20]
WORKERS = 8
MASTER_SEED = 2026110207
STATE = {}


def sha(path):
    with Path(path).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def clean(v):
    if isinstance(v, dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v, (tuple,list,np.ndarray)): return [clean(x) for x in v]
    if isinstance(v, np.generic): return clean(v.item())
    if isinstance(v, float) and not np.isfinite(v): return None
    return v


def dump(path, value):
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def initialize():
    for ds, root in BASE.items():
        a=np.load(root/'panel_inputs.npz'); sp=np.load(root/'shared_splits.npz')
        STATE[ds]=dict(x=a['x_64'], y=a['y'], genes=a['genes_64'], ids=a['sample_ids'],
            groups=a['group_ids'], lineage=a['lineage'], test=sp['outer_test'], inner=sp['inner_fold'],
            nn_seeds=sp['neural_initialization_seeds'], grid=json.load(open(root/'protocol.json'))['neural_grid'])
        for key, akey in [('sample_ids','ids'),('group_ids','groups'),('lineage','lineage')]:
            np.testing.assert_array_equal(sp[key],STATE[ds][akey])


def prepare():
    initialize(); registry=json.load(open(REGISTRY)); records=[]
    for ds, root in BASE.items():
        (OUT/ds/'folds').mkdir(parents=True,exist_ok=True)
        co=registry['cohorts'][ds]
        for file in co['input_and_validation_files'].values():
            assert sha(WS/file['path'])==file['sha256']
            records.append(file)
        for entry in co['models']['neural_network']['outer_models']:
            for file in entry['files'].values():
                assert sha(WS/file['path'])==file['sha256']
                records.append(file)
        np.testing.assert_array_equal(STATE[ds]['genes'],registry['genes_in_input_order'])
        assert len(co['models']['neural_network']['outer_models'])==20
    records.append(dict(path=str(REGISTRY.relative_to(WS)),sha256=sha(REGISTRY)))
    helperroot=WS/'outputs/erastin_genomewide_enet_nn'
    helpers=['model_helpers.py','weighted_mlp.py','make_splits.py','evaluate.py','metric_helpers.py']
    for n in helpers:
        assert sha(OUT/n)==sha(helperroot/n)
        records.append(dict(path=str((helperroot/n).relative_to(WS)),sha256=sha(helperroot/n)))
    p=dict(status='frozen_before_training_local_topk_benchmark',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        assays=list(BASE), panel=64, top_k=TOPK, arms=['top10','top20','full64'], workers=WORKERS,
        outer='Reuse the exact saved 20 lineage-stratified identity-group 80/20 holdouts separately in CTRP and PRISM; Cervix excluded. All evaluated lineages in both outer sets.',
        inner='Reuse the five saved inner tuning folds. Protected singleton training lineages remain training (-2) and never validation.',
        selector='Within each INNER-TRAINING subset separately, make three response-blind lineage/identity-group folds. Fit full64 three-member NNs on each selector-training fold for every candidate in the original assay NN grid. Choose the candidate with minimum mean selector-fold RMSE. Rank all 64 genes by pooled out-of-fold expected MSE increase after exhaustive same-lineage selector-training-donor replacement, using the selected candidate models. No inner-validation or outer-test data enter this function. Protected singleton-lineage groups are excluded as ranking receivers, retained as training observations. Fit the same selector afresh on complete OUTER TRAINING for the final panels.',
        ranking='Descending exact mean delta MSE on selector out-of-fold receivers. Training-donor and receiver identity groups disjoint. Ties use original panel order. Take exactly top10/top20, including nonpositive ranks if necessary. Preserve canonical input order in downstream fits; top10 is a subset of top20 within each selector context.',
        selector_folds=3, selector_seed='SeedSequence([2026110207, assay_index, repeat_zero_based, context_0_to_5]); context 0..4 = saved inner fold; 5 = outer-training selector.',
        predictor='For each K separately, tune original assay NN grid by mean five-fold RMSE using fold-specific selected genes, then refit the selected NN parameters on the outer-training selected panel. Each fit averages the original three NN initialization seeds. All X/y scaling uses current fit training only. No lineage prediction covariates, no culture information, unweighted.',
        neural_grids={ds:s['grid'] for ds,s in STATE.items()},
        neural_fixed=dict(activation='tanh',solver='lbfgs',max_iter=2000,max_fun=40000,tol=.001,early_stopping=False,members=3),
        full64='Reuse exact hash-verified retained full64 models, inner predictions and outer predictions. Validate selected parameters and predictions; no panel selection for this arm.',
        primary_metric='Mean original-unit pooled held-out RMSE over 20 repeats, separately by assay.',
        secondary_metrics=['Spearman','Pearson','MAE','R2','equal-lineage RMSE','equal-large-lineage RMSE'],
        primary_comparisons=[['top10','full64'],['top20','full64']], secondary_comparisons=[['top10','top20']],
        uncertainty=dict(count=5000,seed=2026110208,unit='Paired whole-lineage bootstrap; keep all arms and overlapping holdouts together. Conditional on saved fits; no retraining or multiple-comparison correction.'),
        numerical_policy='Abort on ConvergenceWarning or other runtime/numerical warnings; record the known SciPy optimizer deprecation warning. Resume only checksum-verified checkpoints with this protocol hash.',
        limitations=['Exploratory, previously inspected cohorts; fold-local selection prevents direct test leakage but does not undo earlier adaptive model exploration.',
            'Results evaluate a panel-selection procedure; each repeat can select a different gene list. Consensus frequencies are descriptive, not a tested fixed panel.',
            'Within-lineage replacement can disrupt correlated expression. Predictive contribution is not causal ferroptosis relevance.',
            'CTRP and PRISM have different response provenance/scales and overlapping cell lines. No independent-assay replication or unseen-lineage claim.'],
        scripts={n:sha(OUT/n) for n in ['benchmark.py']+helpers},
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__))
    pp=OUT/'protocol.json'
    if pp.exists():
        old=json.load(open(pp));p['created_utc']=old['created_utc'];assert old==clean(p)
    else: dump(pp,p)
    source=dict(files=records)
    if (OUT/'source_manifest.json').exists(): assert json.load(open(OUT/'source_manifest.json'))==source
    else: dump(OUT/'source_manifest.json',source)
    return sha(pp)


def verify():
    for n,h in json.load(open(OUT/'protocol.json'))['scripts'].items(): assert sha(OUT/n)==h,n
    for row in json.load(open(OUT/'source_manifest.json'))['files']: assert sha(WS/row['path'])==row['sha256'],row['path']


def check_solver(diag):
    caught=[w for member in diag['members'] for w in member['warnings']]
    assert all(w['category']=='DeprecationWarning' and 'disp' in w['message'] for w in caught),caught


def prediction(model,x):
    z=model['x_scaler'].transform(x)
    return np.mean([model['y_scaler'].inverse_transform(m.predict(z)[:,None]).ravel() for m in model['members']],axis=0)


def replacement(model,x,y,lineage,groups,tr,va):
    """Exact held-out reliance, accepting only a selector's local training pool."""
    sx,sy=model['x_scaler'],model['y_scaler'];z=sx.transform(x)
    np.testing.assert_allclose(sx.mean_,x[tr].mean(axis=0),rtol=0,atol=1e-12)
    np.testing.assert_allclose(sy.mean_,y[tr].mean(),rtol=0,atol=1e-12)
    pred=prediction(model,x[va]);base=(pred-y[va])**2;delta=np.empty((len(va),64))
    latent=[(z[va]@m.coefs_[0]+m.intercepts_[0],m.coefs_[0],m.coefs_[1].ravel(),m.intercepts_[1][0]) for m in model['members']]
    checks=[];maxerror=0.
    for label in np.unique(lineage[va]):
        rec=np.flatnonzero(lineage[va]==label);don=tr[lineage[tr]==label]
        assert len(don)>0 and set(groups[don]).isdisjoint(groups[va[rec]])
        for j in range(64):
            diff=z[don,j][None,:]-z[va[rec],j][:,None];p=np.zeros(diff.shape)
            for h,w1,w2,b in latent:
                p+=((np.tanh(h[rec,None,:]+diff[:,:,None]*w1[j])@w2+b)*sy.scale_[0]+sy.mean_[0])/len(latent)
            delta[rec,j]=np.mean((p-y[va[rec],None])**2,axis=1)-base[rec]
            if j in [0,63]:
                xx=np.repeat(x[va[rec[:1]]],len(don),axis=0);xx[:,j]=x[don,j]
                err=float(np.max(np.abs(prediction(model,xx)-p[0])))
                maxerror=max(maxerror,err);assert err<1e-11
        checks.append(dict(lineage=str(label),donors=len(don),receivers=len(rec)))
    return delta,pred,dict(lineage_checks=checks,max_direct_prediction_error=maxerror)


def rank_local(x,y,lineage,groups,grid,seeds,split_seed):
    """No caller validation/test rows, covariates or responses are available here."""
    n=len(y);assignment=inner_folds(lineage,groups,np.zeros(n,bool),split_seed,k=3)
    candidates=np.full((len(grid),n),np.nan);loss=np.empty((len(grid),3));models={};diagnostics=[]
    for fold in range(3):
        tr,va=inner_rows(assignment,fold);assert len(va)>0
        for j,params in enumerate(grid):
            pred,model,diag,_=fitted_neural(params,x,y,tr,va,seeds,np.ones(len(tr)),retain=True)
            check_solver(diag);candidates[j,va]=pred;loss[j,fold]=np.sqrt(np.mean((pred-y[va])**2))
            models[j,fold]=model;diagnostics.append(dict(fold=fold,candidate=j,**diag))
    chosen=int(np.argmin(loss.mean(axis=1)));delta=np.full((n,64),np.nan);audit=[];selected_models=[]
    for fold in range(3):
        tr,va=inner_rows(assignment,fold)
        d,p,check=replacement(models[chosen,fold],x,y,lineage,groups,tr,va)
        np.testing.assert_allclose(p,candidates[chosen,va],rtol=0,atol=1e-12)
        delta[va]=d;audit.append(dict(fold=fold,training_n=len(tr),validation_n=len(va),**check));selected_models.append(models[chosen,fold])
    valid=assignment>=0
    assert np.isfinite(delta[valid]).all() and np.isnan(delta[~valid]).all()
    importance=delta[valid].mean(axis=0);order=np.argsort(-importance,kind='stable')
    arrays=dict(selector_assignment=assignment,importance=importance,ranking=order,receiver_delta_mse=delta,
                candidate_predictions=candidates,selector_fold_rmse=loss,selected_candidate=np.array(chosen))
    meta=dict(training_n=n,ranking_receiver_n=int(valid.sum()),protected_n=int((~valid).sum()),
        selected_parameters=grid[chosen],diagnostics=diagnostics,audit=audit)
    return arrays,meta,selected_models


def selector(ds,r,context,rows,ph):
    root=OUT/ds/'folds'/f'selector__r{r+1:02d}__c{context}'
    if root.with_suffix('.json').exists():
        m=json.load(open(root.with_suffix('.json')));assert m['protocol_sha256']==ph
        for ext in ['npz','joblib']: assert sha(root.with_suffix('.'+ext))==m[ext+'_sha256']
        a=np.load(root.with_suffix('.npz'));np.testing.assert_array_equal(a['context_training_rows'],rows)
        return a['ranking']
    s=STATE[ds];start=time.monotonic()
    seed=int(np.random.SeedSequence([MASTER_SEED,list(BASE).index(ds),r,context]).generate_state(1)[0])
    a,m,models=rank_local(s['x'][rows],s['y'][rows],s['lineage'][rows],s['groups'][rows],s['grid'],s['nn_seeds'][r],seed)
    np.savez_compressed(root.with_suffix('.npz'),context_training_rows=rows,**a)
    joblib.dump(models,root.with_suffix('.joblib'),compress=3)
    dump(root.with_suffix('.json'),dict(dataset=ds,repeat=r+1,context=context,protocol_sha256=ph,
        selector_seed=seed,training_rows_sha256=hashlib.sha256(rows.tobytes()).hexdigest(),
        **m,seconds=time.monotonic()-start,npz_sha256=sha(root.with_suffix('.npz')),joblib_sha256=sha(root.with_suffix('.joblib'))))
    return a['ranking']


def fit_repeat(ds,r,ph):
    out=OUT/ds/'folds';done=out/f'repeat__r{r+1:02d}_complete.json'
    if done.exists():
        m=json.load(open(done));assert m['protocol_sha256']==ph
        for f in m['files']: assert sha(OUT/f['path'])==f['sha256']
        return m
    start=time.monotonic();s=STATE[ds];grid=s['grid'];N=len(s['y']);C=len(grid)
    val=np.full((2,C,N),np.nan);loss=np.empty((2,C,5));masks=np.zeros((5,2,64),bool)
    files=[]
    for fold in range(5):
        root=out/f'inner__r{r+1:02d}__f{fold}';tr,va=inner_rows(s['inner'][r],fold)
        order=selector(ds,r,fold,tr,ph)
        if root.with_suffix('.json').exists():
            m=json.load(open(root.with_suffix('.json')));assert m['protocol_sha256']==ph
            assert sha(root.with_suffix('.npz'))==m['npz_sha256']
            a=np.load(root.with_suffix('.npz'));np.testing.assert_array_equal(a['training_rows'],tr);np.testing.assert_array_equal(a['validation_rows'],va)
            val[:,:,va]=a['validation_predictions'];loss[:,:,fold]=a['fold_rmse'];masks[fold]=a['feature_masks']
        else:
            diagnostics=[]
            for ki,k in enumerate(TOPK):
                indices=np.sort(order[:k]);masks[fold,ki,indices]=True;x=s['x'][:,indices]
                for j,params in enumerate(grid):
                    p,_,diag,_=fitted_neural(params,x,s['y'],tr,va,s['nn_seeds'][r],np.ones(len(tr)),retain=False)
                    check_solver(diag);val[ki,j,va]=p;loss[ki,j,fold]=np.sqrt(np.mean((p-s['y'][va])**2))
                    diagnostics.append(dict(panel=k,candidate=j,**diag))
            np.savez_compressed(root.with_suffix('.npz'),training_rows=tr,validation_rows=va,
                validation_predictions=val[:,:,va],fold_rmse=loss[:,:,fold],feature_masks=masks[fold])
            dump(root.with_suffix('.json'),dict(dataset=ds,repeat=r+1,fold=fold,protocol_sha256=ph,
                diagnostics=diagnostics,npz_sha256=sha(root.with_suffix('.npz'))))
        for ki,k in enumerate(TOPK): np.testing.assert_array_equal(np.flatnonzero(masks[fold,ki]),np.sort(order[:k]))
        files.extend(root.with_suffix('.'+e) for e in ['npz','json'])
    valid=s['inner'][r]>=0
    assert np.isfinite(val[:,:,valid]).all() and np.isnan(val[:,:,~valid]).all()
    tr,te=np.flatnonzero(~s['test'][r]),np.flatnonzero(s['test'][r])
    assert set(s['groups'][tr]).isdisjoint(s['groups'][te])
    order=selector(ds,r,5,tr,ph);pred=np.empty((3,len(te)));outer_masks=np.zeros((2,64),bool);metadata=[]
    for ki,k in enumerate(TOPK):
        chosen=int(np.argmin(loss[ki].mean(axis=1)));params=grid[chosen];indices=np.sort(order[:k]);outer_masks[ki,indices]=True
        p,model,diag,members=fitted_neural(params,s['x'][:,indices],s['y'],tr,te,s['nn_seeds'][r],np.ones(len(tr)),retain=True)
        check_solver(diag);pred[ki]=p;model.update(feature_indices=indices,genes=s['genes'][indices])
        root=out/f'top{k}__r{r+1:02d}';joblib.dump(model,root.with_suffix('.joblib'),compress=3)
        np.savez_compressed(root.with_suffix('.npz'),train_rows=tr,test_rows=te,test_prediction=p,member_test_predictions=members,feature_indices=indices)
        md=dict(panel=k,selected_candidate=chosen,selected_parameters=params,mean_inner_rmse=float(loss[ki,chosen].mean()),genes=s['genes'][indices],fit_diagnostics=diag)
        dump(root.with_suffix('.json'),dict(dataset=ds,repeat=r+1,protocol_sha256=ph,**md,
            npz_sha256=sha(root.with_suffix('.npz')),joblib_sha256=sha(root.with_suffix('.joblib'))));metadata.append(md)
        files.extend(root.with_suffix('.'+e) for e in ['npz','json','joblib'])
    fullroot=BASE[ds]/'folds'/f'neural_network__p64__r{r+1:02d}';base=np.load(fullroot.with_suffix('.npz'));bm=json.load(open(fullroot.with_suffix('.json')))
    np.testing.assert_array_equal(tr,base['train_rows']);np.testing.assert_array_equal(te,base['test_rows'])
    chosen=int(np.argmin(base['inner_fold_rmse'].mean(axis=1)));assert grid[chosen]==bm['selected_parameters']
    pred[2]=base['test_prediction']
    root=out/f'comparison__r{r+1:02d}'
    np.savez_compressed(root.with_suffix('.npz'),train_rows=tr,test_rows=te,test_predictions=pred,
        feature_masks=outer_masks,inner_feature_masks=masks,inner_validation_predictions=val,inner_fold_rmse=loss)
    dump(root.with_suffix('.json'),dict(dataset=ds,repeat=r+1,protocol_sha256=ph,panels=metadata,
        baseline_source=str(fullroot.relative_to(WS)),npz_sha256=sha(root.with_suffix('.npz'))))
    files.extend(root.with_suffix('.'+e) for e in ['npz','json'])
    for context in range(6): files.extend((out/f'selector__r{r+1:02d}__c{context}').with_suffix('.'+e) for e in ['npz','json','joblib'])
    m=dict(dataset=ds,repeat=r+1,protocol_sha256=ph,seconds=time.monotonic()-start,
        genes={str(k):s['genes'][outer_masks[j]].tolist() for j,k in enumerate(TOPK)},
        files=[dict(path=str(p.relative_to(OUT)),sha256=sha(p)) for p in files])
    dump(done,m);return m


def preflight(ph):
    checks=[]
    for ds,s in STATE.items():
        for r in range(20):
            tr,te=np.flatnonzero(~s['test'][r]),np.flatnonzero(s['test'][r])
            assert set(s['groups'][tr]).isdisjoint(s['groups'][te])
            assert set(s['lineage'][tr])==set(s['lineage'][te])
            assert set(np.unique(s['inner'][r])) <= {-2,-1,0,1,2,3,4}
            for fold in range(5):
                a,b=inner_rows(s['inner'][r],fold)
                assert set(a)|set(b)==set(tr) and set(s['groups'][a]).isdisjoint(s['groups'][b])
        # Refit an original full64 fit and inner fold independently with copied code.
        root=BASE[ds]/'folds'/'neural_network__p64__r01';old=np.load(root.with_suffix('.npz'));meta=json.load(open(root.with_suffix('.json')))
        errors=[]
        for tr,va,params,expected in [(old['train_rows'],old['test_rows'],meta['selected_parameters'],old['test_prediction'])]+[
                (*inner_rows(s['inner'][0],0),params,old['inner_validation_predictions'][j,inner_rows(s['inner'][0],0)[1]]) for j,params in enumerate(s['grid'])]:
            p,_,diag,_=fitted_neural(params,s['x'],s['y'],tr,va,s['nn_seeds'][0],np.ones(len(tr)),retain=False)
            check_solver(diag);error=float(np.max(np.abs(p-expected)));assert error<1e-11;errors.append(error)
        # Actual exclusion-invariance check for selection in the first inner fold.
        tr,va=inner_rows(s['inner'][0],0)
        seed=int(np.random.SeedSequence([MASTER_SEED,list(BASE).index(ds),0,0]).generate_state(1)[0])
        a,_,_=rank_local(s['x'][tr],s['y'][tr],s['lineage'][tr],s['groups'][tr],s['grid'],s['nn_seeds'][0],seed)
        xx=s['x'].copy();yy=s['y'].copy();outside=np.ones(len(yy),bool);outside[tr]=False
        xx[outside]=987.;yy[outside]=-321.
        b,_,_=rank_local(xx[tr],yy[tr],s['lineage'][tr],s['groups'][tr],s['grid'],s['nn_seeds'][0],seed)
        np.testing.assert_array_equal(a['ranking'],b['ranking']);np.testing.assert_array_equal(a['importance'],b['importance'])
        checks.append(dict(dataset=ds,baseline_refit_max_error=max(errors),excluded_response_and_expression_mutation_has_no_selection_effect=True,
            ranks_finite=bool(np.isfinite(a['importance']).all()),top10_genes=s['genes'][a['ranking'][:10]].tolist()))
    verify();dump(OUT/'preflight_validation.json',dict(passed=True,protocol_sha256=ph,all_20_splits_audited=True,checks=checks))
    print(json.dumps(clean(dict(preflight_passed=True,checks=checks))),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--preflight-only',action='store_true');args=parser.parse_args()
    ph=prepare();verify()
    if not (OUT/'preflight_validation.json').exists(): preflight(ph)
    assert json.load(open(OUT/'preflight_validation.json'))['protocol_sha256']==ph
    if args.preflight_only:return
    start=time.monotonic()
    with futures.ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        jobs=[pool.submit(fit_repeat,ds,r,ph) for r in range(20) for ds in BASE]
        for done,f in enumerate(futures.as_completed(jobs),1):
            m=f.result();elapsed=time.monotonic()-start
            p=dict(phase='nested_NN_topk',completed=done,total=40,elapsed_seconds=elapsed,
                estimated_remaining_seconds=elapsed/done*(40-done),last_dataset=m['dataset'],last_repeat=m['repeat'])
            dump(OUT/'progress.json',p);print(json.dumps(p),flush=True)
    verify();dump(OUT/'completion.json',dict(complete=True,repeat_comparisons=40,new_outer_ensembles=80,
        selector_contexts=240,protocol_sha256=ph,seconds=time.monotonic()-start))
    print('TOPK_NESTED_FITS_COMPLETE',flush=True)


if __name__=='__main__':main()
