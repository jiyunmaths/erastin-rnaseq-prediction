#!/usr/bin/env python3
"""Nested comparison of nonlinear and matched linear Gaussian-CPD networks."""
import concurrent.futures as futures
import hashlib
import json
import multiprocessing as mp
from pathlib import Path
import platform
import sys
import time

import joblib
import numpy as np
import sklearn

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
WS=OUT.parent.parent
sys.path.insert(0,str(OUT))
import spline_network as sn
FAMILIES=['linear_cpd','spline_cpd']
ALPHAS=[0.,1.,10.,100.,300.,1000.,3000.,10000.,1000000.]
STATE={}


def sha(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()


def clean(x):
    if isinstance(x,dict):return {str(k):clean(v) for k,v in x.items()}
    if isinstance(x,(list,tuple)):return [clean(v) for v in x]
    if isinstance(x,np.ndarray):return clean(x.tolist())
    if isinstance(x,np.integer):return int(x)
    if isinstance(x,(float,np.floating)):return float(x) if np.isfinite(x) else None
    if isinstance(x,np.bool_):return bool(x)
    return x


def dump(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)


def initialize():
    d=np.load(WS/'outputs/erastin_40gene_comparison/comparison_inputs.npz',allow_pickle=False)
    s=np.load(WS/'outputs/erastin_nn_repeated/shared_splits.npz',allow_pickle=False)
    assert np.array_equal(d['sample_ids'],s['sample_ids'])
    STATE.update(x=d['X_panel_log2tpm1'].astype(float),y=d['y_sensitivity'].astype(float),
        ids=d['sample_ids'],groups=d['group_ids'],lineage=d['lineage'],genes=d['panel_symbols'],
        labels=s['holdout_labels'],inner=s['inner_fold_by_holdout'],seeds=s['inner_split_seeds'])


def holdout(r,l,ph):
    token=f'repeat_{r+1:02d}__lineage_{l:02d}'
    meta_path=OUT/'folds'/(token+'.json');array_path=OUT/'folds'/(token+'.npz');model_path=OUT/'folds'/(token+'.joblib')
    if meta_path.exists():
        meta=json.loads(meta_path.read_text())
        assert meta['protocol_sha256']==ph and sha(array_path)==meta['array_sha256'] and sha(model_path)==meta['model_sha256']
        return meta
    begin=time.time();x,y=STATE['x'],STATE['y'];inner=STATE['inner'][r,l];label=STATE['labels'][l]
    tr=np.flatnonzero(inner>=0);te=np.flatnonzero(STATE['lineage']==label)
    assert label not in set(STATE['lineage'][tr]) and set(STATE['groups'][tr]).isdisjoint(STATE['groups'][te])
    val=np.full((2,len(ALPHAS),len(y)),np.nan);loss=np.empty((2,len(ALPHAS),5));diagnostics=[]
    for k in range(5):
        a=tr[inner[tr]!=k];b=tr[inner[tr]==k]
        assert set(a)|set(b)==set(tr) and set(STATE['groups'][a]).isdisjoint(STATE['groups'][b])
        prep=sn.prepare(x[a],y[a])
        for m,family in enumerate(FAMILIES):
            for c,alpha in enumerate(ALPHAS):
                model=sn.fit_prepared(prep,family,alpha);p=sn.predict(model,x[b])
                val[m,c,b]=p['mean'];loss[m,c,k]=np.sqrt(np.mean((y[b]-p['mean'])**2))
                diagnostics.append({'stage':'inner','family':family,'candidate':c,'inner_fold':k,**p['diagnostic']})
    assert np.isfinite(val[:,:,tr]).all() and np.isnan(val[:,:,te]).all()
    selected=np.argmin(loss.mean(axis=2),axis=1)
    prep=sn.prepare(x[tr],y[tr]);models=[];pred=[];variance=[];logpdf=[];logz=[]
    for m,family in enumerate(FAMILIES):
        model=sn.fit_prepared(prep,family,ALPHAS[selected[m]])
        model.update(train_rows=tr,test_rows=te,repeat=r+1,lineage=str(label),protocol_sha256=ph)
        p=sn.predict(model,x[te]);models.append(model);pred.append(p['mean']);variance.append(p['variance']);logz.append(p['logz_standardized'])
        logpdf.append(sn.conditional_logpdf(model,x[te],y[te],p))
        diagnostics.append({'stage':'outer','family':family,'candidate':int(selected[m]),**p['diagnostic']})
    joblib.dump(models,model_path,compress=3)
    np.savez_compressed(array_path,train_rows=tr,test_rows=te,inner_validation_predictions=val,inner_fold_rmse=loss,
        selected_candidates=selected,test_predictions=np.array(pred),test_variances=np.array(variance),
        test_logpdf=np.array(logpdf),test_logz=np.array(logz))
    meta={'repeat':r+1,'lineage_index':l,'lineage':str(label),'inner_seed':int(STATE['seeds'][r]),
        'train_n':len(tr),'test_n':len(te),'selected_alphas':[ALPHAS[i] for i in selected],
        'selected_candidates':selected,'mean_inner_rmse':[loss[m,selected[m]].mean() for m in range(2)],
        'numerical_diagnostics':diagnostics,'protocol_sha256':ph,'model_sha256':sha(model_path),'array_sha256':sha(array_path),
        'seconds':time.time()-begin,'graph_edges':models[0]['edges'],'heldout_lineage_and_identity_exclusion':True}
    dump(meta_path,meta);return meta


def main():
    start=time.time();initialize();(OUT/'folds').mkdir(parents=True,exist_ok=True)
    sources=[]
    for directory,names in [
        ('erastin_40gene_comparison',['comparison_inputs.npz']),
        ('erastin_nn_repeated',['shared_splits.npz','predictions.npz','protocol.json','per_repeat_metrics.tsv','per_repeat_macro_metrics.tsv']),
        ('erastin_bayesian_comparison',['gaussian_network.py','predictions.npz','protocol.json','repeated_summary.tsv']),
        ('erastin_repeated_cv',['repeat_cv.py']),
        ('erastin_nonlinear_comparison',['run_models.py'])]:
        root=WS/'outputs'/directory;known={v['path']:v['sha256'] for v in json.loads((root/'artifact_manifest.json').read_text())['files']}
        for name in names:
            h=sha(root/name);assert h==known[name]
            sources.append({'path':str((root/name).relative_to(WS)),'sha256':h})
    protocol={'status':'frozen_before_real_heldout_results','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'scripts':{name:sha(OUT/name) for name in ['spline_network.py','run_comparison.py']},
        'data':'unchanged 40-gene log2(TPM+1) panel, 745 cancer lines, 734 identity groups',
        'target':'negative reported erastin AUC, continuous, no clipping or discretization',
        'outer_holdouts':'same 24 corrected leave-lineage-out tests as saved ridge and NN',
        'repeats':20,'inner_seeds':[int(v) for v in STATE['seeds']],
        'tuning':'same five saved grouped inner folds; choose minimum equal-weight mean inner RMSE; first candidate wins ties',
        'families':FAMILIES,'alpha_grid':ALPHAS,
        'graph':'maximum spanning tree from Gaussian conditional mutual information between genes given response, learned independently in every current training subset; fixed gene root index0 SLC7A11; response parent of every gene',
        'graph_control':'both new families share the graph-learning rule and identical graph within a training subset; nonlinear structure search is not tested',
        'spline_cpd':'X_j | Y,X_parent ~ Normal(intercept_j + f_j(Y) + g_j(X_parent), sigma_j^2), with additive cubic B-spline means and constant conditional variance',
        'linear_cpd':'matched Gaussian CPDs with linear functions of the same parents, same alpha grid, variance estimator and prior',
        'basis':'four training-quantile knots at 0,1/3,2/3,1; collapse duplicate knots; cubic degree3, include_bias=False; linear extrapolation; each basis column training-standardized',
        'fitting':'each node uses ridge least squares (unpenalized intercept); one common alpha per network; residual maximum-likelihood variance with floor 1e-6 standardized units',
        'scaling':'joint response/gene StandardScaler, basis knots and basis scaling all fitted only on each current training subset',
        'prior':'training response Gaussian; graph/CPD parameters point estimates rather than posterior parameter averaging',
        'inference':'posterior mean E[Y|all 40 observed genes]; numerical integration over response spline intervals, analytic Gaussian tail moments under linear extrapolation',
        'quadrature':{'orders':sn.ORDERS,'mean_tolerance_standardized':sn.MEAN_TOL,'variance_tolerance_standardized':sn.VAR_TOL,
            'log_normalizer_tolerance':sn.LOGZ_TOL,'rule':'double order until all three tolerances pass; fail rather than silently accept unconverged predictions; no finite response truncation'},
        'references':'saved matched 20-repeat ridge and neural network; previous Gaussian TAN as additional reference; no refitting of these baselines',
        'primary_contrasts':'spline BN vs expression ridge and expression neural network; spline vs matched linear-CPD BN isolates conditional nonlinearity',
        'metrics':['spearman','rmse','r2'],'macro':'same 19 lineages with n>=10; undefined lineage Spearman not silently omitted',
        'uncertainty':{'bootstrap_count':5000,'pooled_seed':2026111001,'macro_seed':2026111002,'unit':'whole lineage, all arms and repeats paired; conditional on fixed predictions, no bootstrap refits'},
        'probability_score':'held-out conditional log likelihood using the normalized nonlinear posterior, not a Gaussian approximation to it',
        'audits':'linear limit matches prior Gaussian BN at zero regularization; synthetic nonlinearity; higher-order integration for every selected nonlinear model; independent CPD Ridge refits and scipy adaptive quadrature on prespecified representative holdouts',
        'previous_results_seen':True,'exploratory_same_cohort_followup':True,'training_only_pilot':'used inner-training rows only for timing and numerical convergence, no validation or outer performance examined',
        'workers':6,'environment':{'python':platform.python_version(),'numpy':np.__version__,'sklearn':sklearn.__version__},
        'method_sources':[
            {'url':'https://arxiv.org/abs/2610.30838','use':'related nonlinear Gaussian CPDs with spline means; this custom experiment does not reproduce its structure search or smoothing penalty'},
            {'url':'https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.SplineTransformer.html','use':'cubic B-spline basis and linear extrapolation; local installed version recorded separately'}]}
    path=OUT/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text());protocol['created_utc']=old['created_utc'];assert clean(protocol)==old
    else:dump(path,protocol)
    ph=sha(path);dump(OUT/'source_manifest.json',{'files':sources});dump(OUT/'mathematical_validation.json',sn.verify_math())
    np.savez_compressed(OUT/'shared_splits.npz',sample_ids=STATE['ids'],groups=STATE['groups'],lineage=STATE['lineage'],
        genes=STATE['genes'],labels=STATE['labels'],inner_seeds=STATE['seeds'],inner_fold_by_holdout=STATE['inner'])
    print(json.dumps({'event':'protocol_frozen','sha256':ph,'outer_network_fits':960}),flush=True)
    done=0
    with futures.ProcessPoolExecutor(max_workers=6,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        pending=[pool.submit(holdout,r,l,ph) for r in range(20) for l in range(24)]
        for f in futures.as_completed(pending):
            f.result();done+=1
            if done%12==0:
                progress={'phase':'fitting','completed_lineage_jobs':done,'total_lineage_jobs':480,'elapsed_seconds':time.time()-start}
                dump(OUT/'progress.json',progress);print(json.dumps(progress),flush=True)
    for source in sources:assert sha(WS/source['path'])==source['sha256']
    for name,h in protocol['scripts'].items():assert sha(OUT/name)==h
    assert sha(path)==ph
    dump(OUT/'fit_completion.json',{'status':'complete','outer_network_fits':960,'inner_network_fits':20*24*5*2*len(ALPHAS),
        'elapsed_seconds':time.time()-start,'source_and_script_hashes_unchanged':True})
    dump(OUT/'progress.json',{'phase':'fits_complete','completed_lineage_jobs':480,'total_lineage_jobs':480})
    print(json.dumps({'event':'fits_complete','elapsed_seconds':time.time()-start}),flush=True)


if __name__=='__main__':main()
