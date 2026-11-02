#!/usr/bin/env python3
"""Exact same-lineage held-out gene replacement in saved PRISM/CTRP models."""
import concurrent.futures as futures
import hashlib,json,multiprocessing as mp,shutil,sys,time
from pathlib import Path
import joblib,numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;WS=OUT.parent.parent
SOURCES={'CTRP':WS/'outputs/erastin_elastic_net_comparison','PRISM':WS/'outputs/erastin_prism_direct_comparison'}
MODELS=['ridge','elastic_net','neural_network'];WORKERS=8;STATE={}

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def dump(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n');temp.replace(path)
def table(name,rows):
    d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows);d.to_csv(OUT/name,sep='\t',index=False,float_format='%.12g');return d

def prepare():
    OUT.mkdir(exist_ok=True);(OUT/'repeats').mkdir(exist_ok=True);records=[];inputs={}
    for ds,root in SOURCES.items():
        known={v['path']:v['sha256'] for v in json.load(open(root/'artifact_manifest.json'))['files']};names=['panel_inputs.npz','shared_splits.npz','predictions.npz','protocol.json','validation.json','weighted_mlp.py','elastic_net_gene_selection.tsv']+[f'folds/{f}__p64__r{r+1:02d}.{ext}' for f in MODELS for r in range(20) for ext in ['json','npz','joblib']]
        for name in names:assert sha(root/name)==known[name];records.append(dict(path=str((root/name).relative_to(WS)),sha256=known[name]))
        assert sha(OUT/'weighted_mlp.py')==known['weighted_mlp.py'];a=np.load(root/'panel_inputs.npz');sp=np.load(root/'shared_splits.npz');pred=np.load(root/'predictions.npz')
        arrays=dict(x=a['x_64'],y=a['y'],genes=a['genes_64'],ids=a['sample_ids'],groups=a['group_ids'],lineage=a['lineage'],labels=np.unique(a['lineage']),test=sp['outer_test'],saved_predictions=np.stack([pred[f+'__64'] for f in MODELS]))
        for d in [sp,pred]:
            for key in ['sample_ids','group_ids','lineage']:np.testing.assert_array_equal(d[key],a[key])
        np.savez_compressed(OUT/(ds.lower()+'_inputs.npz'),**arrays);inputs[ds]=arrays
    old=WS/'outputs/erastin_elastic_net_gene_agreement';known={v['path']:v['sha256'] for v in json.load(open(old/'artifact_manifest.json'))['files']}
    for name in ['importance_arrays.npz','gene_importance_summary.tsv','gene_agreement.tsv','summary.json','validation.json']:
        assert sha(old/name)==known[name];records.append(dict(path=str((old/name).relative_to(WS)),sha256=known[name]))
    nativeold=WS/'outputs/erastin_stratified_gene_importance';known={v['path']:v['sha256'] for v in json.load(open(nativeold/'artifact_manifest.json'))['files']};name='bootstrap_draws.npz';assert sha(nativeold/name)==known[name];records.append(dict(path=str((nativeold/name).relative_to(WS)),sha256=known[name]))
    c=inputs['CTRP'];p=inputs['PRISM'];np.testing.assert_array_equal(c['genes'],p['genes']);common=sorted(set(c['labels'])&set(p['labels']));large=[l for l in common if np.sum(c['lineage']==l)>=10 and np.sum(p['lineage']==l)>=10];strict=[l for l in large if np.min(np.sum(c['test']&(c['lineage']==l),axis=1))>=5 and np.min(np.sum(p['test']&(p['lineage']==l),axis=1))>=5]
    matched=sorted(set(c['ids'][c['test'].any(axis=0)])&set(p['ids'][p['test'].any(axis=0)]));matched=[mid for mid in matched if p['lineage'][np.flatnonzero(p['ids']==mid)[0]] in large];assert (len(common),len(large),len(strict),len(matched))==(20,13,7,290)
    for mid in matched:
        ci=np.flatnonzero(c['ids']==mid)[0];pi=np.flatnonzero(p['ids']==mid)[0];np.testing.assert_array_equal(c['x'][ci],p['x'][pi]);assert c['groups'][ci]==p['groups'][pi] and c['lineage'][ci]==p['lineage'][pi]
    protocol=dict(status='fixed_before_PRISM_gene_importance_calculation',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),models=MODELS,genes=c['genes'],datasets={ds:dict(samples=len(a['y']),lineages=len(a['labels']),models=60,repeats=20) for ds,a in inputs.items()},no_refitting=True,
        primary='Same exact expected held-out MSE increase used previously: replace one gene with every same-lineage outer-TRAINING donor value, keeping other genes fixed; average donors then held-out receivers, then the 20 splits. Positive means worse error after replacement. No test response enters donor choice.',
        secondary='RMS prediction change and signed/RMS local derivative per training expression SD are separate output-influence measures. Exact ordinary within-test-lineage permutation is a donor-reference sensitivity analysis; singleton test lineages contain no such permutation information.',
        native_scope='Cell-weighted full native test cohorts, separately for PRISM (78 test lines, 20 lineages) and CTRP (149 test lines, 23 lineages). Also show equal-native-lineage weighting.',
        composition_control='Equal-lineage comparison on the same 13 lineages with cohort n>=10 in both datasets. Average each lineage effect over repeats, then give the 13 lineages equal weight. Compare ranks and relative MSE increase, not raw AUC coefficients across assays.',
        common_lineages=common,common_large_lineages=large,common_strict_lineages=strict,
        matched_cell_control='Restrict to the exact same 290 cell-line IDs from common 13 lineages that were held out at least once in BOTH experiments. Average each cell effect over its own eligible test appearances in each dataset; average cells within each lineage then weight the 13 lineages equally. Donor/training cohorts still differ, so this is a sensitivity analysis, not isolated assay causality.',
        matched_cell_ids=matched,
        lineage_stability='All lineages tabulated; primary PRISM consistency among 13 lineages with cohort n>=10, stricter 7 with >=5 test lines in each dataset. Report positive lineage counts, positive/top10 split counts and rank correlations. Broad positive support is descriptive >=80% of common 13 lineages (>=11/13); do not equate coefficient sign stability with predictive benefit.',
        elastic_net_selection='Nonzero coefficients per outer fit. Always selected=20/20, frequent>=18/20; compare both datasets with signed stability.',
        uncertainty=dict(count=5000,seed=2026111301,native_ctrp='Reuse original lineage_picks23 from the prior CTRP analysis.',native_prism='Whole-lineage bootstrap among 20 PRISM lineages, paired across models and genes.',common13='Same whole-lineage picks among common 13 lineages in both datasets, paired across models and genes.',scope='Conditional pointwise intervals on fixed models, panels and donor distributions. No model refitting or multiple-testing/model-choice correction; overlapping repeats are not independent replications.'),
        limitations=['Replacement disrupts gene-gene correlations; correlated genes can share reliance. This is within-lineage marginal replacement, not fully conditional permutation or a causal perturbation.',
            'Training cohorts, sample sizes, dose ranges, response processing and fitted regularization differ across PRISM and CTRP. Native rank changes cannot be attributed solely to assay biology.',
            'Same numeric split seeds do not imply same test samples across different cohorts. Cross-assay comparisons use paired lineage summaries, not paired repeat-level observations.',
            'CTRP response provenance/normalization remains limited as documented previously. PRISM is the pinned official 20Q2 secondary-screen AUC, with processed values >1 preserved.'],
        workers=WORKERS,script_sha256=sha(Path(__file__)),weighted_mlp_sha256=sha(OUT/'weighted_mlp.py'))
    pp=OUT/'protocol.json'
    if pp.exists():previous=json.load(open(pp));protocol['created_utc']=previous['created_utc'];assert previous==clean(protocol)
    else:dump(pp,protocol)
    dump(OUT/'source_manifest.json',dict(files=records));return sha(pp)

def initialize():
    for ds in SOURCES:
        a=np.load(OUT/(ds.lower()+'_inputs.npz'));STATE[ds]={k:a[k] for k in a.files}

def native(models,x):
    ridge,en,nn=models;pr=(x-ridge['mean'])/ridge['scale']@ridge['coef']+ridge['intercept_after_scaling'];pe=en['y_scaler'].inverse_transform(en['model'].predict(en['x_scaler'].transform(x))[:,None]).ravel();z=nn['x_scaler'].transform(x);pn=np.mean([nn['y_scaler'].inverse_transform(v.predict(z)[:,None]).ravel() for v in nn['members']],axis=0);return np.stack([pr,pe,pn])

def one(ds,r,ph):
    root=OUT/'repeats'/f'{ds.lower()}__r{r+1:02d}'
    if root.with_suffix('.json').exists():
        m=json.load(open(root.with_suffix('.json')));assert m['protocol_sha256']==ph and sha(root.with_suffix('.npz'))==m['array_sha256'];return m
    begin=time.monotonic();s=STATE[ds];x=s['x'];y=s['y'];lin=s['lineage'];groups=s['groups'];tr=np.flatnonzero(~s['test'][r]);te=np.flatnonzero(s['test'][r]);assert set(groups[tr]).isdisjoint(groups[te]) and set(lin[tr])==set(lin[te])==set(s['labels']);models=[]
    for f in MODELS:
        src=SOURCES[ds]/'folds'/f'{f}__p64__r{r+1:02d}';meta=json.load(open(src.with_suffix('.json')));a=np.load(src.with_suffix('.npz'));np.testing.assert_array_equal(a['train_rows'],tr);np.testing.assert_array_equal(a['test_rows'],te);models.append(joblib.load(src.with_suffix('.joblib')))
    ridge,en,nn=models;sx=StandardScaler().fit(x[tr]);sy=StandardScaler().fit(y[tr,None]);z=sx.transform(x)
    for actual,expected in [(ridge['mean'],sx.mean_),(ridge['scale'],sx.scale_)]+[(getattr(v['x_scaler'],key),getattr(sx,key)) for v in [en,nn] for key in ['mean_','scale_']]+[(getattr(v['y_scaler'],key),getattr(sy,key)) for v in [en,nn] for key in ['mean_','scale_']]:np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
    pred=native(models,x[te]);saved_error=float(np.max(np.abs(pred-s['saved_predictions'][:,r,te])));assert saved_error<1e-11;base=(pred-y[te][None,:])**2;beta=np.stack([ridge['coef'],en['model'].coef_*sy.scale_[0]]);grad=np.empty((3,len(te),64));grad[:2]=beta[:,None,:];grad[2]=0.;latent=[]
    for member in nn['members']:
        assert member.activation=='tanh' and member.out_activation_=='identity';w1=member.coefs_[0];w2=member.coefs_[1].ravel();h=z[te]@w1+member.intercepts_[0];latent.append((h,w1,w2,float(member.intercepts_[1][0])));grad[2]+=(((1-np.tanh(h)**2)*w2)@w1.T)*sy.scale_[0]/len(nn['members'])
    D=np.empty_like(grad);S=np.empty_like(grad);T=np.empty_like(grad);TS=np.empty_like(grad);maxnative=maxidentity=maxgrad=maxfull=0.;checks=[];native_full=0
    for label in s['labels']:
        recv=np.flatnonzero(lin[te]==label);testrows=te[recv];trainrows=tr[lin[tr]==label];nt=len(trainrows);nv=len(testrows);donors=np.r_[trainrows,testrows];assert nt>0 and nv>0 and set(groups[trainrows]).isdisjoint(groups[testrows]);first=x[testrows[0]];ref=[]
        for donor in [trainrows[-1],testrows[-1]]:
            xx=np.repeat(first[None,:],64,axis=0);xx[np.arange(64),np.arange(64)]=x[donor];ref.append(native(models,xx))
        eps=1e-5;plus=np.repeat(first[None,:],64,axis=0);minus=plus.copy();plus[np.arange(64),np.arange(64)]+=eps*sx.scale_;minus[np.arange(64),np.arange(64)]-=eps*sx.scale_;fd=(native(models,plus)-native(models,minus))/(2*eps);maxgrad=max(maxgrad,float(np.max(np.abs(fd-grad[:,recv[0]]))));assert maxgrad<1e-8
        for j in range(64):
            diff=z[donors,j][None,:]-z[testrows,j][:,None];pp=np.empty((3,nv,len(donors)));pp[:2]=pred[:2,recv,None]+beta[:,j,None,None]*diff[None,:,:];pp[2]=0.
            for h,w1,w2,bias in latent:pp[2]+=((np.tanh(h[recv,None,:]+diff[:,:,None]*w1[j])@w2+bias)*sy.scale_[0]+sy.mean_[0])/len(latent)
            maxnative=max(maxnative,float(np.max(np.abs(pp[:,0,nt-1]-ref[0][:,j]))),float(np.max(np.abs(pp[:,0,-1]-ref[1][:,j]))));maxidentity=max(maxidentity,float(np.max(np.abs(pp[:,np.arange(nv),nt+np.arange(nv)]-pred[:,recv]))))
            loss=(pp-y[testrows][None,:,None])**2;shift=(pp-pred[:,recv,None])**2;D[:,recv,j]=loss[:,:,:nt].mean(axis=2)-base[:,recv];S[:,recv,j]=shift[:,:,:nt].mean(axis=2);T[:,recv,j]=loss[:,:,nt:].mean(axis=2)-base[:,recv];TS[:,recv,j]=shift[:,:,nt:].mean(axis=2)
            if r in [0,19] and label in ['Lung','Thyroid','Prostate','Ampulla of Vater'] and j in [0,9,17,63]:
                xx=np.repeat(x[testrows],nt,axis=0);xx[:,j]=np.tile(x[trainrows,j],nv);direct=native(models,xx).reshape(3,nv,nt);maxfull=max(maxfull,float(np.max(np.abs(direct-pp[:,:,:nt]))));native_full+=1
        if nv==1:assert np.max(np.abs(T[:,recv]))<1e-12 and np.max(TS[:,recv])<1e-20
        checks.append(dict(lineage=str(label),training_donors=nt,test_receivers=nv,training_groups=len(set(groups[trainrows])),test_groups=len(set(groups[testrows]))))
    assert max(maxnative,maxidentity,maxfull)<1e-11
    np.savez_compressed(root.with_suffix('.npz'),test_rows=te,baseline_prediction=pred,baseline_squared_error=base,train_reference_delta_mse=D,train_reference_mean_squared_prediction_shift=S,test_reference_delta_mse=T,test_reference_mean_squared_prediction_shift=TS,standardized_gradient=grad,ridge_coefficients=beta[0],elastic_net_coefficients=beta[1])
    m=dict(dataset=ds,repeat=r+1,protocol_sha256=ph,saved_prediction_error=saved_error,native_replacement_error=maxnative,identity_replacement_error=maxidentity,finite_difference_gradient_error=maxgrad,full_donor_matrix_error=maxfull,native_full_matrix_checks=native_full,lineage_checks=checks,array_sha256=sha(root.with_suffix('.npz')),seconds=time.monotonic()-begin);dump(root.with_suffix('.json'),m);return m

def main():
    begin=time.monotonic();ph=prepare();initialize();checks=[]
    with futures.ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        jobs=[pool.submit(one,ds,r,ph) for r in range(20) for ds in SOURCES]
        for future in futures.as_completed(jobs):
            checks.append(future.result());v=dict(phase='calculating',completed=len(checks),total=40,elapsed_seconds=time.monotonic()-begin);dump(OUT/'progress.json',v)
            if len(checks)%8==0:print(json.dumps(v),flush=True)
    for a in json.load(open(OUT/'source_manifest.json'))['files']:assert sha(WS/a['path'])==a['sha256']
    p=json.load(open(OUT/'protocol.json'));assert sha(Path(__file__))==p['script_sha256'] and sha(OUT/'weighted_mlp.py')==p['weighted_mlp_sha256'];dump(OUT/'calculation_validation.json',dict(passed=True,no_refitting=True,saved_models=120,repeat_calculations=40,checks=checks,elapsed_seconds=time.monotonic()-begin));dump(OUT/'progress.json',dict(phase='calculation_complete',completed=40,total=40));print('CALCULATION_COMPLETE',flush=True)
if __name__=='__main__':main()
