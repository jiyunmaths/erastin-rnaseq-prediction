#!/usr/bin/env python3
"""Exact held-out same-lineage donor replacement for saved ridge and NN models."""
import concurrent.futures as futures
import hashlib,itertools,json,multiprocessing as mp,shutil,sys,time
from pathlib import Path
import joblib,numpy as np
from sklearn.preprocessing import StandardScaler
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;WS=OUT.parent.parent;SOURCE=WS/'outputs/erastin_stratified_panel_comparison';STATE={};FAMILIES=['ridge','neural_network'];WORKERS=6

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
    if isinstance(v,np.generic):return v.item()
    return v

def dump(path,value):
    temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n');temp.replace(path)

def initialize():
    a=np.load(OUT/'analysis_inputs.npz');STATE.update({k:a[k] for k in a.files})

def native(models,x):
    ridge,nn=models;z=(x-ridge['mean'])/ridge['scale'];pr=z@ridge['coef']+ridge['intercept_after_scaling'];zn=nn['x_scaler'].transform(x)
    pn=np.mean([nn['y_scaler'].inverse_transform(v.predict(zn)[:,None]).ravel() for v in nn['members']],axis=0)
    return np.stack([pr,pn])

def one(r,ph):
    root=OUT/'repeats'/f'r{r+1:02d}'
    if root.with_suffix('.json').exists():
        m=json.loads(root.with_suffix('.json').read_text());assert m['protocol_sha256']==ph and sha(root.with_suffix('.npz'))==m['array_sha256'];return m
    begin=time.monotonic();s=STATE;x=s['x'];y=s['y'];lineage=s['lineage'];groups=s['groups'];mask=s['outer_test'][r];tr=np.flatnonzero(~mask);te=np.flatnonzero(mask)
    assert set(groups[tr]).isdisjoint(groups[te]) and len(set(lineage[tr]))==len(set(lineage[te]))==23
    models=[]
    for f in FAMILIES:
        src=SOURCE/'folds'/f'{f}__p64__r{r+1:02d}';meta=json.loads(src.with_suffix('.json').read_text());a=np.load(src.with_suffix('.npz'));np.testing.assert_array_equal(a['train_rows'],tr);np.testing.assert_array_equal(a['test_rows'],te)
        models.append(joblib.load(src.with_suffix('.joblib')))
    ridge,nn=models;sx=StandardScaler().fit(x[tr]);sy=StandardScaler().fit(y[tr,None])
    for actual,expected in [(ridge['mean'],sx.mean_),(ridge['scale'],sx.scale_),(nn['x_scaler'].mean_,sx.mean_),(nn['x_scaler'].scale_,sx.scale_),(nn['y_scaler'].mean_,sy.mean_),(nn['y_scaler'].scale_,sy.scale_)]:np.testing.assert_allclose(actual,expected,rtol=0,atol=1e-12)
    pred=native(models,x[te]);base=(pred-y[te][None,:])**2;error=float(np.max(np.abs(pred-s['saved_predictions'][:,r,te])));assert error<1e-11
    z=sx.transform(x);p=x.shape[1];n=len(te);delta_train=np.empty((2,n,p));shift_train=np.empty_like(delta_train);delta_test=np.empty_like(delta_train);shift_test=np.empty_like(delta_train);grad=np.empty_like(delta_train)
    grad[0]=ridge['coef'];latent=[];grad[1]=0
    for member in nn['members']:
        assert member.activation=='tanh' and len(member.coefs_)==2 and member.out_activation_=='identity'
        w1=member.coefs_[0];w2=member.coefs_[1].ravel();h=z[te]@w1+member.intercepts_[0];latent.append((h,w1,w2,float(member.intercepts_[1][0])))
        grad[1]+=(((1-np.tanh(h)**2)*w2)@w1.T)*sy.scale_[0]/len(nn['members'])
    native_error=identity_error=gradient_error=exhaustive_error=full_matrix_error=0.;lineage_checks=[]
    for label in s['labels']:
        receiver=np.flatnonzero(lineage[te]==label);testrows=te[receiver];trainrows=tr[lineage[tr]==label];nt=len(trainrows);nv=len(testrows);donors=np.r_[trainrows,testrows]
        assert nt>0 and nv>0 and set(groups[trainrows]).isdisjoint(groups[testrows])
        first=x[testrows[0]];ref=[]
        for donor in [trainrows[-1],testrows[-1]]:
            xx=np.repeat(first[None,:],p,axis=0);xx[np.arange(p),np.arange(p)]=x[donor];ref.append(native(models,xx))
        eps=1e-5;plus=np.repeat(first[None,:],p,axis=0);minus=plus.copy();plus[np.arange(p),np.arange(p)]+=eps*sx.scale_;minus[np.arange(p),np.arange(p)]-=eps*sx.scale_
        finite=(native(models,plus)-native(models,minus))/(2*eps);gradient_error=max(gradient_error,float(np.max(np.abs(finite-grad[:,receiver[0],:]))));assert gradient_error<1e-8
        for j in range(p):
            diff=z[donors,j][None,:]-z[testrows,j][:,None];pp=np.empty((2,nv,len(donors)));pp[0]=pred[0,receiver,None]+diff*ridge['coef'][j];pp[1]=0
            for h,w1,w2,bias in latent:
                v=(np.tanh(h[receiver,None,:]+diff[:,:,None]*w1[j])@w2+bias)*sy.scale_[0]+sy.mean_[0];pp[1]+=v/len(latent)
            native_error=max(native_error,float(np.max(np.abs(pp[:,0,nt-1]-ref[0][:,j]))),float(np.max(np.abs(pp[:,0,-1]-ref[1][:,j]))))
            identity_error=max(identity_error,float(np.max(np.abs(pp[:,np.arange(nv),nt+np.arange(nv)]-pred[:,receiver]))))
            loss=(pp-y[testrows][None,:,None])**2
            delta_train[:,receiver,j]=loss[:,:,:nt].mean(axis=2)-base[:,receiver];delta_test[:,receiver,j]=loss[:,:,nt:].mean(axis=2)-base[:,receiver]
            sh=(pp-pred[:,receiver,None])**2;shift_train[:,receiver,j]=sh[:,:,:nt].mean(axis=2);shift_test[:,receiver,j]=sh[:,:,nt:].mean(axis=2)
            if r==0 and nv in [2,3]:
                losses=[]
                for permutation in itertools.permutations(range(nv)):
                    xx=x[testrows].copy();xx[:,j]=x[testrows[np.array(permutation)],j];dd=native(models,xx);losses.append(np.mean((dd-y[testrows][None,:])**2,axis=1))
                expected=np.mean(losses,axis=0);exhaustive_error=max(exhaustive_error,float(np.max(np.abs(expected-loss[:,:,nt:].mean(axis=(1,2))))))
            if r in [0,19] and label in ['Lung','Thyroid','Ampulla of Vater'] and j in [0,21,40,63]:
                xx=np.repeat(x[testrows],nt,axis=0);xx[:,j]=np.tile(x[trainrows,j],nv);direct=native(models,xx).reshape(2,nv,nt);full_matrix_error=max(full_matrix_error,float(np.max(np.abs(direct-pp[:,:,:nt]))))
        if nv==1:assert np.max(np.abs(delta_test[:,receiver,:]))<1e-12 and np.max(shift_test[:,receiver,:])<1e-20
        lineage_checks.append(dict(lineage=str(label),training_donors=nt,test_receivers=nv,training_donor_groups=len(np.unique(groups[trainrows])),test_groups=len(np.unique(groups[testrows]))))
    assert max(native_error,identity_error,exhaustive_error,full_matrix_error)<1e-11
    np.savez_compressed(root.with_suffix('.npz'),test_rows=te,baseline_prediction=pred,baseline_squared_error=base,
        train_reference_delta_mse=delta_train,train_reference_mean_squared_prediction_shift=shift_train,test_reference_delta_mse=delta_test,
        test_reference_mean_squared_prediction_shift=shift_test,standardized_gradient=grad,ridge_coefficients=ridge['coef'])
    m=dict(repeat=r+1,protocol_sha256=ph,saved_prediction_error=error,native_replacement_error=native_error,identity_replacement_error=identity_error,
        finite_difference_gradient_error=gradient_error,exhaustive_small_test_permutation_error=exhaustive_error,full_donor_matrix_error=full_matrix_error,
        lineage_checks=lineage_checks,array_sha256=sha(root.with_suffix('.npz')),seconds=time.monotonic()-begin)
    dump(root.with_suffix('.json'),m);return m

def prepare():
    OUT.mkdir(exist_ok=True);(OUT/'repeats').mkdir(exist_ok=True);known={v['path']:v['sha256'] for v in json.loads((SOURCE/'artifact_manifest.json').read_text())['files']};names=['panel_inputs.npz','shared_splits.npz','predictions.npz','protocol.json','validation.json','weighted_mlp.py']
    names += [f'folds/{f}__p64__r{r+1:02d}.{ext}' for f in FAMILIES for r in range(20) for ext in ['json','npz','joblib']]
    sources=[]
    for name in names:
        assert sha(SOURCE/name)==known[name];sources.append(dict(path=str((SOURCE/name).relative_to(WS)),sha256=known[name]))
    if not (OUT/'weighted_mlp.py').exists():shutil.copyfile(SOURCE/'weighted_mlp.py',OUT/'weighted_mlp.py')
    assert sha(OUT/'weighted_mlp.py')==known['weighted_mlp.py']
    a=np.load(SOURCE/'panel_inputs.npz');sp=np.load(SOURCE/'shared_splits.npz');pred=np.load(SOURCE/'predictions.npz');genes=a['genes_64']
    for d in [sp,pred]:
        for k in ['sample_ids','group_ids','lineage']:np.testing.assert_array_equal(d[k],a[k])
    assert len(genes)==64 and len(a['y'])==744 and 'Cervix' not in a['lineage']
    p=dict(status='frozen_before_stratified_gene_importance',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        models='All 20 saved unweighted 64-gene ridge models and all 20 saved three-member NN ensembles; no model refitting.',genes=genes,
        split_design='Exact existing 20 lineage-stratified grouped 80/20 splits: 595 training, 149 testing; all 23 lineages present in both, Cervix excluded.',
        primary='Expected held-out MSE increase from replacing one gene with a uniformly chosen same-lineage TRAINING-reference donor value. Enumerate every held-out receiver/training donor pair. No test response enters donor choice.',
        secondary='Exact expected within-held-out-lineage test permutation MSE increase, enumerating every receiver/test-donor pair including self. Singleton test sets have no permutation information and are excluded from secondary ranking scopes.',
        influence='RMS prediction change under the same training-donor replacements; separate from predictive benefit.',
        derivatives='Signed ridge coefficient or NN ensemble local derivative per one TRAINING SD of gene expression, in original negative-AUC target units. NN members are averaged before error/gradient/influence aggregation.',
        primary_scope='Cell-weighted mean over all 149 held-out receivers per split, then mean over 20 splits. Equal-lineage macro23 and weighted/macro19 sensitivity analyses are also reported.',
        lineage_consistency='Primary stability counts/rank correlations among 19 lineages with cohort n>=10; stricter 12-lineage sensitivity with >=5 test lines per split; all 23 lineages tabulated, rare ones descriptive.',
        uncertainty=dict(count=5000,seed=2026110801,unit='Paired whole-lineage bootstrap shared across models/genes, on mean per-split lineage effects; fixed models and donor distributions, no refitting or multiplicity correction. Repeat/test observations overlap.'),
        metrics='Rank all 64 genes by primary MSE increase. Also rank prediction influence and standardized coefficients/gradients. Report positive-lineage counts, per-seed rank/top10/sign stability and pairwise lineage/seed rank correlations.',
        limitations=['Training-reference replacement estimates model reliance under the observed within-lineage training expression distribution; this differs from ordinary within-test permutation importance.',
            'Replacement disrupts gene-gene correlations and may produce atypical expression combinations; redundant or correlated genes can divide importance.',
            'The ridge model uses one coefficient vector for all lineages in each split. Lineage-specific performance importance is not a separately fitted lineage coefficient.',
            'Tiny lineages/test sets yield noisy estimates even when training donors allow computation. No causal gene effects or external biological validation are established.'],
        workers=WORKERS,script_sha256=sha(Path(__file__)),weighted_mlp_sha256=sha(OUT/'weighted_mlp.py'))
    path=OUT/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text());p['created_utc']=old['created_utc'];assert old==clean(p)
    else:dump(path,p)
    dump(OUT/'source_manifest.json',dict(files=sources))
    np.savez_compressed(OUT/'analysis_inputs.npz',x=a['x_64'],y=a['y'],genes=genes,sample_ids=a['sample_ids'],groups=a['group_ids'],lineage=a['lineage'],labels=np.unique(a['lineage']),outer_test=sp['outer_test'],
        saved_predictions=np.stack([pred['ridge__64'],pred['neural_network__64']]),neural_seeds=sp['neural_initialization_seeds'])
    return sha(path)

def main():
    start=time.monotonic();ph=prepare();initialize();checks=[]
    with futures.ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        jobs=[pool.submit(one,r,ph) for r in range(20)]
        for future in futures.as_completed(jobs):
            checks.append(future.result());elapsed=time.monotonic()-start;v=dict(phase='calculating',completed=len(checks),total=20,elapsed_seconds=elapsed);dump(OUT/'progress.json',v);print(json.dumps(v),flush=True)
    for a in json.loads((OUT/'source_manifest.json').read_text())['files']:assert sha(WS/a['path'])==a['sha256']
    p=json.loads((OUT/'protocol.json').read_text());assert sha(Path(__file__))==p['script_sha256'] and sha(OUT/'weighted_mlp.py')==p['weighted_mlp_sha256']
    dump(OUT/'calculation_validation.json',dict(passed=True,no_refitting=True,saved_models=40,checks=checks,elapsed_seconds=time.monotonic()-start));dump(OUT/'progress.json',dict(phase='calculation_complete',completed=20,total=20));print('CALCULATION_COMPLETE',flush=True)
if __name__=='__main__':main()
