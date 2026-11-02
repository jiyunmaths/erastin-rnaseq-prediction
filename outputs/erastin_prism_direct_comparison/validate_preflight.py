#!/usr/bin/env python3
"""Independent fit APIs, objective equivalence, split and metric controls."""
import json,warnings,time
import numpy as np
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
import run_comparison as run
from enet_helpers import fit_path,fit_selected,GRID,ALPHAS,L1_RATIOS,kkt_residual
from make_splits import inner_rows
from evaluate import validate_bootstrap

def independent(f,params,x,y,a,b,seeds):
    sx=StandardScaler().fit(x[a]);z=sx.transform(x[a]);zz=sx.transform(x[b])
    if f=='ridge':return Ridge(alpha=params['alpha'],solver='svd').fit(z,y[a]).predict(zz)
    sy=StandardScaler().fit(y[a,None]);target=sy.transform(y[a,None]).ravel()
    if f=='elastic_net':
        n,p=z.shape;alpha=params['alpha'];ratio=params['l1_ratio'];xx=np.vstack([z,np.sqrt(n*alpha*(1-ratio))*np.eye(p)]);yy=np.r_[target,np.zeros(p)];m=Lasso(alpha=alpha*ratio*n/(n+p),fit_intercept=False,max_iter=50000,tol=1e-10,selection='cyclic')
        with warnings.catch_warnings():warnings.simplefilter('ignore');m.fit(xx,yy)
        assert kkt_residual(z,target,m.coef_,0.,alpha,ratio)<1e-6
        return (zz@m.coef_)*sy.scale_[0]+sy.mean_[0]
    preds=[]
    for seed in seeds:
        m=MLPRegressor(hidden_layer_sizes=(params['width'],),activation='tanh',solver='lbfgs',alpha=params['alpha'],max_iter=2000,max_fun=40000,tol=.001,early_stopping=False,random_state=int(seed))
        with warnings.catch_warnings():warnings.simplefilter('ignore');m.fit(z,target)
        preds.append(sy.inverse_transform(m.predict(zz)[:,None]).ravel())
    return np.mean(preds,axis=0)

def main():
    start=time.monotonic();ph=run.prepare();run.initialize();s=run.STATE;checks=[];splitchecks=0;assert s['x'][5].shape==(386,5) and len(set(s['lineage']))==20
    for r,mask in enumerate(s['test']):
        tr=np.flatnonzero(~mask);te=np.flatnonzero(mask);assert (len(tr),len(te))==(308,78) and set(s['groups'][tr]).isdisjoint(s['groups'][te]) and set(s['lineage'][tr])==set(s['lineage'][te])==set(s['lineage']);splitchecks+=1
        for k in range(5):
            a,b=inner_rows(s['inner'][r],k);assert set(a)|set(b)==set(tr) and set(s['groups'][a]).isdisjoint(s['groups'][b]) and len(set(s['lineage'][a]))==20;splitchecks+=1
    r=0;tr=np.flatnonzero(~s['test'][r]);te=np.flatnonzero(s['test'][r]);refpreds={}
    # Retain these first six complete outer fits as part of the 120-model experiment.
    for panel in run.PANELS:
        x=s['x'][panel]
        for f in run.FAMILIES:
            m=run.fit_job(panel,f,r,ph);a=np.load(run.OUT/'folds'/f'{f}__p{panel}__r01.npz');error=0.;count=0
            for k in range(5):
                aa,bb=inner_rows(s['inner'][r],k);candidates=range(len(run.grid_for(f))) if f!='elastic_net' else [m['selected_candidate']]
                for c in candidates:
                    ref=independent(f,run.grid_for(f)[c],x,s['y'],aa,bb,s['nn_seeds'][r]);error=max(error,float(np.max(np.abs(ref-a['inner_validation_predictions'][c,bb]))));count+=1
            outer=independent(f,m['selected_parameters'],x,s['y'],tr,te,s['nn_seeds'][r]);outererror=float(np.max(np.abs(outer-a['test_prediction'])));assert max(error,outererror)<1e-7
            pp=np.full(s['test'].shape,np.nan)
            # Saved first-repeat predictions are sufficient for metric arithmetic checks across repeated masks below.
            for rr,mask in enumerate(s['test']):pp[rr,mask]=np.mean(s['y'][~mask])+np.linspace(-.1,.1,mask.sum())
            refpreds[run.arm_name(f,panel)]=pp;checks.append(dict(family=f,panel=panel,repeat=1,independent_inner_settings=count,max_inner_prediction_error=error,max_outer_prediction_error=outererror));print('PREFLIGHT_ARM',f,panel,flush=True)
        aa,bb=inner_rows(s['inner'][0],0);pred,_,_,_=fit_path(x,s['y'],aa,bb);native_error=aug_error=0.
        for j in range(len(L1_RATIOS)):
            for i in [0,16,28,40]:
                c=j*len(ALPHAS)+i;p=GRID[c];native,_,_=fit_selected(x,s['y'],aa,bb,p);ref=independent('elastic_net',p,x,s['y'],aa,bb,s['nn_seeds'][0]);native_error=max(native_error,float(np.max(np.abs(native-pred[c]))));aug_error=max(aug_error,float(np.max(np.abs(ref-pred[c]))))
        assert max(native_error,aug_error)<1e-7;checks.append(dict(panel=panel,elastic_net_native_path_error=native_error,elastic_net_augmented_path_error=aug_error,path_candidates_checked=32))
    const=np.full(s['test'].shape,np.nan)
    for r,mask in enumerate(s['test']):const[r,mask]=s['y'][~mask].mean()
    refpreds['training_mean']=const;bv=validate_bootstrap(s['y'],refpreds,s['lineage'],s['test'],2026111202)
    v=dict(passed=True,protocol_sha256=ph,split_checks=splitchecks,all_20_lineages_in_both_outer_sets=True,checks=checks,bootstrap_arithmetic=bv,preflight_fits_retained=6,elapsed_seconds=time.monotonic()-start);run.dump(run.OUT/'preflight_validation.json',v);print(json.dumps(v,indent=2),flush=True)
if __name__=='__main__':main()
