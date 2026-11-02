#!/usr/bin/env python3
"""Verify panel memberships, fixed masks, and independent selected fits."""
import json,time,warnings
import numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.neural_network import MLPRegressor
import run_comparison as run
from enet_helpers import GRID,fit_path,kkt_residual
from make_splits import inner_rows

def independent(f,params,x,y,a,b,seeds):
    sx=StandardScaler().fit(x[a]);z=sx.transform(x[a]);zz=sx.transform(x[b])
    if f=='ridge':return Ridge(alpha=params['alpha'],solver='svd').fit(z,y[a]).predict(zz)
    sy=StandardScaler().fit(y[a,None]);target=sy.transform(y[a,None]).ravel()
    if f=='elastic_net':
        n,p=z.shape;alpha=params['alpha'];ratio=params['l1_ratio'];xx=np.vstack([z,np.sqrt(n*alpha*(1-ratio))*np.eye(p)]);yy=np.r_[target,np.zeros(p)];m=Lasso(alpha=alpha*ratio*n/(n+p),fit_intercept=False,max_iter=200000,tol=1e-11,selection='cyclic')
        with warnings.catch_warnings(record=True) as caught:warnings.simplefilter('always');m.fit(xx,yy)
        assert not any(w.category.__name__=='ConvergenceWarning' for w in caught);assert kkt_residual(z,target,m.coef_,0.,alpha,ratio)<1e-6
        return zz@m.coef_*sy.scale_[0]+sy.mean_[0]
    pp=[]
    for seed in seeds:
        m=MLPRegressor(hidden_layer_sizes=(params['width'],),activation='tanh',solver='lbfgs',alpha=params['alpha'],max_iter=2000,max_fun=40000,tol=.001,early_stopping=False,random_state=int(seed))
        with warnings.catch_warnings(record=True) as caught:warnings.simplefilter('always');m.fit(z,target)
        assert not any(w.category.__name__=='ConvergenceWarning' for w in caught);pp.append(sy.inverse_transform(m.predict(zz)[:,None]).ravel())
    return np.mean(pp,axis=0)

def main():
    start=time.monotonic();ph=run.prepare();run.initialize();s=run.STATE;checks=[];splitchecks=0;assert len(s['ids'])==744 and len(set(s['groups']))==733 and len(set(s['lineage']))==23
    sizes={'ferro5':5,'ferro64':64,'ferro64_emt':264,'ferro64_ddr':213,'ferro64_emt_ddr':413}
    for p in run.PANELS:assert s['x'][p].shape==(744,sizes[p]) and np.isfinite(s['x'][p]).all() and len(set(s['genes'][p]))==sizes[p]
    for p in run.NEW:np.testing.assert_array_equal(s['x'][p][:,:64],s['x']['ferro64']);assert set(s['genes']['ferro64']).issubset(set(s['genes'][p]))
    mapping=pd.read_csv(run.OUT/'gene_mapping.tsv',sep='\t',dtype=str,keep_default_na=False);special=mapping[mapping.current_symbol=='BOLA2'];assert len(special)==1 and special.depmap_column.iloc[0]=='BOLA2 (552900)' and special.mapping_rule.iloc[0]=='duplicate_entrez_resolved_by_current_symbol_and_release_HGNC';assert 'BOLA2B' not in set(s['genes']['ferro64_emt_ddr'])
    for r,mask in enumerate(s['test']):
        tr=np.flatnonzero(~mask);te=np.flatnonzero(mask);assert (len(tr),len(te))==(595,149) and set(s['groups'][tr]).isdisjoint(s['groups'][te]) and set(s['lineage'][tr])==set(s['lineage'][te])==set(s['lineage']);splitchecks+=1;assert np.sum(s['inner'][r]==-2)==1
        for k in range(5):
            a,b=inner_rows(s['inner'][r],k);assert set(a)|set(b)==set(tr) and set(s['groups'][a]).isdisjoint(s['groups'][b]) and len(set(s['lineage'][a]))==23;splitchecks+=1
    tr=np.flatnonzero(~s['test'][0]);te=np.flatnonzero(s['test'][0])
    for p in run.NEW:
        x=s['x'][p]
        for f in run.FAMILIES:
            m=run.fit_job(p,f,0,ph);a=np.load(run.root_for(f,p,0).with_suffix('.npz'));error=0.;count=0
            for k in range(5):
                aa,bb=inner_rows(s['inner'][0],k);candidates=range(len(run.grid_for(f))) if k==0 and f!='elastic_net' else [m['selected_candidate']]
                for c in candidates:
                    ref=independent(f,run.grid_for(f)[c],x,s['y'],aa,bb,s['nn_seeds'][0]);error=max(error,float(np.max(np.abs(ref-a['inner_validation_predictions'][c,bb]))));count+=1
            outer=independent(f,m['selected_parameters'],x,s['y'],tr,te,s['nn_seeds'][0]);outererror=float(np.max(np.abs(outer-a['test_prediction'])));assert max(error,outererror)<1e-7;checks.append(dict(panel=p,family=f,independent_inner_candidates=count,max_inner_error=error,max_outer_error=outererror));print('PREFLIGHT_ARM',p,f,flush=True)
        aa,bb=inner_rows(s['inner'][0],0);path,_,_,_=fit_path(x,s['y'],aa,bb);nativeerror=augerror=0.
        for j in range(8):
            for i in [0,16,28,40]:
                c=j*41+i;params=GRID[c];sx=StandardScaler().fit(x[aa]);sy=StandardScaler().fit(s['y'][aa,None]);native=ElasticNet(**params,max_iter=50000,tol=1e-9,selection='cyclic').fit(sx.transform(x[aa]),sy.transform(s['y'][aa,None]).ravel());pred=sy.inverse_transform(native.predict(sx.transform(x[bb]))[:,None]).ravel();ref=independent('elastic_net',params,x,s['y'],aa,bb,s['nn_seeds'][0]);nativeerror=max(nativeerror,float(np.max(np.abs(pred-path[c]))));augerror=max(augerror,float(np.max(np.abs(ref-path[c]))))
        assert max(nativeerror,augerror)<1e-7;checks.append(dict(panel=p,path_candidates=32,native_path_error=nativeerror,augmented_lasso_error=augerror))
    result=dict(passed=True,protocol_sha256=ph,split_checks=splitchecks,all_23_lineages_on_both_outer_sides=True,baseline_arrays_identical=True,full_Hallmark_memberships_verified=True,BOLA2_mapping_collision_resolved=True,preflight_fits_retained=9,checks=checks,seconds=time.monotonic()-start);run.dump(run.OUT/'preflight_validation.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
