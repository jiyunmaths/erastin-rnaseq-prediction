#!/usr/bin/env python3
"""Independent mixed-feature reconstruction, fit and missing-category controls."""
import copy,json,time,warnings
import numpy as np
from sklearn.linear_model import Ridge,Lasso,ElasticNet
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.exceptions import ConvergenceWarning
import run_comparison as run
from culture_features import CAT_FIELDS,MIN_COUNTS,MISSING,OTHER
from fit_helpers import elastic_path,GRID,kkt
from make_splits import inner_rows
from weighted_mlp import WeightedMLPRegressor

def independent_features(x,meta,a,b):
    # Construct the design from raw current-training values, without using the
    # fitted preprocessor or OneHotEncoder. Keep indicator columns unscaled.
    serum=meta.serum_percent.to_numpy(float);known=serum[a][np.isfinite(serum[a])];median=float(np.median(known)) if len(known) else 0.
    cont=np.column_stack([x,np.where(np.isfinite(serum),serum,median)]);mean=cont[a].mean(axis=0);sd=cont[a].std(axis=0);sd=np.where(sd==0,1.,sd);numeric=(cont-mean)/sd;blocks=[numeric,(~np.isfinite(serum)).astype(float)[:,None]];categories=[]
    for field in CAT_FIELDS:
        values=meta[field].astype(str).to_numpy();u,n=np.unique(values[a],return_counts=True);kept=sorted(v for v,c in zip(u,n) if v not in [MISSING,OTHER] and c>=MIN_COUNTS[field]);cats=kept+[MISSING,OTHER];mapped=np.array([v if v==MISSING or v in kept else OTHER for v in values]);blocks.append(np.column_stack([mapped==v for v in cats]).astype(float));categories.append(cats)
    raw=np.column_stack(blocks);center=raw[a].mean(axis=0);z=raw-center
    return z[a],z[b],dict(serum_median=median,mean=mean,scale=sd,center=center,categories=categories)

def check_preprocessor(panel,a,b,prep=None):
    s=run.STATE;z,zz,prep_actual=run.preprocess(panel,a,b)
    if prep is None:prep=prep_actual
    ref,refv,d=independent_features(s['x'][panel],s['meta'],a,b)
    np.testing.assert_allclose(z,ref,atol=1e-12,rtol=0);np.testing.assert_allclose(zz,refv,atol=1e-12,rtol=0)
    np.testing.assert_allclose(prep.transform(s['x'][panel][a],s['meta'].iloc[a]),ref,atol=1e-12,rtol=0);np.testing.assert_allclose(prep.transform(s['x'][panel][b],s['meta'].iloc[b]),refv,atol=1e-12,rtol=0)
    assert prep.serum_median==d['serum_median'] and prep.category_lists==d['categories'] and prep.train_n==len(a)
    np.testing.assert_allclose(prep.scaler.mean_,d['mean'],atol=1e-12,rtol=0);np.testing.assert_allclose(prep.scaler.scale_,d['scale'],atol=1e-12,rtol=0);np.testing.assert_allclose(prep.center,d['center'],atol=1e-12,rtol=0)
    assert np.max(np.var(z[:,panel+1:],axis=0))<=.250000000001
    return ref,refv,prep

def independent(f,params,z,y,zz,seeds):
    if f=='ridge':return Ridge(alpha=params['alpha'],solver='svd').fit(z,y).predict(zz)
    sy=StandardScaler().fit(y[:,None]);target=sy.transform(y[:,None]).ravel()
    if f=='elastic_net':
        n,p=z.shape;alpha=params['alpha'];ratio=params['l1_ratio'];xx=np.vstack([z,np.sqrt(n*alpha*(1-ratio))*np.eye(p)]);yy=np.r_[target,np.zeros(p)];m=Lasso(alpha=alpha*ratio*n/(n+p),fit_intercept=False,max_iter=50000,tol=1e-10,selection='cyclic')
        with warnings.catch_warnings():warnings.simplefilter('ignore',DeprecationWarning);warnings.simplefilter('error',ConvergenceWarning);m.fit(xx,yy)
        assert kkt(z,target,m.coef_,0.,alpha,ratio)<1e-6
        return zz@m.coef_*sy.scale_[0]+sy.mean_[0]
    pp=[]
    for seed in seeds:
        m=MLPRegressor(hidden_layer_sizes=(params['width'],),activation='tanh',solver='lbfgs',alpha=params['alpha'],max_iter=2000,max_fun=40000,tol=.001,early_stopping=False,random_state=int(seed))
        with warnings.catch_warnings():warnings.simplefilter('ignore',DeprecationWarning);warnings.simplefilter('error',ConvergenceWarning);m.fit(z,target)
        pp.append(sy.inverse_transform(m.predict(zz)[:,None]).ravel())
    return np.mean(pp,axis=0)

def main():
    begin=time.monotonic();ph=run.prepare();run.initialize();s=run.STATE;checks=[];splitchecks=0
    assert len(s['ids'])==386 and len(set(s['groups']))==379 and len(set(s['lineage']))==20
    assert s['meta'].serum_percent.isna().sum()==19 and s['meta'].medium_code.eq(MISSING).sum()==19
    for r,mask in enumerate(s['test']):
        tr=np.flatnonzero(~mask);te=np.flatnonzero(mask);assert (len(tr),len(te))==(308,78) and set(s['groups'][tr]).isdisjoint(s['groups'][te]) and set(s['lineage'][tr])==set(s['lineage'][te])==set(s['lineage']);splitchecks+=1
        for k in range(5):
            a,b=inner_rows(s['inner'][r],k);assert set(a)|set(b)==set(tr) and set(s['groups'][a]).isdisjoint(s['groups'][b]) and len(set(s['lineage'][a]))==20;splitchecks+=1
    tr=np.flatnonzero(~s['test'][0]);te=np.flatnonzero(s['test'][0]);a,b=inner_rows(s['inner'][0],0)
    # Unknown categories and missingness have explicit, distinct destinations;
    # test transformations cannot alter the learned categories/scaling.
    z,zz,prep=check_preprocessor(64,a,b);frozen=copy.deepcopy(prep.__dict__);novel=s['meta'].iloc[b].copy();novel.loc[novel.index[0],CAT_FIELDS]=['unseen_growth','unseen_medium','unseen_basal'];novel.loc[novel.index[1],CAT_FIELDS]=MISSING;novel.loc[novel.index[0],'serum_percent']=np.nan
    mapped=prep.map_categories(novel);assert (mapped[0]==OTHER).all() and (mapped[1]==MISSING).all();transformed=prep.transform(s['x'][64][b],novel);assert np.isfinite(transformed).all();assert prep.category_lists==frozen['category_lists'] and prep.frequent==frozen['frequent'];np.testing.assert_array_equal(prep.scaler.mean_,frozen['scaler'].mean_)
    # Verify equivalence to the unit-weight adapter used by the existing baselines.
    sy=StandardScaler().fit(s['y'][a,None]);target=sy.transform(s['y'][a,None]).ravel();kwargs=dict(hidden_layer_sizes=(16,),activation='tanh',solver='lbfgs',alpha=100,max_iter=2000,max_fun=40000,tol=.001,early_stopping=False,random_state=int(s['nn_seeds'][0,0]));native=MLPRegressor(**kwargs).fit(z,target);weighted=WeightedMLPRegressor(**kwargs).fit(z,target,sample_weight=np.ones(len(a)));parity=float(np.max(np.abs(native.predict(zz)-weighted.predict(zz))));assert parity<1e-7
    for panel in run.PANELS:
        for f in run.FAMILIES:
            m=run.fit_job(panel,f,0,ph);arrays=np.load(run.root_for(f,panel,'culture',0).with_suffix('.npz'));error=0.;count=0
            for k in range(5):
                aa,bb=inner_rows(s['inner'][0],k);z,zz,_=check_preprocessor(panel,aa,bb);candidates=range(len(run.grid_for(f))) if f!='elastic_net' else [m['selected_candidate']]
                for c in candidates:
                    ref=independent(f,run.grid_for(f)[c],z,s['y'][aa],zz,s['nn_seeds'][0]);error=max(error,float(np.max(np.abs(ref-arrays['inner_validation_predictions'][c,bb]))));count+=1
            z,zz,_=check_preprocessor(panel,tr,te);outer=independent(f,m['selected_parameters'],z,s['y'][tr],zz,s['nn_seeds'][0]);outererror=float(np.max(np.abs(outer-arrays['test_prediction'])));assert max(error,outererror)<1e-7
            checks.append(dict(panel=panel,family=f,inner_candidate_fits_checked=count,max_inner_error=error,max_outer_error=outererror));print('PREFLIGHT_ARM',panel,f,flush=True)
        z,zz,_=check_preprocessor(panel,a,b);path,_=elastic_path(z,s['y'][a],zz);nativeerror=augerror=0.
        for j in range(8):
            for i in [0,16,28,40]:
                c=j*41+i;p=GRID[c];sy=StandardScaler().fit(s['y'][a,None]);native=ElasticNet(**p,max_iter=50000,tol=1e-9,selection='cyclic').fit(z,sy.transform(s['y'][a,None]).ravel());pred=sy.inverse_transform(native.predict(zz)[:,None]).ravel();ref=independent('elastic_net',p,z,s['y'][a],zz,s['nn_seeds'][0]);nativeerror=max(nativeerror,float(np.max(np.abs(pred-path[c]))));augerror=max(augerror,float(np.max(np.abs(ref-path[c]))))
        assert max(nativeerror,augerror)<1e-7;checks.append(dict(panel=panel,path_candidates_checked=32,native_error=nativeerror,augmented_lasso_error=augerror))
    v=dict(passed=True,protocol_sha256=ph,split_checks=splitchecks,all_20_lineages_on_both_outer_sides=True,independent_preprocessor_checks_passed=True,unseen_missing_category_controls_passed=True,native_vs_unit_weight_nn_error=parity,checks=checks,preflight_fits_retained=9,elapsed_seconds=time.monotonic()-begin);run.dump(run.OUT/'preflight_validation.json',v);print(json.dumps(v,indent=2),flush=True)
if __name__=='__main__':main()
