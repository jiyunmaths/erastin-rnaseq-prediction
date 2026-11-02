#!/usr/bin/env python3
"""Verify matched data/splits, reused native-NN parity and independent APIs."""
import json,warnings
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import run_panels as run
from make_splits import inner_rows
from evaluate import validate_bootstrap,bootstrap

def native_nn(params,x,y,a,b,seeds):
    sx=StandardScaler().fit(x[a]);sy=StandardScaler().fit(y[a,None]);target=sy.transform(y[a,None]).ravel();preds=[]
    for seed in seeds:
        m=MLPRegressor(hidden_layer_sizes=(params['width'],),activation='tanh',solver='lbfgs',alpha=params['alpha'],max_iter=2000,max_fun=40000,tol=.001,early_stopping=False,random_state=int(seed))
        with warnings.catch_warnings():warnings.simplefilter('ignore');m.fit(sx.transform(x[a]),target)
        preds.append(sy.inverse_transform(m.predict(sx.transform(x[b]))[:,None]).ravel())
    return np.mean(preds,axis=0)

def main():
    ph=run.prepare();run.initialize();s=run.STATE;checks=[];assert len(s['y'])==744 and len(np.unique(s['groups']))==733 and len(np.unique(s['lineage']))==23 and 'Cervix' not in s['lineage']
    assert s['test'].shape==(20,744) and s['inner'].shape==(20,744) and s['nn_seeds'].shape==(20,3)
    split_checks=0
    for r,mask in enumerate(s['test']):
        tr=np.flatnonzero(~mask);te=np.flatnonzero(mask);assert len(tr)==595 and len(te)==149 and set(s['groups'][tr]).isdisjoint(s['groups'][te]) and set(s['lineage'][tr])==set(s['lineage'][te])==set(s['lineage']);split_checks+=1
        for k in range(5):
            a,b=inner_rows(s['inner'][r],k);assert set(a)|set(b)==set(tr) and set(a).isdisjoint(te) and set(b).isdisjoint(te) and set(s['groups'][a]).isdisjoint(s['groups'][b]);assert len(np.unique(s['lineage'][a]))==23;split_checks+=1
    for p in run.PANELS:assert s['x'][p].shape==(744,p) and len(set(s['genes'][p]))==p and np.isfinite(s['x'][p]).all() and np.all(np.ptp(s['x'][p],axis=0)>0)
    for r in [0,19]:
        inner=s['inner'][r];tr=np.flatnonzero(~s['test'][r]);te=np.flatnonzero(s['test'][r]);x=s['x'][64];y=s['y']
        root=run.OUT/'folds'/f'neural_network__p64__r{r+1:02d}';m=json.loads(root.with_suffix('.json').read_text());a=np.load(root.with_suffix('.npz'));loss=np.empty((4,5));maxerr=0.
        for k in range(5):
            aa,bb=inner_rows(inner,k)
            for c,params in enumerate(s['nn_grid']):
                pred=native_nn(params,x,y,aa,bb,s['nn_seeds'][r]);err=float(np.max(np.abs(pred-a['inner_validation_predictions'][c,bb])));maxerr=max(maxerr,err);assert err<1e-10
                loss[c,k]=np.sqrt(np.mean((y[bb]-pred)**2))
        c=int(np.argmin(loss.mean(axis=1)));assert c==m['selected_candidate'];pred=native_nn(s['nn_grid'][c],x,y,tr,te,s['nn_seeds'][r]);err=float(np.max(np.abs(pred-a['test_prediction'])));assert err<1e-10
        checks.append(dict(check='full_native_nn_nested_replay',repeat=r+1,candidates=4,inner_folds=5,inner_prediction_error=maxerr,outer_prediction_error=err))
        for p in run.PANELS:
            xx=s['x'][p];sx=StandardScaler().fit(xx[tr]);alphas=s['alphas'][[0,8,12,16]];pred,_=run.ridge_path(xx,y,tr,te,alphas)
            for c,alpha in enumerate(alphas):
                m=Ridge(alpha=float(alpha),solver='svd').fit(sx.transform(xx[tr]),y[tr]);err=float(np.max(np.abs(pred[:,c]-m.predict(sx.transform(xx[te])))));assert err<1e-10;checks.append(dict(check='ridge_api_parity',panel=p,repeat=r+1,alpha=float(alpha),error=err))
            mu,sd=y[tr].mean(),y[tr].std();target=(y[tr]-mu)/sd;seed=int(s['model_seeds'][r]);params=dict(run.xp.STRUCTURES[5],num_boost_round=600)
            model=xgb.train(run.xp.parameters(params,seed),run.matrix(xx[tr],s['genes'][p],target),num_boost_round=600)
            maxerr=0.
            for n in [100,300,600]:
                pp=run.xp.parameters(params,seed);pp.pop('nthread');pp.pop('seed')
                ref=xgb.XGBRegressor(**pp,n_jobs=1,random_state=seed,n_estimators=n).fit(xx[tr],target);expected=ref.predict(xx[te]);actual=model.predict(run.matrix(xx[te],s['genes'][p]),iteration_range=(0,n));err=float(np.max(np.abs(expected-actual)));assert err<1e-6;maxerr=max(maxerr,err)
            checks.append(dict(check='xgb_native_prefix_and_wrapper_parity',panel=p,repeat=r+1,prefixes=[100,300,600],error=maxerr))
    baseline=np.full((20,744),np.nan)
    for r in range(20):
        a=np.load(run.OUT/'folds'/f'neural_network__p64__r{r+1:02d}.npz');baseline[r,a['test_rows']]=a['test_prediction']
    boot=validate_bootstrap(s['y'],{'baseline':baseline,'identical':baseline.copy()},s['lineage'],s['test'],seed=2026110702)
    v=bootstrap(s['y'],{'baseline':baseline,'identical':baseline.copy()},s['lineage'],s['test'],count=5,seed=2026110703)
    for k in v['baseline']:np.testing.assert_array_equal(v['baseline'][k],v['identical'][k])
    result=dict(passed=True,protocol_sha256=ph,samples=744,lineages_in_both_sets=23,split_checks=split_checks,reused_nn64_models=20,native_nn_replay_repeats=[1,20],checks=checks,bootstrap_validation=boot,identical_prediction_bootstrap_differences_exactly_zero=True)
    run.dump(run.OUT/'preflight_validation.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
