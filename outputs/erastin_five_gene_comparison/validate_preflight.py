#!/usr/bin/env python3
"""Check fixed gene mapping, split reuse and all-candidate native API parity."""
import json,warnings
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
import run_comparison as run
from make_splits import inner_rows
from evaluate import validate_bootstrap

def native_nn(params,x,y,a,b,seeds):
    sx=StandardScaler().fit(x[a]);sy=StandardScaler().fit(y[a,None]);target=sy.transform(y[a,None]).ravel();preds=[]
    for seed in seeds:
        m=MLPRegressor(hidden_layer_sizes=(params['width'],),activation='tanh',solver='lbfgs',alpha=params['alpha'],max_iter=2000,max_fun=40000,tol=.001,early_stopping=False,random_state=int(seed))
        with warnings.catch_warnings():warnings.simplefilter('ignore');m.fit(sx.transform(x[a]),target)
        preds.append(sy.inverse_transform(m.predict(sx.transform(x[b]))[:,None]).ravel())
    return np.mean(preds,axis=0)

def independent(f,params,x,y,a,b,seeds):
    if f=='ridge':
        sx=StandardScaler().fit(x[a]);m=Ridge(alpha=params['alpha'],solver='svd').fit(sx.transform(x[a]),y[a]);return m.predict(sx.transform(x[b]))
    return native_nn(params,x,y,a,b,seeds)

def main():
    ph=run.prepare();run.initialize();s=run.STATE;checks=[];splitchecks=0;assert list(s['genes'][5])==run.FIVE_GENES
    cols=[list(s['genes'][64]).index(g) for g in run.FIVE_GENES];np.testing.assert_array_equal(s['x'][5],s['x'][64][:,cols]);assert s['x'][5].shape==(744,5)
    for r,mask in enumerate(s['test']):
        tr=np.flatnonzero(~mask);te=np.flatnonzero(mask);assert len(tr)==595 and len(te)==149 and set(s['groups'][tr]).isdisjoint(s['groups'][te]);assert set(s['lineage'][tr])==set(s['lineage'][te]) and len(set(s['lineage'][tr]))==23;splitchecks+=1
        for k in range(5):
            a,b=inner_rows(s['inner'][r],k);assert set(a)|set(b)==set(tr) and set(s['groups'][a]).isdisjoint(s['groups'][b]) and len(set(s['lineage'][a]))==23;splitchecks+=1
    assert 'Cervix' not in s['lineage']
    # These four preflight fits are retained as part of the 40 new five-gene models.
    for r in [0,19]:
        inner=s['inner'][r];tr=np.flatnonzero(~s['test'][r]);te=np.flatnonzero(s['test'][r])
        for f in run.FAMILIES:
            m=run.fit_job(5,f,r,ph);a=np.load(run.OUT/'folds'/f'{f}__p5__r{r+1:02d}.npz');loss=np.empty_like(a['inner_fold_rmse']);error=0.
            for k in range(5):
                aa,bb=inner_rows(inner,k)
                for c,params in enumerate(run.grid_for(f)):
                    ref=independent(f,params,s['x'][5],s['y'],aa,bb,s['nn_seeds'][r]);error=max(error,float(np.max(np.abs(ref-a['inner_validation_predictions'][c,bb]))));loss[c,k]=np.sqrt(np.mean((s['y'][bb]-ref)**2))
            assert error<1e-10;chosen=int(np.argmin(loss.mean(axis=1)));assert chosen==m['selected_candidate']
            ref=independent(f,m['selected_parameters'],s['x'][5],s['y'],tr,te,s['nn_seeds'][r]);outererror=float(np.max(np.abs(ref-a['test_prediction'])));assert outererror<1e-10
            checks.append(dict(family=f,panel=5,repeat=r+1,all_inner_candidates_verified=True,inner_prediction_error=error,outer_prediction_error=outererror))
            root=run.OUT/'folds'/f'{f}__p64__r{r+1:02d}';m=json.load(open(root.with_suffix('.json')));a=np.load(root.with_suffix('.npz'));ref=independent(f,m['selected_parameters'],s['x'][64],s['y'],tr,te,s['nn_seeds'][r]);err=float(np.max(np.abs(ref-a['test_prediction'])));assert err<1e-10
            checks.append(dict(family=f,panel=64,repeat=r+1,reused_selected_model=True,outer_prediction_error=err))
    old=np.load(run.WS/'outputs/erastin_stratified_panel_comparison/predictions.npz');boot=validate_bootstrap(s['y'],{'ridge__64':old['ridge__64'],'neural_network__64':old['neural_network__64']},s['lineage'],s['test'],seed=2026110902)
    v=dict(passed=True,protocol_sha256=ph,five_genes=run.FIVE_GENES,five_gene_indices_in_64=cols,samples=744,lineages_in_both_sets=23,split_checks=splitchecks,checks=checks,bootstrap_validation=boot,preflight_fits_retained=4,
        analysis_is_exploratory_due_to_prior_test_informed_gene_selection=True)
    run.dump(run.OUT/'preflight_validation.json',v);print(json.dumps(v,indent=2),flush=True)
if __name__=='__main__':main()
