#!/usr/bin/env python3
"""Check frozen splits, native-path parity and augmented-Lasso equivalence."""
import json,warnings
import numpy as np
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler
import run_comparison as run
from enet_helpers import fit_path,fit_selected,kkt_residual,GRID,ALPHAS,L1_RATIOS
from make_splits import inner_rows
from evaluate import validate_bootstrap

def independent(params,x,y,a,b):
    """Equivalent Lasso on augmented rows; no Gram matrix/path warm starts."""
    sx=StandardScaler().fit(x[a]);sy=StandardScaler().fit(y[a,None]);z=sx.transform(x[a]);target=sy.transform(y[a,None]).ravel();n,p=z.shape
    strength=params['alpha'];ratio=params['l1_ratio']
    xx=np.vstack([z,np.sqrt(n*strength*(1-ratio))*np.eye(p)]);yy=np.r_[target,np.zeros(p)]
    model=Lasso(alpha=strength*ratio*n/(n+p),fit_intercept=False,max_iter=50000,tol=1e-10,selection='cyclic')
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');model.fit(xx,yy)
    assert not any(v.category.__name__=='ConvergenceWarning' for v in caught)
    err=kkt_residual(z,target,model.coef_,0.,strength,ratio);assert err<1e-6
    return (sx.transform(x[b])@model.coef_)*sy.scale_[0]+sy.mean_[0]

def main():
    ph=run.prepare();run.initialize();s=run.STATE;checks=[];splitchecks=0
    genes=['SLC7A11','GPX4','ABCC1','ACSL4','AIFM2'];assert list(s['genes'][5])==genes
    cols=[list(s['genes'][64]).index(g) for g in genes];np.testing.assert_array_equal(s['x'][5],s['x'][64][:,cols]);assert s['x'][5].shape==(744,5)
    for r,mask in enumerate(s['test']):
        tr=np.flatnonzero(~mask);te=np.flatnonzero(mask);assert len(tr)==595 and len(te)==149 and set(s['groups'][tr]).isdisjoint(s['groups'][te]);assert set(s['lineage'][tr])==set(s['lineage'][te]) and len(set(s['lineage'][tr]))==23;splitchecks+=1
        for k in range(5):
            a,b=inner_rows(s['inner'][r],k);assert set(a)|set(b)==set(tr) and set(s['groups'][a]).isdisjoint(s['groups'][b]) and len(set(s['lineage'][a]))==23 and len(set(s['lineage'][b]))==22;splitchecks+=1
    assert 'Cervix' not in s['lineage']
    for r in [0,19]:
        tr=np.flatnonzero(~s['test'][r]);te=np.flatnonzero(s['test'][r]);a,b=inner_rows(s['inner'][r],0)
        for panel in run.PANELS:
            x=s['x'][panel];pred,coef,diag,scale=fit_path(x,s['y'],a,b);error=ind_error=0.
            # Include both endpoints, modest shrinkage, near-ridge and pure Lasso.
            candidates=[j*len(ALPHAS)+i for j in range(len(L1_RATIOS)) for i in [0,16,28,40]]
            for c in candidates:
                ref,_,_=fit_selected(x,s['y'],a,b,GRID[c]);error=max(error,float(np.max(np.abs(ref-pred[c]))))
                alt=independent(GRID[c],x,s['y'],a,b);ind_error=max(ind_error,float(np.max(np.abs(alt-pred[c]))))
            assert error<1e-7 and ind_error<1e-7
            m=run.fit_job(panel,r,ph);saved=np.load(run.OUT/'folds'/f'elastic_net__p{panel}__r{r+1:02d}.npz');params=m['selected_parameters']
            ref=independent(params,x,s['y'],tr,te);outererror=float(np.max(np.abs(ref-saved['test_prediction'])));ie=0.
            for k in range(5):
                aa,bb=inner_rows(s['inner'][r],k);ref=independent(params,x,s['y'],aa,bb);ie=max(ie,float(np.max(np.abs(ref-saved['inner_validation_predictions'][m['selected_candidate'],bb]))))
            assert max(outererror,ie)<1e-7
            checks.append(dict(panel=panel,repeat=r+1,candidates_native_and_augmented_lasso_checked=len(candidates),native_path_max_prediction_error=error,augmented_lasso_path_max_prediction_error=ind_error,selected_outer_augmented_lasso_error=outererror,selected_inner_augmented_lasso_error=ie))
    old=np.load(run.WS/'outputs/erastin_five_gene_comparison/predictions.npz');boot=validate_bootstrap(s['y'],{'ridge__64':old['ridge__64'],'neural_network__64':old['neural_network__64']},s['lineage'],s['test'],seed=2026111002)
    v=dict(passed=True,protocol_sha256=ph,five_genes=genes,five_gene_indices_in_64=cols,samples=744,lineages_in_both_sets=23,split_checks=splitchecks,checks=checks,bootstrap_validation=boot,preflight_fits_retained=4,analysis_is_exploratory_due_to_prior_test_informed_gene_selection=True)
    run.dump(run.OUT/'preflight_validation.json',v);print(json.dumps(v,indent=2),flush=True)
if __name__=='__main__':main()
