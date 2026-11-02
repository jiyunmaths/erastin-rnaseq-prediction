#!/usr/bin/env python3
"""Check fold protection, full-genome solver and independent ridge arithmetic."""
import time
import warnings

import numpy as np
from sklearn.linear_model import ElasticNet,Lasso,Ridge
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

import common as c
import models
from make_splits import inner_rows,outer_split,inner_folds


def main():
    begin=time.monotonic();ph=c.prepare();c.initialize();c.verify()
    split_checks=0
    for assay,s in c.STATE.items():
        for r,mask in enumerate(s['test']):
            np.testing.assert_array_equal(mask,outer_split(s['lineage'],s['groups'],s['seeds'][r]))
            np.testing.assert_array_equal(s['inner'][r],inner_folds(s['lineage'],s['groups'],mask,s['inner_seeds'][r]))
            split_checks+=6
        np.testing.assert_array_equal(c.features(assay,np.arange(len(s['y'])),c.INDEX['core_indices']),s['xcore'])
    s=c.STATE['CTRP'];a,b=inner_rows(s['inner'][0],0)
    x=c.features('CTRP',a);y=s['y'][a];core,candidates=c.INDEX['core_indices'],c.INDEX['candidate_indices']
    fits,diag=models.select_paths(x,y,s['lineage'][a],core,candidates)
    z,target,ix,dd=models.residual_design(x,y,s['lineage'][a],core,candidates)
    # Independent full-genome native solver, no working sets or feature filter.
    comparisons=[]
    for setting in [dict(l1_ratio=.1,alpha_fraction=.5),dict(l1_ratio=.5,alpha_fraction=.2),dict(l1_ratio=.99,alpha_fraction=.1)]:
        j=models.SELECTORS.index(setting);fit=fits[j]
        native=ElasticNet(alpha=fit['alpha'],l1_ratio=setting['l1_ratio'],fit_intercept=False,
            precompute=False,tol=1e-10,max_iter=50000,selection='cyclic')
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');native.fit(z,target)
        assert not any(issubclass(w.category,ConvergenceWarning) for w in caught)
        beta=np.zeros(len(ix));positions=np.searchsorted(ix,fit['feature_indices']);beta[positions]=fit['coefficients']
        error=float(np.max(np.abs(native.coef_-beta)))
        assert error<3e-6,error
        np.testing.assert_array_equal(ix[np.abs(native.coef_)>models.NONZERO],fit['feature_indices'])
        pred_error=float(np.max(np.abs(z@(native.coef_-beta))))
        comparisons.append(dict(**setting,coefficient_max_error=error,training_prediction_max_error=pred_error,
            selected_genes=len(positions),identical_selected_gene_sets=True,native_kkt=models.kkt(z,target,native.coef_,fit['alpha'],setting['l1_ratio'])))
    # SVD residualization checked against an explicit least-squares design.
    controls=np.column_stack([np.ones(len(a)),StandardScaler().fit_transform(x[:,core]),
        (s['lineage'][a,None]==np.unique(s['lineage'][a])[None,:]).astype(float)])
    pos=np.arange(0,len(candidates),997);xx=x[:,candidates[pos]]
    residual=xx-controls@np.linalg.lstsq(controls,xx,rcond=None)[0]
    yy=y-controls@np.linalg.lstsq(controls,y,rcond=None)[0];yy-=yy.mean();yy/=np.sqrt(np.mean(yy**2))
    np.testing.assert_allclose(target,yy,atol=1e-11,rtol=0)
    pos_good=[j for j in pos if candidates[j] in set(ix)]
    residual=x[:,candidates[pos_good]]-controls@np.linalg.lstsq(controls,x[:,candidates[pos_good]],rcond=None)[0]
    residual-=residual.mean(axis=0);residual/=np.sqrt(np.mean(residual**2,axis=0))
    np.testing.assert_allclose(z[:,np.searchsorted(ix,candidates[pos_good])],residual,atol=1e-9,rtol=0)
    # Independent downstream Ridge on one genuine selected panel.
    selected=np.r_[core,fits[7]['feature_indices']]
    test=c.features('CTRP',b,selected);train=x[:,selected]
    pp,_=models.ridge_path(train,y,test,models.ALPHAS)
    scale=StandardScaler().fit(train);ridge_error=0.
    for j,alpha in enumerate(models.ALPHAS):
        native=Ridge(alpha=alpha,solver='svd').fit(scale.transform(train),y)
        err=float(np.max(np.abs(native.predict(scale.transform(test))-pp[j])))
        ridge_error=max(ridge_error,err);assert err<1e-9
    # Small independent augmented-Lasso equivalence for the EN objective.
    rng=np.random.default_rng(2026112302);small=rng.normal(size=(100,12));tt=rng.normal(size=100)
    small-=small.mean(axis=0);tt-=tt.mean();alpha,ratio=.08,.5
    beta,gaps,_,_=models.working_path(np.asfortranarray(small),tt,np.array([alpha]),ratio)
    augmented=np.vstack([small,np.sqrt(len(tt)*alpha*(1-ratio))*np.eye(small.shape[1])])
    response=np.r_[tt,np.zeros(small.shape[1])]
    lasso=Lasso(alpha=alpha*ratio*len(tt)/len(response),fit_intercept=False,tol=1e-12,max_iter=50000).fit(augmented,response)
    equivalence=float(np.max(np.abs(lasso.coef_-beta[:,0])));assert equivalence<1e-6
    # A validation-only response-correlated gene has no training variation.
    synthetic=rng.normal(size=(80,18));synthetic[:60,-1]=0
    synthetic[60:,-1]=rng.normal(size=20);response=rng.normal(size=80);response[60:]=synthetic[60:,-1]
    controls_ix=np.arange(3);candidate_ix=np.arange(3,18)
    chosen,_=models.select_paths(synthetic[:60],response[:60],np.array(['a']*30+['b']*30),controls_ix,candidate_ix,selectors=[models.SELECTORS[0]])
    assert 17 not in chosen[0]['feature_indices']
    c.dump(c.OUT/'preflight_validation.json',dict(passed=True,protocol_sha256=ph,split_checks=split_checks,
        core64_values_exact=True,independent_native_full_genome_checks=comparisons,
        independent_residualization_passed=True,independent_ridge_penalties_checked=len(models.ALPHAS),ridge_prediction_max_error=ridge_error,
        augmented_lasso_coefficient_max_error=equivalence,validation_only_decoy_not_selected=True,
        pilot_diagnostics=diag,seconds=time.monotonic()-begin))
    print('PREFLIGHT_PASSED',time.monotonic()-begin,flush=True)


if __name__=='__main__':main()
