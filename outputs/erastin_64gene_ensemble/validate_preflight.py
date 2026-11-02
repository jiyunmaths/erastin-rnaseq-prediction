#!/usr/bin/env python3
"""Independent simplex/gradient checks, registry and unchanged split checks."""
import time

import numpy as np

import common as c
from models import fit_weights,value_gradient,kkt
from make_splits import outer_split,inner_folds


def grid_minimum(pred,y,fold,resolution=100):
    weights=np.array([[i/resolution,j/resolution,1-(i+j)/resolution] for i in range(resolution+1) for j in range(resolution+1-i)])
    errors=pred@weights.T-y[:,None]
    values=np.mean([np.sqrt(np.mean(errors[fold==k]**2,axis=0)) for k in range(5)],axis=0)
    return float(values.min()),len(weights)


def main():
    begin=time.monotonic();ph=c.prepare();c.verify();split_checks=0;checks=[]
    for assay,s in c.STATE.items():
        for r,mask in enumerate(s['test']):
            np.testing.assert_array_equal(mask,outer_split(s['lineage'],s['groups'],s['seeds'][r]))
            np.testing.assert_array_equal(s['inner'][r],inner_folds(s['lineage'],s['groups'],mask,s['inner_seeds'][r]));split_checks+=6
        p,outer,valid,settings=c.inputs(assay,0);y=s['y'][valid];fold=s['inner'][0,valid]
        w,diag=fit_weights(p,y,fold);objective,gradient=value_gradient(w,p,y,fold)
        minimum,count=grid_minimum(p,y,fold)
        assert objective<=minimum+1e-10
        assert minimum-objective<1e-4
        changed=s['y'].copy();changed[s['test'][0]]=np.arange(np.sum(s['test'][0]))+1000
        modified,_=fit_weights(p,changed[valid],fold);np.testing.assert_array_equal(w,modified)
        checks.append(dict(assay=assay,training_only_weights=True,heldout_response_decoy_unchanged=True,
            simplex_grid_candidates=count,optimizer_better_than_grid_by=minimum-objective,kkt=diag['standardized_kkt_residual']))
    rng=np.random.default_rng(2026112602);n=100;fold=np.arange(n)%5;y=rng.normal(size=n)
    scenarios=[np.column_stack([y+rng.normal(size=n)*sd for sd in [.6,.6,.6]]),
               np.column_stack([y+rng.normal(size=n)*sd for sd in [.2,1.,1.5]])]
    for j,p in enumerate(scenarios):
        w,d=fit_weights(p,y,fold);minimum,count=grid_minimum(p,y,fold,resolution=200)
        objective,gradient=value_gradient(w,p,y,fold);assert objective<=minimum+1e-10 and minimum-objective<1e-4
        at=np.ones(3)/3;v,g=value_gradient(at,p,y,fold);error=0.
        for i in [0,1]:
            direction=np.eye(3)[i]-np.eye(3)[2];h=1e-6
            numeric=(value_gradient(at+h*direction,p,y,fold)[0]-value_gradient(at-h*direction,p,y,fold)[0])/(2*h)
            error=max(error,float(abs(numeric-g@direction)));assert error<1e-7
        checks.append(dict(synthetic=j,simplex_grid_candidates=count,optimizer_better_than_grid_by=minimum-objective,gradient_max_error=error,kkt=d['standardized_kkt_residual']))
    c.dump(c.OUT/'preflight_validation.json',dict(passed=True,protocol_sha256=ph,split_checks=split_checks,
        exact_gene_order_verified=True,registry_base_model_hashes_verified=120,checks=checks,seconds=time.monotonic()-begin))
    print('PREFLIGHT_PASSED',time.monotonic()-begin,flush=True)


if __name__=='__main__':main()
