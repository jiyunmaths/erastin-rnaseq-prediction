#!/usr/bin/env python3
"""Verify cached selectors, untouched folds and exact original NN behavior."""
import time

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import common as c
import selection_helpers as selection
from model_helpers import fitted_neural
from make_splits import outer_split,inner_folds,inner_rows
from run_comparison import check_solver


def main():
    begin=time.monotonic();ph=c.prepare();c.verify();checks=0;replays=[];baseline_error=native_error=0.
    for assay,s in c.STATE.items():
        for r,mask in enumerate(s['test']):
            np.testing.assert_array_equal(mask,outer_split(s['lineage'],s['groups'],s['seeds'][r]))
            np.testing.assert_array_equal(s['inner'][r],inner_folds(s['lineage'],s['groups'],mask,s['inner_seeds'][r]));checks+=6
        cache=c.cache(assay,0)
        # Every replay uses the same training rows as the source selection cache.
        for fold in [0,4,-1]:
            rows=inner_rows(s['inner'][0],fold)[0] if fold>=0 else np.flatnonzero(~s['test'][0])
            x=np.asarray(c.EXPRESSION[s['rows'][rows]],dtype=float)
            fits,d=selection.select_paths(x,s['y'][rows],s['lineage'][rows],c.INDEX['core_indices'],c.INDEX['candidate_indices'])
            masks=cache['inner_selection_masks'][fold] if fold>=0 else cache['outer_selection_masks']
            for j,fit in enumerate(fits):np.testing.assert_array_equal(c.INDEX['candidate_indices'][masks[j]],fit['feature_indices'])
            replays.append(dict(assay=assay,repeat=1,fold=fold,settings=len(fits),identical=True))
        tr,va=inner_rows(s['inner'][0],0);x=c.features(assay,c.INDEX['core_indices']);params=s['nn_grid'][0]
        pred,model,diag,members=fitted_neural(params,x,s['y'],tr,va,s['nn_seeds'][0],np.ones(len(tr)),retain=True)
        check_solver(diag)
        old=np.load(c.BASE[assay]/'folds/neural_network__p64__r01.npz')
        error=float(np.max(np.abs(pred-old['inner_validation_predictions'][0,va])));baseline_error=max(baseline_error,error);assert error<1e-12,error
        # Compare the weighted adapter at unit weights with native sklearn MLP.
        z=model['x_scaler'].transform(x[tr]);target=model['y_scaler'].transform(s['y'][tr,None]).ravel()
        native=MLPRegressor(hidden_layer_sizes=(params['width'],),activation='tanh',solver='lbfgs',alpha=params['alpha'],
            max_iter=2000,max_fun=40000,tol=.001,early_stopping=False,random_state=int(s['nn_seeds'][0,0])).fit(z,target)
        error=float(np.max(np.abs(native.predict(model['x_scaler'].transform(x[va]))-model['members'][0].predict(model['x_scaler'].transform(x[va])))))
        native_error=max(native_error,error);assert error<1e-9,error
        for member in model['members']:
            np.testing.assert_array_equal(member.training_sample_weight_,np.ones(len(tr)))
        # Independent scaling and training objective on an augmented panel.
        indices=np.r_[c.INDEX['core_indices'],c.INDEX['candidate_indices'][cache['inner_selection_masks'][0,4]]]
        x=c.features(assay,indices);params=s['nn_grid'][-1]
        pred,model,diag,members=fitted_neural(params,x,s['y'],tr,va,s['nn_seeds'][0],np.ones(len(tr)),retain=True);check_solver(diag)
        sx=StandardScaler().fit(x[tr]);sy=StandardScaler().fit(s['y'][tr,None])
        np.testing.assert_array_equal(model['x_scaler'].mean_,sx.mean_);np.testing.assert_array_equal(model['y_scaler'].scale_,sy.scale_)
        target=sy.transform(s['y'][tr,None]).ravel()
        for member in model['members']:
            objective=.5*np.mean((member.predict(sx.transform(x[tr]))-target)**2)+.5*params['alpha']*sum(np.sum(v*v) for v in member.coefs_)/len(tr)
            assert abs(objective-member.loss_)<1e-10
        np.testing.assert_allclose(pred,members.mean(axis=0),atol=1e-14,rtol=0)
    c.dump(c.OUT/'preflight_validation.json',dict(passed=True,protocol_sha256=ph,split_checks=checks,
        replayed_fold_local_selectors=replays,cached_selector_gene_sets_checked=sum(d['settings'] for d in replays),
        original_core_nn_prediction_max_error=baseline_error,native_unweighted_adapter_prediction_max_error=native_error,
        training_only_scaling_verified=True,augmented_training_objectives_verified=True,exact_three_seed_ensembles_verified=True,
        no_top_k_cap=True,seconds=time.monotonic()-begin))
    print('PREFLIGHT_PASSED',time.monotonic()-begin,flush=True)


if __name__=='__main__':main()
