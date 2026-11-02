#!/usr/bin/env python3
"""Verify grid reuse, fold-local selectors and stronger-penalty NN objectives."""
import time
import warnings

import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler

import common as c
import selection_helpers as selection
from model_helpers import fitted_neural
from make_splits import inner_rows,outer_split,inner_folds
from run_comparison import check_solver


def main():
    begin=time.monotonic();ph=c.prepare();c.verify();split_checks=0;replay=[];baseline_error=0.;native_error=0.;timings=[]
    for assay,s in c.STATE.items():
        grid=c.grid_for(assay);old=json_grid=[dict(selector=j,**p) for j in range(-1,20) for p in s['old_nn_grid']]
        assert [grid[j] for j in c.old_columns(assay)]==old
        for r,mask in enumerate(s['test']):
            np.testing.assert_array_equal(mask,outer_split(s['lineage'],s['groups'],s['seeds'][r]))
            np.testing.assert_array_equal(s['inner'][r],inner_folds(s['lineage'],s['groups'],mask,s['inner_seeds'][r]));split_checks+=6
        tr,va=inner_rows(s['inner'][0],0);cache=c.cache(assay,0)
        full=np.asarray(c.EXPRESSION[s['rows'][tr]],dtype=float)
        fits,diag=selection.select_paths(full,s['y'][tr],s['lineage'][tr],c.INDEX['core_indices'],c.INDEX['candidate_indices'])
        for j,f in enumerate(fits):np.testing.assert_array_equal(f['feature_indices'],c.INDEX['candidate_indices'][cache['inner_selection_masks'][0,j]])
        replay.append(dict(assay=assay,repeat=1,fold=0,identical_gene_sets=20))
        indices=np.r_[c.INDEX['core_indices'],c.INDEX['candidate_indices'][cache['inner_selection_masks'][0,4]]]
        x=c.features(assay,indices);oldparam=s['old_nn_grid'][0]
        p,_,d,_=fitted_neural(oldparam,x,s['y'],tr,va,s['nn_seeds'][0],np.ones(len(tr)));check_solver(d)
        original=np.load(c.OLD/assay/'folds/inner__r01__f0.npz');col=5*len(s['old_nn_grid'])
        err=float(np.max(np.abs(p-original['validation_predictions'][col])));baseline_error=max(baseline_error,err);assert err<1e-12
        for alpha in [300.,3000.,30000.]:
            params=dict(width=16,alpha=alpha)
            pred,model,d,members=fitted_neural(params,x,s['y'],tr,va,s['nn_seeds'][0],np.ones(len(tr)),retain=True);check_solver(d)
            timings.append(dict(assay=assay,alpha=alpha,genes=len(indices),seconds=d['seconds'],max_iterations=max(v['iterations'] for v in d['members'])))
            sx,sy=model['x_scaler'],model['y_scaler'];target=sy.transform(s['y'][tr,None]).ravel()
            np.testing.assert_allclose(sx.mean_,StandardScaler().fit(x[tr]).mean_,atol=1e-12,rtol=0)
            for v in model['members']:
                loss=.5*np.mean((v.predict(sx.transform(x[tr]))-target)**2)+.5*alpha*sum(np.sum(t*t) for t in v.coefs_)/len(tr)
                assert abs(loss-v.loss_)<1e-10
            np.testing.assert_allclose(pred,members.mean(axis=0),atol=1e-14,rtol=0)
            if alpha==3000.:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore',DeprecationWarning)
                    v=MLPRegressor(hidden_layer_sizes=(16,),activation='tanh',solver='lbfgs',alpha=alpha,max_iter=2000,max_fun=40000,tol=.001,early_stopping=False,random_state=int(s['nn_seeds'][0,0])).fit(sx.transform(x[tr]),target)
                err=float(np.max(np.abs(v.predict(sx.transform(x[va]))-model['members'][0].predict(sx.transform(x[va])))));native_error=max(native_error,err);assert err<1e-9
    c.dump(c.OUT/'preflight_validation.json',dict(passed=True,protocol_sha256=ph,split_checks=split_checks,
        exact_reuse_grid_mapping=True,replayed_selectors=replay,original_candidate_prediction_max_error=baseline_error,
        native_mlp_strong_penalty_prediction_max_error=native_error,training_objectives_verified=True,training_only_scaling=True,
        computational_pilot=timings,seconds=time.monotonic()-begin))
    print('PREFLIGHT_PASSED',time.monotonic()-begin,flush=True)


if __name__=='__main__':main()
