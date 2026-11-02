#!/usr/bin/env python3
"""Verify exclusion, response-blind group stratification, tuning and statistics."""
import json,warnings
import numpy as np
from sklearn.neural_network import MLPRegressor
import run_stratified as run
from make_splits import outer_split,inner_folds,inner_rows,sample_group_subset
from lineage_weights import weights_for
from weighted_mlp import WeightedMLPRegressor
from evaluate import validate_bootstrap

def main():
    ph=run.prepare();run.initialize();s=run.STATE;labels=set(s['lineage']);old=np.load(run.WS/'outputs/erastin_nn_moderate_weights/inputs.npz');keep=old['lineage']!='Cervix'
    assert 'Cervix' not in labels and len(labels)==23 and len(s['ids'])==744
    for k,v in [('sample_ids',s['ids']),('group_ids',s['groups']),('lineage',s['lineage']),('x',s['x']),('y',s['y'])]:np.testing.assert_array_equal(old[k][keep],v)
    for name in ['weighted_mlp.py','lineage_weights.py']:assert run.sha(run.OUT/name)==run.sha(run.WS/'outputs/erastin_nn_moderate_weights'/name)
    groups=s['groups'];lineage=s['lineage'];outer_checks=inner_checks=weight_checks=0;protected=[]
    assert len(np.unique(s['test'],axis=0))==20
    for r,mask in enumerate(s['test']):
        np.testing.assert_array_equal(mask,outer_split(lineage,groups,s['seeds'][r]));np.testing.assert_array_equal(s['inner'][r],inner_folds(lineage,groups,mask,s['inner_seeds'][r]))
        tr=np.flatnonzero(~mask);te=np.flatnonzero(mask);assert set(groups[tr]).isdisjoint(groups[te]);assert set(lineage[tr])==set(lineage[te])==labels;outer_checks+=1
        for l in labels:
            n=np.sum(lineage==l);target=max(1,min(n-1,int(np.floor(.2*n+.5))));assert np.sum(mask&(lineage==l))==target
        for group in np.unique(groups):assert np.unique(mask[groups==group]).size==1
        assert np.sum(mask&(lineage=='Ampulla of Vater'))==np.sum(~mask&(lineage=='Ampulla of Vater'))==1
        inner=s['inner'][r];protected.append(s['ids'][inner==-2].tolist());assert np.all(lineage[inner==-2]=='Ampulla of Vater') and np.sum(inner==-2)==1
        for k in range(5):
            a,b=inner_rows(inner,k);assert set(a)|set(b)==set(tr) and set(groups[a]).isdisjoint(groups[b]) and set(a).isdisjoint(te) and set(b).isdisjoint(te)
            assert set(lineage[a])==labels and set(lineage[b])==labels-{'Ampulla of Vater'};inner_checks+=1
        for a in [tr]+[inner_rows(inner,k)[0] for k in range(5)]:
            for arm in run.ARMS:
                w=weights_for(lineage[a],arm);assert np.isclose(w.mean(),1,atol=1e-12,rtol=0)
                if arm=='capped_inverse_5':assert w.max()<=5
                weight_checks+=1
    rng=np.random.default_rng(2026110602);x=rng.normal(size=(37,4));y=rng.normal(size=37);params=dict(hidden_layer_sizes=(4,),activation='tanh',solver='lbfgs',alpha=100,max_iter=2000,max_fun=40000,tol=.001,random_state=309)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');a=MLPRegressor(**params).fit(x,y);b=WeightedMLPRegressor(**params).fit(x,y,sample_weight=np.ones(len(y)))
    error=float(np.max(np.abs(a.predict(x)-b.predict(x))));assert error==0
    preds={arm:np.full(s['test'].shape,np.nan) for arm in run.ARMS}
    for arm in run.ARMS:
        coef=rng.normal(size=64)
        for r,mask in enumerate(s['test']):preds[arm][r,mask]=s['y'][~mask].mean()+.01*((s['x'][mask]-s['x'][~mask].mean(axis=0))@coef)
    boot=validate_bootstrap(s['y'],preds,lineage,s['test'],seed=2026110603)
    p=np.load(run.OUT/'inputs.npz');coverage=s['test'].sum(axis=0)
    v=dict(passed=True,protocol_sha256=ph,outer_splits_checked=outer_checks,inner_splits_checked=inner_checks,weight_vectors_checked=weight_checks,
        all_23_lineages_in_both_outer_sets=True,all_23_lineages_in_inner_training=True,inner_validation_lineages=22,protected_inner_training_ids_by_repeat=protected,
        identity_group_overlap=0,excluded_cervix_sample_ids=old['sample_ids'][~keep].tolist(),unit_weight_nn_prediction_error=error,
        training_n=[int((~v).sum()) for v in s['test']],testing_n=[int(v.sum()) for v in s['test']],
        test_sample_coverage=dict(ever_tested=int(np.sum(coverage>0)),never_tested=int(np.sum(coverage==0)),minimum_test_repeats=int(coverage.min()),maximum_test_repeats=int(coverage.max())),
        bootstrap_validation=boot,split_generation_has_no_response_argument=True,prior_fit_and_weight_implementations_unchanged=True)
    run.dump(run.OUT/'preflight_validation.json',v);print(json.dumps(v,indent=2),flush=True)
if __name__=='__main__':main()
