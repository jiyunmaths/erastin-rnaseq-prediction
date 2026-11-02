#!/usr/bin/env python3
"""Independent formula, capped-normalization, gradient and prior replay checks."""
import json
from pathlib import Path
import warnings
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network._multilayer_perceptron import _pack
import run_moderate as run
from weighted_mlp import WeightedMLPRegressor
from lineage_weights import SCHEMES,weights_for,weight_audit,CAP

def main():
    ph=run.prepare();run.initialize();s=run.STATE;rng=np.random.default_rng(2026110503);checks=[]
    labels=np.array(['a']*30+['b']*5+['c']*2+['d']);counts=np.array([30,5,2,1])
    np.testing.assert_allclose(weights_for(labels,'capped_inverse_5'),np.repeat([11.5/30,11.5/5,5,5],counts),rtol=0,atol=1e-14)
    for scheme in SCHEMES:
        w=weights_for(labels,scheme);permutation=rng.permutation(len(labels))
        np.testing.assert_array_equal(weights_for(labels[permutation],scheme),w[permutation])
        np.testing.assert_array_equal(weights_for(np.array(['A','B','C','D'])[np.searchsorted(np.unique(labels),labels)],scheme),w)
        np.testing.assert_array_equal(weights_for(np.repeat(['a','b','c'],7),scheme),np.ones(21))
        if scheme=='inverse_sqrt':
            totals=np.array([w[labels==l].sum() for l in np.unique(labels)])
            np.testing.assert_allclose(totals/np.sqrt(counts),np.repeat((totals/np.sqrt(counts))[0],4),rtol=0,atol=1e-13)
        x=rng.normal(size=(len(labels),4));y=rng.normal(size=len(labels))
        kwargs=dict(hidden_layer_sizes=(4,),activation='tanh',solver='lbfgs',alpha=10,max_iter=2000,max_fun=40000,tol=.001,random_state=309)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore');m=WeightedMLPRegressor(**kwargs).fit(x,y,sample_weight=w)
            a=MLPRegressor(**kwargs).fit(x,y);b=WeightedMLPRegressor(**kwargs).fit(x,y,sample_weight=np.ones(len(y)))
        unit_error=float(np.max(np.abs(a.predict(x)-b.predict(x))));assert unit_error==0
        pars=_pack(m.coefs_,m.intercepts_)+rng.normal(size=sum(v.size for v in m.coefs_+m.intercepts_))*.05
        def fg(z):return m._loss_grad_lbfgs(z,x,y[:,None],[x,None,None],[None,None],[np.empty_like(v) for v in m.coefs_],[np.empty_like(v) for v in m.intercepts_])
        loss,g=fg(pars);numeric=approx_derivative(lambda z:np.array([fg(z)[0]]),pars,method='3-point').ravel()
        error=float(np.max(np.abs(g-numeric)));assert error<1e-7;fg(pars)
        expected=.5*np.average((m.predict(x)-y)**2,weights=w)+.5*m.alpha*sum(np.sum(v*v) for v in m.coefs_)/len(y)
        objective_error=float(abs(loss-expected));assert objective_error<1e-12
        checks.append(dict(scheme=scheme,gradient_max_error=error,direct_objective_error=objective_error,unit_weight_error=unit_error,
            toy_weight_audit=weight_audit(labels,scheme)))
    # Every planned fitting membership gets its own independently checked weight vector.
    membership_count=0;ess={v:[] for v in SCHEMES};maxima={v:[] for v in SCHEMES};minima={v:[] for v in SCHEMES}
    for r in range(20):
        for l in range(24):
            inner=s['inner'][r,l];tr=np.flatnonzero(inner>=0)
            for a in [tr]+[tr[inner[tr]!=k] for k in range(5)]:
                ll=s['lineage'][a];u,inv,c=np.unique(ll,return_inverse=True,return_counts=True)
                for scheme in SCHEMES:
                    w=weights_for(ll,scheme)
                    if scheme=='inverse_sqrt':
                        ref=1/np.sqrt(c[inv]);ref/=ref.mean();np.testing.assert_allclose(w,ref,rtol=0,atol=1e-13)
                    else:
                        assert w.max()<=CAP
                        free=w<CAP
                        expected_scale=w[free]*c[inv][free];np.testing.assert_allclose(expected_scale,expected_scale[0],rtol=0,atol=1e-11)
                    assert np.isclose(w.mean(),1,atol=1e-12,rtol=0)
                    ess[scheme].append(float(w.sum()**2/np.dot(w,w)));maxima[scheme].append(float(w.max()));minima[scheme].append(float(w.min()));membership_count+=1
    base=np.load(run.WS/'outputs/erastin_panel_comparison/predictions.npz');prior=np.load(run.WS/'outputs/erastin_lineage_weighted/predictions.npz');replays=[]
    for label in ['Lung','Thyroid']:
        l=list(s['labels']).index(label);tr=np.flatnonzero(s['inner'][0,l]>=0);te=np.flatnonzero(s['lineage']==label)
        for scheme,folder,token,expected in [
            ('unweighted',run.WS/'outputs/erastin_panel_comparison/folds',f'neural_network__p64__r01__l{l:02d}',base['neural_network__64']),
            ('inverse_frequency',run.WS/'outputs/erastin_lineage_weighted/folds',f'neural_network__p64__r01__l{l:02d}',prior['weighted__neural_network__64'])]:
            m=json.loads((folder/(token+'.json')).read_text());p,_,_,_=run.fitted_neural(m['selected_parameters'],s['x'],s['y'],tr,te,s['nn_seeds'][0],weights_for(s['lineage'][tr],scheme))
            err=float(np.max(np.abs(p-expected[0,te])));assert err<1e-10;replays.append(dict(scheme=scheme,lineage=label,max_prediction_error=err))
    result=dict(passed=True,protocol_sha256=ph,checks=checks,actual_data_replays=replays,fitting_membership_weight_checks=membership_count,
        planned_weight_summary={v:dict(effective_sample_size_range=[min(ess[v]),max(ess[v])],weight_range=[min(minima[v]),max(maxima[v])]) for v in SCHEMES})
    run.dump(run.OUT/'preflight_validation.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
