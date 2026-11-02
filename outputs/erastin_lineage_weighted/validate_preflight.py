#!/usr/bin/env python3
"""Independent weighted objective/gradient, unit-weight, and API agreement checks."""
import json
from pathlib import Path
import warnings
import numpy as np
from scipy.optimize._numdiff import approx_derivative
from sklearn.neural_network import MLPRegressor
from sklearn.neural_network._multilayer_perceptron import _pack
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import run_weighted as run
from weighted_mlp import WeightedMLPRegressor,inverse_lineage_weights

OUT=Path(__file__).resolve().parent

def main():
    ph=run.prepare();run.initialize();s=run.STATE;rng=np.random.default_rng(20261104)
    x=rng.normal(size=(37,5));y=rng.normal(size=37);labels=np.array(['a']*30+['b']*5+['c']*2);w=inverse_lineage_weights(labels)
    kwargs=dict(hidden_layer_sizes=(4,),activation='tanh',solver='lbfgs',alpha=10,max_iter=2000,max_fun=40000,tol=.001,random_state=309)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore');orig=MLPRegressor(**kwargs).fit(x,y);unit=WeightedMLPRegressor(**kwargs).fit(x,y,sample_weight=np.ones(37));weighted=WeightedMLPRegressor(**kwargs).fit(x,y,sample_weight=w)
    unit_error=float(np.max(np.abs(orig.predict(x)-unit.predict(x))));assert unit_error<1e-12
    pars=_pack(weighted.coefs_,weighted.intercepts_);gradient_checks=[];loss_checks=[]
    for i in range(3):
        pars=pars+rng.normal(size=pars.size)*.03
        def value_grad(z):
            return weighted._loss_grad_lbfgs(z,x,y[:,None],[x,None,None],[None,None],
                [np.empty_like(a) for a in weighted.coefs_],[np.empty_like(a) for a in weighted.intercepts_])
        loss,analytic=value_grad(pars)
        numeric=approx_derivative(lambda z:np.array([value_grad(z)[0]]),pars,method='3-point').ravel()
        err=float(np.max(np.abs(numeric-analytic)));assert err<1e-7;gradient_checks.append(err)
        value_grad(pars)
        expected=.5*np.average((weighted.predict(x)-y)**2,weights=w)+.5*weighted.alpha*sum(np.sum(a*a) for a in weighted.coefs_)/len(y)
        le=float(abs(loss-expected));assert le<1e-12;loss_checks.append(le)
    checks=[]
    baseline=np.load(run.WS/'outputs/erastin_panel_comparison/predictions.npz')
    for label in ['Lung','Thyroid']:
        l=list(s['labels']).index(label);inner=s['inner'][0,l];tr=np.flatnonzero(inner>=0);te=np.flatnonzero(s['lineage']==label)
        for panel in [40,64,80]:
            xx=s['x'][panel];genes=s['genes'][panel];ww=inverse_lineage_weights(s['lineage'][tr]);sx=StandardScaler().fit(xx[tr])
            for alpha in [s['alphas'][0],100.,s['alphas'][-1]]:
                pred,_=run.ridge_path(xx,s['y'],tr,te,[alpha])
                model=Ridge(alpha=alpha,solver='svd').fit(sx.transform(xx[tr]),s['y'][tr],sample_weight=ww)
                err=float(np.max(np.abs(pred[:,0]-model.predict(sx.transform(xx[te])))));assert err<1e-10
                checks.append(dict(check='weighted_ridge_api',panel=panel,lineage=label,alpha=float(alpha),max_error=err))
            # Unit-weight adapter on actual inputs, exact prior selected NN settings/seeds.
            priorroot=run.WS/'outputs/erastin_panel_comparison'/('replay' if panel==40 else 'folds')/f'neural_network__p{panel}__r01__l{l:02d}'
            m=json.loads(priorroot.with_suffix('.json').read_text())
            p,_,_,_=run.fitted_neural(m['selected_parameters'],xx,s['y'],tr,te,s['nn_seeds'][0],weights=np.ones(len(tr)))
            err=float(np.max(np.abs(p-baseline[f'neural_network__{panel}'][0,te])));assert err<1e-10
            checks.append(dict(check='unit_weight_actual_nn_replay',panel=panel,lineage=label,max_error=err))
        # Native weighted XGBoost vs independently routed sklearn wrapper.
        xx=s['x'][40];genes=s['genes'][40];mu=s['y'][tr].mean();sd=s['y'][tr].std();yy=(s['y'][tr]-mu)/sd
        params=run.xp.GRID[5];seed=int(s['model_seeds'][0]);pp=run.xp.parameters(params,seed)
        native=xgb.train(pp,run.matrix(xx[tr],genes,yy,ww),num_boost_round=params['num_boost_round'])
        pp.pop('nthread');pp.pop('seed')
        wrapped=xgb.XGBRegressor(**pp,n_jobs=1,random_state=seed,n_estimators=params['num_boost_round']).fit(xx[tr],yy,sample_weight=ww)
        err=float(np.max(np.abs(native.predict(run.matrix(xx[te],genes))-wrapped.predict(xx[te]))));assert err<1e-6
        checks.append(dict(check='weighted_xgb_native_vs_wrapper',lineage=label,max_error=err))
        # Explicit weights of one recover original native tree training.
        unit=xgb.train(run.xp.parameters(params,seed),run.matrix(xx[tr],genes,yy,np.ones(len(tr))),num_boost_round=params['num_boost_round'])
        orig=xgb.train(run.xp.parameters(params,seed),run.matrix(xx[tr],genes,yy),num_boost_round=params['num_boost_round'])
        err=float(np.max(np.abs(unit.predict(run.matrix(xx[te],genes))-orig.predict(run.matrix(xx[te],genes)))));assert err==0
        checks.append(dict(check='unit_weight_xgb',lineage=label,max_error=err))
        a=tr[inner[tr]!=0];b=tr[inner[tr]==0];wa=inverse_lineage_weights(s['lineage'][a]);assert len(wa)==len(a)
        run.weight_audit(s['lineage'][a]);assert set(s['groups'][a]).isdisjoint(s['groups'][b])
    result=dict(passed=True,protocol_sha256=ph,script_sha256=run.sha(Path(__file__)),unit_weight_synthetic_mlp_error=unit_error,
        weighted_mlp_gradient_max_errors=gradient_checks,weighted_mlp_direct_objective_errors=loss_checks,
        synthetic_weight_sums=[float(w[labels==v].sum()) for v in np.unique(labels)],checks=checks)
    run.dump(OUT/'preflight_validation.json',result);print(json.dumps(result,indent=2),flush=True)
if __name__=='__main__':main()
