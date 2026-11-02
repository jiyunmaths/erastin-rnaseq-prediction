"""Convex blending of three retained model families using inner predictions."""
from itertools import combinations

import numpy as np
from scipy.optimize import minimize,minimize_scalar

FAMILIES=['ridge','elastic_net','neural_network']
FTOL=1e-12
KKT_LIMIT=2e-6


def value_gradient(weights,prediction,y,fold):
    errors=prediction@weights-y;value=0.;gradient=np.zeros(3)
    labels=np.unique(fold)
    for label in labels:
        rows=fold==label;residual=errors[rows];rmse=np.sqrt(np.mean(residual**2))
        assert rmse>1e-14
        value+=rmse/len(labels)
        gradient+=prediction[rows].T@residual/(rows.sum()*rmse*len(labels))
    return float(value),gradient


def kkt(weights,gradient):
    multiplier=float(weights@gradient)
    active=weights>1e-8
    stationarity=float(np.max(np.abs(gradient[active]-multiplier)))
    inactive=float(np.max(np.maximum(multiplier-gradient[~active],0))) if (~active).any() else 0.
    return max(stationarity,inactive,abs(weights.sum()-1),max(0.,-float(weights.min())))


def fit_weights(prediction,y,fold):
    """Arguments contain only current outer-training inner-validation rows."""
    prediction=np.asarray(prediction,float);y=np.asarray(y,float);fold=np.asarray(fold)
    assert prediction.shape==(len(y),3) and np.isfinite(prediction).all() and np.isfinite(y).all()
    assert set(np.unique(fold))==set(range(5))
    center=float(y.mean());scale=float(y.std());assert scale>0
    p=(prediction-center)/scale;target=(y-center)/scale
    def fun(w):return value_gradient(w,p,target,fold)[0]
    def jac(w):return value_gradient(w,p,target,fold)[1]
    candidates=[np.ones(3)/3,*np.eye(3)]
    diagnostics=[]
    for i,j in combinations(range(3),2):
        def weights(t):
            w=np.zeros(3);w[i]=t;w[j]=1-t;return w
        result=minimize_scalar(lambda t:fun(weights(t)),bounds=(0.,1.),method='bounded',options=dict(xatol=1e-12,maxiter=1000))
        assert result.success
        candidates.append(weights(result.x))
        diagnostics.append(dict(kind='edge',families=[i,j],success=bool(result.success),iterations=int(result.nit)))
    for start in [np.ones(3)/3,np.eye(3)[np.argmin([fun(w) for w in np.eye(3)])]]:
        result=minimize(fun,start,jac=jac,method='SLSQP',bounds=[(0.,1.)]*3,
            constraints=[dict(type='eq',fun=lambda w:w.sum()-1,jac=lambda w:np.ones(3))],options=dict(ftol=FTOL,maxiter=1000))
        w=np.clip(result.x,0.,1.);w[np.abs(w)<1e-10]=0.;w/=w.sum()
        candidates.append(w)
        diagnostics.append(dict(kind='interior',success=bool(result.success),status=int(result.status),message=str(result.message),iterations=int(result.nit)))
    losses=np.array([fun(w) for w in candidates]);best=int(np.argmin(losses));weights=candidates[best]
    value,gradient=value_gradient(weights,p,target,fold);error=kkt(weights,gradient)
    assert error<KKT_LIMIT,dict(weights=weights,gradient=gradient,error=error)
    assert value<=min(losses[:4])+1e-11
    return weights,dict(family_order=FAMILIES,mean_inner_rmse=value*scale,standardized_kkt_residual=error,
        weight_sum=float(weights.sum()),minimum_weight=float(weights.min()),target_center=center,target_scale=scale,
        standardized_gradient=gradient,selected_solver_candidate=best,solver_diagnostics=diagnostics)
