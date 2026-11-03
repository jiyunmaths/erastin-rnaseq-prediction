"""Training-only standardized Elastic Net paths and independent objective checks."""
import warnings
import numpy as np
from sklearn.linear_model import ElasticNet,enet_path
from sklearn.preprocessing import StandardScaler

L1_RATIOS=[.01,.05,.1,.25,.5,.75,.9,1.]
ALPHAS=np.logspace(1,-4,41)
GRID=[dict(alpha=float(alpha),l1_ratio=float(ratio)) for ratio in L1_RATIOS for alpha in ALPHAS]
MAX_ITER=50000
TOL=1e-9

def kkt_residual(z,target,coef,intercept,alpha,l1_ratio):
    residual=z@coef+intercept-target;grad=z.T@residual/len(target)+alpha*(1-l1_ratio)*coef
    active=coef!=0;err=np.where(active,np.abs(grad+alpha*l1_ratio*np.sign(coef)),np.maximum(np.abs(grad)-alpha*l1_ratio,0))
    return max(float(np.max(err)),float(abs(residual.mean())))

def fit_path(x,y,a,b):
    sx=StandardScaler().fit(x[a]);sy=StandardScaler().fit(y[a,None]);z=np.asfortranarray(sx.transform(x[a]));zz=sx.transform(x[b]);target=np.ascontiguousarray(sy.transform(y[a,None]).ravel())
    gram=np.ascontiguousarray(z.T@z);xy=np.ascontiguousarray(z.T@target);pred=np.empty((len(GRID),len(b)));coefs=np.empty((len(GRID),x.shape[1]));diagnostics=[]
    for j,ratio in enumerate(L1_RATIOS):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            alpha,coef,gaps,iterations=enet_path(z,target,l1_ratio=ratio,alphas=ALPHAS,precompute=gram,Xy=xy,copy_X=False,return_n_iter=True,max_iter=MAX_ITER,tol=TOL,selection='cyclic')
        np.testing.assert_allclose(alpha,ALPHAS,rtol=0,atol=0)
        start=j*len(ALPHAS);stop=start+len(ALPHAS);pred[start:stop]=(zz@coef).T*sy.scale_[0]+sy.mean_[0];coefs[start:stop]=coef.T
        grad=(gram@coef-xy[:,None])/len(a)+ALPHAS[None,:]*(1-ratio)*coef
        err=np.where(coef!=0,np.abs(grad+ALPHAS[None,:]*ratio*np.sign(coef)),np.maximum(np.abs(grad)-ALPHAS[None,:]*ratio,0)).max(axis=0)
        assert np.max(err)<1e-6
        diagnostics.append(dict(l1_ratio=ratio,max_kkt_residual=float(np.max(err)),iterations=[int(v) for v in iterations],dual_gaps=gaps,max_dual_gap=float(np.max(gaps)),kkt_residuals=err,
            warnings=[dict(category=v.category.__name__,message=str(v.message)) for v in caught]))
    assert np.isfinite(pred).all() and np.isfinite(coefs).all()
    return pred,coefs,diagnostics,dict(y_mean=float(sy.mean_[0]),y_scale=float(sy.scale_[0]))

def fit_selected(x,y,a,b,params):
    sx=StandardScaler().fit(x[a]);sy=StandardScaler().fit(y[a,None]);z=sx.transform(x[a]);target=sy.transform(y[a,None]).ravel()
    model=ElasticNet(alpha=params['alpha'],l1_ratio=params['l1_ratio'],fit_intercept=True,precompute=False,max_iter=MAX_ITER,tol=TOL,selection='cyclic',warm_start=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always');model.fit(z,target)
    error=kkt_residual(z,target,model.coef_,model.intercept_,model.alpha,model.l1_ratio);assert error<1e-6
    pred=sy.inverse_transform(model.predict(sx.transform(x[b]))[:,None]).ravel()
    objective=.5*np.mean((model.predict(z)-target)**2)+model.alpha*model.l1_ratio*np.abs(model.coef_).sum()+.5*model.alpha*(1-model.l1_ratio)*np.sum(model.coef_**2)
    saved=dict(family='elastic_net',x_scaler=sx,y_scaler=sy,model=model)
    diagnostics=dict(kkt_residual=error,dual_gap=float(model.dual_gap_),iterations=int(model.n_iter_),nonzero_genes=int(np.count_nonzero(model.coef_)),objective=float(objective),warnings=[dict(category=v.category.__name__,message=str(v.message)) for v in caught])
    return pred,saved,diagnostics
