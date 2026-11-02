"""Regularized fits on already preprocessed, centered mixed feature matrices."""
import warnings
import numpy as np
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet,enet_path
from sklearn.neural_network import MLPRegressor
L1_RATIOS=[.01,.05,.1,.25,.5,.75,.9,1.];ALPHAS=np.logspace(1,-4,41);GRID=[dict(alpha=float(a),l1_ratio=float(r)) for r in L1_RATIOS for a in ALPHAS];MAX_ITER=50000;TOL=1e-9

def kkt(z,y,coef,intercept,alpha,ratio):
    residual=z@coef+intercept-y;grad=z.T@residual/len(y)+alpha*(1-ratio)*coef;err=np.where(coef!=0,np.abs(grad+alpha*ratio*np.sign(coef)),np.maximum(np.abs(grad)-alpha*ratio,0));return max(float(np.max(err)),float(abs(residual.mean())))

def ridge_path(z,y,zz,alphas,retain=False):
    mu=float(y.mean());u,s,vt=svd(z,full_matrices=False,check_finite=False,lapack_driver='gesdd');factors=(s[:,None]/(s[:,None]**2+np.asarray(alphas)[None,:]))*(u.T@(y-mu))[:,None];pred=(zz@vt.T)@factors+mu;saved=dict(coef=vt.T@factors[:,0],intercept=mu,alpha=float(alphas[0])) if retain else None;return pred.T,saved

def elastic_path(z,y,zz):
    sy=StandardScaler().fit(y[:,None]);target=np.ascontiguousarray(sy.transform(y[:,None]).ravel());z=np.asfortranarray(z);gram=np.ascontiguousarray(z.T@z);xy=np.ascontiguousarray(z.T@target);pred=np.empty((len(GRID),len(zz)));diagnostics=[]
    for j,ratio in enumerate(L1_RATIOS):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');alpha,coef,gaps,it=enet_path(z,target,l1_ratio=ratio,alphas=ALPHAS,precompute=gram,Xy=xy,copy_X=False,return_n_iter=True,max_iter=MAX_ITER,tol=TOL,selection='cyclic')
        np.testing.assert_array_equal(alpha,ALPHAS);pred[j*41:(j+1)*41]=(zz@coef).T*sy.scale_[0]+sy.mean_[0];grad=(gram@coef-xy[:,None])/len(y)+ALPHAS[None,:]*(1-ratio)*coef;err=np.where(coef!=0,np.abs(grad+ALPHAS[None,:]*ratio*np.sign(coef)),np.maximum(np.abs(grad)-ALPHAS[None,:]*ratio,0)).max(axis=0);assert np.max(err)<1e-6
        diagnostics.append(dict(l1_ratio=ratio,max_kkt_residual=float(np.max(err)),kkt_residuals=err,dual_gaps=gaps,iterations=it,warnings=[dict(category=v.category.__name__,message=str(v.message)) for v in caught]))
    return pred,diagnostics

def fit_neural(z,y,zz,params,seeds,retain=False):
    sy=StandardScaler().fit(y[:,None]);target=sy.transform(y[:,None]).ravel();pred=[];members=[];diagnostics=[]
    for seed in seeds:
        m=MLPRegressor(hidden_layer_sizes=(params['width'],),activation='tanh',solver='lbfgs',alpha=params['alpha'],max_iter=2000,max_fun=40000,tol=.001,early_stopping=False,random_state=int(seed))
        with warnings.catch_warnings(record=True) as caught:warnings.simplefilter('always');m.fit(z,target)
        pred.append(sy.inverse_transform(m.predict(zz)[:,None]).ravel());members.append(m);diagnostics.append(dict(seed=int(seed),iterations=int(m.n_iter_),training_loss=float(m.loss_),warnings=[dict(category=v.category.__name__,message=str(v.message)) for v in caught]))
    return np.mean(pred,axis=0),dict(members=members,y_scaler=sy) if retain else None,dict(members=diagnostics),np.array(pred)

def fit_outer(f,z,y,zz,params,seeds):
    if f=='ridge':p,m=ridge_path(z,y,zz,[params['alpha']],True);return p[0],m,dict(),None
    if f=='neural_network':return fit_neural(z,y,zz,params,seeds,True)
    sy=StandardScaler().fit(y[:,None]);target=sy.transform(y[:,None]).ravel();m=ElasticNet(alpha=params['alpha'],l1_ratio=params['l1_ratio'],max_iter=MAX_ITER,tol=TOL,fit_intercept=True,selection='cyclic',precompute=False)
    with warnings.catch_warnings(record=True) as caught:warnings.simplefilter('always');m.fit(z,target)
    error=kkt(z,target,m.coef_,m.intercept_,m.alpha,m.l1_ratio);assert error<1e-6;objective=.5*np.mean((m.predict(z)-target)**2)+m.alpha*m.l1_ratio*np.abs(m.coef_).sum()+.5*m.alpha*(1-m.l1_ratio)*np.sum(m.coef_**2)
    return sy.inverse_transform(m.predict(zz)[:,None]).ravel(),dict(model=m,y_scaler=sy),dict(kkt_residual=error,objective=float(objective),nonzero_features=int(np.count_nonzero(m.coef_)),iterations=int(m.n_iter_),warnings=[dict(category=v.category.__name__,message=str(v.message)) for v in caught]),None
