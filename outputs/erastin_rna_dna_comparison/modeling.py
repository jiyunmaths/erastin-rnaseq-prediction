"""Training-only mixed-omics preprocessing and the retained model families."""
import warnings
import time
import numpy as np
from scipy.linalg import svd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import ElasticNet,enet_path
from weighted_mlp import WeightedMLPRegressor
import enet_reference as enet

ARMS=['rna','rna_mutation','rna_cnv','rna_mutation_cnv']
ENET_MAX_ITER=500000
ENET_PATH_LIMITS=[500000,2000000,5000000]


def scales(arm):
    mm=[1.,2.] if 'mutation' in arm else [0.]
    cc=[.25,1.] if 'cnv' in arm else [0.]
    return [dict(mutation_scale=m,cnv_scale=c) for m in mm for c in cc]


class OmicsPreprocessor:
    def __init__(self,arm):self.arm=arm

    def fit(self,rna,mutation,cnv):
        self.rna_scaler=StandardScaler().fit(rna)
        self.mutation_keep=np.array([],int);self.cnv_keep=np.array([],int)
        if 'mutation' in self.arm:
            assert np.isfinite(mutation).all() and set(np.unique(mutation))<=set([0.,1.])
            count=mutation.sum(axis=0)
            self.mutation_keep=np.flatnonzero((count>=5)&((len(mutation)-count)>=5))
            self.mutation_mean=mutation[:,self.mutation_keep].mean(axis=0)
        if 'cnv' in self.arm:
            valid=np.isfinite(cnv).sum(axis=0)>=.8*len(cnv)
            candidates=np.flatnonzero(valid)
            med=np.nanmedian(cnv[:,candidates],axis=0)
            filled=np.where(np.isfinite(cnv[:,candidates]),cnv[:,candidates],med)
            variable=np.std(filled,axis=0)>1e-12
            self.cnv_keep=candidates[variable];self.cnv_median=med[variable]
            self.cnv_scaler=StandardScaler().fit(filled[:,variable]) if variable.any() else None
        self.training_n=len(rna)
        return self

    def transform(self,rna,mutation,cnv,mutation_scale=1.,cnv_scale=1.):
        blocks=[self.rna_scaler.transform(rna)]
        if 'mutation' in self.arm:
            # Center binary calls but avoid division by rare-variant SD.
            blocks.append((mutation[:,self.mutation_keep]-self.mutation_mean)*mutation_scale)
        if 'cnv' in self.arm and len(self.cnv_keep):
            c=cnv[:,self.cnv_keep];filled=np.where(np.isfinite(c),c,self.cnv_median)
            blocks.append(self.cnv_scaler.transform(filled)*cnv_scale)
        z=np.column_stack(blocks);assert np.isfinite(z).all()
        return z

    def names(self,s):
        return np.r_[s['genes'],s['mutation_features'][self.mutation_keep],s['cnv_features'][self.cnv_keep]]


def grid(family,arm,source):
    base=([dict(alpha=float(a)) for a in source['ridge_alphas']] if family=='ridge' else enet.GRID if family=='elastic_net' else source['neural_grid'])
    return [dict(**scale,**p) for scale in scales(arm) for p in base]


def check_warnings(caught):
    rows=[dict(category=v.category.__name__,message=str(v.message)) for v in caught]
    assert all(v['category']=='DeprecationWarning' and 'disp' in v['message'] for v in rows),rows
    return rows


def ridge_path(z,y,zz,alphas):
    mean=float(y.mean());u,s,vt=svd(z,full_matrices=False,check_finite=False)
    factors=(s[:,None]/(s[:,None]**2+np.array(alphas)[None,:]))*(u.T@(y-mean))[:,None]
    return ((zz@vt.T)@factors+mean).T,vt.T@factors,mean


def elastic_path(z,y,zz):
    sy=StandardScaler().fit(y[:,None]);target=np.ascontiguousarray(sy.transform(y[:,None]).ravel())
    z=np.asfortranarray(z);gram=np.ascontiguousarray(z.T@z);xy=np.ascontiguousarray(z.T@target)
    predictions=[];diagnostics=[]
    for ratio in enet.L1_RATIOS:
        for limit in ENET_PATH_LIMITS:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                alpha,coef,gaps,it=enet_path(z,target,l1_ratio=ratio,alphas=enet.ALPHAS,precompute=gram,Xy=xy,
                    copy_X=False,return_n_iter=True,max_iter=limit,tol=enet.TOL,selection='cyclic')
            if not any(v.category.__name__=='ConvergenceWarning' for v in caught):break
        check_warnings(caught)
        gradient=(gram@coef-xy[:,None])/len(y)+alpha[None,:]*(1-ratio)*coef
        residual=np.where(coef!=0,np.abs(gradient+alpha[None,:]*ratio*np.sign(coef)),
            np.maximum(np.abs(gradient)-alpha[None,:]*ratio,0)).max(axis=0)
        assert residual.max()<1e-6
        predictions.append((zz@coef).T*sy.scale_[0]+sy.mean_[0])
        diagnostics.append(dict(l1_ratio=ratio,max_kkt_residual=float(residual.max()),max_dual_gap=float(np.max(gaps)),max_iterations=int(np.max(it)),iteration_limit=limit))
    return np.concatenate(predictions),diagnostics


def neural(params,z,y,zz,seeds,retain):
    sy=StandardScaler().fit(y[:,None]);target=sy.transform(y[:,None]).ravel();pred=[];members=[];diagnostics=[]
    for seed in seeds:
        m=WeightedMLPRegressor(hidden_layer_sizes=(params['width'],),activation='tanh',solver='lbfgs',alpha=params['alpha'],
            max_iter=2000,max_fun=40000,tol=.001,early_stopping=False,random_state=int(seed))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');m.fit(z,target,sample_weight=np.ones(len(y)))
        ww=check_warnings(caught);pred.append(sy.inverse_transform(m.predict(zz)[:,None]).ravel())
        diagnostics.append(dict(seed=int(seed),iterations=int(m.n_iter_),training_loss=float(m.loss_),warnings=ww))
        if retain:members.append(m)
    return np.mean(pred,axis=0),dict(y_scaler=sy,members=members),diagnostics,np.array(pred)


def candidates(family,arm,s,tr,va):
    pp=OmicsPreprocessor(arm).fit(s['rna'][tr],s['mutation'][tr],s['cnv'][tr]);pred=[];diagnostics=[]
    for scale in scales(arm):
        z=pp.transform(s['rna'][tr],s['mutation'][tr],s['cnv'][tr],**scale)
        zz=pp.transform(s['rna'][va],s['mutation'][va],s['cnv'][va],**scale)
        if family=='ridge':
            p,_,_=ridge_path(z,s['y'][tr],zz,s['source']['ridge_alphas']);dd=[]
        elif family=='elastic_net':p,dd=elastic_path(z,s['y'][tr],zz)
        else:
            p=[];dd=[]
            for params in s['source']['neural_grid']:
                values,_,d,_=neural(params,z,s['y'][tr],zz,s['current_seeds'],False);p.append(values);dd.append(dict(parameters=params,members=d))
            p=np.array(p)
        pred.append(p);diagnostics.append(dict(scales=scale,fit=dd))
    values=np.concatenate(pred)
    assert len(values)==len(grid(family,arm,s['source'])) and np.isfinite(values).all()
    return values,dict(training_n=len(tr),mutation_keep=pp.mutation_keep,cnv_keep=pp.cnv_keep,
        feature_names=pp.names(s),rna_mean=pp.rna_scaler.mean_,rna_scale=pp.rna_scaler.scale_,diagnostics=diagnostics)


def fit_selected(family,arm,s,tr,te,params):
    start=time.monotonic();pp=OmicsPreprocessor(arm).fit(s['rna'][tr],s['mutation'][tr],s['cnv'][tr])
    scale={k:params[k] for k in ['mutation_scale','cnv_scale']}
    z=pp.transform(s['rna'][tr],s['mutation'][tr],s['cnv'][tr],**scale)
    zz=pp.transform(s['rna'][te],s['mutation'][te],s['cnv'][te],**scale)
    if family=='ridge':
        pred,coef,mean=ridge_path(z,s['y'][tr],zz,[params['alpha']]);pred=pred[0]
        fitted=dict(coef=coef[:,0],intercept=mean);diag={}
    elif family=='elastic_net':
        sy=StandardScaler().fit(s['y'][tr,None]);target=sy.transform(s['y'][tr,None]).ravel()
        m=ElasticNet(alpha=params['alpha'],l1_ratio=params['l1_ratio'],fit_intercept=True,
            max_iter=ENET_MAX_ITER,tol=enet.TOL,selection='cyclic',precompute=False,warm_start=True)
        total_iterations=0
        for continuation in range(10):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always');m.fit(z,target)
            total_iterations+=int(m.n_iter_)
            if not any(v.category.__name__=='ConvergenceWarning' for v in caught):break
        ww=check_warnings(caught);error=enet.kkt_residual(z,target,m.coef_,m.intercept_,params['alpha'],params['l1_ratio']);assert error<1e-6
        pred=sy.inverse_transform(m.predict(zz)[:,None]).ravel();fitted=dict(y_scaler=sy,estimator=m)
        diag=dict(kkt_residual=error,dual_gap=float(m.dual_gap_),iterations=total_iterations,continuations=continuation,nonzero_features=int(np.count_nonzero(m.coef_)),warnings=ww)
    else:
        pred,fitted,members,mp=neural(params,z,s['y'][tr],zz,s['current_seeds'],True)
        diag=dict(members=members)
    assert np.isfinite(pred).all()
    model=dict(family=family,arm=arm,parameters=params,preprocessor=pp,feature_names=pp.names(s),**fitted)
    diag.update(seconds=time.monotonic()-start,features=len(model['feature_names']),mutations=len(pp.mutation_keep),copy_numbers=len(pp.cnv_keep))
    return pred,model,diag


def predict(model,rna,mutation,cnv):
    p=model['parameters'];z=model['preprocessor'].transform(rna,mutation,cnv,p['mutation_scale'],p['cnv_scale'])
    if model['family']=='ridge':return z@model['coef']+model['intercept']
    if model['family']=='elastic_net':return model['y_scaler'].inverse_transform(model['estimator'].predict(z)[:,None]).ravel()
    return np.mean([model['y_scaler'].inverse_transform(m.predict(z)[:,None]).ravel() for m in model['members']],axis=0)
