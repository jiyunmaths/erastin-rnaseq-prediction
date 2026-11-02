"""Additive nonlinear Gaussian CPDs on a training-only TAN structure.

Y is a statistical root; this is not a causal interpretation. Every gene has
Y and at most one gene parent. Means are additive cubic splines of parents.
All genes are observed at prediction time, so p(Y|X) is a one-dimensional
distribution. Integrate spline intervals numerically and linear tails exactly.
"""
import importlib.util
from pathlib import Path

import numpy as np
from scipy.special import log_ndtr, logsumexp, roots_legendre
from sklearn.preprocessing import SplineTransformer, StandardScaler

ROOT=Path(__file__).resolve().parent.parent/'erastin_bayesian_comparison/gaussian_network.py'
spec=importlib.util.spec_from_file_location('linear_bn_source',ROOT)
linear=importlib.util.module_from_spec(spec);spec.loader.exec_module(linear)
ORDERS=(16,32,64,128,256,512)
MEAN_TOL=1e-8
VAR_TOL=1e-7
LOGZ_TOL=1e-8


def fit_basis(values):
    knots=np.unique(np.quantile(values,[0,1/3,2/3,1]))
    assert len(knots)>=2
    spline=SplineTransformer(degree=3,knots=knots[:,None],include_bias=False,extrapolation='linear').fit(values[:,None])
    raw=spline.transform(values[:,None]);scaler=StandardScaler().fit(raw)
    return {'spline':spline,'scaler':scaler,'knots':knots}


def basis_transform(basis,values):
    return basis['scaler'].transform(basis['spline'].transform(np.asarray(values).reshape(-1,1)))


def prepare(x,y):
    scaler=StandardScaler().fit(np.column_stack([y,x]))
    z=scaler.transform(np.column_stack([y,x]));n,p=z.shape;p-=1
    assert np.min(z.var(axis=0))>1e-10
    graph=linear.from_covariance(z.T@z/n,'gaussian_tan')
    parent=graph['gene_parent']
    bases=[fit_basis(z[:,j]) for j in range(p+1)]
    spline_z=[basis_transform(bases[j],z[:,j]) for j in range(p+1)]
    cache={}
    for family in ['linear_cpd','spline_cpd']:
        yy=z[:,[0]] if family=='linear_cpd' else spline_z[0]
        designs=[]
        for j in range(p):
            blocks=[yy]
            if parent[j]>=0:
                blocks.append(z[:,[parent[j]+1]] if family=='linear_cpd' else spline_z[parent[j]+1])
            design=np.column_stack(blocks);center=design.mean(axis=0);target=z[:,j+1];offset=target.mean()
            a=design-center
            u,s,vt=np.linalg.svd(a,full_matrices=False)
            designs.append({'a':a,'center':center,'target':target,'offset':offset,'s':s,'vt':vt,'uty':u.T@(target-offset)})
        cache[family]={'designs':designs,'response_features':yy.shape[1]}
    return {'mean':scaler.mean_,'scale':scaler.scale_,'bases':bases,'parent':parent,'edges':graph['gene_edges'],
            'conditional_mutual_information':graph['conditional_mutual_information'],'training_n':n,'cache':cache}


def fit_prepared(prepared,family,alpha):
    state=prepared['cache'][family];q=state['response_features'];p=len(prepared['parent'])
    response=np.empty((p,q));parents=[];offset=np.empty(p);variance=np.empty(p);node_coefficients=[]
    for j,d in enumerate(state['designs']):
        s=d['s']
        if alpha==0:
            threshold=np.finfo(float).eps*max(d['a'].shape)*s.max()
            factors=np.divide(1.,s,out=np.zeros_like(s),where=s>threshold)
        else:factors=s/(s*s+alpha)
        coef=d['vt'].T@(factors*d['uty']);intercept=d['offset']-d['center']@coef
        residual=d['target']-(d['a']@coef+d['offset'])
        response[j]=coef[:q];parents.append(coef[q:]);offset[j]=intercept
        variance[j]=max(float(np.mean(residual*residual)),1e-6)
        node_coefficients.append(coef)
    model={k:v for k,v in prepared.items() if k!='cache'}
    model.update(family=family,alpha=float(alpha),response_coef=response,parent_coef=parents,
                 intercept=offset,noise_variance=variance,node_coefficients=node_coefficients)
    return model


def observed_residual(model,x):
    z=(x-model['mean'][1:])/model['scale'][1:]
    residual=z-model['intercept']
    transformed={}
    for j,parent in enumerate(model['parent']):
        if parent>=0:
            if parent not in transformed:
                transformed[parent]=z[:,[parent]] if model['family']=='linear_cpd' else basis_transform(model['bases'][parent+1],z[:,parent])
            residual[:,j]-=transformed[parent]@model['parent_coef'][j]
    return residual


def information(model,x):
    residual=observed_residual(model,x);b=model['response_coef'];v=model['noise_variance']
    return (residual/v)@b,b.T@(b/v[:,None])


def tail_terms(model,h,g,side):
    basis=model['bases'][0];boundary=basis['knots'][0 if side=='left' else -1]
    if side=='left':
        f=basis_transform(basis,[boundary-1,boundary]);slope=f[1]-f[0];value=f[1]
    else:
        f=basis_transform(basis,[boundary,boundary+1]);slope=f[1]-f[0];value=f[0]
    intercept=value-slope*boundary
    precision=1+slope@g@slope
    linear_term=h@slope-intercept@g@slope
    constant=h@intercept-.5*intercept@g@intercept
    mean=linear_term/precision;sd=1/np.sqrt(precision)
    standard=(boundary-mean)/sd
    logcdf=log_ndtr(standard if side=='left' else -standard)
    logpdf=-.5*standard**2-.5*np.log(2*np.pi)
    ratio=np.exp(logpdf-logcdf)
    sign=-1 if side=='left' else 1
    first=mean+sign*sd*ratio
    second=mean**2+sd**2+sign*sd*(mean+boundary)*ratio
    mass=constant+linear_term**2/(2*precision)+.5*np.log(2*np.pi/precision)+logcdf
    return mass,first,second


def integrate_order(model,h,g,order):
    knots=model['bases'][0]['knots'];nodes,weights=roots_legendre(order)
    grid=np.concatenate([(a+b)/2+(b-a)/2*nodes for a,b in zip(knots[:-1],knots[1:])])
    w=np.concatenate([(b-a)/2*weights for a,b in zip(knots[:-1],knots[1:])])
    phi=basis_transform(model['bases'][0],grid)
    base=-.5*grid**2-.5*np.einsum('ij,jk,ik->i',phi,g,phi)
    logmass=h@phi.T+base+np.log(w)
    left=tail_terms(model,h,g,'left');right=tail_terms(model,h,g,'right')
    combined=np.column_stack([left[0],logmass,right[0]])
    logz=logsumexp(combined,axis=1)
    probability=np.exp(combined-logz[:,None])
    mean=probability[:,0]*left[1]+probability[:,1:-1]@grid+probability[:,-1]*right[1]
    second=probability[:,0]*left[2]+probability[:,1:-1]@(grid*grid)+probability[:,-1]*right[2]
    variance=second-mean*mean
    assert np.min(variance)>-1e-8 and np.isfinite(mean).all() and np.isfinite(logz).all()
    return mean,np.maximum(variance,1e-12),logz


def predict(model,x,minimum_order=16):
    h,g=information(model,x)
    if model['family']=='linear_cpd':
        precision=1+g[0,0];mean=h[:,0]/precision;variance=np.full(len(x),1/precision)
        logz=.5*np.log(2*np.pi/precision)+h[:,0]**2/(2*precision)
        diagnostic={'order':0,'mean_error':0.,'variance_error':0.,'logz_error':0.,'converged':True}
    else:
        previous=None;converged=False
        for order in ORDERS:
            if order<minimum_order:continue
            current=integrate_order(model,h,g,order)
            if previous is not None:
                differences=[float(np.max(np.abs(a-b))) for a,b in zip(current,previous)]
                if differences[0]<MEAN_TOL and differences[1]<VAR_TOL and differences[2]<LOGZ_TOL:
                    converged=True;break
            previous=current
        if not converged:raise RuntimeError(f'Quadrature did not converge: {differences}')
        mean,variance,logz=current
        diagnostic={'order':int(order),'mean_error':differences[0],'variance_error':differences[1],
                    'logz_error':differences[2],'converged':True}
    return {'mean':model['mean'][0]+model['scale'][0]*mean,
            'variance':model['scale'][0]**2*variance,'logz_standardized':logz,'diagnostic':diagnostic}


def conditional_logpdf(model,x,y,prediction=None):
    if prediction is None:prediction=predict(model,x)
    h,g=information(model,x);z=(y-model['mean'][0])/model['scale'][0]
    phi=z[:,None] if model['family']=='linear_cpd' else basis_transform(model['bases'][0],z)
    score=-.5*z*z+np.sum(h*phi,axis=1)-.5*np.einsum('ij,jk,ik->i',phi,g,phi)
    return score-prediction['logz_standardized']-np.log(model['scale'][0])


def direct_log_joint(model,x,y_standardized):
    """Independent scalar product-of-CPDs evaluation for numerical audits."""
    z=(np.asarray(x)-model['mean'][1:])/model['scale'][1:]
    phi=np.array([[y_standardized]]) if model['family']=='linear_cpd' else basis_transform(model['bases'][0],[y_standardized])
    means=model['intercept']+model['response_coef']@phi[0]
    for j,parent in enumerate(model['parent']):
        if parent>=0:
            xp=np.array([z[parent]]) if model['family']=='linear_cpd' else basis_transform(model['bases'][parent+1],[z[parent]])[0]
            means[j]+=xp@model['parent_coef'][j]
    residual=z-means;v=model['noise_variance']
    return float(-.5*y_standardized**2-.5*np.log(2*np.pi)-.5*np.sum(np.log(2*np.pi*v)+residual*residual/v))


def verify_math():
    rng=np.random.default_rng(20261110);y=rng.normal(size=400)
    x=np.column_stack([y+.35*y**3+rng.normal(scale=.5,size=400),-.6*y+rng.normal(size=400),rng.normal(size=400)])
    prep=prepare(x,y);linear_model=fit_prepared(prep,'linear_cpd',0.)
    reference=linear.fit_network(x,y,'gaussian_tan',0.)
    p=predict(linear_model,x[:7]);rp,rv=linear.predict(reference,x[:7])
    le=float(np.max(np.abs(p['mean']-rp)));ve=float(np.max(np.abs(p['variance']-rv)))
    assert le<1e-10 and ve<1e-10
    spline=fit_prepared(prep,'spline_cpd',10.)
    pp=predict(spline,x[:7]);fine=predict(spline,x[:7],minimum_order=128)
    qe=float(np.max(np.abs(pp['mean']-fine['mean'])))
    assert qe<1e-8
    # Conditional log densities equal normalized joint log densities up to
    # an x-only constant: compare ratios at arbitrary response values.
    targets=np.array([-.4,.8]);xx=np.repeat(x[[0]],2,axis=0)
    logp=conditional_logpdf(spline,xx,targets)
    logjoint=np.array([direct_log_joint(spline,x[0],(t-spline['mean'][0])/spline['scale'][0]) for t in targets])
    density_error=float(abs(np.diff(logp)[0]-np.diff(logjoint)[0]));assert density_error<1e-10
    # A synthetic nonlinear data-generating relationship gives a nonlinear
    # prediction curve, verifying this is not a renamed linear regressor.
    points=np.zeros((3,3));points[:,0]=[0,1,2]
    curved=predict(spline,points)['mean'];curvature=float(abs(curved[0]-2*curved[1]+curved[2]))
    assert curvature>1e-3
    return {'passed':True,'linear_limit_prediction_error':le,'linear_limit_variance_error':ve,
            'quadrature_refinement_prediction_error':qe,'joint_to_conditional_density_ratio_error':density_error,
            'synthetic_prediction_second_difference':curvature}
