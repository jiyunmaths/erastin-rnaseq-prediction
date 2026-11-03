"""Unchanged NN fit and serialization helpers from the verified weight comparison."""
import hashlib,json,time,warnings
from pathlib import Path
import numpy as np
from sklearn.preprocessing import StandardScaler
from weighted_mlp import WeightedMLPRegressor

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,np.ndarray):return clean(v.tolist())
    if isinstance(v,np.generic):return v.item()
    return v

def dump(path,value):
    tmp=path.with_suffix(path.suffix+'.tmp');tmp.write_text(json.dumps(clean(value),indent=2,allow_nan=False)+'\n');tmp.replace(path)

def fitted_neural(params,x,y,tr,te,seeds,weights,retain=False):
    begin=time.perf_counter();sx=StandardScaler().fit(x[tr]);sy=StandardScaler().fit(y[tr,None])
    a=sx.transform(x[tr]);b=sx.transform(x[te]);target=sy.transform(y[tr,None]).ravel()
    members=[];predictions=[];diagnostics=[]
    for seed in seeds:
        model=WeightedMLPRegressor(hidden_layer_sizes=(params['width'],),activation='tanh',solver='lbfgs',
            alpha=params['alpha'],max_iter=2000,max_fun=40000,tol=.001,early_stopping=False,random_state=int(seed))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(a,target,sample_weight=weights)
        p=sy.inverse_transform(model.predict(b)[:,None]).ravel();assert np.isfinite(p).all();predictions.append(p)
        diagnostics.append(dict(seed=int(seed),iterations=model.n_iter_,training_loss=model.loss_,
            warnings=[dict(category=v.category.__name__,message=str(v.message)) for v in caught]))
        if retain:members.append(model)
    pp=np.array(predictions);saved=dict(family='neural_network',x_scaler=sx,y_scaler=sy,members=members) if retain else None
    return pp.mean(axis=0),saved,dict(seconds=time.perf_counter()-begin,members=diagnostics),pp
