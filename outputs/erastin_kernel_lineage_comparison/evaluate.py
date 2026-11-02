"""Shared-repeat evaluation, preserving undefined correlations for constant fits."""
import numpy as np,pandas as pd
from scipy.stats import spearmanr,pearsonr
from metric_helpers import weighted_ranker
KEYS=['pooled_spearman','pooled_pearson','pooled_rmse','pooled_mae','pooled_r2','macro_all_rmse','macro_large_rmse']

def finite_mean(v,axis=None):
    v=np.asarray(v,float);ok=np.isfinite(v);n=ok.sum(axis=axis);return np.divide(np.where(ok,v,0).sum(axis=axis),n,out=np.full(np.shape(n),np.nan),where=n>0)

def metrics(y,p):
    y=np.asarray(y);p=np.asarray(p);e=y-p;mse=np.mean(e**2);ss=np.sum((y-y.mean())**2);defined=len(y)>=2 and np.ptp(y)>0 and np.ptp(p)>0
    return dict(n=len(y),spearman=float(spearmanr(y,p).statistic) if defined else np.nan,pearson=float(pearsonr(y,p).statistic) if defined else np.nan,rmse=float(np.sqrt(mse)),mae=float(np.mean(np.abs(e))),r2=float(1-len(y)*mse/ss) if ss>0 and len(y)>1 else np.nan)

def score(y,preds,lineage,test):
    labels=np.unique(lineage);large=[l for l in labels if np.sum(lineage==l)>=10];pooled=[];per=[];macro=[]
    for arm,pp in preds.items():
        for r,mask in enumerate(test):
            assert np.isfinite(pp[r,mask]).all() and np.isnan(pp[r,~mask]).all();pooled.append(dict(model=arm,repeat=r+1,**metrics(y[mask],pp[r,mask])));allm=[];largem=[]
            for l in labels:
                lm=mask&(lineage==l);m=metrics(y[lm],pp[r,lm]);assert lm.sum()>0;per.append(dict(model=arm,repeat=r+1,lineage=str(l),cohort_n=int(np.sum(lineage==l)),test_n=int(lm.sum()),**{k:v for k,v in m.items() if k!='n'}));allm.append(m['rmse'])
                if l in large:largem.append(m['rmse'])
            macro.append(dict(model=arm,repeat=r+1,macro_all_rmse=float(np.mean(allm)),macro_large_rmse=float(np.mean(largem))))
    return pd.DataFrame(pooled),pd.DataFrame(per),pd.DataFrame(macro)

def bootstrap(y,preds,lineage,test,count,seed,batch_size=100):
    labels=np.unique(lineage);large=[i for i,l in enumerate(labels) if np.sum(lineage==l)>=10];rng=np.random.default_rng(seed);mrng=np.random.default_rng(seed+1);R=len(test);L=len(labels)
    perdraw={a:{key:np.full((count,R),np.nan) for key in KEYS} for a in preds};ix=[np.flatnonzero(mask) for mask in test];yy=[y[ii] for ii in ix];li=[np.searchsorted(labels,lineage[ii]) for ii in ix];yrank=[weighted_ranker(v) for v in yy];prank={a:[weighted_ranker(pp[r,ix[r]]) for r in range(R)] for a,pp in preds.items()}
    lm={a:np.array([[np.sqrt(np.mean((y[mask&(lineage==l)]-pp[r,mask&(lineage==l)])**2)) for l in labels] for r,mask in enumerate(test)]) for a,pp in preds.items()}
    for offset in range(0,count,batch_size):
        b=min(batch_size,count-offset);sampled=rng.integers(L,size=(b,L));gw=np.array([np.bincount(v,minlength=L) for v in sampled],float);mdraw=mrng.integers(len(large),size=(b,len(large)));stop=offset+b
        for a in preds:
            perdraw[a]['macro_all_rmse'][offset:stop]=gw@lm[a].T/L
            perdraw[a]['macro_large_rmse'][offset:stop]=lm[a][:,large][:,mdraw].mean(axis=2).T
        for r in range(R):
            w=gw[:,li[r]];tot=w.sum(axis=1);yr=yrank[r](w);yrc=yr-(tot[:,None]+1)/2;yrss=(w*yrc**2).sum(axis=1);yc=yy[r][None,:]-(w*yy[r]).sum(axis=1)[:,None]/tot[:,None];yss=(w*yc**2).sum(axis=1)
            for a,pp in preds.items():
                p=pp[r,ix[r]];pr=prank[a][r](w);prc=pr-(tot[:,None]+1)/2;prss=(w*prc**2).sum(axis=1);pc=p[None,:]-(w*p).sum(axis=1)[:,None]/tot[:,None];pss=(w*pc**2).sum(axis=1);sse=w@((yy[r]-p)**2)
                # Numerical cancellation in weighted centering must not define a constant predictor's correlation.
                with np.errstate(divide='ignore',invalid='ignore'):
                    vals=dict(pooled_spearman=(w*yrc*prc).sum(axis=1)/np.sqrt(yrss*prss),pooled_pearson=(w*yc*pc).sum(axis=1)/np.sqrt(yss*pss),pooled_rmse=np.sqrt(sse/tot),pooled_mae=w@np.abs(yy[r]-p)/tot,pooled_r2=1-sse/yss)
                if np.ptp(p)==0:vals['pooled_spearman'][:]=np.nan;vals['pooled_pearson'][:]=np.nan
                for key,v in vals.items():perdraw[a][key][offset:stop,r]=v
    return {a:{k:finite_mean(v,axis=1) for k,v in d.items()} for a,d in perdraw.items()},perdraw

def validate_bootstrap(y,preds,lineage,test,seed):
    draws,per=bootstrap(y,preds,lineage,test,5,seed,batch_size=5);labels=np.unique(lineage);large=[l for l in labels if np.sum(lineage==l)>=10];rng=np.random.default_rng(seed);mrng=np.random.default_rng(seed+1);picks=rng.integers(len(labels),size=(5,len(labels)));mpicks=mrng.integers(len(large),size=(5,len(large)));error=0.;checks=0
    for b in range(5):
        for arm,pp in preds.items():
            expected={k:[] for k in KEYS}
            for r,mask in enumerate(test):
                rows=np.concatenate([np.flatnonzero(mask&(lineage==labels[j])) for j in picks[b]]);m=metrics(y[rows],pp[r,rows])
                for key in ['spearman','pearson','rmse','mae','r2']:expected['pooled_'+key].append(m[key])
                errs={str(l):np.sqrt(np.mean((y[mask&(lineage==l)]-pp[r,mask&(lineage==l)])**2)) for l in labels};expected['macro_all_rmse'].append(np.mean([errs[str(labels[j])] for j in picks[b]]));expected['macro_large_rmse'].append(np.mean([errs[str(large[j])] for j in mpicks[b]]))
            for key,v in expected.items():
                value=finite_mean(v);actual=draws[arm][key][b]
                if np.isnan(value):assert np.isnan(actual)
                else:error=max(error,float(abs(value-actual)));checks+=1;assert abs(value-actual)<1e-10
    return dict(passed=True,expanded_scipy_draws=5,metric_checks=checks,max_error=error,constant_predictions_keep_undefined_correlations=True)
