"""Paired evaluation for repeated stratified holdouts with different test rows."""
import numpy as np
import pandas as pd
from metric_helpers import metrics,weighted_ranker


def score(y,preds,lineage,test_masks):
    labels=np.unique(lineage);eligible=np.array([l for l in labels if np.sum(lineage==l)>=10]);pooled=[];per=[];macro=[]
    for arm,pp in preds.items():
        for r,mask in enumerate(test_masks):
            ix=np.flatnonzero(mask);assert np.isfinite(pp[r,ix]).all() and np.isnan(pp[r,~mask]).all()
            pooled.append(dict(model=arm,repeat=r+1,**metrics(y[ix],pp[r,ix])))
            mm=[];ee=[]
            for label in labels:
                rows=np.flatnonzero(mask&(lineage==label));m=metrics(y[rows],pp[r,rows]);assert len(rows)>0
                per.append(dict(model=arm,repeat=r+1,lineage=str(label),test_n=len(rows),cohort_n=int(np.sum(lineage==label)),**m))
                mm.append(m['rmse'])
                if label in eligible:ee.append(m['rmse'])
            assert len(mm)==23 and len(ee)==19
            macro.append(dict(model=arm,repeat=r+1,macro23_rmse=float(np.mean(mm)),macro19_rmse=float(np.mean(ee))))
    return pd.DataFrame(pooled),pd.DataFrame(per),pd.DataFrame(macro)


def bootstrap(y,preds,lineage,test_masks,count,seed,batch_size=100):
    """One lineage-resampling vector pairs all arms and all 20 outer splits."""
    labels=np.unique(lineage);eligible=[i for i,l in enumerate(labels) if np.sum(lineage==l)>=10];rng=np.random.default_rng(seed);mrng=np.random.default_rng(seed+1)
    nr=len(test_masks);keys=['pooled_spearman','pooled_rmse','pooled_r2','macro23_rmse','macro19_rmse']
    result={arm:{key:np.empty(count) for key in keys} for arm in preds}
    ix=[np.flatnonzero(mask) for mask in test_masks];yy=[y[a] for a in ix];li=[np.searchsorted(labels,lineage[a]) for a in ix]
    yrank=[weighted_ranker(v) for v in yy];prank={arm:[weighted_ranker(pp[r,ix[r]]) for r in range(nr)] for arm,pp in preds.items()}
    lm={arm:np.array([[np.sqrt(np.mean((y[mask&(lineage==l)]-pp[r,mask&(lineage==l)])**2)) for l in labels] for r,mask in enumerate(test_masks)]).mean(axis=0) for arm,pp in preds.items()}
    for offset in range(0,count,batch_size):
        b=min(batch_size,count-offset);sampled=rng.integers(len(labels),size=(b,len(labels)));gw=np.array([np.bincount(v,minlength=len(labels)) for v in sampled],float)
        mdraw=mrng.integers(len(eligible),size=(b,len(eligible)))
        for arm in preds:
            for k in keys[:3]:result[arm][k][offset:offset+b]=0
            result[arm]['macro23_rmse'][offset:offset+b]=gw@lm[arm]/len(labels)
            result[arm]['macro19_rmse'][offset:offset+b]=lm[arm][eligible][mdraw].mean(axis=1)
        for r in range(nr):
            w=gw[:,li[r]];tot=w.sum(axis=1);yr=yrank[r](w);yc=yr-(tot[:,None]+1)/2;yss_rank=(w*yc**2).sum(axis=1)
            ym=(w*yy[r]).sum(axis=1)/tot;yss=(w*(yy[r][None,:]-ym[:,None])**2).sum(axis=1)
            for arm,pp in preds.items():
                pr=prank[arm][r](w);pc=pr-(tot[:,None]+1)/2;den=np.sqrt(yss_rank*(w*pc**2).sum(axis=1))
                rho=np.divide((w*yc*pc).sum(axis=1),den,out=np.full(b,np.nan),where=den!=0);sse=w@((yy[r]-pp[r,ix[r]])**2)
                for key,value in [('pooled_spearman',rho),('pooled_rmse',np.sqrt(sse/tot)),('pooled_r2',1-sse/yss)]:result[arm][key][offset:offset+b]+=value/nr
    assert all(np.isfinite(v).all() for d in result.values() for v in d.values())
    return result


def validate_bootstrap(y,preds,lineage,test_masks,seed):
    count=5;actual=bootstrap(y,preds,lineage,test_masks,count,seed,batch_size=5);labels=np.unique(lineage);eligible=[l for l in labels if np.sum(lineage==l)>=10]
    rng=np.random.default_rng(seed);mrng=np.random.default_rng(seed+1);draws=rng.integers(23,size=(count,23));mdraws=mrng.integers(19,size=(count,19));error=0.
    for b in range(count):
        for arm,pp in preds.items():
            vv=[];m23=[];m19=[]
            for r,mask in enumerate(test_masks):
                rows=np.concatenate([np.flatnonzero(mask&(lineage==labels[j])) for j in draws[b]])
                vv.append(metrics(y[rows],pp[r,rows]));errs={str(l):np.sqrt(np.mean((y[mask&(lineage==l)]-pp[r,mask&(lineage==l)])**2)) for l in labels}
                m23.append(np.mean([errs[str(labels[j])] for j in draws[b]]));m19.append(np.mean([errs[str(eligible[j])] for j in mdraws[b]]))
            expected={key:np.mean([v[metric] for v in vv]) for key,metric in [('pooled_spearman','spearman'),('pooled_rmse','rmse'),('pooled_r2','r2')]}
            expected.update(macro23_rmse=np.mean(m23),macro19_rmse=np.mean(m19))
            for k,v in expected.items():error=max(error,abs(v-actual[arm][k][b]));assert abs(v-actual[arm][k][b])<1e-12
    return dict(passed=True,expanded_scipy_check_draws=count,all_repeats_paired=True,max_error=float(error))
