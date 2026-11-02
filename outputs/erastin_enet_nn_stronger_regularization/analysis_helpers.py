"""Unchanged scoring/paired-bootstrap procedure from the prior EN benchmark."""
import numpy as np
import pandas as pd
import common as c
from evaluate import KEYS,score,bootstrap,finite_mean

def evaluate(assay,preds,protocol):
    s=c.STATE[assay];y,lin,test=s['y'],s['lineage'],s['test']
    pooled,per,macro=score(y,preds,lin,test)
    for d in [pooled,per,macro]:d.insert(0,'assay',assay)
    values,summary={},[]
    for f in preds:
        values[f]={'pooled_'+k:pooled[pooled.model==f].sort_values('repeat')[k].to_numpy() for k in ['spearman','pearson','rmse','mae','r2']}
        values[f].update({k:macro[macro.model==f].sort_values('repeat')[k].to_numpy() for k in ['macro_all_rmse','macro_large_rmse']})
        row=dict(assay=assay,model=f)
        for k,v in values[f].items():
            good=v[np.isfinite(v)];row.update({k+'_mean':float(finite_mean(v)),k+'_sd':float(good.std(ddof=1)) if len(good)>1 else np.nan})
        summary.append(row)
    draws,perdraw=bootstrap(y,preds,lin,test,protocol['uncertainty']['count'],protocol['uncertainty']['seed'])
    paired,effects,wins=[],[],[]
    for first,ref in protocol['primary_comparisons']+protocol['secondary_comparisons']:
        for key in KEYS:
            sign=-1 if key.endswith('rmse') or key.endswith('mae') else 1
            delta=sign*(values[first][key]-values[ref][key]);dd=finite_mean(sign*(perdraw[first][key]-perdraw[ref][key]),axis=1)
            good=dd[np.isfinite(dd)];lo,hi=np.percentile(good,[2.5,97.5]) if len(good) else (np.nan,np.nan)
            paired.append(dict(assay=assay,first_model=first,reference_model=ref,metric=key,improvement=float(finite_mean(delta)),
                favorable_repeats=int(np.sum(delta>1e-12)),tied_repeats=int(np.sum(np.abs(delta)<=1e-12)),defined_repeats=int(np.isfinite(delta).sum()),conditional_ci_low=lo,conditional_ci_high=hi))
        favorable=0
        for label in np.unique(lin):
            a=per[(per.model==first)&(per.lineage==label)].sort_values('repeat');b=per[(per.model==ref)&(per.lineage==label)].sort_values('repeat')
            delta=b.rmse.to_numpy()-a.rmse.to_numpy();favorable+=int(delta.mean()>1e-12)
            effects.append(dict(assay=assay,first_model=first,reference_model=ref,lineage=str(label),cohort_n=int(a.cohort_n.iloc[0]),rmse_improvement=float(delta.mean())))
        wins.append(dict(assay=assay,first_model=first,reference_model=ref,favorable_lineages=favorable,lineages=len(np.unique(lin))))
    np.savez_compressed(c.OUT/assay/'bootstrap_draws.npz',**{f+'__'+k:v for f,d in draws.items() for k,v in d.items()})
    return [pd.DataFrame(summary),pd.DataFrame(paired),pd.DataFrame(effects),pd.DataFrame(wins),pooled,per,macro]
