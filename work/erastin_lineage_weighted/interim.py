"""Read-only descriptive interim scores; never changes the frozen benchmark."""
import sys,json
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path('outputs/erastin_lineage_weighted').resolve()))
import review_weighted as review
run=review.run;run.initialize();s=run.STATE
count=int(sys.argv[1]);base=np.load(run.WS/'outputs/erastin_panel_comparison/predictions.npz')
preds={review.name(f,p):np.full((count,745),np.nan) for f in run.FAMILIES for p in [40,64,80]}
for r in range(count):
 for p in [40,64,80]:
  for f in run.FAMILIES:
   for l in range(24):
    a=np.load(run.OUT/'folds'/f'{f}__p{p}__r{r+1:02d}__l{l:02d}.npz');preds[review.name(f,p)][r,a['test_rows']]=a['test_prediction']
preds.update({review.name(f,p,False):base[f+'__'+str(p)][:count].copy() for f in run.FAMILIES for p in [40,64,80]})
assert all(np.isfinite(v).all() for v in preds.values())
scores,_,macro=review.old.score_predictions(s['y'],s['lineage'],preds,list(range(1,count+1)))
rows=[]
for scope,frame in [('pooled',scores),('macro19',macro)]:
 for f in run.FAMILIES:
  for p in [40,64,80]:
   w=frame[frame.model==review.name(f,p)].sort_values('repeat');u=frame[frame.model==review.name(f,p,False)].sort_values('repeat')
   rows.append(dict(scope=scope,family=f,panel=p,weighted_spearman=float(w.spearman.mean()),original_spearman=float(u.spearman.mean()),
    weighted_rmse=float(w.rmse.mean()),original_rmse=float(u.rmse.mean()),rmse_pct_change=float(100*(w.rmse.mean()/u.rmse.mean()-1)),
    rmse_better_seeds=int(np.sum(w.rmse.to_numpy()<u.rmse.to_numpy())),spearman_better_seeds=int(np.sum(w.spearman.to_numpy()>u.spearman.to_numpy()))))
result=dict(status='PRELIMINARY_DESCRIPTIVE_ONLY',seeds=count,rows=rows)
Path(f'work/erastin_lineage_weighted/interim_{count:02d}.json').write_text(json.dumps(result,indent=2))
print(json.dumps(result,indent=2))
