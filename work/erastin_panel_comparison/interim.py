import json,sys
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
root=Path('outputs/erastin_panel_comparison').resolve();sys.path.insert(0,str(root))
import review_panels as review
review.run.initialize();s=review.run.STATE
base=np.load('outputs/erastin_xgboost_comparison/predictions.npz')
completed=[];pred={review.name(f,p):[] for f in review.FAMILIES for p in review.SIZES}
for r in range(20):
 paths={(f,p):[root/'folds'/f'{f}__p{p}__r{r+1:02d}__l{l:02d}.npz' for l in range(24)] for f in review.FAMILIES for p in [64,80]}
 if not all(path.with_suffix('.json').exists() for arm,pp in paths.items() for path in pp):continue
 completed.append(r+1)
 for f in review.FAMILIES:
  pred[review.name(f,40)].append(base[f][r])
  for p in [64,80]:
   a=np.full(745,np.nan)
   for path in paths[f,p]:
    z=np.load(path);a[z['test_rows']]=z['test_prediction']
   assert np.isfinite(a).all();pred[review.name(f,p)].append(a)
if completed:
 pred={k:np.array(v) for k,v in pred.items()}
 scores,folds,macro=review.old.score_predictions(s['y'],s['lineage'],pred,completed)
 rows=[]
 for scope,frame in [('pooled',scores),('macro',macro)]:
  for arm in pred:
   a=frame[frame.model==arm]
   rows.append(dict(scope=scope,model=arm,**{m:float(a[m].mean()) for m in review.METRICS}))
 result=dict(completed_repeats=completed,interim_mean_metrics=rows)
 Path('work/erastin_panel_comparison/interim_summary.json').write_text(json.dumps(result,indent=2))
 print(json.dumps(result))
else:print(json.dumps(dict(completed_repeats=[])))
