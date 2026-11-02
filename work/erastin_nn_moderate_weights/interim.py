"""Read-only interim scores; never changes fitting or the frozen protocol."""
from pathlib import Path
import sys,json
import numpy as np
sys.path.insert(0,str(Path('outputs/erastin_nn_moderate_weights').resolve()))
import run_moderate as run
import evaluation as ev
from lineage_weights import SCHEMES
run.initialize();s=run.STATE;count=int(sys.argv[1]);preds={scheme:np.full((count,745),np.nan) for scheme in SCHEMES}
for r in range(count):
 for scheme in SCHEMES:
  for l in range(24):
   a=np.load(run.OUT/'folds'/f'{scheme}__r{r+1:02d}__l{l:02d}.npz');preds[scheme][r,a['test_rows']]=a['test_prediction']
base=np.load(run.WS/'outputs/erastin_panel_comparison/predictions.npz');full=np.load(run.WS/'outputs/erastin_lineage_weighted/predictions.npz')
preds.update(unweighted=base['neural_network__64'][:count].copy(),inverse_frequency=full['weighted__neural_network__64'][:count].copy())
scores,_,macro=ev.score_predictions(s['y'],s['lineage'],preds,list(range(1,count+1)))
rows=[]
for scope,frame in [('pooled',scores),('macro19',macro)]:
 u=frame[frame.model=='unweighted'].sort_values('repeat')
 for scheme in preds:
  w=frame[frame.model==scheme].sort_values('repeat')
  rows.append(dict(scope=scope,scheme=scheme,spearman=float(w.spearman.mean()),rmse=float(w.rmse.mean()),rmse_pct_change_vs_unweighted=float(100*(w.rmse.mean()/u.rmse.mean()-1)),rmse_better_seeds=int(np.sum(w.rmse.to_numpy()<u.rmse.to_numpy())),spearman_better_seeds=int(np.sum(w.spearman.to_numpy()>u.spearman.to_numpy()))))
result=dict(status='PRELIMINARY_DESCRIPTIVE',seeds=count,rows=rows)
Path(f'work/erastin_nn_moderate_weights/interim_{count:02d}.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2))
