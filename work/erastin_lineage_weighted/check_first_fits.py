import sys,json
from pathlib import Path
sys.path.insert(0,str(Path('outputs/erastin_lineage_weighted').resolve()))
import run_weighted as run
import review_weighted as review
import joblib,numpy as np,xgboost as xgb
run.initialize();s=run.STATE
checks=[]
for p in [40,64,80]:
 for f in run.FAMILIES:
  root=run.OUT/'folds'/f'{f}__p{p}__r01__l00';m=json.loads(root.with_suffix('.json').read_text());a=np.load(root.with_suffix('.npz'));tr=a['train_rows'];te=a['test_rows'];x=s['x'][p]
  assert m['outer_weights']==run.weight_audit(s['lineage'][tr])
  seeds=m['neural_seeds'] if f=='neural_network' else [m['model_seed']]
  pred=review.independent_prediction(f,m['selected_parameters'],seeds,x,s['y'],tr,te)
  err=float(np.max(np.abs(pred-a['test_prediction'])));assert err<1e-7
  checks.append(dict(family=f,panel=p,max_refit_error=err))
run.dump(run.OUT/'first_fit_validation.json',dict(passed=True,checks=checks))
print(json.dumps(checks))
