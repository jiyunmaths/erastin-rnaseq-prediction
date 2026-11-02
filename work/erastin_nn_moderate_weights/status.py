from pathlib import Path
import json,time
root=Path('outputs/erastin_nn_moderate_weights');a=json.loads((root/'progress.json').read_text());out={k:a[k] for k in ['phase','completed','total'] if k in a}
if a['phase']=='fitting':out.update(percent=round(100*a['completed']/a['total'],1),elapsed_minutes=round(a['elapsed_seconds']/60,1),remaining_minutes=round(a['estimated_remaining_seconds']/60,1),progress_age_seconds=round(time.time()-(root/'progress.json').stat().st_mtime,1))
if (root/'audit_progress.json').exists():out['audited_models']=json.loads((root/'audit_progress.json').read_text())['audited_models']
for kind in ['run','review']:
 p=Path('work/erastin_nn_moderate_weights')/(kind+'.log')
 if p.exists() and 'Traceback' in p.read_text():out[kind+'_error']=p.read_text()[-2500:]
print(json.dumps(out))
