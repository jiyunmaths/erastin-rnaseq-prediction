"""Read-only fit inventory and cost-weighted planning estimate."""
import json,time,statistics
from pathlib import Path
OUT=Path(__file__).resolve().parent
def status():
    pr=json.load(open(OUT/'protocol.json'));cost=json.load(open(OUT/'runtime_cost_estimate.json'));rows=[json.load(open(p)) for p in (OUT/'folds').glob('*.json')];new=[r for r in rows if not r['reused']];base=[r for r in rows if r['reused']];pilot={(r['family'],r['panel']):r['seconds'] for r in cost['pilot_fits']};breakdown=[];remaining=0.
    for (family,panel),seconds in pilot.items():
        done=[r for r in new if (r['family'],r['panel'])==(family,panel)];estimate=statistics.median(r['seconds'] for r in done) if len(done)>1 else seconds;remaining+=(20-len(done))*estimate;breakdown.append(dict(family=family,panel=panel,completed=len(done),total=20,median_or_pilot_seconds=estimate))
    complete=(OUT/'fit_completion.json').exists();minutes=0. if complete else remaining/pr['workers']/60
    return dict(snapshot_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),fits_complete=complete,new_models_completed=len(new),new_models_total=180,reused_models=len(base),breakdown=breakdown,cost_weighted_remaining_minutes=minutes,rough_planning_range_minutes=[.7*minutes,1.8*minutes],eta_limit='Cost-weighted estimate from per-arm pilot/observed median times, not a confidence interval. Active partly finished fits are counted at full remaining cost; CPU contention and seed-specific iterations can vary. Do not use the generic completed-job average, which underweights slow Elastic Net jobs early.')
if __name__=='__main__':print(json.dumps(status(),indent=2))
