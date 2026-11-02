#!/usr/bin/env python3
"""Audit all new stratified fits and summarize matched test predictions."""
import json,time,zipfile
from pathlib import Path
from collections import Counter
import joblib,numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
import run_stratified as run
from make_splits import inner_rows
from lineage_weights import weights_for,weight_audit
from evaluate import score,bootstrap,validate_bootstrap
OUT=run.OUT;WS=run.WS

def table(name,rows):
    a=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows);a.to_csv(OUT/name,sep='\t',index=False,float_format='%.12g');return a

def wait(path):
    while not path.exists():time.sleep(.5)

def main():
    run.initialize();s=run.STATE;x=s['x'];y=s['y'];g=s['groups'];lin=s['lineage'];protocol=json.loads((OUT/'protocol.json').read_text());ph=run.sha(OUT/'protocol.json')
    for name,h in protocol['scripts'].items():assert run.sha(OUT/name)==h
    for name,h in protocol['data_sha256'].items():assert run.sha(OUT/name)==h
    run.dump(OUT/'analysis_source_manifest.json',dict(review_sha256=run.sha(Path(__file__)),report_sha256=run.sha(OUT/'report_stratified.py')))
    preds={arm:np.full(s['test'].shape,np.nan) for arm in run.ARMS};params=[];tuning=[];weights=[];weightgroups=[];warnings=[];objective_rows=[];references=[]
    maxp=maxloss=maxobj=0.;groups_checked=weights_checked=scores_checked=0
    for r in range(20):
        mask=s['test'][r];tr=np.flatnonzero(~mask);te=np.flatnonzero(mask);inner=s['inner'][r]
        for arm in run.ARMS:
            root=OUT/'folds'/f'{arm}__r{r+1:02d}';wait(root.with_suffix('.json'));m=json.loads(root.with_suffix('.json').read_text());a=np.load(root.with_suffix('.npz'))
            assert m['protocol_sha256']==ph and m['arm']==arm and m['repeat']==r+1 and m['panel']==64 and m['train_lineages']==m['test_lineages']==23
            assert m['outer_split_seed']==int(s['seeds'][r]) and m['inner_split_seed']==int(s['inner_seeds'][r]);np.testing.assert_array_equal(m['neural_seeds'],s['nn_seeds'][r])
            np.testing.assert_array_equal(a['train_rows'],tr);np.testing.assert_array_equal(a['test_rows'],te);assert set(g[tr]).isdisjoint(g[te]) and set(lin[tr])==set(lin[te])==set(lin);groups_checked+=1
            for ext in ['npz','joblib']:assert run.sha(root.with_suffix('.'+ext))==m[ext+'_sha256']
            c=m['selected_candidate'];p=s['grid'][c];assert p==m['selected_parameters'] and c==np.argmin(a['inner_fold_rmse'].mean(axis=1))
            assert np.isfinite(a['inner_validation_predictions'][:,inner>=0]).all() and np.isnan(a['inner_validation_predictions'][:,inner<0]).all()
            for k in range(5):
                aa,bb=inner_rows(inner,k);assert set(aa)|set(bb)==set(tr) and set(g[aa]).isdisjoint(g[bb]) and set(lin[aa])==set(lin);groups_checked+=1
                assert m['inner_scaling'][k]['y_mean']==y[aa].mean() and m['inner_scaling'][k]['y_scale']==y[aa].std()
                w=weights_for(lin[aa],arm);np.testing.assert_array_equal(a['inner_training_weights'][k,aa],w);assert np.isnan(a['inner_training_weights'][k,np.setdiff1d(np.arange(len(y)),aa)]).all()
                audit=weight_audit(lin[aa],arm);assert audit==m['inner_scaling'][k]['weights'];weights_checked+=1
                weights.append(dict(arm=arm,repeat=r+1,stage='inner',fold=k,**{key:v for key,v in audit.items() if key not in ['groups','scheme']}))
                weightgroups.extend(dict(arm=arm,repeat=r+1,stage='inner',fold=k,**v) for v in audit['groups'])
                loss=np.sqrt(np.mean((a['inner_validation_predictions'][:,bb]-y[bb][None,:])**2,axis=1));maxloss=max(maxloss,float(np.max(np.abs(loss-a['inner_fold_rmse'][:,k]))));scores_checked+=len(s['grid'])
            w=weights_for(lin[tr],arm);np.testing.assert_array_equal(a['outer_training_weights'],w);audit=weight_audit(lin[tr],arm);assert audit==m['outer_weights'];weights_checked+=1
            weights.append(dict(arm=arm,repeat=r+1,stage='outer',fold=-1,**{key:v for key,v in audit.items() if key not in ['groups','scheme']}));weightgroups.extend(dict(arm=arm,repeat=r+1,stage='outer',fold=-1,**v) for v in audit['groups'])
            model=joblib.load(root.with_suffix('.joblib'));sx=StandardScaler().fit(x[tr]);sy=StandardScaler().fit(y[tr,None]);z=sx.transform(x[tr]);target=sy.transform(y[tr,None]).ravel()
            for key in ['mean_','scale_']:
                np.testing.assert_allclose(getattr(model['x_scaler'],key),getattr(sx,key),atol=1e-12,rtol=0);np.testing.assert_allclose(getattr(model['y_scaler'],key),getattr(sy,key),atol=1e-12,rtol=0)
            assert [v.random_state for v in model['members']]==m['neural_seeds']
            members=np.array([sy.inverse_transform(v.predict(sx.transform(x[te]))[:,None]).ravel() for v in model['members']]);np.testing.assert_allclose(members,a['member_test_predictions'],atol=1e-12,rtol=0)
            pred=members.mean(axis=0);maxp=max(maxp,float(np.max(np.abs(pred-a['test_prediction']))));preds[arm][r,te]=pred
            for j,v in enumerate(model['members']):
                np.testing.assert_array_equal(v.training_sample_weight_,w);assert v.alpha==p['alpha'] and v.hidden_layer_sizes==(p['width'],) and v.activation=='tanh' and v.solver=='lbfgs'
                obj=.5*np.sum(w*(v.predict(z)-target)**2)/len(tr)+.5*v.alpha*sum(np.sum(t*t) for t in v.coefs_)/len(tr);err=float(abs(obj-v.loss_));maxobj=max(maxobj,err);assert err<1e-10
                objective_rows.append(dict(arm=arm,repeat=r+1,member=j,loss_recalculation_error=err,iterations=v.n_iter_,training_objective=obj))
            for d in m['fit_diagnostics']:
                for member in d['members']:warnings.extend(dict(arm=arm,repeat=r+1,stage=d['stage'],**v) for v in member['warnings'])
            if r in [0,19]:
                ref,_,_,_=run.fitted_neural(p,x,y,tr,te,s['nn_seeds'][r],w);err=float(np.max(np.abs(ref-pred)));ie=0.
                for k in range(5):
                    aa,bb=inner_rows(inner,k);ref,_,_,_=run.fitted_neural(p,x,y,aa,bb,s['nn_seeds'][r],weights_for(lin[aa],arm));ie=max(ie,float(np.max(np.abs(ref-a['inner_validation_predictions'][c,bb]))))
                assert max(err,ie)<1e-10;references.append(dict(arm=arm,repeat=r+1,outer_error=err,inner_error=ie,inner_refits=5))
            params.append(dict(model=arm,repeat=r+1,candidate=c,**p,mean_inner_rmse=m['mean_inner_rmse']))
            tuning.extend(dict(model=arm,repeat=r+1,candidate=j,**v,mean_inner_rmse=float(a['inner_fold_rmse'][j].mean())) for j,v in enumerate(s['grid']))
            run.dump(OUT/'audit_progress.json',dict(audited_models=len(params),total=80))
    wait(OUT/'fit_completion.json');assert json.loads((OUT/'fit_completion.json').read_text())['complete']
    assert groups_checked==weights_checked==480 and scores_checked==1600 and len(references)==8 and maxp<1e-10 and maxloss<1e-12
    for arm,pp in preds.items():assert np.isfinite(pp[s['test']]).all() and np.isnan(pp[~s['test']]).all()
    table('selected_parameters.tsv',params);table('tuning_scores.tsv',tuning);table('training_subset_weight_audit.tsv',weights);table('training_lineage_weights.tsv',weightgroups);table('training_objective_checks.tsv',objective_rows)
    counts=Counter((v['category'],v['message']) for v in warnings);warning_summary=[dict(category=k[0],message=k[1],count=n) for k,n in counts.items()]
    bv=validate_bootstrap(y,preds,lin,s['test'],seed=2026110604)
    validation=dict(passed=True,outer_models=80,all_23_lineages_in_both_outer_sets=True,all_23_lineages_in_each_inner_training_set=True,
        identity_group_exclusion_checks=groups_checked,training_weight_checks=weights_checked,candidate_fold_scores_checked=scores_checked,
        prediction_max_error=maxp,inner_rmse_max_error=maxloss,training_objectives_checked=len(objective_rows),training_objective_max_error=maxobj,
        selected_setting_replays=references,bootstrap_validation=bv,preflight=json.loads((OUT/'preflight_validation.json').read_text()),
        warning_count=sum(counts.values()),warning_summary=warning_summary,convergence_warning_count=sum(v['count'] for v in warning_summary if v['category']=='ConvergenceWarning'),
        source_hashes_verified=True,training_only_unweighted_scaling_verified=True)
    run.dump(OUT/'validation.json',validation)
    pooled,per,macro=score(y,preds,lin,s['test']);table('per_repeat_pooled_metrics.tsv',pooled);table('per_repeat_macro_metrics.tsv',macro);table('per_lineage_metrics.tsv',per)
    vals={arm:{} for arm in run.ARMS};summary=[]
    for arm in run.ARMS:
        row=dict(model=arm)
        for key in ['spearman','rmse','r2']:
            v=pooled[pooled.model==arm].sort_values('repeat')[key].to_numpy();vals[arm]['pooled_'+key]=v
        for key in ['macro23_rmse','macro19_rmse']:vals[arm][key]=macro[macro.model==arm].sort_values('repeat')[key].to_numpy()
        for key,v in vals[arm].items():row.update({key+'_mean':v.mean(),key+'_sd':v.std(ddof=1),key+'_min':v.min(),key+'_max':v.max()})
        summary.append(row)
    summary=table('summary.tsv',summary);run.dump(OUT/'progress.json',dict(phase='paired_bootstrap',audited_models=80))
    draws=bootstrap(y,preds,lin,s['test'],protocol['uncertainty']['count'],protocol['uncertainty']['seed']);pairs=protocol['primary_comparisons']+protocol['secondary_comparisons'];paired=[];seedrows=[];effects=[]
    for first,reference in pairs:
        for key in vals[first]:
            sign=-1 if key.endswith('rmse') else 1;delta=sign*(vals[first][key]-vals[reference][key]);lo,hi=np.percentile(sign*(draws[first][key]-draws[reference][key]),[2.5,97.5])
            paired.append(dict(first_model=first,reference_model=reference,metric=key,improvement=delta.mean(),favorable_repeats=int(np.sum(delta>0)),repeats=20,conditional_ci_low=lo,conditional_ci_high=hi))
            seedrows.extend(dict(first_model=first,reference_model=reference,metric=key,repeat=r+1,improvement=float(v)) for r,v in enumerate(delta))
        for label in np.unique(lin):
            a=per[(per.model==first)&(per.lineage==label)].sort_values('repeat');b=per[(per.model==reference)&(per.lineage==label)].sort_values('repeat');d=b.rmse.to_numpy()-a.rmse.to_numpy()
            effects.append(dict(first_model=first,reference_model=reference,lineage=str(label),test_n_min=int(a.test_n.min()),test_n_max=int(a.test_n.max()),rmse_improvement=d.mean(),favorable_repeats=int(np.sum(d>0))))
    paired=table('paired_comparisons.tsv',paired);table('paired_seed_differences.tsv',seedrows);effects=table('lineage_paired_effects.tsv',effects)
    np.savez_compressed(OUT/'predictions.npz',sample_ids=s['ids'],group_ids=g,lineage=lin,y=y,outer_test=s['test'],outer_split_seeds=s['seeds'],**preds)
    np.savez_compressed(OUT/'bootstrap_draws.npz',**{arm+'__'+key:v for arm,d in draws.items() for key,v in d.items()})
    from report_stratified import render
    render(summary,paired,effects,validation)
    for a in json.loads((OUT/'source_manifest.json').read_text())['files']:assert run.sha(WS/a['path'])==a['sha256']
    am=json.loads((OUT/'analysis_source_manifest.json').read_text());assert run.sha(Path(__file__))==am['review_sha256'] and run.sha(OUT/'report_stratified.py')==am['report_sha256']
    run.dump(OUT/'final_summary.json',dict(validation_passed=True,summary=summary.to_dict('records'),outer_models=80,lineages_in_train_and_test=23));run.dump(OUT/'progress.json',dict(phase='complete',validation_passed=True,outer_models=80))
    manifest=dict(files=[dict(path=str(p.relative_to(OUT)),sha256=run.sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts]);run.dump(OUT/'artifact_manifest.json',manifest)
    archive=OUT.parent/'erastin_lineage_stratified_nn_results.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for row in manifest['files']:z.write(OUT/row['path'],'erastin_lineage_stratified_nn/'+row['path'])
        z.write(OUT/'artifact_manifest.json','erastin_lineage_stratified_nn/artifact_manifest.json')
    import hashlib
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in manifest['files']:assert hashlib.sha256(z.read('erastin_lineage_stratified_nn/'+row['path'])).hexdigest()==row['sha256']
    print(json.dumps(dict(event='complete',archive_bytes=archive.stat().st_size)),flush=True)
if __name__=='__main__':main()
