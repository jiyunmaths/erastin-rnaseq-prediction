#!/usr/bin/env python3
"""Audit all nine unweighted arms, including exact 64-gene NN reuse."""
import json,time,zipfile,hashlib
from pathlib import Path
from collections import Counter
import joblib,numpy as np,pandas as pd,xgboost as xgb
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
import run_panels as run
from validate_preflight import native_nn
from make_splits import inner_rows
from evaluate import score,bootstrap,validate_bootstrap
OUT=run.OUT;WS=run.WS

def table(name,rows):
    a=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows);a.to_csv(OUT/name,sep='\t',index=False,float_format='%.12g');return a

def pairs():
    name=run.arm_name;v=[]
    for f in run.FAMILIES:v.extend([(name(f,64),name(f,40)),(name(f,80),name(f,40)),(name(f,80),name(f,64))])
    for p in run.PANELS:v.extend([(name('neural_network',p),name('ridge',p)),(name('xgboost',p),name('ridge',p)),(name('xgboost',p),name('neural_network',p))])
    for f in ['ridge','xgboost']:
        for p in [64,80]:v.append((name(f,p),name('neural_network',40)))
    assert len(v)==len(set(v))==22;return v

def independent(f,params,x,y,a,b,seeds,model_seed):
    sx=StandardScaler().fit(x[a])
    if f=='ridge':return Ridge(alpha=params['alpha'],solver='svd').fit(sx.transform(x[a]),y[a]).predict(sx.transform(x[b]))
    if f=='neural_network':return native_nn(params,x,y,a,b,seeds)
    sy=StandardScaler().fit(y[a,None]);pp=run.xp.parameters(params,model_seed);pp.pop('nthread');pp.pop('seed')
    model=xgb.XGBRegressor(**pp,n_jobs=1,random_state=int(model_seed),n_estimators=params['num_boost_round']).fit(x[a],sy.transform(y[a,None]).ravel())
    return model.predict(x[b]).astype(float)*sy.scale_[0]+sy.mean_[0]

def wait(path):
    while not path.exists():time.sleep(.5)

def audit():
    run.initialize();s=run.STATE;y=s['y'];g=s['groups'];lin=s['lineage'];ph=run.sha(OUT/'protocol.json');protocol=json.loads((OUT/'protocol.json').read_text())
    for name,h in protocol['scripts'].items():assert run.sha(OUT/name)==h
    for name,h in protocol['data_sha256'].items():assert run.sha(OUT/name)==h
    for a in json.loads((OUT/'source_manifest.json').read_text())['files']:assert run.sha(WS/a['path'])==a['sha256']
    for a in json.loads((OUT/'reuse_manifest.json').read_text())['files']:assert run.sha(OUT/a['local'])==a['sha256']
    run.dump(OUT/'analysis_source_manifest.json',dict(review_sha256=run.sha(Path(__file__)),report_sha256=run.sha(OUT/'report_panels.py')))
    shared=np.load(WS/'outputs/erastin_lineage_stratified_nn/shared_splits.npz');np.testing.assert_array_equal(s['test'],shared['outer_test']);np.testing.assert_array_equal(s['inner'],shared['inner_fold'])
    preds={run.arm_name(f,p):np.full(s['test'].shape,np.nan) for f in run.FAMILIES for p in run.PANELS};params=[];tuning=[];warnings=[];objectives=[];references=[]
    maxp=maxloss=maxobj=0.;scores_checked=groups_checked=0;reused=0
    for r in range(20):
        mask=s['test'][r];tr=np.flatnonzero(~mask);te=np.flatnonzero(mask);inner=s['inner'][r]
        for panel in run.PANELS:
            x=s['x'][panel];genes=s['genes'][panel]
            for f in run.FAMILIES:
                arm=run.arm_name(f,panel);root=OUT/'folds'/f'{f}__p{panel}__r{r+1:02d}';wait(root.with_suffix('.json'));m=json.loads(root.with_suffix('.json').read_text());a=np.load(root.with_suffix('.npz'))
                assert m['protocol_sha256']==ph and m['family']==f and m['panel']==panel and m['repeat']==r+1 and m['train_lineages']==m['test_lineages']==23
                assert m['outer_split_seed']==int(s['seeds'][r]) and m['inner_split_seed']==int(s['inner_seeds'][r]) and m['model_seed']==int(s['model_seeds'][r]);np.testing.assert_array_equal(m['neural_seeds'],s['nn_seeds'][r])
                assert m['reused']==(f=='neural_network' and panel==64);reused+=int(m['reused'])
                if m['reused']:assert run.sha(OUT/'source_metadata'/f'{f}__p{panel}__r{r+1:02d}.json')==m['original_metadata_sha256']
                np.testing.assert_array_equal(a['train_rows'],tr);np.testing.assert_array_equal(a['test_rows'],te);assert set(g[tr]).isdisjoint(g[te]) and set(lin[tr])==set(lin[te])==set(lin);groups_checked+=1
                ext='ubj' if f=='xgboost' else 'joblib'
                for suffix in ['npz',ext]:assert run.sha(root.with_suffix('.'+suffix))==m[suffix+'_sha256']
                grid=run.grid_for(f);c=m['selected_candidate'];p=grid[c];assert p==m['selected_parameters'] and c==np.argmin(a['inner_fold_rmse'].mean(axis=1))
                assert np.isfinite(a['inner_validation_predictions'][:,inner>=0]).all() and np.isnan(a['inner_validation_predictions'][:,inner<0]).all()
                for k in range(5):
                    aa,bb=inner_rows(inner,k);assert set(aa)|set(bb)==set(tr) and set(g[aa]).isdisjoint(g[bb]) and set(lin[aa])==set(lin);groups_checked+=1
                    assert m['inner_scaling'][k]['y_mean']==y[aa].mean() and m['inner_scaling'][k]['y_scale']==y[aa].std()
                    loss=np.sqrt(np.mean((a['inner_validation_predictions'][:,bb]-y[bb][None,:])**2,axis=1));maxloss=max(maxloss,float(np.max(np.abs(loss-a['inner_fold_rmse'][:,k]))));scores_checked+=len(grid)
                if f=='xgboost':
                    model=xgb.Booster();model.load_model(root.with_suffix('.ubj'));model.set_param({'nthread':1});assert model.num_features()==panel and model.feature_names==list(genes) and model.num_boosted_rounds()==p['num_boost_round']
                    assert m['y_mean']==y[tr].mean() and m['y_scale']==y[tr].std();pred=model.predict(run.matrix(x[te],genes)).astype(float)*m['y_scale']+m['y_mean']
                else:
                    model=joblib.load(root.with_suffix('.joblib'));sx=StandardScaler().fit(x[tr])
                    if f=='ridge':
                        np.testing.assert_allclose(model['mean'],sx.mean_,atol=1e-12,rtol=0);np.testing.assert_allclose(model['scale'],sx.scale_,atol=1e-12,rtol=0);assert model['alpha']==p['alpha'] and model['intercept_after_scaling']==y[tr].mean();pred=sx.transform(x[te])@model['coef']+model['intercept_after_scaling']
                    else:
                        sy=StandardScaler().fit(y[tr,None]);target=sy.transform(y[tr,None]).ravel();z=sx.transform(x[tr])
                        for key in ['mean_','scale_']:
                            np.testing.assert_allclose(getattr(model['x_scaler'],key),getattr(sx,key),atol=1e-12,rtol=0);np.testing.assert_allclose(getattr(model['y_scaler'],key),getattr(sy,key),atol=1e-12,rtol=0)
                        assert [v.random_state for v in model['members']]==m['neural_seeds']
                        members=np.array([sy.inverse_transform(v.predict(sx.transform(x[te]))[:,None]).ravel() for v in model['members']]);np.testing.assert_allclose(members,a['member_test_predictions'],atol=1e-12,rtol=0);pred=members.mean(axis=0)
                        for j,v in enumerate(model['members']):
                            np.testing.assert_array_equal(v.training_sample_weight_,np.ones(len(tr)));assert v.alpha==p['alpha'] and v.hidden_layer_sizes==(p['width'],)
                            obj=.5*np.mean((v.predict(z)-target)**2)+.5*v.alpha*sum(np.sum(t*t) for t in v.coefs_)/len(tr);err=float(abs(obj-v.loss_));maxobj=max(maxobj,err);assert err<1e-10;objectives.append(dict(model=arm,repeat=r+1,member=j,loss_recalculation_error=err,iterations=v.n_iter_))
                maxp=max(maxp,float(np.max(np.abs(pred-a['test_prediction']))));preds[arm][r,te]=pred
                for d in m['fit_diagnostics']:
                    if 'warning_category' in d:warnings.append(dict(model=arm,repeat=r+1,category=d['warning_category'],message=d['warning']))
                    for member in d.get('members',[]):warnings.extend(dict(model=arm,repeat=r+1,**v) for v in member['warnings'])
                if r in [0,19]:
                    ref=independent(f,p,x,y,tr,te,s['nn_seeds'][r],int(s['model_seeds'][r]));err=float(np.max(np.abs(ref-pred)));ie=0.
                    for k in range(5):
                        aa,bb=inner_rows(inner,k);ref=independent(f,p,x,y,aa,bb,s['nn_seeds'][r],int(s['model_seeds'][r]));ie=max(ie,float(np.max(np.abs(ref-a['inner_validation_predictions'][c,bb]))))
                    assert max(err,ie)<1e-7;references.append(dict(model=arm,repeat=r+1,outer_error=err,inner_error=ie,inner_refits=5))
                params.append(dict(model=arm,family=f,panel=panel,repeat=r+1,candidate=c,**p,mean_inner_rmse=m['mean_inner_rmse'],reused=m['reused']))
                tuning.extend(dict(model=arm,repeat=r+1,candidate=j,**v,mean_inner_rmse=float(a['inner_fold_rmse'][j].mean())) for j,v in enumerate(grid))
                run.dump(OUT/'audit_progress.json',dict(audited_models=len(params),total=180))
    wait(OUT/'fit_completion.json');assert json.loads((OUT/'fit_completion.json').read_text())['complete']
    assert reused==20 and groups_checked==1080 and scores_checked==17100 and len(references)==18 and maxp<1e-10 and maxloss<1e-12
    for arm,pp in preds.items():assert np.isfinite(pp[s['test']]).all() and np.isnan(pp[~s['test']]).all()
    previous=np.load(WS/'outputs/erastin_lineage_stratified_nn/predictions.npz');np.testing.assert_array_equal(preds['neural_network__64'],previous['unweighted'])
    params=table('selected_parameters.tsv',params);table('tuning_scores.tsv',tuning);table('neural_training_objective_checks.tsv',objectives)
    counts=Counter((v['category'],v['message']) for v in warnings);warning_summary=[dict(category=k[0],message=k[1],count=n) for k,n in counts.items()]
    bv=validate_bootstrap(y,preds,lin,s['test'],seed=2026110704)
    validation=dict(passed=True,total_outer_models=180,new_outer_models=160,reused_nn64_outer_models=reused,all_23_lineages_in_both_outer_sets=True,identity_group_exclusion_checks=groups_checked,
        candidate_fold_scores_checked=scores_checked,prediction_max_error=maxp,inner_rmse_max_error=maxloss,neural_training_objectives_checked=len(objectives),neural_objective_max_error=maxobj,independent_selected_setting_refits=references,
        preflight=json.loads((OUT/'preflight_validation.json').read_text()),bootstrap_validation=bv,warning_count=sum(counts.values()),warning_summary=warning_summary,convergence_warning_count=sum(v['count'] for v in warning_summary if v['category']=='ConvergenceWarning'),
        input_source_hashes_verified=True,training_only_scaling_verified=True,unweighted_training_verified=True,byte_identical_shared_splits_verified=True)
    run.dump(OUT/'validation.json',validation);return preds,params,validation,protocol

def evaluate(preds,params,validation,protocol):
    s=run.STATE;y=s['y'];lin=s['lineage'];pooled,per,macro=score(y,preds,lin,s['test']);table('per_repeat_pooled_metrics.tsv',pooled);table('per_repeat_macro_metrics.tsv',macro);table('per_lineage_metrics.tsv',per)
    original=pd.read_csv(WS/'outputs/erastin_lineage_stratified_nn/per_repeat_pooled_metrics.tsv',sep='\t');np.testing.assert_allclose(pooled[pooled.model=='neural_network__64'].sort_values('repeat')[['spearman','rmse','r2']],original[original.model=='unweighted'].sort_values('repeat')[['spearman','rmse','r2']],atol=1e-12,rtol=0)
    vals={arm:{} for arm in preds};summary=[]
    for arm in preds:
        f,p=arm.rsplit('__',1);row=dict(model=arm,family=f,panel=int(p))
        for key in ['spearman','rmse','r2']:vals[arm]['pooled_'+key]=pooled[pooled.model==arm].sort_values('repeat')[key].to_numpy()
        for key in ['macro23_rmse','macro19_rmse']:vals[arm][key]=macro[macro.model==arm].sort_values('repeat')[key].to_numpy()
        for key,v in vals[arm].items():assert np.isfinite(v).all();row.update({key+'_mean':v.mean(),key+'_sd':v.std(ddof=1),key+'_min':v.min(),key+'_max':v.max()})
        summary.append(row)
    summary=table('summary.tsv',summary);run.dump(OUT/'progress.json',dict(phase='paired_bootstrap',audited_models=180))
    draws=bootstrap(y,preds,lin,s['test'],count=5000,seed=protocol['uncertainty']['seed']);paired=[];seedrows=[];effects=[];wins=[]
    for first,reference in pairs():
        for key in vals[first]:
            sign=-1 if key.endswith('rmse') else 1;delta=sign*(vals[first][key]-vals[reference][key]);lo,hi=np.percentile(sign*(draws[first][key]-draws[reference][key]),[2.5,97.5])
            paired.append(dict(first_model=first,reference_model=reference,metric=key,improvement=delta.mean(),favorable_repeats=int(np.sum(delta>0)),repeats=20,conditional_ci_low=lo,conditional_ci_high=hi));seedrows.extend(dict(first_model=first,reference_model=reference,metric=key,repeat=r+1,improvement=float(v)) for r,v in enumerate(delta))
        for label in np.unique(lin):
            a=per[(per.model==first)&(per.lineage==label)].sort_values('repeat');b=per[(per.model==reference)&(per.lineage==label)].sort_values('repeat');d=b.rmse.to_numpy()-a.rmse.to_numpy();effects.append(dict(first_model=first,reference_model=reference,lineage=str(label),test_n=int(a.test_n.iloc[0]),rmse_improvement=d.mean(),favorable_repeats=int(np.sum(d>0))))
        selected=[v for v in effects if v['first_model']==first and v['reference_model']==reference];wins.append(dict(first_model=first,reference_model=reference,rmse_favorable_lineages=sum(v['rmse_improvement']>0 for v in selected),lineages=23))
    paired=table('paired_comparisons.tsv',paired);table('paired_seed_differences.tsv',seedrows);effects=table('lineage_paired_effects.tsv',effects);wins=table('lineage_win_counts.tsv',wins)
    pcounts=[]
    for f in run.FAMILIES:
        cols=['alpha'] if f=='ridge' else ['width','alpha'] if f=='neural_network' else ['max_depth','min_child_weight','reg_lambda','num_boost_round'];pcounts.extend(params[params.family==f].groupby(['model','panel']+cols).size().reset_index(name='outer_fits').to_dict('records'))
    table('parameter_counts.tsv',pcounts);np.savez_compressed(OUT/'predictions.npz',sample_ids=s['ids'],group_ids=s['groups'],lineage=lin,y=y,outer_test=s['test'],outer_split_seeds=s['seeds'],**preds)
    np.savez_compressed(OUT/'bootstrap_draws.npz',**{arm+'__'+key:v for arm,d in draws.items() for key,v in d.items()})
    from report_panels import render
    render(summary,paired,effects,wins,validation)
    for a in json.loads((OUT/'source_manifest.json').read_text())['files']:assert run.sha(WS/a['path'])==a['sha256']
    am=json.loads((OUT/'analysis_source_manifest.json').read_text());assert run.sha(Path(__file__))==am['review_sha256'] and run.sha(OUT/'report_panels.py')==am['report_sha256']
    run.dump(OUT/'final_summary.json',dict(validation_passed=True,summary=summary.to_dict('records'),new_outer_models=160,reused_outer_models=20,total_outer_models=180));run.dump(OUT/'progress.json',dict(phase='complete',validation_passed=True,total_outer_models=180))
    manifest=dict(files=[dict(path=str(p.relative_to(OUT)),sha256=run.sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts]);run.dump(OUT/'artifact_manifest.json',manifest)
    archive=OUT.parent/'erastin_stratified_panel_comparison_results.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for row in manifest['files']:z.write(OUT/row['path'],'erastin_stratified_panel_comparison/'+row['path'])
        z.write(OUT/'artifact_manifest.json','erastin_stratified_panel_comparison/artifact_manifest.json')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in manifest['files']:assert hashlib.sha256(z.read('erastin_stratified_panel_comparison/'+row['path'])).hexdigest()==row['sha256']
    print(json.dumps(dict(event='complete',archive_bytes=archive.stat().st_size)),flush=True)
if __name__=='__main__':
    predictions,parameters,validation,protocol=audit();evaluate(predictions,parameters,validation,protocol)
