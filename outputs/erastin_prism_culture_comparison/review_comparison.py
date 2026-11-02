#!/usr/bin/env python3
"""Audit reused/new fits, then evaluate paired culture augmentation effects."""
import hashlib,json,time,zipfile
from pathlib import Path
from collections import Counter
import joblib,numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
import run_comparison as run
from fit_helpers import kkt
from validate_preflight import check_preprocessor,independent
from make_splits import inner_rows,outer_split,inner_folds
from evaluate import KEYS,score,bootstrap,validate_bootstrap,finite_mean
OUT=run.OUT;WS=run.WS

def table(name,rows):
    a=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows);a.to_csv(OUT/name,sep='\t',index=False,float_format='%.12g');return a
def dump(path,value):
    def safe(v):
        if isinstance(v,dict):return {str(k):safe(x) for k,x in v.items()}
        if isinstance(v,np.ndarray) and v.ndim==0:return safe(v.item())
        if isinstance(v,(list,tuple,np.ndarray)):return [safe(x) for x in v]
        if isinstance(v,np.generic):return safe(v.item())
        if isinstance(v,float) and not np.isfinite(v):return None
        return v
    run.dump(path,safe(value))
def arms():return [(f,p,v) for f in run.FAMILIES for v in ['expression','culture'] for p in ([5,64] if v=='expression' else [0,5,64])]
def pairs(p):return p['primary_comparisons']+p['secondary_comparisons']+[[run.arm_name(f,n,v),'training_mean'] for f,n,v in arms()]

def audit():
    run.initialize();s=run.STATE;y=s['y'];lin=s['lineage'];groups=s['groups'];protocol=json.load(open(OUT/'protocol.json'));ph=run.sha(OUT/'protocol.json');assert json.load(open(OUT/'fit_completion.json'))['complete'];sourcevalidation=json.load(open(run.SOURCE/'validation.json'));assert sourcevalidation['passed']
    for name,h in protocol['scripts'].items():assert run.sha(OUT/name)==h
    for name,h in protocol['data_sha256'].items():assert run.sha(OUT/name)==h
    for row in json.load(open(OUT/'source_manifest.json'))['files']:assert run.sha(WS/row['path'])==row['sha256']
    dump(OUT/'analysis_source_manifest.json',dict(review_sha256=run.sha(Path(__file__)),report_sha256=run.sha(OUT/'report_comparison.py')))
    preds={run.arm_name(f,p,v):np.full(s['test'].shape,np.nan) for f,p,v in arms()};preds['training_mean']=np.full(s['test'].shape,np.nan);params=[];tuning=[];warnings=[];objectives=[];solver=[];refs=[];preprocessing=[];coefficients=[];maxp=maxloss=maxobj=maxkkt=0.;scores_checked=groups_checked=prepchecks=0;oldpred=np.load(run.SOURCE/'predictions.npz');genes=np.load(OUT/'panel_inputs.npz')
    for r in range(20):
        np.testing.assert_array_equal(s['test'][r],outer_split(lin,groups,s['seeds'][r]));np.testing.assert_array_equal(s['inner'][r],inner_folds(lin,groups,s['test'][r],s['inner_seeds'][r]));tr=np.flatnonzero(~s['test'][r]);te=np.flatnonzero(s['test'][r]);inner=s['inner'][r];assert (len(tr),len(te))==(308,78);preds['training_mean'][r,te]=y[tr].mean()
        for f,p,v in arms():
            arm=run.arm_name(f,p,v);root=run.root_for(f,p,v,r);m=json.load(open(root.with_suffix('.json')));a=np.load(root.with_suffix('.npz'));model=joblib.load(root.with_suffix('.joblib'));grid=run.grid_for(f);c=m['selected_candidate'];chosen=grid[c];x=s['x'][p]
            assert m['protocol_sha256']==ph and (m['family'],m['panel'],m['view'],m['repeat'])==(f,p,v,r+1) and m['reused']==(v=='expression');assert m['train_n']==308 and m['test_n']==78 and m['train_lineages']==m['test_lineages']==20
            assert m['outer_split_seed']==int(s['seeds'][r]) and m['inner_split_seed']==int(s['inner_seeds'][r]);np.testing.assert_array_equal(m['neural_seeds'],s['nn_seeds'][r]);np.testing.assert_array_equal(a['train_rows'],tr);np.testing.assert_array_equal(a['test_rows'],te);assert set(groups[tr]).isdisjoint(groups[te]) and set(lin[tr])==set(lin[te])==set(lin);groups_checked+=1
            for ext in ['npz','joblib']:assert run.sha(root.with_suffix('.'+ext))==m[ext+'_sha256']
            assert chosen==m['selected_parameters'] and c==np.argmin(a['inner_fold_rmse'].mean(axis=1));assert abs(m['mean_inner_rmse']-a['inner_fold_rmse'][c].mean())<1e-12;assert np.isfinite(a['inner_validation_predictions'][:,inner>=0]).all() and np.isnan(a['inner_validation_predictions'][:,inner<0]).all()
            for k in range(5):
                aa,bb=inner_rows(inner,k);assert set(aa)|set(bb)==set(tr) and set(groups[aa]).isdisjoint(groups[bb]) and set(lin[aa])==set(lin);groups_checked+=1;loss=np.sqrt(np.mean((a['inner_validation_predictions'][:,bb]-y[bb][None,:])**2,axis=1));maxloss=max(maxloss,float(np.max(np.abs(loss-a['inner_fold_rmse'][:,k]))));scores_checked+=len(grid)
                if v=='expression':assert m['inner_scaling'][k]['y_mean']==y[aa].mean() and m['inner_scaling'][k]['y_scale']==y[aa].std()
                else:
                    _,_,prep=check_preprocessor(p,aa,bb)
                    recorded=dict(m['preprocessing'][k]);assert recorded.pop('fold')==k;assert run.clean(prep.diagnostic(s['meta'].iloc[bb]))==recorded;prepchecks+=1
            if v=='culture':
                z,zz,prep=check_preprocessor(p,tr,te,model['preprocessor']);assert run.clean(prep.diagnostic(s['meta'].iloc[te]))==m['outer_preprocessing'];prepchecks+=1;names=[str(g) for g in genes['genes_'+str(p)]] if p else [];names+=prep.feature_names[p:];preprocessing.append(dict(model=arm,repeat=r+1,**m['outer_preprocessing']))
            else:
                sx=StandardScaler().fit(x[tr]);z=sx.transform(x[tr]);zz=sx.transform(x[te]);names=[str(g) for g in genes['genes_'+str(p)]]
                if f=='ridge':np.testing.assert_allclose(model['mean'],sx.mean_,atol=1e-12,rtol=0);np.testing.assert_allclose(model['scale'],sx.scale_,atol=1e-12,rtol=0)
                else:
                    np.testing.assert_allclose(model['x_scaler'].mean_,sx.mean_,atol=1e-12,rtol=0);np.testing.assert_allclose(model['x_scaler'].scale_,sx.scale_,atol=1e-12,rtol=0)
            if f=='ridge':
                assert model['alpha']==chosen['alpha'];intercept=model['intercept'] if v=='culture' else model['intercept_after_scaling'];assert intercept==y[tr].mean();pred=zz@model['coef']+intercept;coef=model['coef']
            else:
                sy=StandardScaler().fit(y[tr,None]);target=sy.transform(y[tr,None]).ravel();np.testing.assert_allclose(model['y_scaler'].mean_,sy.mean_,atol=1e-12,rtol=0);np.testing.assert_allclose(model['y_scaler'].scale_,sy.scale_,atol=1e-12,rtol=0)
                if f=='neural_network':
                    assert [n.random_state for n in model['members']]==m['neural_seeds'];memberpred=np.array([sy.inverse_transform(n.predict(zz)[:,None]).ravel() for n in model['members']]);np.testing.assert_allclose(memberpred,a['member_test_predictions'],atol=1e-12,rtol=0);pred=memberpred.mean(axis=0)
                    for j,n in enumerate(model['members']):
                        assert n.alpha==chosen['alpha'] and n.hidden_layer_sizes==(chosen['width'],) and n.solver=='lbfgs' and n.activation=='tanh' and not n.early_stopping
                        if v=='expression':np.testing.assert_array_equal(n.training_sample_weight_,np.ones(len(tr)))
                        obj=.5*np.mean((n.predict(z)-target)**2)+.5*n.alpha*sum(np.sum(w*w) for w in n.coefs_)/len(tr);err=float(abs(obj-n.loss_));maxobj=max(maxobj,err);assert err<1e-10;objectives.append(dict(model=arm,repeat=r+1,member=j,loss_error=err,iterations=n.n_iter_))
                else:
                    n=model['model'];assert n.alpha==chosen['alpha'] and n.l1_ratio==chosen['l1_ratio'];pred=sy.inverse_transform(n.predict(zz)[:,None]).ravel();error=kkt(z,target,n.coef_,n.intercept_,n.alpha,n.l1_ratio);maxkkt=max(maxkkt,error);assert error<1e-6;obj=.5*np.mean((n.predict(z)-target)**2)+n.alpha*n.l1_ratio*np.abs(n.coef_).sum()+.5*n.alpha*(1-n.l1_ratio)*np.sum(n.coef_**2);assert abs(obj-m['fit_diagnostics'][-1]['objective'])<1e-12;coef=n.coef_*sy.scale_[0]
                    for d in m['fit_diagnostics']:
                        if d['stage']=='inner':
                            assert len(d['kkt_residuals'])==len(d['dual_gaps'])==len(d['iterations'])==41;assert d['max_kkt_residual']<1e-6 and abs(d['max_kkt_residual']-max(d['kkt_residuals']))<1e-15;maxkkt=max(maxkkt,max(d['kkt_residuals']));solver.append(dict(model=arm,repeat=r+1,stage='inner',fold=d['fold'],max_kkt_residual=d['max_kkt_residual'],max_iterations=max(d['iterations'])))
                        else:solver.append(dict(model=arm,repeat=r+1,stage='outer',fold=-1,max_kkt_residual=error,max_iterations=d['iterations']))
            if f!='neural_network':coefficients.extend(dict(model=arm,repeat=r+1,feature=name,coefficient_original_target_per_model_feature=float(b),culture_feature=j>=p,selected=bool(b!=0)) for j,(name,b) in enumerate(zip(names,coef)))
            error=float(np.max(np.abs(pred-a['test_prediction'])));maxp=max(maxp,error);assert error<1e-10;preds[arm][r,te]=pred
            if v=='expression':np.testing.assert_allclose(pred,oldpred[f+'__'+str(p)][r,te],atol=1e-12,rtol=0)
            if v=='culture' and r in [0,19]:
                ref=independent(f,chosen,z,y[tr],zz,s['nn_seeds'][r]);outererr=float(np.max(np.abs(ref-pred)));innererr=0.
                for k in range(5):
                    aa,bb=inner_rows(inner,k);z1,z2,_=check_preprocessor(p,aa,bb);ref=independent(f,chosen,z1,y[aa],z2,s['nn_seeds'][r]);innererr=max(innererr,float(np.max(np.abs(ref-a['inner_validation_predictions'][c,bb]))))
                assert max(outererr,innererr)<1e-7;refs.append(dict(model=arm,repeat=r+1,outer_error=outererr,inner_error=innererr,inner_refits=5))
            for d in m['fit_diagnostics']:
                warnings.extend(dict(model=arm,repeat=r+1,stage=d['stage'],**w) for w in d.get('warnings',[]))
                for n in d.get('members',[]):warnings.extend(dict(model=arm,repeat=r+1,stage=d['stage'],**w) for w in n['warnings'])
            params.append(dict(model=arm,family=f,panel=p,view=v,repeat=r+1,candidate=c,**chosen,mean_inner_rmse=m['mean_inner_rmse']));tuning.extend(dict(model=arm,repeat=r+1,candidate=j,**cp,mean_inner_rmse=float(a['inner_fold_rmse'][j].mean())) for j,cp in enumerate(grid))
        if r%5==4:print('AUDITED_REPEATS',r+1,flush=True)
    assert groups_checked==1800 and scores_checked==176500 and prepchecks==1080 and len(objectives)==300 and len(refs)==18 and maxloss<1e-12
    for arm,p in preds.items():assert np.isfinite(p[s['test']]).all() and np.isnan(p[~s['test']]).all()
    params=table('selected_parameters.tsv',params);table('tuning_scores.tsv',tuning);table('neural_objective_checks.tsv',objectives);table('elastic_net_solver_checks.tsv',solver);table('linear_coefficients.tsv',coefficients);table('outer_preprocessing.tsv',preprocessing);counts=Counter((w['category'],w['message']) for w in warnings);ws=[dict(category=k[0],message=k[1],count=n) for k,n in counts.items()];bv=validate_bootstrap(y,preds,lin,s['test'],2026111402)
    validation=dict(passed=True,total_outer_models=300,new_outer_models=180,reused_outer_models=120,all_20_lineages_in_both_outer_sets=True,identity_exclusion_checks=groups_checked,candidate_fold_scores_checked=scores_checked,training_only_preprocessor_checks=prepchecks,prediction_max_error=maxp,inner_rmse_max_error=maxloss,elastic_net_max_kkt_residual=maxkkt,elastic_net_inner_candidate_solutions_kkt_checked=164000,elastic_net_outer_objectives_checked=100,neural_objectives_checked=300,neural_objective_max_error=maxobj,independent_selected_refits=refs,warning_summary=ws,convergence_warning_count=sum(w['count'] for w in ws if w['category']=='ConvergenceWarning'),selected_outer_convergence_warning_count=sum(w['category']=='ConvergenceWarning' and w['stage']=='outer' for w in warnings),bootstrap_validation=bv,preflight=json.load(open(OUT/'preflight_validation.json')),source_baseline_validation_passed=True,source_hashes_verified=True,baseline_models_and_predictions_unchanged=True,shared_splits_verified=True,unweighted_training_verified=True)
    dump(OUT/'validation.json',validation);return preds,params,validation,protocol

def evaluate(preds,params,validation,protocol):
    s=run.STATE;y=s['y'];lin=s['lineage'];pooled,per,macro=score(y,preds,lin,s['test']);table('per_repeat_pooled_metrics.tsv',pooled);table('per_repeat_macro_metrics.tsv',macro);table('per_lineage_metrics.tsv',per);vals={arm:{} for arm in preds};summary=[];descriptions={run.arm_name(f,p,v):(f,p,v) for f,p,v in arms()}
    for arm in preds:
        f,p,v=descriptions.get(arm,('training_mean',0,'constant'));row=dict(model=arm,family=f,panel=p,view=v)
        for key in ['spearman','pearson','rmse','mae','r2']:vals[arm]['pooled_'+key]=pooled[pooled.model==arm].sort_values('repeat')[key].to_numpy()
        for key in ['macro_all_rmse','macro_large_rmse']:vals[arm][key]=macro[macro.model==arm].sort_values('repeat')[key].to_numpy()
        for key,a in vals[arm].items():
            valid=a[np.isfinite(a)];row.update({key+'_mean':float(finite_mean(a)),key+'_sd':valid.std(ddof=1) if len(valid)>1 else np.nan,key+'_defined_repeats':len(valid)})
        summary.append(row)
    summary=table('summary.tsv',summary);old=pd.read_csv(run.SOURCE/'summary.tsv',sep='\t').set_index('model')
    for f,p,v in arms():
        if v=='expression':
            a=summary[summary.model==run.arm_name(f,p,v)].iloc[0];b=old.loc[f+'__'+str(p)]
            for key in KEYS:assert abs(a[key+'_mean']-b[key+'_mean'])<1e-11
    dump(OUT/'progress.json',dict(phase='paired_bootstrap',audited_models=300));draws,perdraw=bootstrap(y,preds,lin,s['test'],count=5000,seed=protocol['uncertainty']['seed']);paired=[];seedrows=[];effects=[];wins=[]
    for first,ref in pairs(protocol):
        for key in KEYS:
            sign=-1 if key.endswith('rmse') or key.endswith('mae') else 1;delta=sign*(vals[first][key]-vals[ref][key]);bd=finite_mean(sign*(perdraw[first][key]-perdraw[ref][key]),axis=1);valid=bd[np.isfinite(bd)];lo,hi=np.percentile(valid,[2.5,97.5]) if len(valid) else (np.nan,np.nan);paired.append(dict(first_model=first,reference_model=ref,metric=key,improvement=float(finite_mean(delta)),favorable_repeats=int(np.sum(delta>0)),paired_defined_repeats=int(np.isfinite(delta).sum()),conditional_ci_low=lo,conditional_ci_high=hi,valid_bootstrap_draws=len(valid)));seedrows.extend(dict(first_model=first,reference_model=ref,metric=key,repeat=r+1,improvement=float(v)) for r,v in enumerate(delta))
        for label in np.unique(lin):
            a=per[(per.model==first)&(per.lineage==label)].sort_values('repeat');b=per[(per.model==ref)&(per.lineage==label)].sort_values('repeat');d=b.rmse.to_numpy()-a.rmse.to_numpy();effects.append(dict(first_model=first,reference_model=ref,lineage=str(label),cohort_n=int(a.cohort_n.iloc[0]),test_n=int(a.test_n.iloc[0]),rmse_improvement=d.mean(),favorable_repeats=int(np.sum(d>0))))
        selected=[w for w in effects if w['first_model']==first and w['reference_model']==ref];wins.append(dict(first_model=first,reference_model=ref,rmse_favorable_lineages=sum(w['rmse_improvement']>0 for w in selected),lineages=20))
    paired=table('paired_comparisons.tsv',paired);table('paired_seed_differences.tsv',seedrows);effects=table('lineage_paired_effects.tsv',effects);wins=table('lineage_win_counts.tsv',wins);pcounts=[]
    for f in run.FAMILIES:
        cols=['alpha','l1_ratio'] if f=='elastic_net' else ['alpha'] if f=='ridge' else ['width','alpha'];pcounts.extend(params[params.family==f].groupby(['model','panel','view']+cols).size().reset_index(name='outer_fits').to_dict('records'))
    table('parameter_counts.tsv',pcounts);np.savez_compressed(OUT/'predictions.npz',sample_ids=s['ids'],group_ids=s['groups'],lineage=lin,y=y,outer_test=s['test'],outer_split_seeds=s['seeds'],**preds);np.savez_compressed(OUT/'bootstrap_draws.npz',**{arm+'__'+key:v for arm,d in draws.items() for key,v in d.items()})
    from report_comparison import render
    render(summary,paired,effects,wins,validation)
    for row in json.load(open(OUT/'source_manifest.json'))['files']:assert run.sha(WS/row['path'])==row['sha256']
    am=json.load(open(OUT/'analysis_source_manifest.json'));assert run.sha(Path(__file__))==am['review_sha256'] and run.sha(OUT/'report_comparison.py')==am['report_sha256']
    dump(OUT/'final_summary.json',dict(validation_passed=True,summary=summary.to_dict('records'),new_outer_models=180,reused_outer_models=120,total_outer_models=300));dump(OUT/'progress.json',dict(phase='analysis_complete_pending_visual_qa',validation_passed=True,total_outer_models=300));print('ANALYSIS_COMPLETE',flush=True)

def seal():
    assert json.load(open(OUT/'visual_qa.json'))['passed'];dump(OUT/'progress.json',dict(phase='complete',validation_passed=True,total_outer_models=300));manifest=dict(files=[dict(path=str(p.relative_to(OUT)),sha256=run.sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts]);dump(OUT/'artifact_manifest.json',manifest);archive=OUT.parent/'erastin_prism_culture_comparison_results.zip';prefix=OUT.name+'/'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for row in manifest['files']:z.write(OUT/row['path'],prefix+row['path'])
        z.write(OUT/'artifact_manifest.json',prefix+'artifact_manifest.json')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in manifest['files']:assert hashlib.sha256(z.read(prefix+row['path'])).hexdigest()==row['sha256']
    print(json.dumps(dict(event='complete',archive_bytes=archive.stat().st_size,files=len(manifest['files'])+1)),flush=True)
if __name__=='__main__':
    predictions,parameters,validation,protocol=audit();evaluate(predictions,parameters,validation,protocol)
