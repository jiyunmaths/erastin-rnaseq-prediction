#!/usr/bin/env python3
"""Audit all retained/new gene-only models and evaluate paired module additions."""
import hashlib,json,zipfile
from pathlib import Path
from collections import Counter
import joblib,numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
import run_comparison as run
from enet_helpers import kkt_residual
from validate_preflight import independent
from make_splits import inner_rows,outer_split,inner_folds
from evaluate import KEYS,score,bootstrap,validate_bootstrap,finite_mean
OUT=run.OUT;WS=run.WS
def table(name,v):
    d=v if isinstance(v,pd.DataFrame) else pd.DataFrame(v);d.to_csv(OUT/name,sep='\t',index=False,float_format='%.12g');return d
def dump(path,value):
    def safe(v):
        if isinstance(v,dict):return {str(k):safe(a) for k,a in v.items()}
        if isinstance(v,np.ndarray) and v.ndim==0:return safe(v.item())
        if isinstance(v,(list,tuple,np.ndarray)):return [safe(a) for a in v]
        if isinstance(v,np.generic):return safe(v.item())
        if isinstance(v,float) and not np.isfinite(v):return None
        return v
    run.dump(path,safe(value))
def pairs(p):return p['primary_comparisons']+p['secondary_comparisons']+[[run.arm_name(f,n),'training_mean'] for f in run.FAMILIES for n in run.PANELS]

def audit():
    run.initialize();s=run.STATE;y=s['y'];lin=s['lineage'];g=s['groups'];protocol=json.load(open(OUT/'protocol.json'));ph=run.sha(OUT/'protocol.json');assert json.load(open(OUT/'fit_completion.json'))['complete'] and json.load(open(run.SOURCE/'validation.json'))['passed']
    for n,h in protocol['scripts'].items():assert run.sha(OUT/n)==h
    for n,h in protocol['data_sha256'].items():assert run.sha(OUT/n)==h
    for row in json.load(open(OUT/'source_manifest.json'))['files']:assert run.sha(WS/row['path'])==row['sha256']
    rawcheck=json.load(open(OUT/'raw_expression_validation.json'));assert rawcheck['passed'] and rawcheck['max_error']==0. and rawcheck['entries_checked']==307272
    dump(OUT/'analysis_source_manifest.json',dict(review_sha256=run.sha(Path(__file__)),report_sha256=run.sha(OUT/'report_comparison.py'),raw_expression_validation_sha256=run.sha(OUT/'raw_expression_validation.json'),raw_expression_validation_script_sha256=run.sha(OUT/'verify_expression.py')))
    preds={run.arm_name(f,p):np.full(s['test'].shape,np.nan) for f in run.FAMILIES for p in run.PANELS};preds['training_mean']=np.full(s['test'].shape,np.nan);params=[];tuning=[];warnings=[];objectives=[];solver=[];refs=[];coefficients=[];maxp=maxloss=maxobj=maxkkt=0.;groups_checked=scores_checked=0;oldpred=np.load(run.SOURCE/'predictions.npz');inputs=np.load(OUT/'panel_inputs.npz');mapping=pd.read_csv(OUT/'gene_mapping.tsv',sep='\t',dtype=str);module_ids={v:set(mapping[mapping.module==v].entrez_id) for v in ['emt','ddr']}
    for r in range(20):
        np.testing.assert_array_equal(s['test'][r],outer_split(lin,g,s['seeds'][r]));np.testing.assert_array_equal(s['inner'][r],inner_folds(lin,g,s['test'][r],s['inner_seeds'][r]));tr=np.flatnonzero(~s['test'][r]);te=np.flatnonzero(s['test'][r]);inner=s['inner'][r];assert (len(tr),len(te))==(595,149);preds['training_mean'][r,te]=y[tr].mean()
        for p in run.PANELS:
            x=s['x'][p];genes=s['genes'][p];gene_ids=inputs['entrez_'+p]
            for f in run.FAMILIES:
                arm=run.arm_name(f,p);root=run.root_for(f,p,r);m=json.load(open(root.with_suffix('.json')));a=np.load(root.with_suffix('.npz'));model=joblib.load(root.with_suffix('.joblib'));grid=run.grid_for(f);c=m['selected_candidate'];selected=grid[c]
                assert m['protocol_sha256']==ph and (m['family'],m['panel'],m['repeat'])==(f,p,r+1) and m['gene_count']==x.shape[1] and m['reused']==(p not in run.NEW);assert m['train_n']==595 and m['test_n']==149 and m['train_lineages']==m['test_lineages']==23
                assert m['outer_split_seed']==int(s['seeds'][r]) and m['inner_split_seed']==int(s['inner_seeds'][r]);np.testing.assert_array_equal(m['neural_seeds'],s['nn_seeds'][r]);np.testing.assert_array_equal(a['train_rows'],tr);np.testing.assert_array_equal(a['test_rows'],te);assert set(g[tr]).isdisjoint(g[te]) and set(lin[tr])==set(lin[te])==set(lin);groups_checked+=1
                for ext in ['npz','joblib']:assert run.sha(root.with_suffix('.'+ext))==m[ext+'_sha256']
                assert selected==m['selected_parameters'] and c==np.argmin(a['inner_fold_rmse'].mean(axis=1));assert abs(m['mean_inner_rmse']-a['inner_fold_rmse'][c].mean())<1e-12;assert np.isfinite(a['inner_validation_predictions'][:,inner>=0]).all() and np.isnan(a['inner_validation_predictions'][:,inner<0]).all()
                for k in range(5):
                    aa,bb=inner_rows(inner,k);assert set(aa)|set(bb)==set(tr) and set(g[aa]).isdisjoint(g[bb]) and set(lin[aa])==set(lin);groups_checked+=1;assert m['inner_scaling'][k]['y_mean']==y[aa].mean() and m['inner_scaling'][k]['y_scale']==y[aa].std();loss=np.sqrt(np.mean((a['inner_validation_predictions'][:,bb]-y[bb][None,:])**2,axis=1));maxloss=max(maxloss,float(np.max(np.abs(loss-a['inner_fold_rmse'][:,k]))));scores_checked+=len(grid)
                sx=StandardScaler().fit(x[tr]);z=sx.transform(x[tr]);zz=sx.transform(x[te]);nonzero=np.nan
                if f=='ridge':
                    np.testing.assert_allclose(model['mean'],sx.mean_,atol=1e-12,rtol=0);np.testing.assert_allclose(model['scale'],sx.scale_,atol=1e-12,rtol=0);assert model['alpha']==selected['alpha'] and model['intercept_after_scaling']==y[tr].mean();pred=zz@model['coef']+model['intercept_after_scaling']
                else:
                    sy=StandardScaler().fit(y[tr,None]);target=sy.transform(y[tr,None]).ravel()
                    for n in ['mean_','scale_']:np.testing.assert_allclose(getattr(model['x_scaler'],n),getattr(sx,n),atol=1e-12,rtol=0);np.testing.assert_allclose(getattr(model['y_scaler'],n),getattr(sy,n),atol=1e-12,rtol=0)
                    if f=='neural_network':
                        assert [n.random_state for n in model['members']]==m['neural_seeds'];members=np.array([sy.inverse_transform(n.predict(zz)[:,None]).ravel() for n in model['members']]);np.testing.assert_allclose(members,a['member_test_predictions'],atol=1e-12,rtol=0);pred=members.mean(axis=0)
                        for j,n in enumerate(model['members']):
                            np.testing.assert_array_equal(n.training_sample_weight_,np.ones(len(tr)));assert n.alpha==selected['alpha'] and n.hidden_layer_sizes==(selected['width'],) and n.activation=='tanh' and n.solver=='lbfgs' and not n.early_stopping;obj=.5*np.mean((n.predict(z)-target)**2)+.5*n.alpha*sum(np.sum(w*w) for w in n.coefs_)/len(tr);error=float(abs(obj-n.loss_));maxobj=max(maxobj,error);assert error<1e-10;objectives.append(dict(model=arm,repeat=r+1,member=j,objective_error=error,iterations=n.n_iter_))
                    else:
                        n=model['model'];assert n.alpha==selected['alpha'] and n.l1_ratio==selected['l1_ratio'];pred=sy.inverse_transform(n.predict(zz)[:,None]).ravel();error=kkt_residual(z,target,n.coef_,n.intercept_,n.alpha,n.l1_ratio);maxkkt=max(maxkkt,error);assert error<1e-6;obj=.5*np.mean((n.predict(z)-target)**2)+n.alpha*n.l1_ratio*np.abs(n.coef_).sum()+.5*n.alpha*(1-n.l1_ratio)*np.sum(n.coef_**2);assert abs(obj-m['fit_diagnostics'][-1]['objective'])<1e-12;nonzero=int(np.count_nonzero(n.coef_));coefficients.extend(dict(model=arm,panel=p,repeat=r+1,gene=str(gene),entrez_id=str(ent),in_Hallmark_emt=str(ent) in module_ids['emt'],in_Hallmark_ddr=str(ent) in module_ids['ddr'],selected=bool(beta!=0),coefficient_per_training_expression_sd=float(beta*sy.scale_[0])) for gene,ent,beta in zip(genes,gene_ids,n.coef_))
                        for d in m['fit_diagnostics']:
                            if d['stage']=='inner':
                                assert len(d['kkt_residuals'])==len(d['dual_gaps'])==len(d['iterations'])==41;assert d['max_kkt_residual']<1e-6;maxkkt=max(maxkkt,max(d['kkt_residuals']));solver.append(dict(model=arm,repeat=r+1,stage='inner',fold=d['fold'],max_kkt_residual=d['max_kkt_residual'],max_iterations=max(d['iterations'])))
                            else:solver.append(dict(model=arm,repeat=r+1,stage='outer',fold=-1,max_kkt_residual=error,max_iterations=d['iterations']))
                error=float(np.max(np.abs(pred-a['test_prediction'])));maxp=max(maxp,error);assert error<1e-10;preds[arm][r,te]=pred
                if p not in run.NEW:np.testing.assert_allclose(pred,oldpred[f+'__'+str(x.shape[1])][r,te],atol=1e-12,rtol=0)
                if p in run.NEW and r in [0,19]:
                    ref=independent(f,selected,x,y,tr,te,s['nn_seeds'][r]);outererror=float(np.max(np.abs(ref-pred)));innererror=0.
                    for k in range(5):
                        aa,bb=inner_rows(inner,k);ref=independent(f,selected,x,y,aa,bb,s['nn_seeds'][r]);innererror=max(innererror,float(np.max(np.abs(ref-a['inner_validation_predictions'][c,bb]))))
                    assert max(outererror,innererror)<1e-7;refs.append(dict(model=arm,repeat=r+1,outer_error=outererror,inner_error=innererror,inner_refits=5))
                for d in m['fit_diagnostics']:
                    warnings.extend(dict(model=arm,repeat=r+1,stage=d['stage'],**w) for w in d.get('warnings',[]))
                    for n in d.get('members',[]):warnings.extend(dict(model=arm,repeat=r+1,stage=d['stage'],**w) for w in n['warnings'])
                params.append(dict(model=arm,family=f,panel=p,gene_count=x.shape[1],repeat=r+1,candidate=c,**selected,mean_inner_rmse=m['mean_inner_rmse'],nonzero_genes=nonzero));tuning.extend(dict(model=arm,repeat=r+1,candidate=j,**setting,mean_inner_rmse=float(a['inner_fold_rmse'][j].mean())) for j,setting in enumerate(grid))
        if r%5==4:print('AUDITED_REPEATS',r+1,flush=True)
    assert groups_checked==1800 and scores_checked==174500 and len(objectives)==300 and len(refs)==18 and maxloss<1e-12
    params=table('selected_parameters.tsv',params);table('tuning_scores.tsv',tuning);table('neural_objective_checks.tsv',objectives);table('elastic_net_solver_checks.tsv',solver);coef=table('elastic_net_coefficients.tsv',coefficients);selection=[]
    for (panel,gene),d in coef.groupby(['panel','gene']):selection.append(dict(panel=panel,gene=gene,selected_fits=int(d.selected.sum()),total_fits=20,mean_absolute_coefficient=np.abs(d.coefficient_per_training_expression_sd).mean()))
    table('elastic_net_selection.tsv',selection);counts=Counter((w['category'],w['message']) for w in warnings);ws=[dict(category=k[0],message=k[1],count=n) for k,n in counts.items()];bv=validate_bootstrap(y,preds,lin,s['test'],2026111602)
    validation=dict(passed=True,total_outer_models=300,new_outer_models=180,reused_outer_models=120,all_23_lineages_in_both_outer_sets=True,identity_exclusion_checks=groups_checked,candidate_fold_scores_checked=scores_checked,prediction_max_error=maxp,inner_rmse_max_error=maxloss,elastic_net_max_kkt_residual=maxkkt,elastic_net_inner_candidate_solutions_kkt_checked=164000,elastic_net_outer_objectives_checked=100,neural_objectives_checked=300,neural_objective_max_error=maxobj,independent_selected_refits=refs,warning_summary=ws,convergence_warning_count=sum(w['count'] for w in ws if w['category']=='ConvergenceWarning'),selected_outer_convergence_warning_count=sum(w['category']=='ConvergenceWarning' and w['stage']=='outer' for w in warnings),bootstrap_validation=bv,preflight=json.load(open(OUT/'preflight_validation.json')),source_hashes_verified=True,baseline_models_and_predictions_unchanged=True,shared_splits_verified=True,unweighted_training_verified=True,training_only_scalers_verified=True,culture_predictors=False)
    dump(OUT/'validation.json',validation);return preds,params,validation,protocol

def evaluate(preds,params,validation,protocol):
    s=run.STATE;y=s['y'];lin=s['lineage'];pooled,per,macro=score(y,preds,lin,s['test']);table('per_repeat_pooled_metrics.tsv',pooled);table('per_repeat_macro_metrics.tsv',macro);table('per_lineage_metrics.tsv',per);values={a:{} for a in preds};rows=[]
    for arm in preds:
        f,p=arm.split('__',1) if arm!='training_mean' else ('training_mean','none');row=dict(model=arm,family=f,panel=p,gene_count=s['x'][p].shape[1] if p in s['x'] else 0)
        for key in ['spearman','pearson','rmse','mae','r2']:values[arm]['pooled_'+key]=pooled[pooled.model==arm].sort_values('repeat')[key].to_numpy()
        for key in ['macro_all_rmse','macro_large_rmse']:values[arm][key]=macro[macro.model==arm].sort_values('repeat')[key].to_numpy()
        for key,v in values[arm].items():
            ok=v[np.isfinite(v)];row.update({key+'_mean':float(finite_mean(v)),key+'_sd':ok.std(ddof=1) if len(ok)>1 else np.nan,key+'_defined_repeats':len(ok)})
        rows.append(row)
    summary=table('summary.tsv',rows);old=pd.read_csv(run.SOURCE/'summary.tsv',sep='\t').set_index('model')
    for f in run.FAMILIES:
        for p in ['ferro5','ferro64']:
            a=summary[summary.model==run.arm_name(f,p)].iloc[0];b=old.loc[f+'__'+str(s['x'][p].shape[1])]
            for key in ['pooled_spearman','pooled_rmse','pooled_r2','macro_all_rmse','macro_large_rmse']:
                oldkey={'macro_all_rmse':'macro23_rmse','macro_large_rmse':'macro19_rmse'}.get(key,key);assert abs(a[key+'_mean']-b[oldkey+'_mean'])<1e-11
    dump(OUT/'progress.json',dict(phase='paired_bootstrap',audited_models=300));draws,perdraw=bootstrap(y,preds,lin,s['test'],5000,protocol['uncertainty']['seed']);paired=[];seedrows=[];effects=[];wins=[]
    for first,ref in pairs(protocol):
        for key in KEYS:
            sign=-1 if key.endswith('rmse') or key.endswith('mae') else 1;delta=sign*(values[first][key]-values[ref][key]);d=finite_mean(sign*(perdraw[first][key]-perdraw[ref][key]),axis=1);valid=d[np.isfinite(d)];lo,hi=np.percentile(valid,[2.5,97.5]) if len(valid) else (np.nan,np.nan);paired.append(dict(first_model=first,reference_model=ref,metric=key,improvement=float(finite_mean(delta)),favorable_repeats=int(np.sum(delta>0)),paired_defined_repeats=int(np.isfinite(delta).sum()),conditional_ci_low=lo,conditional_ci_high=hi,valid_bootstrap_draws=len(valid)));seedrows.extend(dict(first_model=first,reference_model=ref,metric=key,repeat=r+1,improvement=float(v)) for r,v in enumerate(delta))
        for label in np.unique(lin):
            a=per[(per.model==first)&(per.lineage==label)].sort_values('repeat');b=per[(per.model==ref)&(per.lineage==label)].sort_values('repeat');d=b.rmse.to_numpy()-a.rmse.to_numpy();effects.append(dict(first_model=first,reference_model=ref,lineage=str(label),cohort_n=int(a.cohort_n.iloc[0]),test_n=int(a.test_n.iloc[0]),rmse_improvement=d.mean(),favorable_repeats=int(np.sum(d>0))))
        chosen=[v for v in effects if v['first_model']==first and v['reference_model']==ref];wins.append(dict(first_model=first,reference_model=ref,rmse_favorable_lineages=sum(v['rmse_improvement']>0 for v in chosen),lineages=23))
    paired=table('paired_comparisons.tsv',paired);table('paired_seed_differences.tsv',seedrows);effects=table('lineage_paired_effects.tsv',effects);wins=table('lineage_win_counts.tsv',wins);pc=[]
    for f in run.FAMILIES:
        cols=['alpha','l1_ratio'] if f=='elastic_net' else ['alpha'] if f=='ridge' else ['width','alpha'];pc.extend(params[params.family==f].groupby(['model','panel','gene_count']+cols).size().reset_index(name='outer_fits').to_dict('records'))
    table('parameter_counts.tsv',pc);np.savez_compressed(OUT/'predictions.npz',sample_ids=s['ids'],group_ids=s['groups'],lineage=lin,y=y,outer_test=s['test'],**preds);np.savez_compressed(OUT/'bootstrap_draws.npz',**{arm+'__'+key:v for arm,d in draws.items() for key,v in d.items()})
    from report_comparison import render
    render(summary,paired,effects,wins,validation)
    for row in json.load(open(OUT/'source_manifest.json'))['files']:assert run.sha(WS/row['path'])==row['sha256']
    am=json.load(open(OUT/'analysis_source_manifest.json'));assert run.sha(Path(__file__))==am['review_sha256'] and run.sha(OUT/'report_comparison.py')==am['report_sha256'];dump(OUT/'final_summary.json',dict(validation_passed=True,summary=summary.to_dict('records'),new_outer_models=180,reused_outer_models=120,total_outer_models=300));dump(OUT/'progress.json',dict(phase='analysis_complete_pending_visual_qa',validation_passed=True,total_outer_models=300));print('ANALYSIS_COMPLETE',flush=True)
def seal():
    assert json.load(open(OUT/'visual_qa.json'))['passed'];dump(OUT/'progress.json',dict(phase='complete',validation_passed=True,total_outer_models=300));manifest=dict(files=[dict(path=str(p.relative_to(OUT)),sha256=run.sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts]);dump(OUT/'artifact_manifest.json',manifest);archive=OUT.parent/'erastin_emt_ddr_comparison_results.zip';prefix=OUT.name+'/'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for row in manifest['files']:z.write(OUT/row['path'],prefix+row['path'])
        z.write(OUT/'artifact_manifest.json',prefix+'artifact_manifest.json')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in manifest['files']:assert hashlib.sha256(z.read(prefix+row['path'])).hexdigest()==row['sha256']
    print(json.dumps(dict(event='complete',archive_bytes=archive.stat().st_size,files=len(manifest['files'])+1)),flush=True)
if __name__=='__main__':
    predictions,parameters,validation,protocol=audit();evaluate(predictions,parameters,validation,protocol)
