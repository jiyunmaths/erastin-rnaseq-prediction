#!/usr/bin/env python3
"""Audit all newly fitted PRISM models and evaluate paired test predictions."""
import json,time,hashlib,zipfile
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

def table(name,rows):
    a=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows);a.to_csv(OUT/name,sep='\t',index=False,float_format='%.12g');return a
def dump(p,v):
    def safe(x):
        if isinstance(x,dict):return {str(k):safe(a) for k,a in x.items()}
        if isinstance(x,np.ndarray) and x.ndim==0:return safe(x.item())
        if isinstance(x,(list,tuple,np.ndarray)):return [safe(a) for a in x]
        if isinstance(x,np.generic):return safe(x.item())
        if isinstance(x,float) and not np.isfinite(x):return None
        return x
    run.dump(p,safe(v))
def pairs(protocol):return protocol['primary_comparisons']+protocol['secondary_comparisons']+[[run.arm_name(f,p),'training_mean'] for f in run.FAMILIES for p in run.PANELS]

def audit():
    run.initialize();s=run.STATE;y=s['y'];g=s['groups'];lin=s['lineage'];ph=run.sha(OUT/'protocol.json');protocol=json.load(open(OUT/'protocol.json'));assert json.load(open(OUT/'fit_completion.json'))['complete']
    for name,h in protocol['scripts'].items():assert run.sha(OUT/name)==h
    for name,h in protocol['data_sha256'].items():assert run.sha(OUT/name)==h
    for a in json.load(open(OUT/'source_manifest.json'))['files']:assert run.sha(WS/a['path'])==a['sha256']
    dump(OUT/'analysis_source_manifest.json',dict(review_sha256=run.sha(Path(__file__)),report_sha256=run.sha(OUT/'report_comparison.py')))
    preds={run.arm_name(f,p):np.full(s['test'].shape,np.nan) for f in run.FAMILIES for p in run.PANELS};preds['training_mean']=np.full(s['test'].shape,np.nan);params=[];tuning=[];warnings=[];objectives=[];references=[];coefficients=[];solver=[]
    maxp=maxloss=maxobj=maxkkt=0.;scores_checked=groups_checked=0
    for r in range(20):
        np.testing.assert_array_equal(s['test'][r],outer_split(lin,g,s['seeds'][r]));np.testing.assert_array_equal(s['inner'][r],inner_folds(lin,g,s['test'][r],s['inner_seeds'][r]));tr=np.flatnonzero(~s['test'][r]);te=np.flatnonzero(s['test'][r]);inner=s['inner'][r];assert (len(tr),len(te))==(308,78);preds['training_mean'][r,te]=y[tr].mean()
        for panel in run.PANELS:
            x=s['x'][panel]
            for f in run.FAMILIES:
                arm=run.arm_name(f,panel);root=OUT/'folds'/f'{f}__p{panel}__r{r+1:02d}';m=json.load(open(root.with_suffix('.json')));a=np.load(root.with_suffix('.npz'))
                assert m['protocol_sha256']==ph and m['family']==f and m['panel']==panel and m['repeat']==r+1 and m['train_lineages']==m['test_lineages']==20 and not m['reused']
                assert m['outer_split_seed']==int(s['seeds'][r]) and m['inner_split_seed']==int(s['inner_seeds'][r]) and m['model_seed']==int(s['model_seeds'][r]);np.testing.assert_array_equal(m['neural_seeds'],s['nn_seeds'][r]);np.testing.assert_array_equal(a['train_rows'],tr);np.testing.assert_array_equal(a['test_rows'],te);assert set(g[tr]).isdisjoint(g[te]) and set(lin[tr])==set(lin[te])==set(lin);groups_checked+=1
                for suffix in ['npz','joblib']:assert run.sha(root.with_suffix('.'+suffix))==m[suffix+'_sha256']
                grid=run.grid_for(f);c=m['selected_candidate'];p=grid[c];assert p==m['selected_parameters'] and c==np.argmin(a['inner_fold_rmse'].mean(axis=1));assert np.isfinite(a['inner_validation_predictions'][:,inner>=0]).all() and np.isnan(a['inner_validation_predictions'][:,inner<0]).all()
                for k in range(5):
                    aa,bb=inner_rows(inner,k);assert set(aa)|set(bb)==set(tr) and set(g[aa]).isdisjoint(g[bb]) and set(lin[aa])==set(lin);groups_checked+=1;assert m['inner_scaling'][k]['y_mean']==y[aa].mean() and m['inner_scaling'][k]['y_scale']==y[aa].std();loss=np.sqrt(np.mean((a['inner_validation_predictions'][:,bb]-y[bb][None,:])**2,axis=1));maxloss=max(maxloss,float(np.max(np.abs(loss-a['inner_fold_rmse'][:,k]))));scores_checked+=len(grid)
                model=joblib.load(root.with_suffix('.joblib'));sx=StandardScaler().fit(x[tr])
                if f=='ridge':
                    np.testing.assert_allclose(model['mean'],sx.mean_,atol=1e-12,rtol=0);np.testing.assert_allclose(model['scale'],sx.scale_,atol=1e-12,rtol=0);assert model['alpha']==p['alpha'] and model['intercept_after_scaling']==y[tr].mean();pred=sx.transform(x[te])@model['coef']+model['intercept_after_scaling']
                else:
                    sy=StandardScaler().fit(y[tr,None]);target=sy.transform(y[tr,None]).ravel();z=sx.transform(x[tr])
                    for key in ['mean_','scale_']:
                        np.testing.assert_allclose(getattr(model['x_scaler'],key),getattr(sx,key),atol=1e-12,rtol=0);np.testing.assert_allclose(getattr(model['y_scaler'],key),getattr(sy,key),atol=1e-12,rtol=0)
                    if f=='neural_network':
                        assert [v.random_state for v in model['members']]==m['neural_seeds'];members=np.array([sy.inverse_transform(v.predict(sx.transform(x[te]))[:,None]).ravel() for v in model['members']]);np.testing.assert_allclose(members,a['member_test_predictions'],atol=1e-12,rtol=0);pred=members.mean(axis=0)
                        for j,v in enumerate(model['members']):
                            np.testing.assert_array_equal(v.training_sample_weight_,np.ones(len(tr)));assert v.alpha==p['alpha'] and v.hidden_layer_sizes==(p['width'],);obj=.5*np.mean((v.predict(z)-target)**2)+.5*v.alpha*sum(np.sum(t*t) for t in v.coefs_)/len(tr);err=float(abs(obj-v.loss_));maxobj=max(maxobj,err);assert err<1e-10;objectives.append(dict(model=arm,repeat=r+1,member=j,loss_recalculation_error=err,iterations=v.n_iter_))
                    else:
                        v=model['model'];assert v.alpha==p['alpha'] and v.l1_ratio==p['l1_ratio'];pred=sy.inverse_transform(v.predict(sx.transform(x[te]))[:,None]).ravel();kkt=kkt_residual(z,target,v.coef_,v.intercept_,v.alpha,v.l1_ratio);maxkkt=max(maxkkt,kkt);assert kkt<1e-6;obj=.5*np.mean((v.predict(z)-target)**2)+v.alpha*v.l1_ratio*np.abs(v.coef_).sum()+.5*v.alpha*(1-v.l1_ratio)*np.sum(v.coef_**2);d=m['fit_diagnostics'][-1];assert abs(obj-d['objective'])<1e-12 and np.count_nonzero(v.coef_)==m['nonzero_genes']
                        for j,gene in enumerate(s['genes'][panel]):coefficients.append(dict(panel=panel,repeat=r+1,gene=str(gene),selected=bool(v.coef_[j]!=0),coefficient_standardized_y=v.coef_[j],coefficient_original_y_per_training_sd=v.coef_[j]*sy.scale_[0],coefficient_original_y_per_log2tpm=v.coef_[j]*sy.scale_[0]/sx.scale_[j]))
                        for d in m['fit_diagnostics']:
                            if d['stage']=='inner':
                                assert len(d['kkt_residuals'])==41 and len(d['dual_gaps'])==41 and len(d['iterations'])==41;maxkkt=max(maxkkt,max(d['kkt_residuals']));assert d['max_kkt_residual']<1e-6;solver.append(dict(panel=panel,repeat=r+1,stage='inner',fold=d['fold'],l1_ratio=d['l1_ratio'],max_kkt_residual=d['max_kkt_residual'],max_iterations=max(d['iterations'])))
                            else:solver.append(dict(panel=panel,repeat=r+1,stage='outer',fold=-1,l1_ratio=p['l1_ratio'],max_kkt_residual=kkt,max_iterations=d['iterations']))
                maxp=max(maxp,float(np.max(np.abs(pred-a['test_prediction']))));preds[arm][r,te]=pred
                for d in m['fit_diagnostics']:
                    warnings.extend(dict(model=arm,repeat=r+1,stage=d['stage'],**v) for v in d.get('warnings',[]))
                    for member in d.get('members',[]):warnings.extend(dict(model=arm,repeat=r+1,stage=d['stage'],**v) for v in member['warnings'])
                if r in [0,19]:
                    ref=independent(f,p,x,y,tr,te,s['nn_seeds'][r]);err=float(np.max(np.abs(ref-pred)));ie=0.
                    for k in range(5):
                        aa,bb=inner_rows(inner,k);ref=independent(f,p,x,y,aa,bb,s['nn_seeds'][r]);ie=max(ie,float(np.max(np.abs(ref-a['inner_validation_predictions'][c,bb]))))
                    assert max(err,ie)<1e-7;references.append(dict(model=arm,repeat=r+1,outer_error=err,inner_error=ie,inner_refits=5))
                params.append(dict(model=arm,family=f,panel=panel,repeat=r+1,candidate=c,**p,mean_inner_rmse=m['mean_inner_rmse'],nonzero_genes=m.get('nonzero_genes',np.nan)));tuning.extend(dict(model=arm,repeat=r+1,candidate=j,**v,mean_inner_rmse=float(a['inner_fold_rmse'][j].mean())) for j,v in enumerate(grid))
    assert groups_checked==720 and scores_checked==70600 and len(references)==12 and maxp<1e-10 and maxloss<1e-12 and len(objectives)==120
    for arm,pp in preds.items():assert np.isfinite(pp[s['test']]).all() and np.isnan(pp[~s['test']]).all()
    params=table('selected_parameters.tsv',params);table('tuning_scores.tsv',tuning);table('neural_training_objective_checks.tsv',objectives);table('elastic_net_solver_checks.tsv',solver);coef=table('elastic_net_coefficients.tsv',coefficients);selection=[]
    for (panel,gene),d in coef.groupby(['panel','gene']):
        b=d.coefficient_original_y_per_training_sd.to_numpy();n=int(np.count_nonzero(b));pos=int(np.sum(b>0));neg=int(np.sum(b<0));selection.append(dict(panel=panel,gene=gene,selected_fits=n,total_fits=20,positive_fits=pos,negative_fits=neg,selection_frequency=n/20,sign_agreement_when_selected=max(pos,neg)/n if n else np.nan,mean_coefficient=b.mean(),mean_absolute_coefficient=np.abs(b).mean()))
    table('elastic_net_gene_selection.tsv',selection);counts=Counter((v['category'],v['message']) for v in warnings);warning_summary=[dict(category=k[0],message=k[1],count=n) for k,n in counts.items()];bv=validate_bootstrap(y,preds,lin,s['test'],2026111204)
    validation=dict(passed=True,total_outer_models=120,new_outer_models=120,reused_outer_models=0,all_20_lineages_in_both_outer_sets=True,identity_group_exclusion_checks=groups_checked,candidate_fold_scores_checked=scores_checked,prediction_max_error=maxp,inner_rmse_max_error=maxloss,elastic_net_max_kkt_residual=maxkkt,elastic_net_outer_objectives_checked=40,elastic_net_inner_candidate_solutions_kkt_checked=65600,neural_training_objectives_checked=len(objectives),neural_objective_max_error=maxobj,independent_selected_setting_refits=references,preflight=json.load(open(OUT/'preflight_validation.json')),bootstrap_validation=bv,warning_count=sum(counts.values()),warning_summary=warning_summary,convergence_warning_count=sum(v['count'] for v in warning_summary if v['category']=='ConvergenceWarning'),selected_outer_convergence_warning_count=sum(v['category']=='ConvergenceWarning' and v['stage']=='outer' for v in warnings),input_source_hashes_verified=True,training_only_scaling_verified=True,unweighted_training_verified=True,shared_splits_verified=True)
    dump(OUT/'validation.json',validation);return preds,params,validation,protocol

def evaluate(preds,params,validation,protocol):
    s=run.STATE;y=s['y'];lin=s['lineage'];pooled,per,macro=score(y,preds,lin,s['test']);table('per_repeat_pooled_metrics.tsv',pooled);table('per_repeat_macro_metrics.tsv',macro);table('per_lineage_metrics.tsv',per);vals={arm:{} for arm in preds};summary=[]
    for arm in preds:
        f,p=arm.rsplit('__',1) if '__' in arm else (arm,0);row=dict(model=arm,family=f,panel=int(p))
        for key in ['spearman','pearson','rmse','mae','r2']:vals[arm]['pooled_'+key]=pooled[pooled.model==arm].sort_values('repeat')[key].to_numpy()
        for key in ['macro_all_rmse','macro_large_rmse']:vals[arm][key]=macro[macro.model==arm].sort_values('repeat')[key].to_numpy()
        for key,v in vals[arm].items():
            vv=v[np.isfinite(v)];row.update({key+'_mean':float(finite_mean(v)),key+'_sd':vv.std(ddof=1) if len(vv)>1 else np.nan,key+'_defined_repeats':len(vv)})
        summary.append(row)
    summary=table('summary.tsv',summary);dump(OUT/'progress.json',dict(phase='paired_bootstrap',audited_models=120));draws,draws_per_repeat=bootstrap(y,preds,lin,s['test'],count=5000,seed=protocol['uncertainty']['seed']);paired=[];seedrows=[];effects=[];wins=[]
    for first,ref in pairs(protocol):
        for key in KEYS:
            sign=-1 if key.endswith('rmse') or key.endswith('mae') else 1;delta=sign*(vals[first][key]-vals[ref][key]);mask=np.isfinite(delta);bd=finite_mean(sign*(draws_per_repeat[first][key]-draws_per_repeat[ref][key]),axis=1);valid=bd[np.isfinite(bd)];lo,hi=np.percentile(valid,[2.5,97.5]) if len(valid) else (np.nan,np.nan)
            paired.append(dict(first_model=first,reference_model=ref,metric=key,improvement=float(finite_mean(delta)),favorable_repeats=int(np.sum(delta>0)),paired_defined_repeats=int(mask.sum()),conditional_ci_low=lo,conditional_ci_high=hi,valid_bootstrap_draws=len(valid)));seedrows.extend(dict(first_model=first,reference_model=ref,metric=key,repeat=r+1,improvement=float(v)) for r,v in enumerate(delta))
        for label in np.unique(lin):
            a=per[(per.model==first)&(per.lineage==label)].sort_values('repeat');b=per[(per.model==ref)&(per.lineage==label)].sort_values('repeat');d=b.rmse.to_numpy()-a.rmse.to_numpy();effects.append(dict(first_model=first,reference_model=ref,lineage=str(label),test_n=int(a.test_n.iloc[0]),rmse_improvement=d.mean(),favorable_repeats=int(np.sum(d>0))))
        selected=[v for v in effects if v['first_model']==first and v['reference_model']==ref];wins.append(dict(first_model=first,reference_model=ref,rmse_favorable_lineages=sum(v['rmse_improvement']>0 for v in selected),lineages=20))
    paired=table('paired_comparisons.tsv',paired);table('paired_seed_differences.tsv',seedrows);effects=table('lineage_paired_effects.tsv',effects);wins=table('lineage_win_counts.tsv',wins);pcounts=[]
    for f in run.FAMILIES:
        cols=['alpha','l1_ratio'] if f=='elastic_net' else ['alpha'] if f=='ridge' else ['width','alpha'];pcounts.extend(params[params.family==f].groupby(['model','panel']+cols).size().reset_index(name='outer_fits').to_dict('records'))
    table('parameter_counts.tsv',pcounts);np.savez_compressed(OUT/'predictions.npz',sample_ids=s['ids'],group_ids=s['groups'],lineage=lin,y=y,outer_test=s['test'],outer_split_seeds=s['seeds'],**preds);np.savez_compressed(OUT/'bootstrap_draws.npz',**{arm+'__'+key:v for arm,d in draws.items() for key,v in d.items()});coef=params[params.family=='elastic_net'];sparsity=[]
    for p in run.PANELS:
        d=coef[coef.panel==p];sparsity.append(dict(panel=p,nonzero_genes_mean=d.nonzero_genes.mean(),nonzero_genes_median=d.nonzero_genes.median(),nonzero_genes_min=int(d.nonzero_genes.min()),nonzero_genes_max=int(d.nonzero_genes.max()),constant_outer_models=int(np.sum(d.nonzero_genes==0))))
    table('elastic_net_sparsity_summary.tsv',sparsity)
    from report_comparison import render
    render(summary,paired,effects,wins,validation)
    for a in json.load(open(OUT/'source_manifest.json'))['files']:assert run.sha(WS/a['path'])==a['sha256']
    am=json.load(open(OUT/'analysis_source_manifest.json'));assert run.sha(Path(__file__))==am['review_sha256'] and run.sha(OUT/'report_comparison.py')==am['report_sha256']
    dump(OUT/'final_summary.json',dict(validation_passed=True,summary=summary.to_dict('records'),new_outer_models=120,reused_outer_models=0,total_outer_models=120));dump(OUT/'progress.json',dict(phase='complete',validation_passed=True,total_outer_models=120));seal()

def seal():
    manifest=dict(files=[dict(path=str(p.relative_to(OUT)),sha256=run.sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts]);dump(OUT/'artifact_manifest.json',manifest);archive=OUT.parent/'erastin_prism_direct_comparison_results.zip';prefix=OUT.name+'/'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for row in manifest['files']:z.write(OUT/row['path'],prefix+row['path'])
        z.write(OUT/'artifact_manifest.json',prefix+'artifact_manifest.json')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in manifest['files']:assert hashlib.sha256(z.read(prefix+row['path'])).hexdigest()==row['sha256']
    print(json.dumps(dict(event='complete',archive_bytes=archive.stat().st_size,files=len(manifest['files'])+1)),flush=True)
if __name__=='__main__':
    predictions,parameters,validation,protocol=audit();evaluate(predictions,parameters,validation,protocol)
