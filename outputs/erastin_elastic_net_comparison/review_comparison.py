#!/usr/bin/env python3
"""Audit shared-split Elastic Net, ridge and NN fits; evaluate paired results."""
import json,hashlib,zipfile
from pathlib import Path
from collections import Counter
import joblib,numpy as np,pandas as pd
from sklearn.preprocessing import StandardScaler
import run_comparison as run
from enet_helpers import kkt_residual
from validate_preflight import independent
from make_splits import inner_rows
from evaluate import score,bootstrap,validate_bootstrap
OUT=run.OUT;WS=run.WS

def table(name,rows):
    a=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows);a.to_csv(OUT/name,sep='\t',index=False,float_format='%.12g');return a

def pairs():
    p=json.load(open(OUT/'protocol.json'));v=p['primary_comparisons']+p['secondary_comparisons'];assert len(v)==len({tuple(x) for x in v})==9;return v

def audit():
    run.initialize();s=run.STATE;y=s['y'];g=s['groups'];lin=s['lineage'];ph=run.sha(OUT/'protocol.json');protocol=json.load(open(OUT/'protocol.json'))
    assert json.load(open(OUT/'fit_completion.json'))['complete']
    for name,h in protocol['scripts'].items():assert run.sha(OUT/name)==h
    for name,h in protocol['data_sha256'].items():assert run.sha(OUT/name)==h
    for a in json.load(open(OUT/'source_manifest.json'))['files']:assert run.sha(WS/a['path'])==a['sha256']
    for a in json.load(open(OUT/'reuse_manifest.json'))['files']:assert run.sha(OUT/a['local'])==a['sha256']
    run.dump(OUT/'analysis_source_manifest.json',dict(review_sha256=run.sha(Path(__file__)),report_sha256=run.sha(OUT/'report_comparison.py')))
    previous=np.load(WS/'outputs/erastin_five_gene_comparison/predictions.npz');shared=np.load(WS/'outputs/erastin_five_gene_comparison/shared_splits.npz')
    for key in ['outer_test','inner_fold','outer_split_seeds','inner_split_seeds','neural_initialization_seeds']:np.testing.assert_array_equal(np.load(OUT/'shared_splits.npz')[key],shared[key])
    preds={run.arm_name(f,p):np.full(s['test'].shape,np.nan) for f in run.FAMILIES for p in run.PANELS};params=[];tuning=[];warnings=[];objectives=[];references=[];coefficients=[];solver=[]
    maxp=maxloss=maxobj=maxkkt=0.;scores_checked=groups_checked=reused=0
    for r in range(20):
        tr=np.flatnonzero(~s['test'][r]);te=np.flatnonzero(s['test'][r]);inner=s['inner'][r]
        for panel in run.PANELS:
            x=s['x'][panel]
            for f in run.FAMILIES:
                arm=run.arm_name(f,panel);root=OUT/'folds'/f'{f}__p{panel}__r{r+1:02d}';m=json.load(open(root.with_suffix('.json')));a=np.load(root.with_suffix('.npz'))
                assert m['protocol_sha256']==ph and m['family']==f and m['panel']==panel and m['repeat']==r+1 and m['train_lineages']==m['test_lineages']==23
                assert m['outer_split_seed']==int(s['seeds'][r]) and m['inner_split_seed']==int(s['inner_seeds'][r]) and m['model_seed']==int(s['model_seeds'][r]);np.testing.assert_array_equal(m['neural_seeds'],s['nn_seeds'][r])
                assert m['reused']==(f!='elastic_net');reused+=int(m['reused'])
                if m['reused']:assert run.sha(OUT/'source_metadata'/f'{f}__p{panel}__r{r+1:02d}.json')==m['original_metadata_sha256']
                np.testing.assert_array_equal(a['train_rows'],tr);np.testing.assert_array_equal(a['test_rows'],te);assert set(g[tr]).isdisjoint(g[te]) and set(lin[tr])==set(lin[te])==set(lin);groups_checked+=1
                for suffix in ['npz','joblib']:assert run.sha(root.with_suffix('.'+suffix))==m[suffix+'_sha256']
                grid=run.grid_for(f);c=m['selected_candidate'];p=grid[c];assert p==m['selected_parameters'] and c==np.argmin(a['inner_fold_rmse'].mean(axis=1))
                assert np.isfinite(a['inner_validation_predictions'][:,inner>=0]).all() and np.isnan(a['inner_validation_predictions'][:,inner<0]).all()
                for k in range(5):
                    aa,bb=inner_rows(inner,k);assert set(aa)|set(bb)==set(tr) and set(g[aa]).isdisjoint(g[bb]) and set(lin[aa])==set(lin);groups_checked+=1
                    assert m['inner_scaling'][k]['y_mean']==y[aa].mean() and m['inner_scaling'][k]['y_scale']==y[aa].std()
                    loss=np.sqrt(np.mean((a['inner_validation_predictions'][:,bb]-y[bb][None,:])**2,axis=1));maxloss=max(maxloss,float(np.max(np.abs(loss-a['inner_fold_rmse'][:,k]))));scores_checked+=len(grid)
                model=joblib.load(root.with_suffix('.joblib'));sx=StandardScaler().fit(x[tr])
                if f=='ridge':
                    np.testing.assert_allclose(model['mean'],sx.mean_,atol=1e-12,rtol=0);np.testing.assert_allclose(model['scale'],sx.scale_,atol=1e-12,rtol=0);assert model['alpha']==p['alpha'] and model['intercept_after_scaling']==y[tr].mean();pred=sx.transform(x[te])@model['coef']+model['intercept_after_scaling']
                else:
                    sy=StandardScaler().fit(y[tr,None]);target=sy.transform(y[tr,None]).ravel();z=sx.transform(x[tr])
                    for key in ['mean_','scale_']:
                        np.testing.assert_allclose(getattr(model['x_scaler'],key),getattr(sx,key),atol=1e-12,rtol=0);np.testing.assert_allclose(getattr(model['y_scaler'],key),getattr(sy,key),atol=1e-12,rtol=0)
                    if f=='neural_network':
                        assert [v.random_state for v in model['members']]==m['neural_seeds']
                        members=np.array([sy.inverse_transform(v.predict(sx.transform(x[te]))[:,None]).ravel() for v in model['members']]);np.testing.assert_allclose(members,a['member_test_predictions'],atol=1e-12,rtol=0);pred=members.mean(axis=0)
                        for j,v in enumerate(model['members']):
                            np.testing.assert_array_equal(v.training_sample_weight_,np.ones(len(tr)));assert v.alpha==p['alpha'] and v.hidden_layer_sizes==(p['width'],)
                            obj=.5*np.mean((v.predict(z)-target)**2)+.5*v.alpha*sum(np.sum(t*t) for t in v.coefs_)/len(tr);err=float(abs(obj-v.loss_));maxobj=max(maxobj,err);assert err<1e-10;objectives.append(dict(model=arm,repeat=r+1,member=j,loss_recalculation_error=err,iterations=v.n_iter_))
                    else:
                        v=model['model'];assert v.alpha==p['alpha'] and v.l1_ratio==p['l1_ratio'] and v.fit_intercept and not v.positive and v.selection=='cyclic'
                        pred=sy.inverse_transform(v.predict(sx.transform(x[te]))[:,None]).ravel();kkt=kkt_residual(z,target,v.coef_,v.intercept_,v.alpha,v.l1_ratio);maxkkt=max(maxkkt,kkt);assert kkt<1e-6
                        obj=.5*np.mean((v.predict(z)-target)**2)+v.alpha*v.l1_ratio*np.abs(v.coef_).sum()+.5*v.alpha*(1-v.l1_ratio)*np.sum(v.coef_**2)
                        d=m['fit_diagnostics'][-1];assert abs(obj-d['objective'])<1e-12 and int(np.count_nonzero(v.coef_))==m['nonzero_genes']==d['nonzero_genes']
                        for j,gene in enumerate(s['genes'][panel]):coefficients.append(dict(panel=panel,repeat=r+1,gene=str(gene),selected=bool(v.coef_[j]!=0),coefficient_standardized_y=v.coef_[j],coefficient_original_y_per_training_sd=v.coef_[j]*sy.scale_[0],coefficient_original_y_per_log2tpm=v.coef_[j]*sy.scale_[0]/sx.scale_[j]))
                        for d in m['fit_diagnostics']:
                            if d['stage']=='inner':
                                assert len(d['kkt_residuals'])==41 and len(d['dual_gaps'])==41 and len(d['iterations'])==41;maxkkt=max(maxkkt,max(d['kkt_residuals']));assert d['max_kkt_residual']<1e-6
                                solver.append(dict(panel=panel,repeat=r+1,stage='inner',fold=d['fold'],l1_ratio=d['l1_ratio'],max_kkt_residual=d['max_kkt_residual'],max_dual_gap=d['max_dual_gap'],max_iterations=max(d['iterations'])))
                            else:solver.append(dict(panel=panel,repeat=r+1,stage='outer',fold=-1,l1_ratio=p['l1_ratio'],max_kkt_residual=kkt,max_dual_gap=d['dual_gap'],max_iterations=d['iterations']))
                        if r in [0,19]:
                            ref=independent(p,x,y,tr,te);err=float(np.max(np.abs(ref-pred)));ie=0.
                            for k in range(5):
                                aa,bb=inner_rows(inner,k);ref=independent(p,x,y,aa,bb);ie=max(ie,float(np.max(np.abs(ref-a['inner_validation_predictions'][c,bb]))))
                            assert max(err,ie)<1e-7;references.append(dict(model=arm,repeat=r+1,outer_error=err,inner_error=ie,inner_refits=5,reference='augmented Lasso'))
                maxp=max(maxp,float(np.max(np.abs(pred-a['test_prediction']))));preds[arm][r,te]=pred
                for d in m['fit_diagnostics']:
                    warnings.extend(dict(model=arm,repeat=r+1,**v) for v in d.get('warnings',[]))
                    for member in d.get('members',[]):warnings.extend(dict(model=arm,repeat=r+1,**v) for v in member['warnings'])
                params.append(dict(model=arm,family=f,panel=panel,repeat=r+1,candidate=c,**p,mean_inner_rmse=m['mean_inner_rmse'],reused=m['reused'],nonzero_genes=m.get('nonzero_genes',np.nan)))
                tuning.extend(dict(model=arm,repeat=r+1,candidate=j,**v,mean_inner_rmse=float(a['inner_fold_rmse'][j].mean())) for j,v in enumerate(grid))
    assert reused==80 and groups_checked==720 and scores_checked==69800 and len(references)==4 and maxp<1e-10 and maxloss<1e-12
    for arm,pp in preds.items():
        assert np.isfinite(pp[s['test']]).all() and np.isnan(pp[~s['test']]).all()
        if not arm.startswith('elastic_net'):np.testing.assert_array_equal(pp,previous[arm])
    params=table('selected_parameters.tsv',params);table('tuning_scores.tsv',tuning);table('neural_training_objective_checks.tsv',objectives);table('elastic_net_solver_checks.tsv',solver);coef=table('elastic_net_coefficients.tsv',coefficients);selection=[]
    for (panel,gene),d in coef.groupby(['panel','gene']):
        b=d.coefficient_original_y_per_training_sd.to_numpy();n=int(np.count_nonzero(b));positive=int(np.sum(b>0));negative=int(np.sum(b<0));selection.append(dict(panel=panel,gene=gene,selected_fits=n,total_fits=20,selection_frequency=n/20,positive_fits=positive,negative_fits=negative,sign_agreement_when_selected=max(positive,negative)/n if n else np.nan,mean_coefficient_original_y_per_training_sd=b.mean(),mean_absolute_coefficient_original_y_per_training_sd=np.mean(np.abs(b)),sd_coefficient_original_y_per_training_sd=b.std(ddof=1)))
    selection=table('elastic_net_gene_selection.tsv',pd.DataFrame(selection).sort_values(['panel','selected_fits','mean_absolute_coefficient_original_y_per_training_sd'],ascending=[True,False,False]));sparsity=[]
    for panel in run.PANELS:
        d=params[(params.family=='elastic_net')&(params.panel==panel)];sparsity.append(dict(panel=panel,nonzero_genes_mean=d.nonzero_genes.mean(),nonzero_genes_median=d.nonzero_genes.median(),nonzero_genes_min=int(d.nonzero_genes.min()),nonzero_genes_max=int(d.nonzero_genes.max()),alpha_min=d.alpha.min(),alpha_max=d.alpha.max(),l1_ratio_min=d.l1_ratio.min(),l1_ratio_max=d.l1_ratio.max(),lower_alpha_boundary_fits=int(np.sum(d.alpha==min(run.ALPHAS))),upper_alpha_boundary_fits=int(np.sum(d.alpha==max(run.ALPHAS))),lowest_l1_ratio_fits=int(np.sum(d.l1_ratio==min(run.L1_RATIOS))),pure_lasso_fits=int(np.sum(d.l1_ratio==1))))
    table('elastic_net_sparsity_summary.tsv',sparsity)
    counts=Counter((v['category'],v['message']) for v in warnings);warning_summary=[dict(category=k[0],message=k[1],count=n) for k,n in counts.items()];bv=validate_bootstrap(y,preds,lin,s['test'],seed=2026111004)
    validation=dict(passed=True,total_outer_models=120,new_outer_models=40,reused_outer_models=reused,all_23_lineages_in_both_outer_sets=True,identity_group_exclusion_checks=groups_checked,candidate_fold_scores_checked=scores_checked,prediction_max_error=maxp,inner_rmse_max_error=maxloss,elastic_net_max_kkt_residual=maxkkt,elastic_net_outer_objectives_checked=40,elastic_net_inner_candidate_solutions_kkt_checked=65600,neural_training_objectives_checked=len(objectives),neural_objective_max_error=maxobj,independent_selected_setting_refits=references,preflight=json.load(open(OUT/'preflight_validation.json')),bootstrap_validation=bv,warning_count=sum(counts.values()),warning_summary=warning_summary,convergence_warning_count=sum(v['count'] for v in warning_summary if v['category']=='ConvergenceWarning'),input_source_hashes_verified=True,training_only_scaling_verified=True,unweighted_training_verified=True,byte_identical_shared_splits_verified=True,all_80_baseline_predictions_exactly_reproduced=True,exploratory_post_selection_comparison=True)
    run.dump(OUT/'validation.json',validation);return preds,params,validation,protocol

def evaluate(preds,params,validation,protocol):
    s=run.STATE;y=s['y'];lin=s['lineage'];pooled,per,macro=score(y,preds,lin,s['test']);table('per_repeat_pooled_metrics.tsv',pooled);table('per_repeat_macro_metrics.tsv',macro);table('per_lineage_metrics.tsv',per)
    original=pd.read_csv(WS/'outputs/erastin_five_gene_comparison/per_repeat_pooled_metrics.tsv',sep='\t')
    for f in run.BASELINES:
        for panel in run.PANELS:
            arm=run.arm_name(f,panel);np.testing.assert_allclose(pooled[pooled.model==arm].sort_values('repeat')[['spearman','rmse','r2']],original[original.model==arm].sort_values('repeat')[['spearman','rmse','r2']],atol=1e-12,rtol=0)
    vals={arm:{} for arm in preds};summary=[]
    for arm in preds:
        f,p=arm.rsplit('__',1);row=dict(model=arm,family=f,panel=int(p))
        for key in ['spearman','rmse','r2']:vals[arm]['pooled_'+key]=pooled[pooled.model==arm].sort_values('repeat')[key].to_numpy()
        for key in ['macro23_rmse','macro19_rmse']:vals[arm][key]=macro[macro.model==arm].sort_values('repeat')[key].to_numpy()
        for key,v in vals[arm].items():assert np.isfinite(v).all();row.update({key+'_mean':v.mean(),key+'_sd':v.std(ddof=1),key+'_min':v.min(),key+'_max':v.max()})
        summary.append(row)
    summary=table('summary.tsv',summary);run.dump(OUT/'progress.json',dict(phase='paired_bootstrap',audited_models=120));draws=bootstrap(y,preds,lin,s['test'],count=5000,seed=protocol['uncertainty']['seed']);paired=[];seedrows=[];effects=[];wins=[]
    for first,reference in pairs():
        for key in vals[first]:
            sign=-1 if key.endswith('rmse') else 1;delta=sign*(vals[first][key]-vals[reference][key]);lo,hi=np.percentile(sign*(draws[first][key]-draws[reference][key]),[2.5,97.5])
            paired.append(dict(first_model=first,reference_model=reference,metric=key,improvement=delta.mean(),favorable_repeats=int(np.sum(delta>0)),repeats=20,conditional_ci_low=lo,conditional_ci_high=hi));seedrows.extend(dict(first_model=first,reference_model=reference,metric=key,repeat=r+1,improvement=float(v)) for r,v in enumerate(delta))
        for label in np.unique(lin):
            a=per[(per.model==first)&(per.lineage==label)].sort_values('repeat');b=per[(per.model==reference)&(per.lineage==label)].sort_values('repeat');d=b.rmse.to_numpy()-a.rmse.to_numpy();effects.append(dict(first_model=first,reference_model=reference,lineage=str(label),test_n=int(a.test_n.iloc[0]),rmse_improvement=d.mean(),favorable_repeats=int(np.sum(d>0))))
        selected=[v for v in effects if v['first_model']==first and v['reference_model']==reference];wins.append(dict(first_model=first,reference_model=reference,rmse_favorable_lineages=sum(v['rmse_improvement']>0 for v in selected),lineages=23))
    paired=table('paired_comparisons.tsv',paired);table('paired_seed_differences.tsv',seedrows);effects=table('lineage_paired_effects.tsv',effects);wins=table('lineage_win_counts.tsv',wins)
    agreement=[]
    from scipy.stats import spearmanr
    for f in run.FAMILIES:
        for r,mask in enumerate(s['test']):
            a=preds[run.arm_name(f,5)][r,mask];b=preds[run.arm_name(f,64)][r,mask];agreement.append(dict(family=f,repeat=r+1,pearson=float(np.corrcoef(a,b)[0,1]),spearman=float(spearmanr(a,b).statistic),prediction_rmse=float(np.sqrt(np.mean((a-b)**2)))))
    table('prediction_agreement.tsv',agreement);pcounts=[]
    for f in run.FAMILIES:
        cols=['alpha','l1_ratio'] if f=='elastic_net' else ['alpha'] if f=='ridge' else ['width','alpha'];pcounts.extend(params[params.family==f].groupby(['model','panel']+cols).size().reset_index(name='outer_fits').to_dict('records'))
    table('parameter_counts.tsv',pcounts);np.savez_compressed(OUT/'predictions.npz',sample_ids=s['ids'],group_ids=s['groups'],lineage=lin,y=y,outer_test=s['test'],outer_split_seeds=s['seeds'],**preds);np.savez_compressed(OUT/'bootstrap_draws.npz',**{arm+'__'+key:v for arm,d in draws.items() for key,v in d.items()})
    from report_comparison import render
    render(summary,paired,effects,wins,validation)
    for a in json.load(open(OUT/'source_manifest.json'))['files']:assert run.sha(WS/a['path'])==a['sha256']
    am=json.load(open(OUT/'analysis_source_manifest.json'));assert run.sha(Path(__file__))==am['review_sha256'] and run.sha(OUT/'report_comparison.py')==am['report_sha256']
    run.dump(OUT/'final_summary.json',dict(validation_passed=True,summary=summary.to_dict('records'),new_outer_models=40,reused_outer_models=80,total_outer_models=120));run.dump(OUT/'progress.json',dict(phase='complete',validation_passed=True,total_outer_models=120));seal()

def seal():
    manifest=dict(files=[dict(path=str(p.relative_to(OUT)),sha256=run.sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts]);run.dump(OUT/'artifact_manifest.json',manifest);archive=OUT.parent/'erastin_elastic_net_comparison_results.zip';prefix=OUT.name+'/'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for row in manifest['files']:z.write(OUT/row['path'],prefix+row['path'])
        z.write(OUT/'artifact_manifest.json',prefix+'artifact_manifest.json')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in manifest['files']:assert hashlib.sha256(z.read(prefix+row['path'])).hexdigest()==row['sha256']
    print(json.dumps(dict(event='complete',archive_bytes=archive.stat().st_size,files=len(manifest['files'])+1)),flush=True)
if __name__=='__main__':
    predictions,parameters,validation,protocol=audit();evaluate(predictions,parameters,validation,protocol)
