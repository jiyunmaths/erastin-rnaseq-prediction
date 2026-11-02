#!/usr/bin/env python3
"""Audit all weighted fits and compare with the nine original unweighted arms."""
import json
import time
from pathlib import Path
import sys
import zipfile
from collections import Counter

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import Ridge
from weighted_mlp import WeightedMLPRegressor as MLPRegressor, inverse_lineage_weights
from sklearn.preprocessing import StandardScaler

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;WS=OUT.parent.parent
sys.path.insert(0,str(OUT))
import run_weighted as run
old=run.module('previous_evaluation',WS/'outputs/erastin_bayesian_comparison/review_bayesian.py')
METRICS=['spearman','rmse','r2'];SIZES=[40,64,80]
FAMILIES=run.FAMILIES
LABELS={'ridge':'Ridge','neural_network':'Neural network','xgboost':'XGBoost'}

def name(f,p,weighted=True):return ('weighted__' if weighted else 'unweighted__')+f+'__'+str(p)
def label(arm):
    weight,f,p=arm.split('__');return LABELS[f]+' / '+p+' genes / '+weight

def table(filename,rows):
    frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT/filename,sep='\t',index=False,float_format='%.12g');return frame

def independent_prediction(f,params,seeds,x,y,a,b):
    w=inverse_lineage_weights(run.STATE['lineage'][a])
    if f=='ridge':
        sx=StandardScaler().fit(x[a]);m=Ridge(alpha=params['alpha'],solver='svd').fit(sx.transform(x[a]),y[a],sample_weight=w)
        return m.predict(sx.transform(x[b]))
    sy=StandardScaler().fit(y[a,None]);target=sy.transform(y[a,None]).ravel()
    if f=='xgboost':
        pp=run.xp.parameters(params,seeds[0]);pp.pop('nthread');pp.pop('seed')
        m=xgb.XGBRegressor(**pp,n_jobs=1,random_state=int(seeds[0]),n_estimators=params['num_boost_round']).fit(x[a],target,sample_weight=w)
        return m.predict(x[b]).astype(float)*sy.scale_[0]+sy.mean_[0]
    sx=StandardScaler().fit(x[a]);members=[]
    for seed in seeds:
        m=MLPRegressor(hidden_layer_sizes=(params['width'],),activation='tanh',solver='lbfgs',alpha=params['alpha'],
                       max_iter=2000,max_fun=40000,tol=.001,early_stopping=False,random_state=int(seed))
        m.fit(sx.transform(x[a]),target,sample_weight=w)
        members.append(sy.inverse_transform(m.predict(sx.transform(x[b]))[:,None]).ravel())
    return np.mean(members,axis=0)

def planned_pairs():
    pairs=[(name(f,p),name(f,p,False)) for f in FAMILIES for p in SIZES]
    for f in FAMILIES:pairs.extend([(name(f,64),name(f,40)),(name(f,80),name(f,40)),(name(f,80),name(f,64))])
    for p in SIZES:pairs.extend([(name('neural_network',p),name('ridge',p)),(name('xgboost',p),name('ridge',p)),(name('xgboost',p),name('neural_network',p))])
    assert len(pairs)==len(set(pairs))==27
    return pairs

def wait_for_path(path):
    while not path.exists():time.sleep(.5)

def audit():
    run.initialize();s=run.STATE;y=s['y'];groups=s['groups'];lineage=s['lineage']
    protocol=json.loads((OUT/'protocol.json').read_text());ph=run.sha(OUT/'protocol.json')
    assert run.sha(OUT/'run_weighted.py')==protocol['script_sha256']
    assert run.sha(OUT/'weighted_mlp.py')==protocol['weighted_mlp_sha256']
    for source in json.loads((OUT/'source_manifest.json').read_text())['files']:assert run.sha(WS/source['path'])==source['sha256']
    run.dump(OUT/'analysis_source_manifest.json',dict(script_sha256=run.sha(Path(__file__)),report_sha256=run.sha(OUT/'report_weighted.py')))
    np.testing.assert_array_equal(np.load(OUT/'shared_splits.npz')['inner_fold_by_holdout'],np.load(WS/'outputs/erastin_xgboost_comparison/shared_splits.npz')['inner_fold_by_holdout'])
    base=np.load(WS/'outputs/erastin_panel_comparison/predictions.npz')
    for key,current in [('sample_ids',s['ids']),('group_ids',groups),('lineage',lineage),('y',y),('inner_split_seeds',s['seeds'])]:np.testing.assert_array_equal(base[key],current)
    preds={name(f,p):np.full((20,745),np.nan) for f in FAMILIES for p in SIZES}
    preds.update({name(f,p,False):base[f+'__'+str(p)].copy() for f in FAMILIES for p in SIZES})
    weight_rows=[];weight_group_rows=[]
    parameters=[];tuning=[];references=[];warnings=[];group_checks=0;score_count=0
    max_p=max_loss=0.
    for r in range(20):
        for panel in SIZES:
            x=s['x'][panel];genes=s['genes'][panel]
            for f in FAMILIES:
                seen=np.zeros(745,int)
                for l,heldout in enumerate(s['labels']):
                    root=OUT/'folds'/f'{f}__p{panel}__r{r+1:02d}__l{l:02d}'
                    wait_for_path(root.with_suffix('.json'))
                    m=json.loads(root.with_suffix('.json').read_text());a=np.load(root.with_suffix('.npz'))
                    tr,te=a['train_rows'],a['test_rows'];inner=s['inner'][r,l]
                    np.testing.assert_array_equal(tr,np.flatnonzero(inner>=0));np.testing.assert_array_equal(te,np.flatnonzero(lineage==heldout))
                    assert set(groups[tr]).isdisjoint(groups[te]) and heldout not in set(lineage[tr]);group_checks+=1;seen[te]+=1
                    outer_w=inverse_lineage_weights(lineage[tr])
                    assert m['outer_weights']==run.weight_audit(lineage[tr])
                    if panel==40 and f=='ridge':record_weights(weight_rows,weight_group_rows,r+1,str(heldout),'outer',-1,m['outer_weights'])
                    assert m['protocol_sha256']==ph and m['panel']==panel and m['family']==f and m['repeat']==r+1 and m['lineage']==heldout
                    assert m['inner_split_seed']==int(s['seeds'][r]) and m['model_seed']==int(s['model_seeds'][r])
                    np.testing.assert_array_equal(m['neural_seeds'],s['nn_seeds'][r])
                    assert run.sha(root.with_suffix('.npz'))==m['npz_sha256']
                    grid=([dict(alpha=float(v)) for v in s['alphas']] if f=='ridge' else run.nn.NN_GRID if f=='neural_network' else run.xp.GRID)
                    c=m['selected_candidate'];params=grid[c]
                    assert params==m['selected_parameters'] and c==np.argmin(a['inner_fold_rmse'].mean(axis=1))
                    assert np.isfinite(a['inner_validation_predictions'][:,tr]).all() and np.isnan(a['inner_validation_predictions'][:,inner<0]).all()
                    for k in range(5):
                        aa=tr[inner[tr]!=k];bb=tr[inner[tr]==k]
                        assert set(aa)|set(bb)==set(tr) and set(groups[aa]).isdisjoint(groups[bb]);group_checks+=1
                        assert m['inner_scaling'][k]['y_mean']==y[aa].mean() and m['inner_scaling'][k]['y_scale']==y[aa].std()
                        assert m['inner_scaling'][k]['weights']==run.weight_audit(lineage[aa])
                        if panel==40 and f=='ridge':record_weights(weight_rows,weight_group_rows,r+1,str(heldout),'inner',k,m['inner_scaling'][k]['weights'])
                        loss=np.sqrt(np.mean((a['inner_validation_predictions'][:,bb]-y[bb][None,:])**2,axis=1))
                        max_loss=max(max_loss,float(np.max(np.abs(loss-a['inner_fold_rmse'][:,k]))));score_count+=len(grid)
                    if f=='xgboost':
                        assert run.sha(root.with_suffix('.ubj'))==m['ubj_sha256']
                        model=xgb.Booster();model.load_model(root.with_suffix('.ubj'));model.set_param({'nthread':1})
                        assert model.num_features()==panel and model.feature_names==list(genes) and model.num_boosted_rounds()==params['num_boost_round']
                        assert m['y_mean']==y[tr].mean() and m['y_scale']==y[tr].std()
                        p=model.predict(run.matrix(x[te],genes)).astype(float)*m['y_scale']+m['y_mean']
                    else:
                        assert run.sha(root.with_suffix('.joblib'))==m['joblib_sha256'];model=joblib.load(root.with_suffix('.joblib'))
                        sx=StandardScaler().fit(x[tr])
                        if f=='ridge':
                            np.testing.assert_allclose(model['mean'],sx.mean_,rtol=0,atol=1e-12);np.testing.assert_allclose(model['scale'],sx.scale_,rtol=0,atol=1e-12)
                            assert model['alpha']==params['alpha']
                            np.testing.assert_array_equal(model['training_sample_weight'],outer_w)
                            np.testing.assert_allclose(model['weighted_y_mean'],np.average(y[tr],weights=outer_w),rtol=0,atol=1e-14)
                            np.testing.assert_allclose(model['weighted_x_mean_after_scaling'],np.average(sx.transform(x[tr]),axis=0,weights=outer_w),rtol=0,atol=1e-14)
                            p=sx.transform(x[te])@model['coef']+model['intercept_after_scaling']
                        else:
                            np.testing.assert_allclose(model['x_scaler'].mean_,sx.mean_,rtol=0,atol=1e-12);np.testing.assert_allclose(model['x_scaler'].scale_,sx.scale_,rtol=0,atol=1e-12)
                            sy=StandardScaler().fit(y[tr,None]);np.testing.assert_allclose(model['y_scaler'].mean_,sy.mean_,rtol=0,atol=1e-12);np.testing.assert_allclose(model['y_scaler'].scale_,sy.scale_,rtol=0,atol=1e-12)
                            members=np.array([model['y_scaler'].inverse_transform(v.predict(sx.transform(x[te]))[:,None]).ravel() for v in model['members']])
                            np.testing.assert_allclose(members,a['member_test_predictions'],rtol=0,atol=1e-12);p=members.mean(axis=0)
                            for v in model['members']:np.testing.assert_array_equal(v.training_sample_weight_,outer_w)
                            assert [v.random_state for v in model['members']]==m['neural_seeds']
                    max_p=max(max_p,float(np.max(np.abs(p-a['test_prediction']))));preds[name(f,panel)][r,te]=p
                    parameters.append(dict(model=name(f,panel),family=f,panel=panel,repeat=r+1,lineage=str(heldout),candidate=c,**params,
                                           mean_inner_rmse=m['mean_inner_rmse'],train_n=len(tr),test_n=len(te)))
                    tuning.extend(dict(model=name(f,panel),repeat=r+1,lineage=str(heldout),candidate=ci,**param,
                                       mean_inner_rmse=float(a['inner_fold_rmse'][ci].mean())) for ci,param in enumerate(grid))
                    for diag in m['fit_diagnostics']:
                        if 'warning' in diag:warnings.append(dict(model=name(f,panel),repeat=r+1,lineage=str(heldout),**diag))
                        for member in diag.get('members',[]):
                            warnings.extend(dict(model=name(f,panel),repeat=r+1,lineage=str(heldout),stage=diag['stage'],**w) for w in member['warnings'])
                    if r in [0,19] and heldout in ['Lung','Thyroid']:
                        seeds=m['neural_seeds'] if f=='neural_network' else [m['model_seed']]
                        ref=independent_prediction(f,params,seeds,x,y,tr,te);err=float(np.max(np.abs(ref-p)));ie=0.
                        for k in range(5):
                            aa=tr[inner[tr]!=k];bb=tr[inner[tr]==k]
                            ref=independent_prediction(f,params,seeds,x,y,aa,bb)
                            ie=max(ie,float(np.max(np.abs(ref-a['inner_validation_predictions'][c,bb]))))
                        assert max(err,ie)<1e-7
                        references.append(dict(model=name(f,panel),repeat=r+1,lineage=str(heldout),outer_refit_error=err,inner_refit_error=ie,inner_refits=5))
                assert np.all(seen==1)
                run.dump(OUT/'audit_progress.json',dict(audited_models=len(parameters),total=4320,last_model=name(f,panel),last_repeat=r+1))
        if (r+1)%2==0:print(json.dumps(dict(event='audit_progress',complete_repeats=r+1)),flush=True)
    wait_for_path(OUT/'fit_completion.json')
    assert json.loads((OUT/'fit_completion.json').read_text())['complete']
    assert len(references)==36 and max_p<1e-10 and max_loss<1e-12
    assert all(np.isfinite(v).all() for v in preds.values())
    assert score_count==480*3*(17+4+36)*5 and group_checks==4320*6
    table('training_subset_weight_audit.tsv',weight_rows);table('training_lineage_weights.tsv',weight_group_rows)
    parameters=table('selected_parameters.tsv',parameters);table('tuning_scores.tsv',tuning)
    warning_counts=Counter((v.get('warning_category',v.get('category','unknown')),v.get('warning',v.get('message',''))) for v in warnings)
    warning_summary=[dict(category=k[0],message=k[1],count=n) for k,n in warning_counts.items()]
    import scipy,sklearn,platform
    run.dump(OUT/'environment.json',dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__,xgboost=xgb.__version__))
    validation=dict(passed=True,new_outer_models=4320,reused_unweighted_prediction_arms=9,group_exclusion_checks=group_checks,
                    reconstructed_candidate_fold_scores=score_count,prediction_max_error=max_p,inner_rmse_max_error=max_loss,
                    independent_refits=references,warning_count=len(warnings),warning_summary=warning_summary,
                    convergence_warning_count=sum(v['count'] for v in warning_summary if v['category']=='ConvergenceWarning'),
                    warning_detail_location='folds/*.json:fit_diagnostics',
                    preflight=json.loads((OUT/'preflight_validation.json').read_text()),
                    training_subset_weight_checks=4320*6,all_training_lineages_equal_total_weight=True,
                    neural_refit_note='NN deterministic refits use the same weighted adapter; independent finite-difference gradients and direct-objective checks are in preflight. Ridge/XGBoost reference refits use separate sklearn APIs.',
                    exact_splits_and_initialization_seeds_verified=True,training_only_scaling_verified=True,
                    source_hashes_verified=True)
    run.dump(OUT/'validation.json',validation)
    return preds,parameters,validation,protocol

def evaluate(preds,params,validation,protocol):
    s=run.STATE;y=s['y'];lineage=s['lineage'];repeats=list(range(1,21))
    scores,folds,macro=old.score_predictions(y,lineage,preds,repeats)
    original=pd.read_csv(WS/'outputs/erastin_panel_comparison/per_repeat_metrics.tsv',sep='\t')
    for f in FAMILIES:
        for p in SIZES:
            np.testing.assert_allclose(scores[scores.model==name(f,p,False)].sort_values('repeat')[METRICS],
                original[original.model==f+'__'+str(p)].sort_values('repeat')[METRICS],rtol=0,atol=1e-12)
    table('per_repeat_metrics.tsv',scores);table('per_lineage_metrics.tsv',folds);table('per_repeat_macro_metrics.tsv',macro)
    rows=[]
    for scope,frame in [('pooled_745',scores),('macro_19',macro)]:
        for arm in preds:
            weighting,f,p=arm.split('__');row=dict(scope=scope,model=arm,weighting=weighting,family=f,panel=int(p))
            for metric in METRICS:
                v=frame[frame.model==arm][metric].to_numpy()
                row.update({metric+'_mean':v.mean(),metric+'_sd':v.std(ddof=1),metric+'_min':v.min(),metric+'_max':v.max()})
            rows.append(row)
    summary=table('summary.tsv',rows)
    run.dump(OUT/'progress.json',dict(phase='paired_lineage_bootstrap',new_models_audited=4320))
    draws=old.rc.repeated_group_bootstrap(y,preds,lineage,count=5000,seed=protocol['uncertainty']['pooled_seed'])
    mdraws=old.bootstrap_macro(folds,list(preds),repeats,protocol['uncertainty']['macro_seed'])
    paired,seeds=old.paired_comparisons(scores,macro,draws,mdraws,planned_pairs(),repeats)
    paired=table('paired_comparisons.tsv',paired);table('paired_seed_differences.tsv',seeds)
    effects=[];wins=[]
    for first,reference in planned_pairs():
        for heldout in s['labels']:
            aa=folds[(folds.model==first)&(folds.lineage==heldout)].sort_values('repeat')
            bb=folds[(folds.model==reference)&(folds.lineage==heldout)].sort_values('repeat')
            row=dict(first_model=first,reference_model=reference,lineage=str(heldout),n=int(aa.test_n.iloc[0]))
            for metric in METRICS:
                diff=(-1 if metric=='rmse' else 1)*(aa[metric].to_numpy()-bb[metric].to_numpy())
                row[metric+'_improvement']=np.mean(diff);row[metric+'_favorable_repeats']=int(np.sum(diff>0))
            effects.append(row)
        eligible=[v for v in effects if v['first_model']==first and v['reference_model']==reference and v['n']>=10]
        wins.append(dict(first_model=first,reference_model=reference,eligible_lineages=19,
                         **{metric+'_favorable_lineages':sum(v[metric+'_improvement']>0 for v in eligible) for metric in METRICS}))
    effects=table('lineage_paired_effects.tsv',effects);wins=table('lineage_win_counts.tsv',wins)
    counts=[]
    for f in FAMILIES:
        columns=['alpha'] if f=='ridge' else ['width','alpha'] if f=='neural_network' else ['max_depth','min_child_weight','reg_lambda','num_boost_round']
        frame=params[params.family==f].groupby(['model','panel']+columns).size().reset_index(name='outer_fits')
        counts.extend(frame.to_dict('records'))
    counts=table('parameter_counts.tsv',counts)
    np.savez_compressed(OUT/'predictions.npz',sample_ids=s['ids'],group_ids=s['groups'],lineage=lineage,y=y,inner_split_seeds=s['seeds'],**preds)
    np.savez_compressed(OUT/'bootstrap_draws.npz',**{scope+'__'+arm+'__'+metric:v for scope,d in [('pooled',draws),('macro',mdraws)] for arm,mm in d.items() for metric,v in mm.items()})
    final=dict(summary=summary.to_dict('records'),validation_passed=True,warning_count=validation['warning_count'],
               lineage_wins=wins.to_dict('records'),distinct_prediction_vectors={arm:len(np.unique(v,axis=0)) for arm,v in preds.items()})
    run.dump(OUT/'final_summary.json',final)
    render(summary,paired,effects,wins,counts,validation)
    for source in json.loads((OUT/'source_manifest.json').read_text())['files']:assert run.sha(WS/source['path'])==source['sha256']
    analysis_sources=json.loads((OUT/'analysis_source_manifest.json').read_text())
    assert run.sha(Path(__file__))==analysis_sources['script_sha256']
    assert run.sha(OUT/'report_weighted.py')==analysis_sources['report_sha256']
    run.dump(OUT/'progress.json',dict(phase='complete',new_models_audited=4320,validation_passed=True))
    archive=seal();print(json.dumps(dict(event='complete',archive_bytes=archive.stat().st_size,summary=run.xp.clean(final)),indent=2),flush=True)

def seal():
    manifest=dict(files=[dict(path=str(p.relative_to(OUT)),sha256=run.sha(p),bytes=p.stat().st_size)
                         for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts])
    run.dump(OUT/'artifact_manifest.json',manifest)
    archive=OUT.parent/'erastin_lineage_weighted_results.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for row in manifest['files']:z.write(OUT/row['path'],'erastin_lineage_weighted/'+row['path'])
        z.write(OUT/'artifact_manifest.json','erastin_lineage_weighted/artifact_manifest.json')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in manifest['files']:
            import hashlib
            assert hashlib.sha256(z.read('erastin_lineage_weighted/'+row['path'])).hexdigest()==row['sha256']
    return archive

def render(summary,paired,effects,wins,counts,validation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter
    colors={'ridge':'#137d8b','neural_network':'#8853a3','xgboost':'#bf6d22'}
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for i,scope in enumerate(['pooled_745','macro_19']):
        for j,metric in enumerate(['spearman','rmse']):
            ax=axes[i,j]
            for f in FAMILIES:
                for weight,style in [('unweighted','--'),('weighted','-')]:
                    v=summary[(summary.scope==scope)&(summary.family==f)&(summary.weighting==weight)].sort_values('panel')
                    ax.errorbar(v.panel,v[metric+'_mean'],yerr=v[metric+'_sd'],marker='o',capsize=2,color=colors[f],linestyle=style,label=LABELS[f]+' / '+weight)
            ax.set_xticks(SIZES);ax.set_xlabel('Number of genes');ax.set_ylabel('Spearman ↑' if metric=='spearman' else 'RMSE ↓')
            ax.set_title('Pooled 745 cell lines' if i==0 else 'Equal mean across 19 lineages (n≥10)')
            ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
            if metric=='rmse':ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.4f}'))
    handles,names=axes[0,0].get_legend_handles_labels();fig.legend(handles,names,loc='outside lower center',ncol=3,fontsize=9)
    fig.suptitle('Inverse-lineage-frequency training weights: 20 matched repeats\nWhiskers show seed SD on the same biological cohort.',fontsize=13)
    fig.savefig(OUT/'weighted_comparison.png',dpi=170);fig.savefig(OUT/'weighted_comparison.svg');plt.close(fig)
    selected=effects[effects.n>=10];lineages=selected.drop_duplicates('lineage').sort_values('n',ascending=False).lineage.tolist()
    fig,axes=plt.subplots(1,2,figsize=(14,9),constrained_layout=True,sharey=True)
    contrasts=[(name(f,p),name(f,p,False)) for f in FAMILIES for p in SIZES]
    for ax,metric in zip(axes,['spearman','rmse']):
        a=np.column_stack([selected[(selected.first_model==first)&(selected.reference_model==reference)].set_index('lineage').loc[lineages,metric+'_improvement'] for first,reference in contrasts])
        limit=float(np.max(np.abs(a)));im=ax.imshow(a,aspect='auto',cmap='RdBu',vmin=-limit,vmax=limit)
        ax.set_xticks(range(9),[LABELS[f]+'\n'+str(p) for f in FAMILIES for p in SIZES],rotation=55,ha='right',fontsize=8)
        ax.set_title('Spearman increase' if metric=='spearman' else 'RMSE reduction');fig.colorbar(im,ax=ax,shrink=.75,pad=.02)
    axes[0].set_yticks(range(19),lineages)
    fig.suptitle('Weighted minus original unweighted performance on unseen lineages\nMean paired effect across 20 repeats; blue favors weighting',fontsize=13)
    fig.savefig(OUT/'lineage_weighting_effects.png',dpi=170);fig.savefig(OUT/'lineage_weighting_effects.svg');plt.close(fig)
    write_report(summary,paired,effects,wins,counts,validation)

def record_weights(rows,group_rows,repeat,holdout,stage,inner_fold,audit):
    context=dict(repeat=repeat,holdout_lineage=holdout,stage=stage,inner_fold=inner_fold)
    rows.append(dict(**context,**{k:v for k,v in audit.items() if k!='groups'}))
    group_rows.extend(dict(**context,**v) for v in audit['groups'])

def write_report(summary,paired,effects,wins,counts,validation):
    from report_weighted import write_report as writer
    writer(summary,paired,effects,wins,counts,validation)

if __name__=='__main__':
    predictions,parameters,validation,protocol=audit();evaluate(predictions,parameters,validation,protocol)
