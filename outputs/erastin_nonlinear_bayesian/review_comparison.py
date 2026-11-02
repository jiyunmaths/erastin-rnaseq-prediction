#!/usr/bin/env python3
"""Numerical audit and matched evaluation of spline Bayesian networks."""
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from scipy.integrate import quad_vec
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;WS=OUT.parent.parent
sys.path.insert(0,str(OUT))
import run_comparison as run
import spline_network as sn


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


old=module('previous_bayesian_review',WS/'outputs/erastin_bayesian_comparison/review_bayesian.py')
METRICS=['spearman','rmse','r2']
NAMES={'spline_cpd':'Nonlinear spline BN','linear_cpd':'Matched linear-CPD BN','gaussian_tan':'Previous Gaussian TAN',
       'ridge':'Ridge, 40 genes','neural_network':'Neural network, 40 genes'}


def table(name,rows):
    frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT/name,sep='\t',index=False,float_format='%.12g');return frame


def independent_cpd_check(model,x,y):
    z=np.column_stack([y,x]);scaler=StandardScaler().fit(z);z=scaler.transform(z)
    if model['family']=='linear_cpd':features=[z[:,[j]] for j in range(z.shape[1])]
    else:
        # New transformer fits; the saved basis is not used for this refit.
        from sklearn.preprocessing import SplineTransformer
        features=[]
        for j in range(z.shape[1]):
            knots=np.unique(np.quantile(z[:,j],[0,1/3,2/3,1]))
            b=SplineTransformer(degree=3,knots=knots[:,None],include_bias=False,extrapolation='linear').fit_transform(z[:,[j]])
            features.append(StandardScaler().fit_transform(b))
    coefficient_error=0.;noise_error=0.;intercept_error=0.
    for j,parent in enumerate(model['parent']):
        a=np.column_stack([features[0]]+([] if parent<0 else [features[parent+1]]))
        ref=Ridge(alpha=model['alpha'],solver='svd').fit(a,z[:,j+1])
        coefficient_error=max(coefficient_error,float(np.max(np.abs(ref.coef_-model['node_coefficients'][j]))))
        intercept_error=max(intercept_error,float(abs(ref.intercept_-model['intercept'][j])))
        var=max(np.mean((z[:,j+1]-ref.predict(a))**2),1e-6)
        noise_error=max(noise_error,float(abs(var-model['noise_variance'][j])))
    assert max(coefficient_error,intercept_error,noise_error)<1e-9
    return {'coefficient_error':coefficient_error,'intercept_error':intercept_error,'noise_variance_error':noise_error}


def independent_quadrature(model,x):
    # Explicit likelihood product, independently assembled gene by gene.
    z=(x-model['mean'][1:])/model['scale'][1:]
    residual=z-model['intercept']
    for j,parent in enumerate(model['parent']):
        if parent>=0:
            raw=model['bases'][parent+1]['spline'].transform(np.array([[z[parent]]]))
            phi=model['bases'][parent+1]['scaler'].transform(raw)[0]
            residual[j]-=phi@model['parent_coef'][j]
    def score(t):
        raw=model['bases'][0]['spline'].transform(np.array([[t]]))
        phi=model['bases'][0]['scaler'].transform(raw)[0]
        r=residual-model['response_coef']@phi
        return -.5*t*t-.5*np.sum(r*r/model['noise_variance'])
    prediction=sn.predict(model,x[None,:]);mz=(prediction['mean'][0]-model['mean'][0])/model['scale'][0]
    shift=max(score(t) for t in np.r_[np.linspace(-10,10,201),mz])
    def integrand(t):
        value=np.exp(score(t)-shift)
        return np.array([value,t*value,t*t*value])
    knots=model['bases'][0]['knots'];bounds=np.r_[-np.inf,knots,np.inf]
    result=np.zeros(3);error=0.
    for a,b in zip(bounds[:-1],bounds[1:]):
        v,e=quad_vec(integrand,a,b,epsabs=1e-10,epsrel=1e-10)
        result+=v;error+=e
    mean=result[1]/result[0];variance=result[2]/result[0]-mean*mean
    logz=shift+np.log(result[0])+.5*np.sum(residual*residual/model['noise_variance'])
    checks={'prediction_error':float(abs(model['mean'][0]+model['scale'][0]*mean-prediction['mean'][0])),
            'variance_error':float(abs(model['scale'][0]**2*variance-prediction['variance'][0])),
            'log_normalizer_error':float(abs(logz-prediction['logz_standardized'][0])),'scipy_integral_error_estimate':float(error)}
    assert max(checks[k] for k in ['prediction_error','variance_error','log_normalizer_error'])<1e-7
    return checks


def main():
    run.initialize();s=run.STATE;x,y=s['x'],s['y'];labels=s['labels'];lineage=s['lineage']
    protocol=json.loads((OUT/'protocol.json').read_text());ph=run.sha(OUT/'protocol.json')
    for name,h in protocol['scripts'].items():assert run.sha(OUT/name)==h
    # Verify the analysis helpers against their sealed preceding run.
    root=WS/'outputs/erastin_bayesian_comparison'
    known={v['path']:v['sha256'] for v in json.loads((root/'artifact_manifest.json').read_text())['files']}
    assert run.sha(root/'review_bayesian.py')==known['review_bayesian.py']
    run.dump(OUT/'analysis_source_manifest.json',{'files':[{'path':str((root/'review_bayesian.py').relative_to(WS)),'sha256':known['review_bayesian.py']}]})
    predictions=np.full((20,2,745),np.nan);variances=np.full_like(predictions,np.nan);logpdf=np.full_like(predictions,np.nan)
    parameters=[];tuning=[];numerical=[];reference_checks=[];quadrature=[];edge_rows=[]
    max_pred=max_var=max_logpdf=max_fine=max_finevar=max_loss=0.
    for r in range(20):
        seen=np.zeros(745,dtype=int)
        for l,label in enumerate(labels):
            token=f'repeat_{r+1:02d}__lineage_{l:02d}'
            meta=json.loads((OUT/'folds'/(token+'.json')).read_text())
            assert meta['protocol_sha256']==ph
            for ext,key in [('npz','array_sha256'),('joblib','model_sha256')]:assert run.sha(OUT/'folds'/(token+'.'+ext))==meta[key]
            data=np.load(OUT/'folds'/(token+'.npz'),allow_pickle=False);models=joblib.load(OUT/'folds'/(token+'.joblib'))
            tr,te=data['train_rows'],data['test_rows'];inner=s['inner'][r,l]
            assert set(te)==set(np.flatnonzero(lineage==label)) and set(tr)==set(np.flatnonzero(inner>=0))
            assert set(s['groups'][tr]).isdisjoint(s['groups'][te]);seen[te]+=1
            np.testing.assert_array_equal(models[0]['edges'],models[1]['edges'])
            if r==0:
                prep=sn.prepare(x[tr],y[tr])
                for model in models:
                    np.testing.assert_array_equal(model['parent'],prep['parent'])
                    for k in ['mean','scale']:np.testing.assert_allclose(model[k],prep[k],atol=1e-12,rtol=0)
                    for a,b in zip(model['bases'],prep['bases']):
                        np.testing.assert_array_equal(a['knots'],b['knots'])
                        np.testing.assert_allclose(a['scaler'].mean_,b['scaler'].mean_,atol=1e-12,rtol=0)
                        np.testing.assert_allclose(a['scaler'].scale_,b['scaler'].scale_,atol=1e-12,rtol=0)
            for m,model in enumerate(models):
                assert model['repeat']==r+1 and model['lineage']==label and model['protocol_sha256']==ph
                assert set(model['train_rows'])==set(tr) and set(model['test_rows'])==set(te)
                scaler=StandardScaler().fit(np.column_stack([y[tr],x[tr]]))
                np.testing.assert_allclose(model['mean'],scaler.mean_,atol=1e-12,rtol=0)
                np.testing.assert_allclose(model['scale'],scaler.scale_,atol=1e-12,rtol=0)
                assert np.all(model['noise_variance']>=1e-6)
                graph=nx.DiGraph();graph.add_nodes_from(range(41));graph.add_edges_from((0,j+1) for j in range(40))
                graph.add_edges_from((int(pa)+1,j+1) for j,pa in enumerate(model['parent']) if pa>=0)
                assert nx.is_directed_acyclic_graph(graph) and len(graph.edges)==79
                chosen=int(data['selected_candidates'][m]);assert chosen==np.argmin(data['inner_fold_rmse'][m].mean(axis=1))
                assert model['alpha']==run.ALPHAS[chosen]
                pp=sn.predict(model,x[te]);lp=sn.conditional_logpdf(model,x[te],y[te],pp)
                max_pred=max(max_pred,float(np.max(np.abs(pp['mean']-data['test_predictions'][m]))))
                max_var=max(max_var,float(np.max(np.abs(pp['variance']-data['test_variances'][m]))))
                max_logpdf=max(max_logpdf,float(np.max(np.abs(lp-data['test_logpdf'][m]))))
                if m==1:
                    refined=sn.predict(model,x[te],minimum_order=max(64,pp['diagnostic']['order']))
                    max_fine=max(max_fine,float(np.max(np.abs(refined['mean']-pp['mean']))))
                    max_finevar=max(max_finevar,float(np.max(np.abs(refined['variance']-pp['variance']))))
                predictions[r,m,te]=pp['mean'];variances[r,m,te]=pp['variance'];logpdf[r,m,te]=lp
                parameters.append({'repeat':r+1,'model':model['family'],'lineage':label,'alpha':model['alpha'],
                    'mean_inner_rmse':data['inner_fold_rmse'][m,chosen].mean(),'quadrature_order':pp['diagnostic']['order'],
                    'train_n':len(tr),'test_n':len(te)})
                for c,alpha in enumerate(run.ALPHAS):
                    losses=[]
                    for k in range(5):
                        aa=tr[inner[tr]!=k];bb=tr[inner[tr]==k]
                        assert set(s['groups'][aa]).isdisjoint(s['groups'][bb])
                        losses.append(np.sqrt(np.mean((y[bb]-data['inner_validation_predictions'][m,c,bb])**2)))
                    max_loss=max(max_loss,float(np.max(np.abs(np.array(losses)-data['inner_fold_rmse'][m,c]))))
                    tuning.append({'repeat':r+1,'model':model['family'],'lineage':label,'alpha':alpha,'mean_inner_rmse':np.mean(losses)})
                if r in [0,19] and label in ['Lung','Thyroid']:
                    reference_checks.append({'repeat':r+1,'model':model['family'],'lineage':label,**independent_cpd_check(model,x[tr],y[tr])})
                    if m==1:
                        for row in te[np.unique([0,len(te)//2,len(te)-1])]:
                            quadrature.append({'repeat':r+1,'lineage':label,'sample_id':str(s['ids'][row]),**independent_quadrature(model,x[row])})
            numerical.extend({'repeat':r+1,'lineage':label,**v} for v in meta['numerical_diagnostics'])
            edge_rows.extend({'repeat':r+1,'lineage':label,'gene_a':s['genes'][i],'gene_b':s['genes'][j]} for i,j in models[0]['edges'])
        assert np.all(seen==1)
        if (r+1)%5==0:
            print(json.dumps({'event':'audit_progress','complete_repeats':r+1}),flush=True)
    assert max(max_pred,max_var,max_logpdf,max_loss)<1e-10 and max_fine<1e-7 and max_finevar<1e-7
    assert np.isfinite(predictions).all() and np.isfinite(variances).all() and np.isfinite(logpdf).all()
    params=table('selected_parameters.tsv',parameters);table('tuning_scores.tsv',tuning);nums=table('numerical_diagnostics.tsv',numerical)
    assert nums.converged.all()
    table('selected_gene_edges.tsv',edge_rows)
    nn=np.load(WS/'outputs/erastin_nn_repeated/predictions.npz',allow_pickle=False)
    bn=np.load(WS/'outputs/erastin_bayesian_comparison/predictions.npz',allow_pickle=False)
    assert np.array_equal(nn['sample_ids'],s['ids']) and np.array_equal(bn['sample_ids'],s['ids'])
    assert np.array_equal(nn['inner_split_seeds'],s['seeds']) and np.array_equal(bn['seeds'][1:],s['seeds'])
    preds={'spline_cpd':predictions[:,1],'linear_cpd':predictions[:,0],'ridge':nn['ridge'],'neural_network':nn['neural_network'],'gaussian_tan':bn['gaussian_tan'][1:]}
    scores,folds,macro=old.score_predictions(y,lineage,preds,list(range(1,21)))
    # Stored baseline scores must agree with the independently rescored arrays.
    saved=pd.read_csv(WS/'outputs/erastin_nn_repeated/per_repeat_metrics.tsv',sep='\t')
    for name in ['ridge','neural_network']:
        a=scores[scores.model==name].sort_values('repeat')[METRICS]
        b=saved[saved.model==name].sort_values('repeat')[METRICS]
        np.testing.assert_allclose(a,b,atol=1e-12,rtol=0)
    table('per_repeat_metrics.tsv',scores);table('per_lineage_metrics.tsv',folds);table('per_repeat_macro_metrics.tsv',macro)
    summaries=[]
    for scope,frame in [('pooled_745',scores),('macro_19',macro)]:
        for name in preds:
            v=frame[frame.model==name];row={'scope':scope,'model':name}
            for metric in METRICS:
                a=v[metric].to_numpy();row.update({metric+'_mean':a.mean(),metric+'_sd':a.std(ddof=1),metric+'_min':a.min(),metric+'_max':a.max()})
            summaries.append(row)
    summary=table('summary.tsv',summaries)
    run.dump(OUT/'progress.json',{'phase':'paired_lineage_bootstrap','models_audited':960})
    bootstrap=old.rc.repeated_group_bootstrap(y,preds,lineage,count=5000,seed=protocol['uncertainty']['pooled_seed'])
    macro_bootstrap=old.bootstrap_macro(folds,list(preds),range(1,21),protocol['uncertainty']['macro_seed'])
    pairs=[('spline_cpd',b) for b in ['ridge','neural_network','linear_cpd','gaussian_tan']]+[('linear_cpd','ridge'),('linear_cpd','neural_network')]
    paired,seeds=old.paired_comparisons(scores,macro,bootstrap,macro_bootstrap,pairs,list(range(1,21)))
    paired=table('paired_comparisons.tsv',paired);table('paired_seed_differences.tsv',seeds)
    effects=[];wins=[]
    for reference in ['ridge','neural_network','linear_cpd','gaussian_tan']:
        for label in labels:
            a=folds[(folds.model=='spline_cpd')&(folds.lineage==label)].sort_values('repeat')
            b=folds[(folds.model==reference)&(folds.lineage==label)].sort_values('repeat')
            row={'reference':reference,'lineage':label,'n':int(np.sum(lineage==label))}
            for metric in METRICS:row[metric+'_improvement']=np.mean((-1 if metric=='rmse' else 1)*(a[metric].to_numpy()-b[metric].to_numpy()))
            effects.append(row)
        subset=[v for v in effects if v['reference']==reference and v['n']>=10]
        wins.append({'reference':reference,'eligible_lineages':19,**{m+'_favorable_lineages':sum(v[m+'_improvement']>0 for v in subset) for m in METRICS}})
    effects=table('lineage_paired_effects.tsv',effects);table('lineage_win_counts.tsv',wins)
    counts=table('parameter_counts.tsv',params.groupby(['model','alpha']).size().reset_index(name='outer_fits'))
    probability=[]
    for r in range(20):
        for m,family in enumerate(run.FAMILIES):
            probability.append({'repeat':r+1,'model':family,'heldout_nll':-logpdf[r,m].mean(),
                'mean_predictive_sd':np.sqrt(variances[r,m]).mean()})
        p=bn['gaussian_tan'][r+1];var=bn['gaussian_tan_variance'][r+1]
        probability.append({'repeat':r+1,'model':'gaussian_tan','heldout_nll':np.mean(.5*(np.log(2*np.pi*var)+(y-p)**2/var)),
                            'mean_predictive_sd':np.sqrt(var).mean()})
    table('per_repeat_probability_scores.tsv',probability)
    prob=table('probability_summary.tsv',pd.DataFrame(probability).groupby('model',as_index=False)[['heldout_nll','mean_predictive_sd']].mean())
    validation={'passed':True,'selected_networks':960,'candidate_loss_sets':20*24*2*len(run.ALPHAS),
        'prediction_max_error':max_pred,'variance_max_error':max_var,'logpdf_max_error':max_logpdf,'inner_rmse_max_error':max_loss,
        'higher_order_quadrature_prediction_max_error':max_fine,'higher_order_quadrature_variance_max_error':max_finevar,
        'all_inner_and_outer_quadratures_converged':True,'inner_and_outer_numerical_diagnostics':len(nums),
        'independent_sklearn_node_regressions':reference_checks,'independent_scipy_posterior_integrations':quadrature,
        'matched_baseline_scores_verified':True,'scalers_knots_group_exclusion_and_acyclic_graphs_verified':True,
        'mathematical_checks':json.loads((OUT/'mathematical_validation.json').read_text())}
    assert len(reference_checks)==8 and len(quadrature)==12
    run.dump(OUT/'validation.json',validation)
    np.savez_compressed(OUT/'predictions.npz',sample_ids=s['ids'],groups=s['groups'],lineage=lineage,y=y,seeds=s['seeds'],**preds,
        **{f+'_variance':variances[:,m] for m,f in enumerate(run.FAMILIES)},**{f+'_logpdf':logpdf[:,m] for m,f in enumerate(run.FAMILIES)})
    np.savez_compressed(OUT/'bootstrap_draws.npz',**{scope+'__'+model+'__'+metric:v for scope,bd in [('pooled',bootstrap),('macro',macro_bootstrap)]
        for model,md in bd.items() for metric,v in md.items()})
    final={'summary':summary.to_dict('records'),'lineage_wins':wins,'parameter_counts':counts.to_dict('records'),
        'probability_scores':prob.to_dict('records'),'distinct_prediction_vectors':{f:len(np.unique(predictions[:,m],axis=0)) for m,f in enumerate(run.FAMILIES)},
        'validation_passed':True}
    run.dump(OUT/'final_summary.json',final)
    render(summary,paired,effects,counts,prob,validation,final)
    for source in json.loads((OUT/'source_manifest.json').read_text())['files']:assert run.sha(WS/source['path'])==source['sha256']
    run.dump(OUT/'progress.json',{'phase':'complete','networks_audited':960,'validation_passed':True})
    archive=seal()
    print(json.dumps({'event':'complete','archive_bytes':archive.stat().st_size,'summary':run.clean(final)},indent=2),flush=True)


def seal():
    manifest={'files':[{'path':str(p.relative_to(OUT)),'sha256':run.sha(p),'bytes':p.stat().st_size}
                       for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts]}
    run.dump(OUT/'artifact_manifest.json',manifest)
    archive=OUT.parent/'erastin_nonlinear_bayesian_results.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for row in manifest['files']:z.write(OUT/row['path'],'erastin_nonlinear_bayesian/'+row['path'])
        z.write(OUT/'artifact_manifest.json','erastin_nonlinear_bayesian/artifact_manifest.json')
    import hashlib
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in manifest['files']:assert hashlib.sha256(z.read('erastin_nonlinear_bayesian/'+row['path'])).hexdigest()==row['sha256']
    return archive


def render(summary,paired,effects,counts,prob,validation,final):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    order=['spline_cpd','neural_network','ridge','linear_cpd','gaussian_tan']
    colors=['#b16d20','#8853a3','#137d8b','#9a8b6a','#7f9b95']
    fig,axes=plt.subplots(2,2,figsize=(11.5,7.5),constrained_layout=True,sharex='col')
    for i,scope in enumerate(['pooled_745','macro_19']):
        v=summary[summary.scope==scope].set_index('model').loc[order]
        for j,metric in enumerate(['spearman','rmse']):
            ax=axes[i,j];ax.scatter(v[metric+'_mean'],np.arange(5),s=55,color=colors,zorder=3)
            ax.errorbar(v[metric+'_mean'],np.arange(5),xerr=v[metric+'_sd'],fmt='none',color='#29363b',capsize=2)
            ax.set_yticks(np.arange(5),[NAMES[k] for k in order]);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
            ax.set_title(('Pooled 745 lines' if i==0 else 'Mean across 19 lineages')+' — '+('Spearman ↑' if j==0 else 'RMSE ↓'))
            ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Nonlinear Bayesian network on the 40-gene panel\n20 matched inner-fold seeds; fixed leave-lineage-out tests',fontsize=13)
    fig.supxlabel('Whiskers show seed SD; these repeats reuse the same biological cohort.',fontsize=9)
    fig.savefig(OUT/'model_comparison.png',dpi=170);fig.savefig(OUT/'model_comparison.svg');plt.close(fig)
    v=effects[effects.n>=10];lineages=v.drop_duplicates('lineage').sort_values('n',ascending=False).lineage
    fig,axes=plt.subplots(1,2,figsize=(11.5,8),constrained_layout=True,sharey=True)
    for ax,metric in zip(axes,['spearman','rmse']):
        for off,ref,color in [(-.17,'ridge','#137d8b'),(.17,'neural_network','#8853a3')]:
            vv=v[v.reference==ref].set_index('lineage').loc[lineages]
            ax.barh(np.arange(19)+off,vv[metric+'_improvement'],height=.3,color=color,label=NAMES[ref])
        ax.axvline(0,color='#555',lw=1);ax.grid(axis='x',alpha=.2)
        ax.set_xlabel('Spearman increase' if metric=='spearman' else 'RMSE reduction');ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(np.arange(19),lineages);axes[0].invert_yaxis()
    handles,legend_labels=axes[1].get_legend_handles_labels();fig.legend(handles,legend_labels,loc='outside lower center',ncol=2)
    fig.suptitle('Nonlinear Bayesian network by held-out lineage\nMean paired change; positive favors the spline network',fontsize=12)
    fig.savefig(OUT/'lineage_comparison.png',dpi=170);fig.savefig(OUT/'lineage_comparison.svg');plt.close(fig)
    point=summary[summary.scope=='pooled_745'].set_index('model')
    macro=summary[summary.scope=='macro_19'].set_index('model')
    rows=['# Nonlinear Bayesian network comparison','',
        '**Adding nonlinear spline conditional means did not improve prediction.** '
        f'The nonlinear network has {100*(point.loc["spline_cpd","rmse_mean"]/point.loc["ridge","rmse_mean"]-1):.2f}% higher pooled RMSE than ridge, '
        f'{100*(point.loc["spline_cpd","rmse_mean"]/point.loc["neural_network","rmse_mean"]-1):.2f}% higher than the neural network, and '
        f'{100*(point.loc["spline_cpd","rmse_mean"]/point.loc["linear_cpd","rmse_mean"]-1):.2f}% higher than its matched linear-CPD control. '
        'It trails all three on pooled RMSE and Spearman in every matched repeat. Paired conditional lineage-bootstrap intervals favor ridge and the neural network. '
        'The nonlinear functions therefore added no predictive benefit in this registered configuration on this cohort.','',
        'A nonlinear additive-spline Bayesian network was fitted and evaluated across the same 20 inner-fold seeds and 24 outer lineage holdouts. '
        'The data remain 745 cancer cell lines, 734 related-identity groups and the approved 40-gene expression panel. '
        'The ridge and neural-network predictions come from the exact matching saved splits.','',
        'The matched linear-CPD network uses the same graph, node-wise fitting scheme and tuning grid, with straight-line conditional means. '
        'This control separates the effect of adding nonlinear conditional means from changes in regularization and fitting relative to the earlier Gaussian TAN.','',
        '## Held-out results','',
        '| Model | Pooled Spearman ↑ | Pooled RMSE ↓ | Pooled R² ↑ | Macro Spearman ↑ | Macro RMSE ↓ |',
        '|---|---:|---:|---:|---:|---:|']
    for name in order:
        a=point.loc[name];b=macro.loc[name]
        rows.append(f'| {NAMES[name]} | {a.spearman_mean:.5f} | {a.rmse_mean:.6f} | {a.r2_mean:.5f} | {b.spearman_mean:.5f} | {b.rmse_mean:.6f} |')
    rows+=['','The table gives means of per-repeat scores, not the score of an ensemble averaging predictions across repeats. '
        'Macro scores weight the same 19 lineages with n≥10 equally; all 24 lineages remain in pooled scores.','',
        f'![Model comparison]({OUT}/model_comparison.png)','',
        '## Paired improvement of the nonlinear network','',
        'Positive means better. Intervals use 5,000 paired resamples of whole held-out lineages, keeping all models and repeats together.','',
        '| Scope | Reference | Metric | Mean improvement | Favorable repeats | Conditional 95% interval |',
        '|---|---|---|---:|---:|---|']
    for row in paired[(paired.first_model=='spline_cpd')&paired.metric.isin(['spearman','rmse'])].itertuples():
        rows.append(f'| {row.scope} | {NAMES[row.reference_model]} | {"Spearman increase" if row.metric=="spearman" else "RMSE reduction"} | '
                    f'{row.improvement:+.6f} | {row.favorable_repeats}/20 | [{row.conditional_ci_low:+.6f}, {row.conditional_ci_high:+.6f}] |')
    rows+=['','Among the 19 larger lineages, averaging effects over 20 repeats:','']
    for row in final['lineage_wins']:
        rows.append(f'- Versus {NAMES[row["reference"]]}: higher Spearman in {row["spearman_favorable_lineages"]}/19; lower RMSE in {row["rmse_favorable_lineages"]}/19.')
    rows+=['',f'![Lineage comparison]({OUT}/lineage_comparison.png)','',
        '## What makes this network nonlinear','',
        'Each gene has a Gaussian conditional distribution whose mean is an additive function of continuous sensitivity and, except for the gene-tree root, '
        'one gene parent: `X_j | Y, X_parent ~ Normal(intercept_j + f_j(Y) + g_j(X_parent), variance_j)`. '
        'The functions use cubic B-splines with four training-quantile knots, duplicate-knot handling, ridge regularization and linear extrapolation. '
        'Knots, basis scalers, response/gene scalers, graph and node parameters are all fitted separately within each current training subset.','',
        'At prediction time all 40 genes are observed. The model combines the likelihood factors with the training Gaussian response prior and integrates '
        'the resulting one-dimensional posterior over sensitivity. Its conditional posterior can be non-Gaussian and its prediction is not restricted to a linear function of expression. '
        'No target discretization, response clipping or finite response-range truncation is used.','',
        'The graph is a maximum spanning tree of Gaussian conditional gene mutual information given response, oriented from the fixed SLC7A11 root. '
        'The two new families share this graph within a training subset. Nonlinear structure search, alternative tree orientations and non-additive interactions were not tested. '
        'With nonlinear CPDs the choice of orientation can matter; the fixed orientation is a modeling assumption, not a causal conclusion.','',
        'Nonlinear Gaussian conditional means estimated with splines are also studied in the recent [semi-linear Gaussian BN preprint](https://arxiv.org/abs/2610.30838). '
        'This experiment is a custom TAN implementation, not a reproduction of that paper’s structure search or smoothing penalty. '
        'The basis implementation follows [SplineTransformer](https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.SplineTransformer.html); the installed library version is recorded in the protocol.','',
        '## Selected regularization and numerical accuracy','',
        'The common node penalty was tuned over [0, 1, 10, 100, 300, 1000, 3000, 10000, 1000000] by equal-weight mean five-fold inner RMSE. '
        'The grid and numerical criteria were fixed before inspecting this experiment’s held-out performance.','',
        '| Network | Alpha | Selected outer fits |','|---|---:|---:|']
    for row in counts.itertuples():rows.append(f'| {NAMES[row.model]} | {row.alpha:g} | {row.outer_fits}/480 |')
    rows+=['',f'Distinct full-cohort prediction vectors across the 20 repeats: nonlinear = {final["distinct_prediction_vectors"]["spline_cpd"]}, '
        f'linear control = {final["distinct_prediction_vectors"]["linear_cpd"]}. These algorithms are deterministic; seeds vary inner tuning assignments only.','',
        'Gaussian tails are integrated analytically because spline extrapolation is linear. Inside the spline knots, quadrature order doubles until posterior mean, '
        'variance and log normalization converge. Every selected nonlinear model was additionally checked at higher integration order.','',
        f'- Maximum change in predicted sensitivity under higher-order integration: {validation["higher_order_quadrature_prediction_max_error"]:.3g}.',
        f'- Maximum change in conditional variance: {validation["higher_order_quadrature_variance_max_error"]:.3g}.',
        '- All inner and outer numerical integrations converged; no unconverged predictions were accepted.',
        '- Independent sklearn Ridge refits checked every node for the first and last seeds in Lung and Thyroid.',
        '- Independent SciPy adaptive integration of the likelihood product checked three fixed sample IDs in each of those four nonlinear holdouts.',
        '- The zero-penalty linear limit reproduced the previous Gaussian BN mathematically, and a synthetic nonlinear relationship verified nonlinear prediction behavior.','',
        '## Conditional density scores','',
        'Negative log likelihood evaluates the fitted normalized response density, using the nonlinear posterior directly rather than substituting a Gaussian approximation. '
        'The models estimate parameters and structure; they do not average over posterior parameter uncertainty.','',
        '| Model | Mean held-out NLL ↓ | Mean conditional predictive SD |','|---|---:|---:|']
    for row in prob.itertuples():rows.append(f'| {NAMES[row.model]} | {row.heldout_nll:.5f} | {row.mean_predictive_sd:.5f} |')
    rows+=['','## Interpretation limits','',
        '- The same cohort and outer lineage tests have now supported several exploratory model comparisons. There is no new external drug screen or clinical validation.',
        '- Repeated seeds are not independent biological replications. Bootstrap intervals condition on the fitted predictions, do not refit models, and are not adjusted for multiple comparisons.',
        '- The same grouped inner folds are preserved; they mix represented training lineages. This study does not evaluate lineage-held-out inner tuning.',
        '- The network’s arrows are a statistical factorization. Learned dependencies and nonlinear shapes are not validated causal gene relationships.',
        '- The response prior is Gaussian, conditional gene variances are constant, functions are additive and each gene has at most one gene parent in addition to response. Other nonlinear Bayesian-network families remain untested.','',
        '## Files','',f'- [All scores]({OUT}/summary.tsv)',f'- [Paired comparisons]({OUT}/paired_comparisons.tsv)',
        f'- [Per-lineage effects]({OUT}/lineage_paired_effects.tsv)',f'- [Protocol]({OUT}/protocol.json)',f'- [Validation]({OUT}/validation.json)','']
    (OUT/'REVIEW.md').write_text('\n'.join(rows))


if __name__=='__main__':
    if '--render-only' in sys.argv:
        def read(name):return pd.read_csv(OUT/name,sep='\t')
        render(read('summary.tsv'),read('paired_comparisons.tsv'),read('lineage_paired_effects.tsv'),read('parameter_counts.tsv'),
            read('probability_summary.tsv'),json.loads((OUT/'validation.json').read_text()),json.loads((OUT/'final_summary.json').read_text()))
        print(json.dumps({'event':'rendered_and_resealed','archive_bytes':seal().stat().st_size}),flush=True)
    else:main()
