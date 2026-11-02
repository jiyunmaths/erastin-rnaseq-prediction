#!/usr/bin/env python3
"""Audit extended penalties and separate regularization from added-gene value."""
import argparse
from collections import Counter
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import common as c
from analysis_helpers import evaluate
from evaluate import validate_bootstrap
from make_splits import inner_rows,outer_split,inner_folds

LABELS=dict(nn64='Original core64 NN',core_extended='Core64 NN: extended penalties',
    old_optional='Previous optional additions NN',optional_extended='Optional additions NN: extended penalties',
    old_forced='Previous forced additions NN',forced_extended='Forced additions NN: extended penalties',
    tab64='Core64 TabPFN',training_mean='Training mean')


def table(name,data):
    d=data if isinstance(data,pd.DataFrame) else pd.DataFrame(data)
    d.to_csv(c.OUT/name,sep='\t',index=False,float_format='%.12g');return d


def audit(assay):
    s=c.STATE[assay];y,lin,groups,test=s['y'],s['lineage'],s['groups'],s['test']
    old=np.load(c.OLD/assay/'predictions.npz');forced=np.load(c.OLD/'forced_additions'/assay/'predictions.npz')
    for a in [old,forced]:
        for k,v in [('sample_ids',s['ids']),('group_ids',groups),('lineage',lin),('y',y),('outer_test',test)]:np.testing.assert_array_equal(a[k],v)
    preds=dict(nn64=old['nn64'].copy(),old_optional=old['enet_selected_nn'].copy(),old_forced=forced['enet_forced_nn'].copy(),tab64=old['tab64'].copy())
    preds.update({f:np.full(test.shape,np.nan) for f in c.ARMS+['training_mean']})
    grid=c.grid_for(assay);n,no=len(s['nn_grid']),len(s['old_nn_grid']);oldcols=c.old_columns(assay)
    newcols=np.setdiff1d(np.arange(len(grid)),oldcols)
    params,occ,objectives=[],[],[];warnings=Counter();score_checks=reused_scores=new_ensembles=new_networks=0
    max_pred=max_loss=max_objective=0.
    for r,mask in enumerate(test):
        tr,te=np.flatnonzero(~mask),np.flatnonzero(mask)
        np.testing.assert_array_equal(mask,outer_split(lin,groups,s['seeds'][r]))
        np.testing.assert_array_equal(s['inner'][r],inner_folds(lin,groups,mask,s['inner_seeds'][r]))
        assert set(groups[tr]).isdisjoint(groups[te]) and set(lin[tr])==set(lin[te])==set(lin)
        done=json.load(open(c.OUT/assay/'folds'/f'repeat_{r+1:02d}_complete.json'));assert done['protocol_sha256']==c.sha(c.OUT/'protocol.json')
        for row in done['files']:assert c.sha(c.OUT/row['path'])==row['sha256']
        tune=np.load(c.OUT/assay/'folds'/f'tuning__r{r+1:02d}.npz');val,loss=tune['inner_validation_predictions'],tune['inner_fold_rmse']
        source=np.load(c.OLD/assay/'folds'/f'enet_selected_nn__r{r+1:02d}.npz')
        np.testing.assert_array_equal(val[oldcols],source['inner_validation_predictions'])
        valid=s['inner'][r]>=0;assert np.isfinite(val[:,valid]).all() and np.isnan(val[:,~valid]).all()
        cache=c.cache(assay,r)
        for fold in range(5):
            aa,b=inner_rows(s['inner'][r],fold);root=c.OUT/assay/'folds'/f'inner__r{r+1:02d}__f{fold}'
            m=json.load(open(root.with_suffix('.json')));a=np.load(root.with_suffix('.npz'))
            original=np.load(c.OLD/assay/'folds'/f'inner__r{r+1:02d}__f{fold}.npz')
            np.testing.assert_array_equal(a['training_rows'],aa);np.testing.assert_array_equal(a['validation_rows'],b)
            np.testing.assert_array_equal(a['reused_columns'],oldcols);np.testing.assert_array_equal(a['new_columns'],newcols)
            np.testing.assert_array_equal(a['validation_predictions'],val[:,b])
            np.testing.assert_array_equal(a['validation_predictions'][oldcols],original['validation_predictions'])
            np.testing.assert_array_equal(a['member_validation_predictions'][oldcols],original['member_validation_predictions'])
            np.testing.assert_allclose(a['member_validation_predictions'][newcols].mean(axis=1),val[newcols[:,None],b],atol=1e-14,rtol=0)
            assert set(aa)|set(b)==set(tr) and set(groups[aa]).isdisjoint(groups[b])
            ll=np.sqrt(np.mean((val[:,b]-y[b][None,:])**2,axis=1));error=float(np.max(np.abs(ll-loss[:,fold])));max_loss=max(max_loss,error);assert error<1e-12
            np.testing.assert_allclose(ll,a['fold_rmse'],atol=1e-12,rtol=0);score_checks+=len(grid);reused_scores+=len(oldcols)
            assert len(m['diagnostics'])==len(newcols)
            for j,d in enumerate(m['diagnostics']):
                idx=int(newcols[j]);setting=grid[idx];assert d['candidate']==idx and d['selector']==setting['selector']
                assert d['params']=={k:setting[k] for k in ['width','alpha']}
                assert [v['seed'] for v in d['members']]==s['nn_seeds'][r].tolist()
                new_ensembles+=1;new_networks+=len(d['members'])
                for v in d['members']:
                    assert np.isfinite(v['training_loss']) and v['iterations']>0
                    warnings.update(w['category'] for w in v['warnings'])
        means=loss.mean(axis=1)
        chosen=dict(core_extended=int(np.argmin(means[:n])),optional_extended=int(np.argmin(means)),forced_extended=n+int(np.argmin(means[n:])))
        oldloss=source['inner_fold_rmse'].mean(axis=1)
        oldmin=dict(core_extended=np.min(oldloss[:no]),optional_extended=np.min(oldloss),forced_extended=np.min(oldloss[no:]))
        for arm in c.ARMS:
            root=c.OUT/assay/'folds'/f'{arm}__r{r+1:02d}';m=json.load(open(root.with_suffix('.json')));a=np.load(root.with_suffix('.npz'));model=joblib.load(root.with_suffix('.joblib'))
            assert m['selected_candidate']==chosen[arm] and m['selected_parameters']==grid[chosen[arm]]
            assert means[chosen[arm]]<=oldmin[arm]+1e-12
            setting=grid[chosen[arm]];sel=setting['selector']
            extra=c.INDEX['candidate_indices'][cache['outer_selection_masks'][sel]] if sel>=0 else np.array([],dtype=int)
            indices=np.r_[c.INDEX['core_indices'],extra]
            np.testing.assert_array_equal(a['feature_indices'],indices);np.testing.assert_array_equal(model['feature_indices'],indices)
            np.testing.assert_array_equal(a['train_rows'],tr);np.testing.assert_array_equal(a['test_rows'],te)
            if arm=='core_extended':assert sel==-1 and len(extra)==0
            if arm=='forced_extended':assert sel>=0 and len(extra)>0
            assert len(indices)==len(np.unique(indices)) and len(model['members'])==3
            x=c.features(assay,indices);sx=StandardScaler().fit(x[tr]);sy=StandardScaler().fit(y[tr,None])
            for attr in ['mean_','scale_']:
                np.testing.assert_allclose(getattr(sx,attr),getattr(model['x_scaler'],attr),atol=1e-12,rtol=0)
                np.testing.assert_allclose(getattr(sy,attr),getattr(model['y_scaler'],attr),atol=1e-12,rtol=0)
            target=sy.transform(y[tr,None]).ravel();members=[];norm=[]
            for j,v in enumerate(model['members']):
                assert v.random_state==int(s['nn_seeds'][r,j]) and v.alpha==setting['alpha'] and v.hidden_layer_sizes==(setting['width'],)
                assert v.activation=='tanh' and v.solver=='lbfgs';np.testing.assert_array_equal(v.training_sample_weight_,np.ones(len(tr)))
                members.append(sy.inverse_transform(v.predict(sx.transform(x[te]))[:,None]).ravel())
                squared_norm=sum(np.sum(t*t) for t in v.coefs_);norm.append(np.sqrt(squared_norm))
                objective=.5*np.mean((v.predict(sx.transform(x[tr]))-target)**2)+.5*v.alpha*squared_norm/len(tr)
                error=float(abs(objective-v.loss_));max_objective=max(max_objective,error);assert error<1e-10
                objectives.append(dict(assay=assay,repeat=r+1,model=arm,member=j,training_loss=v.loss_,recalculation_error=error,iterations=v.n_iter_))
            np.testing.assert_allclose(members,a['member_test_predictions'],atol=1e-12,rtol=0)
            pred=np.mean(members,axis=0);error=float(np.max(np.abs(pred-a['test_prediction'])));max_pred=max(max_pred,error);assert error<1e-12
            preds[arm][r,te]=pred
            d=m['fit_diagnostics']
            if 'members' in d:
                for v in d['members']:warnings.update(w['category'] for w in v['warnings'])
            enmeta=json.load(open(c.ENET/assay/'folds'/f'enet_selected_ridge__r{r+1:02d}.json'))['outer_diagnostics']['solutions'][sel] if sel>=0 else {}
            params.append(dict(assay=assay,repeat=r+1,model=arm,selector=sel,width=setting['width'],nn_alpha=setting['alpha'],
                l1_ratio=enmeta.get('l1_ratio',np.nan),alpha_fraction=enmeta.get('alpha_fraction',np.nan),added_genes=len(extra),total_genes=len(indices),
                prediction_sd=float(np.std(pred)),mean_weight_norm=float(np.mean(norm)),mean_inner_rmse=float(means[chosen[arm]])))
            if sel>=0:
                for idx in extra:
                    pos=np.searchsorted(c.INDEX['candidate_indices'],idx)
                    occ.append(dict(assay=assay,repeat=r+1,model=arm,feature_index=int(idx),gene=c.INDEX['genes'][idx],entrez_id=c.INDEX['entrez_ids'][idx],selector_coefficient=cache['outer_selector_coefficients'][sel,pos]))
        if grid[chosen['optional_extended']]['selector']==-1:
            np.testing.assert_allclose(preds['optional_extended'][r,te],preds['core_extended'][r,te],atol=1e-12,rtol=0)
        preds['training_mean'][r,te]=y[tr].mean()
        if (r+1)%5==0:print('AUDITED',assay,r+1,flush=True)
    assert warnings.get('ConvergenceWarning',0)==0,warnings
    np.savez_compressed(c.OUT/assay/'predictions.npz',sample_ids=s['ids'],group_ids=groups,lineage=lin,y=y,outer_test=test,**preds)
    bv=validate_bootstrap(y,preds,lin,test,2026112503)
    v=dict(passed=True,assay=assay,group_split_checks=120,candidate_fold_scores_checked=score_checks,
        reused_candidate_fold_scores=reused_scores,new_inner_ensembles=new_ensembles,new_inner_networks=new_networks,outer_training_objectives_checked=180,
        prediction_max_error=max_pred,inner_rmse_max_error=max_loss,training_objective_max_error=max_objective,
        warning_counts=dict(warnings),convergence_warning_count=warnings.get('ConvergenceWarning',0),
        all_prior_candidates_reused_exactly=True,all_core64_inputs_retained=True,matched_extended_control_verified=True,
        all_selected_inner_losses_at_most_old_grid_minimum=True,bootstrap_arithmetic=bv)
    c.dump(c.OUT/assay/'validation.json',v);return preds,params,occ,objectives,v


def gene_summary(occ):
    data=table('gene_selection_occurrences.tsv',pd.DataFrame(occ,columns=['assay','repeat','model','feature_index','gene','entrez_id','selector_coefficient']))
    rows=[]
    for (assay,model,idx),g in data.groupby(['assay','model','feature_index']):
        rows.append(dict(assay=assay,model=model,feature_index=int(idx),gene=g.gene.iloc[0],entrez_id=g.entrez_id.iloc[0],selected_fits=len(g),selection_fraction=len(g)/20,
            mean_absolute_selector_coefficient=float(g.selector_coefficient.abs().mean()),selector_sign_consistency=float(max((g.selector_coefficient>0).mean(),(g.selector_coefficient<0).mean()))))
    genes=pd.DataFrame(rows,columns=['assay','model','feature_index','gene','entrez_id','selected_fits','selection_fraction','mean_absolute_selector_coefficient','selector_sign_consistency'])
    genes=genes.sort_values(['assay','model','selected_fits','mean_absolute_selector_coefficient'],ascending=[True,True,False,False]);table('gene_selection_stability.tsv',genes);return genes


def render(summary,paired,wins,params,genes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    comparisons=[('optional_extended','core_extended','Optional vs extended core64'),('forced_extended','core_extended','Forced vs extended core64'),
        ('core_extended','nn64','Extended core64 vs original'),('forced_extended','old_forced','Extended forced vs previous')]
    fig,axes=plt.subplots(2,2,figsize=(12,7))
    for row,assay in enumerate(c.ASSAYS):
        for col,key in enumerate(['pooled_rmse','pooled_spearman']):
            ax=axes[row,col]
            for j,(first,ref,label) in enumerate(comparisons):
                d=paired[(paired.assay==assay)&(paired.first_model==first)&(paired.reference_model==ref)&(paired.metric==key)].iloc[0]
                ax.plot([d.conditional_ci_low,d.conditional_ci_high],[j,j],color='#28768e',lw=2);ax.scatter(d.improvement,j,color='#173e57',s=25,zorder=3)
            ax.axvline(0,color='gray',ls='--',lw=1);ax.set_yticks(range(len(comparisons)),[x[2] for x in comparisons] if col==0 else ['']*len(comparisons));ax.invert_yaxis()
            ax.set_title(assay+' | '+('RMSE reduction' if col==0 else 'Spearman increase'));ax.set_xlabel('Improvement over paired reference');ax.xaxis.set_major_locator(MaxNLocator(nbins=4));ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Stronger neural-network regularization: same core64 and fold-local EN selections',fontsize=13)
    fig.tight_layout(rect=(0,.045,1,.95));fig.text(.5,.01,'Conditional 95% paired lineage-bootstrap intervals; all 20 overlapping splits retained together. Exploratory.',ha='center',fontsize=9)
    fig.savefig(c.OUT/'model_comparison.png',dpi=180);fig.savefig(c.OUT/'model_comparison.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,4))
    for ax,assay in zip(axes,c.ASSAYS):
        for arm,marker in zip(c.ARMS,['o','s','^']):
            d=params[(params.assay==assay)&(params.model==arm)].sort_values('repeat')
            ax.plot(d['repeat'],d.nn_alpha,marker=marker,ms=4,lw=.8,label=arm.replace('_extended',''))
        ax.set_yscale('log');ax.set_title(assay);ax.set_xlabel('Split repeat');ax.set_ylabel('Selected NN L2 penalty');ax.set_xticks([1,5,10,15,20]);ax.grid(alpha=.2);ax.legend(fontsize=8);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Training-validation penalty choices');fig.tight_layout();fig.savefig(c.OUT/'selected_regularization.png',dpi=180);plt.close(fig)
    lines=['# Stronger NN regularization for Elastic Net-selected additions','',
        'The original core64 panel is mandatory. The same fold-local Elastic Net selections, outer/inner splits, three-network initialization seeds and NN architecture/solver are reused. Added L2 penalties are 300, 1,000, 3,000, 10,000 and 30,000; existing values are retained and reused exactly. No top-K gene cap is introduced.','',
        'The extended-grid core64-only control separates a general regularization benefit from the value of extra genes. The optional arm may select no additions; the forced arm selects the best nonempty setting. Both tune jointly inside the original five inner folds across all 20 seeds per assay.','',
        '## Matched-control findings','']
    lines += ['**Retain the core64 panel.** Extending the penalty grid improved the previous forced-expansion models, especially in CTRP, but did not establish lower error than the matched core64-only control. Every core-only fit selected alpha=100; every forced-expansion fit selected alpha=300. Neither end of the extended penalty grid was selected. These findings concern the tested EN/NN procedure and previously inspected cohorts.','']
    for assay in c.ASSAYS:
        for arm in ['optional_extended','forced_extended']:
            b=summary[(summary.assay==assay)&(summary.model=='core_extended')].iloc[0];n=summary[(summary.assay==assay)&(summary.model==arm)].iloc[0]
            d=paired[(paired.assay==assay)&(paired.first_model==arm)&(paired.reference_model=='core_extended')&(paired.metric=='pooled_rmse')].iloc[0]
            p=params[(params.assay==assay)&(params.model==arm)]
            change='unchanged' if abs(d.improvement)<1e-12 else f'{abs(100*d.improvement/b.pooled_rmse_mean):.2f}% '+('lower' if d.improvement>0 else 'higher')
            support='includes no difference' if d.conditional_ci_low<=0<=d.conditional_ci_high else 'favors improvement' if d.conditional_ci_low>0 else 'favors worse performance'
            lines += [f'**{assay}, {arm}:** RMSE {b.pooled_rmse_mean:.6f} -> {n.pooled_rmse_mean:.6f} ({change} error); {d.favorable_repeats}/20 favorable, {d.tied_repeats}/20 ties. Paired reduction {d.improvement:+.6f}, conditional 95% interval [{d.conditional_ci_low:+.6f}, {d.conditional_ci_high:+.6f}], which {support}.',
                f'Added genes: mean {p.added_genes.mean():.1f}, range {p.added_genes.min()}-{p.added_genes.max()}, zero-addition fits {(p.added_genes==0).sum()}/20.','']
    lines += ['## Improvement over the previous forced expansion','','| Assay | Previous forced NN RMSE | Extended-grid forced NN RMSE | Error reduction | Favorable repeats /20 | Conditional 95% RMSE-reduction interval |','|---|---:|---:|---:|---:|---:|']
    for assay in c.ASSAYS:
        b=summary[(summary.assay==assay)&(summary.model=='old_forced')].iloc[0];n=summary[(summary.assay==assay)&(summary.model=='forced_extended')].iloc[0]
        d=paired[(paired.assay==assay)&(paired.first_model=='forced_extended')&(paired.reference_model=='old_forced')&(paired.metric=='pooled_rmse')].iloc[0]
        lines.append(f'| {assay} | {b.pooled_rmse_mean:.6f} | {n.pooled_rmse_mean:.6f} | {100*d.improvement/b.pooled_rmse_mean:.2f}% | {d.favorable_repeats} | [{d.conditional_ci_low:+.6f}, {d.conditional_ci_high:+.6f}] |')
    lines += ['','For PRISM, 1,000 was already present in the original grid: alpha=300 fills the prior 100-to-1,000 gap. No fit selected a penalty above 300. Positive reductions favor the extended grid. These secondary comparisons do not demonstrate an added-gene gain over the core64 control.','',
        '## Across lineages','']
    for assay in c.ASSAYS:
        b=summary[(summary.assay==assay)&(summary.model=='core_extended')].iloc[0];n=summary[(summary.assay==assay)&(summary.model=='forced_extended')].iloc[0]
        d=paired[(paired.assay==assay)&(paired.first_model=='forced_extended')&(paired.reference_model=='core_extended')&(paired.metric=='macro_all_rmse')].iloc[0]
        w=wins[(wins.assay==assay)&(wins.first_model=='forced_extended')&(wins.reference_model=='core_extended')].iloc[0]
        lines += [f'**{assay}, forced additions:** equal-lineage RMSE {b.macro_all_rmse_mean:.6f} -> {n.macro_all_rmse_mean:.6f}; conditional reduction interval [{d.conditional_ci_low:+.6f}, {d.conditional_ci_high:+.6f}]. Mean RMSE improves in {w.favorable_lineages}/{w.lineages} lineages. These secondary intervals include no difference.','']
    lines += ['## Held-out performance','','| Assay | NN procedure | RMSE | Spearman | R2 | Equal-lineage RMSE |','|---|---|---:|---:|---:|---:|']
    for assay in c.ASSAYS:
        for f in LABELS:
            d=summary[(summary.assay==assay)&(summary.model==f)].iloc[0];rho=f'{d.pooled_spearman_mean:.4f}' if np.isfinite(d.pooled_spearman_mean) else 'undefined'
            lines.append(f'| {assay} | {LABELS[f]} | {d.pooled_rmse_mean:.6f} | {rho} | {d.pooled_r2_mean:.4f} | {d.macro_all_rmse_mean:.6f} |')
    lines += ['','All scores are means across the 20 saved holdouts; CTRP and PRISM endpoints remain separate.','',
        '![Matched comparisons](model_comparison.png)','','![Selected penalties](selected_regularization.png)','','## Selected genes and numerical diagnostics','',
        'Full choices are in `selected_parameters.tsv`. Training-only gene selection frequencies are in `gene_selection_stability.tsv`, separately for optional and forced arms. They are not NN feature importance, causal effects, individually validated contributions or independent replications. The selected gene set can vary between splits.','',
        'Prediction standard deviations and mean network-weight norms are saved with selected parameters, so nearly constant highly regularized predictors can be recognized. Grid-boundary counts are in `grid_boundary_audit.json`.','',
        '## Validation and scope','',
        '- Every old candidate is mapped by exact selector/width/penalty and reused unchanged. Only new penalties are fit. Sources are hash-verified against the completed previous benchmark. Source EN masks remain specific to each inner-training/outer-training subset; no consensus list is used.',
        '- NN centering/scaling uses current training only. Gene selection, penalty and width are jointly tuned by equal-mean five-fold RMSE. Core-only and forced choices use the corresponding subsets of the same validation grid.',
        '- Audits recompute every inner RMSE/argmin, verify reused predictions/member predictions, all outer feature sets, scaler parameters, identity exclusion, lineage coverage, seed triplets, unit weights, member predictions and training objectives. Preflight independently checks native stronger-penalty MLP fits and exact old candidate reproduction.',
        '- New fits abort on ConvergenceWarning. SciPy L-BFGS option deprecation warnings are counted separately; they do not change optimization. No penalty is added after examining these results.',
        '- Whole-lineage paired bootstrap preserves all models/repeats together. Intervals are conditional on fitted models, exclude retraining uncertainty and are not multiplicity-adjusted. Repeated holdouts overlap; these are not independent biological replications.',
        '- The expanded grid was motivated by prior results on these cohorts. This remains exploratory internal validation, despite fold-local selection/tuning. A final consensus panel and each individual gene contribution need separate validation. Patient prediction and unseen-lineage transfer are outside scope.',
        '- These EN selectors use linear conditional associations and can miss genes that act only through nonlinear interactions. The benchmark does not exhaust all gene subsets or neural-network architectures.',
        '- The original core64 panel and retained registry remain unchanged. Culture covariates and lineage prediction features are excluded. Protein-coding RNA coverage, finite grid and original endpoint provenance limitations remain.','',
        'Penalty units follow the training objective in the pinned implementation: the L2 term is divided by the training sample count. See [scikit-learn MLPRegressor documentation](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html).','']
    (c.OUT/'REVIEW.md').write_text('\n'.join(lines))


def seal():
    c.verify();assert json.load(open(c.OUT/'validation.json'))['passed'];assert json.load(open(c.OUT/'visual_qa.json'))['passed']
    for name,h in json.load(open(c.OUT/'analysis_source_manifest.json'))['scripts'].items():assert c.sha(c.OUT/name)==h
    c.dump(c.OUT/'progress.json',dict(phase='complete',validation_passed=True))
    files=[dict(path=str(p.relative_to(c.OUT)),sha256=c.sha(p),bytes=p.stat().st_size) for p in sorted(c.OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts and p.suffix!='.log']
    c.dump(c.OUT/'artifact_manifest.json',dict(files=files));print('SEALED',len(files),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');parser.add_argument('--seal',action='store_true');args=parser.parse_args()
    if args.seal:seal();return
    c.initialize();c.verify();protocol=json.load(open(c.OUT/'protocol.json'))
    c.dump(c.OUT/'analysis_source_manifest.json',dict(scripts={n:c.sha(c.OUT/n) for n in ['review_comparison.py','analysis_helpers.py','evaluate.py','metric_helpers.py']}))
    results=[];params=[];occ=[];objectives=[];valid=[]
    for assay in c.ASSAYS:
        if args.wait:
            print('WAITING_FOR',assay,flush=True)
            while len(list((c.OUT/assay/'folds').glob('repeat_*_complete.json')))<20:time.sleep(5)
        preds,pp,oo,tt,vv=audit(assay);results.append(evaluate(assay,preds,protocol));params+=pp;occ+=oo;objectives+=tt;valid.append(vv)
        print('ANALYSIS_COMPLETE',assay,flush=True)
    names=['summary.tsv','paired_comparisons.tsv','lineage_paired_effects.tsv','lineage_win_counts.tsv','per_repeat_pooled_metrics.tsv','per_lineage_metrics.tsv','per_repeat_macro_metrics.tsv']
    frames=[table(name,pd.concat([d[j] for d in results],ignore_index=True)) for j,name in enumerate(names)]
    params=table('selected_parameters.tsv',params);table('neural_training_objectives.tsv',objectives);genes=gene_summary(occ)
    table('parameter_counts.tsv',params.groupby(['assay','model','selector','l1_ratio','alpha_fraction','width','nn_alpha'],dropna=False).size().reset_index(name='outer_fits'))
    boundary=[]
    for assay in c.ASSAYS:
        for arm in c.ARMS:
            d=params[(params.assay==assay)&(params.model==arm)];alphas=[x['alpha'] for x in c.STATE[assay]['nn_grid']]
            boundary.append(dict(assay=assay,model=arm,zero_additions=int((d.added_genes==0).sum()),min_nn_alpha=int((d.nn_alpha==min(alphas)).sum()),max_nn_alpha=int((d.nn_alpha==max(alphas)).sum()),stronger_than_old_max=int((d.nn_alpha>max(x['alpha'] for x in c.STATE[assay]['old_nn_grid'])).sum())))
    c.dump(c.OUT/'grid_boundary_audit.json',dict(records=boundary,post_result_grid_expansion=False))
    c.dump(c.OUT/'validation.json',dict(passed=True,assays=valid,preflight=json.load(open(c.OUT/'preflight_validation.json')),all_core64_retained=True,sources_verified=True,training_only_selection=True))
    render(frames[0],frames[1],frames[3],params,genes);c.verify()
    c.dump(c.OUT/'final_summary.json',dict(validation_passed=True,summary=frames[0].to_dict('records')))
    c.dump(c.OUT/'progress.json',dict(phase='analysis_complete_pending_visual_qa',validation_passed=True));print('ALL_ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':main()
