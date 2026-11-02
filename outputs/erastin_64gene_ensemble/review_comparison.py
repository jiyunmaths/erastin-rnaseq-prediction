#!/usr/bin/env python3
"""Audit retained-model blending and evaluate paired outer-test predictions."""
import argparse
from itertools import combinations
import json

import joblib
import numpy as np
import pandas as pd

import common as c
from models import value_gradient,kkt,KKT_LIMIT
from analysis_helpers import evaluate
from evaluate import validate_bootstrap
from make_splits import outer_split,inner_folds,inner_rows

LABELS=dict(ridge='Ridge',elastic_net='Elastic Net',neural_network='Neural network',equal_blend='Equal three-way mean',
    learned_blend='Learned convex blend',inner_best_single='Inner-selected single family',training_mean='Training mean')


def table(name,data):
    d=data if isinstance(data,pd.DataFrame) else pd.DataFrame(data)
    d.to_csv(c.OUT/name,sep='\t',index=False,float_format='%.12g');return d


def predict_base(model,family,x):
    if family=='ridge':
        assert len(model['mean'])==64 and len(model['coef'])==64
        return ((x-model['mean'])/model['scale'])@model['coef']+model['intercept_after_scaling']
    assert model['x_scaler'].n_features_in_==64
    z=model['x_scaler'].transform(x);sy=model['y_scaler']
    if family=='elastic_net':return sy.inverse_transform(model['model'].predict(z)[:,None]).ravel()
    assert len(model['members'])==3
    return np.mean([sy.inverse_transform(v.predict(z)[:,None]).ravel() for v in model['members']],axis=0)


def audit(assay):
    s=c.STATE[assay];y,lin,groups,test=s['y'],s['lineage'],s['groups'],s['test']
    baseline=np.load(s['source']/'predictions.npz')
    for key,val in [('sample_ids',s['ids']),('group_ids',groups),('lineage',lin),('y',y),('outer_test',test)]:np.testing.assert_array_equal(baseline[key],val)
    preds={family:baseline[family+'__64'].copy() for family in c.FAMILIES}
    preds.update({f:np.full(test.shape,np.nan) for f in c.ARMS+['training_mean']})
    weights=[];base_parameters=[];correlations=[];max_pred=max_base=max_inner=max_kkt=max_source_aggregate=0.;base_scores=0
    for r,mask in enumerate(test):
        tr,te=np.flatnonzero(~mask),np.flatnonzero(mask)
        np.testing.assert_array_equal(mask,outer_split(lin,groups,s['seeds'][r]))
        np.testing.assert_array_equal(s['inner'][r],inner_folds(lin,groups,mask,s['inner_seeds'][r]))
        assert set(groups[tr]).isdisjoint(groups[te]) and set(lin[tr])==set(lin[te])==set(lin)
        root=c.OUT/assay/'folds'/f'ensemble__r{r+1:02d}'
        a=np.load(root.with_suffix('.npz'));m=json.load(open(root.with_suffix('.json')))
        assert m['protocol_sha256']==c.sha(c.OUT/'protocol.json') and m['npz_sha256']==c.sha(root.with_suffix('.npz'))
        inner,outer,valid,settings=c.inputs(assay,r)
        np.testing.assert_array_equal(a['train_rows'],tr);np.testing.assert_array_equal(a['test_rows'],te)
        np.testing.assert_array_equal(a['inner_rows'],np.flatnonzero(valid))
        np.testing.assert_array_equal(a['inner_base_predictions'],inner);np.testing.assert_array_equal(a['test_base_predictions'],outer)
        assert not np.any(valid&mask) and np.all(s['inner'][r,valid]>=0)
        for j,family in enumerate(c.FAMILIES):
            src=c.source_root(assay,family,r);t=np.load(src.with_suffix('.npz'));meta=json.load(open(src.with_suffix('.json')))
            model=joblib.load(src.with_suffix('.joblib'))
            np.testing.assert_array_equal(t['train_rows'],tr);np.testing.assert_array_equal(t['test_rows'],te)
            record=s['registry']['models'][family]['outer_models'][r]
            assert record['selected_parameters']==meta['selected_parameters']
            loss=t['inner_fold_rmse'];val=t['inner_validation_predictions']
            assert np.isfinite(val[:,valid]).all() and np.isnan(val[:,~valid]).all()
            for fold in range(5):
                aa,b=inner_rows(s['inner'][r],fold)
                assert set(aa)|set(b)==set(tr) and set(groups[aa]).isdisjoint(groups[b])
                expected=np.sqrt(np.mean((val[:,b]-y[b][None,:])**2,axis=1))
                error=float(np.max(np.abs(expected-loss[:,fold])));max_inner=max(max_inner,error);assert error<1e-12
                base_scores+=len(expected)
            chosen=int(np.argmin(loss.mean(axis=1)));assert chosen==meta['selected_candidate']
            np.testing.assert_array_equal(inner[:,j],val[chosen,valid])
            error=float(np.max(np.abs(outer[:,j]-preds[family][r,te])));max_source_aggregate=max(max_source_aggregate,error)
            np.testing.assert_allclose(outer[:,j],preds[family][r,te],atol=1e-12,rtol=0)
            restored=predict_base(model,family,s['x'][te]);error=float(np.max(np.abs(restored-outer[:,j])));max_base=max(max_base,error);assert error<1e-10
            base_parameters.append(dict(assay=assay,repeat=r+1,family=family,selected_candidate=chosen,mean_inner_rmse=float(loss[chosen].mean()),parameters=json.dumps(meta['selected_parameters'],sort_keys=True)))
        w=a['weights'];np.testing.assert_array_equal(w,m['learned_weights'])
        assert w.min()>=0 and abs(w.sum()-1)<1e-12
        scale=float(y[valid].std());center=float(y[valid].mean());p=(inner-center)/scale;target=(y[valid]-center)/scale
        objective,gradient=value_gradient(w,p,target,s['inner'][r,valid]);error=kkt(w,gradient);max_kkt=max(max_kkt,error);assert error<KKT_LIMIT
        assert abs(objective*scale-m['weight_fit']['mean_inner_rmse'])<1e-12
        base_loss=np.array([value_gradient(row,inner,y[valid],s['inner'][r,valid])[0] for row in np.eye(3)])
        equal_loss=value_gradient(np.ones(3)/3,inner,y[valid],s['inner'][r,valid])[0]
        assert objective*scale<=min(equal_loss,base_loss.min())+1e-10
        best=int(np.argmin(base_loss));assert int(a['selected_single_family'])==best and m['selected_single_family']==c.FAMILIES[best]
        direct=np.column_stack([np.mean(outer,axis=1),sum(w[j]*outer[:,j] for j in range(3)),outer[:,best]])
        error=float(np.max(np.abs(direct-a['ensemble_test_predictions'])));max_pred=max(max_pred,error);assert error<1e-12
        assert np.all(direct>=outer.min(axis=1)[:,None]-1e-12) and np.all(direct<=outer.max(axis=1)[:,None]+1e-12)
        for j,arm in enumerate(c.ARMS):preds[arm][r,te]=direct[:,j]
        preds['training_mean'][r,te]=y[tr].mean()
        weights.append(dict(assay=assay,repeat=r+1,ridge=w[0],elastic_net=w[1],neural_network=w[2],selected_single_family=c.FAMILIES[best],
            ridge_inner_rmse=base_loss[0],elastic_net_inner_rmse=base_loss[1],neural_network_inner_rmse=base_loss[2],equal_inner_rmse=equal_loss,
            blend_inner_rmse=objective*scale,meta_training_rows=int(valid.sum()),protected_training_rows=int(np.sum(s['inner'][r]==-2)),standardized_kkt_residual=kkt(w,gradient)))
        for i,j in combinations(range(3),2):
            e1,e2=outer[:,i]-y[te],outer[:,j]-y[te]
            pc=float(np.corrcoef(outer[:,i],outer[:,j])[0,1]) if np.ptp(outer[:,i])>0 and np.ptp(outer[:,j])>0 else np.nan
            ec=float(np.corrcoef(e1,e2)[0,1]) if np.ptp(e1)>0 and np.ptp(e2)>0 else np.nan
            correlations.append(dict(assay=assay,repeat=r+1,first_family=c.FAMILIES[i],second_family=c.FAMILIES[j],prediction_correlation=pc,error_correlation=ec,prediction_difference_rms=float(np.sqrt(np.mean((outer[:,i]-outer[:,j])**2)))))
    np.savez_compressed(c.OUT/assay/'predictions.npz',sample_ids=s['ids'],group_ids=groups,lineage=lin,y=y,outer_test=test,**preds)
    bv=validate_bootstrap(y,preds,lin,test,2026112603)
    v=dict(passed=True,assay=assay,group_split_checks=120,retained_base_outer_predictions_reproduced=60,base_candidate_fold_scores_checked=base_scores,
        base_prediction_max_error=max_base,source_aggregate_prediction_max_difference=max_source_aggregate,ensemble_prediction_max_error=max_pred,base_inner_rmse_max_error=max_inner,blend_standardized_kkt_max=max_kkt,
        weights_nonnegative_sum_one=True,outer_test_labels_excluded_from_weight_fit=True,all_base_predictions_unchanged=True,
        equal_weight_and_model_selection_controls_verified=True,bootstrap_arithmetic=bv)
    c.dump(c.OUT/assay/'validation.json',v);return preds,weights,base_parameters,correlations,v


def render(summary,paired,wins,weights,weight_summary,correlations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    comparisons=[(a,b,LABELS[a]+' vs '+LABELS[b]) for a in ['equal_blend','learned_blend'] for b in c.FAMILIES]
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for row,assay in enumerate(c.ASSAYS):
        for col,key in enumerate(['pooled_rmse','pooled_spearman']):
            ax=axes[row,col]
            for j,(first,ref,label) in enumerate(comparisons):
                d=paired[(paired.assay==assay)&(paired.first_model==first)&(paired.reference_model==ref)&(paired.metric==key)].iloc[0]
                ax.plot([d.conditional_ci_low,d.conditional_ci_high],[j,j],color='#28768e',lw=2);ax.scatter(d.improvement,j,color='#173e57',s=25,zorder=3)
            ax.axvline(0,color='gray',ls='--',lw=1);ax.set_yticks(range(len(comparisons)),[x[2] for x in comparisons] if col==0 else ['']*len(comparisons));ax.invert_yaxis()
            ax.set_title(assay+' | '+('RMSE reduction' if col==0 else 'Spearman increase'));ax.set_xlabel('Improvement over paired base family');ax.xaxis.set_major_locator(MaxNLocator(nbins=4));ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Ensembling the retained core64 ridge, Elastic Net and neural network',fontsize=13)
    fig.tight_layout(rect=(0,.045,1,.95));fig.text(.5,.01,'Conditional 95% paired lineage-bootstrap intervals; 20 overlapping holdouts per assay. Exploratory.',ha='center',fontsize=9)
    fig.savefig(c.OUT/'model_comparison.png',dpi=180);fig.savefig(c.OUT/'model_comparison.svg');plt.close(fig)
    fig,axes=plt.subplots(2,1,figsize=(10,6),sharex=True)
    for ax,assay in zip(axes,c.ASSAYS):
        d=weights[weights.assay==assay].sort_values('repeat');bottom=np.zeros(len(d))
        for family,color in zip(c.FAMILIES,['#28768e','#e79f37','#5a9b69']):
            ax.bar(d['repeat'],d[family],bottom=bottom,color=color,label=LABELS[family],width=.8);bottom+=d[family].to_numpy()
        ax.set_title(assay);ax.set_ylim(0,1);ax.set_ylabel('Family weight');ax.spines[['top','right']].set_visible(False)
    axes[0].legend(ncol=3,loc='upper center',bbox_to_anchor=(.5,1.30),fontsize=9);axes[-1].set_xlabel('Split repeat');axes[-1].set_xticks(range(1,21))
    fig.suptitle('Blend weights fitted using inner-validation predictions only',fontsize=12);fig.tight_layout(rect=(0,0,1,.94));fig.savefig(c.OUT/'ensemble_weights.png',dpi=180);plt.close(fig)
    lines=['# Ensemble of the retained 64-gene models','',
        'All original 64 genes and the 120 saved outer ridge/Elastic Net/neural-network models are retained unchanged. The NN is its original three-network mean and contributes as one model family. CTRP and PRISM remain separate.','',
        'Two ensemble rules are tested: an equal three-family mean and a convex blend with nonnegative weights summing to one. The blend has no intercept, calibration or clipping and may assign zero to a family. An inner-selected single-family control tests whether blending helps beyond choosing a family.','',
        '## Findings','',
        '**No convincing gain over the retained neural network was established.** The equal mean has a negligible CTRP RMSE edge with an interval spanning zero and slightly higher mean PRISM error. Learned weights give higher mean error in both assays; PRISM is worse than the NN and the equal mean under the conditional intervals. Keep the retained registry unchanged.','',
        '| Assay | Ensemble | RMSE reduction vs NN | Conditional 95% interval | Favorable repeats /20 |','|---|---|---:|---:|---:|']
    for assay in c.ASSAYS:
        for arm in ['equal_blend','learned_blend']:
            d=paired[(paired.assay==assay)&(paired.first_model==arm)&(paired.reference_model=='neural_network')&(paired.metric=='pooled_rmse')].iloc[0]
            lines.append(f'| {assay} | {LABELS[arm]} | {d.improvement:+.6f} | [{d.conditional_ci_low:+.6f}, {d.conditional_ci_high:+.6f}] | {d.favorable_repeats} |')
    lines += ['', 'Both blends outperform the inner-selected single-family control under conditional RMSE intervals. That control itself has higher mean error than the retained NN. Outperformance of this selection-only control does not establish a gain over each retained family.','',
        '## Held-out performance','','| Assay | Procedure | RMSE | Spearman | R2 | Equal-lineage RMSE |','|---|---|---:|---:|---:|---:|']
    for assay in c.ASSAYS:
        for f in LABELS:
            d=summary[(summary.assay==assay)&(summary.model==f)].iloc[0];rho=f'{d.pooled_spearman_mean:.4f}' if np.isfinite(d.pooled_spearman_mean) else 'undefined'
            lines.append(f'| {assay} | {LABELS[f]} | {d.pooled_rmse_mean:.6f} | {rho} | {d.pooled_r2_mean:.4f} | {d.macro_all_rmse_mean:.6f} |')
    lines += ['','Scores are means across the unchanged 20 lineage-stratified identity-group holdouts per assay. All evaluated lineages appear in both outer training and testing.','',
        '![Paired comparisons](model_comparison.png)','','## Paired RMSE results','','| Assay | Ensemble | Reference | RMSE reduction | Conditional 95% interval | Favorable repeats /20 |','|---|---|---|---:|---:|---:|']
    for _,d in paired[paired.metric=='pooled_rmse'].iterrows():
        lines.append(f'| {d.assay} | {LABELS[d.first_model]} | {LABELS[d.reference_model]} | {d.improvement:+.6f} | [{d.conditional_ci_low:+.6f}, {d.conditional_ci_high:+.6f}] | {d.favorable_repeats} |')
    lines += ['','Positive reductions favor the ensemble. Intervals including zero do not establish equivalence. Lineage-specific effects and model-selection-only contrasts are saved in the paired tables.','',
        '## Learned model-family weights','','| Assay | Family | Mean | SD across splits | Zero-weight fits /20 |','|---|---|---:|---:|---:|']
    for _,d in weight_summary.iterrows():lines.append(f'| {d.assay} | {LABELS[d.family]} | {d["mean"]:.3f} | {d["sd"]:.3f} | {d.zero_fits} |')
    lines += ['','![Weight stability](ensemble_weights.png)','',
        'The equal blend always uses 1/3 per family. Learned weights describe predictive mixing, not gene importance or causal contributions. Correlated base predictors can admit unstable or nearly interchangeable weights. Splits overlap, so the weight SD is descriptive rather than an independent-replication interval.','',
        'Per-repeat prediction/error correlations are in `base_error_correlation.tsv`; weights and inner objectives are in `ensemble_weights.tsv`.','',
        '## Base-model error correlation','',
        'The table averages held-out residual Pearson correlations across the 20 repeats. Their high values are consistent with limited error diversification. These are post-evaluation diagnostics and did not train the weights.','',
        '| Assay | Pair | Mean prediction correlation | Mean error correlation |','|---|---|---:|---:|']
    for (assay,first,second),d in correlations.groupby(['assay','first_family','second_family']):
        lines.append(f'| {assay} | {LABELS[first]} / {LABELS[second]} | {d.prediction_correlation.mean():.4f} | {d.error_correlation.mean():.4f} |')
    lines += ['',
        '## Validation and interpretation','',
        '- The same saved five inner folds are used in each of the 20 outer splits. Weight fitting uses only inner-validation rows at the retained base settings; it never receives outer-test labels. The protected inner-training singleton in CTRP has no inner-validation prediction and is omitted from weight fitting, while remaining in the saved base training models.',
        '- Base hyperparameters were selected using these inner labels, which now also train the blend weights. Inner blend losses are optimistic training diagnostics and are not independent ensemble validation. No further cross-fitting of base hyperparameter selection is performed. All accuracy claims use the untouched outer-test response for that split.',
        '- Learned weights minimize equal-mean five-fold RMSE. All vertices, all simplex edges, uniform weights and two SLSQP starts are considered; feasibility/global convex KKT checks must pass. The optimizer can fall back to one family.',
        '- Registry/source hashes, ordered core64 inputs, every original inner RMSE and retained setting, all 120 outer base-model predictions, ensemble arithmetic, convex-hull bounds and split/group exclusions are audited. No base model is retrained.',
        '- Whole-lineage paired bootstrap preserves all models and overlapping repeats together. Intervals are conditional on these models, omit retraining uncertainty and are not multiplicity-adjusted. Twenty holdouts are not independent biological replications.',
        '- These repeatedly inspected cohorts support exploratory internal comparison, not fresh confirmatory accuracy. The finite panel/learner choices, original endpoint provenance limitations and within-represented-lineage scope remain. No patient or unseen-lineage prediction is validated.',
        '- Training is gene-only and unweighted. Culture/lineage prediction covariates are excluded, and the retained model registry remains unchanged.','',
        'Methods references: [scikit-learn stacking documentation](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html) and [SciPy SLSQP documentation](https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html).','']
    (c.OUT/'REVIEW.md').write_text('\n'.join(lines))


def seal():
    c.verify();assert json.load(open(c.OUT/'validation.json'))['passed'];assert json.load(open(c.OUT/'visual_qa.json'))['passed']
    for name,h in json.load(open(c.OUT/'analysis_source_manifest.json'))['scripts'].items():assert c.sha(c.OUT/name)==h
    c.dump(c.OUT/'progress.json',dict(phase='complete',validation_passed=True))
    files=[dict(path=str(p.relative_to(c.OUT)),sha256=c.sha(p),bytes=p.stat().st_size) for p in sorted(c.OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts and p.suffix!='.log']
    c.dump(c.OUT/'artifact_manifest.json',dict(files=files));print('SEALED',len(files),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--seal',action='store_true');args=parser.parse_args()
    if args.seal:seal();return
    c.initialize();c.verify();protocol=json.load(open(c.OUT/'protocol.json'))
    assert json.load(open(c.OUT/'completion.json'))['complete']
    c.dump(c.OUT/'analysis_source_manifest.json',dict(scripts={name:c.sha(c.OUT/name) for name in ['review_comparison.py','analysis_helpers.py','evaluate.py','metric_helpers.py']}))
    results=[];weights=[];parameters=[];correlations=[];valid=[]
    for assay in c.ASSAYS:
        preds,ww,pp,cc,vv=audit(assay);results.append(evaluate(assay,preds,protocol));weights+=ww;parameters+=pp;correlations+=cc;valid.append(vv)
        print('ANALYSIS_COMPLETE',assay,flush=True)
    names=['summary.tsv','paired_comparisons.tsv','lineage_paired_effects.tsv','lineage_win_counts.tsv','per_repeat_pooled_metrics.tsv','per_lineage_metrics.tsv','per_repeat_macro_metrics.tsv']
    frames=[table(name,pd.concat([d[j] for d in results],ignore_index=True)) for j,name in enumerate(names)]
    weights=table('ensemble_weights.tsv',weights);table('base_selected_parameters.tsv',parameters);correlations=table('base_error_correlation.tsv',correlations)
    stats=[]
    for assay in c.ASSAYS:
        d=weights[weights.assay==assay]
        for family in c.FAMILIES:
            x=d[family];stats.append(dict(assay=assay,family=family,mean=x.mean(),sd=x.std(ddof=1),median=x.median(),minimum=x.min(),maximum=x.max(),zero_fits=int((x<=1e-8).sum())))
    stats=table('weight_summary.tsv',stats)
    c.dump(c.OUT/'validation.json',dict(passed=True,assays=valid,preflight=json.load(open(c.OUT/'preflight_validation.json')),registry_unchanged=True,outer_test_labels_excluded_from_weights=True,base_models_retrained=0))
    render(frames[0],frames[1],frames[3],weights,stats,correlations);c.verify()
    c.dump(c.OUT/'final_summary.json',dict(validation_passed=True,summary=frames[0].to_dict('records'),weights=stats.to_dict('records')))
    c.dump(c.OUT/'progress.json',dict(phase='analysis_complete_pending_visual_qa',validation_passed=True));print('ALL_ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':main()
