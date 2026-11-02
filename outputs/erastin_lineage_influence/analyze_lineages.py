#!/usr/bin/env python3
"""Audit lineage weights and decompose evaluation influence of all nine arms."""
import hashlib
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;WS=OUT.parent.parent;SOURCE=WS/'outputs/erastin_panel_comparison'
FAMILIES=['ridge','neural_network','xgboost'];SIZES=[40,64,80]
ARMS=[f+'__'+str(p) for f in FAMILIES for p in SIZES]
TITLES={'ridge':'Ridge','neural_network':'Neural network','xgboost':'XGBoost'}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,(float,np.floating)):return float(v) if np.isfinite(v) else None
    if isinstance(v,np.bool_):return bool(v)
    return v
def dump(path,obj):path.write_text(json.dumps(clean(obj),indent=2,allow_nan=False)+'\n')
def table(filename,rows):
    f=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    f.to_csv(OUT/filename,sep='\t',index=False,float_format='%.12g');return f
def score(y,p):
    return dict(rmse=float(np.sqrt(np.mean((y-p)**2))),spearman=float(spearmanr(y,p).statistic) if np.ptp(y)>0 and np.ptp(p)>0 else np.nan)
def label(arm):
    f,p=arm.rsplit('__',1);return TITLES[f]+' '+p

def main():
    known={r['path']:r['sha256'] for r in json.loads((SOURCE/'artifact_manifest.json').read_text())['files']}
    sources=[]
    for filename in ['predictions.npz','shared_splits.npz','per_repeat_metrics.tsv','per_lineage_metrics.tsv','summary.tsv','run_panels.py']:
        assert sha(SOURCE/filename)==known[filename]
        sources.append(dict(path=str((SOURCE/filename).relative_to(WS)),sha256=known[filename]))
    d=np.load(SOURCE/'predictions.npz');s=np.load(SOURCE/'shared_splits.npz')
    for key in ['sample_ids','group_ids','lineage']:np.testing.assert_array_equal(d[key],s[key])
    np.testing.assert_array_equal(d['inner_split_seeds'],s['inner_split_seeds'])
    y=d['y'];lineage=d['lineage'];labels=s['holdout_labels'];N=len(y)
    sizes=np.array([np.sum(lineage==l) for l in labels]);eligible=sizes>=10
    assert N==745 and len(labels)==24 and eligible.sum()==19 and sizes[eligible].sum()==721
    protocol=dict(status='fixed_before_lineage_decomposition',script_sha256=sha(Path(__file__)),models=ARMS,repeats=20,
        samples=N,lineages=24,interpretation='Lineage weighting and influence on SAVED MODEL EVALUATION; no models retrained.',
        learned_lineage_coefficients='None: models use expression only. Lineage labels define split membership.',
        training='Each current training cell has equal explicit loss weight; no inverse-lineage balancing. XGBoost uses uniform row sampling at 0.8.',
        pooled_weight='n_lineage/745 for MSE. Pooled RMSE is sqrt(sum(weight*lineage MSE)), not a weighted average of lineage RMSE.',
        macro_weight='1/19 for each lineage with n>=10, zero for the other five; mean of lineage RMSE and Spearman.',
        prior_gene_importance_weight='n_lineage/721 among the same 19 larger lineages.',
        error_share='Mean across repeats of SSE_lineage/SSE_all; model-dependent difficulty/size contribution, not learned feature importance.',
        spearman_decomposition='Exact global-rank covariance split into within-lineage and between-lineage terms; each normalized by global rank SD product.',
        omission='Recompute pooled scores omitting one lineage from EVALUATION only, without retraining. Does not measure training-data influence.',
        skill_baseline='Training-only mean response separately for each outer holdout. Signed lineage MSE gains sum to the overall gain versus this baseline.',
        panel_gain='Signed lineage contributions (n/745)*(MSE_smaller-MSE_larger); macro Spearman contributions are local rank-correlation differences divided by 19.',
        uncertainty='Descriptive averages over the same 20 repeats. No new p-values or independent biological replications.')
    dump(OUT/'protocol.json',protocol);dump(OUT/'source_manifest.json',dict(files=sources))
    weights=table('lineage_weights.tsv',[dict(lineage=str(l),n=int(n),pooled_mse_weight=n/N,
        macro_metric_weight=1/19 if n>=10 else 0,prior_gene_importance_weight=n/721 if n>=10 else 0,
        eligible_macro=bool(n>=10)) for l,n in zip(labels,sizes)]).sort_values('n',ascending=False)
    table('lineage_weights.tsv',weights)
    baseline=np.empty((20,N));training=[];training_mix=np.empty((24,24))
    for l,heldout in enumerate(labels):
        for r in range(20):
            tr=np.flatnonzero(s['inner_fold_by_holdout'][r,l]>=0);te=np.flatnonzero(lineage==heldout)
            assert set(d['group_ids'][tr]).isdisjoint(d['group_ids'][te]) and heldout not in set(lineage[tr])
            if r:np.testing.assert_array_equal(tr,np.flatnonzero(s['inner_fold_by_holdout'][0,l]>=0))
            baseline[r,te]=y[tr].mean()
        for k,training_lineage in enumerate(labels):
            count=int(np.sum(lineage[tr]==training_lineage));training_mix[l,k]=count/len(tr)
            training.append(dict(held_out_lineage=str(heldout),training_lineage=str(training_lineage),train_lineage_n=count,
                                 total_train_n=len(tr),training_loss_share=count/len(tr),applies_to='all nine model-panel arms and all repeats'))
    assert np.max(np.abs(training_mix.sum(axis=1)-1))<1e-14 and np.max(np.diag(training_mix))==0
    table('training_lineage_weights.tsv',training)
    saved_scores=pd.read_csv(SOURCE/'per_repeat_metrics.tsv',sep='\t')
    saved_lineage=pd.read_csv(SOURCE/'per_lineage_metrics.tsv',sep='\t')
    ry=rankdata(y);ry-=ry.mean()
    rows=[];reconstructed=[];max_metric=max_sse=max_cov=0.
    error_arrays={};rho_arrays={};lineage_mse={};lineage_rho={};omissions={}
    for arm in ARMS:
        pred=d[arm];assert pred.shape==(20,N) and np.isfinite(pred).all()
        errors=(pred-y[None,:])**2;total_sse=errors.sum(axis=1);base_error=(baseline-y[None,:])**2
        mse=np.empty((20,24));rhos=np.empty_like(mse);er=np.empty_like(mse);rc=np.empty_like(mse)
        for r,p in enumerate(pred):
            full=score(y,p);expected=saved_scores[(saved_scores.model==arm)&(saved_scores.repeat==r+1)].iloc[0]
            max_metric=max(max_metric,abs(full['rmse']-expected.rmse),abs(full['spearman']-expected.spearman))
            rp=rankdata(p);rp-=rp.mean();den=np.sqrt(np.sum(ry**2)*np.sum(rp**2))
            for l,heldout in enumerate(labels):
                ix=lineage==heldout;local=score(y[ix],p[ix]);without=score(y[~ix],p[~ix])
                sse=float(errors[r,ix].sum());mse[r,l]=local['rmse']**2;rhos[r,l]=local['spearman'];er[r,l]=sse/total_sse[r]
                between=float(ix.sum()*ry[ix].mean()*rp[ix].mean()/den)
                within=float(np.sum((ry[ix]-ry[ix].mean())*(rp[ix]-rp[ix].mean()))/den)
                rc[r,l]=within+between
                rows.append(dict(model=arm,repeat=r+1,lineage=str(heldout),n=int(ix.sum()),
                    pooled_mse_weight=ix.sum()/N,macro_metric_weight=1/19 if ix.sum()>=10 else 0,
                    mse=local['rmse']**2,rmse=local['rmse'],spearman=local['spearman'],
                    pooled_mse_contribution=sse/N,total_squared_error_share=sse/total_sse[r],
                    pooled_spearman_contribution=within+between,within_global_rank_contribution=within,between_global_rank_contribution=between,
                    rmse_without_lineage=without['rmse'],rmse_change_when_omitted=without['rmse']-full['rmse'],
                    spearman_without_lineage=without['spearman'],spearman_change_when_omitted=without['spearman']-full['spearman'],
                    weighted_mse_gain_vs_training_mean=float((base_error[r,ix]-errors[r,ix]).sum()/N)))
            max_sse=max(max_sse,abs(np.sum(mse[r]*sizes/N)-np.mean(errors[r])),abs(er[r].sum()-1))
            max_cov=max(max_cov,abs(rc[r].sum()-full['spearman']))
            reconstructed.append(dict(model=arm,repeat=r+1,pooled_rmse_from_lineages=np.sqrt(np.sum(mse[r]*sizes/N)),
                pooled_spearman_from_rank_contributions=rc[r].sum(),macro_rmse=np.mean(np.sqrt(mse[r,eligible])),macro_spearman=np.mean(rhos[r,eligible])))
        error_arrays[arm]=er;rho_arrays[arm]=rc;lineage_mse[arm]=mse;lineage_rho[arm]=rhos
    assert max(max_metric,max_sse,max_cov)<1e-11
    detail=table('per_repeat_lineage_contributions.tsv',rows)
    keys=['model','lineage','n','pooled_mse_weight','macro_metric_weight']
    numerical=[c for c in detail.columns if c not in keys+['repeat']]
    mean=detail.groupby(keys,dropna=False)[numerical].mean().reset_index()
    table('per_model_lineage_contributions.tsv',mean)
    replay=table('metric_reconstruction.tsv',reconstructed)
    # Verify the existing per-lineage metrics independently from predictions.
    joined=detail.merge(saved_lineage[['repeat','model','lineage','rmse','spearman']],on=['repeat','model','lineage'],suffixes=('_new','_saved'),validate='one_to_one')
    np.testing.assert_allclose(joined.rmse_new,joined.rmse_saved,rtol=0,atol=1e-11)
    np.testing.assert_allclose(joined.spearman_new,joined.spearman_saved,rtol=0,atol=1e-11,equal_nan=True)
    expansion=[];expansion_summaries=[]
    for f in FAMILIES:
        for large,small in [(64,40),(80,40),(80,64)]:
            a=f+'__'+str(large);b=f+'__'+str(small)
            gains=(lineage_mse[b]-lineage_mse[a])*sizes[None,:]/N
            rank_gain=(lineage_rho[a]-lineage_rho[b])/19
            means=gains.mean(axis=0);gross_positive=np.maximum(means,0).sum();gross_negative=-np.minimum(means,0).sum()
            for l,heldout in enumerate(labels):
                expansion.append(dict(first_model=a,reference_model=b,lineage=str(heldout),n=int(sizes[l]),
                    weighted_mse_reduction=float(means[l]),gross_positive_gain_share=float(max(means[l],0)/gross_positive) if gross_positive else 0,
                    gross_deterioration_share=float(max(-means[l],0)/gross_negative) if gross_negative else 0,
                    favorable_mse_repeats=int(np.sum(gains[:,l]>0)),
                    macro_spearman_change_contribution=float(np.mean(rank_gain[:,l])) if eligible[l] else 0))
            total=float(means.sum());direct=np.mean((d[b]-y[None,:])**2)-np.mean((d[a]-y[None,:])**2)
            assert abs(total-direct)<1e-13
            macro_change=float(np.mean((lineage_rho[a]-lineage_rho[b])[:,eligible]))
            assert abs(np.mean(rank_gain[:,eligible],axis=0).sum()-macro_change)<1e-13
            expansion_summaries.append(dict(first_model=a,reference_model=b,total_mse_reduction=total,
                macro_spearman_change=macro_change,gross_positive_mse_reduction=gross_positive,gross_mse_deterioration=gross_negative))
    expansion=table('panel_gain_lineage_contributions.tsv',expansion);table('panel_gain_summary.tsv',expansion_summaries)
    omission_summary=[]
    for arm in ARMS:
        v=detail[detail.model==arm]
        for heldout in labels:
            z=v[v.lineage==heldout]
            omission_summary.append(dict(model=arm,lineage=str(heldout),n=int(z.n.iloc[0]),
                rmse_change_when_omitted=z.rmse_change_when_omitted.mean(),spearman_change_when_omitted=z.spearman_change_when_omitted.mean(),
                mean_rmse_without=z.rmse_without_lineage.mean(),mean_spearman_without=z.spearman_without_lineage.mean()))
    table('evaluation_omission_sensitivity.tsv',omission_summary)
    summary=[]
    for arm in ARMS:
        v=mean[mean.model==arm];ss=detail[detail.model==arm]
        top_error=v.nlargest(3,'total_squared_error_share');top_skill=v.nlargest(3,'weighted_mse_gain_vs_training_mean')
        summary.append(dict(model=arm,top3_error_lineages=top_error.lineage.tolist(),top3_error_share=float(top_error.total_squared_error_share.sum()),
            top3_training_mean_skill_lineages=top_skill.lineage.tolist(),
            pooled_spearman=float(v.pooled_spearman_contribution.sum()),
            within_global_rank_component=float(v.within_global_rank_contribution.sum()),
            between_global_rank_component=float(v.between_global_rank_contribution.sum()),
            mean_rmse=float(saved_scores[saved_scores.model==arm].rmse.mean()),
            median_within_lineage_spearman=float(v[v.n>=10].spearman.median())))
    summary=table('model_lineage_summary.tsv',summary)
    check=dict(passed=True,models=9,repeats=20,lineages=24,no_refitting=True,explicit_weights_identical_across_models=True,
        held_out_lineage_training_weight_zero=True,training_mix_rows_sum_to_one=True,
        exact_mse_and_error_share_decomposition_max_error=max_sse,
        exact_spearman_global_rank_covariance_decomposition_max_error=max_cov,
        saved_metric_reconstruction_max_error=max_metric,per_lineage_metrics_reproduced=True,
        note='Rank components are an algebraic decomposition of global ranks, not within-lineage Spearman or causal attribution.')
    dump(OUT/'validation.json',check)
    render(weights,mean,expansion,summary,check)
    for source in sources:assert sha(WS/source['path'])==source['sha256']
    assert sha(Path(__file__))==protocol['script_sha256']
    dump(OUT/'progress.json',dict(phase='complete',validation_passed=True,model_arms=9))
    manifest=dict(files=[dict(path=p.name,bytes=p.stat().st_size,sha256=sha(p)) for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='artifact_manifest.json'])
    dump(OUT/'artifact_manifest.json',manifest);archive=OUT.parent/'erastin_lineage_influence_results.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for r in manifest['files']:z.write(OUT/r['path'],'erastin_lineage_influence/'+r['path'])
        z.write(OUT/'artifact_manifest.json','erastin_lineage_influence/artifact_manifest.json')
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps(dict(validation=check,summary=clean(summary.to_dict('records')),archive_bytes=archive.stat().st_size),indent=2))

def render(weights,mean,expansion,summary,validation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    labels=weights.lineage.tolist();short=[label(a) for a in ARMS]
    fig,axes=plt.subplots(1,2,figsize=(16,10),constrained_layout=True,sharey=True)
    error=mean.pivot(index='lineage',columns='model',values='total_squared_error_share').loc[labels,ARMS].to_numpy()*100
    rank=mean.pivot(index='lineage',columns='model',values='pooled_spearman_contribution').loc[labels,ARMS].to_numpy()
    im=axes[0].imshow(error,aspect='auto',cmap='YlOrRd');fig.colorbar(im,ax=axes[0],shrink=.65,label='Share of total squared error (%)')
    bound=float(np.max(np.abs(rank)));im=axes[1].imshow(rank,aspect='auto',cmap='RdBu',norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound))
    fig.colorbar(im,ax=axes[1],shrink=.65,label='Signed contribution to pooled Spearman')
    for ax in axes:ax.set_xticks(range(9),short,rotation=60,ha='right',fontsize=9)
    axes[0].set_yticks(range(24),[f'{l} (n={int(weights.set_index("lineage").loc[l,"n"])})' for l in labels])
    axes[0].set_title('Where prediction error accumulates');axes[1].set_title('Which lineages drive the pooled ranking score')
    fig.suptitle('Lineage contributions to the nine saved model evaluations\nMean over 20 matched repeats; these are evaluation contributions, not learned coefficients',fontsize=13)
    fig.savefig(OUT/'lineage_contributions.png',dpi=170);fig.savefig(OUT/'lineage_contributions.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12,9),constrained_layout=True,sharey=True)
    yy=np.arange(24)
    for off,col,color,title in [(-.2,'pooled_mse_weight','#137d8b','Pooled MSE'),(.2,'macro_metric_weight','#8853a3','Equal-lineage macro')]:
        axes[0].barh(yy+off,weights[col]*100,height=.36,color=color,label=title)
    axes[0].set_yticks(yy,labels);axes[0].invert_yaxis();axes[0].set_xlabel('Explicit evaluation weight (%)');axes[0].legend(fontsize=9)
    nn=expansion[(expansion.first_model=='neural_network__64')&(expansion.reference_model=='neural_network__40')].set_index('lineage').loc[labels]
    values=nn.weighted_mse_reduction*1000
    axes[1].barh(yy,values,color=np.where(values>=0,'#137d8b','#b65e40'));axes[1].axvline(0,color='#444',lw=1)
    axes[1].set_xlabel('Contribution to pooled MSE reduction × 1,000');axes[1].set_title('Neural network: 64 versus 40 genes\nPositive favors 64 genes')
    for ax in axes:ax.grid(axis='x',alpha=.18);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Sample weighting and the lineages driving panel-expansion gains',fontsize=13)
    fig.savefig(OUT/'weights_and_nn64_gains.png',dpi=170);fig.savefig(OUT/'weights_and_nn64_gains.svg');plt.close(fig)
    rows=['# Lineage weights and evaluation influence','',
        '**There are no learned lineage coefficients in these models.** All nine current arms use expression features only. '
        'Lineage labels define the outer holdouts and identify groups for evaluation. A separate fitted model predicts each held-out lineage; '
        'that lineage has zero training weight in its own outer model.','',
        'Within a training subset, cells have equal explicit loss weight. There is no inverse-lineage balancing. XGBoost samples rows uniformly at rate 0.8. '
        'These are sampling/loss weights, not estimates of each training lineage’s actual influence on fitted parameters. '
        'Inner tuning minimizes the equal-weight mean of five fold RMSE values, not an equal-lineage objective.','',
        '## Explicit evaluation weights','',
        '| Lineage | Cells | Pooled MSE weight | Macro score weight | Prior gene-importance weight |',
        '|---|---:|---:|---:|---:|']
    for v in weights.itertuples():rows.append(f'| {v.lineage} | {v.n} | {100*v.pooled_mse_weight:.2f}% | {100*v.macro_metric_weight:.2f}% | {100*v.prior_gene_importance_weight:.2f}% |')
    rows+=['','These weights are identical for ridge, neural network and XGBoost at all three panel sizes. '
        'Pooled MSE weights are n/745. Pooled RMSE is the square root of the weighted sum of lineage MSE, not the weighted average of lineage RMSE. '
        'Macro RMSE and macro Spearman give 1/19 to each lineage with n≥10; the other five receive zero macro weight. '
        'The previous gene-importance primary summary used n/721 over those 19 larger lineages.','',
        f'![Weights and gain contributions]({OUT}/weights_and_nn64_gains.png)','','## Model-dependent contributions','',
        '| Model | Three largest squared-error contributors | Combined error share | Within-global-rank component | Between-lineage-rank component | Pooled Spearman |',
        '|---|---|---:|---:|---:|---:|']
    for v in summary.itertuples():rows.append(f'| {label(v.model)} | {", ".join(v.top3_error_lineages)} | {100*v.top3_error_share:.1f}% | {v.within_global_rank_component:.4f} | {v.between_global_rank_component:.4f} | {v.pooled_spearman:.4f} |')
    rows+=['','Error share is SSE from a lineage divided by total SSE, averaged over repeats. It combines lineage size and prediction difficulty. '
        'A large error share does not mean that lineage is a highly weighted input feature.','',
        'The Spearman terms exactly decompose covariance of the full-cohort response/prediction ranks into within-lineage and between-lineage parts, normalized by the same global denominator. '
        'They sum to pooled Spearman. The within-global-rank component is not the mean of local within-lineage Spearman scores; local ranks use a different scale. '
        'Components are descriptive associations, not causal effects or independent predictive abilities.','',
        f'![Model lineage contributions]({OUT}/lineage_contributions.png)','','## What drives the 64-gene neural-network gain','']
    before=summary[summary.model=='neural_network__40'].iloc[0]
    after=summary[summary.model=='neural_network__64'].iloc[0]
    rows+=['The neural network pooled Spearman increase decomposes as follows:', '',
        f'- Between-lineage global-rank component: {before.between_global_rank_component:.6f} → {after.between_global_rank_component:.6f} '
        f'(change {after.between_global_rank_component-before.between_global_rank_component:+.6f}).',
        f'- Within-lineage global-rank component: {before.within_global_rank_component:.6f} → {after.within_global_rank_component:.6f} '
        f'(change {after.within_global_rank_component-before.within_global_rank_component:+.6f}).',
        f'- Their sum changes by {after.pooled_spearman-before.pooled_spearman:+.6f}. The increase comes from the between-lineage term in this algebraic decomposition; '
        'it does not establish a causal role for lineage or a learned categorical lineage feature.', '']
    z=expansion[(expansion.first_model=='neural_network__64')&(expansion.reference_model=='neural_network__40')]
    for title,frame in [('Largest positive contributions',z.nlargest(5,'weighted_mse_reduction')),('Largest negative contributions',z.nsmallest(5,'weighted_mse_reduction'))]:
        rows.extend([title+':',''])
        for v in frame.itertuples():rows.append(f'- {v.lineage}, n={v.n}: weighted MSE reduction {v.weighted_mse_reduction:+.8f}; '
            f'{100*v.gross_positive_gain_share:.1f}% of gross positive gain, {100*v.gross_deterioration_share:.1f}% of gross deterioration.')
        rows.append('')
    worst=z.nsmallest(3,'macro_spearman_change_contribution')
    rows+=['Largest negative contributions to the equal-lineage Spearman change:','']
    for v in worst.itertuples():rows.append(f'- {v.lineage}: {v.macro_spearman_change_contribution:+.5f} of the mean Spearman change.')
    soft=z[z.lineage=='Soft Tissue'].iloc[0]
    total_macro=float(z.macro_spearman_change_contribution.sum())
    rows+=['',f'Soft Tissue has only {soft.n} cells ({100*soft.n/745:.2f}% of pooled MSE weight) but receives 5.26% macro weight. '
        f'Its contribution to the macro-Spearman change is {soft.macro_spearman_change_contribution:+.6f}, compared with '
        f'{total_macro:+.6f} overall; the remaining 18 eligible lineages sum to {total_macro-soft.macro_spearman_change_contribution:+.6f}. '
        'This is a descriptive explanation of the aggregate score tradeoff; the small lineage itself has substantial sampling uncertainty.']
    rows+=['','Signed weighted MSE contributions sum exactly to the total MSE reduction. Gross-positive percentages normalize only positive contributions, so they must not be interpreted as percentages of the net gain. '
        'Expansion effects for every family and both larger panels are provided in the data tables.','',
        '## Omission sensitivity and limits','',
        'The omission table removes one lineage from scoring and recomputes pooled RMSE and Spearman, keeping all trained models and predictions fixed. '
        'It is a score-sensitivity diagnostic. It does not estimate the effect of removing that lineage from training, which would require additional model fitting. '
        'An RMSE change below zero means the omitted lineage was raising pooled prediction error; a Spearman change above zero means its omission improves the pooled ranking score.','',
        'All scores reuse the same 20 seeds and biological cohort. No new model fitting, external validation or causal lineage-importance claim is made. '
        'Small lineages have imprecise descriptive metrics, and a singleton has no defined within-lineage correlation. '
        'The original conditional bootstrap intervals for panel comparisons remain in the parent experiment; this decomposition does not add new hypothesis tests.','',
        f'Validation passed: MSE reconstruction error ≤ {validation["exact_mse_and_error_share_decomposition_max_error"]:.3g}; '
        f'global-rank Spearman decomposition error ≤ {validation["exact_spearman_global_rank_covariance_decomposition_max_error"]:.3g}.','',
        '## Files','',f'- [Weights]({OUT}/lineage_weights.tsv)',f'- [Training weight by held-out lineage]({OUT}/training_lineage_weights.tsv)',
        f'- [All model/lineage contributions]({OUT}/per_model_lineage_contributions.tsv)',
        f'- [Panel-expansion gain contributions]({OUT}/panel_gain_lineage_contributions.tsv)',
        f'- [Evaluation omission sensitivity]({OUT}/evaluation_omission_sensitivity.tsv)',f'- [Validation]({OUT}/validation.json)','']
    (OUT/'REVIEW.md').write_text('\n'.join(rows))

if __name__=='__main__':main()
