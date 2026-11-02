"""Render results and scientific figures for the 64-gene NN weight comparison."""
from pathlib import Path
import numpy as np
import pandas as pd

OUT=Path(__file__).resolve().parent
ARMS=['unweighted','inverse_sqrt','capped_inverse_5','inverse_frequency']
LABELS={'unweighted':'Unweighted','inverse_sqrt':'Inverse square root','capped_inverse_5':'Capped inverse, max 5','inverse_frequency':'Full inverse frequency (previous)'}

def render(summary,paired,effects,wins,counts,validation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter
    colors=['#333c48','#138c9c','#7e56a3','#bd7f41']
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    names=['Unweighted','Inverse\nsquare root','Capped\ninverse (5)','Full inverse\n(previous)']
    for i,scope in enumerate(['pooled_745','macro_19']):
        v=summary[summary.scope==scope].set_index('model').loc[ARMS]
        for j,metric in enumerate(['spearman','rmse']):
            ax=axes[i,j];ax.errorbar(range(4),v[metric+'_mean'],yerr=v[metric+'_sd'],color='#adb2b9',fmt='none',capsize=4)
            ax.scatter(range(4),v[metric+'_mean'],color=colors,s=65,zorder=3)
            ax.axhline(v.loc['unweighted',metric+'_mean'],color=colors[0],linestyle='--',alpha=.4)
            ax.set_xticks(range(4),names,fontsize=9);ax.set_ylabel('Spearman ↑' if metric=='spearman' else 'RMSE ↓')
            ax.set_title('Pooled 745 cell lines' if i==0 else 'Equal mean over 19 lineages (n≥10)');ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
            if metric=='rmse':ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.4f}'))
    fig.suptitle('64-gene neural network: moderate lineage weighting\nMeans ± seed SD across 20 matched leave-lineage-out repeats',fontsize=13)
    fig.savefig(OUT/'weighting_comparison.png',dpi=170);fig.savefig(OUT/'weighting_comparison.svg');plt.close(fig)
    e=effects[(effects.reference_model=='unweighted')&(effects.n>=10)];lineages=e.drop_duplicates('lineage').sort_values('n',ascending=False).lineage.tolist()
    fig,axes=plt.subplots(1,2,figsize=(8.8,8),constrained_layout=True,sharey=True)
    for ax,metric in zip(axes,['spearman','rmse']):
        a=np.column_stack([e[e.first_model==arm].set_index('lineage').loc[lineages,metric+'_improvement'] for arm in ['inverse_sqrt','capped_inverse_5']]);limit=float(np.max(np.abs(a)))
        im=ax.imshow(a,aspect='auto',cmap='RdBu',vmin=-limit,vmax=limit);ax.set_xticks([0,1],['Inverse\nsquare root','Capped\ninverse (5)']);ax.set_title('Spearman increase' if metric=='spearman' else 'RMSE reduction');fig.colorbar(im,ax=ax,shrink=.8,pad=.03)
    axes[0].set_yticks(range(19),lineages,fontsize=9)
    fig.suptitle('Changes relative to unweighted training\nMean paired effect over 20 seeds; blue favors weighting',fontsize=13)
    fig.savefig(OUT/'lineage_effects.png',dpi=170);fig.savefig(OUT/'lineage_effects.svg');plt.close(fig)
    write_report(summary,paired,effects,wins,counts,validation)

def write_report(summary,paired,effects,wins,counts,validation):
    pooled=summary[summary.scope=='pooled_745'].set_index('model');macro=summary[summary.scope=='macro_19'].set_index('model')
    higher_pool=[v for v in ['inverse_sqrt','capped_inverse_5'] if pooled.loc[v,'rmse_mean']>pooled.loc['unweighted','rmse_mean']]
    higher_macro=[v for v in ['inverse_sqrt','capped_inverse_5'] if macro.loc[v,'rmse_mean']>macro.loc['unweighted','rmse_mean']]
    if len(higher_pool)==len(higher_macro)==2:
        verdict='Neither moderate weighting scheme improved mean pooled or equal-lineage RMSE over the unweighted baseline.'
    else:
        verdict='The weighting effects differ by scheme and scoring scope; paired estimates below quantify the changes.'
    best_pool=pooled.rmse_mean.idxmin();best_macro=macro.rmse_mean.idxmin()
    rows=['# Moderate lineage weighting for the 64-gene neural network','',f'**{verdict}** '
        f'Lowest mean pooled RMSE: {LABELS[best_pool]} ({pooled.loc[best_pool,"rmse_mean"]:.6f}). '
        f'Lowest mean equal-lineage RMSE: {LABELS[best_macro]} ({macro.loc[best_macro,"rmse_mean"]:.6f}).','',
        'Completed 960 new nested outer fits: two weighting schemes × 20 saved seeds × 24 held-out lineages. '
        'The comparison uses the same 745 cell lines, 734 identity groups, 64 genes, labels, inner/outer splits and three neural initializations per repeat as before. '
        'Verified unweighted predictions are the primary baseline; the previous full inverse-frequency predictions are included as context.','',
        '## Scores','',
        'Values are means of per-repeat scores, not scores after averaging predictions across repeats. Pooled scores include all 745 lines. '
        'Macro scores equally average the same 19 lineages with n≥10.','',
        '| Training weights | Pooled Spearman ↑ | Pooled RMSE ↓ | Pooled R² ↑ | Macro Spearman ↑ | Macro RMSE ↓ |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        a=pooled.loc[arm];b=macro.loc[arm];rows.append(f'| {LABELS[arm]} | {a.spearman_mean:.5f} | {a.rmse_mean:.6f} | {a.r2_mean:.5f} | {b.spearman_mean:.5f} | {b.rmse_mean:.6f} |')
    rows+=['',f'![Weight comparison]({OUT}/weighting_comparison.png)','','## Primary paired effects versus unweighted training','',
        'Positive improvement favors the new scheme: Spearman(new) − Spearman(unweighted), or RMSE(unweighted) − RMSE(new). '
        'Intervals use 5,000 paired whole-lineage bootstrap draws and are conditional on fitted predictions; they do not refit models or adjust for multiplicity. '
        'The same draws pair all arms and all repeats.','',
        '| Scheme | Scope | Metric | Mean improvement | Favorable seeds | Conditional 95% interval |',
        '|---|---|---|---:|---:|---|']
    for v in paired.itertuples():
        if v.reference_model!='unweighted' or v.metric=='r2':continue
        rows.append(f'| {LABELS[v.first_model]} | {v.scope} | {v.metric} | {v.improvement:+.6f} | {v.favorable_repeats}/20 | [{v.conditional_ci_low:+.6f}, {v.conditional_ci_high:+.6f}] |')
    rows+=['','## Consistency across held-out lineages','',
        '| Scheme | Lineages with higher Spearman | Lineages with lower RMSE |',
        '|---|---:|---:|']
    for v in wins.itertuples():
        if v.reference_model=='unweighted':rows.append(f'| {LABELS[v.first_model]} | {v.spearman_favorable_lineages}/19 | {v.rmse_favorable_lineages}/19 |')
    rows+=['','These counts compare mean scores across seeds within each lineage. They are descriptive signs, not independently established lineage-specific effects.',
        '',f'![Lineage effects]({OUT}/lineage_effects.png)','','## Training weights and controls','',
        '- **Inverse square root:** weights are proportional to 1/√n_lineage and normalized to mean one within each current fitting subset. Total lineage weight is proportional to √n_lineage.',
        '- **Capped inverse:** final weights are min(c/n_lineage, 5), with c solved jointly with clipping so the weights average one. Thus the final normalized weight never exceeds five; uncapped lineages retain inverse-frequency ratios. The cap was fixed before new test predictions were inspected, with no cap search.',
        '- All counts and normalizations use only current fitting rows, independently for each inner fit and outer refit. Every represented training lineage participates, including very small lineages.',
        '- X/y StandardScaler fitting remains unweighted and training-only. Mean-one sample weights preserve the original L2 scaling.',
        '- The original four width/penalty candidates are retained: 16/32 tanh units and alpha 10/100, L-BFGS, max_iter 2000, max_fun 40000, tolerance 0.001, and the same three initialization seeds per repeat. Member predictions are averaged before scoring.',
        '- Selection remains the lowest unweighted equal-mean five-fold inner RMSE, with the original candidate order breaking ties. The held-out lineage and related identity groups are excluded from fitting/tuning.',
        '- Weight schemes and cap are fixed experimental arms; no arm is selected separately using outer-test labels.','',
        '| Scheme | Outer per-cell weight range | Outer weight effective sample size range |',
        '|---|---:|---:|']
    audit=pd.read_csv(OUT/'training_subset_weight_audit.tsv',sep='\t');outer=audit[audit.stage=='outer']
    for arm in ARMS:
        a=outer[outer.scheme==arm];rows.append(f'| {LABELS[arm]} | {a.weight_min.min():.4f}–{a.weight_max.max():.4f} | {a.effective_sample_size.min():.1f}–{a.effective_sample_size.max():.1f} |')
    rows+=['','Weight effective sample size is (sum w)²/sum(w²), a concentration diagnostic rather than an independent biological sample count. '
        'Full inverse-frequency weights give equal lineage totals; the moderate schemes deliberately relax this.','',
        '## Validation','',
        f'- All {validation["new_outer_models"]:,} new models reconstructed saved held-out predictions; maximum error {validation["prediction_max_error"]:.3g}.',
        f'- All {validation["reconstructed_candidate_fold_scores"]:,} inner candidate-fold RMSE values and {validation["training_weight_vector_checks"]:,} actual fitting-subset weight vectors were reconstructed and checked.',
        f'- All {validation["direct_weighted_objectives_checked"]:,} selected network members had their weighted training objective recomputed independently from predictions and coefficients; maximum discrepancy {validation["weighted_objective_max_error"]:.3g}.',
        '- Preflight checked independent square-root and capped-weight formulas, permutation/label invariance, balanced-lineage unit weights, finite-difference gradients, and direct objective calculations. Current fitting code exactly replayed original unweighted and full inverse-frequency predictions for Lung/Thyroid in seed 1.',
        '- Selected new settings were replayed for both schemes, Lung/Thyroid, seeds 1/20, and their five corresponding inner fits. These replay the same verified weighted adapter; finite-difference and objective checks provide independent mathematical validation.',
        f'- Convergence warnings: {validation["convergence_warning_count"]}; recorded warnings: {validation["warning_count"]:,}. Detailed categories are saved in validation.json and per-fit metadata.',
        '- Exact shared split arrays, identity exclusions, seed triplets, input/source hashes, prior baseline scores, scaler values and bootstrap statistics versus explicitly expanded SciPy calculations were verified.','',
        '## Interpretation limits','',
        'This is an exploratory follow-up motivated by earlier results on the same cohort. Twenty seeds perturb tuning and initialization while reusing the biological samples; they are not independent biological validations. '
        'Intervals describe variation across observed held-out lineages conditional on the fitted models. They do not establish general performance across all unseen cancer types. '
        'Inner folds mix training lineages, as in the existing protocol; outer tests alone assess transfer to unseen lineages. '
        'Macro-19 scoring excludes the five smallest cohort lineages, whereas every represented lineage contributes during training. '
        'The upstream harmonized erastin AUC provenance limitation remains unresolved, and no independent response cohort was evaluated.','',
        '## Saved artifacts','',
        '- summary.tsv, per_repeat_metrics.tsv, per_repeat_macro_metrics.tsv and per_lineage_metrics.tsv: all four arms.',
        '- paired_comparisons.tsv and paired_seed_differences.tsv: primary comparisons and secondary comparisons against full inverse weighting and between the two new schemes.',
        '- lineage_paired_effects.tsv and lineage_win_counts.tsv: held-out lineage consistency.',
        '- training_subset_weight_audit.tsv and training_lineage_weights.tsv: exact weights and concentration diagnostics.',
        '- selected_parameters.tsv, tuning_scores.tsv, model_training_objective_checks.tsv, validation.json, preflight_validation.json and protocol.json: settings and audits.',
        '- predictions.npz, bootstrap_draws.npz, inputs.npz, shared_splits.npz and folds/: matched inputs, all predictions and all 960 newly fitted models.',
        '- Source scripts, pinned environment versions, source manifests and checksummed artifact manifest.','']
    (OUT/'REVIEW.md').write_text('\n'.join(rows))
