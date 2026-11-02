"""Render the new lineage-stratified design and its model comparison."""
import json
from pathlib import Path
import numpy as np,pandas as pd
OUT=Path(__file__).resolve().parent
ARMS=['unweighted','inverse_sqrt','capped_inverse_5','inverse_frequency']
LABELS={'unweighted':'Unweighted','inverse_sqrt':'Inverse square root','capped_inverse_5':'Capped inverse (max 5)','inverse_frequency':'Full inverse frequency'}

def render(summary,paired,effects,validation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter
    split=pd.read_csv(OUT/'split_lineage_counts.tsv',sep='\t');first=split[split.repeat==1].sort_values('cohort_n');y=np.arange(len(first))
    fig,ax=plt.subplots(figsize=(9,8),constrained_layout=True)
    ax.barh(y,first.train_n,color='#367f9c',label='Training (595)');ax.barh(y,first.test_n,left=first.train_n,color='#d89546',label='Testing (149)')
    for i,v in enumerate(first.itertuples()):ax.text(v.cohort_n+1,i,f'{v.train_n} / {v.test_n}',va='center',fontsize=9)
    ax.set_yticks(y,first.lineage,fontsize=9);ax.set_xlim(0,143);ax.set_xlabel('Cell lines per lineage');ax.spines[['top','right']].set_visible(False);ax.legend(loc='lower right')
    ax.set_title('All 23 lineages appear in training and testing in every repeat\nLabels show train / test counts; Cervix excluded',fontsize=12)
    fig.savefig(OUT/'split_balance.png',dpi=170);fig.savefig(OUT/'split_balance.svg');plt.close(fig)
    s=summary.set_index('model').loc[ARMS];colors=['#333c48','#138c9c','#7e56a3','#bd7f41'];names=['Unweighted','Inverse\nsquare root','Capped\ninverse (5)','Full inverse\nfrequency']
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    keys=['pooled_spearman','pooled_rmse','macro23_rmse','macro19_rmse'];titles=['Pooled test Spearman ↑','Pooled test RMSE ↓','Equal-lineage RMSE: all 23 ↓','Equal-lineage RMSE: 19 larger lineages ↓']
    for ax,key,title in zip(axes.flat,keys,titles):
        ax.errorbar(range(4),s[key+'_mean'],yerr=s[key+'_sd'],fmt='none',capsize=3,color='#aeb5bb');ax.scatter(range(4),s[key+'_mean'],color=colors,s=55,zorder=3)
        ax.axhline(s.loc['unweighted',key+'_mean'],color=colors[0],linestyle='--',alpha=.35);ax.set_xticks(range(4),names,fontsize=9);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
        if key!='pooled_spearman':ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.4f}'))
    fig.suptitle('64-gene neural network: lineage-stratified 80/20 testing\nMeans ± split SD over 20 shared splits; all arms newly fitted',fontsize=13)
    fig.savefig(OUT/'model_comparison.png',dpi=170);fig.savefig(OUT/'model_comparison.svg');plt.close(fig)
    write_report(summary,paired,effects,validation,split)

def write_report(summary,paired,effects,validation,split):
    s=summary.set_index('model');best_pool=s.pooled_rmse_mean.idxmin();best_macro=s.macro23_rmse_mean.idxmin();pre=validation['preflight']
    rows=['# Lineage-stratified validation of the 64-gene neural network','',
        '**All 23 retained lineages appear in both training and testing in every one of the 20 repeats.** Cervix (one cell line) was excluded as requested. '
        'Each split has **595 training lines and 149 test lines**, with no related identity group crossing the split. '
        'The cohort contains 744 cell lines and 733 identity groups.','',
        f'Lowest mean pooled RMSE: **{LABELS[best_pool]} ({s.loc[best_pool,"pooled_rmse_mean"]:.6f})**. '
        f'Lowest equal-lineage RMSE across all 23 lineages: **{LABELS[best_macro]} ({s.loc[best_macro,"macro23_rmse_mean"]:.6f})**. '
        'These are descriptive model comparisons on the existing cohort; intervals below quantify conditional variation across observed lineages.','',
        '## Exactly how training and testing are split','',
        'Each lineage contributes approximately 80% of its samples to training and 20% to testing. The nearest feasible rounded sample count is used, with at least one identity group on each side. '
        'Whole identity groups are sampled uniformly among subsets meeting that sample-count target. All related cell-line identities stay together. '
        'The sample counts below are identical across the 20 repeats, while selected identity groups change with the seed. '
        'RNA expression and erastin response values are not used to generate splits.','',
        '| Lineage | Total | Training | Testing |','|---|---:|---:|---:|']
    for v in split[split.repeat==1].sort_values('cohort_n',ascending=False).itertuples():rows.append(f'| {v.lineage} | {v.cohort_n} | {v.train_n} | {v.test_n} |')
    rows+=['',f'![Train/test representation]({OUT}/split_balance.png)','','## Inner tuning and the rare-lineage exception','',
        'The outer 149-line test set is never used for preprocessing, weighting, candidate selection or fitting. '
        'Within the 595-line training set, five group-stratified validation folds choose among the original four neural-network candidates. '
        'All 23 lineages remain represented in every inner training set.','',
        '**Ampulla of Vater has only two cell lines:** one is outer training and one is outer testing in each repeat. '
        'Its single outer-training line is retained in every inner training fit and is never an inner-validation point. '
        'Consequently each inner validation set contains 22 lineages; it is mathematically impossible for that single training line to occur on both disjoint inner sides. '
        'The outer Ampulla test line remains entirely untouched by tuning.','',
        'The protocol retains minimum unweighted mean five-fold RMSE selection, training-only unweighted X/y standardization, '
        '16/32 tanh units × alpha 10/100, L-BFGS and the same three initialization seeds per repeat. '
        'All four weighting arms share exactly the same outer and inner split assignments and initialization seeds. '
        'All 80 outer models were newly fitted, including the unweighted baseline and full inverse-frequency reference.','',
        '## Results','',
        'Scores are computed separately on each repeat’s test set and then averaged. The test sets overlap across repeats; no stacked 20×149 sample count is treated as independent. '
        'Macro-23 RMSE equally averages all 23 lineage-specific test RMSEs. Macro-19 RMSE uses the same 19 lineages with at least 10 total cohort samples as a secondary measure.','',
        '| Training weights | Pooled Spearman ↑ | Pooled RMSE ↓ | Pooled R² ↑ | Macro-23 RMSE ↓ | Macro-19 RMSE ↓ |',
        '|---|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        v=s.loc[arm];rows.append(f'| {LABELS[arm]} | {v.pooled_spearman_mean:.5f} | {v.pooled_rmse_mean:.6f} | {v.pooled_r2_mean:.5f} | {v.macro23_rmse_mean:.6f} | {v.macro19_rmse_mean:.6f} |')
    rows+=['',f'![Model comparison]({OUT}/model_comparison.png)','','## Paired differences versus unweighted training','',
        'Positive improvement favors weighting: higher Spearman/R² or lower RMSE. Conditional intervals use 5,000 whole-lineage bootstrap draws shared across all four models and all 20 split-specific test sets. '
        'Macro-19 uses the same paired resampling restricted to its eligible lineage set. Models are not refitted in the bootstrap and intervals are not adjusted for multiplicity.','',
        '| Weights | Metric | Mean improvement | Favorable repeats | Conditional 95% interval |',
        '|---|---|---:|---:|---|']
    for v in paired.itertuples():
        if v.reference_model!='unweighted' or v.metric in ['pooled_r2','macro19_rmse']:continue
        rows.append(f'| {LABELS[v.first_model]} | {v.metric} | {v.improvement:+.6f} | {v.favorable_repeats}/20 | [{v.conditional_ci_low:+.6f}, {v.conditional_ci_high:+.6f}] |')
    rows+=['','## Consistency across lineages','',
        '| Weights | Lineages with lower mean test RMSE than unweighted |','|---|---:|']
    for arm in ARMS[1:]:
        a=effects[(effects.first_model==arm)&(effects.reference_model=='unweighted')];rows.append(f'| {LABELS[arm]} | {int(np.sum(a.rmse_improvement>0))}/23 |')
    rows+=['','Counts describe point estimates across repeated splits, not independently established lineage effects. '
        'Ampulla, Biliary and Prostate each have one test line per repeat; lineage RMSE is defined, but single-point Spearman is not. '
        'Other small test subsets also yield unstable correlations, so no macro Spearman is used. Per-lineage correlations are saved where mathematically defined.','',
        '## Weights and validation','',
        '- Unweighted assigns weight one. Inverse square root is proportional to 1/√n_lineage. Full inverse frequency gives equal total training weight to each represented lineage. Capped inverse uses min(c/n_lineage, 5), with c solved to keep the final mean one and final maximum five.',
        '- All counts and normalizations use the current inner training or outer training rows only. All represented lineages participate, including small ones. The cap and schemes were fixed before inspecting these test predictions.',
        f'- Validated {validation["outer_models"]} newly fitted models, {validation["candidate_fold_scores_checked"]:,} inner candidate-fold scores, {validation["training_weight_checks"]} training weight vectors and {validation["training_objectives_checked"]} independently reconstructed weighted training objectives.',
        f'- Saved prediction maximum error: {validation["prediction_max_error"]:.3g}; inner RMSE maximum error: {validation["inner_rmse_max_error"]:.3g}; weighted objective maximum error: {validation["training_objective_max_error"]:.3g}.',
        '- Selected settings for all four arms in repeats 1 and 20 replayed both outer predictions and all five inner predictions. The same verified weighted-loss adapter is used; preflight verifies unchanged fitting code, unit-weight agreement and the new split-dependent bootstrap against explicit SciPy expansion.',
        f'- Convergence warnings: {validation["convergence_warning_count"]}; total recorded warnings: {validation["warning_count"]:,}. Per-fit diagnostics and categories are saved.',
        '- Verified input/source hashes, Cervix removal, reproducible response-blind split generation, exact all-lineage representation, zero identity-group overlap, training-only scaling and common seeds across arms.',
        '',f'Across the 20 random holdouts, {pre["test_sample_coverage"]["ever_tested"]}/744 distinct cell lines appeared in at least one test set; '
        f'{pre["test_sample_coverage"]["never_tested"]} were never tested. Repeated 80/20 holdouts do not guarantee every sample receives a test prediction. '
        'Coverage by sample and identity group is saved.','',
        '## What this experiment measures','',
        'This experiment predicts new cell lines from lineages already represented in training. The previous leave-lineage-out experiments measured transfer to absent lineages. '
        'Their predictions are not reused and their scores should not be pooled with this experiment. '
        'Changing the design also changes the training size, cohort and number of test samples per lineage.','',
        'The existing cohort has already informed earlier model and weighting choices, so these are exploratory results rather than untouched external validation. '
        'Repeated splits reuse samples and overlapping training sets; twenty seeds are not twenty independent biological replications. '
        'The lineage bootstrap is conditional on fitted predictions and describes variation across these observed lineages. '
        'Tiny lineage test subsets are particularly uncertain. The upstream harmonized erastin AUC provenance limitation remains unresolved.','',
        '## Files','',
        '- split_lineage_counts.tsv and inner_split_lineage_counts.tsv: representation in every outer/inner split.',
        '- shared_splits.npz, inputs.npz and sample_test_coverage.tsv: fixed memberships, matched inputs and test coverage.',
        '- summary.tsv, paired_comparisons.tsv, paired_seed_differences.tsv and lineage_paired_effects.tsv: all four weighting arms.',
        '- per_repeat_pooled_metrics.tsv, per_repeat_macro_metrics.tsv and per_lineage_metrics.tsv: individual repeat and lineage scores.',
        '- protocol.json, preflight_validation.json, validation.json and artifact_manifest.json: design and audit trail.',
        '- predictions.npz, bootstrap_draws.npz, folds/ and source scripts: all new models, held-out predictions and reproducible analysis.','']
    (OUT/'REVIEW.md').write_text('\n'.join(rows))
