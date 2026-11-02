"""Scientific figures and report for the nine unweighted stratified arms."""
from pathlib import Path
import json
import numpy as np,pandas as pd
OUT=Path(__file__).resolve().parent
FAMILIES=['ridge','neural_network','xgboost'];PANELS=[40,64,80]
LABELS={'ridge':'Ridge','neural_network':'Neural network','xgboost':'XGBoost'}

def name(f,p):return f+'__'+str(p)
def label(arm):
    f,p=arm.rsplit('__',1);return f'{LABELS[f]}, {p} genes'

def render(summary,paired,effects,wins,validation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter
    colors={'ridge':'#137d8b','neural_network':'#8853a3','xgboost':'#bf6d22'}
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,key,title in zip(axes.flat,['pooled_spearman','pooled_rmse','macro23_rmse','macro19_rmse'],['Pooled test Spearman ↑','Pooled test RMSE ↓','Equal-lineage RMSE: all 23 ↓','Equal-lineage RMSE: 19 larger lineages ↓']):
        for f in FAMILIES:
            a=summary[summary.family==f].sort_values('panel');ax.errorbar(a.panel,a[key+'_mean'],yerr=a[key+'_sd'],color=colors[f],marker='o',capsize=3,label=LABELS[f])
        ax.set_xticks(PANELS);ax.set_xlabel('Number of genes');ax.set_title(title);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
        if key!='pooled_spearman':ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.4f}'))
    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=3)
    fig.suptitle('Unweighted ferroptosis panels: lineage-stratified 80/20 testing\n20 matched splits; whiskers show split SD on the same cohort',fontsize=13)
    fig.savefig(OUT/'panel_comparison.png',dpi=170);fig.savefig(OUT/'panel_comparison.svg');plt.close(fig)
    selected=[(name(f,p),name(f,40)) for f in FAMILIES for p in [64,80]];counts=pd.read_csv(OUT/'split_lineage_counts.tsv',sep='\t');order=counts[counts.repeat==1].sort_values('cohort_n',ascending=False);lineages=order.lineage.tolist()
    a=np.column_stack([effects[(effects.first_model==first)&(effects.reference_model==reference)].set_index('lineage').loc[lineages,'rmse_improvement'] for first,reference in selected]);limit=float(np.max(np.abs(a)))
    fig,ax=plt.subplots(figsize=(10,9),constrained_layout=True);im=ax.imshow(a,aspect='auto',cmap='RdBu',vmin=-limit,vmax=limit)
    ax.set_yticks(range(23),[f'{v.lineage} (n={v.test_n})' for v in order.itertuples()],fontsize=9);ax.set_xticks(range(6),[LABELS[f]+'\n'+str(p)+' vs 40' for f in FAMILIES for p in [64,80]],rotation=35,ha='right',fontsize=9)
    ax.set_title('Where larger panels reduce held-out RMSE\nMean paired reduction over 20 splits; blue favors expansion',fontsize=12);fig.colorbar(im,ax=ax,shrink=.8,label='RMSE reduction');fig.savefig(OUT/'lineage_panel_effects.png',dpi=170);fig.savefig(OUT/'lineage_panel_effects.svg');plt.close(fig)
    write_report(summary,paired,effects,wins,validation)

def write_report(summary,paired,effects,wins,validation):
    s=summary.set_index('model');best_pool=s.pooled_rmse_mean.idxmin();best_rank=s.pooled_spearman_mean.idxmax();best_macro=s.macro23_rmse_mean.idxmin();protocol=json.load(open(OUT/'protocol.json'))
    rows=['# Unweighted 40-, 64- and 80-gene panels on lineage-stratified splits','',
        f'**Lowest mean pooled RMSE: {label(best_pool)} ({s.loc[best_pool,"pooled_rmse_mean"]:.6f}).** '
        f'Highest mean pooled Spearman: {label(best_rank)} ({s.loc[best_rank,"pooled_spearman_mean"]:.5f}). '
        f'Lowest mean equal-lineage RMSE: {label(best_macro)} ({s.loc[best_macro,"macro23_rmse_mean"]:.6f}). '
        'These are descriptive point estimates; the paired intervals below assess differences conditional on the fitted predictions.','',
        'Completed all nine model/panel combinations across the **same 20 saved lineage-stratified 80/20 splits**, excluding the singleton Cervix line. '
        'Every split contains **595 training and 149 test cell lines**, with **all 23 retained lineages on both sides** and **zero identity-group overlap**. '
        'The cohort contains 744 lines and 733 identity groups. All training is unweighted.','',
        'Eight arms contributed 160 newly fitted outer models. The 20 completed unweighted 64-gene NN models were reused after exact input/split checks, artifact-hash validation and full nested native-MLP replays in repeats 1 and 20. '
        'The nine arms therefore comprise 180 outer models.','',
        '## Mean test scores across 20 splits','',
        '| Model | Genes | Pooled Spearman ↑ | Pooled RMSE ↓ | Pooled R² ↑ | Macro-23 RMSE ↓ | Macro-19 RMSE ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for f in FAMILIES:
        for p in PANELS:
            v=s.loc[name(f,p)];rows.append(f'| {LABELS[f]} | {p} | {v.pooled_spearman_mean:.5f} | {v.pooled_rmse_mean:.6f} | {v.pooled_r2_mean:.5f} | {v.macro23_rmse_mean:.6f} | {v.macro19_rmse_mean:.6f} |')
    rows+=['','Scores are calculated separately on each repeat’s test set and then averaged; predictions are not averaged across repeats before scoring. '
        'Macro-23 RMSE equally averages all 23 lineage-specific RMSE values. Macro-19 uses the 19 cohort lineages with at least 10 samples as a secondary metric. '
        'Very small test subsets make lineage correlations unstable or undefined, so no macro Spearman is used.','',
        f'![Panel comparison]({OUT}/panel_comparison.png)','','## Paired panel expansion effects','',
        'Positive improvement favors the larger panel: higher Spearman or lower RMSE. Intervals use 5,000 paired whole-lineage draws shared across all models and all split-specific test sets. '
        'They are conditional on fitted predictions, do not refit models and are not adjusted for multiple comparisons.','',
        '| Model | Contrast | Metric | Mean improvement | Favorable splits | Conditional 95% interval |',
        '|---|---|---|---:|---:|---|']
    for v in paired.itertuples():
        af,ap=v.first_model.rsplit('__',1);bf,bp=v.reference_model.rsplit('__',1)
        if af!=bf or v.metric not in ['pooled_spearman','pooled_rmse','macro23_rmse']:continue
        rows.append(f'| {LABELS[af]} | {ap} vs {bp} | {v.metric} | {v.improvement:+.6f} | {v.favorable_repeats}/20 | [{v.conditional_ci_low:+.6f}, {v.conditional_ci_high:+.6f}] |')
    rows+=['','## Cross-model comparisons at matched panel sizes','',
        '| Genes | Contrast | Metric | Mean improvement | Favorable splits | Conditional 95% interval |',
        '|---|---|---|---:|---:|---|']
    for v in paired.itertuples():
        af,ap=v.first_model.rsplit('__',1);bf,bp=v.reference_model.rsplit('__',1)
        if af==bf or ap!=bp or v.metric not in ['pooled_rmse','macro23_rmse']:continue
        rows.append(f'| {ap} | {LABELS[af]} vs {LABELS[bf]} | {v.metric} | {v.improvement:+.6f} | {v.favorable_repeats}/20 | [{v.conditional_ci_low:+.6f}, {v.conditional_ci_high:+.6f}] |')
    rows+=['','All 22 planned paired comparisons and all five metrics are saved in paired_comparisons.tsv.','',
        '## Lineage consistency of panel expansion','',
        '| Model | Contrast | Lineages with lower mean test RMSE |',
        '|---|---|---:|']
    for v in wins.itertuples():
        af,ap=v.first_model.rsplit('__',1);bf,bp=v.reference_model.rsplit('__',1)
        if af==bf:rows.append(f'| {LABELS[af]} | {ap} vs {bp} | {v.rmse_favorable_lineages}/23 |')
    rows+=['','These are descriptive signs of mean effects across splits, not separately established effects in each lineage. '
        'The heatmap labels show the per-repeat test sample count for each lineage; one-point lineages are particularly uncertain.','',
        f'![Lineage panel effects]({OUT}/lineage_panel_effects.png)','','## Fixed data, splits and model settings','',
        '- All models use the same erastin sensitivity target, negative harmonized erastin AUC (higher means more sensitive), and DepMap log2(TPM+1) expression. The existing 40/64/80 panels and their order are unchanged; larger panels contain the smaller panels as prefixes.',
        '- All 20 outer and five-fold inner assignments are byte-identical to the preceding lineage-stratified experiment. Every outer train/test pair contains all 23 lineages. Related identities stay together.',
        '- Ampulla has only two total lines: one outer train and one outer test. Its single training line is retained in every inner training fit and never used for inner validation. Thus each inner training set contains 23 lineages and each inner validation set contains 22. The outer Ampulla test is never used for tuning.',
        '- Ridge: original 17-value alpha grid, train-only StandardScaler for X, target centering and SVD. No sample weights.',
        '- Neural network: original four width/penalty candidates (16/32 tanh units and alpha 10/100), three fixed per-repeat initializations, L-BFGS with max_iter 2000, max_fun 40000 and tolerance 0.001. X/y scalers use training rows only. The verified adapter receives exact unit weights and reproduces native MLPRegressor predictions.',
        '- XGBoost: original 36 candidates: depth 1/2/3 × minimum child weight 5/20 × L2 penalty 1/20 × 100/300/600 rounds. Fixed CPU hist, max_bin 64, learning rate 0.03, uniform row/column sampling 0.8, and the matched per-repeat sampling seeds. No sample weights; target mean/SD use current training rows only.',
        '- Each arm is tuned independently by minimum unweighted equal-mean five-fold inner RMSE, preserving the original grid order for ties. Hyperparameters, panels and seeds are not changed after observing outer-test predictions.',
        '', '## Validation','',
        f'- Audited all {validation["total_outer_models"]} outer models ({validation["new_outer_models"]} new and {validation["reused_nn64_outer_models"]} reused). Maximum saved-prediction reconstruction error {validation["prediction_max_error"]:.3g}.',
        f'- Recomputed all {validation["candidate_fold_scores_checked"]:,} candidate-fold scores; maximum RMSE discrepancy {validation["inner_rmse_max_error"]:.3g}. Verified selected grid indices, source hashes and exact panel/input alignment.',
        f'- Completed {validation["identity_group_exclusion_checks"]:,} outer/inner identity-exclusion checks. Every inner training set retains all 23 lineages.',
        f'- Recomputed unweighted objectives for all {validation["neural_training_objectives_checked"]} selected NN members; maximum discrepancy {validation["neural_objective_max_error"]:.3g}. Verified all saved NN sample weights equal one.',
        '- Preflight full nested native-MLP replays for the reused NN64 arm in repeats 1/20 had exact zero discrepancy in all candidate inner and selected outer predictions. Independent ridge APIs and XGBoost native/prefix/wrapper checks passed for all three panel sizes in repeats 1/20.',
        '- Selected settings were independently refitted for all nine arms in repeats 1/20, including five inner fits each. Ridge uses a separate sklearn fit, NN uses native MLPRegressor, and XGBoost uses its sklearn wrapper.',
        '- Bootstrap metric calculations were checked against explicitly expanded SciPy samples using the actual split-specific predictions. Reused NN64 predictions exactly match the preceding stratified experiment.',
        f'- Convergence warnings: {validation["convergence_warning_count"]}; total recorded warnings: {validation["warning_count"]:,}. Categories and per-fit diagnostics are saved.','',
        '## Interpretation limits','',
        'This design tests new cell lines from lineages represented in training. It does not measure transfer to absent cancer lineages, and its scores should not be pooled with the earlier leave-lineage-out results. '
        'Twenty split-specific test sets overlap and reuse this biological cohort; their 20×149 predictions are not independent observations. '
        'The whole-lineage bootstrap is conditional on fitted predictions and summarizes variability across the observed lineages, not independent biological replication. '
        'Only 724 of 744 distinct cell lines happen to be tested at least once across these random holdouts; coverage is saved by sample. '
        'Tiny lineage test sets remain uncertain. Earlier analyses of this same cohort informed panel/model exploration, so no winner is externally validated. '
        'The harmonized AUC upstream provenance limitation remains unresolved.','',
        '## Files','',
        '- summary.tsv: all nine arms, mean scores and split dispersion.',
        '- paired_comparisons.tsv and paired_seed_differences.tsv: planned panel/family contrasts and conditional intervals.',
        '- lineage_paired_effects.tsv and lineage_win_counts.tsv: test-lineage consistency.',
        '- split_lineage_counts.tsv, inner_split_lineage_counts.tsv and sample_test_coverage.tsv: all saved membership audits.',
        '- per_repeat_pooled_metrics.tsv, per_repeat_macro_metrics.tsv and per_lineage_metrics.tsv: individual split/lineage scores.',
        '- selected_parameters.tsv, tuning_scores.tsv, parameter_counts.tsv, validation.json, preflight_validation.json and protocol.json: tuning and audit trail.',
        '- predictions.npz, bootstrap_draws.npz, panel_inputs.npz, shared_splits.npz and folds/: matched data, all 180 models and held-out predictions.',
        '- source_metadata/ and reuse_manifest.json: exact provenance for the 20 reused NN64 models.',
        '- Source scripts, source hashes and a checksummed artifact manifest.','']
    (OUT/'REVIEW.md').write_text('\n'.join(rows))
