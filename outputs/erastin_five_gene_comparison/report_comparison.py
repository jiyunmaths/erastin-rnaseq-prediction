"""Render the exploratory fixed five-gene versus full-panel comparison."""
import json
from pathlib import Path
import numpy as np,pandas as pd
OUT=Path(__file__).resolve().parent;FAMILIES=['ridge','neural_network'];LABELS={'ridge':'Ridge','neural_network':'Neural network'}
ORDER=['ridge__5','ridge__64','neural_network__5','neural_network__64']

def name(f,p):return f+'__'+str(p)

def render(summary,paired,effects,wins,validation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter
    s=summary.set_index('model').loc[ORDER];colors=['#63adba','#137d8b','#b993cd','#8853a3'];names=['Ridge\n5 genes','Ridge\n64 genes','Neural network\n5 genes','Neural network\n64 genes']
    fig,axes=plt.subplots(2,2,figsize=(11,8),constrained_layout=True)
    for ax,key,title in zip(axes.flat,['pooled_spearman','pooled_rmse','macro23_rmse','macro19_rmse'],['Pooled test Spearman ↑','Pooled test RMSE ↓','Equal-lineage RMSE: all 23 ↓','Equal-lineage RMSE: larger 19 ↓']):
        ax.errorbar(range(4),s[key+'_mean'],yerr=s[key+'_sd'],fmt='none',capsize=3,color='#adb5bd');ax.scatter(range(4),s[key+'_mean'],color=colors,s=60,zorder=3)
        ax.set_xticks(range(4),names,fontsize=9);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
        if key!='pooled_spearman':ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.4f}'))
    fig.suptitle('Exploratory five-gene versus 64-gene prediction\n20 matched lineage-stratified splits; whiskers show split SD',fontsize=13)
    fig.savefig(OUT/'panel_comparison.png',dpi=170);fig.savefig(OUT/'panel_comparison.svg');plt.close(fig)
    primary=paired[(paired.first_model.isin(['ridge__5','neural_network__5']))&(paired.reference_model.isin(['ridge__64','neural_network__64']))].copy()
    fig,axes=plt.subplots(1,3,figsize=(12,3.8),constrained_layout=True,sharey=True)
    for ax,key,title in zip(axes,['pooled_spearman','pooled_rmse','macro23_rmse'],['Pooled Spearman increase','Pooled RMSE reduction','Equal-lineage RMSE reduction']):
        for i,f in enumerate(FAMILIES):
            a=primary[(primary.first_model==name(f,5))&(primary.reference_model==name(f,64))&(primary.metric==key)].iloc[0]
            ax.hlines(i,a.conditional_ci_low,a.conditional_ci_high,color=colors[2*i+1],lw=2);ax.scatter(a.improvement,i,color=colors[2*i+1],s=55,zorder=3)
        ax.axvline(0,color='#555',linestyle='--',lw=1);ax.set_title(title,fontsize=11);ax.set_yticks([0,1],[LABELS[f] for f in FAMILIES]);ax.grid(axis='x',alpha=.18);ax.spines[['top','right']].set_visible(False);ax.ticklabel_format(axis='x',style='sci',scilimits=(-2,2))
    axes[0].invert_yaxis();fig.suptitle('Five-gene improvement over the 64-gene panel: positive favors five\nConditional 95% lineage intervals; prior gene selection is not adjusted for',fontsize=12)
    fig.savefig(OUT/'paired_differences.png',dpi=170);fig.savefig(OUT/'paired_differences.svg');plt.close(fig)
    count=pd.read_csv(OUT/'split_lineage_counts.tsv',sep='\t');order=count[count.repeat==1].sort_values('cohort_n',ascending=False);labels=order.lineage.tolist()
    vals=np.column_stack([effects[(effects.first_model==name(f,5))&(effects.reference_model==name(f,64))].set_index('lineage').loc[labels,'rmse_improvement'] for f in FAMILIES]);limit=float(np.max(np.abs(vals)))
    fig,ax=plt.subplots(figsize=(7.8,8.5),constrained_layout=True);im=ax.imshow(vals,aspect='auto',cmap='RdBu',vmin=-limit,vmax=limit)
    ax.set_yticks(range(23),[f'{v.lineage} (n={v.test_n})' for v in order.itertuples()],fontsize=9);ax.set_xticks([0,1],['Ridge: 5 vs 64','NN: 5 vs 64']);fig.colorbar(im,ax=ax,shrink=.75,label='Mean test RMSE reduction')
    ax.set_title('Lineage-specific panel differences\nBlue favors five genes; tiny test subsets are uncertain',fontsize=12);fig.savefig(OUT/'lineage_comparison.png',dpi=170);fig.savefig(OUT/'lineage_comparison.svg');plt.close(fig)
    write_report(summary,paired,effects,wins,validation)

def write_report(summary,paired,effects,wins,validation):
    s=summary.set_index('model');best=s.pooled_rmse_mean.idxmin();f,p=best.rsplit('__',1)
    rows=['# Exploratory five-gene versus 64-gene ridge and neural-network comparison','',
        '**Fixed five-gene panel: SLC7A11, GPX4, ABCC1, ACSL4, AIFM2.** '
        f'Lowest mean pooled RMSE among the four fitted arms: **{LABELS[f]}, {p} genes ({s.loc[best,"pooled_rmse_mean"]:.6f})**.','',
        '**Selection limitation:** these five genes were chosen after examining held-out importance results from this same cohort and these same 20 splits. '
        'Test responses therefore influenced feature selection. Refitting models and tuning hyperparameters inside each training split does not remove that bias. '
        'This is a retrospective exploratory comparison, not independent validation of the reduced panel, and its conditional intervals do not adjust for the gene-selection step.','',
        'Both panels use the exact saved 20 lineage-stratified grouped 80/20 splits: **595 training and 149 testing lines per repeat**, '
        '744 lines and 733 identity groups after excluding the singleton Cervix line. All 23 retained lineages occur on both sides of every outer split, with zero identity-group overlap. '
        'The five-gene panel uses 59 fewer genes (92.2% fewer measured features).','',
        '## Mean test scores across 20 splits','',
        '| Model | Genes | Pooled Spearman ↑ | Pooled RMSE ↓ | Pooled R² ↑ | Macro-23 RMSE ↓ | Macro-19 RMSE ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for f in FAMILIES:
        for p in [5,64]:
            v=s.loc[name(f,p)];rows.append(f'| {LABELS[f]} | {p} | {v.pooled_spearman_mean:.5f} | {v.pooled_rmse_mean:.6f} | {v.pooled_r2_mean:.5f} | {v.macro23_rmse_mean:.6f} | {v.macro19_rmse_mean:.6f} |')
    rows+=['','Scores are calculated separately within each repeat’s held-out test set and then averaged. Test observations are not stacked as independent samples. '
        'Macro-23 RMSE equally averages all 23 lineage-specific test RMSEs; macro-19 restricts the average to the 19 cohort lineages with at least 10 total samples. '
        'No macro Spearman is calculated because some lineage test sets contain only one line.','',
        f'![Panel comparison]({OUT}/panel_comparison.png)','','## Five-gene changes relative to the matched 64-gene model','',
        '| Model | Pooled RMSE change (%) | Macro-23 RMSE change (%) | Splits with lower pooled RMSE | Lineages with lower mean RMSE |',
        '|---|---:|---:|---:|---:|']
    for f in FAMILIES:
        first=name(f,5);reference=name(f,64);a=s.loc[first];b=s.loc[reference];v=paired[(paired.first_model==first)&(paired.reference_model==reference)&(paired.metric=='pooled_rmse')].iloc[0];w=wins[(wins.first_model==first)&(wins.reference_model==reference)].iloc[0]
        rows.append(f'| {LABELS[f]} | {100*(a.pooled_rmse_mean/b.pooled_rmse_mean-1):+.3f}% | {100*(a.macro23_rmse_mean/b.macro23_rmse_mean-1):+.3f}% | {int(v.favorable_repeats)}/20 | {int(w.rmse_favorable_lineages)}/23 |')
    rows+=['','Positive percentage changes in the table above mean higher error for the five-gene panel. '
        'In the paired-effect table below, positive improvement instead means the five-gene model is better: Spearman(5)−Spearman(64), or RMSE(64)−RMSE(5).','',
        '| Model | Metric | Mean improvement | Favorable splits | Conditional 95% interval |',
        '|---|---|---:|---:|---|']
    for f in FAMILIES:
        for v in paired[(paired.first_model==name(f,5))&(paired.reference_model==name(f,64))].itertuples():
            rows.append(f'| {LABELS[f]} | {v.metric} | {v.improvement:+.6f} | {v.favorable_repeats}/20 | [{v.conditional_ci_low:+.6f}, {v.conditional_ci_high:+.6f}] |')
    rows+=['','Intervals use 5,000 paired whole-lineage bootstrap draws shared across models and all 20 split-specific test sets. '
        'They are conditional on the fitted models and fixed, previously selected gene panel. Models and feature selection are not repeated in the bootstrap. '
        'There is no adjustment for multiple comparisons or post-selection optimism. '
        '**An interval containing zero does not establish equivalence or noninferiority; no equivalence margin was specified.**','',
        f'![Paired differences]({OUT}/paired_differences.png)','','## Lineage-specific performance','',
        '| Lineage | Test lines per split | Ridge RMSE reduction: 5 vs 64 | NN RMSE reduction: 5 vs 64 |',
        '|---|---:|---:|---:|']
    a=effects[(effects.first_model=='ridge__5')&(effects.reference_model=='ridge__64')].set_index('lineage');b=effects[(effects.first_model=='neural_network__5')&(effects.reference_model=='neural_network__64')].set_index('lineage')
    count=pd.read_csv(OUT/'split_lineage_counts.tsv',sep='\t');labels=count[count.repeat==1].sort_values('cohort_n',ascending=False).lineage
    for l in labels:rows.append(f'| {l} | {int(a.loc[l,"test_n"])} | {a.loc[l,"rmse_improvement"]:+.6f} | {b.loc[l,"rmse_improvement"]:+.6f} |')
    rows+=['','Positive reductions favor the five-gene panel. These are mean effects across overlapping random holdouts and are descriptive, especially for one- to three-line test subsets.','',
        f'![Lineage comparison]({OUT}/lineage_comparison.png)','','## Prediction agreement','',
        '| Family | Mean Pearson correlation of 5- and 64-gene predictions | Mean Spearman correlation | Mean prediction-to-prediction RMSE |',
        '|---|---:|---:|---:|']
    agreement=pd.read_csv(OUT/'prediction_agreement.tsv',sep='\t').groupby('family')[['pearson','spearman','prediction_rmse']].mean()
    for f in FAMILIES:
        v=agreement.loc[f];rows.append(f'| {LABELS[f]} | {v.pearson:.4f} | {v.spearman:.4f} | {v.prediction_rmse:.6f} |')
    rows+=['','Prediction agreement quantifies similarity between fitted outputs; it does not establish equal accuracy or independent validation.','',
        '## Fitting and validation','',
        '- The five genes were fixed in the user-specified order before these fits. Their columns exactly match the verified 64-gene expression matrix. No per-split gene reselection or additional gene search is performed.',
        '- The five-gene models were fitted from scratch using only those five features. Both panels were separately tuned using the same five inner validation folds and the same original grids. All training is unweighted.',
        '- Ridge retains the original 17-alpha grid, training-only StandardScaler and target centering. NN retains 16/32 tanh units × alpha 10/100, L-BFGS, max_iter 2000, max_fun 40000, tolerance 0.001 and the same three initialization seeds per repeat. X/y scalers use current training rows only.',
        '- Every inner training set retains all 23 lineages. Ampulla has one outer-training line, so that line stays in every inner training fit and never serves as inner validation; inner validation contains 22 lineages. The outer Ampulla test remains untouched.',
        f'- Audited {validation["total_outer_models"]} outer models: {validation["new_outer_models"]} newly fitted five-gene models plus {validation["reused_64gene_outer_models"]} verified 64-gene models. Both reused prediction arrays and scores exactly match the prior comparison.',
        f'- Reconstructed all {validation["candidate_fold_scores_checked"]:,} candidate-fold scores; maximum discrepancy {validation["inner_rmse_max_error"]:.3g}. Saved-prediction maximum discrepancy {validation["prediction_max_error"]:.3g}.',
        f'- Verified {validation["identity_group_exclusion_checks"]} outer/inner identity exclusions, common split arrays, training-only scalers and source hashes. All saved NN weights equal one.',
        f'- Recomputed unweighted objectives for all {validation["neural_training_objectives_checked"]} selected NN members; maximum error {validation["neural_objective_max_error"]:.3g}.',
        '- Preflight independently replayed every inner candidate and selected outer prediction for five-gene ridge/NN models in repeats 1/20, and selected reused 64-gene fits. Independent selected-setting checks cover all four arms in repeats 1/20 and all corresponding inner folds.',
        f'- Convergence warnings: {validation["convergence_warning_count"]}; total recorded warnings: {validation["warning_count"]:,}. Full categories and per-fit diagnostics are saved.',
        '- Bootstrap calculations were independently checked against explicitly expanded SciPy samples using the actual test predictions.','',
        '## Interpretation limits','',
        'This analysis answers the requested retrospective comparison on these saved splits. It cannot provide an unbiased confirmatory estimate for a five-gene panel selected using those earlier test responses. '
        'A fresh cohort or prospectively reserved test set is needed to validate the fixed panel independently. '
        'Twenty repeats share the same biological samples and overlapping training/test sets, so they are not 20 independent biological validations. '
        'Only 724/744 lines happen to occur in at least one test set across the random holdouts; coverage is saved. '
        'This design tests new lines from represented cancer lineages, not absent-lineage transfer. The harmonized erastin AUC upstream provenance limitation remains unresolved.','',
        '## Files','',
        '- summary.tsv and paired_comparisons.tsv: aggregate scores and conditional paired differences.',
        '- per_repeat_pooled_metrics.tsv, per_repeat_macro_metrics.tsv, paired_seed_differences.tsv and per_lineage_metrics.tsv: split-level and lineage results.',
        '- lineage_paired_effects.tsv, lineage_win_counts.tsv and prediction_agreement.tsv: consistency and prediction similarity.',
        '- selected_parameters.tsv, tuning_scores.tsv, parameter_counts.tsv, protocol.json, preflight_validation.json and validation.json: tuning and audit trail.',
        '- panel_inputs.npz, shared_splits.npz, predictions.npz, bootstrap_draws.npz and folds/: matched data, all models and held-out predictions.',
        '- source_metadata/, source_manifest.json and reuse_manifest.json: original 64-gene artifacts and exact reuse provenance.',
        '- Checksummed artifact manifest and all analysis scripts.','']
    (OUT/'REVIEW.md').write_text('\n'.join(rows))
