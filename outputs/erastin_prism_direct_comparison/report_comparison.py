"""Report new PRISM-only nested fits and shared-split model comparisons."""
import json
from pathlib import Path
import numpy as np,pandas as pd
OUT=Path(__file__).resolve().parent;FAMILIES=['ridge','elastic_net','neural_network'];LABELS={'ridge':'Ridge','elastic_net':'Elastic Net','neural_network':'Neural network','training_mean':'Training mean'}
ORDER=[f+'__'+str(p) for f in FAMILIES for p in [5,64]]+['training_mean'];COLORS=['#63adba','#137d8b','#f4ad62','#c96913','#b993cd','#8853a3','#555555']

def name(f,p):return f+'__'+str(p)
def pretty(arm):
    if arm=='training_mean':return 'Training-mean control'
    f,p=arm.rsplit('__',1);return LABELS[f]+' ('+p+')'
def fmt(v,d=6):return f'{v:.{d}f}' if np.isfinite(v) else 'NA'

def render(summary,paired,effects,wins,validation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter
    s=summary.set_index('model').loc[ORDER];labels=[('Neural\nnetwork' if f=='neural_network' else LABELS[f])+'\n'+str(p)+' genes' for f in FAMILIES for p in [5,64]]+['Training\nmean'];fig,axes=plt.subplots(2,2,figsize=(14,8.5),constrained_layout=True)
    for ax,key,title in zip(axes.flat,['pooled_rmse','pooled_spearman','pooled_r2','macro_all_rmse'],['Pooled test RMSE ↓','Pooled test Spearman ↑','Pooled test R² ↑','Equal-lineage test RMSE: all 20 ↓']):
        ax.errorbar(range(7),s[key+'_mean'],yerr=s[key+'_sd'],fmt='none',capsize=3,color='#adb5bd');ax.scatter(range(7),s[key+'_mean'],color=COLORS,s=55,zorder=3);ax.set_xticks(range(7),labels,fontsize=8);ax.set_xlim(-.5,6.5);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
        if key.endswith('rmse'):ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.4f}'))
    fig.suptitle('Models trained and tuned directly on PRISM erastin AUC\n20 shared lineage-stratified splits; points = split means, whiskers = split SD',fontsize=13);fig.savefig(OUT/'model_comparison.png',dpi=170);fig.savefig(OUT/'model_comparison.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12.5,4.5),constrained_layout=True,sharey=True)
    for ax,key,title in zip(axes,['pooled_rmse','pooled_spearman','macro_all_rmse'],['Pooled RMSE reduction','Pooled Spearman increase','Equal-lineage RMSE reduction']):
        for i,f in enumerate(FAMILIES):
            v=paired[(paired.first_model==name(f,5))&(paired.reference_model==name(f,64))&(paired.metric==key)].iloc[0]
            if np.isfinite(v.improvement):ax.hlines(i,v.conditional_ci_low,v.conditional_ci_high,color=COLORS[2*i+1],lw=2);ax.scatter(v.improvement,i,color=COLORS[2*i+1],s=45,zorder=3)
        ax.axvline(0,color='#555',linestyle='--',lw=1);ax.set_yticks(range(3),[LABELS[f] for f in FAMILIES]);ax.set_title(title,fontsize=11);ax.grid(axis='x',alpha=.18);ax.spines[['top','right']].set_visible(False);ax.ticklabel_format(axis='x',style='sci',scilimits=(-2,2))
    axes[0].invert_yaxis();fig.suptitle('Five-gene improvement over 64 genes: positive favors five\nConditional 95% lineage bootstrap intervals; no refitting or comparison adjustment',fontsize=12);fig.savefig(OUT/'paired_panel_differences.png',dpi=170);fig.savefig(OUT/'paired_panel_differences.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(9.5,5),constrained_layout=True)
    for i,arm in enumerate(ORDER[:-1]):
        v=paired[(paired.first_model==arm)&(paired.reference_model=='training_mean')&(paired.metric=='pooled_rmse')].iloc[0];ax.hlines(i,v.conditional_ci_low,v.conditional_ci_high,color=COLORS[i],lw=2);ax.scatter(v.improvement,i,color=COLORS[i],s=45,zorder=3)
    ax.set_yticks(range(6),[pretty(a) for a in ORDER[:-1]]);ax.invert_yaxis();ax.axvline(0,color='#555',linestyle='--',lw=1);ax.set_xlabel('Mean test RMSE reduction versus training-mean prediction');ax.set_title('Does expression improve on a training-mean control?\nConditional 95% lineage bootstrap intervals; positive favors the expression model',fontsize=11);ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False);fig.savefig(OUT/'training_mean_comparison.png',dpi=170);fig.savefig(OUT/'training_mean_comparison.svg');plt.close(fig)
    count=pd.read_csv(OUT/'split_lineage_counts.tsv',sep='\t');order=count[count.repeat==1].sort_values('cohort_n',ascending=False);lineages=order.lineage.tolist();values=np.column_stack([effects[(effects.first_model==name(f,5))&(effects.reference_model==name(f,64))].set_index('lineage').loc[lineages,'rmse_improvement'] for f in FAMILIES]);limit=float(np.max(np.abs(values)));fig,ax=plt.subplots(figsize=(8,8),constrained_layout=True);im=ax.imshow(values,aspect='auto',cmap='RdBu',vmin=-limit,vmax=limit);ax.set_yticks(range(20),[f'{v.lineage} (test n={v.test_n})' for v in order.itertuples()],fontsize=9);ax.set_xticks(range(3),[LABELS[f] for f in FAMILIES]);fig.colorbar(im,ax=ax,shrink=.75,label='Mean RMSE reduction: 5 versus 64');ax.set_title('Lineage-specific panel differences\nBlue favors five genes; tiny test subsets are uncertain',fontsize=12);fig.savefig(OUT/'lineage_panel_differences.png',dpi=170);fig.savefig(OUT/'lineage_panel_differences.svg');plt.close(fig)
    write_report(summary,paired,effects,wins,validation)

def write_report(summary,paired,effects,wins,validation):
    s=summary.set_index('model');models=s.loc[ORDER[:-1]];best=models.pooled_rmse_mean.idxmin();base=s.loc['training_mean'];protocol=json.load(open(OUT/'protocol.json'));params=pd.read_csv(OUT/'selected_parameters.tsv',sep='\t');coverage=pd.read_csv(OUT/'sample_test_coverage.tsv',sep='\t');counts=pd.read_csv(OUT/'inner_split_lineage_counts.tsv',sep='\t');split=counts.groupby(['repeat','inner_fold']).agg(train_lineages=('train_n',lambda v:int(np.sum(v>0))),validation_lineages=('validation_n',lambda v:int(np.sum(v>0))));sparsity=pd.read_csv(OUT/'elastic_net_sparsity_summary.tsv',sep='\t')
    best_rho=models.pooled_spearman_mean.idxmax();gain=100*(base.pooled_rmse_mean-s.loc[best,'pooled_rmse_mean'])/base.pooled_rmse_mean
    rows=['# Direct PRISM erastin training: five versus 64 genes','',
        f'**All 120 models were newly trained on PRISM erastin AUC.** Lowest mean pooled test RMSE among the six expression arms: **{pretty(best)}**, {s.loc[best,"pooled_rmse_mean"]:.6f}. The training-mean control has mean RMSE {base.pooled_rmse_mean:.6f}. This is an internal repeated-holdout benchmark; it is separate from the previous frozen CTRP-to-PRISM transfer test.','',
        f'**The models learn a modest PRISM response signal.** The best mean RMSE is {gain:.2f}% lower than the training-mean control; all six expression arms improve on that control in 19/20 splits and their conditional RMSE-reduction intervals stay above zero. The largest mean Spearman is {s.loc[best_rho,"pooled_spearman_mean"]:.4f} for {pretty(best_rho)}. Mean test R² across the six arms ranges from {models.pooled_r2_mean.min():.4f} to {models.pooled_r2_mean.max():.4f}.','',
        'Five-gene panels have lower mean pooled and equal-lineage RMSE than their matched 64-gene counterparts, while 64-gene panels have slightly higher mean Spearman. The five-versus-64 paired intervals include zero, so the panel comparison remains uncertain. Within 64 genes, NN has a small RMSE advantage over ridge and Elastic Net under conditional, unadjusted intervals; five-gene ridge still has the lowest mean RMSE among all six arms.','',
        'The fixed five-gene panel is **SLC7A11, GPX4, ABCC1, ACSL4, AIFM2**. Both panels are separately tuned inside each outer training set, using the same inner folds. No PRISM-driven gene reselection or lineage training weights are used.','',
        '## Test scores','',
        '| Model | Genes | Mean Spearman ↑ | Mean RMSE ↓ | Mean R² ↑ | Mean MAE ↓ | Macro-20 RMSE ↓ | Macro-13 RMSE ↓ |','|---|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ORDER:
        v=s.loc[arm];rows.append(f'| {LABELS[v.family]} | {int(v.panel) if v.panel else "–"} | {fmt(v.pooled_spearman_mean,4)} | {fmt(v.pooled_rmse_mean)} | {fmt(v.pooled_r2_mean,4)} | {fmt(v.pooled_mae_mean)} | {fmt(v.macro_all_rmse_mean)} | {fmt(v.macro_large_rmse_mean)} |')
    rows+=['','Each metric is calculated on one 78-line test set and then averaged over 20 splits; predictions are not stacked into independent repeated observations. Macro-20 RMSE equally weights all 20 lineage-specific test RMSEs. Macro-13 is secondary and equally weights the 13 lineages with at least 10 cohort lines. Correlations are undefined for constant predictions and are retained as NA; their defined-repeat counts are saved. The training-mean control uses only each outer-training target mean, so no test response enters the control prediction.','',f'![Model comparison]({OUT}/model_comparison.png)','','## Five-gene versus 64-gene comparisons','',
        '| Family | Pooled RMSE reduction: 5 vs 64 | Relative RMSE reduction | Favorable splits | Favorable lineages | Conditional 95% interval |','|---|---:|---:|---:|---:|---|']
    for f in FAMILIES:
        first=name(f,5);ref=name(f,64);v=paired[(paired.first_model==first)&(paired.reference_model==ref)&(paired.metric=='pooled_rmse')].iloc[0];w=wins[(wins.first_model==first)&(wins.reference_model==ref)].iloc[0];pct=100*v.improvement/s.loc[ref,'pooled_rmse_mean'];rows.append(f'| {LABELS[f]} | {v.improvement:+.6f} | {pct:+.3f}% | {int(v.favorable_repeats)}/20 | {int(w.rmse_favorable_lineages)}/20 | [{v.conditional_ci_low:+.6f}, {v.conditional_ci_high:+.6f}] |')
    rows+=['','Positive improvement favors the five-gene panel. No equivalence or noninferiority margin was specified; intervals containing zero do not establish equal performance.','',
        '| Family | Metric | Improvement: 5 vs 64 | Favorable / defined splits | Conditional 95% interval |','|---|---|---:|---:|---|']
    for f in FAMILIES:
        for v in paired[(paired.first_model==name(f,5))&(paired.reference_model==name(f,64))&(paired.metric.isin(['pooled_spearman','macro_all_rmse']))].itertuples():rows.append(f'| {LABELS[f]} | {v.metric} | {fmt(v.improvement)} | {v.favorable_repeats}/{v.paired_defined_repeats} | [{fmt(v.conditional_ci_low)}, {fmt(v.conditional_ci_high)}] |')
    rows+=['',f'![Panel differences]({OUT}/paired_panel_differences.png)','','## Model comparisons within each panel','',
        '| Panel | First model vs reference | Mean pooled RMSE reduction | Favorable splits | Conditional 95% interval |','|---:|---|---:|---:|---|']
    for first,ref in protocol['secondary_comparisons']:
        v=paired[(paired.first_model==first)&(paired.reference_model==ref)&(paired.metric=='pooled_rmse')].iloc[0];p=int(first.rsplit('__',1)[1]);rows.append(f'| {p} | {pretty(first)} vs {pretty(ref)} | {v.improvement:+.6f} | {int(v.favorable_repeats)}/20 | [{v.conditional_ci_low:+.6f}, {v.conditional_ci_high:+.6f}] |')
    rows+=['','## Training-mean control','',
        '| Model | Mean RMSE reduction vs training mean | Favorable splits | Conditional 95% interval |','|---|---:|---:|---|']
    for arm in ORDER[:-1]:
        v=paired[(paired.first_model==arm)&(paired.reference_model=='training_mean')&(paired.metric=='pooled_rmse')].iloc[0];rows.append(f'| {pretty(arm)} | {v.improvement:+.6f} | {int(v.favorable_repeats)}/20 | [{v.conditional_ci_low:+.6f}, {v.conditional_ci_high:+.6f}] |')
    rows+=['','Positive reductions indicate a lower mean test error than predicting the training-set mean. Unlike the prior cross-assay test, these RMSE and R² values use the same PRISM response endpoint for training and testing. They assess direct within-PRISM prediction and are not directly interchangeable with the earlier CTRP benchmark values.','',f'![Training-mean control]({OUT}/training_mean_comparison.png)','','## Cohort and split design','',
        '- **386 cancer cell lines, 379 identity groups, 20 lineages.** Every lineage has at least three identity groups, so all lines can enter the benchmark without introducing a singleton-lineage exception. Cervix was already absent from the retained PRISM cohort.',
        '- Each of the 20 response-blind grouped lineage splits has **308 training and 78 testing lines** (79.8%/20.2%), with all 20 lineages on both sides and zero identity-group overlap. Per-lineage test counts use the closest feasible intact-group subset to a 20% target.',
        '- The same 20 numeric outer seeds, inner seeds and NN initialization triplets as the prior comparisons are reused. The split masks are newly generated for this PRISM cohort; CTRP masks are not copied onto different samples.',
        f'- Five inner folds per outer split. Every inner training fit retains all 20 lineages; inner validation contains {int(split.validation_lineages.min())}–{int(split.validation_lineages.max())} lineages because rare groups cannot populate every fold. Protected training-only rows: {int((np.load(OUT/"shared_splits.npz")["inner_fold"]==-2).sum())}.',
        f'- {int((coverage.test_appearances>0).sum())}/386 lines occur in at least one outer test set across the 20 random holdouts; {int((coverage.test_appearances==0).sum())} never appear in a test set. Individual counts and every lineage split are saved.',
        '- Official response source: [Broad DepMap PRISM Repurposing 20Q2 v1](https://doi.org/10.6084/m9.figshare.20564034.v1), erastin BRD-A25004090-001-08-4, HTS002 secondary-screen dose-response AUC. The cohort comes from the completed source/STR/expression-matching audit; all 386 matched lines are retained, including nine excluded from the previous transfer scoring only because they had no eligible old held-out fit.',
        '- Target is negative reported PRISM AUC; higher means more sensitive. Retain reported values above 1 and finite processed curves. No clipping, IC50 substitution, response imputation, extra response-quality cutoff or CTRP target values are used.',
        '- Expression is the same verified DepMap 24Q4 log2(TPM+1) matrix and gene mapping. Every X/y scaler is fitted on current training rows only.','',
        '## Fixed tuning grids','',
        '- **Ridge:** 17 alpha values from 0.0001 to 10,000, logarithmically spaced; training-only X standardization and target centering.',
        '- **Elastic Net:** 41 explicit alphas from 10 to 0.0001 × L1 ratios 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1 (328 candidates). Standardize X/y in current training. Cyclic coordinate descent, tol 1e-9 and max_iter 50,000. Selected outer models are native sklearn ElasticNet fits.',
        '- **NN:** 16/32 tanh hidden units × alpha 1, 10, 100, 1,000 (eight candidates). Three independently initialized members averaged for every candidate and outer fit. L-BFGS, max_iter 2,000, max_fun 40,000, tol 0.001, no early stopping; training-only X/y standardization. The grid expands the prior 10/100-alpha search at both ends before any fitting.',
        '- All arms minimize the equal-mean five-fold unweighted RMSE in original PRISM sensitivity units. Each arm, panel and outer split is independently tuned inside the common inner folds. Ties select the first fixed grid member. No outer test response enters fitting or tuning.',
        '- Ridge and Elastic Net alpha numbers are not directly comparable because their objective normalizations and target scaling differ.','',
        '## Selected settings and Elastic Net sparsity','',
        '| Family | Panel | Selected settings |','|---|---:|---|']
    for f in FAMILIES:
        for p in [5,64]:
            d=params[(params.family==f)&(params.panel==p)];cols=['alpha','l1_ratio'] if f=='elastic_net' else ['alpha'] if f=='ridge' else ['width','alpha'];settings=d.groupby(cols).size().reset_index(name='count');text=[]
            for _,v in settings.iterrows():text.append(', '.join(c+'='+f'{v[c]:g}' for c in cols)+f' ({int(v["count"])}/20)')
            rows.append(f'| {LABELS[f]} | {p} | '+ '; '.join(text)+' |')
    rows+=['','| Elastic Net input | Median nonzero genes | Range | Constant outer fits |','|---|---:|---:|---:|']
    for v in sparsity.itertuples():rows.append(f'| {v.panel} | {v.nonzero_genes_median:g} | {v.nonzero_genes_min}–{v.nonzero_genes_max} | {v.constant_outer_models}/20 |')
    rows+=['','Nonzero coefficients describe fitted sparsity, not causal gene importance. Complete split-specific coefficients and selection frequencies are saved; no new compact panel is inferred from test performance.','',
        f'![Lineage panel effects]({OUT}/lineage_panel_differences.png)','','## Validation and uncertainty','',
        f'- Audited all {validation["total_outer_models"]} newly trained outer models, reconstructed {validation["candidate_fold_scores_checked"]:,} candidate-fold scores and checked {validation["identity_group_exclusion_checks"]} outer/inner identity exclusions.',
        f'- Saved prediction maximum reconstruction error {validation["prediction_max_error"]:.3g}; inner RMSE error {validation["inner_rmse_max_error"]:.3g}. Verified training-only scalers, common masks and exact source hashes.',
        f'- Recomputed unweighted NN objectives for all {validation["neural_training_objectives_checked"]} selected ensemble members; maximum error {validation["neural_objective_max_error"]:.3g}. All saved training weights equal one.',
        f'- Checked all 65,600 inner Elastic Net candidate solutions and 40 selected outer objectives; maximum optimality-condition residual {validation["elastic_net_max_kkt_residual"]:.3g}.',
        '- Preflight replayed all inner ridge/NN candidates and selected outer fits in repeat 1 using independent native sklearn APIs. Native Elastic Net paths and augmented-row Lasso equivalence were checked across all mixing ratios and representative alphas. Selected inner/outer fits for every arm were independently replayed in repeats 1 and 20.',
        f'- Convergence warnings across all candidate/selected fits: {validation["convergence_warning_count"]}; selected outer-member convergence warnings: {validation["selected_outer_convergence_warning_count"]}. Detailed warning categories are saved.',
        '- Uncertainty uses 5,000 paired whole-lineage bootstrap draws shared across arms and repeat-specific test sets. Fits and feature selection are not repeated; no multiplicity or model-choice correction is applied. Correlation differences use common defined repeats; no undefined correlation is silently set to zero. Bootstrap arithmetic was independently checked against expanded SciPy samples, including constant-prediction controls.',
        '', '## Interpretation limits','',
        'This is an exploratory internal PRISM benchmark. The five genes were fixed from earlier CTRP importance before direct PRISM fitting, but the PRISM transfer results and this cohort were already inspected. There is no untouched final test set in this experiment. The same biological lines recur across overlapping random holdouts, so 20 seeds are not independent biological replications. The design evaluates new lines from represented lineages; it does not establish absent-lineage, patient or clinical generalization. Public processed AUC/STR checks do not replace raw-well or replicate-level QC.','',
        '## Files','',
        '- summary.tsv, paired_comparisons.tsv, paired_seed_differences.tsv, lineage_win_counts.tsv: mean scores, paired intervals and consistency.',
        '- per_repeat_pooled_metrics.tsv, per_repeat_macro_metrics.tsv, per_lineage_metrics.tsv: complete split/lineage results, including the training-mean control.',
        '- panel_inputs.npz, shared_splits.npz, split_lineage_counts.tsv, inner_split_lineage_counts.tsv, sample_test_coverage.tsv: matched features, targets, seeds and all train/validation/test memberships.',
        '- selected_parameters.tsv, tuning_scores.tsv, folds/: all tuning scores and all 120 PRISM-trained models with held-out predictions.',
        '- elastic_net_coefficients.tsv, elastic_net_gene_selection.tsv, elastic_net_sparsity_summary.tsv: fitted coefficients and stability.',
        '- protocol.json, preflight_validation.json, validation.json, source_manifest.json, artifact_manifest.json and scripts: fixed fitting choices, source provenance and checksums.','']
    (OUT/'REVIEW.md').write_text('\n'.join(rows))
