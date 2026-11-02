"""Static scientific figures and report for the Elastic Net comparison."""
import json
from pathlib import Path
import numpy as np,pandas as pd
OUT=Path(__file__).resolve().parent;FAMILIES=['ridge','elastic_net','neural_network'];LABELS={'ridge':'Ridge','elastic_net':'Elastic Net','neural_network':'Neural network'}
ORDER=[f+'__'+str(p) for f in FAMILIES for p in [5,64]]
PRIMARY=[('elastic_net__5','ridge__5'),('elastic_net__64','ridge__64'),('elastic_net__5','neural_network__5'),('elastic_net__64','neural_network__64'),('elastic_net__5','elastic_net__64')]
SHORT=['EN5 vs Ridge5','EN64 vs Ridge64','EN5 vs NN5','EN64 vs NN64','EN5 vs EN64']

def pretty(arm):
    f,p=arm.rsplit('__',1);return f'{LABELS[f]} ({p})'

def render(summary,paired,effects,wins,validation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter
    s=summary.set_index('model').loc[ORDER];colors=['#63adba','#137d8b','#f4ad62','#c96913','#b993cd','#8853a3'];names=[('Neural\nnetwork' if f=='neural_network' else LABELS[f])+'\n'+str(p)+' genes' for f in FAMILIES for p in [5,64]]
    fig,axes=plt.subplots(2,2,figsize=(12.5,8),constrained_layout=True)
    for ax,key,title in zip(axes.flat,['pooled_spearman','pooled_rmse','macro23_rmse','macro19_rmse'],['Pooled test Spearman ↑','Pooled test RMSE ↓','Equal-lineage RMSE: all 23 ↓','Equal-lineage RMSE: larger 19 ↓']):
        ax.errorbar(range(6),s[key+'_mean'],yerr=s[key+'_sd'],fmt='none',capsize=3,color='#adb5bd');ax.scatter(range(6),s[key+'_mean'],color=colors,s=60,zorder=3);ax.set_xticks(range(6),names,fontsize=9);ax.set_title(title);ax.grid(axis='y',alpha=.2);ax.spines[['top','right']].set_visible(False)
        if key!='pooled_spearman':ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.4f}'))
    fig.suptitle('Elastic Net, ridge and neural network: 5 versus 64 genes\n20 matched lineage-stratified splits; whiskers show split SD',fontsize=13);fig.savefig(OUT/'model_comparison.png',dpi=170);fig.savefig(OUT/'model_comparison.svg');plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4.8),constrained_layout=True,sharey=True)
    for ax,key,title in zip(axes,['pooled_spearman','pooled_rmse','macro23_rmse'],['Pooled Spearman increase','Pooled RMSE reduction','Equal-lineage RMSE reduction']):
        for i,(first,ref) in enumerate(PRIMARY):
            a=paired[(paired.first_model==first)&(paired.reference_model==ref)&(paired.metric==key)].iloc[0];ax.hlines(i,a.conditional_ci_low,a.conditional_ci_high,color='#c96913',lw=2);ax.scatter(a.improvement,i,color='#c96913',s=55,zorder=3)
        ax.axvline(0,color='#555',linestyle='--',lw=1);ax.set_title(title,fontsize=11);ax.set_yticks(range(5),SHORT);ax.grid(axis='x',alpha=.18);ax.spines[['top','right']].set_visible(False);ax.ticklabel_format(axis='x',style='sci',scilimits=(-2,2))
    axes[0].invert_yaxis();fig.suptitle('Positive improvement favors the first model in each label\nConditional 95% lineage intervals; prior panel selection is not adjusted for',fontsize=12);fig.savefig(OUT/'paired_differences.png',dpi=170);fig.savefig(OUT/'paired_differences.svg');plt.close(fig)
    sel=pd.read_csv(OUT/'elastic_net_gene_selection.tsv',sep='\t');params=pd.read_csv(OUT/'selected_parameters.tsv',sep='\t');fig,axes=plt.subplots(1,3,figsize=(13,7.4),constrained_layout=True,gridspec_kw={'width_ratios':[.9,1.15,1.3]})
    for ax,p,top in zip(axes[:2],[5,64],[5,20]):
        d=sel[sel.panel==p].sort_values(['selected_fits','mean_absolute_coefficient_original_y_per_training_sd'],ascending=False).head(top).iloc[::-1];ax.barh(d.gene,d.selection_frequency*100,color='#c96913');ax.set_xlim(0,106);ax.set_xlabel('Fits with a nonzero coefficient (%)');ax.set_title(f'{p}-gene panel'+('\nTop 20 by selection / coefficient' if p==64 else ''),fontsize=11);ax.set_xticks([0,25,50,75,100]);ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    for p,c in [(5,'#f4ad62'),(64,'#c96913')]:
        d=params[(params.family=='elastic_net')&(params.panel==p)].sort_values('repeat');axes[2].plot(d.repeat,d.nonzero_genes,'o-',label=f'{p}-gene input',color=c,ms=4)
    axes[2].set_title('Genes retained in each outer fit',fontsize=11);axes[2].set_xlabel('Split repeat');axes[2].set_ylabel('Nonzero coefficients');axes[2].set_xticks([1,5,10,15,20]);axes[2].set_ylim(0,66);axes[2].legend();axes[2].grid(alpha=.2);axes[2].spines[['top','right']].set_visible(False);fig.suptitle('Elastic Net sparsity and selection stability\nSelection frequency describes these fitted models; it is not causal importance',fontsize=12);fig.savefig(OUT/'elastic_net_selection.png',dpi=170);fig.savefig(OUT/'elastic_net_selection.svg');plt.close(fig)
    count=pd.read_csv(OUT/'split_lineage_counts.tsv',sep='\t');order=count[count.repeat==1].sort_values('cohort_n',ascending=False);labels=order.lineage.tolist();contrasts=PRIMARY[:4];vals=np.column_stack([effects[(effects.first_model==a)&(effects.reference_model==b)].set_index('lineage').loc[labels,'rmse_improvement'] for a,b in contrasts]);limit=float(np.max(np.abs(vals)))
    fig,ax=plt.subplots(figsize=(9,8.6),constrained_layout=True);im=ax.imshow(vals,aspect='auto',cmap='RdBu',vmin=-limit,vmax=limit);ax.set_yticks(range(23),[f'{v.lineage} (n={v.test_n})' for v in order.itertuples()],fontsize=9);ax.set_xticks(range(4),SHORT[:4],rotation=20,ha='right');fig.colorbar(im,ax=ax,shrink=.75,label='Mean test RMSE reduction');ax.set_title('Elastic Net performance by lineage\nBlue favors Elastic Net; tiny test subsets are uncertain',fontsize=12);fig.savefig(OUT/'lineage_comparison.png',dpi=170);fig.savefig(OUT/'lineage_comparison.svg');plt.close(fig)
    write_report(summary,paired,effects,wins,validation)

def write_report(summary,paired,effects,wins,validation):
    s=summary.set_index('model');best=s.pooled_rmse_mean.idxmin();selection=pd.read_csv(OUT/'elastic_net_gene_selection.tsv',sep='\t');sparsity=pd.read_csv(OUT/'elastic_net_sparsity_summary.tsv',sep='\t');parameters=pd.read_csv(OUT/'selected_parameters.tsv',sep='\t')
    rows=['# Exploratory Elastic Net comparison','',f'Lowest mean pooled RMSE among the six arms: **{pretty(best)}**, {s.loc[best,"pooled_rmse_mean"]:.6f}. All arms use the exact same 20 lineage-stratified outer splits and five inner folds.','',
        '**Elastic Net provides no clear accuracy improvement here.** With 64 input genes, mean pooled RMSE is 0.171% higher than ridge and 0.203% higher than NN, although Elastic Net wins 13/20 and 11/20 individual splits respectively. Its mean macro-23 RMSE is also higher, and paired conditional intervals include zero. The five-gene Elastic Net is numerically close to five-gene ridge. Its selected L1 ratio is 0.01 in all 20 repeats, retaining every input gene. The 64-gene models are sparser (median 37 retained, range 17–58), but the selected subsets vary across splits.','',
        '**Five-gene panel:** SLC7A11, GPX4, ABCC1, ACSL4, AIFM2. These genes were chosen after examining held-out importance from this same cohort and these same splits. Test responses therefore influenced panel selection. Training-only scaling and nested tuning do not remove that earlier bias. This is an exploratory comparison; its intervals do not adjust for panel or model selection.','',
        '## Test performance','',
        '| Model | Genes | Pooled Spearman ↑ | Pooled RMSE ↓ | Pooled R² ↑ | Macro-23 RMSE ↓ | Macro-19 RMSE ↓ |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm in ORDER:
        v=s.loc[arm];rows.append(f'| {LABELS[v.family]} | {v.panel} | {v.pooled_spearman_mean:.5f} | {v.pooled_rmse_mean:.6f} | {v.pooled_r2_mean:.5f} | {v.macro23_rmse_mean:.6f} | {v.macro19_rmse_mean:.6f} |')
    rows+=['','Each metric is computed within a repeat’s test set and then averaged over 20 repeats. Macro-23 RMSE gives every lineage equal weight. Macro-19 is secondary and restricts to 19 lineages with at least 10 cohort samples. No macro Spearman is reported because some test lineage subsets contain one line.','',f'![Model comparison]({OUT}/model_comparison.png)','','## Paired Elastic Net contrasts','',
        '| Comparison | Mean pooled RMSE reduction | RMSE reduction (%) | Favorable splits | Favorable lineages | Conditional 95% interval |','|---|---:|---:|---:|---:|---|']
    for first,ref in PRIMARY:
        v=paired[(paired.first_model==first)&(paired.reference_model==ref)&(paired.metric=='pooled_rmse')].iloc[0];w=wins[(wins.first_model==first)&(wins.reference_model==ref)].iloc[0];pct=100*(s.loc[ref,'pooled_rmse_mean']-s.loc[first,'pooled_rmse_mean'])/s.loc[ref,'pooled_rmse_mean'];rows.append(f'| {pretty(first)} vs {pretty(ref)} | {v.improvement:+.6f} | {pct:+.3f}% | {int(v.favorable_repeats)}/20 | {int(w.rmse_favorable_lineages)}/23 | [{v.conditional_ci_low:+.6f}, {v.conditional_ci_high:+.6f}] |')
    rows+=['','Positive improvement favors the first model. Favorable lineages count mean lineage-specific RMSE reductions across repeats; this is descriptive, particularly for one- to three-line test subsets.','',
        '| Comparison | Metric | Mean improvement | Favorable splits | Conditional 95% interval |','|---|---|---:|---:|---|']
    for first,ref in PRIMARY:
        for v in paired[(paired.first_model==first)&(paired.reference_model==ref)&(paired.metric.isin(['pooled_spearman','macro23_rmse']))].itertuples():rows.append(f'| {pretty(first)} vs {pretty(ref)} | {v.metric} | {v.improvement:+.6f} | {v.favorable_repeats}/20 | [{v.conditional_ci_low:+.6f}, {v.conditional_ci_high:+.6f}] |')
    rows+=['','Intervals use 5,000 paired whole-lineage bootstrap draws, shared across arms and all repeat-specific test sets. They are conditional on fitted models, fixed splits and chosen panels. No refitting, feature reselection or multiple-comparison adjustment occurs. A zero-crossing interval does not establish equivalence or noninferiority; no equivalence margin was defined.','',f'![Paired differences]({OUT}/paired_differences.png)','','## Sparsity','',
        '| Input panel | Mean retained | Median retained | Retained range | Selected alpha range | Selected L1 ratio range |','|---|---:|---:|---:|---:|---:|']
    for v in sparsity.itertuples():rows.append(f'| {v.panel} | {v.nonzero_genes_mean:.2f} | {v.nonzero_genes_median:g} | {v.nonzero_genes_min}–{v.nonzero_genes_max} | {v.alpha_min:.6g}–{v.alpha_max:.6g} | {v.l1_ratio_min:g}–{v.l1_ratio_max:g} |')
    rows+=['','Nonzero means a literal nonzero solver coefficient. This summarizes model compression, not a biologically validated gene set. L1 ratio 0 is pure ridge and 1 is pure Lasso; the fitted grid starts at 0.01. The saved ridge arms supply the L2 comparator. Elastic Net alpha numbers cannot be compared directly with Ridge alpha because objective normalization and target scaling differ.','']
    for p in [5,64]:
        d=selection[selection.panel==p].sort_values(['selected_fits','mean_absolute_coefficient_original_y_per_training_sd'],ascending=False).head(5 if p==5 else 20);rows+= [f'### {p}-gene panel: '+('all five genes' if p==5 else 'top 20 by selection frequency, then coefficient magnitude'),'',
            '| Gene | Selected fits | Positive / negative fits | Mean signed coefficient | Mean absolute coefficient |','|---|---:|---:|---:|---:|']
        for v in d.itertuples():rows.append(f'| {v.gene} | {v.selected_fits}/20 | {v.positive_fits} / {v.negative_fits} | {v.mean_coefficient_original_y_per_training_sd:+.6f} | {v.mean_absolute_coefficient_original_y_per_training_sd:.6f} |')
        rows+=['','Coefficients are original target units per one training-set expression SD. Positive predicts higher sensitivity (negative harmonized erastin AUC), conditional on the other panel genes. Correlated features can share or exchange coefficients; coefficient magnitude and frequency are not causal effects or significance tests.','']
    rows+= [f'![Elastic Net gene selection]({OUT}/elastic_net_selection.png)','','## Hyperparameters','',
        '| Input panel | Alpha | L1 ratio | Outer fits |','|---|---:|---:|---:|']
    d=parameters[parameters.family=='elastic_net'].groupby(['panel','alpha','l1_ratio']).size().reset_index(name='count')
    for v in d.itertuples():rows.append(f'| {v.panel} | {v.alpha:.8g} | {v.l1_ratio:g} | {v.count} |')
    for v in sparsity.itertuples():rows.append(f'\nPanel {v.panel}: lowest-alpha boundary selected in {v.lower_alpha_boundary_fits}/20 fits, highest-alpha boundary in {v.upper_alpha_boundary_fits}/20, lowest L1 ratio in {v.lowest_l1_ratio_fits}/20 and pure Lasso in {v.pure_lasso_fits}/20. The grid was fixed before fitting and was not expanded after examining test performance.')
    rows+=['',f'![Lineage comparison]({OUT}/lineage_comparison.png)','','## Design and validation','',
        '- 744 cell lines, 733 identity groups, 23 lineages after excluding the singleton Cervix line. Every repeat has 595 training and 149 testing lines, all 23 lineages on each side and zero identity-group overlap.',
        '- All six arms use unweighted training. The 80 saved ridge/NN models are copied and hash-verified; 40 new Elastic Net models were fitted. All copied baseline predictions and scores exactly reproduce the prior five-gene comparison.',
        '- Elastic Net scales X and y using only current training rows. Each inner fold fits its own scalers. Tune 41 explicit alpha values from 10 to 0.0001 times L1 ratios 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1, selecting the lowest equal-mean five-fold RMSE in original target units. Ties select the first grid member.',
        '- Objective on standardized X/y: squared error/(2n) + alpha × L1 ratio × sum(abs(beta)) + alpha × (1 − L1 ratio) × sum(beta²)/2, with an unpenalized intercept. Inner paths use training-only Gram matrices and exact centering. Selected outer models use native sklearn ElasticNet; cyclic coordinate descent, tolerance 1e-9, max_iter 50,000.',
        '- Ridge retains the existing 17-alpha grid and training-only standardization. NN retains the existing 16/32 tanh units × alpha 10/100 grid, three initialization seeds and L-BFGS settings. Their tuning and saved models are unchanged.',
        '- Ampulla has one outer-training line, retained in every inner training fit and excluded from inner validation. Thus inner training has 23 lineages and inner validation has 22; outer Ampulla testing stays untouched.',
        f'- Audited all {validation["total_outer_models"]} outer models and reconstructed {validation["candidate_fold_scores_checked"]:,} candidate-fold RMSE scores. Saved prediction max error {validation["prediction_max_error"]:.3g}; inner RMSE max error {validation["inner_rmse_max_error"]:.3g}.',
        f'- Verified {validation["identity_group_exclusion_checks"]} outer/inner identity exclusions, current-training scalers, unweighted NN objectives for {validation["neural_training_objectives_checked"]} ensemble members, all source hashes and byte-identical splits.',
        f'- Checked Elastic Net optimality conditions for all 65,600 inner candidate solutions and 40 selected outer fits; maximum residual {validation["elastic_net_max_kkt_residual"]:.3g}. Recomputed all 40 selected Elastic Net objectives.',
        '- Native independent ElasticNet refits and an equivalent augmented-row Lasso reproduce representative inner paths across all eight mixing ratios and alpha endpoints. An independent augmented Lasso also reproduces selected inner and outer models for both panels in repeats 1 and 20.',
        f'- Convergence warnings: {validation["convergence_warning_count"]}; recorded warnings overall: {validation["warning_count"]:,}. Bootstrap arithmetic checked against explicitly expanded SciPy samples.',
        '', '## Interpretation limits','',
        'These repeats test new cell lines from represented cancer lineages. They do not assess transfer to absent lineages. The same biological samples recur across overlapping training/test sets; 20 seeds are not 20 independent biological validations. Only 724/744 lines occur in at least one test set. The full dataset has informed earlier model and panel choices, especially the five-gene panel, so a fresh cohort or prospectively reserved test set is needed for confirmation. The upstream harmonized erastin AUC provenance limitation remains unresolved.','',
        '## Saved artifacts','',
        '- summary.tsv, paired_comparisons.tsv, paired_seed_differences.tsv: means, conditional intervals and split consistency.',
        '- per_repeat_pooled_metrics.tsv, per_repeat_macro_metrics.tsv, per_lineage_metrics.tsv, lineage_paired_effects.tsv, lineage_win_counts.tsv: complete split and lineage scores.',
        '- elastic_net_coefficients.tsv, elastic_net_gene_selection.tsv, elastic_net_sparsity_summary.tsv: all coefficients and selection frequencies.',
        '- selected_parameters.tsv, tuning_scores.tsv, parameter_counts.tsv, elastic_net_solver_checks.tsv: tuning and numerical diagnostics.',
        '- protocol.json, preflight_validation.json, validation.json: frozen fitting choices and completed audit.',
        '- panel_inputs.npz, shared_splits.npz, predictions.npz, bootstrap_draws.npz, folds/: data, saved splits, all fitted models and held-out predictions.',
        '- source_metadata/, source_manifest.json, reuse_manifest.json, analysis scripts and artifact_manifest.json: exact reuse provenance, reproducibility and checksums.','']
    (OUT/'REVIEW.md').write_text('\n'.join(rows))
