"""Render predictive, model-sensitivity and lineage-consistency summaries."""
from pathlib import Path
import numpy as np,pandas as pd
OUT=Path(__file__).resolve().parent;MODELS=['ridge','neural_network'];LABELS={'ridge':'Ridge','neural_network':'Neural network'}

def render(summary,lineages,rankstats,cross,sensitivity,result,validation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    primary=summary[summary.scope=='pooled23'];order=[]
    for model in MODELS:
        for gene in primary[primary.model==model].sort_values('rank').head(10).gene:
            if gene not in order:order.append(gene)
    score=primary.pivot(index='gene',columns='model',values='rank');order=sorted(order,key=lambda g:score.loc[g].mean());colors={'ridge':'#137d8b','neural_network':'#8853a3'}
    fig,axes=plt.subplots(1,2,figsize=(11,max(7,.38*len(order)+1.4)),constrained_layout=True,sharey=True);yy=np.arange(len(order))
    for ax,model in zip(axes,MODELS):
        a=primary[primary.model==model].set_index('gene').loc[order];ax.barh(yy,a.relative_mse_increase_percent,color=colors[model],alpha=.8)
        ax.hlines(yy,a.relative_ci_low,a.relative_ci_high,color='#263744',lw=1);ax.scatter(a.relative_mse_increase_percent,yy,color='#263744',s=12,zorder=3);ax.axvline(0,color='#555',lw=.8)
        ax.set_title(LABELS[model]);ax.set_xlabel('Expected MSE increase (% of baseline MSE)');ax.grid(axis='x',alpha=.18);ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(yy,order);axes[0].invert_yaxis();fig.suptitle('Top predictive genes in the 64-gene models\nSame-lineage training-reference replacements; conditional 95% lineage intervals',fontsize=12)
    fig.savefig(OUT/'top_predictive_genes.png',dpi=170);fig.savefig(OUT/'top_predictive_genes.svg');plt.close(fig)
    eligible=lineages[lineages.eligible19];info=eligible.drop_duplicates('lineage').sort_values('cohort_n',ascending=False);labels=info.lineage.tolist()
    fig,axes=plt.subplots(1,2,figsize=(17,max(7,.38*len(order)+1.8)),constrained_layout=True,sharey=True)
    for ax,model in zip(axes,MODELS):
        frame=eligible[eligible.model==model];ranks=frame.pivot(index='gene',columns='lineage',values='rank').loc[order,labels].to_numpy();values=frame.pivot(index='gene',columns='lineage',values='delta_mse').loc[order,labels].to_numpy()
        im=ax.imshow(ranks,aspect='auto',cmap='viridis_r',vmin=1,vmax=64)
        for i in range(len(order)):
            for j in range(19):
                if values[i,j]<=0:ax.text(j,i,'×',ha='center',va='center',fontsize=8,color='white' if ranks[i,j]>32 else '#23323a')
        ax.set_xticks(range(19),[f'{v.lineage} (n={v.test_n})' for v in info.itertuples()],rotation=65,ha='right',fontsize=8);ax.set_title(LABELS[model]);fig.colorbar(im,ax=ax,shrink=.7,label='Rank among all 64 genes (1 = highest)')
    axes[0].set_yticks(range(len(order)),order,fontsize=9);fig.suptitle('Importance varies across represented cancer lineages\nMean effects over 20 shared splits; × = nonpositive held-out predictive benefit',fontsize=12)
    fig.savefig(OUT/'lineage_gene_ranks.png',dpi=170);fig.savefig(OUT/'lineage_gene_ranks.svg');plt.close(fig)
    slope_order=[]
    for model in MODELS:
        for gene in primary[primary.model==model].sort_values('slope_rms_rank').head(10).gene:
            if gene not in slope_order:slope_order.append(gene)
    pivot=primary.pivot(index='gene',columns='model',values='slope_rms_rank');slope_order=sorted(slope_order,key=lambda g:pivot.loc[g].mean());yy=np.arange(len(slope_order))
    fig,axes=plt.subplots(1,2,figsize=(11,max(7,.36*len(slope_order)+1.5)),constrained_layout=True,sharey=True)
    repeat=pd.read_csv(OUT/'per_repeat_importance.tsv',sep='\t');repeat=repeat[repeat.scope=='pooled23']
    for ax,model in zip(axes,MODELS):
        a=primary[primary.model==model].set_index('gene').loc[slope_order];sd=repeat[repeat.model==model].groupby('gene').mean_standardized_slope.std().loc[slope_order]
        ax.barh(yy,a.mean_standardized_slope,color=np.where(a.mean_standardized_slope>=0,'#198396','#c07a3d'));ax.errorbar(a.mean_standardized_slope,yy,xerr=sd,fmt='none',ecolor='#263744',capsize=2)
        ax.axvline(0,color='#555',lw=.8);ax.set_title('Standardized ridge coefficients' if model=='ridge' else 'Mean local neural-network derivatives');ax.set_xlabel('Predicted negative AUC per training SD of expression');ax.grid(axis='x',alpha=.18);ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(yy,slope_order);axes[0].invert_yaxis();fig.suptitle('Model sensitivity and direction, separate from predictive benefit\nPositive slope predicts greater sensitivity; whiskers show split SD',fontsize=12)
    fig.savefig(OUT/'standardized_sensitivity.png',dpi=170);fig.savefig(OUT/'standardized_sensitivity.svg');plt.close(fig)
    write_report(summary,lineages,rankstats,cross,sensitivity,result,validation)

def write_report(summary,lineages,rankstats,cross,sensitivity,result,validation):
    primary=summary[summary.scope=='pooled23'];rows=['# Gene importance in the lineage-stratified 64-gene models','',
        '**Top genes by held-out predictive benefit:** Ridge: '+', '.join(result['top10_predictive']['ridge'][:5])+'. Neural network: '+', '.join(result['top10_predictive']['neural_network'][:5])+'.','',
        'Reused all 20 unweighted 64-gene ridge models and all 20 saved three-member neural-network ensembles. No models were retrained. '
        'The cohort and splits are unchanged: Cervix excluded, 744 lines, 23 represented lineages, 595 training/149 testing per split, and no related identity group crossing train/test. '
        'Ranks below are among the complete 64-gene panel.','',
        '## Definition of predictive importance','',
        'For each held-out line and each gene, replace that gene’s expression with every available **same-lineage training-line value**, holding the other genes fixed. '
        'Average the resulting squared-error increase over training donors, held-out receivers, and the 20 splits. '
        'All receiver/donor pairs are evaluated exactly; no permutation Monte Carlo approximation is used. The neural ensemble prediction is averaged before calculating error. '
        'Positive importance means replacement worsens held-out accuracy; negative importance means replacement improves it on this cohort.','',
        'This is a **training-reference donor-replacement measure**, not ordinary test-set permutation importance. '
        'It allows estimates even when a lineage has just one test line, but does not create extra biological information for those tiny lineages. '
        'Donors use training expression only and never test responses. Ordinary within-test permutation importance is provided separately wherever at least two test lines exist.','',
        'Primary aggregate rankings weight the 149 test receivers equally within each split, then average split-level effects. '
        'Equal-lineage and larger-lineage sensitivity analyses are saved. Cross-lineage consistency is summarized for the 19 lineages with cohort n≥10; a stricter analysis uses the 12 lineages with at least five test lines per split. '
        'All 23 lineages are tabulated, including the four smaller, descriptive lineages.','']
    for model in MODELS:
        rows+=['## '+LABELS[model]+' predictive ranking','',
            '| Rank | Gene | Panel membership | MSE increase (%) | Conditional 95% interval (%) | Positive lineages /19 | Positive splits /20 | Top-ten splits /20 | Seed rank range | Direction |',
            '|---:|---|---|---:|---|---:|---:|---:|---|---|']
        for v in primary[primary.model==model].sort_values('rank').head(15).itertuples():
            rows.append(f'| {v.rank:g} | {v.gene} | {"Added 24" if v.added_24 else "Original 40"} | {v.relative_mse_increase_percent:+.3f} | [{v.relative_ci_low:+.3f}, {v.relative_ci_high:+.3f}] | {v.positive_lineages_19}/19 | {v.positive_repeats}/20 | {v.top10_repeats}/20 | {v.rank_min:g}–{v.rank_max:g} | {"Higher sensitivity" if v.mean_standardized_slope>0 else "Lower sensitivity"} |')
        positive=result['positive_pointwise_intervals'][model];rows+=['','Genes with pointwise conditional intervals entirely above zero: '+(', '.join(positive) if positive else 'none')+'.','']
    rows+=['','MSE increase (%) divides the expected error increase by the unshuffled model’s baseline MSE. Intervals use 5,000 paired whole-lineage draws, conditional on the saved models and donor distributions; they do not refit models and are not adjusted for 64 genes or other comparisons. '
        'Direction is the mean model derivative per training SD of expression. The target is negative erastin AUC: a positive derivative predicts greater sensitivity. It is not a causal gene-perturbation effect.','',
        f'![Top predictive genes]({OUT}/top_predictive_genes.png)','','## Consistency across lineages and splits','',
        '| Model | Importance measure | Unit | Median rank correlation | Interquartile range |','|---|---|---|---:|---|']
    for v in rankstats.itertuples():
        if v.unit=='between_lineages12' and v.measure!='predictive_importance':continue
        rows.append(f'| {LABELS[v.model]} | {v.measure} | {v.unit} | {v.median:.3f} | [{v.q25:.3f}, {v.q75:.3f}] |')
    rows+=['','Correlation is over all 64 gene ranks. Split-level rankings vary with held-out responses, training donor distributions and fitted models; the 20 splits reuse overlapping samples. '
        'Low lineage-to-lineage predictive-rank agreement indicates context-dependent error contributions, even when model coefficients or sensitivity rankings agree. '
        'The primary ranking can therefore be reproducible in aggregate without implying that every lineage uses the same genes for accurate prediction.','',
        '| Comparison | Measure | Rank correlation |','|---|---|---:|']
    for v in cross[cross.scope=='pooled23'].itertuples():rows.append(f'| Ridge vs neural network | {v.measure} | {v.spearman:.3f} |')
    rows+=['',f'![Lineage ranks]({OUT}/lineage_gene_ranks.png)','','## Model weights and local sensitivity','',
        'The ridge model has **one coefficient vector per split, applied to every lineage**. It has no independently fitted lineage-specific coefficient vector. '
        'Its lineage-specific predictive importance differs because expression variation and residual prediction errors differ across held-out lineages. '
        'The neural network has context-dependent derivatives; ensemble gradients are averaged before summarizing.','',
        '| Model | Top ten genes by RMS standardized coefficient/derivative |','|---|---|']
    for model in MODELS:rows.append('| '+LABELS[model]+' | '+', '.join(result['top10_sensitivity'][model])+' |')
    rows+=['','A large coefficient/derivative or RMS prediction change measures model influence. It does not guarantee useful held-out predictive signal. '
        'Per-gene signed coefficients, local gradient summaries, prediction-change ranks and sign stability are saved separately from the error-benefit ranking.','',
        f'![Standardized sensitivity]({OUT}/standardized_sensitivity.png)','','## Donor-reference and lineage-weighting checks','',
        '| Model | Training-reference vs test-permutation rank correlation | Top-ten overlap |','|---|---:|---:|']
    for v in sensitivity.itertuples():rows.append(f'| {LABELS[v.model]} | {v.rank_correlation:.3f} | {v.top10_overlap}/10 |')
    rows+=['','This check uses the same 20 lineages with at least two test lines for both methods. Ordinary test permutations include self replacements and use much smaller donor pools; the two measures need not be identical. '
        'Ampulla, Biliary and Prostate have one test line per split and provide no within-test permutation information; their training-reference results remain descriptive.','',
        '| Model | Cell-weighted all-23 top five | Equal-lineage all-23 top five | Cell-weighted larger-19 top five |','|---|---|---|---|']
    for model in MODELS:
        tops=[', '.join(summary[(summary.model==model)&(summary.scope==scope)].sort_values('rank').head(5).gene) for scope in ['pooled23','macro23','pooled19']];rows.append('| '+LABELS[model]+' | '+' | '.join(tops)+' |')
    rows+=['','## Validation and limits','',
        f'- All {validation["saved_models"]} saved models reproduced their held-out predictions; maximum error {validation["saved_prediction_error"]:.3g}. Input/model source hashes and training scalers were verified.',
        f'- Native complete-input replacement checks covered every gene, lineage and split; maximum error {validation["native_replacement_error"]:.3g}. All training donor identities are disjoint from test receiver identities.',
        f'- Analytical gradients matched independent finite differences; maximum discrepancy {validation["finite_difference_gradient_error"]:.3g}. Exact self-replacement checks passed.',
        f'- Exhaustive native permutations for two- and three-line test subsets matched the exact expected secondary loss; maximum error {validation["exhaustive_small_test_permutation_error"]:.3g}.',
        f'- Selected full receiver/training-donor matrices were verified by native prediction for Lung, Thyroid and Ampulla in splits 1/20; maximum error {validation["full_donor_matrix_error"]:.3g}.',
        f'- Cell-level and lineage-level aggregations matched; maximum error {validation["max_cell_lineage_aggregation_error"]:.3g}. Explicit lineage-bootstrap expansion matched within {validation["bootstrap_expansion_error"]:.3g}.',
        '- Replacement preserves the observed lineage context and one-gene donor distribution but disrupts correlations with the other 63 genes. Atypical expression combinations and redundancy can alter importance. Strongest measured gene correlations are saved.',
        '- High importance of an added gene does not isolate its contribution to the improvement from retraining a 40-gene model as a 64-gene model. Individual importance values are not additive independent contributions.',
        '- Negative importance is not evidence of an opposite biological mechanism; low importance is not evidence of biological irrelevance. Derivatives describe fitted associations, not gene knockout or overexpression outcomes.',
        '- The same cell-line cohort has already supported panel/model exploration. Repeated splits and reused test samples are not independent biological validations. No external screen or causal intervention was evaluated.','',
        '## Files','',
        '- gene_importance_summary.tsv: all 64 genes, both models, four aggregation scopes, conditional intervals, direction and stability.',
        '- per_lineage_importance.tsv, lineage_information.tsv and per_repeat_importance.tsv: lineage/split estimates and information limits.',
        '- ridge_coefficients_by_split.tsv and standardized_sensitivity_summary.tsv: direct coefficients and neural derivatives.',
        '- rank_correlation_summary.tsv, rank_correlations.tsv and cross_model_rank_correlations.tsv: agreement across lineages, splits and models.',
        '- donor_reference_sensitivity.tsv and donor_reference_sensitivity_summary.tsv: matched test-permutation checks.',
        '- strongest_gene_correlations.tsv: descriptive pooled and within-lineage-centered expression correlations.',
        '- importance_arrays.npz, bootstrap_draws.npz and repeats/: exact derived arrays and conditional draws.',
        '- protocol.json, source_manifest.json, calculation_validation.json, validation.json and source scripts: definitions, provenance and numerical checks.','']
    (OUT/'REVIEW.md').write_text('\n'.join(rows))
