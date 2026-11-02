"""Report selection stability separately from matched held-out predictive benefit."""
from pathlib import Path
import numpy as np,pandas as pd
OUT=Path(__file__).resolve().parent
MODELS=['elastic_net','ridge','neural_network'];LABELS={'elastic_net':'Elastic Net','ridge':'Ridge','neural_network':'Neural network'}
CORE=['SLC7A11','GPX4','ABCC1','ACSL4','AIFM2'];COLORS=['#c96913','#137d8b','#8853a3']

def render(combined,summary,perlineage,cross,result,validation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    d=combined[combined.selected_fits>=18].sort_values(['selected_fits','elastic_net__rank'],ascending=[False,True]);genes=d.gene.tolist();n=len(d)
    fig,axes=plt.subplots(1,2,figsize=(11,9),constrained_layout=True,gridspec_kw={'width_ratios':[1,1.7]});ax=axes[0];ax.barh(np.arange(n),d.selection_frequency*100,color=['#c96913' if v==20 else '#f0a95e' for v in d.selected_fits]);ax.set_yticks(range(n),genes);ax.invert_yaxis();ax.set_xlim(0,112);ax.set_xticks([0,25,50,75,100]);ax.set_xlabel('Outer fits selecting gene (%)');ax.set_title('Elastic Net selection stability',fontsize=12);ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    for i,v in enumerate(d.selected_fits):ax.text(d.selection_frequency.iloc[i]*100+1,i,f'{v}/20',va='center',fontsize=8)
    vals=np.column_stack([d[m+'__rank'].to_numpy() for m in MODELS]);ax=axes[1];im=ax.imshow(vals,aspect='auto',cmap='viridis_r',vmin=1,vmax=64);ax.set_xticks(range(3),[LABELS[m] for m in MODELS]);ax.set_yticks(range(n),genes);ax.set_title('Matched held-out predictive rank\n1 = greatest MSE increase after replacement',fontsize=12)
    for i in range(n):
        for j in range(3):ax.text(j,i,f'{vals[i,j]:g}',ha='center',va='center',color='white' if vals[i,j]>37 else 'black',fontsize=10)
    fig.colorbar(im,ax=ax,shrink=.7,label='Rank among all 64 genes');fig.suptitle('Frequently selected Elastic Net genes and agreement across models\nSame 20 lineage-stratified splits; gene selection and predictive importance are distinct',fontsize=13);fig.savefig(OUT/'selection_and_rank_agreement.png',dpi=170);fig.savefig(OUT/'selection_and_rank_agreement.svg');plt.close(fig)
    primary=summary[summary.scope=='pooled23'];top=primary[primary.model=='elastic_net'].sort_values('rank').head(12).gene.tolist();fig,ax=plt.subplots(figsize=(11,7.8),constrained_layout=True)
    for i,(m,c) in enumerate(zip(MODELS,COLORS)):
        dd=primary[primary.model==m].set_index('gene').loc[top];ys=np.arange(len(top))+(i-1)*.22;ax.hlines(ys,dd.conditional_ci_low*1e4,dd.conditional_ci_high*1e4,color=c,alpha=.65,lw=1.5);ax.scatter(dd.delta_mse*1e4,ys,label=LABELS[m],s=24,color=c,zorder=3)
    ax.set_yticks(range(len(top)),top);ax.invert_yaxis();ax.axvline(0,color='#555',linestyle='--',lw=1);ax.set_xlabel('Increase in held-out MSE after same-lineage training-donor replacement (×10⁻⁴)');ax.set_title('Top 12 Elastic Net genes by matched held-out predictive importance\nPoints: mean effect; intervals: conditional pointwise 95% lineage bootstrap',fontsize=12);ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False);ax.legend(loc='lower right');fig.savefig(OUT/'matched_predictive_importance.png',dpi=170);fig.savefig(OUT/'matched_predictive_importance.svg');plt.close(fig)
    info=perlineage[(perlineage.model=='elastic_net')&(perlineage.gene=='SLC7A11')&(perlineage.cohort_n>=10)].sort_values('cohort_n',ascending=False);lineages=info.lineage.tolist();values=[]
    for m in MODELS:values.append(perlineage[(perlineage.model==m)&perlineage.lineage.isin(lineages)&perlineage.gene.isin(CORE)].pivot(index='lineage',columns='gene',values='delta_mse').loc[lineages,CORE].to_numpy()*1e4)
    limit=max(float(np.max(np.abs(v))) for v in values);fig,axes=plt.subplots(1,3,figsize=(14,8),constrained_layout=True,sharey=True)
    for ax,m,v in zip(axes,MODELS,values):
        im=ax.imshow(v,aspect='auto',cmap='RdBu',vmin=-limit,vmax=limit);ax.set_xticks(range(5),CORE,rotation=45,ha='right');ax.set_yticks(range(len(lineages)),[f'{v.lineage} (test n={v.test_n})' for v in info.itertuples()],fontsize=9);ax.set_title(LABELS[m])
    fig.colorbar(im,ax=axes,shrink=.8,label='Mean held-out MSE increase (×10⁻⁴)');fig.suptitle('Core-gene predictive importance varies across lineages\n19 lineages with cohort n ≥10; blue means replacement worsens prediction',fontsize=13);fig.savefig(OUT/'core_lineage_importance.png',dpi=170);fig.savefig(OUT/'core_lineage_importance.svg');plt.close(fig)
    write_report(combined,summary,cross,result,validation)

def write_report(combined,summary,cross,result,validation):
    d=combined.set_index('gene');always=combined[combined.selected_fits==20].sort_values('elastic_net__rank');primary=summary[summary.scope=='pooled23'];rows=['# Elastic Net gene selection and agreement with ridge and NN','',
        f'**{len(always)} of 64 genes are selected in all 20 Elastic Net outer fits**, with consistent coefficient signs. All five previously highlighted genes—SLC7A11, GPX4, ABCC1, ACSL4 and AIFM2—are in this set. '+
        'Selection is checked on the saved outer-training models. The models are not refitted or reselected for this analysis.','',
        '## Always selected genes','',
        '| Gene | Selected fits | EN direction | Mean EN coefficient | EN predictive rank | Ridge rank | NN rank |','|---|---:|---|---:|---:|---:|---:|']
    for v in always.itertuples():
        a=d.loc[v.gene];direction='Lower sensitivity' if a.mean_coefficient<0 else 'Higher sensitivity';rows.append(f'| {v.gene} | {int(a.selected_fits)}/20 | {direction} | {a.mean_coefficient:+.6f} | {a["elastic_net__rank"]:g} | {a["ridge__rank"]:g} | {a["neural_network__rank"]:g} |')
    rows+=['','Coefficients are original target units per training-set expression SD. The target is negative harmonized erastin AUC, so higher predictions mean greater sensitivity. Direction is a conditional model association, not a causal biological effect. All always-selected genes have one sign across all 20 fits.','',
        '**Selected in 19/20 fits:** '+', '.join(result['selected_19_of_20'])+'.',
        '**Selected in 18/20 fits:** '+', '.join(result['selected_18_of_20'])+'.','',
        f'![Selection and rank agreement]({OUT}/selection_and_rank_agreement.png)','','## Matched predictive importance','',
        'The same importance definition is used for all three models: replace one gene in a held-out line with each same-lineage training-donor value, average the resulting increase in squared error, cell-weight the 149 held-out lines per split, then average over 20 splits. This is exact expected training-reference replacement. Donors do not cross lineages or use test responses. It is not fully conditional permutation because other gene values remain fixed.','',
        'Ridge coefficients and NN derivatives describe output sensitivity. They are separate from the replacement effect on held-out prediction error. The table above and the correlations below use matched replacement MSE importance for all models.','',
        '| Comparison | Predictive-rank Spearman, all 64 | Top-five overlap | Top-ten overlap |','|---|---:|---:|---:|']
    for ref in ['ridge','neural_network']:
        rho=cross[(cross.first_model=='elastic_net')&(cross.reference_model==ref)&(cross.scope=='pooled23')&(cross.measure=='predictive_importance')].spearman.iloc[0];over={v['top_k']:v['overlap'] for v in result['top_overlap'] if v['first_model']=='elastic_net' and v['reference_model']==ref};rows.append(f'| Elastic Net vs {LABELS[ref]} | {rho:.4f} | {over[5]}/5 | {over[10]}/10 |')
    rows+=['','Global predictive ranks by model:','']
    for m in MODELS:rows.append(f'- **{LABELS[m]} top ten:** '+', '.join(result['top10_predictive'][m])+'.')
    rows+=['','### Five previously highlighted genes','',
        '| Gene | EN rank | Ridge rank | NN rank | EN MSE increase (%) | EN positive splits | EN top-ten splits | Positive lineages: EN / ridge / NN |','|---|---:|---:|---:|---:|---:|---:|---|']
    for gene in CORE:
        v=d.loc[gene];counts=' / '.join(str(int(v[m+'__positive_lineages_19']))+'/19' for m in MODELS);rows.append(f'| {gene} | {v["elastic_net__rank"]:g} | {v["ridge__rank"]:g} | {v["neural_network__rank"]:g} | {v["elastic_net__relative_mse_increase_percent"]:+.3f}% | {int(v["elastic_net__positive_repeats"])}/20 | {int(v["elastic_net__top10_repeats"])}/20 | {counts} |')
    rows+=['','MSE increases are relative to each model’s own baseline held-out MSE. Positive importance means replacement worsens accuracy; negative means replacement improves accuracy under this perturbation. The ranking is distinct from the coefficient magnitude ranking.','',f'![Matched predictive importance]({OUT}/matched_predictive_importance.png)','','### Pointwise uncertainty','',
        'Genes with a positive lower bound in the conditional pointwise 95% interval:','']
    for m in MODELS:rows.append(f'- **{LABELS[m]}:** '+', '.join(result['positive_pointwise_intervals'][m])+'.')
    rows+=['','These intervals reuse exactly the existing 5,000 paired whole-lineage bootstrap draws. Fits, panels and donor distributions stay fixed. They do not adjust for 64-gene comparisons, model choice, prior gene selection or repeated use of this cohort. A positive interval is descriptive evidence of model reliance under this replacement rule, not causal evidence or confirmatory significance.','',
        '## Coefficients, local derivatives and lineages','',
        '| Comparison | RMS standardized sensitivity rank correlation | RMS prediction-influence rank correlation |','|---|---:|---:|']
    for ref in ['ridge','neural_network']:
        c=cross[(cross.first_model=='elastic_net')&(cross.reference_model==ref)&(cross.scope=='pooled23')].set_index('measure');rows.append(f'| Elastic Net vs {LABELS[ref]} | {c.loc["standardized_sensitivity","spearman"]:.4f} | {c.loc["prediction_influence","spearman"]:.4f} |')
    rows+=['','Standardized sensitivity is RMS coefficient/local derivative per training SD. Prediction influence is RMS prediction change under the same donor replacement. Both measure how outputs move; neither alone measures whether that movement improves held-out accuracy.','',
        '| Model | Median predictive-rank correlation across splits | Across 19 lineages | Across stricter 12 lineages |','|---|---:|---:|---:|']
    for m in MODELS:
        v={row['unit']:row['median'] for row in result['stability'] if row['model']==m};rows.append(f'| {LABELS[m]} | {v["between_splits"]:.4f} | {v["between_lineages19"]:.4f} | {v["between_lineages12"]:.4f} |')
    rows+=['','A linear Elastic Net uses one coefficient vector across every lineage in a split. Therefore stable selection and direction do not establish equal lineage-specific predictive benefit. The 19-lineage analysis excludes lineages with fewer than 10 cohort lines; the stricter 12-lineage analysis requires at least five test lines per split. All 23 lineages remain in the saved tables.','',f'![Core lineage importance]({OUT}/core_lineage_importance.png)','','## Validation and limitations','',
        '- Reused the same 744 lines, 733 identity groups and 20 lineage-stratified 595/149 splits; every retained lineage appears in both outer sets and Cervix is excluded. No refitting or panel reduction was performed.',
        '- Verified exact feature, sample, target and test-mask identity with the earlier ridge/NN importance dataset. All 40 reused model files and prediction arrays match their original importance sources.',
        '- Reconstructed all 64 Elastic Net coefficients for all 20 fits. Selection counts, signs and original-y coefficients match the prior Elastic Net comparison.',
        f'- Verified {validation["train_test_identity_exclusion_checks"]} outer/donor group exclusions. Donor-moment calculations agree with all {validation["all_gene_donor_pair_moment_checks"]:,} gene-by-lineage-by-repeat exhaustive donor-pair checks.',
        '- Independently replaced input values and called native ElasticNet prediction for selected full donor matrices in repeats 1 and 20. Finite differences verify all 64 standardized coefficient slopes in every split.',
        f'- Reconstructed reused ridge/NN ranks, mean importance and bootstrap arrays. Maximum mean-importance discrepancy {validation["reused_importance_score_max_error"]:.3g}; independent bootstrap expansion error {validation["bootstrap_expansion_max_error"]:.3g}.',
        '- Correlated genes may share or exchange coefficients, and replacements can break gene correlations. A gene can always be selected yet provide weak or negative marginal held-out importance. An unselected gene can still matter biologically.',
        '- Repeats overlap and do not represent independent biological replications. The cohort informed earlier gene/model choices; a fresh cohort is required to validate a proposed compact panel.',
        '', '## Files','',
        '- gene_agreement.tsv: all 64 genes, Elastic Net selection/coefficients and directly matched three-model ranks, importance, intervals and directions.',
        '- elastic_net_selection.tsv, gene_importance_summary.tsv, per_repeat_importance.tsv, per_lineage_importance.tsv: full stability and importance results.',
        '- cross_model_rank_correlations.tsv, top_gene_overlap.tsv, rank_stability.tsv: global, split and lineage agreement.',
        '- importance_arrays.npz and bootstrap_draws.npz: all aggregated arrays and exact paired bootstrap values.',
        '- protocol.json, source_manifest.json, validation.json, scripts and artifact_manifest.json: reproducible methods and checksums.','']
    (OUT/'REVIEW.md').write_text('\n'.join(rows))
