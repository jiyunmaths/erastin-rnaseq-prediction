#!/usr/bin/env python3
"""Figures and report for assay agreement and lineage-specific gene reliance."""
import json
from pathlib import Path
import numpy as np,pandas as pd
import calculate_importance as calc
OUT=calc.OUT;MODELS=calc.MODELS;DATASETS=['CTRP','PRISM'];LABELS={'ridge':'Ridge','elastic_net':'Elastic Net','neural_network':'Neural network'};COLORS={'ridge':'#137d8b','elastic_net':'#c96913','neural_network':'#8853a3'}
CORE=['AIFM2','SLC7A11','ACSL4','GPX4','ABCC1'];FOCUS=CORE+['SLC3A2','SLC40A1','POR','GCLC']

def rank_string(summary,ds,gene,scope='native_pooled'):
    d=summary[(summary.dataset==ds)&(summary.gene==gene)&(summary.scope==scope)].set_index('model');return ' / '.join(f'{d.loc[m,"rank"]:g}' for m in MODELS)
def count_string(summary,ds,gene,key):
    d=summary[(summary.dataset==ds)&(summary.gene==gene)&(summary.scope=='native_pooled')].set_index('model');return ' / '.join(f'{d.loc[m,key]:g}' for m in MODELS)

def lineage_tables(perlineage,protocol):
    p=perlineage[perlineage.common13].pivot(index=['lineage','gene'],columns=['dataset','model'],values='delta_mse');assert not p.isna().any().any();positive=p>0
    rows=[]
    for (lineage,gene),row in p.iterrows():
        v=dict(lineage=lineage,gene=gene,strict7=lineage in protocol['common_strict_lineages'])
        for ds in DATASETS:
            for m in MODELS:v[ds+'__'+m+'__delta_mse']=row[(ds,m)]
            v[ds+'__all_three_positive']=bool((row.loc[ds]>0).all())
        v['both_assays_all_six_positive']=bool((row>0).all());rows.append(v)
    tab=calc.table('common_lineage_gene_concordance.tsv',rows);result=[]
    for gene,g in tab.groupby('gene'):
        pos=g[g.both_assays_all_six_positive];result.append(dict(gene=gene,CTRP_all_three_positive_lineages=int(g['CTRP__all_three_positive'].sum()),PRISM_all_three_positive_lineages=int(g['PRISM__all_three_positive'].sum()),both_assays_all_six_positive_lineages=len(pos),both_assays_all_six_positive_strict7=int(pos.strict7.sum()),lineages=', '.join(pos.lineage)))
    return calc.table('gene_lineage_concordance_summary.tsv',result)

def render(summary,perlineage,selection,cross,within,stability,protocol,result,validation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    primary=summary[summary.scope=='native_pooled'];sets=[set(primary[(primary.dataset=='PRISM')&(primary.model==m)&(primary['rank']<=10)].gene) for m in MODELS];consensus=set.intersection(*sets);avg=primary[primary.dataset=='PRISM'].groupby('gene')['rank'].mean();genes=CORE+sorted(consensus-set(CORE),key=lambda g:avg.loc[g]);table=lineage_tables(perlineage,protocol)
    matrix=np.column_stack([primary[(primary.dataset==ds)&(primary.model==m)].set_index('gene').loc[genes,'rank'] for ds in DATASETS for m in MODELS]);fig,ax=plt.subplots(figsize=(10.5,6.5),constrained_layout=True);im=ax.imshow(matrix,aspect='auto',cmap='viridis_r',vmin=1,vmax=64);ax.set_yticks(range(len(genes)),genes);ax.set_xticks(range(6),[ds+'\n'+LABELS[m] for ds in DATASETS for m in MODELS],fontsize=9);ax.axvline(2.5,color='white',lw=3)
    for i in range(len(genes)):
        for j in range(6):ax.text(j,i,f'{matrix[i,j]:g}',ha='center',va='center',color='white' if matrix[i,j]>36 else 'black')
    fig.colorbar(im,ax=ax,shrink=.75,label='Predictive rank among 64 genes');ax.set_title('Gene importance changes between CTRP and PRISM\nOriginal five genes plus PRISM genes in every model’s top ten; rank 1 = strongest',fontsize=12);fig.savefig(OUT/'native_rank_comparison.png',dpi=170);fig.savefig(OUT/'native_rank_comparison.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,4.5),constrained_layout=True)
    scopes=['native_pooled','common13_macro','matched_cells_common13_macro'];names=['Full native\ncohorts','Same 13 lineages\nequal weight','Same 290 cells\nequal lineage weight'];ax=axes[0]
    for i,m in enumerate(MODELS):
        d=cross[(cross.model==m)&(cross.measure=='predictive_importance')].set_index('scope');v=d.loc[scopes,'spearman'].to_numpy();xx=np.arange(3)+(i-1)*.23;ax.bar(xx,v,width=.22,color=COLORS[m],label=LABELS[m])
        for x,y in zip(xx,v):ax.text(x,y+.015,f'{y:.2f}',ha='center',fontsize=9)
    ax.set_xticks(range(3),names,fontsize=9);ax.set_ylim(0,1.08);ax.set_title('Agreement across assays\nSpearman of all 64 gene-importance ranks',fontsize=11);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False)
    combos=[('ridge','elastic_net'),('ridge','neural_network'),('elastic_net','neural_network')];ax=axes[1]
    for i,ds in enumerate(DATASETS):
        v=[within[(within.dataset==ds)&(within.scope=='native_pooled')&(within.first_model==a)&(within.reference_model==b)].spearman.iloc[0] for a,b in combos];xx=np.arange(3)+(i-.5)*.30;ax.bar(xx,v,width=.29,label=ds,color=['#137d8b','#c96913'][i])
        for x,y in zip(xx,v):ax.text(x,y+.015,f'{y:.2f}',ha='center',fontsize=9)
    ax.set_xticks(range(3),['Ridge vs\nElastic Net','Ridge vs\nNeural network','Elastic Net vs\nNeural network'],fontsize=9);ax.set_ylim(0,1.08);ax.set_title('Agreement across models within each assay\nSpearman of all 64 gene-importance ranks',fontsize=11);ax.legend(fontsize=8);ax.grid(axis='y',alpha=.18);ax.spines[['top','right']].set_visible(False);fig.suptitle('Cohort controls partly improve assay agreement; model agreement is stronger',fontsize=12);fig.savefig(OUT/'agreement_controls.png',dpi=170);fig.savefig(OUT/'agreement_controls.svg');plt.close(fig)
    plotgenes=CORE+['SLC3A2','SLC40A1','GCLC'];order=perlineage[(perlineage.dataset=='PRISM')&(perlineage.model=='ridge')&(perlineage.gene=='AIFM2')&perlineage.common13].sort_values('cohort_n',ascending=False);lineages=order.lineage.tolist();values=[]
    for ds in DATASETS:
        row=[]
        for m in MODELS:
            v=perlineage[(perlineage.dataset==ds)&(perlineage.model==m)&perlineage.common13].pivot(index='lineage',columns='gene',values='delta_mse').loc[lineages,plotgenes].to_numpy();baseline=summary[(summary.dataset==ds)&(summary.model==m)&(summary.scope=='common13_macro')].baseline_mse.iloc[0];row.append(v/baseline*100)
        values.append(row)
    limit=max(float(np.max(np.abs(v))) for row in values for v in row);fig,axes=plt.subplots(2,3,figsize=(15.5,10.5),constrained_layout=True,sharex=True,sharey=True)
    for i,ds in enumerate(DATASETS):
        for j,m in enumerate(MODELS):
            ax=axes[i,j];im=ax.imshow(values[i][j],aspect='auto',cmap='RdBu',vmin=-limit,vmax=limit);ax.set_title(ds+' — '+LABELS[m],fontsize=11);ax.set_xticks(range(len(plotgenes)),plotgenes,rotation=50,ha='right',fontsize=9);ax.set_yticks(range(13),lineages,fontsize=9)
    fig.colorbar(im,ax=axes,shrink=.78,label='Lineage ΔMSE / model’s common-13 baseline MSE (%)');fig.suptitle('Positive global importance does not imply benefit in every lineage\nSame 13 lineages; blue = replacement worsens error, red = replacement improves error',fontsize=13);fig.savefig(OUT/'lineage_importance_comparison.png',dpi=170);fig.savefig(OUT/'lineage_importance_comparison.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(13,7.8),constrained_layout=True,sharey=True)
    for ax,ds in zip(axes,DATASETS):
        for i,m in enumerate(MODELS):
            d=primary[(primary.dataset==ds)&(primary.model==m)].set_index('gene').loc[genes];yy=np.arange(len(genes))+(i-1)*.23;ax.hlines(yy,d.relative_ci_low,d.relative_ci_high,color=COLORS[m],lw=1.4,alpha=.7);ax.scatter(d.relative_mse_increase_percent,yy,label=LABELS[m],color=COLORS[m],s=25,zorder=3)
        ax.axvline(0,color='#666',linestyle='--',lw=1);ax.set_yticks(range(len(genes)),genes);ax.set_title(ds+' native pooled importance');ax.set_xlabel('Increase in held-out MSE relative to model baseline (%)');ax.grid(axis='x',alpha=.18);ax.spines[['top','right']].set_visible(False)
    axes[0].invert_yaxis();axes[1].legend(fontsize=9,loc='lower right');fig.suptitle('Exact same-lineage training-donor gene replacement\nPoints = mean importance; intervals = conditional pointwise 95% lineage bootstrap',fontsize=12);fig.savefig(OUT/'relative_importance_intervals.png',dpi=170);fig.savefig(OUT/'relative_importance_intervals.svg');plt.close(fig)
    write_report(summary,perlineage,selection,cross,within,stability,protocol,result,validation,table)

def write_report(summary,perlineage,selection,cross,within,stability,protocol,result,validation,concordance):
    primary=summary[summary.scope=='native_pooled'];con=concordance.set_index('gene');rows=['# PRISM versus CTRP: 64-gene model importance and lineage consistency','',
        '**AIFM2, SLC7A11 and ACSL4 are the clearest shared high-ranking predictors across the two datasets.** AIFM2 ranks first in all three PRISM models. GPX4 and ABCC1, which were among the top five in CTRP, are substantially less useful in the 64-gene PRISM models. These are model reliance results, not causal gene effects.','',
        '**Lineage consistency is limited.** In PRISM, AIFM2 has positive held-out predictive importance in 8/13 sufficiently represented common lineages, SLC7A11 in 9/13, and ACSL4 in 7–8/13. No PRISM gene reaches the predefined descriptive threshold of positive importance in at least 11/13 common lineages. Native predictive-rank agreement between assays is low, and only partly improves after matching lineage composition or exact receiver cell lines.','',
        '## What was compared','',
        '- All **120 saved 64-gene models**: 20 ridge, 20 Elastic Net and 20 three-member NN ensembles from each dataset. No model or gene panel was refitted.',
        '- **CTRP:** 744 lines, 23 lineages, 149 held-out lines per repeat. **PRISM:** 386 lines, 20 lineages, 78 held-out lines per repeat. Both use previously saved unweighted grouped lineage-stratified 80/20 splits, with all represented lineages on both sides.',
        '- Identical importance definition in both datasets: replace one gene in a held-out receiver with every same-lineage outer-training donor value, then average the increase in held-out squared error. Positive values mean replacement worsens accuracy. All donor/receiver combinations are enumerated; donor choice uses no test responses.',
        '- Main native summaries weight held-out lines within each repeat, then average repeats. Relative importance divides the mean MSE increase by the model’s own mean baseline MSE. Comparisons of raw coefficients across assays are avoided because response processing, dose intervals, training sizes and regularization differ.',
        '- Controls: equally weight the same **13 lineages** with at least ten cohort lines in both datasets; repeat on the **290 exact same cell-line IDs** held out at least once in both studies, averaging each cell’s valid test appearances before equal lineage weighting. A stricter lineage analysis uses **seven common lineages** with at least five test lines per repeat in both datasets.',
        '', '## Native full-cohort gene rankings','',
        'Ranks are shown in **ridge / Elastic Net / neural network** order. Rank 1 means the greatest positive held-out MSE increase after replacement.','',
        '| Gene | CTRP ranks | PRISM ranks | PRISM positive lineages, out of 13 |','|---|---|---|---|']
    for gene in FOCUS:rows.append(f'| {gene} | {rank_string(summary,"CTRP",gene)} | {rank_string(summary,"PRISM",gene)} | {count_string(summary,"PRISM",gene,"positive_common13")} |')
    rows+=['','SLC3A2, SLC40A1, POR and GCLC rise in the PRISM native rankings. High point ranks alone do not establish reproducible benefit: SLC3A2, SLC40A1 and POR have conditional native importance intervals containing zero in all three PRISM models. AIFM2, ACSL4 and GCLC have positive lower interval bounds in all three; SLC7A11 does in ridge and NN. Intervals are pointwise and unadjusted.','',
        '**GPX4:** ranks 44–48 in PRISM, has negative mean native predictive importance in every model, and enters none of their top tens in any PRISM repeat. This does not imply biological irrelevance; it means these fitted expression models do not gain reliable marginal held-out predictive benefit from this feature under this replacement rule.','',
        '**ABCC1:** falls to ranks 22/46/27, with weak positive or negative mean predictive importance. Its mean coefficient/local-derivative association reverses from higher expression predicting greater sensitivity in CTRP to predicting lower sensitivity in PRISM. Correlation, shrinkage and cohort/assay differences prevent a causal interpretation of that reversal.','',
        f'![Native gene ranks]({OUT}/native_rank_comparison.png)','','## Global importance and uncertainty','',
        'Relative MSE increase is distinct from coefficient magnitude and selection frequency. Replacing a frequently selected feature can still fail to worsen held-out accuracy.','']
    for ds in DATASETS:
        rows += [f'### {ds}: top ten by model','']
        for m in MODELS:rows.append('- **'+LABELS[m]+':** '+', '.join(result['top10_native'][ds][m])+'.')
        rows+=['','Positive lower bounds in conditional pointwise native 95% intervals:','']
        for m in MODELS:rows.append('- **'+LABELS[m]+':** '+', '.join(result['positive_pointwise_native_intervals'][ds][m])+'.')
    rows+=['',f'![Relative importance and intervals]({OUT}/relative_importance_intervals.png)','','## Agreement across models and assays','',
        '| Comparison within assay | CTRP rank correlation | PRISM rank correlation |','|---|---:|---:|']
    for first,ref in [('ridge','elastic_net'),('ridge','neural_network'),('elastic_net','neural_network')]:
        vals=[within[(within.dataset==ds)&(within.scope=='native_pooled')&(within.first_model==first)&(within.reference_model==ref)].spearman.iloc[0] for ds in DATASETS];rows.append(f'| {LABELS[first]} vs {LABELS[ref]} | {vals[0]:.4f} | {vals[1]:.4f} |')
    rows+=['','| Model | CTRP–PRISM native rank correlation | Same 13 lineages, equal weight | Same 290 receiver lines, equal lineage weight |','|---|---:|---:|---:|']
    for m in MODELS:
        d=cross[(cross.model==m)&(cross.measure=='predictive_importance')].set_index('scope');rows.append(f'| {LABELS[m]} | {d.loc["native_pooled","spearman"]:.4f} | {d.loc["common13_macro","spearman"]:.4f} | {d.loc["matched_cells_common13_macro","spearman"]:.4f} |')
    rows+=['','These are correlations across all 64 gene-importance values. Matching the lineage mixture improves agreement, but exact receiver matching still leaves only modest agreement. The matched-cell control uses each dataset’s own valid held-out predictions, trained models and donor distributions; it cannot isolate an assay-specific biological effect. Same numeric split seeds do not imply identical test samples in the two cohorts.','',
        '| Gene | Same-13-lineage CTRP ranks | Same-13-lineage PRISM ranks | Same-290-cell CTRP ranks | Same-290-cell PRISM ranks |','|---|---|---|---|---|']
    for gene in FOCUS:rows.append(f'| {gene} | {rank_string(summary,"CTRP",gene,"common13_macro")} | {rank_string(summary,"PRISM",gene,"common13_macro")} | {rank_string(summary,"CTRP",gene,"matched_cells_common13_macro")} | {rank_string(summary,"PRISM",gene,"matched_cells_common13_macro")} |')
    rows+=['','AIFM2, SLC7A11 and ACSL4 remain top-ten predictors in both assays and all three models under both composition controls. The GPX4 drop persists after matching the exact receiver cell lines. SLC40A1’s point rank is more sensitive to the receiver/cohort weighting choice.','',f'![Agreement controls]({OUT}/agreement_controls.png)','','## Which effects persist across lineages?','',
        'For each lineage, average the held-out MSE replacement effect across the 20 repeats. Counts below refer to positive mean effects, not statistically significant effects in individual lineages. The common 13-lineage set avoids comparing a 19-lineage CTRP denominator with a 13-lineage PRISM denominator.','',
        '| Gene | CTRP positive lineages: R / EN / NN | PRISM positive lineages: R / EN / NN | Positive in both assays and all three models |','|---|---|---|---:|']
    for gene in FOCUS:rows.append(f'| {gene} | {count_string(summary,"CTRP",gene,"positive_common13")} | {count_string(summary,"PRISM",gene,"positive_common13")} | {int(con.loc[gene,"both_assays_all_six_positive_lineages"])}/13 |')
    rows+=['','For the three shared leading genes, the lineages with positive point effects in **all six dataset/model combinations** are:','']
    for gene in ['AIFM2','SLC7A11','ACSL4']:rows.append(f'- **{gene}:** '+con.loc[gene,'lineages']+'.')
    rows+=['','These lists are descriptive intersections. A positive point effect in a three-line test subset is not a lineage-specific validation. Complete counts, ranks and positive-repeat frequencies for all 64 genes and all lineages are saved.','',
        '| Model | CTRP median correlation across common 13 lineage rankings | PRISM median correlation across common 13 | PRISM stricter seven |','|---|---:|---:|---:|']
    for m in MODELS:
        vals=[]
        for ds,unit in [('CTRP','between_common13_lineages__predictive_importance'),('PRISM','between_common13_lineages__predictive_importance'),('PRISM','between_strict7_lineages__predictive_importance')]:vals.append(stability[(stability.dataset==ds)&(stability.model==m)&(stability.unit==unit)]['median'].iloc[0])
        rows.append(f'| {LABELS[m]} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} |')
    rows+=['','Lineage-specific predictive rankings vary substantially in both assays; the stricter seven-lineage subset does not recover uniform rankings. Conversely, sensitivity ranks are identical across lineages by construction for ridge and Elastic Net, and almost identical for the strongly regularized NN. A globally shared coefficient vector or stable derivative is not evidence that the feature improves prediction equally in every lineage.','',f'![Lineage importance]({OUT}/lineage_importance_comparison.png)','','## Elastic Net selection and directions','',
        '| Gene | CTRP selections | PRISM selections | CTRP mean direction | PRISM mean direction |','|---|---:|---:|---|---|']
    selected=selection[selection.model=='elastic_net'].set_index(['dataset','gene'])
    for gene in FOCUS:
        c=selected.loc[('CTRP',gene)];p=selected.loc[('PRISM',gene)];direction=lambda v:'Higher sensitivity' if v>0 else 'Lower sensitivity' if v<0 else 'Zero';rows.append(f'| {gene} | {int(c.selected_fits)}/20 | {int(p.selected_fits)}/20 | {direction(c.mean_coefficient)} | {direction(p.mean_coefficient)} |')
    rows+=['','Always selected in PRISM: **'+', '.join(result['always_selected_EN']['PRISM'])+'**. Always selected in CTRP: '+', '.join(result['always_selected_EN']['CTRP'])+'. Only AIFM2 and ACSL4 are always selected in both. Direction means the model association of higher expression with negative-AUC sensitivity, conditional on the other genes. It is not a causal perturbation direction.','',
        '## Uncertainty, validation and interpretation limits','',
        '- Source model/data hashes, gene order, sample IDs, training/test rows, saved predictions and training-only scalers were verified for both datasets. Identity groups never cross training versus test/donor roles.',
        f'- Recalculated all 120 saved-model importance profiles. Previous CTRP native rankings are reproduced exactly; maximum previous-array/score discrepancy is {validation["CTRP_previous_importance_max_error"]:.3g}.',
        f'- Exact NN hidden-state replacements and linear coefficient replacements were checked against native input replacement across all 64 genes and every lineage, with additional full donor-matrix checks. Maximum native error {validation["native_replacement_error"]:.3g}; finite-difference derivative error {validation["finite_difference_gradient_error"]:.3g}.',
        f'- Verified {validation["train_test_and_donor_identity_checks"]} outer/donor identity exclusions. Elastic Net selection counts/signs reproduce prior fit audits. Independent bootstrap expansion discrepancy {validation["bootstrap_expansion_max_error"]:.3g}.',
        '- Uncertainty uses 5,000 whole-lineage bootstrap draws, paired across genes/models. Original CTRP lineage draws are reused; common-13 comparisons share identical lineage draws in both datasets. Intervals condition on the saved fits and donor distributions, with no refitting, gene/model reselection or multiplicity correction.',
        '- Replacement is marginal within lineage: it can break gene–gene correlations and create atypical combinations. Redundant predictors can divide importance. A weak/negative predictive effect is not evidence against biological involvement.',
        '- Training cohorts, tested dose ranges, response processing and fitted regularization differ between assays. The existing CTRP normalization/provenance limitation remains. Even matching receiver cells does not match model training cohorts or isolate assay biology.',
        '- Repeated holdouts overlap. Tiny lineage test subsets remain uncertain. The cohort has informed prior modeling choices; these are exploratory model explanations, not independent causal or clinical validation.','',
        '## Files','',
        '- gene_comparison.tsv: all 64 genes, both assays and all models, native ranks/effects, controlled ranks, selection and lineage/split consistency.',
        '- gene_importance_summary.tsv, per_repeat_importance.tsv, per_lineage_importance.tsv: full global, split and lineage summaries.',
        '- common_lineage_gene_concordance.tsv, gene_lineage_concordance_summary.tsv, gene_lineage_effect_agreement.tsv: exact positive-effect intersections and matching lineages.',
        '- cross_assay_rank_correlations.tsv, within_assay_model_agreement.tsv, cross_assay_top_gene_overlap.tsv, rank_stability.tsv: global/cohort-controlled/model/lineage agreement.',
        '- coefficient_selection.tsv, coefficient_sensitivity_by_repeat.tsv, donor_reference_sensitivity.tsv: weights, derivatives and donor-reference checks.',
        '- matched_cell_control.tsv, ctrp_importance_arrays.npz, prism_importance_arrays.npz, bootstrap_draws.npz, repeats/: matched receivers and complete numerical arrays.',
        '- protocol.json, source_manifest.json, calculation_validation.json, validation.json, analysis scripts and artifact_manifest.json: methods and reproducibility. Original fitted models remain dependencies at their recorded source paths.','']
    (OUT/'REVIEW.md').write_text('\n'.join(rows))

def main():
    summary=pd.read_csv(OUT/'gene_importance_summary.tsv',sep='\t');per=pd.read_csv(OUT/'per_lineage_importance.tsv',sep='\t');sel=pd.read_csv(OUT/'coefficient_selection.tsv',sep='\t');cross=pd.read_csv(OUT/'cross_assay_rank_correlations.tsv',sep='\t');within=pd.read_csv(OUT/'within_assay_model_agreement.tsv',sep='\t');stable=pd.read_csv(OUT/'rank_stability.tsv',sep='\t');protocol=json.load(open(OUT/'protocol.json'));result=json.load(open(OUT/'summary.json'));validation=json.load(open(OUT/'validation.json'));assert validation['passed'];render(summary,per,sel,cross,within,stable,protocol,result,validation)
    manifest=json.load(open(OUT/'analysis_source_manifest.json'));assert manifest['review_sha256']==calc.sha(OUT/'review_importance.py');manifest['report_sha256']=calc.sha(Path(__file__));calc.dump(OUT/'analysis_source_manifest.json',manifest);print('REPORT_RENDERED',flush=True)
if __name__=='__main__':main()
