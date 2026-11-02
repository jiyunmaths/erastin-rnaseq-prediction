"""Plots and a self-contained report for frozen-model PRISM transfer."""
import json
from pathlib import Path
import numpy as np,pandas as pd
OUT=Path(__file__).resolve().parent
FAMILIES=['ridge','elastic_net','neural_network'];LABELS={'ridge':'Ridge','elastic_net':'Elastic Net','neural_network':'Neural network'};COLORS={'ridge':'#137d8b','elastic_net':'#c96913','neural_network':'#8853a3'}
SCOPES=['novel_identity_groups','shared_groups_held_out'];TITLES={'novel_identity_groups':'Novel cell-line identity groups','shared_groups_held_out':'Shared groups, held out of training','combined_held_out':'All eligible groups (secondary)'}

def pretty(arm):
    f,p=arm.rsplit('__',1);return LABELS[f]+' ('+p+' genes)'

def render(summary,ci,paired,cohort,predictions,diagnostics,validation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(13,8),constrained_layout=True)
    for i,scope in enumerate(SCOPES):
        mask=cohort.novel_identity_group.to_numpy() if i==0 else ~cohort.novel_identity_group.to_numpy()&cohort.scored.to_numpy()
        for j,f in enumerate(FAMILIES):
            ax=axes[i,j];arm=f+'__64';a=summary[(summary.scope==scope)&(summary.model==arm)]
            if a.empty:ax.set_axis_off();continue
            v=a.iloc[0];ax.scatter(predictions[arm][mask],cohort.prism_sensitivity.to_numpy()[mask],s=26,alpha=.8,color=COLORS[f],edgecolors='none');ax.set_title(f'{LABELS[f]}, 64 genes\n{TITLES[scope]}\nn = {int(v.n)}; Spearman = {v.spearman:.3f}',fontsize=10);ax.set_xlabel('Predicted sensitivity (−CTRP AUC)');ax.set_ylabel('Measured sensitivity (−PRISM AUC)');ax.grid(alpha=.15);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Frozen CTRP-trained models evaluated on PRISM erastin\nNo PRISM-response fitting or calibration; higher values mean more sensitivity',fontsize=13);fig.savefig(OUT/'prism_prediction_scatter.png',dpi=170);fig.savefig(OUT/'prism_prediction_scatter.svg');plt.close(fig)
    arms=[f+'__'+str(p) for f in FAMILIES for p in [64,5]];fig,axes=plt.subplots(1,2,figsize=(12,5.7),constrained_layout=True,sharey=True)
    for ax,scope in zip(axes,SCOPES):
        for i,arm in enumerate(arms):
            d=ci[(ci.scope==scope)&(ci.model==arm)&(ci.metric=='spearman')&(ci.unit=='identity_group')]
            if d.empty:continue
            v=d.iloc[0];f,p=arm.rsplit('__',1);color=COLORS[f] if p=='64' else '#aab0b5';ax.hlines(i,v.conditional_ci_low,v.conditional_ci_high,color=color,lw=2);ax.scatter(v.point,i,s=45,color=color,zorder=3)
        ax.axvline(0,color='#777',linestyle='--',lw=1);ax.set_title(TITLES[scope],fontsize=11);ax.set_yticks(range(6),[pretty(a) for a in arms]);ax.set_xlabel('Spearman with measured PRISM sensitivity');ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    axes[0].invert_yaxis();fig.suptitle('PRISM transfer accuracy: 64-gene primary and five-gene exploratory arms\nConditional 95% paired identity-group bootstrap intervals',fontsize=12);fig.savefig(OUT/'prism_correlation_comparison.png',dpi=170);fig.savefig(OUT/'prism_correlation_comparison.svg');plt.close(fig)
    mask=cohort.original_benchmark_auc.notna();a=diagnostics[0];fig,ax=plt.subplots(figsize=(6.5,5.5),constrained_layout=True);ax.scatter(-cohort.loc[mask,'original_benchmark_auc'],cohort.loc[mask,'prism_sensitivity'],s=22,alpha=.7,color='#526779',edgecolors='none');ax.set_xlabel('Measured CTRP sensitivity (−harmonized AUC)');ax.set_ylabel('Measured PRISM sensitivity (−reported AUC)');ax.set_title('Observed agreement between assays\n'+f'n = {a["n"]}; Spearman = {a["spearman"]:.3f}; Pearson = {a["pearson"]:.3f}',fontsize=12);ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False);fig.savefig(OUT/'cross_assay_measurement_agreement.png',dpi=170);fig.savefig(OUT/'cross_assay_measurement_agreement.svg');plt.close(fig)
    write_report(summary,ci,paired,cohort,diagnostics,validation)

def write_report(summary,ci,paired,cohort,diagnostics,validation):
    info=json.load(open(OUT/'cohort_summary.json'));novel=summary[(summary.scope=='novel_identity_groups')&summary.model.str.endswith('__64')];best=novel.loc[novel.spearman.idxmax()] if len(novel) else None
    rows=['# Frozen-model validation on DepMap PRISM erastin data','',
        'The previous models were trained on **CTRP erastin sensitivity**, using DepMap RNA expression. This analysis evaluates those saved models on **official PRISM Repurposing 20Q2 secondary-screen erastin dose-response AUC**, without PRISM-label training, feature selection, recalibration or hyperparameter tuning. It is a cross-assay transfer test, not a new within-PRISM training benchmark.','']
    if best is not None:rows.append(f'Among the 64-gene models, the highest point Spearman on novel identity groups is **{pretty(best.model)}: {best.spearman:.3f}**. The primary test contains {info["novel_lines"]} lines in {info["novel_groups"]} identity groups; this is a small external test set. Ranking a model first here does not establish a reliable performance difference.')
    rows+=['','**Interpretation:** the saved models show modest positive ranking transfer. The 64-gene point correlations are about 0.28–0.30 on novel identities and about 0.37 on shared identities held out of training. All 64-gene identity-bootstrap intervals on the small novel set include zero. Paired primary model-difference intervals also include zero, so this test does not show a clear winner among ridge, Elastic Net and NN. Raw PRISM-AUC R² is negative without recalibration; ranking transfer does not imply calibrated absolute AUC prediction.','',
        'The five-gene models have higher point Spearman on novel identities (about 0.34–0.36), but lower point Spearman on shared identities (about 0.30–0.31). Paired five-versus-64 intervals include zero in both populations. These results do not establish superiority or equivalence of a reduced panel.','',
        '## Data identity and cohort','',
        '- Official dataset: [Broad DepMap PRISM Repurposing 20Q2 v1](https://doi.org/10.6084/m9.figshare.20564034.v1). The portal currently serves an automated-verification page, so the official Figshare archive was used. This is a pinned release, not a claim of the latest PRISM data.',
        '- Exact compound mapping: **erastin = DPC-002562 = CTRP:63578 = BRD-A25004090-001-08-4**. The BRD identifier matches the original official PortalCompounds mapping. Related compounds and names containing erastin were not pooled.',
        '- Erastin is from **HTS002**, eight doses spanning approximately 0.00061034–10 µM. These are secondary-screen dose-response AUC values; primary-screen single-dose log-fold change is not used. Historical target annotations are source metadata, not updated mechanistic claims.',
        f'- Found {info["raw_erastin_rows"]} erastin curve rows; {info["STR_passed_rows"]} explicitly pass STR profiling. After matching finite AUC, same-release expression, cancer-cell-line metadata and represented lineages, retained **{info["matched_lines"]} lines, {info["matched_identity_groups"]} identity groups and {info["lineages"]} lineages**.',
        f'- **Novel groups:** {info["novel_lines"]} lines in {info["novel_groups"]} groups and {info["novel_lineages"]} lineages, absent from the entire original 744-line CTRP benchmark. All 20 frozen split-specific fits may predict these lines.',
        f'- **Shared groups:** {info["shared_lines"]} lines overlap the benchmark by identity group. Only {info["shared_scored_lines"]} have at least one saved fit that excludes their entire identity group from training. The {info["never_heldout_shared_lines"]} never-held-out lines are excluded from scoring.',
        '- Identity groups are reconstructed across all release models using patient, RRID, CCLE and stripped cell-line identifiers, then checked against every original training group. Cervix remains excluded and all retained PRISM lineages occur in training.',
        '- The same DepMap 24Q4 log2(TPM+1) features, Entrez mappings, gene order, source float32 precision and saved training-only scalers are used. No test-data standardization or expression imputation is performed.',
        f'- PRISM AUC is kept exactly as reported: range {info["auc_min"]:.4f}–{info["auc_max"]:.4f}, including {info["auc_greater_than_one"]} values above 1. No response clipping, sign optimization or arbitrary filtering of flat, insensitive curves is used.','',
        '## Primary accuracy: novel identity groups','',
        'Outcome is continuous sensitivity (negative PRISM AUC). **Spearman correlation is the primary accuracy measure**; there is no artificial sensitive/resistant classification threshold. Positive correlation means predictions correctly rank greater erastin sensitivity.','',
        '| Model | Genes | Test lines | Spearman | Conditional 95% identity-group interval | Pearson | Within-lineage-centered Pearson |','|---|---:|---:|---:|---|---:|---:|']
    for f in FAMILIES:
        for p in [64,5]:
            arm=f+'__'+str(p);d=summary[(summary.scope=='novel_identity_groups')&(summary.model==arm)]
            if d.empty:continue
            v=d.iloc[0];c=ci[(ci.scope=='novel_identity_groups')&(ci.model==arm)&(ci.metric=='spearman')&(ci.unit=='identity_group')].iloc[0];rows.append(f'| {LABELS[f]} | {p} | {int(v.n)} | {v.spearman:.4f} | [{c.conditional_ci_low:.4f}, {c.conditional_ci_high:.4f}] | {v.pearson:.4f} | {v.within_lineage_centered_pearson:.4f} |')
    rows+=['','The 64-gene models are the primary comparison. Five-gene results are exploratory: those genes were chosen using earlier held-out importance from the CTRP cohort. Novel groups were absent from that cohort, but the model and panel investigation as a whole is exploratory.','',
        '## Secondary accuracy: shared identities held out of training','',
        'For a shared identity, only fits whose 595 training rows contain no member of that identity group are eligible. Eligible held-out predictions are averaged per line. Some lines have fewer eligible fits than others; counts are saved. This is an independent response assay on shared biological models, not an independent set of cell-line identities.','',
        '| Model | Genes | Test lines | Spearman | Conditional 95% identity-group interval | Pearson | Within-lineage-centered Pearson |','|---|---:|---:|---:|---|---:|---:|']
    for f in FAMILIES:
        for p in [64,5]:
            arm=f+'__'+str(p);d=summary[(summary.scope=='shared_groups_held_out')&(summary.model==arm)]
            if d.empty:continue
            v=d.iloc[0];c=ci[(ci.scope=='shared_groups_held_out')&(ci.model==arm)&(ci.metric=='spearman')&(ci.unit=='identity_group')].iloc[0];rows.append(f'| {LABELS[f]} | {p} | {int(v.n)} | {v.spearman:.4f} | [{c.conditional_ci_low:.4f}, {c.conditional_ci_high:.4f}] | {v.pearson:.4f} | {v.within_lineage_centered_pearson:.4f} |')
    rows+=['',f'![PRISM rank accuracy]({OUT}/prism_correlation_comparison.png)','',f'![PRISM prediction scatter]({OUT}/prism_prediction_scatter.png)','','## Paired primary model differences','',
        '| Population | First model vs reference | Spearman improvement | Conditional 95% identity-group interval |','|---|---|---:|---|']
    for scope in SCOPES:
        d=paired[(paired.scope==scope)&(paired.unit=='identity_group')&(paired.metric=='spearman')&paired.first_model.str.endswith('__64')&paired.reference_model.str.endswith('__64')]
        for v in d.itertuples():rows.append(f'| {TITLES[scope]} | {pretty(v.first_model)} vs {pretty(v.reference_model)} | {v.improvement:+.4f} | [{v.conditional_ci_low:+.4f}, {v.conditional_ci_high:+.4f}] |')
    rows+=['','Intervals use 5,000 paired whole-identity-group bootstrap draws with all six predictions resampled together. Whole-lineage bootstrap intervals are a sensitivity analysis in confidence_intervals.tsv. These intervals are conditional on the saved models and do not include retraining uncertainty or adjust for multiple model choices. Twenty overlapping training splits are not 20 independent validations.','',
        '## Agreement between measured assays','']
    for v in diagnostics:rows.append(f'- {v["scope"]}: n={v["n"]}, measured CTRP-versus-PRISM Spearman **{v["spearman"]:.4f}**, Pearson **{v["pearson"]:.4f}**.')
    rows+=['','This describes cross-assay agreement in observed responses; it is not an estimated upper bound on expression-model accuracy. Differences in dose ranges, pooled versus original assay conditions and AUC processing can all affect transfer.','',f'![Measured assay agreement]({OUT}/cross_assay_measurement_agreement.png)','','## Calibration diagnostics','',
        '**Raw AUC error is not interchangeable with the earlier CTRP test error.** The saved model predicts negative harmonized CTRP AUC, whereas PRISM supplies a different reported AUC endpoint, with a different tested dose range and values above 1. The upstream CTRP normalization/clipping recipe remains unresolved. No PRISM calibration was fitted. The following numbers describe uncalibrated cross-assay disagreement, not a directly comparable within-assay regression benchmark.','',
        '| Population | Model | Raw AUC RMSE | Raw AUC MAE | Raw AUC R² |','|---|---|---:|---:|---:|']
    for scope in SCOPES:
        for f in FAMILIES:
            d=summary[(summary.scope==scope)&(summary.model==f+'__64')]
            if d.empty:continue
            v=d.iloc[0];rows.append(f'| {TITLES[scope]} | {LABELS[f]} (64) | {v.uncalibrated_auc_rmse:.4f} | {v.uncalibrated_auc_mae:.4f} | {v.uncalibrated_auc_r2:.4f} |')
    rows+=['','Within-lineage-centered Pearson is a descriptive association after subtracting each evaluation population’s lineage means from predictions and responses. It is not prediction recalibration and does not use a trained residual model. Per-lineage correlations require at least five test lines and are saved with their sample counts; smaller subsets are descriptive only.','',
        '## Validation','',
        f'- Verified official download sizes/MD5 checksums and source SHA-256 hashes. Reconstructed all original identity groups and exactly matched {info["gene_expression_values_exactly_verified"]:,} original 64-gene expression entries.',
        f'- Loaded and checked all {validation["models_checked"]} saved outer models (six arms × 20 splits). Every scored prediction is from a model whose training data excludes the test identity group.',
        '- Reconstructed native NN predictions independently from their tanh weights and Elastic Net predictions from coefficients. Reproduced original CTRP held-out predictions for all overlapping eligible lines; no scaler or model was fitted.',
        '- Independently verified bootstrap correlations and raw error metrics against explicitly expanded SciPy samples for both bootstrap units and all analysis populations.',
        '- Every raw erastin record, exclusion reason, screen choice, matched expression value, eligible-model mask and per-line prediction is retained. All source hashes were rechecked after scoring.',
        '', '## Files','',
        '- matched_cohort.tsv and cohort_audit.tsv: complete source/QC/matching decisions and novel/shared identity classification.',
        '- summary.tsv, confidence_intervals.tsv and paired_comparisons.tsv: ensemble accuracy and conditional paired comparisons.',
        '- per_repeat_metrics.tsv, per_lineage_metrics.tsv, cross_assay_measurement_agreement.tsv and lineage_cohort_counts.tsv: repeat, lineage and assay diagnostics.',
        '- matched_prism_inputs.npz, predictions.npz and ensemble_predictions.tsv: matched data, exact eligible models and predictions.',
        '- prism_erastin_all_rows.tsv, erastin_treatment_metadata.tsv, gene_mapping.tsv, official release manifest and README: inspectable source subset and mappings.',
        '- protocol.json, analysis_source_manifest.json, source_manifest.json, validation.json and scripts: analysis design and reproducibility. Original saved models and full raw releases remain dependencies at the source paths recorded in the manifest.','']
    (OUT/'REVIEW.md').write_text('\n'.join(rows))
