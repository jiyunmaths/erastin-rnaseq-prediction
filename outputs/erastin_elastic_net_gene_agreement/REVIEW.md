# Elastic Net gene selection and agreement with ridge and NN

**13 of 64 genes are selected in all 20 Elastic Net outer fits**, with consistent coefficient signs. All five previously highlighted genes—SLC7A11, GPX4, ABCC1, ACSL4 and AIFM2—are in this set. Selection is checked on the saved outer-training models. The models are not refitted or reselected for this analysis.

## Always selected genes

| Gene | Selected fits | EN direction | Mean EN coefficient | EN predictive rank | Ridge rank | NN rank |
|---|---:|---|---:|---:|---:|---:|
| SLC7A11 | 20/20 | Lower sensitivity | -0.019322 | 1 | 1 | 1 |
| ACSL4 | 20/20 | Higher sensitivity | +0.011969 | 2 | 4 | 2 |
| GPX4 | 20/20 | Lower sensitivity | -0.009180 | 3 | 2 | 3 |
| AIFM2 | 20/20 | Lower sensitivity | -0.009434 | 4 | 5 | 4 |
| ABCC1 | 20/20 | Higher sensitivity | +0.008540 | 5 | 3 | 5 |
| FADS1 | 20/20 | Higher sensitivity | +0.007315 | 6 | 8 | 11 |
| ACSL3 | 20/20 | Lower sensitivity | -0.008705 | 7 | 7 | 7 |
| LAMP2 | 20/20 | Lower sensitivity | -0.008150 | 8 | 13 | 13 |
| CYB5R1 | 20/20 | Lower sensitivity | -0.008061 | 10 | 14 | 14 |
| CTSD | 20/20 | Lower sensitivity | -0.007402 | 14 | 18 | 16 |
| FADS2 | 20/20 | Higher sensitivity | +0.003922 | 17 | 16 | 17 |
| GCLC | 20/20 | Lower sensitivity | -0.006195 | 18 | 19 | 19 |
| ATF4 | 20/20 | Higher sensitivity | +0.003886 | 22 | 27 | 30 |

Coefficients are original target units per training-set expression SD. The target is negative harmonized erastin AUC, so higher predictions mean greater sensitivity. Direction is a conditional model association, not a causal biological effect. All always-selected genes have one sign across all 20 fits.

**Selected in 19/20 fits:** GSR, LRP8, GOT1, DHFR.
**Selected in 18/20 fits:** SLC3A2, LPCAT3, PROM2, SPR.

![Selection and rank agreement](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_elastic_net_gene_agreement/selection_and_rank_agreement.png)

## Matched predictive importance

The same importance definition is used for all three models: replace one gene in a held-out line with each same-lineage training-donor value, average the resulting increase in squared error, cell-weight the 149 held-out lines per split, then average over 20 splits. This is exact expected training-reference replacement. Donors do not cross lineages or use test responses. It is not fully conditional permutation because other gene values remain fixed.

Ridge coefficients and NN derivatives describe output sensitivity. They are separate from the replacement effect on held-out prediction error. The table above and the correlations below use matched replacement MSE importance for all models.

| Comparison | Predictive-rank Spearman, all 64 | Top-five overlap | Top-ten overlap |
|---|---:|---:|---:|
| Elastic Net vs Ridge | 0.9165 | 5/5 | 8/10 |
| Elastic Net vs Neural network | 0.9155 | 5/5 | 7/10 |

Global predictive ranks by model:

- **Elastic Net top ten:** SLC7A11, ACSL4, GPX4, AIFM2, ABCC1, FADS1, ACSL3, LAMP2, GOT1, CYB5R1.
- **Ridge top ten:** SLC7A11, GPX4, ABCC1, ACSL4, AIFM2, GOT1, ACSL3, FADS1, LPCAT3, LRP8.
- **Neural network top ten:** SLC7A11, ACSL4, GPX4, AIFM2, ABCC1, GOT1, ACSL3, LRP8, GSR, DHFR.

### Five previously highlighted genes

| Gene | EN rank | Ridge rank | NN rank | EN MSE increase (%) | EN positive splits | EN top-ten splits | Positive lineages: EN / ridge / NN |
|---|---:|---:|---:|---:|---:|---:|---|
| SLC7A11 | 1 | 1 | 1 | +5.171% | 20/20 | 20/20 | 14/19 / 13/19 / 13/19 |
| GPX4 | 3 | 2 | 3 | +1.390% | 19/20 | 16/20 | 13/19 / 14/19 / 14/19 |
| ABCC1 | 5 | 3 | 5 | +1.318% | 18/20 | 17/20 | 14/19 / 15/19 / 14/19 |
| ACSL4 | 2 | 4 | 2 | +1.842% | 19/20 | 17/20 | 12/19 / 12/19 / 12/19 |
| AIFM2 | 4 | 5 | 4 | +1.382% | 18/20 | 18/20 | 13/19 / 14/19 / 14/19 |

MSE increases are relative to each model’s own baseline held-out MSE. Positive importance means replacement worsens accuracy; negative means replacement improves accuracy under this perturbation. The ranking is distinct from the coefficient magnitude ranking.

![Matched predictive importance](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_elastic_net_gene_agreement/matched_predictive_importance.png)

### Pointwise uncertainty

Genes with a positive lower bound in the conditional pointwise 95% interval:

- **Elastic Net:** SLC7A11, ACSL4, GPX4, AIFM2, ABCC1, GOT1.
- **Ridge:** SLC7A11, GPX4, ABCC1, ACSL4, AIFM2, GOT1, LRP8.
- **Neural network:** SLC7A11, ACSL4, GPX4, AIFM2, ABCC1, GOT1, LRP8.

These intervals reuse exactly the existing 5,000 paired whole-lineage bootstrap draws. Fits, panels and donor distributions stay fixed. They do not adjust for 64-gene comparisons, model choice, prior gene selection or repeated use of this cohort. A positive interval is descriptive evidence of model reliance under this replacement rule, not causal evidence or confirmatory significance.

## Coefficients, local derivatives and lineages

| Comparison | RMS standardized sensitivity rank correlation | RMS prediction-influence rank correlation |
|---|---:|---:|
| Elastic Net vs Ridge | 0.9619 | 0.9642 |
| Elastic Net vs Neural network | 0.9594 | 0.9565 |

Standardized sensitivity is RMS coefficient/local derivative per training SD. Prediction influence is RMS prediction change under the same donor replacement. Both measure how outputs move; neither alone measures whether that movement improves held-out accuracy.

| Model | Median predictive-rank correlation across splits | Across 19 lineages | Across stricter 12 lineages |
|---|---:|---:|---:|
| Elastic Net | 0.3406 | 0.0217 | 0.0462 |
| Ridge | 0.3690 | 0.0360 | 0.0558 |
| Neural network | 0.3641 | 0.0313 | 0.0543 |

A linear Elastic Net uses one coefficient vector across every lineage in a split. Therefore stable selection and direction do not establish equal lineage-specific predictive benefit. The 19-lineage analysis excludes lineages with fewer than 10 cohort lines; the stricter 12-lineage analysis requires at least five test lines per split. All 23 lineages remain in the saved tables.

![Core lineage importance](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_elastic_net_gene_agreement/core_lineage_importance.png)

## Validation and limitations

- Reused the same 744 lines, 733 identity groups and 20 lineage-stratified 595/149 splits; every retained lineage appears in both outer sets and Cervix is excluded. No refitting or panel reduction was performed.
- Verified exact feature, sample, target and test-mask identity with the earlier ridge/NN importance dataset. All 40 reused model files and prediction arrays match their original importance sources.
- Reconstructed all 64 Elastic Net coefficients for all 20 fits. Selection counts, signs and original-y coefficients match the prior Elastic Net comparison.
- Verified 480 outer/donor group exclusions. Donor-moment calculations agree with all 29,440 gene-by-lineage-by-repeat exhaustive donor-pair checks.
- Independently replaced input values and called native ElasticNet prediction for selected full donor matrices in repeats 1 and 20. Finite differences verify all 64 standardized coefficient slopes in every split.
- Reconstructed reused ridge/NN ranks, mean importance and bootstrap arrays. Maximum mean-importance discrepancy 4.97e-16; independent bootstrap expansion error 4.34e-19.
- Correlated genes may share or exchange coefficients, and replacements can break gene correlations. A gene can always be selected yet provide weak or negative marginal held-out importance. An unselected gene can still matter biologically.
- Repeats overlap and do not represent independent biological replications. The cohort informed earlier gene/model choices; a fresh cohort is required to validate a proposed compact panel.

## Files

- gene_agreement.tsv: all 64 genes, Elastic Net selection/coefficients and directly matched three-model ranks, importance, intervals and directions.
- elastic_net_selection.tsv, gene_importance_summary.tsv, per_repeat_importance.tsv, per_lineage_importance.tsv: full stability and importance results.
- cross_model_rank_correlations.tsv, top_gene_overlap.tsv, rank_stability.tsv: global, split and lineage agreement.
- importance_arrays.npz and bootstrap_draws.npz: all aggregated arrays and exact paired bootstrap values.
- protocol.json, source_manifest.json, validation.json, scripts and artifact_manifest.json: reproducible methods and checksums.
