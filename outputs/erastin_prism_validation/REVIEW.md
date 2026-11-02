# Frozen-model validation on DepMap PRISM erastin data

The previous models were trained on **CTRP erastin sensitivity**, using DepMap RNA expression. This analysis evaluates those saved models on **official PRISM Repurposing 20Q2 secondary-screen erastin dose-response AUC**, without PRISM-label training, feature selection, recalibration or hyperparameter tuning. It is a cross-assay transfer test, not a new within-PRISM training benchmark.

Among the 64-gene models, the highest point Spearman on novel identity groups is **Elastic Net (64 genes): 0.296**. The primary test contains 43 lines in 42 identity groups; this is a small external test set. Ranking a model first here does not establish a reliable performance difference.

**Interpretation:** the saved models show modest positive ranking transfer. The 64-gene point correlations are about 0.28–0.30 on novel identities and about 0.37 on shared identities held out of training. All 64-gene identity-bootstrap intervals on the small novel set include zero. Paired primary model-difference intervals also include zero, so this test does not show a clear winner among ridge, Elastic Net and NN. Raw PRISM-AUC R² is negative without recalibration; ranking transfer does not imply calibrated absolute AUC prediction.

The five-gene models have higher point Spearman on novel identities (about 0.34–0.36), but lower point Spearman on shared identities (about 0.30–0.31). Paired five-versus-64 intervals include zero in both populations. These results do not establish superiority or equivalence of a reduced panel.

## Data identity and cohort

- Official dataset: [Broad DepMap PRISM Repurposing 20Q2 v1](https://doi.org/10.6084/m9.figshare.20564034.v1). The portal currently serves an automated-verification page, so the official Figshare archive was used. This is a pinned release, not a claim of the latest PRISM data.
- Exact compound mapping: **erastin = DPC-002562 = CTRP:63578 = BRD-A25004090-001-08-4**. The BRD identifier matches the original official PortalCompounds mapping. Related compounds and names containing erastin were not pooled.
- Erastin is from **HTS002**, eight doses spanning approximately 0.00061034–10 µM. These are secondary-screen dose-response AUC values; primary-screen single-dose log-fold change is not used. Historical target annotations are source metadata, not updated mechanistic claims.
- Found 396 erastin curve rows; 390 explicitly pass STR profiling. After matching finite AUC, same-release expression, cancer-cell-line metadata and represented lineages, retained **386 lines, 379 identity groups and 20 lineages**.
- **Novel groups:** 43 lines in 42 groups and 13 lineages, absent from the entire original 744-line CTRP benchmark. All 20 frozen split-specific fits may predict these lines.
- **Shared groups:** 343 lines overlap the benchmark by identity group. Only 334 have at least one saved fit that excludes their entire identity group from training. The 9 never-held-out lines are excluded from scoring.
- Identity groups are reconstructed across all release models using patient, RRID, CCLE and stripped cell-line identifiers, then checked against every original training group. Cervix remains excluded and all retained PRISM lineages occur in training.
- The same DepMap 24Q4 log2(TPM+1) features, Entrez mappings, gene order, source float32 precision and saved training-only scalers are used. No test-data standardization or expression imputation is performed.
- PRISM AUC is kept exactly as reported: range 0.5582–1.5127, including 62 values above 1. No response clipping, sign optimization or arbitrary filtering of flat, insensitive curves is used.

## Primary accuracy: novel identity groups

Outcome is continuous sensitivity (negative PRISM AUC). **Spearman correlation is the primary accuracy measure**; there is no artificial sensitive/resistant classification threshold. Positive correlation means predictions correctly rank greater erastin sensitivity.

| Model | Genes | Test lines | Spearman | Conditional 95% identity-group interval | Pearson | Within-lineage-centered Pearson |
|---|---:|---:|---:|---|---:|---:|
| Ridge | 64 | 43 | 0.2756 | [-0.0493, 0.5604] | 0.1419 | 0.1552 |
| Ridge | 5 | 43 | 0.3411 | [0.0404, 0.5997] | 0.2804 | 0.2190 |
| Elastic Net | 64 | 43 | 0.2961 | [-0.0225, 0.5745] | 0.1964 | 0.1810 |
| Elastic Net | 5 | 43 | 0.3389 | [0.0384, 0.5985] | 0.2807 | 0.2191 |
| Neural network | 64 | 43 | 0.2842 | [-0.0429, 0.5673] | 0.1534 | 0.1609 |
| Neural network | 5 | 43 | 0.3579 | [0.0595, 0.6116] | 0.3012 | 0.2561 |

The 64-gene models are the primary comparison. Five-gene results are exploratory: those genes were chosen using earlier held-out importance from the CTRP cohort. Novel groups were absent from that cohort, but the model and panel investigation as a whole is exploratory.

## Secondary accuracy: shared identities held out of training

For a shared identity, only fits whose 595 training rows contain no member of that identity group are eligible. Eligible held-out predictions are averaged per line. Some lines have fewer eligible fits than others; counts are saved. This is an independent response assay on shared biological models, not an independent set of cell-line identities.

| Model | Genes | Test lines | Spearman | Conditional 95% identity-group interval | Pearson | Within-lineage-centered Pearson |
|---|---:|---:|---:|---|---:|---:|
| Ridge | 64 | 334 | 0.3708 | [0.2701, 0.4609] | 0.3415 | 0.2756 |
| Ridge | 5 | 334 | 0.3026 | [0.1983, 0.3999] | 0.2940 | 0.2530 |
| Elastic Net | 64 | 334 | 0.3712 | [0.2680, 0.4627] | 0.3372 | 0.2765 |
| Elastic Net | 5 | 334 | 0.3034 | [0.1992, 0.4009] | 0.2947 | 0.2539 |
| Neural network | 64 | 334 | 0.3749 | [0.2752, 0.4638] | 0.3462 | 0.2794 |
| Neural network | 5 | 334 | 0.3084 | [0.2054, 0.4049] | 0.2905 | 0.2435 |

![PRISM rank accuracy](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_prism_validation/prism_correlation_comparison.png)

![PRISM prediction scatter](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_prism_validation/prism_prediction_scatter.png)

## Paired primary model differences

| Population | First model vs reference | Spearman improvement | Conditional 95% identity-group interval |
|---|---|---:|---|
| Novel cell-line identity groups | Elastic Net (64 genes) vs Ridge (64 genes) | +0.0205 | [-0.0270, +0.0689] |
| Novel cell-line identity groups | Elastic Net (64 genes) vs Neural network (64 genes) | +0.0119 | [-0.0390, +0.0645] |
| Novel cell-line identity groups | Neural network (64 genes) vs Ridge (64 genes) | +0.0086 | [-0.0155, +0.0406] |
| Shared groups, held out of training | Elastic Net (64 genes) vs Ridge (64 genes) | +0.0003 | [-0.0192, +0.0205] |
| Shared groups, held out of training | Elastic Net (64 genes) vs Neural network (64 genes) | -0.0038 | [-0.0224, +0.0153] |
| Shared groups, held out of training | Neural network (64 genes) vs Ridge (64 genes) | +0.0041 | [-0.0021, +0.0104] |

Intervals use 5,000 paired whole-identity-group bootstrap draws with all six predictions resampled together. Whole-lineage bootstrap intervals are a sensitivity analysis in confidence_intervals.tsv. These intervals are conditional on the saved models and do not include retraining uncertainty or adjust for multiple model choices. Twenty overlapping training splits are not 20 independent validations.

## Agreement between measured assays

- all_exact_CTRP_PRISM_overlap: n=341, measured CTRP-versus-PRISM Spearman **0.3526**, Pearson **0.3006**.
- scored_shared_overlap: n=332, measured CTRP-versus-PRISM Spearman **0.3537**, Pearson **0.3099**.

This describes cross-assay agreement in observed responses; it is not an estimated upper bound on expression-model accuracy. Differences in dose ranges, pooled versus original assay conditions and AUC processing can all affect transfer.

![Measured assay agreement](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_prism_validation/cross_assay_measurement_agreement.png)

## Calibration diagnostics

**Raw AUC error is not interchangeable with the earlier CTRP test error.** The saved model predicts negative harmonized CTRP AUC, whereas PRISM supplies a different reported AUC endpoint, with a different tested dose range and values above 1. The upstream CTRP normalization/clipping recipe remains unresolved. No PRISM calibration was fitted. The following numbers describe uncalibrated cross-assay disagreement, not a directly comparable within-assay regression benchmark.

| Population | Model | Raw AUC RMSE | Raw AUC MAE | Raw AUC R² |
|---|---|---:|---:|---:|
| Novel cell-line identity groups | Ridge (64) | 0.1852 | 0.1061 | -0.2049 |
| Novel cell-line identity groups | Elastic Net (64) | 0.1833 | 0.1056 | -0.1808 |
| Novel cell-line identity groups | Neural network (64) | 0.1848 | 0.1060 | -0.2005 |
| Shared groups, held out of training | Ridge (64) | 0.1559 | 0.1053 | -0.1878 |
| Shared groups, held out of training | Elastic Net (64) | 0.1563 | 0.1054 | -0.1927 |
| Shared groups, held out of training | Neural network (64) | 0.1561 | 0.1054 | -0.1895 |

Within-lineage-centered Pearson is a descriptive association after subtracting each evaluation population’s lineage means from predictions and responses. It is not prediction recalibration and does not use a trained residual model. Per-lineage correlations require at least five test lines and are saved with their sample counts; smaller subsets are descriptive only.

## Validation

- Verified official download sizes/MD5 checksums and source SHA-256 hashes. Reconstructed all original identity groups and exactly matched 47,616 original 64-gene expression entries.
- Loaded and checked all 120 saved outer models (six arms × 20 splits). Every scored prediction is from a model whose training data excludes the test identity group.
- Reconstructed native NN predictions independently from their tanh weights and Elastic Net predictions from coefficients. Reproduced original CTRP held-out predictions for all overlapping eligible lines; no scaler or model was fitted.
- Independently verified bootstrap correlations and raw error metrics against explicitly expanded SciPy samples for both bootstrap units and all analysis populations.
- Every raw erastin record, exclusion reason, screen choice, matched expression value, eligible-model mask and per-line prediction is retained. All source hashes were rechecked after scoring.

## Files

- matched_cohort.tsv and cohort_audit.tsv: complete source/QC/matching decisions and novel/shared identity classification.
- summary.tsv, confidence_intervals.tsv and paired_comparisons.tsv: ensemble accuracy and conditional paired comparisons.
- per_repeat_metrics.tsv, per_lineage_metrics.tsv, cross_assay_measurement_agreement.tsv and lineage_cohort_counts.tsv: repeat, lineage and assay diagnostics.
- matched_prism_inputs.npz, predictions.npz and ensemble_predictions.tsv: matched data, exact eligible models and predictions.
- prism_erastin_all_rows.tsv, erastin_treatment_metadata.tsv, gene_mapping.tsv, official release manifest and README: inspectable source subset and mappings.
- protocol.json, analysis_source_manifest.json, source_manifest.json, validation.json and scripts: analysis design and reproducibility. Original saved models and full raw releases remain dependencies at the source paths recorded in the manifest.
