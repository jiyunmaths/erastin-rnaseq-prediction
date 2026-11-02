# Corrected leave-lineage-out comparison

The corrected leave-lineage-out results do not support consistent overall superiority of panel-focused BulkFormer over the 40-gene expression baseline.

Reused the saved embeddings for 745 cancer cell lines in 734 identity groups. Each of 24 lineages was completely excluded from supervised training and tuning when it was evaluated. Original outer holdouts, labels, ridge penalty grid and train-only standardization were preserved.

The direct corrected rerun uses the original seed (20261031). The sensitivity analysis uses the same 20 fixed seeds as the preceding grouped-CV study. Each seed changes only the five inner folds; the 24 outer lineage tests stay fixed.

## Pooled predictions across all 24 held-out lineages

Means across the 20 fixed inner-fold seeds; 745 held-out predictions per seed. These are means of metrics, not metrics of an averaged-prediction ensemble.

| Model | Spearman ↑ | RMSE ↓ | R² ↑ |
|---|---:|---:|---:|
| 40-gene expression | 0.4238 | 0.12075 | 0.1737 |
| Panel-focused BulkFormer | 0.3694 | 0.12380 | 0.1315 |
| All-gene BulkFormer | 0.3105 | 0.12692 | 0.0870 |

## Equal weight for each of 19 larger lineages

Mean within-lineage scores, then averaged over the 20 seeds. The established n ≥ 10 threshold excludes five small lineages from this summary; all 24 remain in pooled metrics. Pooled correlation can reflect between-lineage differences, so it is not a substitute for within-lineage ranking.

| Model | Mean within-lineage Spearman ↑ | Mean within-lineage RMSE ↓ | Mean within-lineage R² ↑ |
|---|---:|---:|---:|
| 40-gene expression | 0.2301 | 0.12125 | -0.0877 |
| Panel-focused BulkFormer | 0.2334 | 0.12206 | -0.1130 |
| All-gene BulkFormer | 0.1882 | 0.12358 | -0.1407 |

## Panel-focused BulkFormer versus expression

Positive changes favor panel-focused BulkFormer. Intervals are paired, conditional 95% whole-lineage bootstrap intervals (5,000 draws).

| Summary | Metric | Mean improvement | 95% interval | Favorable inner-fold seeds |
|---|---|---:|---:|---:|
| Pooled | Spearman increase | -0.05438 | [-0.17633, +0.04793] | 0/20 |
| Pooled | RMSE reduction | -0.00305 | [-0.00928, +0.00246] | 0/20 |
| Pooled | R² increase | -0.04227 | [-0.13624, +0.03435] | 0/20 |
| Equal-weight lineages | Spearman increase | +0.00328 | [-0.08261, +0.09760] | 16/20 |
| Equal-weight lineages | RMSE reduction | -0.00081 | [-0.00716, +0.00515] | 1/20 |
| Equal-weight lineages | R² increase | -0.02524 | [-0.16055, +0.09452] | 0/20 |

Among the 19 lineages with at least 10 samples, using the mean effect across 20 seeds:

- Panel-focused BulkFormer has higher Spearman in 10/19 lineages.
- Panel-focused BulkFormer has lower RMSE in 9/19 lineages.
- Panel-focused BulkFormer has higher R² in 9/19 lineages.

![Per-lineage effects](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_lineage_corrected/lineage_effects.png)

![Inner-seed sensitivity](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_lineage_corrected/inner_seed_effects.png)

## Direct original-seed correction

| Model | Legacy Spearman | Corrected Spearman | Legacy RMSE | Corrected RMSE |
|---|---:|---:|---:|---:|
| 40-gene expression | 0.4254 | 0.4235 | 0.12073 | 0.12088 |
| Panel-focused BulkFormer | 0.3624 | 0.3695 | 0.12420 | 0.12407 |
| All-gene BulkFormer | 0.2977 | 0.3173 | 0.12789 | 0.12612 |

The original predictions were reproduced before the corrected evaluation. The prior bug shuffled a view of a pandas index and corrupted lineage-count lookup. The corrected splitter copies group IDs and verifies tracked counts against actual fold membership. Related-identity groups remained intact in both versions. No original artifacts were overwritten.

## Interpretation and checks

- These are unseen lineages for the supervised ridge head. BulkFormer pretraining overlap is unresolved; no pretraining-disjoint claim is made.
- Panel-focused embeddings pool the selected 40 gene tokens after processing the full transcriptome. This is not an assay requiring only 40 measured genes.
- The 20 inner-fold seeds reuse the same data and outer tests. They quantify tuning-split sensitivity, not 20 independent experiments.
- The bootstrap resamples complete held-out lineages, keeping models and seeds paired. Models are not retrained in each draw, and training data overlap across outer holdouts. Intervals are conditional descriptive uncertainty, not a complete estimate of future-lineage risk.
- Results retain the archived CTRP AUC and DepMap 24Q4 limitations. No external screen or clinical cohort was added.
- All saved outer models were independently reconstructed. Selected penalties match recorded inner-fold minima. Independent sklearn Pipeline/GridSearchCV fits were checked for the Lung and Thyroid holdouts across all three primary models.
- See `protocol.json`, `validation.json`, `independent_review.json`, `shared_splits.npz`, prediction/model files and `artifact_manifest.json` for the audit trail.
