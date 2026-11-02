# Repeated erastin sensitivity comparison

Completed 20 fixed seeds on the same 745 cancer cell lines (734 identity groups), reusing saved embeddings. Each repeat used five outer folds and four inner folds, shared across models; scaling and ridge selection used training data only. Target: negative reported erastin AUC, so larger values mean greater sensitivity.

| Model | Mean Spearman ± SD | Mean RMSE ± SD | Mean R² ± SD |
|---|---:|---:|---:|
| 40-gene expression | 0.4435 ± 0.0085 | 0.1194 ± 0.0006 | 0.1922 ± 0.0076 |
| Panel-focused BulkFormer | 0.4598 ± 0.0098 | 0.1180 ± 0.0007 | 0.2114 ± 0.0096 |
| All-gene BulkFormer | 0.4280 ± 0.0107 | 0.1199 ± 0.0007 | 0.1850 ± 0.0096 |
| Lineage-only control | 0.3397 ± 0.0079 | 0.1247 ± 0.0004 | 0.1181 ± 0.0056 |

SD describes variation under repartitioning, not the standard error of independent datasets.

| Panel-focused improvement over expression | Mean change | Favorable seeds | Conditional 95% group-bootstrap interval |
|---|---:|---:|---:|
| Spearman increase | 0.01632 | 19/20 | [-0.01945, 0.05068] |
| RMSE reduction | 0.00143 | 19/20 | [-0.00098, 0.00381] |
| R² increase | 0.01916 | 19/20 | [-0.01334, 0.05114] |

The mean RMSE reduction is 1.19%. All three paired intervals include zero.

The win counts measure split stability in this cohort. The 20 repeats reuse the same cells and are not independent studies. The 5,000 bootstrap draws resample identity groups and retain all models and repeats together; intervals condition on fitted out-of-fold predictions and do not include bootstrap refitting.

## Split-generation correction

Validation uncovered a bug in the earlier split generator: shuffling a NumPy view also shuffled the pandas index used for lineage counts. Identity groups stayed intact, but lineage balancing was corrupted. The corrected generator copies IDs before shuffling and verifies actual lineage counts for every outer and inner split. Across the original seed and the 20 follow-up seeds, maximum outer-fold lineage deviation from the ideal count fell from up to 9 cells to 0.8 cells.

The first repeated run is preserved in `../erastin_repeated_cv_legacy_splits/`. Its results were visible before the corrected rerun. The correction was motivated by the reproduced indexing bug; seeds, features, models, ridge grid, metrics and bootstrap design were unchanged. `legacy_vs_corrected.tsv` reports both versions. Original input data, embeddings and single-seed results were preserved.

## Scope and verification

This supports a within-cohort stability assessment, not an external validation claim. Panel-focused BulkFormer pools the 40 selected gene tokens after processing the full transcriptome; it is not a model requiring only 40 measured genes. Checkpoint pretraining overlap and original AUC processing details remain unresolved.

The prior pooled leave-lineage-out analysis favored raw expression. Its inner folds used the legacy generator, and that analysis was not rerun here; these repeated grouped splits do not establish improvement on unseen lineages.

All 400 saved outer models reproduce their out-of-fold predictions, all selected penalties match the minimum inner-fold RMSE, and all source hashes remain unchanged. An independent sklearn Pipeline/GridSearchCV refit reproduces the selected penalties and predictions for all three primary models on repeat 1, outer fold 0. See `independent_review.json` and `validation.json`.

Detailed tables, shared splits, predictions, fitted models, frozen protocol and audit files are included in `../erastin_repeated_cv_results.zip`.
