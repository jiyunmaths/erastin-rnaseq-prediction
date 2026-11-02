# Exploratory five-gene versus 64-gene ridge and neural-network comparison

**Fixed five-gene panel: SLC7A11, GPX4, ABCC1, ACSL4, AIFM2.** Lowest mean pooled RMSE among the four fitted arms: **Neural network, 64 genes (0.116968)**.

**Selection limitation:** these five genes were chosen after examining held-out importance results from this same cohort and these same 20 splits. Test responses therefore influenced feature selection. Refitting models and tuning hyperparameters inside each training split does not remove that bias. This is a retrospective exploratory comparison, not independent validation of the reduced panel, and its conditional intervals do not adjust for the gene-selection step.

Both panels use the exact saved 20 lineage-stratified grouped 80/20 splits: **595 training and 149 testing lines per repeat**, 744 lines and 733 identity groups after excluding the singleton Cervix line. All 23 retained lineages occur on both sides of every outer split, with zero identity-group overlap. The five-gene panel uses 59 fewer genes (92.2% fewer measured features).

## Mean test scores across 20 splits

| Model | Genes | Pooled Spearman ↑ | Pooled RMSE ↓ | Pooled R² ↑ | Macro-23 RMSE ↓ | Macro-19 RMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Ridge | 5 | 0.40979 | 0.120231 | 0.16443 | 0.112458 | 0.117511 |
| Ridge | 64 | 0.46062 | 0.117005 | 0.20874 | 0.110655 | 0.115007 |
| Neural network | 5 | 0.40971 | 0.120052 | 0.16692 | 0.112192 | 0.117439 |
| Neural network | 64 | 0.46175 | 0.116968 | 0.20941 | 0.110514 | 0.114937 |

Scores are calculated separately within each repeat’s held-out test set and then averaged. Test observations are not stacked as independent samples. Macro-23 RMSE equally averages all 23 lineage-specific test RMSEs; macro-19 restricts the average to the 19 cohort lineages with at least 10 total samples. No macro Spearman is calculated because some lineage test sets contain only one line.

![Panel comparison](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_five_gene_comparison/panel_comparison.png)

## Five-gene changes relative to the matched 64-gene model

| Model | Pooled RMSE change (%) | Macro-23 RMSE change (%) | Splits with lower pooled RMSE | Lineages with lower mean RMSE |
|---|---:|---:|---:|---:|
| Ridge | +2.757% | +1.629% | 4/20 | 12/23 |
| Neural network | +2.637% | +1.519% | 4/20 | 10/23 |

Positive percentage changes in the table above mean higher error for the five-gene panel. In the paired-effect table below, positive improvement instead means the five-gene model is better: Spearman(5)−Spearman(64), or RMSE(64)−RMSE(5).

| Model | Metric | Mean improvement | Favorable splits | Conditional 95% interval |
|---|---|---:|---:|---|
| Ridge | pooled_spearman | -0.050826 | 4/20 | [-0.105022, +0.001025] |
| Ridge | pooled_rmse | -0.003226 | 4/20 | [-0.006937, +0.000520] |
| Ridge | pooled_r2 | -0.044316 | 4/20 | [-0.098572, +0.007011] |
| Ridge | macro23_rmse | -0.001803 | 9/20 | [-0.005618, +0.002078] |
| Ridge | macro19_rmse | -0.002505 | 6/20 | [-0.006552, +0.001602] |
| Neural network | pooled_spearman | -0.052042 | 4/20 | [-0.102128, -0.001224] |
| Neural network | pooled_rmse | -0.003084 | 4/20 | [-0.006484, +0.000295] |
| Neural network | pooled_r2 | -0.042487 | 4/20 | [-0.091688, +0.003382] |
| Neural network | macro23_rmse | -0.001678 | 8/20 | [-0.005372, +0.002082] |
| Neural network | macro19_rmse | -0.002502 | 6/20 | [-0.006282, +0.001370] |

Intervals use 5,000 paired whole-lineage bootstrap draws shared across models and all 20 split-specific test sets. They are conditional on the fitted models and fixed, previously selected gene panel. Models and feature selection are not repeated in the bootstrap. There is no adjustment for multiple comparisons or post-selection optimism. **An interval containing zero does not establish equivalence or noninferiority; no equivalence margin was specified.**

![Paired differences](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_five_gene_comparison/paired_differences.png)

## Lineage-specific performance

| Lineage | Test lines per split | Ridge RMSE reduction: 5 vs 64 | NN RMSE reduction: 5 vs 64 |
|---|---:|---:|---:|
| Lung | 24 | +0.000584 | -0.000288 |
| Lymphoid | 19 | -0.007597 | -0.006443 |
| Esophagus/Stomach | 11 | -0.001099 | -0.000787 |
| CNS/Brain | 9 | +0.002878 | +0.002227 |
| Bowel | 9 | -0.011453 | -0.011065 |
| Myeloid | 8 | -0.013306 | -0.010398 |
| Skin | 8 | -0.017772 | -0.017080 |
| Pancreas | 8 | +0.001302 | +0.000820 |
| Ovary/Fallopian Tube | 7 | +0.009971 | +0.009597 |
| Breast | 7 | -0.004692 | -0.004870 |
| Head and Neck | 6 | -0.012277 | -0.011035 |
| Uterus | 5 | +0.006265 | +0.004257 |
| Liver | 4 | -0.018698 | -0.018154 |
| Kidney | 4 | +0.008078 | +0.009804 |
| Bladder/Urinary Tract | 4 | +0.008110 | +0.006683 |
| Peripheral Nervous System | 3 | +0.001572 | +0.001461 |
| Soft Tissue | 3 | +0.001218 | -0.001752 |
| Bone | 3 | -0.010212 | -0.010813 |
| Thyroid | 2 | +0.009540 | +0.010306 |
| Pleura | 2 | -0.012720 | -0.011864 |
| Biliary Tract | 1 | -0.004462 | -0.001184 |
| Prostate | 1 | +0.019474 | +0.020839 |
| Ampulla of Vater | 1 | +0.003827 | +0.001137 |

Positive reductions favor the five-gene panel. These are mean effects across overlapping random holdouts and are descriptive, especially for one- to three-line test subsets.

![Lineage comparison](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_five_gene_comparison/lineage_comparison.png)

## Prediction agreement

| Family | Mean Pearson correlation of 5- and 64-gene predictions | Mean Spearman correlation | Mean prediction-to-prediction RMSE |
|---|---:|---:|---:|
| Ridge | 0.7611 | 0.7317 | 0.039247 |
| Neural network | 0.7564 | 0.7264 | 0.039298 |

Prediction agreement quantifies similarity between fitted outputs; it does not establish equal accuracy or independent validation.

## Fitting and validation

- The five genes were fixed in the user-specified order before these fits. Their columns exactly match the verified 64-gene expression matrix. No per-split gene reselection or additional gene search is performed.
- The five-gene models were fitted from scratch using only those five features. Both panels were separately tuned using the same five inner validation folds and the same original grids. All training is unweighted.
- Ridge retains the original 17-alpha grid, training-only StandardScaler and target centering. NN retains 16/32 tanh units × alpha 10/100, L-BFGS, max_iter 2000, max_fun 40000, tolerance 0.001 and the same three initialization seeds per repeat. X/y scalers use current training rows only.
- Every inner training set retains all 23 lineages. Ampulla has one outer-training line, so that line stays in every inner training fit and never serves as inner validation; inner validation contains 22 lineages. The outer Ampulla test remains untouched.
- Audited 80 outer models: 40 newly fitted five-gene models plus 40 verified 64-gene models. Both reused prediction arrays and scores exactly match the prior comparison.
- Reconstructed all 4,200 candidate-fold scores; maximum discrepancy 8.33e-17. Saved-prediction maximum discrepancy 1.11e-16.
- Verified 480 outer/inner identity exclusions, common split arrays, training-only scalers and source hashes. All saved NN weights equal one.
- Recomputed unweighted objectives for all 120 selected NN members; maximum error 5.55e-17.
- Preflight independently replayed every inner candidate and selected outer prediction for five-gene ridge/NN models in repeats 1/20, and selected reused 64-gene fits. Independent selected-setting checks cover all four arms in repeats 1/20 and all corresponding inner folds.
- Convergence warnings: 0; total recorded warnings: 2,520. Full categories and per-fit diagnostics are saved.
- Bootstrap calculations were independently checked against explicitly expanded SciPy samples using the actual test predictions.

## Interpretation limits

This analysis answers the requested retrospective comparison on these saved splits. It cannot provide an unbiased confirmatory estimate for a five-gene panel selected using those earlier test responses. A fresh cohort or prospectively reserved test set is needed to validate the fixed panel independently. Twenty repeats share the same biological samples and overlapping training/test sets, so they are not 20 independent biological validations. Only 724/744 lines happen to occur in at least one test set across the random holdouts; coverage is saved. This design tests new lines from represented cancer lineages, not absent-lineage transfer. The harmonized erastin AUC upstream provenance limitation remains unresolved.

## Files

- summary.tsv and paired_comparisons.tsv: aggregate scores and conditional paired differences.
- per_repeat_pooled_metrics.tsv, per_repeat_macro_metrics.tsv, paired_seed_differences.tsv and per_lineage_metrics.tsv: split-level and lineage results.
- lineage_paired_effects.tsv, lineage_win_counts.tsv and prediction_agreement.tsv: consistency and prediction similarity.
- selected_parameters.tsv, tuning_scores.tsv, parameter_counts.tsv, protocol.json, preflight_validation.json and validation.json: tuning and audit trail.
- panel_inputs.npz, shared_splits.npz, predictions.npz, bootstrap_draws.npz and folds/: matched data, all models and held-out predictions.
- source_metadata/, source_manifest.json and reuse_manifest.json: original 64-gene artifacts and exact reuse provenance.
- Checksummed artifact manifest and all analysis scripts.
