# Unweighted 40-, 64- and 80-gene panels on lineage-stratified splits

**Lowest mean pooled RMSE: Neural network, 64 genes (0.116968).** Highest mean pooled Spearman: Neural network, 64 genes (0.46175). Lowest mean equal-lineage RMSE: Neural network, 64 genes (0.110514). These are descriptive point estimates; the paired intervals below assess differences conditional on the fitted predictions.

Completed all nine model/panel combinations across the **same 20 saved lineage-stratified 80/20 splits**, excluding the singleton Cervix line. Every split contains **595 training and 149 test cell lines**, with **all 23 retained lineages on both sides** and **zero identity-group overlap**. The cohort contains 744 lines and 733 identity groups. All training is unweighted.

Eight arms contributed 160 newly fitted outer models. The 20 completed unweighted 64-gene NN models were reused after exact input/split checks, artifact-hash validation and full nested native-MLP replays in repeats 1 and 20. The nine arms therefore comprise 180 outer models.

## Mean test scores across 20 splits

| Model | Genes | Pooled Spearman ↑ | Pooled RMSE ↓ | Pooled R² ↑ | Macro-23 RMSE ↓ | Macro-19 RMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Ridge | 40 | 0.44883 | 0.117993 | 0.19508 | 0.111978 | 0.115700 |
| Ridge | 64 | 0.46062 | 0.117005 | 0.20874 | 0.110655 | 0.115007 |
| Ridge | 80 | 0.45092 | 0.117558 | 0.20095 | 0.111562 | 0.115742 |
| Neural network | 40 | 0.45163 | 0.117904 | 0.19676 | 0.111531 | 0.115336 |
| Neural network | 64 | 0.46175 | 0.116968 | 0.20941 | 0.110514 | 0.114937 |
| Neural network | 80 | 0.45325 | 0.117415 | 0.20298 | 0.111317 | 0.115631 |
| XGBoost | 40 | 0.44447 | 0.117990 | 0.19488 | 0.110774 | 0.114556 |
| XGBoost | 64 | 0.44925 | 0.117789 | 0.19770 | 0.110795 | 0.114187 |
| XGBoost | 80 | 0.43327 | 0.118664 | 0.18615 | 0.111724 | 0.115024 |

Scores are calculated separately on each repeat’s test set and then averaged; predictions are not averaged across repeats before scoring. Macro-23 RMSE equally averages all 23 lineage-specific RMSE values. Macro-19 uses the 19 cohort lineages with at least 10 samples as a secondary metric. Very small test subsets make lineage correlations unstable or undefined, so no macro Spearman is used.

![Panel comparison](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_stratified_panel_comparison/panel_comparison.png)

## Paired panel expansion effects

Positive improvement favors the larger panel: higher Spearman or lower RMSE. Intervals use 5,000 paired whole-lineage draws shared across all models and all split-specific test sets. They are conditional on fitted predictions, do not refit models and are not adjusted for multiple comparisons.

| Model | Contrast | Metric | Mean improvement | Favorable splits | Conditional 95% interval |
|---|---|---|---:|---:|---|
| Ridge | 64 vs 40 | pooled_spearman | +0.011787 | 13/20 | [-0.000196, +0.024473] |
| Ridge | 64 vs 40 | pooled_rmse | +0.000988 | 16/20 | [+0.000055, +0.001880] |
| Ridge | 64 vs 40 | macro23_rmse | +0.001323 | 16/20 | [-0.000535, +0.003665] |
| Ridge | 80 vs 40 | pooled_spearman | +0.002088 | 12/20 | [-0.016471, +0.021065] |
| Ridge | 80 vs 40 | pooled_rmse | +0.000435 | 15/20 | [-0.000833, +0.001600] |
| Ridge | 80 vs 40 | macro23_rmse | +0.000415 | 12/20 | [-0.002296, +0.003572] |
| Ridge | 80 vs 64 | pooled_spearman | -0.009700 | 3/20 | [-0.020981, +0.001108] |
| Ridge | 80 vs 64 | pooled_rmse | -0.000553 | 5/20 | [-0.001231, +0.000056] |
| Ridge | 80 vs 64 | macro23_rmse | -0.000908 | 6/20 | [-0.002085, +0.000196] |
| Neural network | 64 vs 40 | pooled_spearman | +0.010122 | 12/20 | [-0.002422, +0.023379] |
| Neural network | 64 vs 40 | pooled_rmse | +0.000936 | 17/20 | [+0.000011, +0.001691] |
| Neural network | 64 vs 40 | macro23_rmse | +0.001017 | 14/20 | [-0.000303, +0.002614] |
| Neural network | 80 vs 40 | pooled_spearman | +0.001623 | 10/20 | [-0.017265, +0.020228] |
| Neural network | 80 vs 40 | pooled_rmse | +0.000488 | 14/20 | [-0.000817, +0.001514] |
| Neural network | 80 vs 40 | macro23_rmse | +0.000215 | 12/20 | [-0.001837, +0.002565] |
| Neural network | 80 vs 64 | pooled_spearman | -0.008499 | 6/20 | [-0.019134, +0.000680] |
| Neural network | 80 vs 64 | pooled_rmse | -0.000448 | 6/20 | [-0.001102, +0.000108] |
| Neural network | 80 vs 64 | macro23_rmse | -0.000803 | 6/20 | [-0.001923, +0.000244] |
| XGBoost | 64 vs 40 | pooled_spearman | +0.004782 | 9/20 | [-0.013361, +0.019601] |
| XGBoost | 64 vs 40 | pooled_rmse | +0.000201 | 12/20 | [-0.000912, +0.001235] |
| XGBoost | 64 vs 40 | macro23_rmse | -0.000021 | 10/20 | [-0.001485, +0.001353] |
| XGBoost | 80 vs 40 | pooled_spearman | -0.011198 | 6/20 | [-0.040023, +0.010082] |
| XGBoost | 80 vs 40 | pooled_rmse | -0.000674 | 9/20 | [-0.002120, +0.000759] |
| XGBoost | 80 vs 40 | macro23_rmse | -0.000950 | 6/20 | [-0.002626, +0.000630] |
| XGBoost | 80 vs 64 | pooled_spearman | -0.015981 | 4/20 | [-0.032139, -0.001942] |
| XGBoost | 80 vs 64 | pooled_rmse | -0.000875 | 6/20 | [-0.001554, -0.000145] |
| XGBoost | 80 vs 64 | macro23_rmse | -0.000929 | 7/20 | [-0.001626, -0.000228] |

## Cross-model comparisons at matched panel sizes

| Genes | Contrast | Metric | Mean improvement | Favorable splits | Conditional 95% interval |
|---|---|---|---:|---:|---|
| 40 | Neural network vs Ridge | pooled_rmse | +0.000090 | 8/20 | [-0.000407, +0.000699] |
| 40 | Neural network vs Ridge | macro23_rmse | +0.000446 | 13/20 | [-0.000333, +0.001291] |
| 40 | XGBoost vs Ridge | pooled_rmse | +0.000003 | 10/20 | [-0.001494, +0.001881] |
| 40 | XGBoost vs Ridge | macro23_rmse | +0.001204 | 15/20 | [-0.001370, +0.003764] |
| 40 | XGBoost vs Neural network | pooled_rmse | -0.000086 | 11/20 | [-0.001456, +0.001497] |
| 40 | XGBoost vs Neural network | macro23_rmse | +0.000758 | 11/20 | [-0.001280, +0.002752] |
| 64 | Neural network vs Ridge | pooled_rmse | +0.000037 | 11/20 | [-0.000257, +0.000361] |
| 64 | Neural network vs Ridge | macro23_rmse | +0.000141 | 12/20 | [-0.000167, +0.000468] |
| 64 | XGBoost vs Ridge | pooled_rmse | -0.000784 | 5/20 | [-0.002352, +0.001139] |
| 64 | XGBoost vs Ridge | macro23_rmse | -0.000140 | 8/20 | [-0.001817, +0.001504] |
| 64 | XGBoost vs Neural network | pooled_rmse | -0.000821 | 5/20 | [-0.002200, +0.000960] |
| 64 | XGBoost vs Neural network | macro23_rmse | -0.000281 | 7/20 | [-0.002031, +0.001346] |
| 80 | Neural network vs Ridge | pooled_rmse | +0.000142 | 11/20 | [-0.000136, +0.000402] |
| 80 | Neural network vs Ridge | macro23_rmse | +0.000246 | 15/20 | [-0.000103, +0.000594] |
| 80 | XGBoost vs Ridge | pooled_rmse | -0.001106 | 5/20 | [-0.002627, +0.001022] |
| 80 | XGBoost vs Ridge | macro23_rmse | -0.000162 | 10/20 | [-0.002121, +0.001807] |
| 80 | XGBoost vs Neural network | pooled_rmse | -0.001249 | 5/20 | [-0.002685, +0.000783] |
| 80 | XGBoost vs Neural network | macro23_rmse | -0.000407 | 10/20 | [-0.002327, +0.001539] |

All 22 planned paired comparisons and all five metrics are saved in paired_comparisons.tsv.

## Lineage consistency of panel expansion

| Model | Contrast | Lineages with lower mean test RMSE |
|---|---|---:|
| Ridge | 64 vs 40 | 16/23 |
| Ridge | 80 vs 40 | 10/23 |
| Ridge | 80 vs 64 | 10/23 |
| Neural network | 64 vs 40 | 14/23 |
| Neural network | 80 vs 40 | 11/23 |
| Neural network | 80 vs 64 | 9/23 |
| XGBoost | 64 vs 40 | 14/23 |
| XGBoost | 80 vs 40 | 9/23 |
| XGBoost | 80 vs 64 | 6/23 |

These are descriptive signs of mean effects across splits, not separately established effects in each lineage. The heatmap labels show the per-repeat test sample count for each lineage; one-point lineages are particularly uncertain.

![Lineage panel effects](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_stratified_panel_comparison/lineage_panel_effects.png)

## Fixed data, splits and model settings

- All models use the same erastin sensitivity target, negative harmonized erastin AUC (higher means more sensitive), and DepMap log2(TPM+1) expression. The existing 40/64/80 panels and their order are unchanged; larger panels contain the smaller panels as prefixes.
- All 20 outer and five-fold inner assignments are byte-identical to the preceding lineage-stratified experiment. Every outer train/test pair contains all 23 lineages. Related identities stay together.
- Ampulla has only two total lines: one outer train and one outer test. Its single training line is retained in every inner training fit and never used for inner validation. Thus each inner training set contains 23 lineages and each inner validation set contains 22. The outer Ampulla test is never used for tuning.
- Ridge: original 17-value alpha grid, train-only StandardScaler for X, target centering and SVD. No sample weights.
- Neural network: original four width/penalty candidates (16/32 tanh units and alpha 10/100), three fixed per-repeat initializations, L-BFGS with max_iter 2000, max_fun 40000 and tolerance 0.001. X/y scalers use training rows only. The verified adapter receives exact unit weights and reproduces native MLPRegressor predictions.
- XGBoost: original 36 candidates: depth 1/2/3 × minimum child weight 5/20 × L2 penalty 1/20 × 100/300/600 rounds. Fixed CPU hist, max_bin 64, learning rate 0.03, uniform row/column sampling 0.8, and the matched per-repeat sampling seeds. No sample weights; target mean/SD use current training rows only.
- Each arm is tuned independently by minimum unweighted equal-mean five-fold inner RMSE, preserving the original grid order for ties. Hyperparameters, panels and seeds are not changed after observing outer-test predictions.

## Validation

- Audited all 180 outer models (160 new and 20 reused). Maximum saved-prediction reconstruction error 1.11e-16.
- Recomputed all 17,100 candidate-fold scores; maximum RMSE discrepancy 8.33e-17. Verified selected grid indices, source hashes and exact panel/input alignment.
- Completed 1,080 outer/inner identity-exclusion checks. Every inner training set retains all 23 lineages.
- Recomputed unweighted objectives for all 180 selected NN members; maximum discrepancy 5.55e-17. Verified all saved NN sample weights equal one.
- Preflight full nested native-MLP replays for the reused NN64 arm in repeats 1/20 had exact zero discrepancy in all candidate inner and selected outer predictions. Independent ridge APIs and XGBoost native/prefix/wrapper checks passed for all three panel sizes in repeats 1/20.
- Selected settings were independently refitted for all nine arms in repeats 1/20, including five inner fits each. Ridge uses a separate sklearn fit, NN uses native MLPRegressor, and XGBoost uses its sklearn wrapper.
- Bootstrap metric calculations were checked against explicitly expanded SciPy samples using the actual split-specific predictions. Reused NN64 predictions exactly match the preceding stratified experiment.
- Convergence warnings: 0; total recorded warnings: 3,780. Categories and per-fit diagnostics are saved.

## Interpretation limits

This design tests new cell lines from lineages represented in training. It does not measure transfer to absent cancer lineages, and its scores should not be pooled with the earlier leave-lineage-out results. Twenty split-specific test sets overlap and reuse this biological cohort; their 20×149 predictions are not independent observations. The whole-lineage bootstrap is conditional on fitted predictions and summarizes variability across the observed lineages, not independent biological replication. Only 724 of 744 distinct cell lines happen to be tested at least once across these random holdouts; coverage is saved by sample. Tiny lineage test sets remain uncertain. Earlier analyses of this same cohort informed panel/model exploration, so no winner is externally validated. The harmonized AUC upstream provenance limitation remains unresolved.

## Files

- summary.tsv: all nine arms, mean scores and split dispersion.
- paired_comparisons.tsv and paired_seed_differences.tsv: planned panel/family contrasts and conditional intervals.
- lineage_paired_effects.tsv and lineage_win_counts.tsv: test-lineage consistency.
- split_lineage_counts.tsv, inner_split_lineage_counts.tsv and sample_test_coverage.tsv: all saved membership audits.
- per_repeat_pooled_metrics.tsv, per_repeat_macro_metrics.tsv and per_lineage_metrics.tsv: individual split/lineage scores.
- selected_parameters.tsv, tuning_scores.tsv, parameter_counts.tsv, validation.json, preflight_validation.json and protocol.json: tuning and audit trail.
- predictions.npz, bootstrap_draws.npz, panel_inputs.npz, shared_splits.npz and folds/: matched data, all 180 models and held-out predictions.
- source_metadata/ and reuse_manifest.json: exact provenance for the 20 reused NN64 models.
- Source scripts, source hashes and a checksummed artifact manifest.
