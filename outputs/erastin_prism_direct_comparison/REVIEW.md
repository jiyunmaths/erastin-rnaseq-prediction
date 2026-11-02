# Direct PRISM erastin training: five versus 64 genes

**All 120 models were newly trained on PRISM erastin AUC.** Lowest mean pooled test RMSE among the six expression arms: **Ridge (5)**, 0.137092. The training-mean control has mean RMSE 0.144030. This is an internal repeated-holdout benchmark; it is separate from the previous frozen CTRP-to-PRISM transfer test.

**The models learn a modest PRISM response signal.** The best mean RMSE is 4.82% lower than the training-mean control; all six expression arms improve on that control in 19/20 splits and their conditional RMSE-reduction intervals stay above zero. The largest mean Spearman is 0.3616 for Neural network (64). Mean test R² across the six arms ranges from 0.0450 to 0.0728.

Five-gene panels have lower mean pooled and equal-lineage RMSE than their matched 64-gene counterparts, while 64-gene panels have slightly higher mean Spearman. The five-versus-64 paired intervals include zero, so the panel comparison remains uncertain. Within 64 genes, NN has a small RMSE advantage over ridge and Elastic Net under conditional, unadjusted intervals; five-gene ridge still has the lowest mean RMSE among all six arms.

The fixed five-gene panel is **SLC7A11, GPX4, ABCC1, ACSL4, AIFM2**. Both panels are separately tuned inside each outer training set, using the same inner folds. No PRISM-driven gene reselection or lineage training weights are used.

## Test scores

| Model | Genes | Mean Spearman ↑ | Mean RMSE ↓ | Mean R² ↑ | Mean MAE ↓ | Macro-20 RMSE ↓ | Macro-13 RMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge | 5 | 0.3450 | 0.137092 | 0.0728 | 0.095242 | 0.107695 | 0.124220 |
| Ridge | 64 | 0.3601 | 0.138611 | 0.0514 | 0.095470 | 0.109207 | 0.126198 |
| Elastic Net | 5 | 0.3390 | 0.137380 | 0.0687 | 0.095382 | 0.107930 | 0.124548 |
| Elastic Net | 64 | 0.3491 | 0.139082 | 0.0450 | 0.095568 | 0.109586 | 0.126558 |
| Neural network | 5 | 0.3435 | 0.137392 | 0.0682 | 0.095508 | 0.107551 | 0.124217 |
| Neural network | 64 | 0.3616 | 0.138203 | 0.0569 | 0.095186 | 0.108525 | 0.125982 |
| Training mean | – | NA | 0.144030 | -0.0286 | 0.100367 | 0.119545 | 0.129545 |

Each metric is calculated on one 78-line test set and then averaged over 20 splits; predictions are not stacked into independent repeated observations. Macro-20 RMSE equally weights all 20 lineage-specific test RMSEs. Macro-13 is secondary and equally weights the 13 lineages with at least 10 cohort lines. Correlations are undefined for constant predictions and are retained as NA; their defined-repeat counts are saved. The training-mean control uses only each outer-training target mean, so no test response enters the control prediction.

![Model comparison](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_prism_direct_comparison/model_comparison.png)

## Five-gene versus 64-gene comparisons

| Family | Pooled RMSE reduction: 5 vs 64 | Relative RMSE reduction | Favorable splits | Favorable lineages | Conditional 95% interval |
|---|---:|---:|---:|---:|---|
| Ridge | +0.001519 | +1.096% | 15/20 | 11/20 | [-0.000606, +0.004371] |
| Elastic Net | +0.001702 | +1.224% | 16/20 | 12/20 | [-0.000244, +0.004273] |
| Neural network | +0.000811 | +0.586% | 13/20 | 9/20 | [-0.001669, +0.004319] |

Positive improvement favors the five-gene panel. No equivalence or noninferiority margin was specified; intervals containing zero do not establish equal performance.

| Family | Metric | Improvement: 5 vs 64 | Favorable / defined splits | Conditional 95% interval |
|---|---|---:|---:|---|
| Ridge | pooled_spearman | -0.015143 | 8/20 | [-0.076823, 0.063223] |
| Ridge | macro_all_rmse | 0.001512 | 14/20 | [-0.002073, 0.005021] |
| Elastic Net | pooled_spearman | -0.010115 | 8/20 | [-0.064836, 0.060453] |
| Elastic Net | macro_all_rmse | 0.001656 | 16/20 | [-0.001572, 0.004639] |
| Neural network | pooled_spearman | -0.018167 | 6/20 | [-0.083020, 0.066867] |
| Neural network | macro_all_rmse | 0.000974 | 12/20 | [-0.002453, 0.004359] |

![Panel differences](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_prism_direct_comparison/paired_panel_differences.png)

## Model comparisons within each panel

| Panel | First model vs reference | Mean pooled RMSE reduction | Favorable splits | Conditional 95% interval |
|---:|---|---:|---:|---|
| 5 | Elastic Net (5) vs Ridge (5) | -0.000288 | 8/20 | [-0.000555, -0.000008] |
| 64 | Elastic Net (64) vs Ridge (64) | -0.000471 | 7/20 | [-0.000905, +0.000056] |
| 5 | Neural network (5) vs Ridge (5) | -0.000300 | 8/20 | [-0.000954, +0.000625] |
| 64 | Neural network (64) vs Ridge (64) | +0.000408 | 15/20 | [+0.000043, +0.000758] |
| 5 | Neural network (5) vs Elastic Net (5) | -0.000012 | 9/20 | [-0.000737, +0.000939] |
| 64 | Neural network (64) vs Elastic Net (64) | +0.000879 | 15/20 | [+0.000123, +0.001512] |

## Training-mean control

| Model | Mean RMSE reduction vs training mean | Favorable splits | Conditional 95% interval |
|---|---:|---:|---|
| Ridge (5) | +0.006938 | 19/20 | [+0.003707, +0.010763] |
| Ridge (64) | +0.005419 | 19/20 | [+0.002056, +0.008689] |
| Elastic Net (5) | +0.006650 | 19/20 | [+0.003469, +0.010434] |
| Elastic Net (64) | +0.004948 | 19/20 | [+0.001840, +0.008074] |
| Neural network (5) | +0.006638 | 19/20 | [+0.002999, +0.011154] |
| Neural network (64) | +0.005827 | 19/20 | [+0.002305, +0.009306] |

Positive reductions indicate a lower mean test error than predicting the training-set mean. Unlike the prior cross-assay test, these RMSE and R² values use the same PRISM response endpoint for training and testing. They assess direct within-PRISM prediction and are not directly interchangeable with the earlier CTRP benchmark values.

![Training-mean control](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_prism_direct_comparison/training_mean_comparison.png)

## Cohort and split design

- **386 cancer cell lines, 379 identity groups, 20 lineages.** Every lineage has at least three identity groups, so all lines can enter the benchmark without introducing a singleton-lineage exception. Cervix was already absent from the retained PRISM cohort.
- Each of the 20 response-blind grouped lineage splits has **308 training and 78 testing lines** (79.8%/20.2%), with all 20 lineages on both sides and zero identity-group overlap. Per-lineage test counts use the closest feasible intact-group subset to a 20% target.
- The same 20 numeric outer seeds, inner seeds and NN initialization triplets as the prior comparisons are reused. The split masks are newly generated for this PRISM cohort; CTRP masks are not copied onto different samples.
- Five inner folds per outer split. Every inner training fit retains all 20 lineages; inner validation contains 17–20 lineages because rare groups cannot populate every fold. Protected training-only rows: 0.
- 370/386 lines occur in at least one outer test set across the 20 random holdouts; 16 never appear in a test set. Individual counts and every lineage split are saved.
- Official response source: [Broad DepMap PRISM Repurposing 20Q2 v1](https://doi.org/10.6084/m9.figshare.20564034.v1), erastin BRD-A25004090-001-08-4, HTS002 secondary-screen dose-response AUC. The cohort comes from the completed source/STR/expression-matching audit; all 386 matched lines are retained, including nine excluded from the previous transfer scoring only because they had no eligible old held-out fit.
- Target is negative reported PRISM AUC; higher means more sensitive. Retain reported values above 1 and finite processed curves. No clipping, IC50 substitution, response imputation, extra response-quality cutoff or CTRP target values are used.
- Expression is the same verified DepMap 24Q4 log2(TPM+1) matrix and gene mapping. Every X/y scaler is fitted on current training rows only.

## Fixed tuning grids

- **Ridge:** 17 alpha values from 0.0001 to 10,000, logarithmically spaced; training-only X standardization and target centering.
- **Elastic Net:** 41 explicit alphas from 10 to 0.0001 × L1 ratios 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1 (328 candidates). Standardize X/y in current training. Cyclic coordinate descent, tol 1e-9 and max_iter 50,000. Selected outer models are native sklearn ElasticNet fits.
- **NN:** 16/32 tanh hidden units × alpha 1, 10, 100, 1,000 (eight candidates). Three independently initialized members averaged for every candidate and outer fit. L-BFGS, max_iter 2,000, max_fun 40,000, tol 0.001, no early stopping; training-only X/y standardization. The grid expands the prior 10/100-alpha search at both ends before any fitting.
- All arms minimize the equal-mean five-fold unweighted RMSE in original PRISM sensitivity units. Each arm, panel and outer split is independently tuned inside the common inner folds. Ties select the first fixed grid member. No outer test response enters fitting or tuning.
- Ridge and Elastic Net alpha numbers are not directly comparable because their objective normalizations and target scaling differ.

## Selected settings and Elastic Net sparsity

| Family | Panel | Selected settings |
|---|---:|---|
| Ridge | 5 | alpha=31.6228 (8/20); alpha=100 (12/20) |
| Ridge | 64 | alpha=316.228 (3/20); alpha=1000 (17/20) |
| Elastic Net | 5 | alpha=0.0237137, l1_ratio=0.75 (1/20); alpha=0.0316228, l1_ratio=1 (2/20); alpha=0.0421697, l1_ratio=0.75 (4/20); alpha=0.0421697, l1_ratio=1 (2/20); alpha=0.0562341, l1_ratio=0.9 (1/20); alpha=0.0749894, l1_ratio=0.25 (2/20); alpha=0.1, l1_ratio=0.25 (1/20); alpha=0.177828, l1_ratio=0.01 (1/20); alpha=0.237137, l1_ratio=0.01 (4/20); alpha=0.237137, l1_ratio=0.1 (1/20); alpha=0.562341, l1_ratio=0.01 (1/20) |
| Elastic Net | 64 | alpha=0.0749894, l1_ratio=0.9 (1/20); alpha=0.1, l1_ratio=0.75 (1/20); alpha=0.133352, l1_ratio=0.75 (1/20); alpha=0.133352, l1_ratio=0.9 (1/20); alpha=0.562341, l1_ratio=0.1 (1/20); alpha=0.749894, l1_ratio=0.05 (1/20); alpha=1, l1_ratio=0.01 (1/20); alpha=1, l1_ratio=0.05 (2/20); alpha=1.33352, l1_ratio=0.01 (1/20); alpha=1.33352, l1_ratio=0.05 (1/20); alpha=1.77828, l1_ratio=0.01 (2/20); alpha=2.37137, l1_ratio=0.01 (3/20); alpha=3.16228, l1_ratio=0.01 (4/20) |
| Neural network | 5 | width=16, alpha=10 (10/20); width=32, alpha=10 (10/20) |
| Neural network | 64 | width=16, alpha=100 (18/20); width=32, alpha=100 (2/20) |

| Elastic Net input | Median nonzero genes | Range | Constant outer fits |
|---|---:|---:|---:|
| 5 | 4 | 3–5 | 0/20 |
| 64 | 36.5 | 6–55 | 0/20 |

Nonzero coefficients describe fitted sparsity, not causal gene importance. Complete split-specific coefficients and selection frequencies are saved; no new compact panel is inferred from test performance.

![Lineage panel effects](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_prism_direct_comparison/lineage_panel_differences.png)

## Validation and uncertainty

- Audited all 120 newly trained outer models, reconstructed 70,600 candidate-fold scores and checked 720 outer/inner identity exclusions.
- Saved prediction maximum reconstruction error 2.22e-16; inner RMSE error 1.11e-16. Verified training-only scalers, common masks and exact source hashes.
- Recomputed unweighted NN objectives for all 120 selected ensemble members; maximum error 5.55e-17. All saved training weights equal one.
- Checked all 65,600 inner Elastic Net candidate solutions and 40 selected outer objectives; maximum optimality-condition residual 4.17e-10.
- Preflight replayed all inner ridge/NN candidates and selected outer fits in repeat 1 using independent native sklearn APIs. Native Elastic Net paths and augmented-row Lasso equivalence were checked across all mixing ratios and representative alphas. Selected inner/outer fits for every arm were independently replayed in repeats 1 and 20.
- Convergence warnings across all candidate/selected fits: 0; selected outer-member convergence warnings: 0. Detailed warning categories are saved.
- Uncertainty uses 5,000 paired whole-lineage bootstrap draws shared across arms and repeat-specific test sets. Fits and feature selection are not repeated; no multiplicity or model-choice correction is applied. Correlation differences use common defined repeats; no undefined correlation is silently set to zero. Bootstrap arithmetic was independently checked against expanded SciPy samples, including constant-prediction controls.

## Interpretation limits

This is an exploratory internal PRISM benchmark. The five genes were fixed from earlier CTRP importance before direct PRISM fitting, but the PRISM transfer results and this cohort were already inspected. There is no untouched final test set in this experiment. The same biological lines recur across overlapping random holdouts, so 20 seeds are not independent biological replications. The design evaluates new lines from represented lineages; it does not establish absent-lineage, patient or clinical generalization. Public processed AUC/STR checks do not replace raw-well or replicate-level QC.

## Files

- summary.tsv, paired_comparisons.tsv, paired_seed_differences.tsv, lineage_win_counts.tsv: mean scores, paired intervals and consistency.
- per_repeat_pooled_metrics.tsv, per_repeat_macro_metrics.tsv, per_lineage_metrics.tsv: complete split/lineage results, including the training-mean control.
- panel_inputs.npz, shared_splits.npz, split_lineage_counts.tsv, inner_split_lineage_counts.tsv, sample_test_coverage.tsv: matched features, targets, seeds and all train/validation/test memberships.
- selected_parameters.tsv, tuning_scores.tsv, folds/: all tuning scores and all 120 PRISM-trained models with held-out predictions.
- elastic_net_coefficients.tsv, elastic_net_gene_selection.tsv, elastic_net_sparsity_summary.tsv: fitted coefficients and stability.
- protocol.json, preflight_validation.json, validation.json, source_manifest.json, artifact_manifest.json and scripts: fixed fitting choices, source provenance and checksums.
