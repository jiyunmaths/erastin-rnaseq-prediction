# Lineage-stratified validation of the 64-gene neural network

**All 23 retained lineages appear in both training and testing in every one of the 20 repeats.** Cervix (one cell line) was excluded as requested. Each split has **595 training lines and 149 test lines**, with no related identity group crossing the split. The cohort contains 744 cell lines and 733 identity groups.

Lowest mean pooled RMSE: **Unweighted (0.116968)**. Lowest equal-lineage RMSE across all 23 lineages: **Unweighted (0.110514)**. These are descriptive model comparisons on the existing cohort; intervals below quantify conditional variation across observed lineages.

## Exactly how training and testing are split

Each lineage contributes approximately 80% of its samples to training and 20% to testing. The nearest feasible rounded sample count is used, with at least one identity group on each side. Whole identity groups are sampled uniformly among subsets meeting that sample-count target. All related cell-line identities stay together. The sample counts below are identical across the 20 repeats, while selected identity groups change with the seed. RNA expression and erastin response values are not used to generate splits.

| Lineage | Total | Training | Testing |
|---|---:|---:|---:|
| Lung | 121 | 97 | 24 |
| Lymphoid | 95 | 76 | 19 |
| Esophagus/Stomach | 55 | 44 | 11 |
| CNS/Brain | 47 | 38 | 9 |
| Bowel | 44 | 35 | 9 |
| Myeloid | 41 | 33 | 8 |
| Skin | 40 | 32 | 8 |
| Pancreas | 38 | 30 | 8 |
| Ovary/Fallopian Tube | 37 | 30 | 7 |
| Breast | 34 | 27 | 7 |
| Head and Neck | 29 | 23 | 6 |
| Uterus | 26 | 21 | 5 |
| Liver | 22 | 18 | 4 |
| Kidney | 22 | 18 | 4 |
| Bladder/Urinary Tract | 19 | 15 | 4 |
| Peripheral Nervous System | 15 | 12 | 3 |
| Soft Tissue | 13 | 10 | 3 |
| Bone | 13 | 10 | 3 |
| Thyroid | 10 | 8 | 2 |
| Pleura | 9 | 7 | 2 |
| Biliary Tract | 6 | 5 | 1 |
| Prostate | 6 | 5 | 1 |
| Ampulla of Vater | 2 | 1 | 1 |

![Train/test representation](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_lineage_stratified_nn/split_balance.png)

## Inner tuning and the rare-lineage exception

The outer 149-line test set is never used for preprocessing, weighting, candidate selection or fitting. Within the 595-line training set, five group-stratified validation folds choose among the original four neural-network candidates. All 23 lineages remain represented in every inner training set.

**Ampulla of Vater has only two cell lines:** one is outer training and one is outer testing in each repeat. Its single outer-training line is retained in every inner training fit and is never an inner-validation point. Consequently each inner validation set contains 22 lineages; it is mathematically impossible for that single training line to occur on both disjoint inner sides. The outer Ampulla test line remains entirely untouched by tuning.

The protocol retains minimum unweighted mean five-fold RMSE selection, training-only unweighted X/y standardization, 16/32 tanh units × alpha 10/100, L-BFGS and the same three initialization seeds per repeat. All four weighting arms share exactly the same outer and inner split assignments and initialization seeds. All 80 outer models were newly fitted, including the unweighted baseline and full inverse-frequency reference.

## Results

Scores are computed separately on each repeat’s test set and then averaged. The test sets overlap across repeats; no stacked 20×149 sample count is treated as independent. Macro-23 RMSE equally averages all 23 lineage-specific test RMSEs. Macro-19 RMSE uses the same 19 lineages with at least 10 total cohort samples as a secondary measure.

| Training weights | Pooled Spearman ↑ | Pooled RMSE ↓ | Pooled R² ↑ | Macro-23 RMSE ↓ | Macro-19 RMSE ↓ |
|---|---:|---:|---:|---:|---:|
| Unweighted | 0.46175 | 0.116968 | 0.20941 | 0.110514 | 0.114937 |
| Inverse square root | 0.45347 | 0.117473 | 0.20235 | 0.111182 | 0.115536 |
| Capped inverse (max 5) | 0.43361 | 0.118591 | 0.18680 | 0.112439 | 0.116777 |
| Full inverse frequency | 0.42797 | 0.119026 | 0.18081 | 0.112858 | 0.116994 |

![Model comparison](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_lineage_stratified_nn/model_comparison.png)

## Paired differences versus unweighted training

Positive improvement favors weighting: higher Spearman/R² or lower RMSE. Conditional intervals use 5,000 whole-lineage bootstrap draws shared across all four models and all 20 split-specific test sets. Macro-19 uses the same paired resampling restricted to its eligible lineage set. Models are not refitted in the bootstrap and intervals are not adjusted for multiplicity.

| Weights | Metric | Mean improvement | Favorable repeats | Conditional 95% interval |
|---|---|---:|---:|---|
| Inverse square root | pooled_spearman | -0.008279 | 5/20 | [-0.021777, +0.003708] |
| Inverse square root | pooled_rmse | -0.000506 | 5/20 | [-0.001145, +0.000011] |
| Inverse square root | macro23_rmse | -0.000669 | 6/20 | [-0.001801, +0.000512] |
| Capped inverse (max 5) | pooled_spearman | -0.028143 | 1/20 | [-0.056854, -0.005557] |
| Capped inverse (max 5) | pooled_rmse | -0.001623 | 1/20 | [-0.003067, -0.000442] |
| Capped inverse (max 5) | macro23_rmse | -0.001925 | 3/20 | [-0.004391, +0.000623] |
| Full inverse frequency | pooled_spearman | -0.033779 | 1/20 | [-0.063929, -0.010628] |
| Full inverse frequency | pooled_rmse | -0.002058 | 0/20 | [-0.003495, -0.000818] |
| Full inverse frequency | macro23_rmse | -0.002344 | 1/20 | [-0.004963, +0.000281] |

## Consistency across lineages

| Weights | Lineages with lower mean test RMSE than unweighted |
|---|---:|
| Inverse square root | 9/23 |
| Capped inverse (max 5) | 7/23 |
| Full inverse frequency | 6/23 |

Counts describe point estimates across repeated splits, not independently established lineage effects. Ampulla, Biliary and Prostate each have one test line per repeat; lineage RMSE is defined, but single-point Spearman is not. Other small test subsets also yield unstable correlations, so no macro Spearman is used. Per-lineage correlations are saved where mathematically defined.

## Weights and validation

- Unweighted assigns weight one. Inverse square root is proportional to 1/√n_lineage. Full inverse frequency gives equal total training weight to each represented lineage. Capped inverse uses min(c/n_lineage, 5), with c solved to keep the final mean one and final maximum five.
- All counts and normalizations use the current inner training or outer training rows only. All represented lineages participate, including small ones. The cap and schemes were fixed before inspecting these test predictions.
- Validated 80 newly fitted models, 1,600 inner candidate-fold scores, 480 training weight vectors and 240 independently reconstructed weighted training objectives.
- Saved prediction maximum error: 0; inner RMSE maximum error: 8.33e-17; weighted objective maximum error: 5.55e-17.
- Selected settings for all four arms in repeats 1 and 20 replayed both outer predictions and all five inner predictions. The same verified weighted-loss adapter is used; preflight verifies unchanged fitting code, unit-weight agreement and the new split-dependent bootstrap against explicit SciPy expansion.
- Convergence warnings: 0; total recorded warnings: 5,040. Per-fit diagnostics and categories are saved.
- Verified input/source hashes, Cervix removal, reproducible response-blind split generation, exact all-lineage representation, zero identity-group overlap, training-only scaling and common seeds across arms.

Across the 20 random holdouts, 724/744 distinct cell lines appeared in at least one test set; 20 were never tested. Repeated 80/20 holdouts do not guarantee every sample receives a test prediction. Coverage by sample and identity group is saved.

## What this experiment measures

This experiment predicts new cell lines from lineages already represented in training. The previous leave-lineage-out experiments measured transfer to absent lineages. Their predictions are not reused and their scores should not be pooled with this experiment. Changing the design also changes the training size, cohort and number of test samples per lineage.

The existing cohort has already informed earlier model and weighting choices, so these are exploratory results rather than untouched external validation. Repeated splits reuse samples and overlapping training sets; twenty seeds are not twenty independent biological replications. The lineage bootstrap is conditional on fitted predictions and describes variation across these observed lineages. Tiny lineage test subsets are particularly uncertain. The upstream harmonized erastin AUC provenance limitation remains unresolved.

## Files

- split_lineage_counts.tsv and inner_split_lineage_counts.tsv: representation in every outer/inner split.
- shared_splits.npz, inputs.npz and sample_test_coverage.tsv: fixed memberships, matched inputs and test coverage.
- summary.tsv, paired_comparisons.tsv, paired_seed_differences.tsv and lineage_paired_effects.tsv: all four weighting arms.
- per_repeat_pooled_metrics.tsv, per_repeat_macro_metrics.tsv and per_lineage_metrics.tsv: individual repeat and lineage scores.
- protocol.json, preflight_validation.json, validation.json and artifact_manifest.json: design and audit trail.
- predictions.npz, bootstrap_draws.npz, folds/ and source scripts: all new models, held-out predictions and reproducible analysis.
