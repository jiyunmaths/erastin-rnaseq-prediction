# XGBoost comparison on the 40-gene panel

**The tested XGBoost pipeline did not consistently improve on ridge or the neural network.** Mean pooled RMSE was 0.53% higher than ridge and 0.76% higher than the neural network. XGBoost improved pooled RMSE in 2/20 and 1/20 repeats, respectively, and pooled Spearman in 0/20 against either baseline. All paired lineage-bootstrap intervals include zero, so the observed performance differences remain uncertain. Equal-lineage Spearman was close to both baselines; equal-lineage RMSE was higher than both. These results do not justify replacing the existing leading models.

Completed 20 matched repeats on 745 cancer cell lines, 734 identity groups, and the same 24 leave-lineage-out tests. Ridge and neural-network predictions were reused from the exact saved inner split seeds. Scores below are means of per-repeat scores, not scores after averaging predictions across repeats.

| Model | Pooled Spearman ↑ | Pooled RMSE ↓ | Pooled R² ↑ | Macro Spearman ↑ | Macro RMSE ↓ |
|---|---:|---:|---:|---:|---:|
| XGBoost | 0.39710 | 0.121394 | 0.16488 | 0.22743 | 0.121739 |
| Neural network | 0.42824 | 0.120484 | 0.17737 | 0.22628 | 0.120872 |
| Ridge | 0.42380 | 0.120750 | 0.17373 | 0.23008 | 0.121250 |

Macro scores weight the same 19 lineages with n≥10 equally. All 24 lineages remain in pooled scores.

![Model comparison](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_xgboost_comparison/model_comparison.png)

## Paired results

Positive means XGBoost improves on the reference. Intervals are from 5,000 paired whole-lineage bootstrap resamples.

| Scope | Reference | Metric | Mean improvement | Favorable repeats | Conditional 95% interval |
|---|---|---|---:|---:|---|
| pooled_745 | Ridge | Spearman increase | -0.026698 | 0/20 | [-0.076313, +0.020040] |
| pooled_745 | Ridge | RMSE reduction | -0.000644 | 2/20 | [-0.002697, +0.001373] |
| pooled_745 | Neural network | Spearman increase | -0.031131 | 0/20 | [-0.080651, +0.016169] |
| pooled_745 | Neural network | RMSE reduction | -0.000910 | 1/20 | [-0.002885, +0.000965] |
| macro_19 | Ridge | Spearman increase | -0.002642 | 10/20 | [-0.055494, +0.049815] |
| macro_19 | Ridge | RMSE reduction | -0.000489 | 5/20 | [-0.002962, +0.001789] |
| macro_19 | Neural network | Spearman increase | +0.001152 | 12/20 | [-0.051015, +0.055525] |
| macro_19 | Neural network | RMSE reduction | -0.000867 | 1/20 | [-0.003112, +0.001260] |

Among the 19 larger lineages, using effects averaged across repeats:

- Versus Ridge: higher Spearman in 9/19; lower RMSE in 9/19.
- Versus Neural network: higher Spearman in 9/19; lower RMSE in 8/19.

Compared with the neural network, Kidney, Liver, Myeloid and Bladder/Urinary Tract had lower RMSE in all 20 repeats. Bowel, CNS/Brain, Thyroid, Breast, Skin and Bone had higher RMSE in all 20 repeats. These are descriptive results on the same held-out cell lines, not independently replicated lineage-specific effects.

![Lineage comparison](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_xgboost_comparison/lineage_comparison.png)

## Fitting and tuning

XGBoost CPU histogram trees used squared-error loss, learning rate 0.03, row and feature subsampling 0.8, 64 histogram bins, and one CPU thread per fit. The frozen grid crossed depth 1/2/3, minimum child weight 5/20, L2 leaf regularization 1/20, and 100/300/600 boosting rounds (36 candidates). The minimum equal-weight mean RMSE across the five saved inner folds selected the configuration. Each 600-tree training path supplied fixed prefixes; the selected length was then refitted on all outer-training rows. No early stopping or held-out-lineage tuning was used.

The 40 log2(TPM+1) features were used unchanged. Response scaling was fitted independently within each current training subset and predictions were returned to the original negative-AUC scale. Each repeat changes both the saved inner assignments and the recorded XGBoost row/column sampling seed. Outer holdouts are fixed. Each outer prediction uses one XGBoost model; the existing neural network uses its unchanged three-member ensemble.

The implementation uses the official [XGBoost training API](https://xgboost.readthedocs.io/en/stable/python/python_api.html) and [tree parameters](https://xgboost.readthedocs.io/en/stable/parameter.html). Installed CPU package version and wheel hash are recorded in the protocol.

Selected settings:

| Depth | Minimum child weight | L2 penalty | Boosting rounds | Outer fits |
|---:|---:|---:|---:|---:|
| 1 | 5 | 1 | 300 | 16 |
| 1 | 5 | 1 | 600 | 1 |
| 1 | 5 | 20 | 300 | 2 |
| 1 | 5 | 20 | 600 | 1 |
| 1 | 20 | 20 | 300 | 1 |
| 2 | 5 | 1 | 300 | 16 |
| 2 | 5 | 20 | 300 | 13 |
| 2 | 20 | 1 | 300 | 5 |
| 2 | 20 | 20 | 300 | 47 |
| 3 | 5 | 1 | 100 | 73 |
| 3 | 5 | 20 | 100 | 30 |
| 3 | 5 | 20 | 300 | 19 |
| 3 | 20 | 1 | 100 | 111 |
| 3 | 20 | 20 | 100 | 96 |
| 3 | 20 | 20 | 300 | 49 |

The upper tested depth (3) was selected in 378/480 fits, and the upper boosting length (600) in 2/480. These counts describe the frozen search limits; deeper trees and other learning rates were not tested. The grid was not expanded after viewing results.

## Validation

- All 480 saved outer models reproduced their predictions (maximum absolute error 0).
- All 86,400 candidate-fold scores were reconstructed; maximum RMSE difference 1.11e-16.
- Identity exclusion, held-out lineage exclusion, training-only target scaling, and exact split/baseline alignment passed.
- Independent sklearn-API refits checked Lung and Thyroid in repeats 1 and 20, including all five selected inner fits stopped directly at the selected prefix length.
- Recorded training warnings: 0.
- Frequency-weighted bootstrap calculations were checked against explicit lineage replication and SciPy ranks.

## Interpretation limits

- This cohort has supported multiple exploratory comparisons. There is no independent drug-screen or clinical validation; intervals are not adjusted for multiple comparisons.
- Seeds are not independent biological replications. Bootstrap intervals condition on saved predictions, do not refit models, and do not account fully for dependence from overlapping training sets.
- Outer tests hold out lineages, while the preserved group-disjoint inner folds mix represented training lineages. Inner tuning is not itself leave-lineage-out.
- Search budgets differ: 36 XGBoost candidates versus four neural-network candidates. The comparison evaluates these specified pipelines, not all configurations of each model family.
- The target remains negative reported CTRP erastin AUC; upstream AUC processing provenance remains unresolved.

## Reuse

The archive contains the 40-gene input matrix, shared splits, saved UBJ boosters, prediction arrays, selected settings, all candidate validation predictions, and audit tables. For a saved booster, predict the gene columns in the order of `panel_inputs.npz:genes`, then multiply by the matching metadata `y_scale` and add `y_mean`. These are evaluation models trained without their respective held-out lineage, not a final model trained on the full cohort.

- [All scores](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_xgboost_comparison/summary.tsv)
- [Paired comparisons](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_xgboost_comparison/paired_comparisons.tsv)
- [Per-lineage effects](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_xgboost_comparison/lineage_paired_effects.tsv)
- [Protocol](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_xgboost_comparison/protocol.json)
- [Validation](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_xgboost_comparison/validation.json)
