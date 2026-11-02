# Lineage weights and evaluation influence

**There are no learned lineage coefficients in these models.** All nine current arms use expression features only. Lineage labels define the outer holdouts and identify groups for evaluation. A separate fitted model predicts each held-out lineage; that lineage has zero training weight in its own outer model.

Within a training subset, cells have equal explicit loss weight. There is no inverse-lineage balancing. XGBoost samples rows uniformly at rate 0.8. These are sampling/loss weights, not estimates of each training lineage’s actual influence on fitted parameters. Inner tuning minimizes the equal-weight mean of five fold RMSE values, not an equal-lineage objective.

## Explicit evaluation weights

| Lineage | Cells | Pooled MSE weight | Macro score weight | Prior gene-importance weight |
|---|---:|---:|---:|---:|
| Lung | 121 | 16.24% | 5.26% | 16.78% |
| Lymphoid | 95 | 12.75% | 5.26% | 13.18% |
| Esophagus/Stomach | 55 | 7.38% | 5.26% | 7.63% |
| CNS/Brain | 47 | 6.31% | 5.26% | 6.52% |
| Bowel | 44 | 5.91% | 5.26% | 6.10% |
| Myeloid | 41 | 5.50% | 5.26% | 5.69% |
| Skin | 40 | 5.37% | 5.26% | 5.55% |
| Pancreas | 38 | 5.10% | 5.26% | 5.27% |
| Ovary/Fallopian Tube | 37 | 4.97% | 5.26% | 5.13% |
| Breast | 34 | 4.56% | 5.26% | 4.72% |
| Head and Neck | 29 | 3.89% | 5.26% | 4.02% |
| Uterus | 26 | 3.49% | 5.26% | 3.61% |
| Kidney | 22 | 2.95% | 5.26% | 3.05% |
| Liver | 22 | 2.95% | 5.26% | 3.05% |
| Bladder/Urinary Tract | 19 | 2.55% | 5.26% | 2.64% |
| Peripheral Nervous System | 15 | 2.01% | 5.26% | 2.08% |
| Soft Tissue | 13 | 1.74% | 5.26% | 1.80% |
| Bone | 13 | 1.74% | 5.26% | 1.80% |
| Thyroid | 10 | 1.34% | 5.26% | 1.39% |
| Pleura | 9 | 1.21% | 0.00% | 0.00% |
| Biliary Tract | 6 | 0.81% | 0.00% | 0.00% |
| Prostate | 6 | 0.81% | 0.00% | 0.00% |
| Ampulla of Vater | 2 | 0.27% | 0.00% | 0.00% |
| Cervix | 1 | 0.13% | 0.00% | 0.00% |

These weights are identical for ridge, neural network and XGBoost at all three panel sizes. Pooled MSE weights are n/745. Pooled RMSE is the square root of the weighted sum of lineage MSE, not the weighted average of lineage RMSE. Macro RMSE and macro Spearman give 1/19 to each lineage with n≥10; the other five receive zero macro weight. The previous gene-importance primary summary used n/721 over those 19 larger lineages.

![Weights and gain contributions](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_lineage_influence/weights_and_nn64_gains.png)

## Model-dependent contributions

| Model | Three largest squared-error contributors | Combined error share | Within-global-rank component | Between-lineage-rank component | Pooled Spearman |
|---|---|---:|---:|---:|---:|
| Ridge 40 | Lymphoid, Lung, Ovary/Fallopian Tube | 38.0% | 0.2058 | 0.2180 | 0.4238 |
| Ridge 64 | Lymphoid, Lung, Ovary/Fallopian Tube | 38.2% | 0.2031 | 0.2343 | 0.4374 |
| Ridge 80 | Lymphoid, Lung, Ovary/Fallopian Tube | 37.8% | 0.1957 | 0.2373 | 0.4331 |
| Neural network 40 | Lymphoid, Lung, Ovary/Fallopian Tube | 38.3% | 0.2105 | 0.2178 | 0.4282 |
| Neural network 64 | Lymphoid, Lung, Ovary/Fallopian Tube | 38.3% | 0.2039 | 0.2339 | 0.4378 |
| Neural network 80 | Lymphoid, Lung, Ovary/Fallopian Tube | 37.9% | 0.1979 | 0.2362 | 0.4342 |
| XGBoost 40 | Lymphoid, Lung, Bowel | 38.1% | 0.2150 | 0.1821 | 0.3971 |
| XGBoost 64 | Lymphoid, Lung, Ovary/Fallopian Tube | 38.9% | 0.2075 | 0.1870 | 0.3945 |
| XGBoost 80 | Lymphoid, Lung, Ovary/Fallopian Tube | 38.3% | 0.1992 | 0.1898 | 0.3890 |

Error share is SSE from a lineage divided by total SSE, averaged over repeats. It combines lineage size and prediction difficulty. A large error share does not mean that lineage is a highly weighted input feature.

The Spearman terms exactly decompose covariance of the full-cohort response/prediction ranks into within-lineage and between-lineage parts, normalized by the same global denominator. They sum to pooled Spearman. The within-global-rank component is not the mean of local within-lineage Spearman scores; local ranks use a different scale. Components are descriptive associations, not causal effects or independent predictive abilities.

![Model lineage contributions](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_lineage_influence/lineage_contributions.png)

## What drives the 64-gene neural-network gain

The neural network pooled Spearman increase decomposes as follows:

- Between-lineage global-rank component: 0.217785 → 0.233905 (change +0.016119).
- Within-lineage global-rank component: 0.210450 → 0.203897 (change -0.006553).
- Their sum changes by +0.009566. The increase comes from the between-lineage term in this algebraic decomposition; it does not establish a causal role for lineage or a learned categorical lineage feature.

Largest positive contributions:

- Esophagus/Stomach, n=55: weighted MSE reduction +0.00009295; 28.2% of gross positive gain, 0.0% of gross deterioration.
- Lymphoid, n=95: weighted MSE reduction +0.00005306; 16.1% of gross positive gain, 0.0% of gross deterioration.
- Head and Neck, n=29: weighted MSE reduction +0.00005232; 15.9% of gross positive gain, 0.0% of gross deterioration.
- Lung, n=121: weighted MSE reduction +0.00003221; 9.8% of gross positive gain, 0.0% of gross deterioration.
- Pancreas, n=38: weighted MSE reduction +0.00002819; 8.6% of gross positive gain, 0.0% of gross deterioration.

Largest negative contributions:

- Bowel, n=44: weighted MSE reduction -0.00003443; 0.0% of gross positive gain, 24.4% of gross deterioration.
- Uterus, n=26: weighted MSE reduction -0.00002704; 0.0% of gross positive gain, 19.2% of gross deterioration.
- Thyroid, n=10: weighted MSE reduction -0.00001733; 0.0% of gross positive gain, 12.3% of gross deterioration.
- Breast, n=34: weighted MSE reduction -0.00001702; 0.0% of gross positive gain, 12.1% of gross deterioration.
- Biliary Tract, n=6: weighted MSE reduction -0.00001269; 0.0% of gross positive gain, 9.0% of gross deterioration.

Largest negative contributions to the equal-lineage Spearman change:

- Soft Tissue: -0.01684 of the mean Spearman change.
- Bone: -0.00561 of the mean Spearman change.
- Bowel: -0.00492 of the mean Spearman change.

Soft Tissue has only 13 cells (1.74% of pooled MSE weight) but receives 5.26% macro weight. Its contribution to the macro-Spearman change is -0.016845, compared with -0.016175 overall; the remaining 18 eligible lineages sum to +0.000670. This is a descriptive explanation of the aggregate score tradeoff; the small lineage itself has substantial sampling uncertainty.

Signed weighted MSE contributions sum exactly to the total MSE reduction. Gross-positive percentages normalize only positive contributions, so they must not be interpreted as percentages of the net gain. Expansion effects for every family and both larger panels are provided in the data tables.

## Omission sensitivity and limits

The omission table removes one lineage from scoring and recomputes pooled RMSE and Spearman, keeping all trained models and predictions fixed. It is a score-sensitivity diagnostic. It does not estimate the effect of removing that lineage from training, which would require additional model fitting. An RMSE change below zero means the omitted lineage was raising pooled prediction error; a Spearman change above zero means its omission improves the pooled ranking score.

All scores reuse the same 20 seeds and biological cohort. No new model fitting, external validation or causal lineage-importance claim is made. Small lineages have imprecise descriptive metrics, and a singleton has no defined within-lineage correlation. The original conditional bootstrap intervals for panel comparisons remain in the parent experiment; this decomposition does not add new hypothesis tests.

Validation passed: MSE reconstruction error ≤ 4.44e-16; global-rank Spearman decomposition error ≤ 1.67e-16.

## Files

- [Weights](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_lineage_influence/lineage_weights.tsv)
- [Training weight by held-out lineage](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_lineage_influence/training_lineage_weights.tsv)
- [All model/lineage contributions](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_lineage_influence/per_model_lineage_contributions.tsv)
- [Panel-expansion gain contributions](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_lineage_influence/panel_gain_lineage_contributions.tsv)
- [Evaluation omission sensitivity](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_lineage_influence/evaluation_omission_sensitivity.tsv)
- [Validation](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_lineage_influence/validation.json)
