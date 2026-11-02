# Repeated neural-network versus ridge comparison

The neural network improved pooled Spearman in 16/20 repeats and reduced pooled RMSE in 20/20. The mean improvements were +0.00443 in Spearman and +0.000266 in RMSE reduction. Paired uncertainty and lineage-level behavior are reported below.

Both models use the same 40-gene expression panel, 745 cancer cell lines, 734 identity groups, and 24 fixed leave-lineage-out tests. Each repeat pairs neural-network predictions with ridge predictions from the exact same saved inner-fold assignment.

Twenty inner-fold seeds and twenty fresh, fixed neural initialization trios were registered before fitting. Each neural prediction averages three members. Inner splits and initializations change jointly, so this study assesses their combined sensitivity and does not separate their contributions. The original single-split result is excluded from the repeat averages.

## Mean held-out performance

Means of per-repeat metrics; SD describes variation across repeats. These are not scores of an averaged-prediction ensemble.

| Scope | Model | Spearman mean ± SD ↑ | RMSE mean ± SD ↓ | R² mean ± SD ↑ |
|---|---|---:|---:|---:|
| Pooled, 745 cells | Ridge | 0.42380 ± 0.00413 | 0.12075 ± 0.00021 | 0.17373 ± 0.00282 |
| Pooled, 745 cells | Neural network | 0.42824 ± 0.00052 | 0.12048 ± 0.00003 | 0.17737 ± 0.00044 |
| Equal-weight mean, 19 lineages | Ridge | 0.23008 ± 0.00360 | 0.12125 ± 0.00020 | -0.08775 ± 0.00565 |
| Equal-weight mean, 19 lineages | Neural network | 0.22628 ± 0.00197 | 0.12087 ± 0.00004 | -0.07708 ± 0.00072 |

## Paired neural-network improvement

Positive means better than the matched ridge model. Intervals use 5,000 paired resamples of whole held-out lineages.

| Scope | Metric | Mean improvement | Favorable repeats | Conditional 95% interval |
|---|---|---:|---:|---:|
| Pooled cells | Spearman increase | +0.004433 | 16/20 | [-0.003640, +0.013419] |
| Pooled cells | RMSE reduction | +0.000266 | 20/20 | [-0.000215, +0.000941] |
| Pooled cells | R² increase | +0.003640 | 20/20 | [-0.002883, +0.014235] |
| Equal-weight lineages | Spearman increase | -0.003794 | 3/20 | [-0.029570, +0.015369] |
| Equal-weight lineages | RMSE reduction | +0.000378 | 20/20 | [-0.000172, +0.001005] |
| Equal-weight lineages | R² increase | +0.010666 | 20/20 | [-0.001574, +0.025171] |

The mean pooled RMSE reduction relative to ridge is 0.22%. Both paired pooled improvement intervals include zero. The error reduction was consistent across these seeds, but a clear advantage across lineages is not established.

![Paired repeat effects](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_nn_repeated/paired_seed_effects.png)

## Behavior by held-out lineage

Using mean effects across all 20 repeats, among the 19 lineages with n ≥ 10:

- Neural network has higher Spearman in 9/19 lineages.
- Neural network has lower RMSE in 10/19 lineages.
- Neural network has higher R² in 10/19 lineages.

![Lineage effects](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_nn_repeated/lineage_effects.png)

## Protocol and audit

- The neural grid is unchanged: one tanh layer with 16 or 32 units, L2 alpha 10 or 100; four candidates. Each candidate is a mean of three fixed initialization seeds.
- X and target scalers are fitted only on each current training subset. L-BFGS uses max_iter=2000, max_fun=40000, tol=0.001, without a random early-stopping validation split.
- Each outer fold retunes candidates by equal-weight mean five-fold inner RMSE. Held-out lineages and related identities never enter supervised training or tuning.
- The unchanged neural pipeline exactly replays the earlier Lung and Thyroid tests, including every candidate loss and selected setting.
- The 480 new selected ensemble models and all 1,920 candidate validation-prediction sets are audited. Independent sklearn pipelines refit both neural network and ridge on the first and last repeats for Lung and Thyroid.
- Neural member fits with convergence warnings: 0/30240; selected outer members: 0/1440.
- Selected settings, seed triplets, predictions, inner losses, fitted models, hashes and numerical audits are saved alongside this review.

## Limits

- The repeats reuse the same cohort and outer lineage tests. Win counts describe stability, not independent replication; no independent-seed t-test or binomial significance test is used.
- Pooled bootstrap draws resample 24 complete lineages, preserving all model and repeat predictions together. Macro draws resample the 19 lineages with n≥10. Models are not retrained within draws, and training sets overlap across holdouts.
- Five smaller lineages remain in pooled scores and detailed tables but are excluded from macro scores and lineage win counts.
- This follow-up was motivated by the earlier marginal neural-network gain. No external screen or independent cohort was added.
- Inner validation mixes represented training lineages; the same established design was preserved for a matched comparison. This study does not evaluate lineage-held-out inner tuning.
- Only the 40-gene expression panel is used. The archived CTRP AUC and DepMap 24Q4 data limitations remain unchanged.
