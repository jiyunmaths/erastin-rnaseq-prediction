# Nonlinear Bayesian network comparison

**Adding nonlinear spline conditional means did not improve prediction.** The nonlinear network has 4.08% higher pooled RMSE than ridge, 4.31% higher than the neural network, and 2.38% higher than its matched linear-CPD control. It trails all three on pooled RMSE and Spearman in every matched repeat. Paired conditional lineage-bootstrap intervals favor ridge and the neural network. The nonlinear functions therefore added no predictive benefit in this registered configuration on this cohort.

A nonlinear additive-spline Bayesian network was fitted and evaluated across the same 20 inner-fold seeds and 24 outer lineage holdouts. The data remain 745 cancer cell lines, 734 related-identity groups and the approved 40-gene expression panel. The ridge and neural-network predictions come from the exact matching saved splits.

The matched linear-CPD network uses the same graph, node-wise fitting scheme and tuning grid, with straight-line conditional means. This control separates the effect of adding nonlinear conditional means from changes in regularization and fitting relative to the earlier Gaussian TAN.

## Held-out results

| Model | Pooled Spearman ↑ | Pooled RMSE ↓ | Pooled R² ↑ | Macro Spearman ↑ | Macro RMSE ↓ |
|---|---:|---:|---:|---:|---:|
| Nonlinear spline BN | 0.37107 | 0.125678 | 0.10490 | 0.20558 | 0.125321 |
| Neural network, 40 genes | 0.42824 | 0.120484 | 0.17737 | 0.22628 | 0.120872 |
| Ridge, 40 genes | 0.42380 | 0.120750 | 0.17373 | 0.23008 | 0.121250 |
| Matched linear-CPD BN | 0.38198 | 0.122762 | 0.14596 | 0.20670 | 0.122941 |
| Previous Gaussian TAN | 0.38167 | 0.122823 | 0.14512 | 0.20591 | 0.123185 |

The table gives means of per-repeat scores, not the score of an ensemble averaging predictions across repeats. Macro scores weight the same 19 lineages with n≥10 equally; all 24 lineages remain in pooled scores.

![Model comparison](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_nonlinear_bayesian/model_comparison.png)

## Paired improvement of the nonlinear network

Positive means better. Intervals use 5,000 paired resamples of whole held-out lineages, keeping all models and repeats together.

| Scope | Reference | Metric | Mean improvement | Favorable repeats | Conditional 95% interval |
|---|---|---|---:|---:|---|
| pooled_745 | Ridge, 40 genes | Spearman increase | -0.052735 | 0/20 | [-0.093844, -0.015292] |
| pooled_745 | Ridge, 40 genes | RMSE reduction | -0.004928 | 0/20 | [-0.006994, -0.002337] |
| pooled_745 | Neural network, 40 genes | Spearman increase | -0.057168 | 0/20 | [-0.094278, -0.023904] |
| pooled_745 | Neural network, 40 genes | RMSE reduction | -0.005195 | 0/20 | [-0.007043, -0.002911] |
| pooled_745 | Matched linear-CPD BN | Spearman increase | -0.010916 | 0/20 | [-0.020933, -0.000071] |
| pooled_745 | Matched linear-CPD BN | RMSE reduction | -0.002916 | 0/20 | [-0.004536, -0.001218] |
| pooled_745 | Previous Gaussian TAN | Spearman increase | -0.010602 | 0/20 | [-0.019866, +0.000680] |
| pooled_745 | Previous Gaussian TAN | RMSE reduction | -0.002855 | 0/20 | [-0.004564, -0.001060] |
| macro_19 | Ridge, 40 genes | Spearman increase | -0.024491 | 0/20 | [-0.071743, +0.026279] |
| macro_19 | Ridge, 40 genes | RMSE reduction | -0.004071 | 0/20 | [-0.005902, -0.002358] |
| macro_19 | Neural network, 40 genes | Spearman increase | -0.020698 | 0/20 | [-0.063389, +0.023305] |
| macro_19 | Neural network, 40 genes | RMSE reduction | -0.004448 | 0/20 | [-0.006121, -0.002793] |
| macro_19 | Matched linear-CPD BN | Spearman increase | -0.001117 | 16/20 | [-0.018452, +0.018837] |
| macro_19 | Matched linear-CPD BN | RMSE reduction | -0.002380 | 0/20 | [-0.003720, -0.001099] |
| macro_19 | Previous Gaussian TAN | Spearman increase | -0.000323 | 16/20 | [-0.016408, +0.016591] |
| macro_19 | Previous Gaussian TAN | RMSE reduction | -0.002136 | 0/20 | [-0.003531, -0.000813] |

Among the 19 larger lineages, averaging effects over 20 repeats:

- Versus Ridge, 40 genes: higher Spearman in 7/19; lower RMSE in 3/19.
- Versus Neural network, 40 genes: higher Spearman in 7/19; lower RMSE in 2/19.
- Versus Matched linear-CPD BN: higher Spearman in 7/19; lower RMSE in 5/19.
- Versus Previous Gaussian TAN: higher Spearman in 8/19; lower RMSE in 6/19.

![Lineage comparison](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_nonlinear_bayesian/lineage_comparison.png)

## What makes this network nonlinear

Each gene has a Gaussian conditional distribution whose mean is an additive function of continuous sensitivity and, except for the gene-tree root, one gene parent: `X_j | Y, X_parent ~ Normal(intercept_j + f_j(Y) + g_j(X_parent), variance_j)`. The functions use cubic B-splines with four training-quantile knots, duplicate-knot handling, ridge regularization and linear extrapolation. Knots, basis scalers, response/gene scalers, graph and node parameters are all fitted separately within each current training subset.

At prediction time all 40 genes are observed. The model combines the likelihood factors with the training Gaussian response prior and integrates the resulting one-dimensional posterior over sensitivity. Its conditional posterior can be non-Gaussian and its prediction is not restricted to a linear function of expression. No target discretization, response clipping or finite response-range truncation is used.

The graph is a maximum spanning tree of Gaussian conditional gene mutual information given response, oriented from the fixed SLC7A11 root. The two new families share this graph within a training subset. Nonlinear structure search, alternative tree orientations and non-additive interactions were not tested. With nonlinear CPDs the choice of orientation can matter; the fixed orientation is a modeling assumption, not a causal conclusion.

Nonlinear Gaussian conditional means estimated with splines are also studied in the recent [semi-linear Gaussian BN preprint](https://arxiv.org/abs/2610.30838). This experiment is a custom TAN implementation, not a reproduction of that paper’s structure search or smoothing penalty. The basis implementation follows [SplineTransformer](https://scikit-learn.org/1.6/modules/generated/sklearn.preprocessing.SplineTransformer.html); the installed library version is recorded in the protocol.

## Selected regularization and numerical accuracy

The common node penalty was tuned over [0, 1, 10, 100, 300, 1000, 3000, 10000, 1000000] by equal-weight mean five-fold inner RMSE. The grid and numerical criteria were fixed before inspecting this experiment’s held-out performance.

| Network | Alpha | Selected outer fits |
|---|---:|---:|
| Matched linear-CPD BN | 1000 | 20/480 |
| Matched linear-CPD BN | 3000 | 460/480 |
| Nonlinear spline BN | 0 | 1/480 |
| Nonlinear spline BN | 1 | 13/480 |
| Nonlinear spline BN | 10 | 8/480 |
| Nonlinear spline BN | 100 | 2/480 |
| Nonlinear spline BN | 10000 | 456/480 |

Distinct full-cohort prediction vectors across the 20 repeats: nonlinear = 7, linear control = 1. These algorithms are deterministic; seeds vary inner tuning assignments only.

Gaussian tails are integrated analytically because spline extrapolation is linear. Inside the spline knots, quadrature order doubles until posterior mean, variance and log normalization converge. Every selected nonlinear model was additionally checked at higher integration order.

- Maximum change in predicted sensitivity under higher-order integration: 1.89e-15.
- Maximum change in conditional variance: 2.39e-16.
- All inner and outer numerical integrations converged; no unconverged predictions were accepted.
- Independent sklearn Ridge refits checked every node for the first and last seeds in Lung and Thyroid.
- Independent SciPy adaptive integration of the likelihood product checked three fixed sample IDs in each of those four nonlinear holdouts.
- The zero-penalty linear limit reproduced the previous Gaussian BN mathematically, and a synthetic nonlinear relationship verified nonlinear prediction behavior.

## Conditional density scores

Negative log likelihood evaluates the fitted normalized response density, using the nonlinear posterior directly rather than substituting a Gaussian approximation. The models estimate parameters and structure; they do not average over posterior parameter uncertainty.

| Model | Mean held-out NLL ↓ | Mean conditional predictive SD |
|---|---:|---:|
| Previous Gaussian TAN | -0.67190 | 0.12998 |
| Matched linear-CPD BN | -0.67219 | 0.12999 |
| Nonlinear spline BN | -0.62452 | 0.12935 |

## Interpretation limits

- The same cohort and outer lineage tests have now supported several exploratory model comparisons. There is no new external drug screen or clinical validation.
- Repeated seeds are not independent biological replications. Bootstrap intervals condition on the fitted predictions, do not refit models, and are not adjusted for multiple comparisons.
- The same grouped inner folds are preserved; they mix represented training lineages. This study does not evaluate lineage-held-out inner tuning.
- The network’s arrows are a statistical factorization. Learned dependencies and nonlinear shapes are not validated causal gene relationships.
- The response prior is Gaussian, conditional gene variances are constant, functions are additive and each gene has at most one gene parent in addition to response. Other nonlinear Bayesian-network families remain untested.

## Files

- [All scores](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_nonlinear_bayesian/summary.tsv)
- [Paired comparisons](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_nonlinear_bayesian/paired_comparisons.tsv)
- [Per-lineage effects](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_nonlinear_bayesian/lineage_paired_effects.tsv)
- [Protocol](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_nonlinear_bayesian/protocol.json)
- [Validation](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_nonlinear_bayesian/validation.json)
