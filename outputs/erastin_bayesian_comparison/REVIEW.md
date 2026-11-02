# Probabilistic graphical models for erastin sensitivity

**Neither Gaussian Bayesian network improves on the 40-gene ridge or neural-network models.** The tree-augmented network has 1.72% higher pooled RMSE than ridge and 1.94% higher RMSE than the neural network. It loses on pooled RMSE and Spearman in all 20 matched repeats; the corresponding paired 95% lineage-bootstrap intervals favor the existing models. Its small pooled improvement over the naive Bayesian network has intervals crossing zero, and its equal-weight lineage RMSE is slightly worse than naive.

Two continuous Gaussian Bayesian networks were tested on the same 40-gene expression panel and corrected leave-lineage-out splits. All 24 lineage holdouts were preserved, including 745 cancer cell lines and 734 identity groups. The network structure, expression/response scaling and distribution parameters were learned separately inside every training fold.

The tree-augmented model (TAN) adds 39 learned gene–gene dependencies to the naive network. The naive model assumes genes are independent given continuous sensitivity. Both predict the conditional mean of negative erastin AUC. No response discretization, BulkFormer refitting, gene filtering or prediction clipping was used.

## Matched comparison across 20 repeats

These are means of per-repeat held-out scores. Outer lineage tests stay fixed; seeds vary the inner tuning assignments. Ridge, neural-network and BulkFormer ridge predictions are reused from the exact matching saved seeds.

| Model | Pooled Spearman ↑ | Pooled RMSE ↓ | Pooled R² ↑ | Macro Spearman ↑ | Macro RMSE ↓ |
|---|---:|---:|---:|---:|---:|
| Neural network, 40 genes | 0.42824 | 0.120484 | 0.17737 | 0.22628 | 0.120872 |
| Ridge, 40 genes | 0.42380 | 0.120750 | 0.17373 | 0.23008 | 0.121250 |
| Tree-augmented Gaussian BN | 0.38167 | 0.122823 | 0.14512 | 0.20591 | 0.123185 |
| Naive Gaussian BN | 0.37836 | 0.123291 | 0.13859 | 0.20006 | 0.123109 |
| Ridge, panel BulkFormer | 0.36943 | 0.123800 | 0.13146 | 0.23336 | 0.122056 |
| Ridge, all-gene BulkFormer | 0.31053 | 0.126925 | 0.08702 | 0.18817 | 0.123584 |

![Repeated comparison](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_bayesian_comparison/model_comparison.png)

## Paired change for the tree-augmented Bayesian network

Positive values mean improvement. Intervals use 5,000 paired resamples of whole held-out lineages, conditional on the saved predictions.

| Scope | Compared with | Metric | Mean improvement | Favorable repeats | Conditional 95% interval |
|---|---|---|---:|---:|---|
| pooled_745 | Ridge, 40 genes | Spearman increase | -0.042133 | 0/20 | [-0.077866, -0.008826] |
| pooled_745 | Ridge, 40 genes | RMSE reduction | -0.002073 | 0/20 | [-0.003954, -0.000203] |
| pooled_745 | Neural network, 40 genes | Spearman increase | -0.046566 | 0/20 | [-0.079048, -0.016297] |
| pooled_745 | Neural network, 40 genes | RMSE reduction | -0.002339 | 0/20 | [-0.003979, -0.000779] |
| pooled_745 | Naive Gaussian BN | Spearman increase | +0.003311 | 20/20 | [-0.005816, +0.011573] |
| pooled_745 | Naive Gaussian BN | RMSE reduction | +0.000468 | 20/20 | [-0.000674, +0.001491] |
| macro_19 | Ridge, 40 genes | Spearman increase | -0.024169 | 0/20 | [-0.067138, +0.023146] |
| macro_19 | Ridge, 40 genes | RMSE reduction | -0.001935 | 0/20 | [-0.003724, -0.000086] |
| macro_19 | Neural network, 40 genes | Spearman increase | -0.020375 | 0/20 | [-0.057532, +0.020273] |
| macro_19 | Neural network, 40 genes | RMSE reduction | -0.002313 | 0/20 | [-0.003810, -0.000819] |
| macro_19 | Naive Gaussian BN | Spearman increase | +0.005841 | 20/20 | [-0.006650, +0.019215] |
| macro_19 | Naive Gaussian BN | RMSE reduction | -0.000076 | 0/20 | [-0.001074, +0.000858] |

Among the 19 lineages with n≥10:

- TAN versus Ridge, 40 genes: higher Spearman in 8/19 and lower RMSE in 5/19.
- TAN versus Neural network, 40 genes: higher Spearman in 7/19 and lower RMSE in 3/19.
- TAN versus Naive Gaussian BN: higher Spearman in 11/19 and lower RMSE in 10/19.

![Lineage comparison](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_bayesian_comparison/lineage_comparison.png)

## Original-seed comparison including random forest

Random forest was previously fitted for inner seed 20261031 only. This table compares every arm at that same original seed; it does not compare a one-seed forest with an averaged 20-seed predictor. The original seed is excluded from the repeated means above.

| Model / representation | Spearman ↑ | RMSE ↓ | R² ↑ | Macro Spearman ↑ |
|---|---:|---:|---:|---:|
| neural network / panel expression | 0.42878 | 0.120411 | 0.17837 | 0.22466 |
| ridge / panel expression | 0.42345 | 0.120881 | 0.17194 | 0.23034 |
| random forest / panel expression | 0.40043 | 0.120894 | 0.17176 | 0.23588 |
| Tree-augmented Gaussian BN | 0.38167 | 0.122823 | 0.14512 | 0.20591 |
| Naive Gaussian BN | 0.37836 | 0.123291 | 0.13859 | 0.20006 |
| random forest / panel contextual embedding | 0.35326 | 0.123326 | 0.13810 | 0.18906 |
| ridge / panel contextual embedding | 0.36953 | 0.124071 | 0.12765 | 0.23055 |
| random forest / all gene contextual embedding | 0.32844 | 0.124707 | 0.11869 | 0.19827 |
| ridge / all gene contextual embedding | 0.31727 | 0.126121 | 0.09859 | 0.18458 |
| neural network / panel contextual embedding | 0.35033 | 0.128955 | 0.05762 | 0.26362 |
| neural network / all gene contextual embedding | 0.28205 | 0.130622 | 0.03310 | 0.20415 |

## Regularization and the preserved initial run

The initial grid was [0, .01, .03, .1, .2, .4, .6, .8]. All 504 naive fits and 483/504 TAN fits selected the upper boundary by inner validation. After seeing that tuning-boundary problem (and the initial pooled test scores), a documented amendment added [.9, .95, .99, 1]. The reason was the inner-selection boundary, not a target held-out score. The revised grid was fixed before its results were scored. The full original run is preserved in the results archive; all original candidate validation predictions were reproduced exactly. The extension left every selected model and held-out prediction unchanged; the final selected values are inside the extended range. This remains an exploratory follow-up on an already examined cohort.

| Model | Selected shrinkage | Number of follow-up outer fits |
|---|---:|---:|
| Naive Gaussian BN | 0.8 | 480/480 |
| Tree-augmented Gaussian BN | 0.6 | 20/480 |
| Tree-augmented Gaussian BN | 0.8 | 460/480 |

Distinct complete held-out prediction vectors among the 20 repeats: naive = 1; TAN = 1. Deterministic fits can repeat exactly when inner tuning selects the same settings. This is stability under tuning assignments, not twenty independently trained stochastic models or independent biological replications.

## Predictive uncertainty

The Bayesian networks also return a Gaussian conditional variance. These intervals use fitted distribution parameters; they do not integrate parameter/graph uncertainty and were not recalibrated on the held-out responses.

| Model | Mean Gaussian NLL ↓ | Coverage of nominal 90% interval | Mean interval width |
|---|---:|---:|---:|
| Naive Gaussian BN | -0.6692 | 92.9% | 0.4277 |
| Tree-augmented Gaussian BN | -0.6719 | 93.8% | 0.4276 |

## Learned gene dependencies

The network is learned from the training expression and sensitivity values, not imported from a protein-interaction database. The response is a root in the probabilistic factorization; this arrow choice does not claim that drug sensitivity causes gene expression. Likewise, the orientation of the gene tree is arbitrary for Gaussian inference. Gene pairs below are therefore reported without causal direction.

| Gene pair | Selection frequency across 480 TAN fits |
|---|---:|
| ACSL3 — MBOAT1 | 100.0% |
| ACSL3 — NFE2L2 | 100.0% |
| ACSL3 — SCD | 100.0% |
| ACSL4 — LAMP2 | 100.0% |
| CTSD — ABHD12 | 100.0% |
| CTSD — CYB5R1 | 100.0% |
| CTSD — LAMP2 | 100.0% |
| CTSD — POR | 100.0% |
| DHFR — HSPA8 | 100.0% |
| GCLC — GSR | 100.0% |
| GSS — NFS1 | 100.0% |
| LPCAT3 — SREBF1 | 100.0% |
| NADK — NGLY1 | 100.0% |
| PLA2G6 — ATF4 | 100.0% |
| PLA2G6 — HDDC3 | 100.0% |

![Gene dependency stability](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_bayesian_comparison/gene_dependency_stability.png)

Edge counts reflect overlapping training cohorts. They are descriptive stability measures, not independent confirmation of physical or regulatory interactions.

## Method and interpretation limits

- TAN uses a maximum spanning tree of Gaussian conditional mutual information between genes given response, then linear Gaussian conditional distributions at the nodes. This is a continuous-response adaptation of the [TAN factorization](https://dang.cs.technion.ac.il/journal_papers/friedman1997Bayesian.pdf); the original paper evaluates classification.
- Gaussian networks have a joint Gaussian representation and permit exact response conditioning; their conditional means are linear in the observed expression. See the [linear Gaussian network documentation](https://pgmpy.org/api/generated/models/pgmpy.models.LinearGaussianBayesianNetwork.html). The implementation is saved here and uses NumPy/scikit-learn; pgmpy was not installed or used.
- Covariance shrinkage uses S=(1−λ)C+λI after training-only standardization. The [diagonal shrinkage form](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ledoit_wolf.html) is familiar, but λ here is chosen by inner RMSE, not by the Ledoit–Wolf estimator.
- These are Bayesian-network factorizations with estimated parameters, not a full Bayesian posterior over structures and parameters. The comparison does not test nonlinear, mixture, discrete or biologically constrained Bayesian networks.
- Each outer holdout tunes λ using the same five grouped inner folds as prior models. These inner folds mix represented training lineages; this study does not add lineage-held-out inner tuning.
- Pooled metrics include differences between lineage means. Macro metrics weight the same 19 larger lineages equally. A constant within-lineage predictor has undefined Spearman; if it occurs, macro Spearman remains undefined rather than dropping that lineage.
- Confidence intervals resample 24 lineages for pooled metrics and 19 for macro metrics. All arms and repeats remain paired; models are not refitted. Training sets overlap, the repeated samples are not independent datasets, and exploratory comparisons are not multiplicity adjusted.
- The same archived CTRP labels and DepMap 24Q4 inputs are reused. No external drug screen or clinical cohort has been added. BulkFormer reference arms retain the earlier unresolved pretraining-overlap limitation.

## Audit and files

All 1008 selected Bayesian-network fits were audited, including training-only scalers, identity exclusion, acyclic graphs, positive-definite covariances, exact prediction reconstruction and independent covariance-based conditioning. All 12096 candidate loss sets were reconstructed. For original/first/last seeds in Lung and Thyroid, every candidate was independently refitted with networkx tree learning, augmented-data least squares CPDs and separate covariance conditioning. Known-network synthetic checks verify the exact conditional mean/variance, tree-root invariance and naive-network formula.

- [Repeated score table](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_bayesian_comparison/repeated_summary.tsv)
- [Original-seed comparison including random forest](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_bayesian_comparison/original_seed_summary.tsv)
- [Paired repeated comparisons](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_bayesian_comparison/paired_repeated_comparisons.tsv)
- [Gene edge stability](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_bayesian_comparison/gene_edge_stability.tsv)
- [Protocol](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_bayesian_comparison/protocol.json)
- [Validation](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_bayesian_comparison/validation.json)
