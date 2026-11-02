# Gene importance in the lineage-stratified 64-gene models

**Top genes by held-out predictive benefit:** Ridge: SLC7A11, GPX4, ABCC1, ACSL4, AIFM2. Neural network: SLC7A11, ACSL4, GPX4, AIFM2, ABCC1.

Reused all 20 unweighted 64-gene ridge models and all 20 saved three-member neural-network ensembles. No models were retrained. The cohort and splits are unchanged: Cervix excluded, 744 lines, 23 represented lineages, 595 training/149 testing per split, and no related identity group crossing train/test. Ranks below are among the complete 64-gene panel.

## Definition of predictive importance

For each held-out line and each gene, replace that gene’s expression with every available **same-lineage training-line value**, holding the other genes fixed. Average the resulting squared-error increase over training donors, held-out receivers, and the 20 splits. All receiver/donor pairs are evaluated exactly; no permutation Monte Carlo approximation is used. The neural ensemble prediction is averaged before calculating error. Positive importance means replacement worsens held-out accuracy; negative importance means replacement improves it on this cohort.

This is a **training-reference donor-replacement measure**, not ordinary test-set permutation importance. It allows estimates even when a lineage has just one test line, but does not create extra biological information for those tiny lineages. Donors use training expression only and never test responses. Ordinary within-test permutation importance is provided separately wherever at least two test lines exist.

Primary aggregate rankings weight the 149 test receivers equally within each split, then average split-level effects. Equal-lineage and larger-lineage sensitivity analyses are saved. Cross-lineage consistency is summarized for the 19 lineages with cohort n≥10; a stricter analysis uses the 12 lineages with at least five test lines per split. All 23 lineages are tabulated, including the four smaller, descriptive lineages.

## Ridge predictive ranking

| Rank | Gene | Panel membership | MSE increase (%) | Conditional 95% interval (%) | Positive lineages /19 | Positive splits /20 | Top-ten splits /20 | Seed rank range | Direction |
|---:|---|---|---:|---|---:|---:|---:|---|---|
| 1 | SLC7A11 | Original 40 | +4.222 | [+1.231, +7.285] | 13/19 | 20/20 | 20/20 | 1–9 | Lower sensitivity |
| 2 | GPX4 | Original 40 | +1.538 | [+0.552, +2.404] | 14/19 | 18/20 | 15/20 | 1–60 | Lower sensitivity |
| 3 | ABCC1 | Added 24 | +1.442 | [+0.328, +2.428] | 15/19 | 18/20 | 17/20 | 1–63 | Higher sensitivity |
| 4 | ACSL4 | Original 40 | +1.414 | [+0.244, +2.369] | 12/19 | 19/20 | 14/20 | 1–60 | Higher sensitivity |
| 5 | AIFM2 | Original 40 | +1.265 | [+0.276, +2.207] | 14/19 | 18/20 | 16/20 | 2–61 | Lower sensitivity |
| 6 | GOT1 | Added 24 | +0.809 | [+0.133, +1.437] | 12/19 | 17/20 | 14/20 | 3–63 | Higher sensitivity |
| 7 | ACSL3 | Original 40 | +0.743 | [-0.327, +1.844] | 11/19 | 18/20 | 7/20 | 4–58 | Lower sensitivity |
| 8 | FADS1 | Added 24 | +0.723 | [-0.192, +1.385] | 14/19 | 17/20 | 11/20 | 2–62 | Higher sensitivity |
| 9 | LPCAT3 | Original 40 | +0.666 | [-0.244, +1.485] | 11/19 | 17/20 | 9/20 | 3–58 | Higher sensitivity |
| 10 | LRP8 | Original 40 | +0.662 | [+0.055, +1.331] | 15/19 | 16/20 | 9/20 | 3–63 | Lower sensitivity |
| 11 | DHFR | Original 40 | +0.629 | [-0.120, +1.304] | 13/19 | 18/20 | 9/20 | 5–58 | Higher sensitivity |
| 12 | GSR | Original 40 | +0.621 | [-0.243, +1.439] | 11/19 | 17/20 | 8/20 | 3–62 | Lower sensitivity |
| 13 | LAMP2 | Original 40 | +0.607 | [-0.341, +1.438] | 10/19 | 17/20 | 8/20 | 5–63 | Lower sensitivity |
| 14 | CYB5R1 | Original 40 | +0.575 | [-0.254, +1.241] | 14/19 | 16/20 | 10/20 | 3–64 | Lower sensitivity |
| 15 | CNDP2 | Original 40 | +0.544 | [-0.070, +1.260] | 12/19 | 17/20 | 6/20 | 4–62 | Higher sensitivity |

Genes with pointwise conditional intervals entirely above zero: SLC7A11, GPX4, ABCC1, ACSL4, AIFM2, GOT1, LRP8.

## Neural network predictive ranking

| Rank | Gene | Panel membership | MSE increase (%) | Conditional 95% interval (%) | Positive lineages /19 | Positive splits /20 | Top-ten splits /20 | Seed rank range | Direction |
|---:|---|---|---:|---|---:|---:|---:|---|---|
| 1 | SLC7A11 | Original 40 | +3.837 | [+1.022, +6.682] | 13/19 | 20/20 | 20/20 | 1–7 | Lower sensitivity |
| 2 | ACSL4 | Original 40 | +1.364 | [+0.255, +2.284] | 12/19 | 18/20 | 14/20 | 1–57 | Higher sensitivity |
| 3 | GPX4 | Original 40 | +1.338 | [+0.454, +2.117] | 14/19 | 18/20 | 15/20 | 1–57 | Lower sensitivity |
| 4 | AIFM2 | Original 40 | +1.236 | [+0.277, +2.127] | 14/19 | 18/20 | 16/20 | 2–61 | Lower sensitivity |
| 5 | ABCC1 | Added 24 | +1.207 | [+0.199, +2.084] | 14/19 | 18/20 | 16/20 | 2–64 | Higher sensitivity |
| 6 | GOT1 | Added 24 | +0.716 | [+0.090, +1.292] | 12/19 | 17/20 | 11/20 | 3–63 | Higher sensitivity |
| 7 | ACSL3 | Original 40 | +0.700 | [-0.323, +1.740] | 11/19 | 18/20 | 8/20 | 4–60 | Lower sensitivity |
| 8 | LRP8 | Original 40 | +0.692 | [+0.086, +1.352] | 13/19 | 16/20 | 11/20 | 3–63 | Lower sensitivity |
| 9 | GSR | Original 40 | +0.688 | [-0.211, +1.555] | 11/19 | 17/20 | 9/20 | 3–62 | Lower sensitivity |
| 10 | DHFR | Original 40 | +0.680 | [-0.059, +1.340] | 13/19 | 18/20 | 11/20 | 4–57 | Higher sensitivity |
| 11 | FADS1 | Added 24 | +0.675 | [-0.155, +1.256] | 14/19 | 17/20 | 9/20 | 2–62 | Higher sensitivity |
| 12 | LPCAT3 | Original 40 | +0.583 | [-0.236, +1.329] | 11/19 | 16/20 | 7/20 | 3–56 | Higher sensitivity |
| 13 | LAMP2 | Original 40 | +0.562 | [-0.326, +1.333] | 10/19 | 17/20 | 9/20 | 5–62 | Lower sensitivity |
| 14 | CYB5R1 | Original 40 | +0.542 | [-0.252, +1.185] | 14/19 | 15/20 | 11/20 | 4–64 | Lower sensitivity |
| 15 | CNDP2 | Original 40 | +0.496 | [-0.080, +1.144] | 11/19 | 17/20 | 6/20 | 4–59 | Higher sensitivity |

Genes with pointwise conditional intervals entirely above zero: SLC7A11, ACSL4, GPX4, AIFM2, ABCC1, GOT1, LRP8.


MSE increase (%) divides the expected error increase by the unshuffled model’s baseline MSE. Intervals use 5,000 paired whole-lineage draws, conditional on the saved models and donor distributions; they do not refit models and are not adjusted for 64 genes or other comparisons. Direction is the mean model derivative per training SD of expression. The target is negative erastin AUC: a positive derivative predicts greater sensitivity. It is not a causal gene-perturbation effect.

![Top predictive genes](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_stratified_gene_importance/top_predictive_genes.png)

## Consistency across lineages and splits

| Model | Importance measure | Unit | Median rank correlation | Interquartile range |
|---|---|---|---:|---|
| Ridge | predictive_importance | between_splits | 0.369 | [0.251, 0.493] |
| Ridge | prediction_influence | between_splits | 0.784 | [0.759, 0.811] |
| Ridge | standardized_sensitivity | between_splits | 0.787 | [0.764, 0.816] |
| Neural network | predictive_importance | between_splits | 0.364 | [0.240, 0.490] |
| Neural network | prediction_influence | between_splits | 0.790 | [0.761, 0.812] |
| Neural network | standardized_sensitivity | between_splits | 0.792 | [0.762, 0.814] |
| Ridge | predictive_importance | between_lineages19 | 0.036 | [-0.096, 0.179] |
| Ridge | prediction_influence | between_lineages19 | 0.882 | [0.857, 0.913] |
| Ridge | standardized_sensitivity | between_lineages19 | 1.000 | [1.000, 1.000] |
| Ridge | predictive_importance | between_lineages12 | 0.056 | [-0.039, 0.203] |
| Neural network | predictive_importance | between_lineages19 | 0.031 | [-0.100, 0.167] |
| Neural network | prediction_influence | between_lineages19 | 0.888 | [0.862, 0.917] |
| Neural network | standardized_sensitivity | between_lineages19 | 1.000 | [1.000, 1.000] |
| Neural network | predictive_importance | between_lineages12 | 0.054 | [-0.034, 0.210] |

Correlation is over all 64 gene ranks. Split-level rankings vary with held-out responses, training donor distributions and fitted models; the 20 splits reuse overlapping samples. Low lineage-to-lineage predictive-rank agreement indicates context-dependent error contributions, even when model coefficients or sensitivity rankings agree. The primary ranking can therefore be reproducible in aggregate without implying that every lineage uses the same genes for accurate prediction.

| Comparison | Measure | Rank correlation |
|---|---|---:|
| Ridge vs neural network | predictive_importance | 0.991 |
| Ridge vs neural network | prediction_influence | 0.991 |
| Ridge vs neural network | standardized_sensitivity | 0.989 |

![Lineage ranks](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_stratified_gene_importance/lineage_gene_ranks.png)

## Model weights and local sensitivity

The ridge model has **one coefficient vector per split, applied to every lineage**. It has no independently fitted lineage-specific coefficient vector. Its lineage-specific predictive importance differs because expression variation and residual prediction errors differ across held-out lineages. The neural network has context-dependent derivatives; ensemble gradients are averaged before summarizing.

| Model | Top ten genes by RMS standardized coefficient/derivative |
|---|---|
| Ridge | SLC7A11, ACSL4, GPX4, ABCC1, AIFM2, CYB5R1, ACSL3, LAMP2, FADS1, GSR |
| Neural network | SLC7A11, ACSL4, AIFM2, GPX4, ABCC1, CYB5R1, ACSL3, LAMP2, CTSD, GSR |

A large coefficient/derivative or RMS prediction change measures model influence. It does not guarantee useful held-out predictive signal. Per-gene signed coefficients, local gradient summaries, prediction-change ranks and sign stability are saved separately from the error-benefit ranking.

![Standardized sensitivity](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_stratified_gene_importance/standardized_sensitivity.png)

## Donor-reference and lineage-weighting checks

| Model | Training-reference vs test-permutation rank correlation | Top-ten overlap |
|---|---:|---:|
| Ridge | 0.977 | 8/10 |
| Neural network | 0.979 | 9/10 |

This check uses the same 20 lineages with at least two test lines for both methods. Ordinary test permutations include self replacements and use much smaller donor pools; the two measures need not be identical. Ampulla, Biliary and Prostate have one test line per split and provide no within-test permutation information; their training-reference results remain descriptive.

| Model | Cell-weighted all-23 top five | Equal-lineage all-23 top five | Cell-weighted larger-19 top five |
|---|---|---|---|
| Ridge | SLC7A11, GPX4, ABCC1, ACSL4, AIFM2 | GPX4, SLC7A11, ACSL4, ACSL3, ABCC1 | SLC7A11, ABCC1, ACSL4, GPX4, AIFM2 |
| Neural network | SLC7A11, ACSL4, GPX4, AIFM2, ABCC1 | GPX4, SLC7A11, ACSL4, ACSL3, AIFM2 | SLC7A11, ACSL4, AIFM2, ABCC1, GPX4 |

## Validation and limits

- All 40 saved models reproduced their held-out predictions; maximum error 0. Input/model source hashes and training scalers were verified.
- Native complete-input replacement checks covered every gene, lineage and split; maximum error 3.33e-16. All training donor identities are disjoint from test receiver identities.
- Analytical gradients matched independent finite differences; maximum discrepancy 1.72e-11. Exact self-replacement checks passed.
- Exhaustive native permutations for two- and three-line test subsets matched the exact expected secondary loss; maximum error 2.78e-17.
- Selected full receiver/training-donor matrices were verified by native prediction for Lung, Thyroid and Ampulla in splits 1/20; maximum error 3.33e-16.
- Cell-level and lineage-level aggregations matched; maximum error 7.59e-19. Explicit lineage-bootstrap expansion matched within 2.17e-19.
- Replacement preserves the observed lineage context and one-gene donor distribution but disrupts correlations with the other 63 genes. Atypical expression combinations and redundancy can alter importance. Strongest measured gene correlations are saved.
- High importance of an added gene does not isolate its contribution to the improvement from retraining a 40-gene model as a 64-gene model. Individual importance values are not additive independent contributions.
- Negative importance is not evidence of an opposite biological mechanism; low importance is not evidence of biological irrelevance. Derivatives describe fitted associations, not gene knockout or overexpression outcomes.
- The same cell-line cohort has already supported panel/model exploration. Repeated splits and reused test samples are not independent biological validations. No external screen or causal intervention was evaluated.

## Files

- gene_importance_summary.tsv: all 64 genes, both models, four aggregation scopes, conditional intervals, direction and stability.
- per_lineage_importance.tsv, lineage_information.tsv and per_repeat_importance.tsv: lineage/split estimates and information limits.
- ridge_coefficients_by_split.tsv and standardized_sensitivity_summary.tsv: direct coefficients and neural derivatives.
- rank_correlation_summary.tsv, rank_correlations.tsv and cross_model_rank_correlations.tsv: agreement across lineages, splits and models.
- donor_reference_sensitivity.tsv and donor_reference_sensitivity_summary.tsv: matched test-permutation checks.
- strongest_gene_correlations.tsv: descriptive pooled and within-lineage-centered expression correlations.
- importance_arrays.npz, bootstrap_draws.npz and repeats/: exact derived arrays and conditional draws.
- protocol.json, source_manifest.json, calculation_validation.json, validation.json and source scripts: definitions, provenance and numerical checks.
