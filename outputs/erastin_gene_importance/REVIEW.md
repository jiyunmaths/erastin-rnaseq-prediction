# Gene importance for erastin sensitivity prediction

Both models prioritize SLC7A11, ACSL4 and AIFM2 in that order. GPX4 and CYB5R1 complete the same top-five set. The full 40-gene importance rankings agree strongly between models (Spearman 0.985; 9/10 genes shared in their top tens). Rankings are much less uniform across lineages.

This analysis reuses every saved ridge fit and three-member neural ensemble from the 20 matched repeats. The 24 outer leave-lineage-out tests, fitted weights, hyperparameters, and expression panel are unchanged. No models were retrained, and no genes were selected using these importance results.

## How importance was measured

For each saved model, replace one gene at a time using expression values from donors in the same held-out lineage. Calculate the increase in mean squared prediction error relative to the unchanged predictions. All receiver/donor pairs are evaluated, including self-replacement. For additive squared error, this is exactly the expected loss over all random within-lineage permutations; it has no permutation Monte Carlo noise. Neural member predictions are averaged before scoring the ensemble.

Positive importance means shuffling the gene worsens prediction, so that gene helps the fitted model on those held-out cells. Negative importance means shuffling improves prediction. This measures model reliance and usefulness, not causal gene effects. The perturbation preserves each lineage’s marginal expression distribution, so the primary importance measures within-lineage predictive value and does not attribute predictions arising solely from differences between lineage means.

Primary rankings weight cells equally across the same 19 lineages with n≥10 (721 lines), then average effects across all 20 repeats. Equal-weight lineage rankings and a descriptive analysis of all 23 non-singleton lineages (744 lines) are also supplied. The singleton Cervix lineage has no within-lineage permutation information.

## Leading genes

Importance is expected MSE increase ×1,000. “Helpful lineages” counts positive mean importance across 20 repeats; it is a descriptive count, not a significance threshold. Directions use ridge coefficients and mean local neural slopes, each standardized to a one-training-SD increase in that gene. The target is negative erastin AUC: higher means greater sensitivity.

| Gene | Ridge rank | Ridge importance | NN rank | NN importance | Helpful lineages, ridge / NN | Higher expression predicts |
|---|---:|---:|---:|---:|---:|---|
| SLC7A11 | 1 | 0.714 | 1 | 0.538 | 16/19 / 15/19 | Lower sensitivity |
| ACSL4 | 2 | 0.376 | 2 | 0.310 | 14/19 / 13/19 | Greater sensitivity |
| AIFM2 | 3 | 0.250 | 3 | 0.231 | 13/19 / 13/19 | Lower sensitivity |
| GPX4 | 4 | 0.199 | 5 | 0.144 | 17/19 / 17/19 | Lower sensitivity |
| CYB5R1 | 5 | 0.177 | 4 | 0.152 | 15/19 / 15/19 | Lower sensitivity |
| LPCAT3 | 6 | 0.157 | 6 | 0.128 | 13/19 / 13/19 | Greater sensitivity |
| DHFR | 7 | 0.121 | 7 | 0.115 | 11/19 / 11/19 | Greater sensitivity |
| LRP8 | 8 | 0.121 | 8 | 0.104 | 13/19 / 13/19 | Lower sensitivity |
| GCLC | 10 | 0.092 | 9 | 0.093 | 11/19 / 9/19 | Lower sensitivity |
| LAMP2 | 9 | 0.115 | 11 | 0.081 | 11/19 / 11/19 | Lower sensitivity |

![Importance and conditional intervals](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_gene_importance/gene_importance.png)

## Consistency across seeds and lineages

- **Ridge across seeds:** top-five gene rank ranges: SLC7A11 #1–1, ACSL4 #2–2, AIFM2 #3–3, GPX4 #4–4, CYB5R1 #5–5.
- **Neural network across seeds:** top-five gene rank ranges: SLC7A11 #1–1, ACSL4 #2–2, AIFM2 #3–3, CYB5R1 #4–4, GPX4 #5–5.
- **Ridge across lineages:** median pairwise correlation of the 40-gene importance rankings is 0.065 (interquartile range -0.079–0.254). These 171 lineage-pair correlations are descriptive and dependent.
- **Neural network across lineages:** median pairwise correlation of the 40-gene importance rankings is 0.071 (interquartile range -0.085–0.233). These 171 lineage-pair correlations are descriptive and dependent.

Within a given lineage, ridge and neural gene rankings agree much more closely: median rank correlation 0.982. Model agreement does not imply a universal ranking across cancer types.

| Gene | Ridge slope sign across 19 lineage means | NN slope sign across 19 lineage means |
|---|---:|---:|
| SLC7A11 | 0 positive / 19 negative | 0 positive / 19 negative |
| ACSL4 | 19 positive / 0 negative | 19 positive / 0 negative |
| AIFM2 | 0 positive / 19 negative | 0 positive / 19 negative |
| GPX4 | 0 positive / 19 negative | 0 positive / 19 negative |
| CYB5R1 | 0 positive / 19 negative | 0 positive / 19 negative |

For example, SLC7A11 ranks first in Lung and CNS/Brain in both models, but has negative importance in Myeloid, Bone and Soft Tissue. GPX4 has a smaller overall effect but positive importance in 17/19 lineages in both models. Differences between lineage rankings can reflect biological differences, limited sample size, measurement noise, and model error; this analysis does not separate those sources.

| Held-out lineage | n | Ridge top three | Neural-network top three |
|---|---:|---|---|
| Lung | 121 | SLC7A11, AIFM2, ACSL4 | SLC7A11, GCLC, AIFM2 |
| Lymphoid | 95 | ACSL4, CYB5R1, GPX4 | ACSL4, CYB5R1, CTSD |
| Esophagus/Stomach | 55 | SLC7A11, ACSL4, GPX4 | ACSL4, SLC7A11, AIFM2 |
| CNS/Brain | 47 | SLC7A11, LAMP2, AIFM2 | SLC7A11, LAMP2, CTSD |
| Bowel | 44 | SLC7A11, GPX4, GCLC | SLC7A11, GCLC, GSR |
| Myeloid | 41 | MBOAT2, ACSL3, ACSL4 | ACSL4, LPCAT3, MBOAT2 |
| Skin | 40 | SLC7A11, AIFM2, ACSL3 | AIFM2, SLC7A11, ACSL3 |
| Pancreas | 38 | ACSL4, SLC7A11, CNDP2 | ACSL4, SLC7A11, CNDP2 |
| Ovary/Fallopian Tube | 37 | SLC7A11, AIFM2, ACSL4 | SLC7A11, AIFM2, ACSL4 |
| Breast | 34 | SLC7A11, CNDP2, GPX4 | SLC7A11, CNDP2, GPX4 |
| Head and Neck | 29 | ACSL4, DHFR, LPCAT3 | ACSL4, DHFR, LPCAT3 |
| Uterus | 26 | AIFM2, DHFR, CNDP2 | AIFM2, DHFR, CNDP2 |
| Kidney | 22 | LPCAT3, AIFM2, SLC7A11 | AIFM2, LPCAT3, CTSD |
| Liver | 22 | ACSL4, DHFR, LPCAT3 | ACSL4, DHFR, CYB5R1 |
| Bladder/Urinary Tract | 19 | ACSL4, SLC7A11, CYB5R1 | ACSL4, SLC7A11, LRP8 |
| Peripheral Nervous System | 15 | SLC7A11, LAMP2, AIFM2 | AIFM2, LAMP2, SLC7A11 |
| Bone | 13 | ACSL3, LPCAT3, LAMP2 | ACSL3, LPCAT3, LAMP2 |
| Soft Tissue | 13 | ACSL3, LRP8, ACSL4 | ACSL3, LRP8, ACSL4 |
| Thyroid | 10 | GCLC, CYB5R1, AIFM2 | GCLC, CTSD, CYB5R1 |

![Lineage ranks](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_gene_importance/lineage_gene_ranks.png)

[All 40 genes, absolute importance heatmap](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_gene_importance/lineage_gene_heatmap.png)

## Weighting and uncertainty

| Summary | Ridge top five | Neural-network top five |
|---|---|---|
| weighted_19_lineages | SLC7A11, ACSL4, AIFM2, GPX4, CYB5R1 | SLC7A11, ACSL4, AIFM2, CYB5R1, GPX4 |
| macro_19_lineages | SLC7A11, ACSL4, AIFM2, GPX4, LRP8 | SLC7A11, ACSL4, AIFM2, LRP8, CYB5R1 |
| weighted_23_nonsingleton_lineages | SLC7A11, ACSL4, AIFM2, GPX4, CYB5R1 | SLC7A11, ACSL4, AIFM2, CYB5R1, GPX4 |

Intervals use 5,000 paired resamples of the 19 complete held-out lineages, preserving every gene, model and repeat together. They condition on the saved models and observed donor distributions; models are not refitted. The intervals are pointwise and exploratory, with no adjustment for examining 40 genes. Training sets overlap between holdouts, and neither repeats nor positive-lineage counts are independent biological replications.

- Ridge genes with pointwise primary intervals entirely above zero: SLC7A11, ACSL4, AIFM2, GPX4, CYB5R1, LPCAT3, LRP8.
- Neural network genes with pointwise primary intervals entirely above zero: SLC7A11, ACSL4, AIFM2, CYB5R1, GPX4, LPCAT3, LRP8.

## Interpretation limits

- The ranking describes these two predictive models, the 40 measured genes, and this archived erastin-response cohort. It does not prove a gene causes erastin resistance or sensitivity.
- Shuffling a gene breaks its correlation with other genes and may generate atypical expression combinations. Correlated genes may share predictive information, so small importance is not evidence of biological irrelevance. The full within-lineage-centered expression correlation matrix is supplied.
- Direction is a partial model response with other features fixed. A consistent coefficient or local-gradient sign can coexist with negative held-out importance in some lineages because predictive calibration and gene–response relationships vary.
- Small lineages provide imprecise individual assessments. The n≥10 cutoff and the same 19 primary lineages are inherited from the preceding comparisons, not chosen after seeing gene scores.
- Gene importance is not additive across features and cannot be read as a percentage of explained erastin sensitivity. No knockout, causal mediation, external drug screen, or clinical validation was performed.
- This analysis uses the 40-gene expression panel only, not BulkFormer embeddings.

## Validation and files

All 960 saved model/holdout combinations reproduced their stored predictions (maximum absolute error 1.1e-16). Native sklearn replacement predictions were checked for all 40 genes in every holdout (maximum error 3.3e-16). Neural analytical gradients matched independent finite differences (maximum error 1.7e-11). All ridge exact-permutation results additionally matched a closed-form loss calculation; exhaustive permutations of the two-cell lineage validated the all-donor identity. Training-only scalers, held-out identity exclusion, every model hash, and source-data hashes were checked.

- [All-gene summary, ranks and intervals](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_gene_importance/gene_importance_summary.tsv)
- [Per-lineage gene importance and direction](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_gene_importance/per_lineage_importance.tsv)
- [Per-repeat importance and ranks](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_gene_importance/per_repeat_importance.tsv)
- [Within-lineage gene correlations](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_gene_importance/within_lineage_gene_correlations.tsv)
- [Registered analysis specification](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_gene_importance/protocol.json)
- [Validation](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_gene_importance/validation.json)
