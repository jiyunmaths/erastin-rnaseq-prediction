# Importance of the 24 added genes in the 64-gene neural network

**Top genes by held-out predictive benefit: ABCC1, FADS1, GOT1, FAR1, FADS2.** Top genes by prediction influence: ABCC1, FADS1, GOT1, SLC3A2, FADS2. All rankings here are among the 24 additions, not among the complete 64-gene panel.

We reused all 480 saved outer neural-network ensembles (20 repeats × 24 lineage holdouts). Each ensemble averages its three member predictions before squared error is evaluated. No models were retrained. For each gene and holdout, every receiver/donor pair within that lineage was evaluated, including self replacements. This exactly computes the expectation of additive squared-error loss over random within-lineage permutations, without permutation Monte Carlo noise.

Primary rankings weight the 721 cells in the same 19 lineages with n≥10. Equal-lineage rankings are provided as a sensitivity analysis. The five smaller lineages are descriptive only, and the singleton has no within-lineage permutation information.

| Gene | Rank | MSE increase (%) | Positive lineages | Top-five seeds | Seed rank range | Prediction slope |
|---|---:|---:|---:|---:|---|---|
| ABCC1 | 1 | +1.258 | 18/19 | 20/20 | 1–1 | higher sensitivity |
| FADS1 | 2 | +0.708 | 13/19 | 20/20 | 2–2 | higher sensitivity |
| GOT1 | 3 | +0.642 | 12/19 | 20/20 | 3–3 | higher sensitivity |
| FAR1 | 4 | +0.351 | 11/19 | 20/20 | 4–4 | higher sensitivity |
| FADS2 | 5 | +0.343 | 13/19 | 20/20 | 5–5 | higher sensitivity |
| AGPS | 6 | +0.273 | 11/19 | 0/20 | 6–6 | lower sensitivity |
| SLC3A2 | 7 | +0.194 | 10/19 | 0/20 | 7–7 | lower sensitivity |
| BECN1 | 8 | +0.183 | 12/19 | 0/20 | 8–8 | higher sensitivity |
| PRDX6 | 9 | +0.067 | 10/19 | 0/20 | 9–9 | lower sensitivity |
| IREB2 | 10 | +0.035 | 14/19 | 0/20 | 10–10 | higher sensitivity |
| SEPHS2 | 11 | +0.006 | 11/19 | 0/20 | 11–11 | higher sensitivity |
| FTH1 | 12 | -0.014 | 11/19 | 0/20 | 12–12 | higher sensitivity |
| SPR | 13 | -0.034 | 8/19 | 0/20 | 13–13 | lower sensitivity |
| CTH | 14 | -0.044 | 9/19 | 0/20 | 14–14 | higher sensitivity |
| PTS | 15 | -0.080 | 4/19 | 0/20 | 15–15 | higher sensitivity |
| TMEM164 | 16 | -0.094 | 5/19 | 0/20 | 16–16 | higher sensitivity |
| GLS2 | 17 | -0.101 | 7/19 | 0/20 | 17–18 | lower sensitivity |
| CBS | 18 | -0.104 | 8/19 | 0/20 | 17–18 | higher sensitivity |
| BAP1 | 19 | -0.149 | 10/19 | 0/20 | 19–20 | lower sensitivity |
| PROM2 | 20 | -0.151 | 8/19 | 0/20 | 19–20 | lower sensitivity |
| GNPAT | 21 | -0.220 | 9/19 | 0/20 | 21–21 | higher sensitivity |
| SLC40A1 | 22 | -0.228 | 6/19 | 0/20 | 22–23 | lower sensitivity |
| ELOVL5 | 23 | -0.231 | 7/19 | 0/20 | 22–23 | lower sensitivity |
| GCH1 | 24 | -0.253 | 6/19 | 0/20 | 24–24 | lower sensitivity |

MSE increase (%) means the expected MSE change after shuffling divided by the unshuffled baseline MSE. Positive importance means disrupting the gene worsens held-out accuracy. Negative importance means the disruption improves accuracy on this cohort. Influence is the RMS prediction change and can be large even for a gene with unhelpful error contributions. Direction is the mean local derivative per training SD of expression; the response is negative erastin AUC, so higher predicted response means greater sensitivity.

![Importance](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_added_gene_importance/added_gene_importance.png)

## Consistency across seeds and lineages

- Median pairwise rank correlation between seed-level aggregate rankings: 0.999.
- Median pairwise rank correlation between lineage-specific importance rankings: 0.077.
- Median lineage-to-lineage rank correlation for prediction influence: 0.786.
- Pointwise conditional 95% intervals entirely above zero: ABCC1.

Stable aggregate ranks across seeds reflect stability of these fitted models on the same cohort. They do not imply stable importance across cancer lineages or independent biological replication. The per-lineage table includes positive-repeat counts, rank ranges, direction and influence for every gene.

![Lineage importance](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_added_gene_importance/lineage_gene_importance.png)

## Weighting and lineage exceptions

Giving each lineage equal weight changes the top five to ABCC1, FADS1, BECN1, AGPS, FAR1. The first two genes remain ABCC1, FADS1, while the rest of the ranking depends more on lineage weighting.

The leading gene, ABCC1, has negative predictive importance in Bowel. Negative importance means its shuffled predictions are more accurate on those held-out cells; it does not establish an opposite biological gene effect.

## Joint contribution of the added block

- weighted_19: joint permutation of the 24 additions changes MSE by +0.000392 (+2.72%; conditional 95% interval [-0.000083, +0.000711]); positive importance in 10/19 lineages.
- macro_19: joint permutation of the 24 additions changes MSE by +0.000074 (+0.50%; conditional 95% interval [-0.000435, +0.000534]); positive importance in 10/19 lineages.

The joint permutation preserves dependence among the added genes while disrupting their links to the original 40 genes and response. It measures reliance of the fitted 64-gene model on this block. It does not decompose the improvement over a separately trained 40-gene model, and individual importance values must not be summed as independent contributions.

## Limits and validation

- All within-lineage substitutions preserve the marginal expression distribution of the shuffled gene. They can disrupt correlations with other genes and produce atypical combinations. The strongest within-lineage correlated partner of each addition is reported separately.
- Low or negative importance may reflect redundant information, regularization, limited within-lineage variation or estimation error. It is not evidence of biological irrelevance or a causal gene effect.
- Bootstrap intervals resample whole eligible lineages, condition on the saved models and donor distributions, and do not refit models. Intervals and rankings are exploratory and not adjusted for multiple comparisons.
- The same cohort has supported several model comparisons; no external screen or clinical validation was added.
- All 480 baseline predictions reconstructed (maximum error 0). Native replacement checks, finite-difference gradients and exhaustive two-cell permutations passed.
- Maximum native swap prediction discrepancy: 3.33e-16; maximum gradient discrepancy: 1.8e-11.

## Files

- [All gene rankings and intervals](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_added_gene_importance/gene_importance_summary.tsv)
- [Per-lineage results](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_added_gene_importance/per_lineage_importance.tsv)
- [Per-seed results](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_added_gene_importance/per_repeat_importance.tsv)
- [Correlated genes](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_added_gene_importance/strongest_gene_correlations.tsv)
- [Protocol](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_added_gene_importance/protocol.json)
- [Validation](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_added_gene_importance/validation.json)
