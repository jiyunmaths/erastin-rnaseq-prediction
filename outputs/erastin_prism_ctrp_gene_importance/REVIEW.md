# PRISM versus CTRP: 64-gene model importance and lineage consistency

**AIFM2, SLC7A11 and ACSL4 are the clearest shared high-ranking predictors across the two datasets.** AIFM2 ranks first in all three PRISM models. GPX4 and ABCC1, which were among the top five in CTRP, are substantially less useful in the 64-gene PRISM models. These are model reliance results, not causal gene effects.

**Lineage consistency is limited.** In PRISM, AIFM2 has positive held-out predictive importance in 8/13 sufficiently represented common lineages, SLC7A11 in 9/13, and ACSL4 in 7–8/13. No PRISM gene reaches the predefined descriptive threshold of positive importance in at least 11/13 common lineages. Native predictive-rank agreement between assays is low, and only partly improves after matching lineage composition or exact receiver cell lines.

## What was compared

- All **120 saved 64-gene models**: 20 ridge, 20 Elastic Net and 20 three-member NN ensembles from each dataset. No model or gene panel was refitted.
- **CTRP:** 744 lines, 23 lineages, 149 held-out lines per repeat. **PRISM:** 386 lines, 20 lineages, 78 held-out lines per repeat. Both use previously saved unweighted grouped lineage-stratified 80/20 splits, with all represented lineages on both sides.
- Identical importance definition in both datasets: replace one gene in a held-out receiver with every same-lineage outer-training donor value, then average the increase in held-out squared error. Positive values mean replacement worsens accuracy. All donor/receiver combinations are enumerated; donor choice uses no test responses.
- Main native summaries weight held-out lines within each repeat, then average repeats. Relative importance divides the mean MSE increase by the model’s own mean baseline MSE. Comparisons of raw coefficients across assays are avoided because response processing, dose intervals, training sizes and regularization differ.
- Controls: equally weight the same **13 lineages** with at least ten cohort lines in both datasets; repeat on the **290 exact same cell-line IDs** held out at least once in both studies, averaging each cell’s valid test appearances before equal lineage weighting. A stricter lineage analysis uses **seven common lineages** with at least five test lines per repeat in both datasets.

## Native full-cohort gene rankings

Ranks are shown in **ridge / Elastic Net / neural network** order. Rank 1 means the greatest positive held-out MSE increase after replacement.

| Gene | CTRP ranks | PRISM ranks | PRISM positive lineages, out of 13 |
|---|---|---|---|
| AIFM2 | 5 / 4 / 4 | 1 / 1 / 1 | 8 / 8 / 8 |
| SLC7A11 | 1 / 1 / 1 | 5 / 8 / 4 | 9 / 9 / 9 |
| ACSL4 | 4 / 2 / 2 | 6 / 4 / 7 | 8 / 7 / 8 |
| GPX4 | 2 / 3 / 3 | 47 / 44 / 48 | 7 / 5 / 3 |
| ABCC1 | 3 / 5 / 5 | 22 / 46 / 27 | 8 / 6 / 7 |
| SLC3A2 | 26 / 25 / 22 | 2 / 2 / 2 | 10 / 10 / 9 |
| SLC40A1 | 39 / 49 / 38 | 3 / 3 / 5 | 8 / 9 / 9 |
| POR | 22 / 19 / 20 | 7 / 5 / 6 | 7 / 7 / 8 |
| GCLC | 19 / 18 / 19 | 8 / 6 / 8 | 9 / 8 / 7 |

SLC3A2, SLC40A1, POR and GCLC rise in the PRISM native rankings. High point ranks alone do not establish reproducible benefit: SLC3A2, SLC40A1 and POR have conditional native importance intervals containing zero in all three PRISM models. AIFM2, ACSL4 and GCLC have positive lower interval bounds in all three; SLC7A11 does in ridge and NN. Intervals are pointwise and unadjusted.

**GPX4:** ranks 44–48 in PRISM, has negative mean native predictive importance in every model, and enters none of their top tens in any PRISM repeat. This does not imply biological irrelevance; it means these fitted expression models do not gain reliable marginal held-out predictive benefit from this feature under this replacement rule.

**ABCC1:** falls to ranks 22/46/27, with weak positive or negative mean predictive importance. Its mean coefficient/local-derivative association reverses from higher expression predicting greater sensitivity in CTRP to predicting lower sensitivity in PRISM. Correlation, shrinkage and cohort/assay differences prevent a causal interpretation of that reversal.

![Native gene ranks](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_prism_ctrp_gene_importance/native_rank_comparison.png)

## Global importance and uncertainty

Relative MSE increase is distinct from coefficient magnitude and selection frequency. Replacing a frequently selected feature can still fail to worsen held-out accuracy.

### CTRP: top ten by model

- **Ridge:** SLC7A11, GPX4, ABCC1, ACSL4, AIFM2, GOT1, ACSL3, FADS1, LPCAT3, LRP8.
- **Elastic Net:** SLC7A11, ACSL4, GPX4, AIFM2, ABCC1, FADS1, ACSL3, LAMP2, GOT1, CYB5R1.
- **Neural network:** SLC7A11, ACSL4, GPX4, AIFM2, ABCC1, GOT1, ACSL3, LRP8, GSR, DHFR.

Positive lower bounds in conditional pointwise native 95% intervals:

- **Ridge:** SLC7A11, GPX4, ABCC1, ACSL4, AIFM2, GOT1, LRP8.
- **Elastic Net:** SLC7A11, ACSL4, GPX4, AIFM2, ABCC1, GOT1.
- **Neural network:** SLC7A11, ACSL4, GPX4, AIFM2, ABCC1, GOT1, LRP8.
### PRISM: top ten by model

- **Ridge:** AIFM2, SLC3A2, SLC40A1, LRP8, SLC7A11, ACSL4, POR, GCLC, NADK, NFE2L2.
- **Elastic Net:** AIFM2, SLC3A2, SLC40A1, ACSL4, POR, GCLC, LRP8, SLC7A11, NFE2L2, NADK.
- **Neural network:** AIFM2, SLC3A2, LRP8, SLC7A11, SLC40A1, POR, ACSL4, GCLC, CYB5R1, NFE2L2.

Positive lower bounds in conditional pointwise native 95% intervals:

- **Ridge:** AIFM2, SLC7A11, ACSL4, GCLC.
- **Elastic Net:** AIFM2, ACSL4, GCLC.
- **Neural network:** AIFM2, SLC7A11, ACSL4, GCLC, CYB5R1.

![Relative importance and intervals](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_prism_ctrp_gene_importance/relative_importance_intervals.png)

## Agreement across models and assays

| Comparison within assay | CTRP rank correlation | PRISM rank correlation |
|---|---:|---:|
| Ridge vs Elastic Net | 0.9165 | 0.8000 |
| Ridge vs Neural network | 0.9915 | 0.9424 |
| Elastic Net vs Neural network | 0.9155 | 0.7718 |

| Model | CTRP–PRISM native rank correlation | Same 13 lineages, equal weight | Same 290 receiver lines, equal lineage weight |
|---|---:|---:|---:|
| Ridge | 0.1781 | 0.4082 | 0.3163 |
| Elastic Net | 0.1902 | 0.3601 | 0.3116 |
| Neural network | 0.2447 | 0.4112 | 0.3141 |

These are correlations across all 64 gene-importance values. Matching the lineage mixture improves agreement, but exact receiver matching still leaves only modest agreement. The matched-cell control uses each dataset’s own valid held-out predictions, trained models and donor distributions; it cannot isolate an assay-specific biological effect. Same numeric split seeds do not imply identical test samples in the two cohorts.

| Gene | Same-13-lineage CTRP ranks | Same-13-lineage PRISM ranks | Same-290-cell CTRP ranks | Same-290-cell PRISM ranks |
|---|---|---|---|---|
| AIFM2 | 3 / 3 / 3 | 2 / 3 / 1 | 2 / 2 / 2 | 1 / 1 / 1 |
| SLC7A11 | 1 / 1 / 1 | 4 / 6 / 5 | 1 / 1 / 1 | 5 / 7 / 5 |
| ACSL4 | 2 / 2 / 2 | 1 / 1 / 2 | 9 / 5 / 8 | 8 / 4 / 8 |
| GPX4 | 6 / 6 / 6 | 53 / 43 / 48 | 3 / 3 / 3 | 55 / 52 / 49 |
| ABCC1 | 5 / 4 / 5 | 17 / 19 / 18 | 4 / 4 / 4 | 17 / 19 / 21 |
| SLC3A2 | 20 / 18 / 19 | 8 / 2 / 4 | 16 / 15 / 16 | 2 / 2 / 2 |
| SLC40A1 | 34 / 30 / 32 | 3 / 4 / 7 | 29 / 44 / 27 | 15 / 9 / 14 |
| POR | 55 / 56 / 52 | 18 / 17 / 14 | 28 / 29 / 26 | 3 / 3 / 3 |
| GCLC | 22 / 50 / 23 | 6 / 5 / 6 | 13 / 11 / 11 | 6 / 6 / 9 |

AIFM2, SLC7A11 and ACSL4 remain top-ten predictors in both assays and all three models under both composition controls. The GPX4 drop persists after matching the exact receiver cell lines. SLC40A1’s point rank is more sensitive to the receiver/cohort weighting choice.

![Agreement controls](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_prism_ctrp_gene_importance/agreement_controls.png)

## Which effects persist across lineages?

For each lineage, average the held-out MSE replacement effect across the 20 repeats. Counts below refer to positive mean effects, not statistically significant effects in individual lineages. The common 13-lineage set avoids comparing a 19-lineage CTRP denominator with a 13-lineage PRISM denominator.

| Gene | CTRP positive lineages: R / EN / NN | PRISM positive lineages: R / EN / NN | Positive in both assays and all three models |
|---|---|---|---:|
| AIFM2 | 11 / 10 / 11 | 8 / 8 / 8 | 7/13 |
| SLC7A11 | 10 / 11 / 10 | 9 / 9 / 9 | 7/13 |
| ACSL4 | 9 / 9 / 9 | 8 / 7 / 8 | 5/13 |
| GPX4 | 10 / 9 / 10 | 7 / 5 / 3 | 1/13 |
| ABCC1 | 11 / 10 / 10 | 8 / 6 / 7 | 4/13 |
| SLC3A2 | 8 / 8 / 9 | 10 / 10 / 9 | 7/13 |
| SLC40A1 | 5 / 7 / 5 | 8 / 9 / 9 | 4/13 |
| POR | 6 / 6 / 7 | 7 / 7 / 8 | 3/13 |
| GCLC | 5 / 5 / 5 | 9 / 8 / 7 | 4/13 |

For the three shared leading genes, the lineages with positive point effects in **all six dataset/model combinations** are:

- **AIFM2:** Bladder/Urinary Tract, CNS/Brain, Esophagus/Stomach, Lung, Ovary/Fallopian Tube, Pancreas, Uterus.
- **SLC7A11:** Bladder/Urinary Tract, Breast, CNS/Brain, Lung, Ovary/Fallopian Tube, Pancreas, Skin.
- **ACSL4:** Bladder/Urinary Tract, CNS/Brain, Kidney, Liver, Ovary/Fallopian Tube.

These lists are descriptive intersections. A positive point effect in a three-line test subset is not a lineage-specific validation. Complete counts, ranks and positive-repeat frequencies for all 64 genes and all lineages are saved.

| Model | CTRP median correlation across common 13 lineage rankings | PRISM median correlation across common 13 | PRISM stricter seven |
|---|---:|---:|---:|
| Ridge | 0.0544 | 0.0453 | 0.1270 |
| Elastic Net | 0.0419 | 0.0107 | 0.0495 |
| Neural network | 0.0556 | 0.0378 | 0.0882 |

Lineage-specific predictive rankings vary substantially in both assays; the stricter seven-lineage subset does not recover uniform rankings. Conversely, sensitivity ranks are identical across lineages by construction for ridge and Elastic Net, and almost identical for the strongly regularized NN. A globally shared coefficient vector or stable derivative is not evidence that the feature improves prediction equally in every lineage.

![Lineage importance](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_prism_ctrp_gene_importance/lineage_importance_comparison.png)

## Elastic Net selection and directions

| Gene | CTRP selections | PRISM selections | CTRP mean direction | PRISM mean direction |
|---|---:|---:|---|---|
| AIFM2 | 20/20 | 20/20 | Lower sensitivity | Lower sensitivity |
| SLC7A11 | 20/20 | 16/20 | Lower sensitivity | Lower sensitivity |
| ACSL4 | 20/20 | 20/20 | Higher sensitivity | Higher sensitivity |
| GPX4 | 20/20 | 5/20 | Lower sensitivity | Lower sensitivity |
| ABCC1 | 20/20 | 9/20 | Higher sensitivity | Lower sensitivity |
| SLC3A2 | 18/20 | 20/20 | Lower sensitivity | Lower sensitivity |
| SLC40A1 | 5/20 | 18/20 | Lower sensitivity | Lower sensitivity |
| POR | 17/20 | 20/20 | Lower sensitivity | Lower sensitivity |
| GCLC | 20/20 | 17/20 | Lower sensitivity | Lower sensitivity |

Always selected in PRISM: **AIFM2, SLC3A2, POR, ACSL4**. Always selected in CTRP: SLC7A11, ACSL4, AIFM2, GPX4, ACSL3, ABCC1, LAMP2, CYB5R1, CTSD, FADS1, GCLC, FADS2, ATF4. Only AIFM2 and ACSL4 are always selected in both. Direction means the model association of higher expression with negative-AUC sensitivity, conditional on the other genes. It is not a causal perturbation direction.

## Uncertainty, validation and interpretation limits

- Source model/data hashes, gene order, sample IDs, training/test rows, saved predictions and training-only scalers were verified for both datasets. Identity groups never cross training versus test/donor roles.
- Recalculated all 120 saved-model importance profiles. Previous CTRP native rankings are reproduced exactly; maximum previous-array/score discrepancy is 4.39e-16.
- Exact NN hidden-state replacements and linear coefficient replacements were checked against native input replacement across all 64 genes and every lineage, with additional full donor-matrix checks. Maximum native error 3.33e-16; finite-difference derivative error 2.18e-11.
- Verified 900 outer/donor identity exclusions. Elastic Net selection counts/signs reproduce prior fit audits. Independent bootstrap expansion discrepancy 2.17e-19.
- Uncertainty uses 5,000 whole-lineage bootstrap draws, paired across genes/models. Original CTRP lineage draws are reused; common-13 comparisons share identical lineage draws in both datasets. Intervals condition on the saved fits and donor distributions, with no refitting, gene/model reselection or multiplicity correction.
- Replacement is marginal within lineage: it can break gene–gene correlations and create atypical combinations. Redundant predictors can divide importance. A weak/negative predictive effect is not evidence against biological involvement.
- Training cohorts, tested dose ranges, response processing and fitted regularization differ between assays. The existing CTRP normalization/provenance limitation remains. Even matching receiver cells does not match model training cohorts or isolate assay biology.
- Repeated holdouts overlap. Tiny lineage test subsets remain uncertain. The cohort has informed prior modeling choices; these are exploratory model explanations, not independent causal or clinical validation.

## Files

- gene_comparison.tsv: all 64 genes, both assays and all models, native ranks/effects, controlled ranks, selection and lineage/split consistency.
- gene_importance_summary.tsv, per_repeat_importance.tsv, per_lineage_importance.tsv: full global, split and lineage summaries.
- common_lineage_gene_concordance.tsv, gene_lineage_concordance_summary.tsv, gene_lineage_effect_agreement.tsv: exact positive-effect intersections and matching lineages.
- cross_assay_rank_correlations.tsv, within_assay_model_agreement.tsv, cross_assay_top_gene_overlap.tsv, rank_stability.tsv: global/cohort-controlled/model/lineage agreement.
- coefficient_selection.tsv, coefficient_sensitivity_by_repeat.tsv, donor_reference_sensitivity.tsv: weights, derivatives and donor-reference checks.
- matched_cell_control.tsv, ctrp_importance_arrays.npz, prism_importance_arrays.npz, bootstrap_draws.npz, repeats/: matched receivers and complete numerical arrays.
- protocol.json, source_manifest.json, calculation_validation.json, validation.json, analysis scripts and artifact_manifest.json: methods and reproducibility. Original fitted models remain dependencies at their recorded source paths.
