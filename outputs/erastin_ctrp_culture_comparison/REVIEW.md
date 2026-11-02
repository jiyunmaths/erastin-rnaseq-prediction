# Adding DepMap culture metadata to CTRP erastin prediction

**Completed 180 new outer fits and reused the 120 unchanged gene-only models.** All 300 models passed the checks below. Culture augmentation lowered mean pooled test RMSE in 3/6 matched gene-panel/model comparisons. Of the six conditional 95% intervals, 0 lie above zero (benefit), 0 below zero (harm), and 6 include zero. These are exploratory, unadjusted comparisons.

Lowest mean pooled RMSE among all 15 fitted arms: **Neural network, 64 genes only**, 0.116968. Ranking the observed means is descriptive and does not establish superiority.

**The culture annotations describe DepMap onboarding. They are not verified CTRP assay conditions.** Any predictive association may reflect lineage, source, or other cell-line properties; it cannot establish a causal growth-medium effect.

## What was added

| Covariate | Coverage / encoding |
|---|---|
| GrowthPattern | 548 adherent, 45 mixed, 148 suspension, 3 unknown. Unknown is explicit missing. |
| Medium identity | 704/744 annotated; 87 distinct MF codes. Training categories with fewer than 5 rows are pooled as OTHER; unseen codes use OTHER. |
| Basal-medium family | Recipe prefix before first +; same training-only rare-category grouping. |
| Recorded serum percentage | Explicit % FBS in 703 recipes; training-median imputation for 41 unknown percentages (40 absent media plus one serum-free recipe with no numeric FBS %), plus a missingness indicator. No zero inferred. |

The same PRISM covariate set is retained. PlateCoating is the literal None for all 744 lines; ModelDerivationMaterial and CulturedResistanceDrug are empty. ModelTreatment has one EBV record and is outside the fixed covariate set. SerumFreeMedia has 703 False, one True and 40 missing; no extra serum-free flag is added. The True recipe lacks explicit FBS % and remains missing numerically. No basal nutrient concentrations or unrecorded supplements were inferred. Exact full recipe is not encoded separately from its one-to-one medium code.

The downloaded Model.csv has an important schema inconsistency: **OnboardedMedia contains MF codes; FormulationID contains recipe text**, while the accompanying README descriptions imply the opposite. Code–recipe mapping is one-to-one in the retained cohort. Raw column names and values are preserved in culture_metadata.tsv and the observed mapping is saved in medium_dictionary.tsv. Source files match their original recorded SHA256 hashes.

Source: [DepMap 24Q4 Public v1](https://doi.org/10.25452/figshare.plus.27993248.v1). Outcome: unchanged harmonized CTRP erastin AUC from the [recorded redistribution archive](https://doi.org/10.5281/zenodo.22125121). Archive checksums were verified in the original preparation; the exact upstream AUC generation recipe remains unverified. The local source README and source manifests are the provenance for the field audit.

## Gene-only versus gene-plus-culture test scores

| Family | Genes | RMSE genes only | RMSE + culture | Spearman genes only | Spearman + culture | R² genes only | R² + culture |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge | 5 | 0.120231 | 0.120204 | 0.4098 | 0.4213 | 0.1644 | 0.1649 |
| Ridge | 64 | 0.117005 | 0.117161 | 0.4606 | 0.4596 | 0.2087 | 0.2066 |
| Elastic Net | 5 | 0.120222 | 0.119873 | 0.4097 | 0.4245 | 0.1646 | 0.1697 |
| Elastic Net | 64 | 0.117205 | 0.117323 | 0.4600 | 0.4578 | 0.2062 | 0.2045 |
| Neural network | 5 | 0.120052 | 0.119943 | 0.4097 | 0.4237 | 0.1669 | 0.1680 |
| Neural network | 64 | 0.116968 | 0.117055 | 0.4618 | 0.4603 | 0.2094 | 0.2081 |

Metrics are computed separately on each 149-line test set, then averaged over 20 repeats; predictions are not stacked as independent observations. Higher negative-AUC predictions mean greater erastin sensitivity. The original harmonized CTRP target array is retained without transformation beyond its existing sign convention.

![Model comparison](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_ctrp_culture_comparison/model_comparison.png)

## Paired augmentation effects

| Family | Genes | Mean RMSE reduction | Relative reduction | Favorable splits | Favorable lineages | Conditional 95% interval |
|---|---:|---:|---:|---:|---:|---|
| Ridge | 5 | +0.000027 | +0.023% | 8/20 | 11/20 | [-0.001611, +0.001431] |
| Ridge | 64 | -0.000156 | -0.134% | 8/20 | 13/20 | [-0.000422, +0.000149] |
| Elastic Net | 5 | +0.000349 | +0.290% | 12/20 | 10/20 | [-0.000914, +0.001462] |
| Elastic Net | 64 | -0.000117 | -0.100% | 7/20 | 11/20 | [-0.000235, +0.000007] |
| Neural network | 5 | +0.000108 | +0.090% | 9/20 | 10/20 | [-0.001969, +0.001940] |
| Neural network | 64 | -0.000087 | -0.075% | 8/20 | 12/20 | [-0.000338, +0.000182] |

Positive reduction favors adding culture. Favorable lineages count a positive mean within-lineage RMSE reduction; small lineages can have only one held-out line per repeat, so these counts are descriptive. An interval containing zero does not establish equivalence.

![Paired effects](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_ctrp_culture_comparison/paired_culture_effects.png)

| Family | Genes | Macro-23 RMSE genes only | Macro-23 RMSE + culture | Mean macro-23 reduction | Conditional 95% interval |
|---|---:|---:|---:|---:|---|
| Ridge | 5 | 0.112458 | 0.112806 | -0.000348 | [-0.001913, +0.001312] |
| Ridge | 64 | 0.110655 | 0.110745 | -0.000091 | [-0.000401, +0.000243] |
| Elastic Net | 5 | 0.112425 | 0.112538 | -0.000114 | [-0.001320, +0.001197] |
| Elastic Net | 64 | 0.111193 | 0.111234 | -0.000041 | [-0.000176, +0.000089] |
| Neural network | 5 | 0.112192 | 0.112551 | -0.000358 | [-0.002416, +0.002026] |
| Neural network | 64 | 0.110514 | 0.110563 | -0.000049 | [-0.000322, +0.000246] |

Macro-23 gives equal weight to the 23 lineage-specific test RMSEs. Macro-19, MAE, Pearson, R², full repeat values and all paired intervals are also saved.

![Lineage effects](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_ctrp_culture_comparison/lineage_culture_effects.png)

## Culture-only controls

| Family | RMSE | Spearman | R² | Macro-23 RMSE |
|---|---:|---:|---:|---:|
| Ridge | 0.127962 | 0.2108 | 0.0553 | 0.119042 |
| Elastic Net | 0.127968 | 0.2066 | 0.0552 | 0.119109 |
| Neural network | 0.128005 | 0.2135 | 0.0545 | 0.119555 |
| Outer-training mean | 0.132083 | NA | -0.0062 | 0.120777 |

## Five versus 64 genes with culture included

| Family | RMSE reduction: 64 vs 5 | Relative reduction | Favorable splits for 64 | Conditional 95% interval |
|---|---:|---:|---:|---|
| Ridge | +0.003043 | +2.531% | 16/20 | [+0.000011, +0.006395] |
| Elastic Net | +0.002551 | +2.128% | 16/20 | [-0.000359, +0.005753] |
| Neural network | +0.002888 | +2.408% | 14/20 | [-0.000095, +0.006156] |

All three 64-gene augmented models have lower mean RMSE than their five-gene counterparts. Only the ridge panel interval excludes zero, narrowly, under these conditional unadjusted intervals. This is a previously specified panel contrast; positive reductions in this table favor 64 genes.


All culture-only models use the same preprocessing, tuning grids and inner/outer splits as augmented arms. Constant predictors retain undefined correlations as NA; defined-repeat counts are saved. The training-mean prediction uses only current outer-training outcomes.

## Unchanged split design and tuning

- 744 cell lines, 733 identity groups, 23 lineages. Cervix remains excluded as in the baseline. Exact prior shared 20 outer masks and five inner folds per repeat; unweighted training. Each outer split has 595 training and 149 testing lines, with every lineage on both sides and no shared identity group. No new cohort exclusion.
- Every inner training fold contains all 23 lineages; validation contains 22. The single outer-training Ampulla line remains in every inner training fit and is never used for inner validation. Categories, rare-category pooling, serum imputation and all scaling are fitted inside the current inner training fold, then refitted on outer training for the selected model. Lineage and cell identity are not predictor columns.
- Standardize expression and serum percentage only. One-hot categories and missingness indicators are centered without variance standardization. Reserved missing and OTHER columns distinguish absent annotations from rare/unseen categories. Column means from training center the final joint design.
- Ridge: 17 alphas, 1e-4 to 1e4. Elastic Net retains the same 41 alphas, 10 to 1e-4, crossed with L1 ratios .01/.05/.1/.25/.5/.75/.9/1. Mixed-feature fits use tighter tol 1e-11 and max_iter 200,000; the reused baselines used tol 1e-9 and max_iter 50,000. This is a numerical precision change, not a new tuning grid.
- Overlapping medium indicators make pure-Lasso coefficients potentially nonunique. Inner candidates at L1 ratio 1 use native cold starts to match outer refits. Other mixing ratios use the original warm-start path algorithm. Native prediction parity is checked; augmented-row Lasso equivalence is assessed by objective and optimality residual for pure Lasso because unseen-combination predictions need not be unique across optimal coefficient vectors.
- Neural network: 16/32 tanh units, alpha 10/100; three initializations averaged using the same saved triplets. L-BFGS, max_iter 2,000, max_fun 40,000, tol .001, no early stopping. Native unweighted MLP parity with the previous unit-weight adapter passed the independent preflight.
- Select by equal-mean five-fold RMSE in original target units, with first grid member breaking ties. The 5-gene panel remains SLC7A11, GPX4, ABCC1, ACSL4, AIFM2. The 64-gene panel and expression values are unchanged.
- Culture adds columns without changing the candidate grids. Penalization and selected architecture can still change when features are added; this is an operational prediction comparison, not an isolated causal culture-effect estimate.

| Input genes | Outer joint-feature count range |
|---:|---:|
| 0 | 33–38 |
| 5 | 38–43 |
| 64 | 97–102 |

## Validation and interpretation limits

- All 300 outer models audited: 180 newly fitted, 120 baseline model files reused byte-for-byte. Original baseline test predictions and mean scores reproduced. Verified all source/input/script hashes and split masks.
- Replayed all 40 selected gene-only Elastic Net settings at the tighter numerical precision; maximum held-out prediction change 1.55e-10. This checks selected-fit precision without changing saved baseline tuning or predictions.
- Reconstructed 174,500 candidate-fold scores, 1,800 identity exclusions and 1,080 training-only design reconstructions. Maximum saved-prediction error 2.22e-16; inner-RMSE error 1.11e-16.
- Recomputed all 300 selected NN-member objectives (maximum error 5.55e-17); checked 100 selected outer Elastic Net objectives and recorded residuals for 164,000 inner candidate solutions. Maximum Elastic Net optimality residual 4.74e-10.
- Independently reconstructed mixed-feature matrices using raw values and explicit indicators. Verified unseen/missing category handling, training-only medians/scalers/category thresholds, native prediction parity, augmented-Lasso objective/optimality equivalence for pure Lasso and prediction equivalence for other mixing ratios, and the previous unit-weight NN adapter. Independent selected-setting inner/outer refits passed for every new arm in repeats 1 and 20.
- Convergence warnings: 0 across candidate/outer fits; selected outer convergence warnings: 0. Other warning categories are saved.
- Intervals use 5,000 paired whole-lineage bootstrap draws shared across arms/repeats. They condition on fitted models and the observed cohort/splits; no model refitting, multiplicity correction or post-selection correction. Independent expanded-sample SciPy checks passed, including undefined constant-prediction correlations.
- This is an exploratory internal CTRP comparison informed by earlier results. The five-gene panel was chosen after earlier held-out importance analysis on these same splits; tuning inside training does not remove this earlier test-informed selection. Overlapping 20 holdouts are not independent biological replications; there is no untouched final test cohort. Evaluation is for new cell lines from represented lineages. It does not establish unseen-lineage, unseen-medium, patient or clinical generalization.
- Onboarding media are proxies for the requested growth conditions. Missingness, rare recipes and highly imbalanced growth patterns limit what can be learned. This analysis does not identify a causal medium effect or show that changing a cell line’s culture recipe changes erastin response.

## Files

- summary.tsv, paired_comparisons.tsv, paired_seed_differences.tsv: overall accuracy and paired effects.
- per_lineage_metrics.tsv, lineage_paired_effects.tsv, lineage_win_counts.tsv: lineage-specific accuracy and effect consistency.
- culture_metadata.tsv, medium_dictionary.tsv, metadata_coverage.json, culture_coverage_by_lineage.tsv: raw/derived onboarding fields and overall/lineage coverage audit.
- panel_inputs.npz, shared_splits.npz: unchanged RNA, outcomes, identities and train/test/validation memberships.
- folds/, selected_parameters.tsv, tuning_scores.tsv, outer_preprocessing.tsv: saved fitted models, predictions and tuning/preprocessing diagnostics. Use the saved preprocessor before new culture-model predictions.
- linear_coefficients.tsv: fitted coefficients in the model design (not causal importance; continuous and binary columns have different scaling conventions).
- protocol.json, source_manifest.json, preflight_validation.json, validation.json, visual_qa.json, artifact_manifest.json and scripts: reproducibility and integrity evidence.
