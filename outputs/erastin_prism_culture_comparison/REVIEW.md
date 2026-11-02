# Adding DepMap culture metadata to PRISM erastin prediction

**Completed 180 new outer fits and reused the 120 unchanged gene-only models.** All 300 models passed the checks below. Culture augmentation lowered mean pooled test RMSE in 1/6 matched gene-panel/model comparisons. Of the six conditional 95% intervals, 0 lie above zero (benefit), 0 below zero (harm), and 6 include zero. These are exploratory, unadjusted comparisons.

Lowest mean pooled RMSE among all 15 fitted arms: **Ridge, 5 genes only**, 0.137092. Ranking the observed means is descriptive and does not establish superiority.

**The culture annotations describe DepMap onboarding. They are not verified PRISM assay conditions.** Any predictive association may reflect lineage, source, or other cell-line properties; it cannot establish a causal growth-medium effect.

## What was added

| Covariate | Coverage / encoding |
|---|---|
| GrowthPattern | 369 adherent, 12 mixed, 4 suspension, 1 unknown. Unknown is explicit missing. |
| Medium identity | 367/386 annotated; 55 distinct MF codes. Training categories with fewer than 5 rows are pooled as OTHER; unseen codes use OTHER. |
| Basal-medium family | Recipe prefix before first +; same training-only rare-category grouping. |
| Recorded serum percentage | Explicit % FBS in each of the 367 nonempty recipes; training-median imputation for 19 missing values, plus a missingness indicator. |

Excluded fields have no measured contrast: PlateCoating is the literal None for all 386 lines; all 367 nonmissing SerumFreeMedia flags are False, with the same 19 missing-media rows. ModelDerivationMaterial, ModelTreatment and CulturedResistanceDrug are entirely empty. No basal nutrient concentrations or unrecorded supplements were inferred. Exact full recipe is not encoded separately from its one-to-one medium code.

The downloaded Model.csv has an important schema inconsistency: **OnboardedMedia contains MF codes; FormulationID contains recipe text**, while the accompanying README descriptions imply the opposite. Code–recipe mapping is one-to-one in the retained cohort. Raw column names and values are preserved in culture_metadata.tsv and the observed mapping is saved in medium_dictionary.tsv. Source files match their original recorded SHA256 hashes.

Source: [DepMap 24Q4 Public v1](https://doi.org/10.25452/figshare.plus.27993248.v1). Outcome: [PRISM Repurposing 20Q2 v1](https://doi.org/10.6084/m9.figshare.20564034.v1), erastin BRD-A25004090-001-08-4, HTS002 secondary-screen dose-response AUC. The local source README and source manifests are the provenance for the field audit.

## Gene-only versus gene-plus-culture test scores

| Family | Genes | RMSE genes only | RMSE + culture | Spearman genes only | Spearman + culture | R² genes only | R² + culture |
|---|---:|---:|---:|---:|---:|---:|---:|
| Ridge | 5 | 0.137092 | 0.137534 | 0.3450 | 0.3238 | 0.0728 | 0.0670 |
| Ridge | 64 | 0.138611 | 0.138564 | 0.3601 | 0.3617 | 0.0514 | 0.0521 |
| Elastic Net | 5 | 0.137380 | 0.137913 | 0.3390 | 0.3256 | 0.0687 | 0.0618 |
| Elastic Net | 64 | 0.139082 | 0.139178 | 0.3491 | 0.3477 | 0.0450 | 0.0437 |
| Neural network | 5 | 0.137392 | 0.139104 | 0.3435 | 0.3067 | 0.0682 | 0.0448 |
| Neural network | 64 | 0.138203 | 0.138247 | 0.3616 | 0.3612 | 0.0569 | 0.0564 |

Metrics are computed separately on each 78-line test set, then averaged over 20 repeats; predictions are not stacked as independent observations. Higher negative-AUC predictions mean greater erastin sensitivity. Original AUC values above 1 remain unchanged.

![Model comparison](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_prism_culture_comparison/model_comparison.png)

## Paired augmentation effects

| Family | Genes | Mean RMSE reduction | Relative reduction | Favorable splits | Favorable lineages | Conditional 95% interval |
|---|---:|---:|---:|---:|---:|---|
| Ridge | 5 | -0.000442 | -0.322% | 9/20 | 11/20 | [-0.001738, +0.000894] |
| Ridge | 64 | +0.000047 | +0.034% | 12/20 | 12/20 | [-0.000183, +0.000263] |
| Elastic Net | 5 | -0.000532 | -0.387% | 7/20 | 7/20 | [-0.001348, +0.000285] |
| Elastic Net | 64 | -0.000096 | -0.069% | 8/20 | 11/20 | [-0.000257, +0.000058] |
| Neural network | 5 | -0.001711 | -1.246% | 4/20 | 9/20 | [-0.003704, +0.000367] |
| Neural network | 64 | -0.000044 | -0.032% | 11/20 | 11/20 | [-0.000365, +0.000251] |

Positive reduction favors adding culture. Favorable lineages count a positive mean within-lineage RMSE reduction; small lineages can have only one held-out line per repeat, so these counts are descriptive. An interval containing zero does not establish equivalence.

![Paired effects](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_prism_culture_comparison/paired_culture_effects.png)

| Family | Genes | Macro-20 RMSE genes only | Macro-20 RMSE + culture | Mean macro-20 reduction | Conditional 95% interval |
|---|---:|---:|---:|---:|---|
| Ridge | 5 | 0.107695 | 0.107678 | +0.000017 | [-0.001764, +0.001850] |
| Ridge | 64 | 0.109207 | 0.108986 | +0.000221 | [-0.000084, +0.000571] |
| Elastic Net | 5 | 0.107930 | 0.108546 | -0.000617 | [-0.001860, +0.000616] |
| Elastic Net | 64 | 0.109586 | 0.109550 | +0.000036 | [-0.000148, +0.000218] |
| Neural network | 5 | 0.107551 | 0.107685 | -0.000134 | [-0.002564, +0.002567] |
| Neural network | 64 | 0.108525 | 0.108292 | +0.000233 | [-0.000165, +0.000690] |

Macro-20 gives equal weight to the 20 lineage-specific test RMSEs. Macro-13, MAE, Pearson, R², full repeat values and all paired intervals are also saved.

![Lineage effects](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_prism_culture_comparison/lineage_culture_effects.png)

## Culture-only controls

| Family | RMSE | Spearman | R² | Macro-20 RMSE |
|---|---:|---:|---:|---:|
| Ridge | 0.144750 | 0.0632 | -0.0382 | 0.119753 |
| Elastic Net | 0.144659 | 0.0363 | -0.0370 | 0.119686 |
| Neural network | 0.144903 | 0.0327 | -0.0398 | 0.119986 |
| Outer-training mean | 0.144030 | NA | -0.0286 | 0.119545 |

All culture-only models use the same preprocessing, tuning grids and inner/outer splits as augmented arms. Constant predictors retain undefined correlations as NA; defined-repeat counts are saved. The training-mean prediction uses only current outer-training outcomes.

## Unchanged split design and tuning

- 386 cell lines, 379 identity groups, 20 lineages. Exact prior shared 20 outer masks and five inner folds per repeat; unweighted training. Each outer split has 308 training and 78 testing lines, with every lineage on both sides and no shared identity group. No new cohort exclusion.
- Every inner training fold contains all 20 lineages; inner validation contains 17–20. Categories, rare-category pooling, serum imputation and all scaling are fitted inside the current inner training fold, then refitted on outer training for the selected model. Lineage and cell identity are not predictor columns.
- Standardize expression and serum percentage only. One-hot categories and missingness indicators are centered without variance standardization. Reserved missing and OTHER columns distinguish absent annotations from rare/unseen categories. Column means from training center the final joint design.
- Ridge: 17 alphas, 1e-4 to 1e4. Elastic Net: 41 alphas, 10 to 1e-4, crossed with L1 ratios .01/.05/.1/.25/.5/.75/.9/1; tol 1e-9, max_iter 50,000.
- Neural network: 16/32 tanh units, alpha 1/10/100/1,000; three initializations averaged using the same saved triplets. L-BFGS, max_iter 2,000, max_fun 40,000, tol .001, no early stopping. Native unweighted MLP parity with the previous unit-weight adapter passed the independent preflight.
- Select by equal-mean five-fold RMSE in original target units, with first grid member breaking ties. The 5-gene panel remains SLC7A11, GPX4, ABCC1, ACSL4, AIFM2. The 64-gene panel and expression values are unchanged.
- Culture adds columns without changing the candidate grids. Penalization and selected architecture can still change when features are added; this is an operational prediction comparison, not an isolated causal culture-effect estimate.

| Input genes | Outer joint-feature count range |
|---:|---:|
| 0 | 26–30 |
| 5 | 31–35 |
| 64 | 90–94 |

## Validation and interpretation limits

- All 300 outer models audited: 180 newly fitted, 120 baseline model files reused byte-for-byte. Original baseline test predictions and mean scores reproduced. Verified all source/input/script hashes and split masks.
- Reconstructed 176,500 candidate-fold scores, 1,800 identity exclusions and 1,080 training-only design reconstructions. Maximum saved-prediction error 2.22e-16; inner-RMSE error 1.11e-16.
- Recomputed all 300 selected NN-member objectives (maximum error 5.55e-17); checked 100 selected outer Elastic Net objectives and recorded residuals for 164,000 inner candidate solutions. Maximum Elastic Net optimality residual 7.4e-10.
- Independently reconstructed mixed-feature matrices using raw values and explicit indicators. Verified unseen/missing category handling, training-only medians/scalers/category thresholds, path/native/augmented-Lasso parity and the previous unit-weight NN adapter. Independent selected-setting inner/outer refits passed for every new arm in repeats 1 and 20.
- Convergence warnings: 0 across candidate/outer fits; selected outer convergence warnings: 0. Other warning categories are saved.
- Intervals use 5,000 paired whole-lineage bootstrap draws shared across arms/repeats. They condition on fitted models and the observed cohort/splits; no model refitting, multiplicity correction or post-selection correction. Independent expanded-sample SciPy checks passed, including undefined constant-prediction correlations.
- This is an exploratory internal PRISM comparison informed by earlier results. Overlapping 20 holdouts are not independent biological replications; there is no untouched final test cohort. Evaluation is for new cell lines from represented lineages. It does not establish unseen-lineage, unseen-medium, patient or clinical generalization.
- Onboarding media are proxies for the requested growth conditions. Missingness, rare recipes and highly imbalanced growth patterns limit what can be learned. This analysis does not identify a causal medium effect or show that changing a cell line’s culture recipe changes erastin response.

## Files

- summary.tsv, paired_comparisons.tsv, paired_seed_differences.tsv: overall accuracy and paired effects.
- per_lineage_metrics.tsv, lineage_paired_effects.tsv, lineage_win_counts.tsv: lineage-specific accuracy and effect consistency.
- culture_metadata.tsv, medium_dictionary.tsv, metadata_coverage.json: raw/derived onboarding fields and coverage audit.
- panel_inputs.npz, shared_splits.npz: unchanged RNA, outcomes, identities and train/test/validation memberships.
- folds/, selected_parameters.tsv, tuning_scores.tsv, outer_preprocessing.tsv: saved fitted models, predictions and tuning/preprocessing diagnostics. Use the saved preprocessor before new culture-model predictions.
- linear_coefficients.tsv: fitted coefficients in the model design (not causal importance; continuous and binary columns have different scaling conventions).
- protocol.json, source_manifest.json, preflight_validation.json, validation.json, visual_qa.json, artifact_manifest.json and scripts: reproducibility and integrity evidence.
