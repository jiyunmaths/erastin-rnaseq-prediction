# Inverse-lineage-frequency training weights for erastin prediction

Lowest pooled RMSE across the 18 evaluated arms: **Neural network, 64 genes (unweighted) (0.119700)**. Lowest equal-lineage mean RMSE: **Neural network, 64 genes (unweighted) (0.120358)**. Highest equal-lineage mean Spearman: **Ridge, 40 genes (unweighted) (0.23008)**.

All nine weighted model/panel combinations completed the same 20 saved leave-lineage-out repeats: 745 cell lines, 734 identity groups, 24 lineage holdouts and five inner tuning folds. All 4,320 weighted outer models were newly fitted. The nine verified original unweighted prediction arrays were reused for paired comparisons.

## Results

Each entry is the mean score across 20 repeats. Scores were computed separately for each repeat; predictions were not averaged across repeats before scoring. Pooled metrics include all 745 lines. Macro metrics are equal-weight means over the same 19 lineages with n≥10.

| Model | Genes | Training | Pooled Spearman ↑ | Pooled RMSE ↓ | Pooled R² ↑ | Macro Spearman ↑ | Macro RMSE ↓ |
|---|---:|---|---:|---:|---:|---:|---:|
| Ridge | 40 | Original | 0.42380 | 0.120750 | 0.17373 | 0.23008 | 0.121250 |
| Ridge | 40 | Weighted | 0.39650 | 0.122246 | 0.15313 | 0.17769 | 0.123149 |
| Ridge | 64 | Original | 0.43744 | 0.119984 | 0.18418 | 0.21004 | 0.120673 |
| Ridge | 64 | Weighted | 0.39874 | 0.122110 | 0.15501 | 0.16353 | 0.123117 |
| Ridge | 80 | Original | 0.43307 | 0.120515 | 0.17694 | 0.18584 | 0.121473 |
| Ridge | 80 | Weighted | 0.38412 | 0.122898 | 0.14406 | 0.14757 | 0.124217 |
| Neural network | 40 | Original | 0.42824 | 0.120484 | 0.17737 | 0.22628 | 0.120872 |
| Neural network | 40 | Weighted | 0.39794 | 0.122023 | 0.15621 | 0.17307 | 0.122874 |
| Neural network | 64 | Original | 0.43780 | 0.119700 | 0.18804 | 0.21011 | 0.120358 |
| Neural network | 64 | Weighted | 0.39759 | 0.121869 | 0.15834 | 0.17356 | 0.122968 |
| Neural network | 80 | Original | 0.43416 | 0.120244 | 0.18065 | 0.18317 | 0.121215 |
| Neural network | 80 | Weighted | 0.38824 | 0.122424 | 0.15067 | 0.15399 | 0.123967 |
| XGBoost | 40 | Original | 0.39710 | 0.121394 | 0.16488 | 0.22743 | 0.121739 |
| XGBoost | 40 | Weighted | 0.38860 | 0.121953 | 0.15717 | 0.20477 | 0.123225 |
| XGBoost | 64 | Original | 0.39449 | 0.121708 | 0.16056 | 0.20159 | 0.121914 |
| XGBoost | 64 | Weighted | 0.37073 | 0.122896 | 0.14408 | 0.16911 | 0.124173 |
| XGBoost | 80 | Original | 0.38899 | 0.122079 | 0.15542 | 0.19540 | 0.122649 |
| XGBoost | 80 | Weighted | 0.36657 | 0.123102 | 0.14121 | 0.17101 | 0.124620 |

![Weighted comparison](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_lineage_weighted/weighted_comparison.png)

## Matched effect of reweighting

Positive improvement favors inverse-frequency weighting: Spearman(weighted) − Spearman(original), or RMSE(original) − RMSE(weighted). Intervals use 5,000 paired whole-lineage bootstrap draws, averaging the per-repeat contrasts within each draw. They are conditional on the fitted predictions, do not refit models, and are not adjusted for multiple comparisons.

| Scope | Model | Genes | Metric | Mean improvement | Favorable repeats | Conditional 95% interval |
|---|---|---:|---|---:|---:|---|
| pooled_745 | Ridge | 40 | spearman | -0.027302 | 0/20 | [-0.065590, -0.000239] |
| pooled_745 | Ridge | 40 | rmse | -0.001496 | 0/20 | [-0.003614, +0.000190] |
| pooled_745 | Ridge | 64 | spearman | -0.038708 | 0/20 | [-0.086985, -0.004829] |
| pooled_745 | Ridge | 64 | rmse | -0.002126 | 0/20 | [-0.004942, -0.000033] |
| pooled_745 | Ridge | 80 | spearman | -0.048943 | 0/20 | [-0.098305, -0.009310] |
| pooled_745 | Ridge | 80 | rmse | -0.002383 | 0/20 | [-0.005222, -0.000040] |
| pooled_745 | Neural network | 40 | spearman | -0.030292 | 0/20 | [-0.066963, -0.004307] |
| pooled_745 | Neural network | 40 | rmse | -0.001540 | 0/20 | [-0.003639, +0.000195] |
| pooled_745 | Neural network | 64 | spearman | -0.040212 | 0/20 | [-0.086851, -0.007230] |
| pooled_745 | Neural network | 64 | rmse | -0.002170 | 0/20 | [-0.004973, -0.000111] |
| pooled_745 | Neural network | 80 | spearman | -0.045919 | 0/20 | [-0.090644, -0.007674] |
| pooled_745 | Neural network | 80 | rmse | -0.002180 | 0/20 | [-0.004924, +0.000129] |
| pooled_745 | XGBoost | 40 | spearman | -0.008507 | 4/20 | [-0.039309, +0.018393] |
| pooled_745 | XGBoost | 40 | rmse | -0.000560 | 4/20 | [-0.002656, +0.001064] |
| pooled_745 | XGBoost | 64 | spearman | -0.023767 | 0/20 | [-0.066779, +0.014209] |
| pooled_745 | XGBoost | 64 | rmse | -0.001189 | 0/20 | [-0.003791, +0.001066] |
| pooled_745 | XGBoost | 80 | spearman | -0.022417 | 2/20 | [-0.070671, +0.021789] |
| pooled_745 | XGBoost | 80 | rmse | -0.001023 | 2/20 | [-0.003776, +0.001282] |
| macro_19 | Ridge | 40 | spearman | -0.052385 | 0/20 | [-0.094536, -0.011036] |
| macro_19 | Ridge | 40 | rmse | -0.001899 | 0/20 | [-0.003732, -0.000218] |
| macro_19 | Ridge | 64 | spearman | -0.046506 | 0/20 | [-0.091981, +0.001345] |
| macro_19 | Ridge | 64 | rmse | -0.002444 | 0/20 | [-0.004912, -0.000218] |
| macro_19 | Ridge | 80 | spearman | -0.038272 | 0/20 | [-0.076912, +0.000929] |
| macro_19 | Ridge | 80 | rmse | -0.002744 | 0/20 | [-0.005395, -0.000301] |
| macro_19 | Neural network | 40 | spearman | -0.053208 | 0/20 | [-0.084111, -0.022293] |
| macro_19 | Neural network | 40 | rmse | -0.002001 | 0/20 | [-0.003775, -0.000333] |
| macro_19 | Neural network | 64 | spearman | -0.036552 | 0/20 | [-0.079183, +0.008037] |
| macro_19 | Neural network | 64 | rmse | -0.002609 | 0/20 | [-0.005083, -0.000362] |
| macro_19 | Neural network | 80 | spearman | -0.029179 | 0/20 | [-0.072306, +0.014662] |
| macro_19 | Neural network | 80 | rmse | -0.002752 | 0/20 | [-0.005355, -0.000231] |
| macro_19 | XGBoost | 40 | spearman | -0.022668 | 3/20 | [-0.071555, +0.024430] |
| macro_19 | XGBoost | 40 | rmse | -0.001485 | 1/20 | [-0.003103, +0.000091] |
| macro_19 | XGBoost | 64 | spearman | -0.032483 | 0/20 | [-0.084836, +0.018638] |
| macro_19 | XGBoost | 64 | rmse | -0.002258 | 0/20 | [-0.004384, -0.000216] |
| macro_19 | XGBoost | 80 | spearman | -0.024390 | 4/20 | [-0.082443, +0.032719] |
| macro_19 | XGBoost | 80 | rmse | -0.001971 | 0/20 | [-0.004329, +0.000476] |

## Consistency across unseen lineages

Counts below compare mean held-out performance across repeats, among the 19 lineages with n≥10. A positive count reflects a favorable point estimate, not a separately established effect in that lineage.

| Model | Genes | Higher lineage Spearman | Lower lineage RMSE |
|---|---:|---:|---:|
| Ridge | 40 | 6/19 | 6/19 |
| Ridge | 64 | 5/19 | 7/19 |
| Ridge | 80 | 7/19 | 6/19 |
| Neural network | 40 | 5/19 | 6/19 |
| Neural network | 64 | 6/19 | 6/19 |
| Neural network | 80 | 7/19 | 6/19 |
| XGBoost | 40 | 6/19 | 8/19 |
| XGBoost | 64 | 7/19 | 5/19 |
| XGBoost | 80 | 7/19 | 6/19 |

![Lineage effects](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_lineage_weighted/lineage_weighting_effects.png)

## Panel expansion under weighted training

| Model | Panel contrast | Scope | Metric | Mean improvement | Favorable repeats | Conditional 95% interval |
|---|---|---|---|---:|---:|---|
| Ridge | 64 vs 40 | pooled_745 | spearman | +0.002237 | 19/20 | [-0.032973, +0.039611] |
| Ridge | 64 vs 40 | pooled_745 | rmse | +0.000136 | 19/20 | [-0.001652, +0.001790] |
| Ridge | 80 vs 40 | pooled_745 | spearman | -0.012376 | 0/20 | [-0.048941, +0.030289] |
| Ridge | 80 vs 40 | pooled_745 | rmse | -0.000652 | 0/20 | [-0.002649, +0.001350] |
| Ridge | 80 vs 64 | pooled_745 | spearman | -0.014613 | 0/20 | [-0.037974, +0.006962] |
| Ridge | 80 vs 64 | pooled_745 | rmse | -0.000788 | 0/20 | [-0.001801, +0.000219] |
| Neural network | 64 vs 40 | pooled_745 | spearman | -0.000354 | 7/20 | [-0.035343, +0.038825] |
| Neural network | 64 vs 40 | pooled_745 | rmse | +0.000154 | 20/20 | [-0.001647, +0.001850] |
| Neural network | 80 vs 40 | pooled_745 | spearman | -0.009704 | 0/20 | [-0.042381, +0.032732] |
| Neural network | 80 vs 40 | pooled_745 | rmse | -0.000400 | 0/20 | [-0.002450, +0.001626] |
| Neural network | 80 vs 64 | pooled_745 | spearman | -0.009351 | 0/20 | [-0.031189, +0.012115] |
| Neural network | 80 vs 64 | pooled_745 | rmse | -0.000554 | 0/20 | [-0.001716, +0.000479] |
| XGBoost | 64 vs 40 | pooled_745 | spearman | -0.017870 | 1/20 | [-0.049560, +0.006403] |
| XGBoost | 64 vs 40 | pooled_745 | rmse | -0.000943 | 1/20 | [-0.002367, +0.000404] |
| XGBoost | 80 vs 40 | pooled_745 | spearman | -0.022022 | 1/20 | [-0.070462, +0.013564] |
| XGBoost | 80 vs 40 | pooled_745 | rmse | -0.001149 | 0/20 | [-0.002828, +0.000392] |
| XGBoost | 80 vs 64 | pooled_745 | spearman | -0.004152 | 7/20 | [-0.038700, +0.024636] |
| XGBoost | 80 vs 64 | pooled_745 | rmse | -0.000206 | 5/20 | [-0.001836, +0.001116] |
| Ridge | 64 vs 40 | macro_19 | spearman | -0.014158 | 0/20 | [-0.053613, +0.025001] |
| Ridge | 64 vs 40 | macro_19 | rmse | +0.000033 | 19/20 | [-0.001772, +0.001990] |
| Ridge | 80 vs 40 | macro_19 | spearman | -0.030122 | 0/20 | [-0.069746, +0.006969] |
| Ridge | 80 vs 40 | macro_19 | rmse | -0.001067 | 0/20 | [-0.003420, +0.001566] |
| Ridge | 80 vs 64 | macro_19 | spearman | -0.015964 | 0/20 | [-0.040459, +0.008042] |
| Ridge | 80 vs 64 | macro_19 | rmse | -0.001100 | 0/20 | [-0.002494, +0.000247] |
| Neural network | 64 vs 40 | macro_19 | spearman | +0.000482 | 16/20 | [-0.032836, +0.033829] |
| Neural network | 64 vs 40 | macro_19 | rmse | -0.000094 | 1/20 | [-0.001898, +0.001958] |
| Neural network | 80 vs 40 | macro_19 | spearman | -0.019083 | 0/20 | [-0.056856, +0.015889] |
| Neural network | 80 vs 40 | macro_19 | rmse | -0.001093 | 0/20 | [-0.003524, +0.001743] |
| Neural network | 80 vs 64 | macro_19 | spearman | -0.019565 | 0/20 | [-0.044225, +0.005294] |
| Neural network | 80 vs 64 | macro_19 | rmse | -0.000999 | 0/20 | [-0.002534, +0.000402] |
| XGBoost | 64 vs 40 | macro_19 | spearman | -0.035658 | 2/20 | [-0.071518, -0.000129] |
| XGBoost | 64 vs 40 | macro_19 | rmse | -0.000948 | 3/20 | [-0.002428, +0.000478] |
| XGBoost | 80 vs 40 | macro_19 | spearman | -0.033760 | 2/20 | [-0.081766, +0.015865] |
| XGBoost | 80 vs 40 | macro_19 | rmse | -0.001396 | 1/20 | [-0.003200, +0.000619] |
| XGBoost | 80 vs 64 | macro_19 | spearman | +0.001898 | 11/20 | [-0.037221, +0.041790] |
| XGBoost | 80 vs 64 | macro_19 | rmse | -0.000448 | 5/20 | [-0.002060, +0.001110] |

## Exact weighting and preserved controls

For each current fitting subset, $w_i=N/(K n_{\ell(i)})$, where N is its training sample count, K its number of represented lineages, and n its training count for the sample’s lineage. Weights are recomputed using only that subset. They sum to N, average one, and give each represented lineage total weight N/K. All represented training lineages participate, including lineages with fewer than 10 samples; the n≥10 restriction applies only to reported macro scoring.

- Outer refit per-cell weights range from 0.2336 to 32.3043; K ranges from 23 to 23. Weight effective sample size, (sum w)²/sum(w²), ranges from 196.9 to 312.2; this is a weight-concentration diagnostic, not a biological sample count.
- Inner-fit per-cell weights range from 0.2322 to 26.9545; K ranges from 22 to 23.
- Lineages represented by fewer than 10 training samples receive 17.39–21.74% of total outer training weight. The macro-19 score excludes lineages with fewer than 10 total cohort samples, so it is not the identical population balance as the all-lineage training objective.
- X/y standardization remains unweighted and uses only current training rows, matching the original comparison. Ridge uses weighted centering for its unpenalized intercept.
- Hyperparameter selection remains the minimum unweighted equal-mean five-fold inner RMSE. This experiment changes training loss weights while retaining the original tuning objective.
- Ridge retains the original 17-alpha grid. Neural network retains the four width/penalty candidates (16/32 tanh units, alpha 10/100), L-BFGS, and the same three initializations per repeat. XGBoost retains the original 36 candidates, CPU hist, 0.03 learning rate, 0.8 row/column sampling, and per-repeat seeds.
- Weights multiply the squared-error training terms in all families. Normalization to mean one retains the original regularization scale. XGBoost retains uniform row sampling at 0.8; equality is in full-set weight totals, not each stochastic subsample. Its minimum-child-weight grid now operates on weighted Hessian mass.
- Held-out lineage membership and related identity groups are excluded from both inner fitting and outer refitting. No lineage label is supplied as an input feature. All repeats and lineages are retained.

| Training lineage | Outer-refit per-cell weight range | Training cell-count range when present |
|---|---:|---:|
| Cervix | 27.130–32.304 | 1–1 |
| Ampulla of Vater | 13.565–16.174 | 2–2 |
| Biliary Tract | 4.522–5.391 | 6–6 |
| Prostate | 4.522–5.391 | 6–6 |
| Pleura | 3.014–3.594 | 9–9 |
| Thyroid | 2.713–3.235 | 10–10 |
| Soft Tissue | 2.087–2.488 | 13–13 |
| Bone | 2.087–2.488 | 13–13 |
| Peripheral Nervous System | 1.809–2.157 | 15–15 |
| Bladder/Urinary Tract | 1.428–1.703 | 19–19 |
| Kidney | 1.233–1.470 | 22–22 |
| Liver | 1.233–1.470 | 22–22 |
| Uterus | 1.043–1.244 | 26–26 |
| Head and Neck | 0.936–1.115 | 29–29 |
| Breast | 0.798–0.951 | 34–34 |
| Ovary/Fallopian Tube | 0.733–0.874 | 37–37 |
| Pancreas | 0.714–0.851 | 38–38 |
| Skin | 0.678–0.809 | 40–40 |
| Myeloid | 0.662–0.789 | 41–41 |
| Bowel | 0.617–0.735 | 44–44 |
| CNS/Brain | 0.577–0.688 | 47–47 |
| Esophagus/Stomach | 0.493–0.588 | 55–55 |
| Lymphoid | 0.286–0.341 | 95–95 |
| Lung | 0.234–0.267 | 121–121 |

## Validation and reproducibility

- All 4,320 weighted outer models reconstructed their saved predictions; maximum absolute discrepancy 2.22e-16.
- All 410,400 candidate-fold scores were recomputed; maximum RMSE discrepancy 1.11e-16. Selected candidates match the original unweighted inner-scoring rule.
- All 25,920 fitting-subset weight summaries were reconstructed from their training memberships; equal total lineage weights and training-only scaling were checked.
- Preflight: neural-network unit weights exactly reproduced the original implementation on synthetic data and all three actual panels for Lung/Thyroid in repeat 1. Weighted neural gradients matched independent finite differences at three parameter points; direct weighted-objective calculations also matched. Weighted ridge and XGBoost agreed with separate sklearn API checks.
- Selected settings were refitted for all nine weighted arms in Lung/Thyroid for repeats 1/20, including the five corresponding inner fits. Ridge and XGBoost use separate sklearn APIs for these checks; neural refits use the verified weighted adapter and check deterministic replay.
- Training warnings: 90,720; convergence warnings: 0. Warning summaries and per-fit diagnostics are saved.
- Original prediction scores, source hashes, exact saved split arrays, sample order, targets and initialization seeds were verified. Full weighted models, inner predictions, tuning tables, weight audits, bootstrap draws and source scripts are included in the archive.

## Interpretation limits

These results concern transfer to held-out cancer lineages in this reused cell-line cohort. The 20 repeats perturb inner tuning and initialization, not the biological sample; consistency across seeds does not supply 20 independent biological validations. Whole-lineage bootstrap intervals reflect uncertainty across the observed held-out lineages, conditional on the fitted models. Extremely small training lineages receive large per-cell weights and can substantially affect fitting. Equal training weight does not guarantee equal accuracy, and no observed effect establishes causal ferroptosis biology. Inner folds mix training lineages, as in the original protocol; only outer tests measure unseen-lineage transfer. The original harmonized erastin AUC provenance limitation remains unresolved, and no independent drug-response cohort was used.

## Files

- `summary.tsv`: all 18 arms, pooled and macro scores with seed dispersion.
- `paired_comparisons.tsv` and `paired_seed_differences.tsv`: matching reweighting contrasts plus weighted panel/family comparisons.
- `lineage_paired_effects.tsv` and `per_lineage_metrics.tsv`: held-out lineage effects.
- `training_subset_weight_audit.tsv` and `training_lineage_weights.tsv`: exact training-weight totals, counts and concentration.
- `selected_parameters.tsv`, `tuning_scores.tsv`, `parameter_counts.tsv`, `validation.json`, `preflight_validation.json`, `protocol.json`: controls and audits.
- `predictions.npz`, `bootstrap_draws.npz`, `folds/`: predictions, uncertainty draws and all fitted weighted models.
