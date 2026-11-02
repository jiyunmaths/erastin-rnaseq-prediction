# Comparison of 40-, 64- and 80-gene ferroptosis panels

**Lowest pooled RMSE: Neural network / 64 genes (0.119700).** The lowest equal-lineage mean RMSE is Neural network / 64 genes (0.120358); the highest equal-lineage mean Spearman is Ridge / 40 genes (0.23008). Panel choice therefore needs to reflect both prediction error and ranking within held-out lineages. These are descriptive comparisons on this reused cohort, not external validation.

Completed the same 20 saved leave-lineage-out repeats across 745 cell lines, 734 identity groups and 24 outer lineage holdouts. The 40-gene predictions were reused after exact nested-pipeline replays. The six expanded-panel arms contributed 2,880 newly fitted outer models. All nine arms share response labels, sample ordering, inner splits, per-repeat model seeds and model-specific tuning grids.

| Model | Genes | Pooled Spearman ↑ | Pooled RMSE ↓ | Pooled R² ↑ | Macro Spearman ↑ | Macro RMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Ridge | 40 | 0.42380 | 0.120750 | 0.17373 | 0.23008 | 0.121250 |
| Ridge | 64 | 0.43744 | 0.119984 | 0.18418 | 0.21004 | 0.120673 |
| Ridge | 80 | 0.43307 | 0.120515 | 0.17694 | 0.18584 | 0.121473 |
| Neural network | 40 | 0.42824 | 0.120484 | 0.17737 | 0.22628 | 0.120872 |
| Neural network | 64 | 0.43780 | 0.119700 | 0.18804 | 0.21011 | 0.120358 |
| Neural network | 80 | 0.43416 | 0.120244 | 0.18065 | 0.18317 | 0.121215 |
| XGBoost | 40 | 0.39710 | 0.121394 | 0.16488 | 0.22743 | 0.121739 |
| XGBoost | 64 | 0.39449 | 0.121708 | 0.16056 | 0.20159 | 0.121914 |
| XGBoost | 80 | 0.38899 | 0.122079 | 0.15542 | 0.19540 | 0.122649 |

Values are means of per-repeat scores, not scores after averaging predictions across repeats. Macro metrics weight the same 19 lineages with n≥10 equally; pooled metrics include all 24 lineages.

![Panel comparison](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_panel_comparison/panel_comparison.png)

## Paired panel expansion effects

Positive means the larger panel improves on the smaller panel within the same model family. Intervals use 5,000 paired whole-lineage bootstrap draws, conditional on the fitted predictions.

| Scope | Model | Panel contrast | Metric | Improvement | Favorable repeats | Conditional 95% interval |
|---|---|---|---|---:|---:|---|
| pooled_745 | Ridge | 64 vs 40 | Spearman increase | +0.013642 | 20/20 | [-0.007831, +0.039266] |
| pooled_745 | Ridge | 64 vs 40 | RMSE reduction | +0.000766 | 20/20 | [-0.000406, +0.002179] |
| pooled_745 | Ridge | 80 vs 40 | Spearman increase | +0.009265 | 20/20 | [-0.017053, +0.035106] |
| pooled_745 | Ridge | 80 vs 40 | RMSE reduction | +0.000235 | 18/20 | [-0.001273, +0.001670] |
| pooled_745 | Ridge | 80 vs 64 | Spearman increase | -0.004378 | 0/20 | [-0.022620, +0.008463] |
| pooled_745 | Ridge | 80 vs 64 | RMSE reduction | -0.000531 | 0/20 | [-0.001437, +0.000121] |
| pooled_745 | Neural network | 64 vs 40 | Spearman increase | +0.009566 | 20/20 | [-0.010457, +0.030419] |
| pooled_745 | Neural network | 64 vs 40 | RMSE reduction | +0.000784 | 20/20 | [-0.000290, +0.001853] |
| pooled_745 | Neural network | 80 vs 40 | Spearman increase | +0.005923 | 20/20 | [-0.018840, +0.028231] |
| pooled_745 | Neural network | 80 vs 40 | RMSE reduction | +0.000240 | 20/20 | [-0.001214, +0.001396] |
| pooled_745 | Neural network | 80 vs 64 | Spearman increase | -0.003643 | 0/20 | [-0.021418, +0.009102] |
| pooled_745 | Neural network | 80 vs 64 | RMSE reduction | -0.000544 | 0/20 | [-0.001469, +0.000120] |
| pooled_745 | XGBoost | 64 vs 40 | Spearman increase | -0.002610 | 9/20 | [-0.023847, +0.012743] |
| pooled_745 | XGBoost | 64 vs 40 | RMSE reduction | -0.000314 | 9/20 | [-0.001493, +0.000806] |
| pooled_745 | XGBoost | 80 vs 40 | Spearman increase | -0.008112 | 5/20 | [-0.038219, +0.010700] |
| pooled_745 | XGBoost | 80 vs 40 | RMSE reduction | -0.000686 | 3/20 | [-0.001971, +0.000314] |
| pooled_745 | XGBoost | 80 vs 64 | Spearman increase | -0.005503 | 7/20 | [-0.023390, +0.007592] |
| pooled_745 | XGBoost | 80 vs 64 | RMSE reduction | -0.000372 | 8/20 | [-0.001144, +0.000220] |
| macro_19 | Ridge | 64 vs 40 | Spearman increase | -0.020038 | 0/20 | [-0.074873, +0.025578] |
| macro_19 | Ridge | 64 vs 40 | RMSE reduction | +0.000577 | 20/20 | [-0.000864, +0.002079] |
| macro_19 | Ridge | 80 vs 40 | Spearman increase | -0.044235 | 0/20 | [-0.104125, +0.008831] |
| macro_19 | Ridge | 80 vs 40 | RMSE reduction | -0.000223 | 4/20 | [-0.002262, +0.001639] |
| macro_19 | Ridge | 80 vs 64 | Spearman increase | -0.024197 | 0/20 | [-0.069373, +0.013957] |
| macro_19 | Ridge | 80 vs 64 | RMSE reduction | -0.000800 | 0/20 | [-0.001787, +0.000148] |
| macro_19 | Neural network | 64 vs 40 | Spearman increase | -0.016175 | 0/20 | [-0.061843, +0.022407] |
| macro_19 | Neural network | 64 vs 40 | RMSE reduction | +0.000514 | 20/20 | [-0.000752, +0.001830] |
| macro_19 | Neural network | 80 vs 40 | Spearman increase | -0.043112 | 0/20 | [-0.098074, +0.005978] |
| macro_19 | Neural network | 80 vs 40 | RMSE reduction | -0.000342 | 0/20 | [-0.002288, +0.001406] |
| macro_19 | Neural network | 80 vs 64 | Spearman increase | -0.026938 | 0/20 | [-0.066546, +0.004995] |
| macro_19 | Neural network | 80 vs 64 | RMSE reduction | -0.000856 | 0/20 | [-0.001830, +0.000057] |
| macro_19 | XGBoost | 64 vs 40 | Spearman increase | -0.025842 | 4/20 | [-0.062281, +0.007576] |
| macro_19 | XGBoost | 64 vs 40 | RMSE reduction | -0.000175 | 9/20 | [-0.001661, +0.001176] |
| macro_19 | XGBoost | 80 vs 40 | Spearman increase | -0.032038 | 2/20 | [-0.071084, +0.005966] |
| macro_19 | XGBoost | 80 vs 40 | RMSE reduction | -0.000910 | 4/20 | [-0.002599, +0.000689] |
| macro_19 | XGBoost | 80 vs 64 | Spearman increase | -0.006195 | 8/20 | [-0.022317, +0.010091] |
| macro_19 | XGBoost | 80 vs 64 | RMSE reduction | -0.000735 | 4/20 | [-0.001370, -0.000134] |

## Lineage consistency

| Model | Panel contrast | Lineages with higher Spearman | Lineages with lower RMSE |
|---|---|---:|---:|
| Ridge | 64 vs 40 | 8/19 | 12/19 |
| Ridge | 80 vs 40 | 8/19 | 11/19 |
| Ridge | 80 vs 64 | 9/19 | 6/19 |
| Neural network | 64 vs 40 | 7/19 | 11/19 |
| Neural network | 80 vs 40 | 8/19 | 10/19 |
| Neural network | 80 vs 64 | 10/19 | 7/19 |
| XGBoost | 64 vs 40 | 8/19 | 10/19 |
| XGBoost | 80 vs 40 | 7/19 | 8/19 |
| XGBoost | 80 vs 64 | 10/19 | 6/19 |

![Lineage effects](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_panel_comparison/lineage_comparison.png)

## Models and validation

- Ridge: the original 17-value alpha grid, training-only feature standardization, target centering and SVD solution.
- Neural network: original four width/penalty candidates (16/32 units and alpha 10/100), tanh activation, L-BFGS, and the same three initialization seeds per repeat. X/y scalers are fitted on the current training subset.
- XGBoost: original 36 candidates with depth 1/2/3, minimum child weight 5/20, L2 penalty 1/20 and 100/300/600 rounds; learning rate 0.03, row/column sampling 0.8, CPU hist and max_bin 64. Target scaling uses current training rows only.
- Each panel selects its settings using the minimum equal-weight mean five-fold inner RMSE. The outer test lineage and related identities are excluded from tuning and refitting.
- All 2,880 new outer models reconstructed their predictions; maximum absolute difference 1.11e-16.
- All 273,600 candidate-fold scores were reconstructed; maximum RMSE discrepancy 1.11e-16.
- Independent sklearn refits checked all three families and both expanded panels in Lung/Thyroid for repeats 1/20, including five selected inner refits for each.
- Recorded training warnings: 60480; convergence warnings: 0. Full diagnostics and selected settings are saved.

## Interpretation limits

- Repeated seeds reuse the same biological cohort and are not independent experiments. Bootstrap intervals do not refit models, are not multiplicity-adjusted and do not fully represent dependence from overlapping training sets.
- Gene panels were literature-defined before these fits, after earlier results on this cohort were seen. Some source papers also use CTRP/CCLE. These results are exploratory and require independent screening data for validation.
- All models preserve the same group-disjoint inner folds, which mix the represented training lineages. Inner tuning is not itself leave-lineage-out.
- The fixed grids compare specified pipelines. Boundaries in selected settings can motivate a separately planned follow-up; no outer-score-based grid expansion was performed here.
- The target remains negative reported CTRP erastin AUC. Upstream AUC processing provenance remains unresolved. This is prediction of the saved response labels, not a causal or clinical validation.

## Files

- [All scores](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_panel_comparison/summary.tsv)
- [Paired contrasts, including cross-model comparisons](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_panel_comparison/paired_comparisons.tsv)
- [Per-lineage paired effects](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_panel_comparison/lineage_paired_effects.tsv)
- [Selected settings](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_panel_comparison/selected_parameters.tsv)
- [Protocol](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_panel_comparison/protocol.json)
- [Validation](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_panel_comparison/validation.json)
- [Panel definitions](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_panel_comparison/panel_definitions.json)

- Warning summary: 60480 × DeprecationWarning: scipy.optimize: The `disp` and `iprint` options of the L-BFGS-B solver are deprecated and will be removed in SciPy 1.18.0.