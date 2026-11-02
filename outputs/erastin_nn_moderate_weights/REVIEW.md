# Moderate lineage weighting for the 64-gene neural network

**Neither moderate weighting scheme improved mean pooled or equal-lineage RMSE over the unweighted baseline.** Lowest mean pooled RMSE: Unweighted (0.119700). Lowest mean equal-lineage RMSE: Unweighted (0.120358).

Completed 960 new nested outer fits: two weighting schemes × 20 saved seeds × 24 held-out lineages. The comparison uses the same 745 cell lines, 734 identity groups, 64 genes, labels, inner/outer splits and three neural initializations per repeat as before. Verified unweighted predictions are the primary baseline; the previous full inverse-frequency predictions are included as context.

## Scores

Values are means of per-repeat scores, not scores after averaging predictions across repeats. Pooled scores include all 745 lines. Macro scores equally average the same 19 lineages with n≥10.

| Training weights | Pooled Spearman ↑ | Pooled RMSE ↓ | Pooled R² ↑ | Macro Spearman ↑ | Macro RMSE ↓ |
|---|---:|---:|---:|---:|---:|
| Unweighted | 0.43780 | 0.119700 | 0.18804 | 0.21011 | 0.120358 |
| Inverse square root | 0.42811 | 0.120159 | 0.18180 | 0.19994 | 0.121068 |
| Capped inverse, max 5 | 0.40252 | 0.121441 | 0.16425 | 0.18070 | 0.122676 |
| Full inverse frequency (previous) | 0.39759 | 0.121869 | 0.15834 | 0.17356 | 0.122968 |

![Weight comparison](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_nn_moderate_weights/weighting_comparison.png)

## Primary paired effects versus unweighted training

Positive improvement favors the new scheme: Spearman(new) − Spearman(unweighted), or RMSE(unweighted) − RMSE(new). Intervals use 5,000 paired whole-lineage bootstrap draws and are conditional on fitted predictions; they do not refit models or adjust for multiplicity. The same draws pair all arms and all repeats.

| Scheme | Scope | Metric | Mean improvement | Favorable seeds | Conditional 95% interval |
|---|---|---|---:|---:|---|
| Inverse square root | pooled_745 | spearman | -0.009688 | 0/20 | [-0.030897, +0.005513] |
| Inverse square root | pooled_745 | rmse | -0.000459 | 0/20 | [-0.001633, +0.000435] |
| Capped inverse, max 5 | pooled_745 | spearman | -0.035279 | 0/20 | [-0.079743, -0.005192] |
| Capped inverse, max 5 | pooled_745 | rmse | -0.001741 | 0/20 | [-0.004265, +0.000123] |
| Inverse square root | macro_19 | spearman | -0.010167 | 0/20 | [-0.040195, +0.018062] |
| Inverse square root | macro_19 | rmse | -0.000709 | 0/20 | [-0.001685, +0.000169] |
| Capped inverse, max 5 | macro_19 | spearman | -0.029408 | 0/20 | [-0.072045, +0.010632] |
| Capped inverse, max 5 | macro_19 | rmse | -0.002318 | 0/20 | [-0.004574, -0.000277] |

## Consistency across held-out lineages

| Scheme | Lineages with higher Spearman | Lineages with lower RMSE |
|---|---:|---:|
| Inverse square root | 10/19 | 8/19 |
| Capped inverse, max 5 | 7/19 | 6/19 |

These counts compare mean scores across seeds within each lineage. They are descriptive signs, not independently established lineage-specific effects.

![Lineage effects](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_nn_moderate_weights/lineage_effects.png)

## Training weights and controls

- **Inverse square root:** weights are proportional to 1/√n_lineage and normalized to mean one within each current fitting subset. Total lineage weight is proportional to √n_lineage.
- **Capped inverse:** final weights are min(c/n_lineage, 5), with c solved jointly with clipping so the weights average one. Thus the final normalized weight never exceeds five; uncapped lineages retain inverse-frequency ratios. The cap was fixed before new test predictions were inspected, with no cap search.
- All counts and normalizations use only current fitting rows, independently for each inner fit and outer refit. Every represented training lineage participates, including very small lineages.
- X/y StandardScaler fitting remains unweighted and training-only. Mean-one sample weights preserve the original L2 scaling.
- The original four width/penalty candidates are retained: 16/32 tanh units and alpha 10/100, L-BFGS, max_iter 2000, max_fun 40000, tolerance 0.001, and the same three initialization seeds per repeat. Member predictions are averaged before scoring.
- Selection remains the lowest unweighted equal-mean five-fold inner RMSE, with the original candidate order breaking ties. The held-out lineage and related identity groups are excluded from fitting/tuning.
- Weight schemes and cap are fixed experimental arms; no arm is selected separately using outer-test labels.

| Scheme | Outer per-cell weight range | Outer weight effective sample size range |
|---|---:|---:|
| Unweighted | 1.0000–1.0000 | 624.0–744.0 |
| Inverse square root | 0.5300–6.2247 | 528.4–628.6 |
| Capped inverse, max 5 | 0.2501–5.0000 | 363.0–432.7 |
| Full inverse frequency (previous) | 0.2336–32.3043 | 196.9–312.2 |

Weight effective sample size is (sum w)²/sum(w²), a concentration diagnostic rather than an independent biological sample count. Full inverse-frequency weights give equal lineage totals; the moderate schemes deliberately relax this.

## Validation

- All 960 new models reconstructed saved held-out predictions; maximum error 0.
- All 19,200 inner candidate-fold RMSE values and 5,760 actual fitting-subset weight vectors were reconstructed and checked.
- All 2,880 selected network members had their weighted training objective recomputed independently from predictions and coefficients; maximum discrepancy 5.55e-17.
- Preflight checked independent square-root and capped-weight formulas, permutation/label invariance, balanced-lineage unit weights, finite-difference gradients, and direct objective calculations. Current fitting code exactly replayed original unweighted and full inverse-frequency predictions for Lung/Thyroid in seed 1.
- Selected new settings were replayed for both schemes, Lung/Thyroid, seeds 1/20, and their five corresponding inner fits. These replay the same verified weighted adapter; finite-difference and objective checks provide independent mathematical validation.
- Convergence warnings: 0; recorded warnings: 60,480. Detailed categories are saved in validation.json and per-fit metadata.
- Exact shared split arrays, identity exclusions, seed triplets, input/source hashes, prior baseline scores, scaler values and bootstrap statistics versus explicitly expanded SciPy calculations were verified.

## Interpretation limits

This is an exploratory follow-up motivated by earlier results on the same cohort. Twenty seeds perturb tuning and initialization while reusing the biological samples; they are not independent biological validations. Intervals describe variation across observed held-out lineages conditional on the fitted models. They do not establish general performance across all unseen cancer types. Inner folds mix training lineages, as in the existing protocol; outer tests alone assess transfer to unseen lineages. Macro-19 scoring excludes the five smallest cohort lineages, whereas every represented lineage contributes during training. The upstream harmonized erastin AUC provenance limitation remains unresolved, and no independent response cohort was evaluated.

## Saved artifacts

- summary.tsv, per_repeat_metrics.tsv, per_repeat_macro_metrics.tsv and per_lineage_metrics.tsv: all four arms.
- paired_comparisons.tsv and paired_seed_differences.tsv: primary comparisons and secondary comparisons against full inverse weighting and between the two new schemes.
- lineage_paired_effects.tsv and lineage_win_counts.tsv: held-out lineage consistency.
- training_subset_weight_audit.tsv and training_lineage_weights.tsv: exact weights and concentration diagnostics.
- selected_parameters.tsv, tuning_scores.tsv, model_training_objective_checks.tsv, validation.json, preflight_validation.json and protocol.json: settings and audits.
- predictions.npz, bootstrap_draws.npz, inputs.npz, shared_splits.npz and folds/: matched inputs, all predictions and all 960 newly fitted models.
- Source scripts, pinned environment versions, source manifests and checksummed artifact manifest.
