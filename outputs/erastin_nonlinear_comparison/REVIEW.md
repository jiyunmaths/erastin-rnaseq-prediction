# Random-forest and neural-network comparison

The small neural-network ensemble on 40-gene expression had the best pooled point estimates, with only a marginal gain over expression ridge. Its paired Spearman and RMSE improvement intervals include zero. Neither nonlinear prediction head established an overall BulkFormer advantage on unseen lineages in this bounded search.

Completed nested leave-lineage-out evaluation on 745 cell lines (734 identity groups, 24 lineages), reusing the saved expression and BulkFormer representations. Ridge, random forest and a small neural-network ensemble use the same corrected outer holdouts and five inner folds.

**This is a model-family comparison on one fixed inner split scheme (seed 20261031), not a repeat over the previous 20 split seeds.** The three neural-network initializations are ensemble members, not independent validation datasets.

| Prediction head | Representation | Pooled Spearman ↑ | Pooled RMSE ↓ | Pooled R² ↑ | Mean within-lineage Spearman ↑ |
|---|---|---:|---:|---:|---:|
| Ridge | 40-gene expression | 0.4235 | 0.12088 | 0.1719 | 0.2303 |
| Ridge | Panel-focused BulkFormer | 0.3695 | 0.12407 | 0.1277 | 0.2305 |
| Ridge | All-gene BulkFormer | 0.3173 | 0.12612 | 0.0986 | 0.1846 |
| Random forest | 40-gene expression | 0.4004 | 0.12089 | 0.1718 | 0.2359 |
| Random forest | Panel-focused BulkFormer | 0.3533 | 0.12333 | 0.1381 | 0.1891 |
| Random forest | All-gene BulkFormer | 0.3284 | 0.12471 | 0.1187 | 0.1983 |
| Neural network (3-member mean) | 40-gene expression | 0.4288 | 0.12041 | 0.1784 | 0.2247 |
| Neural network (3-member mean) | Panel-focused BulkFormer | 0.3503 | 0.12896 | 0.0576 | 0.2636 |
| Neural network (3-member mean) | All-gene BulkFormer | 0.2820 | 0.13062 | 0.0331 | 0.2042 |

The lowest observed pooled RMSE among the nine fixed arms was 0.12041 for Neural network (3-member mean) on 40-gene expression. This is an exploratory ranking using held-out results, not a separately validated model-selection claim.

Panel-focused BulkFormer with a neural network had the highest mean within-lineage Spearman (0.2636), but its pooled RMSE was worse (0.12896). Improvements in within-lineage ranking did not translate into better overall response prediction.

The prespecified nested selector chose the model and representation using only inner-validation RMSE in each outer fold. Its pooled Spearman was 0.3362, RMSE 0.12487, and R² 0.1164; the expression ridge reference was 0.4235, 0.12088, and 0.1719.

![Model-family comparison](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_nonlinear_comparison/model_comparison.png)

## Paired comparisons against expression ridge

Positive improvement favors the first model. Intervals are conditional 95% whole-lineage bootstrap intervals; all models remain paired.

| Model | Spearman improvement [95% interval] | RMSE reduction [95% interval] |
|---|---:|---:|
| Ridge / Panel-focused BulkFormer | -0.05392 [-0.16604, +0.04148] | -0.00319 [-0.00899, +0.00193] |
| Ridge / All-gene BulkFormer | -0.10618 [-0.24268, +0.01790] | -0.00524 [-0.01068, +0.00074] |
| Random forest / 40-gene expression | -0.02302 [-0.07768, +0.03207] | -0.00001 [-0.00223, +0.00227] |
| Random forest / Panel-focused BulkFormer | -0.07019 [-0.15721, +0.00651] | -0.00245 [-0.00565, +0.00121] |
| Random forest / All-gene BulkFormer | -0.09501 [-0.19259, -0.00033] | -0.00383 [-0.00746, +0.00071] |
| Neural network (3-member mean) / 40-gene expression | +0.00533 [-0.00326, +0.01623] | +0.00047 [-0.00011, +0.00133] |
| Neural network (3-member mean) / Panel-focused BulkFormer | -0.07313 [-0.21891, +0.04785] | -0.00807 [-0.01879, +0.00054] |
| Neural network (3-member mean) / All-gene BulkFormer | -0.14141 [-0.30639, +0.01464] | -0.00974 [-0.01917, -0.00078] |
| Nested model/representation selector | -0.08723 [-0.17161, -0.01074] | -0.00399 [-0.00823, -0.00005] |

## Does panel-focused BulkFormer help within each prediction head?

| Head | Pooled Spearman change | Pooled RMSE reduction | Lineages with higher Spearman | Lineages with lower RMSE |
|---|---:|---:|---:|---:|
| Ridge | -0.05392 | -0.00319 | 10/19 | 9/19 |
| Random forest | -0.04717 | -0.00243 | 8/19 | 8/19 |
| Neural network (3-member mean) | -0.07846 | -0.00854 | 10/19 | 9/19 |

## Fixed search and validation

- Random forest: 128 trees; leaf sizes 5 or 15; per-split feature sampling sqrt(p) or 30% of features. Four candidates per outer fold; fixed forest seed.
- Neural network: one tanh hidden layer with 16 or 32 units; L2 alpha 10 or 100. Four candidates; each prediction averages three fixed initializations. L-BFGS, max_iter 2000, max_fun 40000, tolerance 0.001. No random early-stopping validation split.
- Neural-network X and target scaling are fitted on each current training subset only. Predictions are returned to the original negative-AUC scale before tuning and scoring.
- Both new heads select the minimum equal-weight mean inner-fold RMSE. Ridge predictions and tuning losses are reused from the exact matching original-seed corrected run.
- The nested selector chooses among all nine independently tuned head/representation arms using inner RMSE only. It selected random forest on panel-focused embeddings in 17 holdouts and ridge on panel-focused embeddings in seven, yet performed worse than expression ridge on the outer holdouts. Its choices are saved in nested_selection.tsv.
- Inner validation folds contain lineages also represented in their training folds. The selector result suggests that this tuning design may be a poor guide to unseen-lineage transfer; inner lineage-held-out tuning is a separate, untested alternative.
- All 72 neural outer fits selected alpha=100, the stronger regularization value in the fixed search. The expression networks selected 16 hidden units in all 24 holdouts. This does not establish that the tested regularization range is optimal.
- Related identity groups are kept together in every inner and outer split. Each held-out lineage is absent from supervised fitting and tuning.
- Macro summaries and lineage win counts use the 19 lineages with at least 10 samples. Five smaller lineages remain in pooled scores and detailed tables.
- All candidate validation predictions, losses, selected outer models and optimizer diagnostics are saved. An independent audit recomputes tuning decisions, reconstructs predictions, verifies train-only scaling, and refits the Lung holdout with separate sklearn code.
- Neural-network member fits with convergence warnings: 0/4536; selected outer members: 0/216. No candidate was removed because of an outer-test score.

## Interpretation limits

- This follow-up was designed after the ridge results were known. The search is bounded and is not an exhaustive comparison of all forests or neural networks.
- Bootstrap draws resample complete held-out lineages (24 pooled, 19 macro), with fixed predictions. Models are not retrained within draws, and training sets overlap across holdouts.
- Comparisons and intervals are descriptive; there is no multiplicity-adjusted claim of a best model. The nested selector provides a separate, training-only rule for selecting an arm.
- BulkFormer embeddings see the full transcriptome. Panel-focused pooling is not a 40-gene-only input model. Checkpoint pretraining overlap remains unresolved.
- The same archived CTRP AUC and DepMap 24Q4 data are used; no external drug screen or clinical cohort was added.

Audit and reproduction files: protocol.json, source_manifest.json, per_lineage_metrics.tsv, tuning_scores.tsv, predictions.npz, paired_comparisons.tsv, validation.json, and folds/ (selected fitted models, candidate validation predictions, diagnostics and checksums).
