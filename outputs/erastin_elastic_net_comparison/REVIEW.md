# Exploratory Elastic Net comparison

Lowest mean pooled RMSE among the six arms: **Neural network (64)**, 0.116968. All arms use the exact same 20 lineage-stratified outer splits and five inner folds.

**Elastic Net provides no clear accuracy improvement here.** With 64 input genes, mean pooled RMSE is 0.171% higher than ridge and 0.203% higher than NN, although Elastic Net wins 13/20 and 11/20 individual splits respectively. Its mean macro-23 RMSE is also higher, and paired conditional intervals include zero. The five-gene Elastic Net is numerically close to five-gene ridge. Its selected L1 ratio is 0.01 in all 20 repeats, retaining every input gene. The 64-gene models are sparser (median 37 retained, range 17–58), but the selected subsets vary across splits.

**Five-gene panel:** SLC7A11, GPX4, ABCC1, ACSL4, AIFM2. These genes were chosen after examining held-out importance from this same cohort and these same splits. Test responses therefore influenced panel selection. Training-only scaling and nested tuning do not remove that earlier bias. This is an exploratory comparison; its intervals do not adjust for panel or model selection.

## Test performance

| Model | Genes | Pooled Spearman ↑ | Pooled RMSE ↓ | Pooled R² ↑ | Macro-23 RMSE ↓ | Macro-19 RMSE ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Ridge | 5 | 0.40979 | 0.120231 | 0.16443 | 0.112458 | 0.117511 |
| Ridge | 64 | 0.46062 | 0.117005 | 0.20874 | 0.110655 | 0.115007 |
| Elastic Net | 5 | 0.40965 | 0.120222 | 0.16462 | 0.112425 | 0.117501 |
| Elastic Net | 64 | 0.46005 | 0.117205 | 0.20619 | 0.111193 | 0.115165 |
| Neural network | 5 | 0.40971 | 0.120052 | 0.16692 | 0.112192 | 0.117439 |
| Neural network | 64 | 0.46175 | 0.116968 | 0.20941 | 0.110514 | 0.114937 |

Each metric is computed within a repeat’s test set and then averaged over 20 repeats. Macro-23 RMSE gives every lineage equal weight. Macro-19 is secondary and restricts to 19 lineages with at least 10 cohort samples. No macro Spearman is reported because some test lineage subsets contain one line.

![Model comparison](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_elastic_net_comparison/model_comparison.png)

## Paired Elastic Net contrasts

| Comparison | Mean pooled RMSE reduction | RMSE reduction (%) | Favorable splits | Favorable lineages | Conditional 95% interval |
|---|---:|---:|---:|---:|---|
| Elastic Net (5) vs Ridge (5) | +0.000009 | +0.008% | 9/20 | 10/23 | [-0.000038, +0.000066] |
| Elastic Net (64) vs Ridge (64) | -0.000201 | -0.171% | 13/20 | 11/23 | [-0.000650, +0.000262] |
| Elastic Net (5) vs Neural network (5) | -0.000170 | -0.142% | 9/20 | 10/23 | [-0.000680, +0.000304] |
| Elastic Net (64) vs Neural network (64) | -0.000238 | -0.203% | 11/20 | 10/23 | [-0.000772, +0.000268] |
| Elastic Net (5) vs Elastic Net (64) | -0.003016 | -2.574% | 3/20 | 11/23 | [-0.006315, +0.000456] |

Positive improvement favors the first model. Favorable lineages count mean lineage-specific RMSE reductions across repeats; this is descriptive, particularly for one- to three-line test subsets.

| Comparison | Metric | Mean improvement | Favorable splits | Conditional 95% interval |
|---|---|---:|---:|---|
| Elastic Net (5) vs Ridge (5) | pooled_spearman | -0.000140 | 10/20 | [-0.000831, +0.000779] |
| Elastic Net (5) vs Ridge (5) | macro23_rmse | +0.000033 | 10/20 | [-0.000033, +0.000105] |
| Elastic Net (64) vs Ridge (64) | pooled_spearman | -0.000571 | 14/20 | [-0.009761, +0.007697] |
| Elastic Net (64) vs Ridge (64) | macro23_rmse | -0.000538 | 9/20 | [-0.001446, +0.000279] |
| Elastic Net (5) vs Neural network (5) | pooled_spearman | -0.000056 | 7/20 | [-0.009026, +0.006970] |
| Elastic Net (5) vs Neural network (5) | macro23_rmse | -0.000232 | 7/20 | [-0.000852, +0.000399] |
| Elastic Net (64) vs Neural network (64) | pooled_spearman | -0.001702 | 14/20 | [-0.011698, +0.006312] |
| Elastic Net (64) vs Neural network (64) | macro23_rmse | -0.000679 | 8/20 | [-0.001574, +0.000153] |
| Elastic Net (5) vs Elastic Net (64) | pooled_spearman | -0.050396 | 4/20 | [-0.098691, +0.000921] |
| Elastic Net (5) vs Elastic Net (64) | macro23_rmse | -0.001232 | 8/20 | [-0.004799, +0.002425] |

Intervals use 5,000 paired whole-lineage bootstrap draws, shared across arms and all repeat-specific test sets. They are conditional on fitted models, fixed splits and chosen panels. No refitting, feature reselection or multiple-comparison adjustment occurs. A zero-crossing interval does not establish equivalence or noninferiority; no equivalence margin was defined.

![Paired differences](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_elastic_net_comparison/paired_differences.png)

## Sparsity

| Input panel | Mean retained | Median retained | Retained range | Selected alpha range | Selected L1 ratio range |
|---|---:|---:|---:|---:|---:|
| 5 | 5.00 | 5 | 5–5 | 0.0237137–0.0749894 | 0.01–0.01 |
| 64 | 36.50 | 37 | 17–58 | 0.0316228–0.562341 | 0.01–1 |

Nonzero means a literal nonzero solver coefficient. This summarizes model compression, not a biologically validated gene set. L1 ratio 0 is pure ridge and 1 is pure Lasso; the fitted grid starts at 0.01. The saved ridge arms supply the L2 comparator. Elastic Net alpha numbers cannot be compared directly with Ridge alpha because objective normalization and target scaling differ.

### 5-gene panel: all five genes

| Gene | Selected fits | Positive / negative fits | Mean signed coefficient | Mean absolute coefficient |
|---|---:|---:|---:|---:|
| SLC7A11 | 20/20 | 0 / 20 | -0.035085 | 0.035085 |
| AIFM2 | 20/20 | 0 / 20 | -0.026162 | 0.026162 |
| ACSL4 | 20/20 | 20 / 0 | +0.023959 | 0.023959 |
| GPX4 | 20/20 | 0 / 20 | -0.017883 | 0.017883 |
| ABCC1 | 20/20 | 20 / 0 | +0.014100 | 0.014100 |

Coefficients are original target units per one training-set expression SD. Positive predicts higher sensitivity (negative harmonized erastin AUC), conditional on the other panel genes. Correlated features can share or exchange coefficients; coefficient magnitude and frequency are not causal effects or significance tests.

### 64-gene panel: top 20 by selection frequency, then coefficient magnitude

| Gene | Selected fits | Positive / negative fits | Mean signed coefficient | Mean absolute coefficient |
|---|---:|---:|---:|---:|
| SLC7A11 | 20/20 | 0 / 20 | -0.019322 | 0.019322 |
| ACSL4 | 20/20 | 20 / 0 | +0.011969 | 0.011969 |
| AIFM2 | 20/20 | 0 / 20 | -0.009434 | 0.009434 |
| GPX4 | 20/20 | 0 / 20 | -0.009180 | 0.009180 |
| ACSL3 | 20/20 | 0 / 20 | -0.008705 | 0.008705 |
| ABCC1 | 20/20 | 20 / 0 | +0.008540 | 0.008540 |
| LAMP2 | 20/20 | 0 / 20 | -0.008150 | 0.008150 |
| CYB5R1 | 20/20 | 0 / 20 | -0.008061 | 0.008061 |
| CTSD | 20/20 | 0 / 20 | -0.007402 | 0.007402 |
| FADS1 | 20/20 | 20 / 0 | +0.007315 | 0.007315 |
| GCLC | 20/20 | 0 / 20 | -0.006195 | 0.006195 |
| FADS2 | 20/20 | 20 / 0 | +0.003922 | 0.003922 |
| ATF4 | 20/20 | 20 / 0 | +0.003886 | 0.003886 |
| GSR | 19/20 | 0 / 19 | -0.005537 | 0.005537 |
| LRP8 | 19/20 | 0 / 19 | -0.005137 | 0.005137 |
| GOT1 | 19/20 | 19 / 0 | +0.005072 | 0.005072 |
| DHFR | 19/20 | 19 / 0 | +0.004104 | 0.004104 |
| SLC3A2 | 18/20 | 0 / 18 | -0.004076 | 0.004076 |
| LPCAT3 | 18/20 | 18 / 0 | +0.004054 | 0.004054 |
| PROM2 | 18/20 | 0 / 18 | -0.003836 | 0.003836 |

Coefficients are original target units per one training-set expression SD. Positive predicts higher sensitivity (negative harmonized erastin AUC), conditional on the other panel genes. Correlated features can share or exchange coefficients; coefficient magnitude and frequency are not causal effects or significance tests.

![Elastic Net gene selection](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_elastic_net_comparison/elastic_net_selection.png)

## Hyperparameters

| Input panel | Alpha | L1 ratio | Outer fits |
|---|---:|---:|---:|
| 5 | 0.023713737 | 0.01 | 1 |
| 5 | 0.031622777 | 0.01 | 5 |
| 5 | 0.04216965 | 0.01 | 7 |
| 5 | 0.056234133 | 0.01 | 3 |
| 5 | 0.074989421 | 0.01 | 4 |
| 64 | 0.031622777 | 1 | 1 |
| 64 | 0.04216965 | 1 | 1 |
| 64 | 0.056234133 | 0.75 | 1 |
| 64 | 0.13335214 | 0.25 | 2 |
| 64 | 0.17782794 | 0.1 | 1 |
| 64 | 0.23713737 | 0.05 | 1 |
| 64 | 0.23713737 | 0.1 | 4 |
| 64 | 0.31622777 | 0.05 | 5 |
| 64 | 0.4216965 | 0.05 | 2 |
| 64 | 0.56234133 | 0.01 | 2 |

Panel 5: lowest-alpha boundary selected in 0/20 fits, highest-alpha boundary in 0/20, lowest L1 ratio in 20/20 and pure Lasso in 0/20. The grid was fixed before fitting and was not expanded after examining test performance.

Panel 64: lowest-alpha boundary selected in 0/20 fits, highest-alpha boundary in 0/20, lowest L1 ratio in 2/20 and pure Lasso in 2/20. The grid was fixed before fitting and was not expanded after examining test performance.

![Lineage comparison](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_elastic_net_comparison/lineage_comparison.png)

## Design and validation

- 744 cell lines, 733 identity groups, 23 lineages after excluding the singleton Cervix line. Every repeat has 595 training and 149 testing lines, all 23 lineages on each side and zero identity-group overlap.
- All six arms use unweighted training. The 80 saved ridge/NN models are copied and hash-verified; 40 new Elastic Net models were fitted. All copied baseline predictions and scores exactly reproduce the prior five-gene comparison.
- Elastic Net scales X and y using only current training rows. Each inner fold fits its own scalers. Tune 41 explicit alpha values from 10 to 0.0001 times L1 ratios 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 1, selecting the lowest equal-mean five-fold RMSE in original target units. Ties select the first grid member.
- Objective on standardized X/y: squared error/(2n) + alpha × L1 ratio × sum(abs(beta)) + alpha × (1 − L1 ratio) × sum(beta²)/2, with an unpenalized intercept. Inner paths use training-only Gram matrices and exact centering. Selected outer models use native sklearn ElasticNet; cyclic coordinate descent, tolerance 1e-9, max_iter 50,000.
- Ridge retains the existing 17-alpha grid and training-only standardization. NN retains the existing 16/32 tanh units × alpha 10/100 grid, three initialization seeds and L-BFGS settings. Their tuning and saved models are unchanged.
- Ampulla has one outer-training line, retained in every inner training fit and excluded from inner validation. Thus inner training has 23 lineages and inner validation has 22; outer Ampulla testing stays untouched.
- Audited all 120 outer models and reconstructed 69,800 candidate-fold RMSE scores. Saved prediction max error 1.11e-16; inner RMSE max error 8.33e-17.
- Verified 720 outer/inner identity exclusions, current-training scalers, unweighted NN objectives for 120 ensemble members, all source hashes and byte-identical splits.
- Checked Elastic Net optimality conditions for all 65,600 inner candidate solutions and 40 selected outer fits; maximum residual 4.74e-10. Recomputed all 40 selected Elastic Net objectives.
- Native independent ElasticNet refits and an equivalent augmented-row Lasso reproduce representative inner paths across all eight mixing ratios and alpha endpoints. An independent augmented Lasso also reproduces selected inner and outer models for both panels in repeats 1 and 20.
- Convergence warnings: 0; recorded warnings overall: 2,520. Bootstrap arithmetic checked against explicitly expanded SciPy samples.

## Interpretation limits

These repeats test new cell lines from represented cancer lineages. They do not assess transfer to absent lineages. The same biological samples recur across overlapping training/test sets; 20 seeds are not 20 independent biological validations. Only 724/744 lines occur in at least one test set. The full dataset has informed earlier model and panel choices, especially the five-gene panel, so a fresh cohort or prospectively reserved test set is needed for confirmation. The upstream harmonized erastin AUC provenance limitation remains unresolved.

## Saved artifacts

- summary.tsv, paired_comparisons.tsv, paired_seed_differences.tsv: means, conditional intervals and split consistency.
- per_repeat_pooled_metrics.tsv, per_repeat_macro_metrics.tsv, per_lineage_metrics.tsv, lineage_paired_effects.tsv, lineage_win_counts.tsv: complete split and lineage scores.
- elastic_net_coefficients.tsv, elastic_net_gene_selection.tsv, elastic_net_sparsity_summary.tsv: all coefficients and selection frequencies.
- selected_parameters.tsv, tuning_scores.tsv, parameter_counts.tsv, elastic_net_solver_checks.tsv: tuning and numerical diagnostics.
- protocol.json, preflight_validation.json, validation.json: frozen fitting choices and completed audit.
- panel_inputs.npz, shared_splits.npz, predictions.npz, bootstrap_draws.npz, folds/: data, saved splits, all fitted models and held-out predictions.
- source_metadata/, source_manifest.json, reuse_manifest.json, analysis scripts and artifact_manifest.json: exact reuse provenance, reproducibility and checksums.
