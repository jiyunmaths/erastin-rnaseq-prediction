# EMT and DNA repair gene additions for CTRP erastin prediction

**Culture variables are excluded.** The full Hallmark EMT and DNA-repair sets selected by the user were added to the fixed 64-gene ferroptosis panel. All 300 models passed validation: 180 new expanded-panel fits plus 120 unchanged five-/64-gene reference models.

Expanded panels reduced mean pooled RMSE in **5/9** same-family comparisons with 64 genes. Conditional unadjusted 95% intervals: 0 entirely favor expansion, 2 entirely favor the baseline, and 7 include zero. Lowest observed mean RMSE across all 15 fitted arms: **Elastic Net, 64 + both**, 0.116265; Spearman 0.4748. Observed ranking is descriptive.

## Fixed panels and verified mappings

| Panel | Genes | Additions beyond 64 |
|---|---:|---:|
| Original five-gene reference | 5 | — |
| Ferroptosis baseline | 64 | 0 |
| Baseline + Hallmark EMT | 264 | 200 |
| Baseline + Hallmark DNA Repair | 213 | 149 |
| Baseline + both Hallmarks | 413 | 349 |

All **200/200 EMT** and **150/150 DNA-repair** genes have expression columns, with no missing values in any retained line. GPX4 belongs to the DNA-repair Hallmark and the original 64-gene panel, so it is included once. The selected EMT and repair sets otherwise do not overlap. Original five- and 64-gene arrays are bit-identical to the saved baseline inputs.

Gene membership comes from the official current [Hallmark EMT](https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION.html) and [Hallmark DNA Repair](https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_DNA_REPAIR.html) snapshots downloaded on 2026-11-02. JSON membership and the HTML table of mapped NCBI identifiers agree. Download hashes and snapshots are saved; no release number was inferred from a copyright date.

**Mapping exception resolved:** DepMap has both BOLA2B (552900) and BOLA2 (552900) columns. The release Gene.csv identifies BOLA2 as HGNC:29488 / ENSG00000183336 / Entrez 552900, while BOLA2B has a distinct HGNC/Ensembl/Entrez identity. We retain the exact BOLA2 (552900) column; the mislabeled paralog column is not substituted or averaged. Every choice and candidate column is in gene_mapping.tsv.

Hallmark sets describe coherent expression programs, not comprehensive mechanistic inventories. The selected repair set includes RAD51, RAD52, ERCC1–5, PCNA and TP53, but does not contain BRCA1/BRCA2; the EMT set includes VIM, FN1 and CDH2 but not ZEB1/ZEB2. No unselected core-gene extras were silently added.

## Biological rationale

EMT is a reasonable hypothesis because a mesenchymal therapy-resistant state has been linked to dependence on lipid-peroxide detoxification, particularly GPX4 inhibition ([Viswanathan et al., Nature 2017](https://www.nature.com/articles/nature23007)). This is a rationale to test expression context, not proof that every EMT-associated gene predicts erastin response.
DNA-repair context may matter, but inducer specificity is important: a primary BRCA1 study found that BRCA1 deficiency promotes erastin resistance while increasing GPX4-inhibitor sensitivity ([Lei et al., Cancer Discovery 2024](https://pubmed.ncbi.nlm.nih.gov/38552003/)). Its BRCA1 result motivates a broader future repair-gene experiment; this Hallmark-only comparison does not directly test BRCA1, BRCA2 or functional homologous-recombination deficiency.

## Held-out accuracy

| Family | Input panel | Genes | Mean RMSE ↓ | Mean Spearman ↑ | Mean R² ↑ | Macro-23 RMSE ↓ |
|---|---|---:|---:|---:|---:|---:|
| Ridge | 5 genes | 5 | 0.120231 | 0.4098 | 0.1644 | 0.112458 |
| Ridge | 64 genes | 64 | 0.117005 | 0.4606 | 0.2087 | 0.110655 |
| Ridge | 64 + EMT | 264 | 0.117011 | 0.4591 | 0.2089 | 0.109231 |
| Ridge | 64 + DNA repair | 213 | 0.116924 | 0.4641 | 0.2101 | 0.109818 |
| Ridge | 64 + both | 413 | 0.116588 | 0.4621 | 0.2143 | 0.108667 |
| Elastic Net | 5 genes | 5 | 0.120222 | 0.4097 | 0.1646 | 0.112425 |
| Elastic Net | 64 genes | 64 | 0.117205 | 0.4600 | 0.2062 | 0.111193 |
| Elastic Net | 64 + EMT | 264 | 0.116798 | 0.4562 | 0.2117 | 0.109345 |
| Elastic Net | 64 + DNA repair | 213 | 0.116610 | 0.4705 | 0.2143 | 0.110550 |
| Elastic Net | 64 + both | 413 | 0.116265 | 0.4748 | 0.2190 | 0.109864 |
| Neural network | 5 genes | 5 | 0.120052 | 0.4097 | 0.1669 | 0.112192 |
| Neural network | 64 genes | 64 | 0.116968 | 0.4618 | 0.2094 | 0.110514 |
| Neural network | 64 + EMT | 264 | 0.119076 | 0.4277 | 0.1793 | 0.111301 |
| Neural network | 64 + DNA repair | 213 | 0.118247 | 0.4465 | 0.1914 | 0.110433 |
| Neural network | 64 + both | 413 | 0.122613 | 0.4127 | 0.1293 | 0.114515 |

Each score is computed on a 149-line test set, then averaged over the 20 splits. The target remains negative harmonized CTRP erastin AUC; higher means more sensitive. No clipping, response redefinition, culture predictor or extra cohort exclusion. Macro-23 equally weights lineage-specific RMSEs; Macro-19 equally weights the 19 lineages with at least 10 cohort lines. Full Pearson, MAE, R², repeat and lineage tables are saved.

![Model comparison](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_emt_ddr_comparison/model_comparison.png)

## Paired gains versus the 64-gene baseline

| Family | Added program | Mean RMSE reduction | Relative reduction | Favorable splits | Favorable lineages | Conditional 95% interval |
|---|---|---:|---:|---:|---:|---|
| Ridge | EMT | -0.000006 | -0.005% | 11/20 | 12/23 | [-0.001458, +0.001770] |
| Ridge | DNA repair | +0.000081 | +0.069% | 10/20 | 13/23 | [-0.001340, +0.001678] |
| Ridge | both | +0.000416 | +0.356% | 13/20 | 14/23 | [-0.001400, +0.002560] |
| Elastic Net | EMT | +0.000407 | +0.347% | 13/20 | 12/23 | [-0.000766, +0.001782] |
| Elastic Net | DNA repair | +0.000596 | +0.508% | 13/20 | 14/23 | [-0.000735, +0.002031] |
| Elastic Net | both | +0.000941 | +0.802% | 14/20 | 15/23 | [-0.000295, +0.002413] |
| Neural network | EMT | -0.002109 | -1.803% | 4/20 | 10/23 | [-0.004371, -0.000032] |
| Neural network | DNA repair | -0.001280 | -1.094% | 7/20 | 9/23 | [-0.003333, +0.001045] |
| Neural network | both | -0.005645 | -4.826% | 3/20 | 7/23 | [-0.008806, -0.002662] |

Positive improvement favors the expanded panel. Intervals crossing zero do not establish equivalence. A favorable lineage has a positive mean RMSE reduction; tiny lineage test sets make these counts descriptive. All nine primary effects are exploratory and unadjusted for multiplicity.

![Paired effects](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_emt_ddr_comparison/paired_pathway_effects.png)

## Does combining both programs improve on each alone?

| Family | Both versus | Mean RMSE reduction | Favorable splits | Conditional 95% interval |
|---|---|---:|---:|---|
| Ridge | 64 + EMT | +0.000422 | 12/20 | [-0.000757, +0.001582] |
| Ridge | 64 + DNA repair | +0.000335 | 14/20 | [-0.000790, +0.001529] |
| Elastic Net | 64 + EMT | +0.000533 | 15/20 | [-0.000655, +0.001731] |
| Elastic Net | 64 + DNA repair | +0.000345 | 13/20 | [-0.000296, +0.001108] |
| Neural network | 64 + EMT | -0.003536 | 0/20 | [-0.005789, -0.001336] |
| Neural network | 64 + DNA repair | -0.004365 | 2/20 | [-0.006583, -0.002589] |

![Lineage effects](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_emt_ddr_comparison/lineage_pathway_effects.png)

## Fixed fitting and validation

- 744 lines, 733 identity groups, 23 lineages; Cervix remains excluded. All exact prior 20 masks, five inner folds and NN initialization triplets are reused. Every split has 595 training / 149 testing lines, all 23 lineages on both sides and no identity-group overlap.
- Every inner training set retains all 23 lineages. The single outer-training Ampulla line is always inner training and never inner validation; inner validation has 22 lineages. Hyperparameter choice uses only current inner validation outcomes.
- All models are unweighted. Training-only StandardScaler for X; target centering for ridge and training-only y standardization for Elastic Net/NN. No response-based external gene filtering, all-cohort scaler, lineage predictor or culture feature.
- Independently re-extracted all 307,272 expanded expression entries (744 lines × 413 unique genes) directly from the original checksum-verified CSV. Every panel matches exactly under the original float32-storage convention; the chosen BOLA2 column was verified explicitly. See raw_expression_validation.json.
- The grids and solver settings are unchanged from the saved CTRP gene-only models. Ridge: 17 alphas from 1e-4 to 1e4. Elastic Net: 41 alphas from 10 to 1e-4 × L1 ratios .01/.05/.1/.25/.5/.75/.9/1; tol 1e-9, max_iter 50,000. NN: 16/32 tanh units × alpha 10/100; three initialization members averaged, L-BFGS, tol .001, max_iter 2,000, max_fun 40,000, no early stopping.
- Minimize the equal-mean five-fold RMSE in original outcome units, with first grid candidate breaking ties. The same finite grid is separately tuned for each panel/family/repeat; this comparison does not establish the best possible architecture for a 413-gene panel.
- Reconstructed 174,500 candidate-fold scores and 1,800 outer/inner identity exclusions. Maximum prediction-reconstruction error 2.22e-16, inner-RMSE error 1.67e-16. Original baseline model files/predictions/mean scores reproduce exactly.
- Recomputed all 300 selected NN-member unweighted objectives, maximum error 5.55e-17; checked 100 selected Elastic Net objectives and recorded optimality residuals for 164,000 inner candidate solutions, maximum residual 7.47e-10.
- Native sklearn ridge/NN fits and augmented-row Lasso refits independently verify the retained pilot models. EN paths/native fits are checked across every mixing ratio and representative alphas. Selected inner/outer settings for every expanded arm are independently replayed in repeats 1 and 20.
- Convergence warnings: 0 across candidate/selected fits; selected outer convergence warnings: 0. Full warning categories and selected settings are saved.
- Confidence intervals use 5,000 paired whole-lineage bootstrap draws, shared across models and repeat-specific test sets. They condition on trained models and overlapping holdouts; no refitting, multiplicity or post-selection correction. Expanded-sample SciPy arithmetic checks passed, including constant-prediction controls.

## Interpretation limits

New gene membership is fixed from external Hallmark definitions, but these cell lines have already informed many modelling choices. The five-gene reference was itself selected after prior held-out importance on these splits. This is an exploratory internal benchmark, not an untouched final test cohort. Twenty overlapping splits are not independent biological replications. The design evaluates new lines from represented lineages, not unseen-lineage, patient or clinical transfer.
Adding a large gene set can improve prediction by providing more predictors without demonstrating pathway-specific biology. Matched-size random gene controls and fresh independent response data would be needed to assess that specificity. EMT expression in hematologic lines should not be interpreted as evidence that they underwent EMT, and repair-gene RNA does not measure DNA-repair competence or mutation status.
The unchanged CTRP outcome comes from a previously checksum-verified harmonized redistribution archive ([recorded source](https://doi.org/10.5281/zenodo.22125121)); its exact upstream AUC generation recipe remains unverified. Erastin response must not be treated as interchangeable with sensitivity to direct GPX4 inhibitors.

## Files

- summary.tsv, paired_comparisons.tsv, paired_seed_differences.tsv, per_lineage_metrics.tsv, lineage_paired_effects.tsv and lineage_win_counts.tsv: accuracy, uncertainty and consistency.
- panel_inputs.npz, panel_definitions.json, gene_mapping.tsv, coverage_summary.json, sources/: exact selected genes, stable identifiers, inputs, mappings and source snapshots.
- folds/, selected_parameters.tsv, tuning_scores.tsv: all saved models, predictions and tuning decisions. elastic_net_coefficients.tsv and elastic_net_selection.tsv describe fitted coefficients/selection, not causal importance.
- shared_splits.npz, split_lineage_counts.tsv, inner_split_lineage_counts.tsv and sample_test_coverage.tsv: unchanged memberships and coverage.
- protocol.json, source_manifest.json, preflight_validation.json, validation.json, visual_qa.json, artifact_manifest.json and scripts: fixed choices, provenance, independent checks and checksums.
