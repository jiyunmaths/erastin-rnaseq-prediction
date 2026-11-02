# Genome-wide Elastic Net feature selection beyond the retained core64

Results and interpretation: [REVIEW.md](REVIEW.md).

This experiment searches the existing 19,176 unique measured protein-coding RNA
features, including 19,112 genes outside the retained 64. It reuses the frozen
expression mapping, cohorts and all 20 identity-group/lineage-stratified outer
holdouts from `../erastin_genomewide_search/`. All five original inner folds
are reused, including the original singleton-lineage training rule. No culture
covariates, lineage predictors or sample weighting enter prediction.

Elastic Net is the **selector**. Each current training subset independently
projects candidate expression and the response off the core64 and training
lineage design, scales residuals, then fits all eligible outside genes jointly.
Only training constants and numerical aliases are excluded. There is no
preliminary marginal-response shortlist. Alpha is defined relative to that
training subset's null-solution threshold. Four L1 ratios and five relative
penalties give 20 selector settings.

Ridge predicts from the original core64 plus the selected outside genes. It
uses fresh training-only scaling and no lineage/culture input. The selector
setting and 17 ridge penalties are jointly chosen by mean five-fold inner RMSE.
An additional family of 17 core-only ridge settings supplies a zero-addition
fallback: 357 candidates per inner fold, 71,400 candidate-fold scores across
both assays. After tuning, selection is repeated on outer training alone.
No fixed top-K cap is applied. All 64 original features remain downstream
inputs even if their fitted ridge coefficient is small.

The working-set solver only accelerates the same multivariate Elastic Net
objective. It repeatedly expands the active set using global KKT violations;
every eligible gene is checked at every penalty. Restricted coordinate descent,
full-genome KKT residuals and dual-gap checks must all pass. The preflight checks
native full-genome ElasticNet coefficients/selected genes, augmented-Lasso
equivalence, independent least-squares residualization, native downstream ridge,
all saved splits and a validation-only decoy gene.

Use the pinned original analysis environment from the project root:

```bash
export PYTHONPATH="$PWD/work/bulkformer/packages"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/work/bulkformer/matplotlib_cache"
export PYTHONDONTWRITEBYTECODE=1
PYTHON=/home/sji/miniforge3/envs/deeplearning/bin/python
$PYTHON outputs/erastin_genomewide_elastic_net/validate_preflight.py
$PYTHON outputs/erastin_genomewide_elastic_net/run_comparison.py
$PYTHON outputs/erastin_genomewide_elastic_net/review_comparison.py
```

The review supports `--wait` during fitting. Inspect both generated figures,
record `visual_qa.json`, then use `--seal` to finalize the artifact manifest.
Frozen protocol/source checks prevent incompatible resumption.

`gene_selection_stability.tsv` reports genes chosen by the selected outer model,
not genes that happen to be nonzero in any searched setting. Counts have a
denominator of 20 per assay, including core-only fits. Selector and downstream
ridge coefficient summaries are conditional on selection and use different
scales. They are regularized associations, not causal importance. Full solver
diagnostics and paired lineage effects remain saved locally.

The pre-existing core64 ridge, Elastic Net, NN, TabPFN and whole-transcriptome
ridge predictions are reused unchanged as references. The retained registry
stays unchanged. This experiment adds 40 outer ridge models and their nested
feature-selection paths; it does not refit the neural network/TabPFN or directly
fit an unconstrained genome-wide Elastic Net predictor.

The cohorts have informed earlier model searches. Nesting controls direct
feature-selection leakage but does not create an untouched confirmatory test.
Selection frequencies come from overlapping training sets, and the two assays
share cell lines. A consensus fixed panel assembled from all repeats needs
separate validation. Noncoding RNA, DNA variants, patient prediction and
unseen-lineage generalization are outside this benchmark.
