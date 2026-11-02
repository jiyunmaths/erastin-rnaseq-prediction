# Reproduce the hierarchical lineage-slope benchmark

Results are in [REVIEW.md](REVIEW.md). This experiment extends the previous
random-intercept model; it does not repeat that model under another name.

For each assay separately, the model is

`response = global intercept + shared gene effects + lineage offset + local gene effects`.

All 64 gene slopes can deviate by lineage, with Gaussian shrinkage toward shared
slopes. Local deviations use expression centered on each lineage's current
training mean. The original global expression scaling is used for both shared
and local slopes. A lineage with one training line has zero local slope design
and exactly zero fitted slope deviation.

The convex penalized fit is a Gaussian mixed-model MAP estimate with prior/noise
ratios tuned by inner CV. It is not full Bayesian posterior sampling. The dual
kernel is equivalent to ridge on explicit lineage-by-gene features; preflight
independently checks all 425 settings in both assays against that representation.

- Same 744 CTRP lines/23 lineages and 386 PRISM lines/20 lineages, 64 genes,
  identity groups, 20 outer 80/20 holdouts and five inner folds.
- Every evaluated lineage appears in outer training and testing. Identity groups
  remain disjoint; CTRP Cervix remains excluded, and Ampulla's single training
  identity group is always inner training.
- All scalers and lineage means fit only the current training subset. Training
  remains unweighted. No culture covariates, expanded genes or assay pooling.
- Fixed grid: 17 global penalties, five intercept variance weights and five
  local-slope variance weights (425 settings). Three arms use this same path:
  matched intercept-only (85), slopes with zero-slope fallback (425), and required
  positive slope variance (340).
- The older intercept-only model uses a different intercept-penalty grid and is
  shown separately. Ridge, Elastic Net, NN, SVR, kernel ridge and lineage-only
  controls are also reused. There are 120 new arm/repeat bundles and 280 verified
  baseline bundles; identical selected fits may serve multiple reported arms.

From the project root, with the pinned environment:

```bash
export PYTHONPATH="$PWD/work/bulkformer/packages"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/work/bulkformer/matplotlib_cache"
export PYTHONDONTWRITEBYTECODE=1
python outputs/erastin_hierarchical_slopes/validate_preflight.py
python outputs/erastin_hierarchical_slopes/run_benchmark.py
python outputs/erastin_hierarchical_slopes/review_benchmark.py
```

The runner uses six workers and resumes checksummed completed repeats. Fitting
code and grids are frozen before fitting. Analysis hashes are recorded separately.
Inspect the figure, record `visual_qa.json`, then run `review_benchmark.py --seal`
to produce the artifact manifest. Changed model specifications belong in a new
experiment directory.

`lineage_effect_norms.tsv` summarizes offset and local-slope magnitude by lineage;
these descriptive fitted effects are not causal gene-importance estimates or
posterior evidence for biological heterogeneity. `selected_parameters.tsv` records
effective degrees of freedom and variance choices. All comparisons remain
exploratory on repeatedly inspected, overlapping holdouts. No unseen-lineage or
patient-performance claim is made. The retained registry remains unchanged.
