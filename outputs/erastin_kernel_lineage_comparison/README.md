# Reproduce the kernel and lineage benchmark

The scientific results are in [REVIEW.md](REVIEW.md). The retained model registry
is unchanged; these are additional exploratory candidates.

Both assays keep their original RNA inputs, response values, identity groups,
20 outer 80/20 splits and five inner folds. CTRP has 744 lines/23 lineages and
PRISM has 386 lines/20 lineages. All outer train/test sets contain every evaluated
lineage. The singleton CTRP Cervix lineage remains excluded. CTRP Ampulla's sole
outer-training identity group is always inner training, never inner validation.

The fixed grids contain 24 SVR, 91 kernel ridge, 102 hierarchical ridge and six
lineage-only candidates. Selection minimizes equal-mean inner-fold RMSE in the
original assay-specific response units. Standardization uses only the current
training subset. There are 160 new outer fits, including 40 lineage-only controls,
and 120 verified saved ridge/Elastic Net/neural-network baselines. A training-mean
control is computed for each split. Kernel ridge includes a linear-only fallback;
hierarchical ridge includes a zero-lineage-offset fallback. Lineage membership is
an input only to the hierarchical and lineage-only arms. No culture covariates,
lineage frequency weights, expanded gene panels or assay pooling are used.

From the project root, use the pinned analysis environment:

```bash
export PYTHONPATH="$PWD/work/bulkformer/packages"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/work/bulkformer/matplotlib_cache"
export PYTHONDONTWRITEBYTECODE=1
python outputs/erastin_kernel_lineage_comparison/validate_preflight.py
python outputs/erastin_kernel_lineage_comparison/run_benchmark.py
python outputs/erastin_kernel_lineage_comparison/review_finalized.py
```

These commands verify and reuse completed model bundles. Do not change the frozen
scripts or grids to overwrite this experiment. Start a new experiment directory
for a revised specification. The runner uses six worker processes; the preflight
and post-fit audits compare custom paths against independent native APIs or
augmented least squares. Source files and every model bundle are checksummed.

The finalized review preserves the frozen original analysis and evaluates the
lineage-only control as exactly constant within each lineage. This removes BLAS
roundoff that otherwise creates spurious within-lineage correlations. It does not
change fits or tuning. The analysis-only correction is recorded in
`analysis_source_manifest.json`.

After inspecting the rendered figure, record the visual inspection in
`visual_qa.json`, then run `review_benchmark.py --seal` to build the final artifact
manifest. The manifest excludes mutable execution logs and Python caches.

The original source experiments and downloaded data must remain available locally;
they are referenced by `source_manifest.json`. Large arrays and fitted binaries
remain excluded from Git under the project policy. These repeatedly used,
overlapping holdouts provide exploratory comparisons; they are not independent
replications or an untouched external validation cohort.
