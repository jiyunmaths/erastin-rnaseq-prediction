# Ensemble of retained core64 ridge, Elastic Net and neural network

Results and interpretation: [REVIEW.md](REVIEW.md).

Exactly the 120 retained outer models are reused: three families, 20 repeats,
and separate CTRP/PRISM cohorts. No base model is refitted or retuned. The
original 64 input genes/order, unweighted RNA-only training, group exclusions,
lineage-stratified holdouts and five inner folds are unchanged. Culture and
lineage prediction covariates are excluded.

Three prediction rules are evaluated:

- Equal blend: one-third of each family prediction. The NN's original
  three-network mean counts as one family, so this is a three-family average.
- Learned blend: nonnegative weights summing to one, fitted by mean five-fold
  inner RMSE using the saved inner predictions at the retained base settings.
  No intercept, calibration, clipping or outer-response-based tuning is used.
  Zero weights are allowed; the optimization includes all vertices/edges.
- Inner-selected single family: lowest mean inner RMSE, as a model-selection
  control for whether blending helps beyond choosing a family.

Weight optimization receives only rows with saved inner assignment >=0. The
CTRP protected singleton training row has no inner-validation prediction and
is omitted from weight fitting; the retained base models still use all their
original training rows. Each ensemble predicts the same original outer test.

The original inner labels select base hyperparameters and now also train the
weights. These inner losses are optimistic training diagnostics. They are not
independently cross-validated ensemble accuracy, and no extra cross-fitting
of base hyperparameter selection is added. Only outer-test scores evaluate
the full procedure. The cohorts have informed earlier experiments, so results
remain exploratory rather than untouched confirmation.

Use the pinned original environment from the repository root:

```bash
export PYTHONPATH="$PWD/work/bulkformer/packages"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/work/bulkformer/matplotlib_cache"
export PYTHONDONTWRITEBYTECODE=1
PYTHON=/home/sji/miniforge3/envs/deeplearning/bin/python
$PYTHON outputs/erastin_64gene_ensemble/validate_preflight.py
$PYTHON outputs/erastin_64gene_ensemble/run_comparison.py
$PYTHON outputs/erastin_64gene_ensemble/review_comparison.py
```

Inspect both figures, record `visual_qa.json`, then run the review with `--seal`
to finalize the artifact manifest. Frozen script/source hashes protect resuming.

The preflight checks a dense simplex grid, analytic gradients, convex KKT
conditions, unchanged gene order/registry hashes, all saved split generators,
and a held-out-response decoy. Audits independently reproduce all retained
outer predictions, every base inner RMSE/selection, ensemble arithmetic,
weight feasibility, convex-hull bounds, group exclusions and bootstrap
arithmetic. Model binaries are referenced by their existing registry paths.

`ensemble_weights.tsv` and `weight_summary.tsv` describe learned family weights;
`base_error_correlation.tsv` describes per-repeat prediction/error correlation.
Weights are not gene importance or causal effects. Highly correlated models
may have unstable or nearly interchangeable weights. Splits overlap, so weight
SDs and win counts do not represent independent replication.

The paired whole-lineage bootstrap keeps all models/overlapping repeats
together. Its intervals are conditional on the fitted models, omit retraining
uncertainty and are not multiplicity-adjusted. Original endpoint provenance,
panel-selection history and within-represented-lineage limitations remain.
The retained core64 registry is unchanged; no patient or unseen-lineage
performance is established.
