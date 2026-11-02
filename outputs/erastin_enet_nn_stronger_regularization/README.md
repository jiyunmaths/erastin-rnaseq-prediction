# Stronger NN regularization with Elastic Net-selected additions

Results and interpretation: [REVIEW.md](REVIEW.md).

The retained core64 genes, all 20 Elastic Net settings, fold-local gene masks,
20 lineage-stratified identity-group holdouts, five inner folds and three-network
initialization seed triplets are unchanged. No top-K truncation is applied.
Prediction remains RNA-only, unweighted and free of culture/lineage covariates.

New L2 penalties are **300, 1,000, 3,000, 10,000 and 30,000**. Values already
present are reused, alongside every previously tested setting. Hidden widths
stay at 16/32, with the original tanh/L-BFGS solver, tolerances and scaling.
The pinned objective is `0.5*MSE + 0.5*alpha*sum(weight matrices squared)/n`;
biases are unpenalized and X/y scaling uses current training only.

Three arms use the same extended NN grid:

- `core_extended`: original core64 only, with the penalty/width retuned.
- `optional_extended`: jointly tune zero additions or 20 EN settings, penalty
  and width by mean five-fold inner RMSE.
- `forced_extended`: tune the best nonempty EN setting on the same grid.

The matched extended-grid core-only control separates the value of extra genes
from a general benefit of stronger regularization. Existing core64, prior
optional/forced expansions and TabPFN predictions remain saved references.
Outer NN fits are reused when an unchanged old setting wins; identical choices
among new arms share one fit before separate output bundles are saved.

There are 63,000 candidate-fold scores: 25,200 reused scores and 37,800 new
three-member inner ensembles (113,400 networks). There are 120 logical outer
ensemble bundles across the three arms and both assays. Original fold-local
selection caches and all old predictions are source-hash verified. Completed
inner folds are checkpointed for safe resumption.

From the project root, use the original pinned analysis environment:

```bash
export PYTHONPATH="$PWD/work/bulkformer/packages"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/work/bulkformer/matplotlib_cache"
export PYTHONDONTWRITEBYTECODE=1
PYTHON=/home/sji/miniforge3/envs/deeplearning/bin/python
$PYTHON outputs/erastin_enet_nn_stronger_regularization/validate_preflight.py
$PYTHON outputs/erastin_enet_nn_stronger_regularization/run_comparison.py
$PYTHON outputs/erastin_enet_nn_stronger_regularization/review_comparison.py
```

The review supports `--wait`. Inspect both generated figures, save
`visual_qa.json`, then use `--seal` to finalize the artifact manifest.
Preserve frozen fitting code/source hashes when resuming. New fits abort on
ConvergenceWarning; SciPy solver-option deprecations are counted separately.

Audit outputs include every inner RMSE/choice, reused candidate/member
predictions, all outer scalers, seed triplets, unit weights, member predictions,
training objectives, selected gene masks and split checks. Selected penalties,
added-gene counts, prediction standard deviations and weight norms are saved in
`selected_parameters.tsv`. Grid limits are reported without post-test expansion.

The expanded grid was motivated by earlier results from these same cohorts.
Results remain exploratory internal validation. Paired whole-lineage bootstrap
keeps all overlapping repeats/models together; intervals omit retraining
uncertainty and are not multiplicity-adjusted. Gene-selection frequencies are
training associations, not neural-network importance, individual incremental
benefits, causal effects or independent replications. A consensus panel needs
separate validation. Original endpoint, protein-coding RNA and finite-grid
limitations remain. The retained panel/model registry stays unchanged.
