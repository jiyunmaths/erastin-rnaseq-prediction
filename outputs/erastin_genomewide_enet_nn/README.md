# Elastic Net-selected additions with the neural network

Results and interpretation: [REVIEW.md](REVIEW.md).

The original 64 genes remain mandatory. All outside genes selected by each
Elastic Net setting are appended, without a top-K cap. Prediction remains
RNA-only, unweighted and free of culture/lineage model covariates.

The completed genome-wide EN benchmark saved 20 selector paths separately in
each current inner-training fold and outer-training set. Those hash-verified
caches are reused here; no whole-cohort list or ridge-chosen selector enters
NN tuning. Preflight reproduces all 20 selectors in the first/last inner folds
and outer training of repeat 1 for each assay (120 exact gene-set checks).
The unchanged selectors adjust training expression/response for the core64
and training lineage, standardize residuals, and jointly consider all eligible
outside genes. Constants/numerical aliases alone are filtered.

NN validation jointly chooses an EN setting and the original NN parameters:

- CTRP: widths 16/32, L2 penalties 10/100.
- PRISM: widths 16/32, L2 penalties 1/10/100/1000.
- Both: an additional core-only fallback family, tanh activation, L-BFGS,
  max_iter 2000, max_fun 40000, tolerance 0.001, no early stopping, and a mean
  of three networks with the original initialization seed triplet.

All 20 original lineage-stratified, identity-group 80/20 holdouts and five
inner folds are unchanged. Every evaluated lineage is in outer train/test.
The same singleton inner-training handling and Cervix exclusion apply.
Scaling is fit separately on each current NN training subset. Selection uses
mean five-fold RMSE. Core-only grid predictions and selected outer models
are reused exactly from the verified baseline.

There are 84 CTRP and 168 PRISM joint candidates, 25,200 candidate-fold scores,
24,000 new inner ensembles (72,000 networks), and 40 outer ensemble bundles.
Completed inner folds are checkpointed, so resuming preserves finished work.
New fits abort on a ConvergenceWarning; SciPy solver-option deprecation
warnings are recorded separately.

Use the original pinned analysis environment from the project root:

```bash
export PYTHONPATH="$PWD/work/bulkformer/packages"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/work/bulkformer/matplotlib_cache"
export PYTHONDONTWRITEBYTECODE=1
PYTHON=/home/sji/miniforge3/envs/deeplearning/bin/python
$PYTHON outputs/erastin_genomewide_enet_nn/validate_preflight.py
$PYTHON outputs/erastin_genomewide_enet_nn/run_comparison.py
$PYTHON outputs/erastin_genomewide_enet_nn/review_comparison.py
```

The review supports `--wait` during fitting. After inspecting both scientific
figures, save `visual_qa.json` and run the review with `--seal` to finalize
the artifact manifest. Do not alter frozen fitting code or source hashes
when resuming.

Audits verify ensemble means, every inner RMSE/selected setting, source gene
masks, feature counts, scaling, outer member predictions and training
objectives, seed triplets, unit weights, identity exclusion and lineage
coverage. Paired whole-lineage bootstrap preserves models and overlapping
repeats together. Existing core64 NN/ridge/EN/TabPFN, EN-selected ridge and
whole-genome ridge predictions are reused as references. The scoring
procedure is copied unchanged from the prior EN benchmark, with its source
recorded in `helper_sources.json`.

Selection frequencies in `gene_selection_stability.tsv` are training-only
EN associations under the NN-selected settings. They are not NN feature
importance, causal effects or independent replications. A final consensus
panel assembled after this search needs separate validation. Previously
inspected cohorts, finite NN/EN grids, protein-coding RNA scope and original
endpoint provenance limitations keep the results exploratory.

The starting project snapshot was committed and pushed as
`7aa826d7ace8702d777fafc8240f5b34dd4744b2` before this experiment began.
The retained core64 registry is unchanged.

A separately registered best-nonempty diagnostic is summarized in
[forced_additions/REVIEW.md](forced_additions/REVIEW.md). Its initial output-path
bookkeeping correction was archived before any diagnostic fits; primary fitting
and selection were unchanged. Both registrations record their timing/scope.
