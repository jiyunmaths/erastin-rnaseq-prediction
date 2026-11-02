# Reproduce the ferroptosis multi-task benchmark

Results and interpretation are in [REVIEW.md](REVIEW.md). This experiment keeps the
retained 64-gene model registry unchanged.

The study uses the same 744 CTRP cell lines, 733 identity groups, 23 lineages and
20 lineage-stratified 80/20 splits as the retained benchmark. All held-out identity
groups are excluded from **all four tasks**, in outer evaluation and inner tuning.
There are no additional auxiliary-only cell lines and no cross-assay pooling.
Inference requires only the 64 RNA-expression values.

The four separate targets are negative CTRP AUC for erastin, 1S,3R-RSL-3, ML162 and
ML210, with 744/721/721/690 observed responses. Missing labels are masked. Training
task means and scales use observed responses in the current training subset only.
Cell contributions are unweighted within each task; per-task loss normalization
and auxiliary-task weights are explicit in the frozen protocol. All tuning uses
erastin validation RMSE, never auxiliary validation outcomes.

- Coupled linear regression has 68 candidates: 17 ridge penalties and four
  coupling strengths, including zero. Coefficients shrink toward a shared mean;
  zero coupling reproduces the original single-task ridge objective.
- The shared neural network has 12 candidates: hidden width 16/32, penalty 10/100
  and total auxiliary loss weight 0/0.25/1. Each fit averages the same three saved
  initialization seeds. Hidden layers and the erastin head start identically
  across task-weight choices. The matched single-task arm uses weight zero;
  a separate positive-weight arm tests forced sharing.
- The neural objective and optimizer are explicit float64 implementations with
  independent autograd and finite-difference checks. The matched neural control
  is necessary because optimizer settings differ from the retained sklearn NN.
- Eighty outer arm/repeat bundles are saved, with identical selected neural fits
  reused when possible. One hundred saved baseline models (ridge, Elastic Net,
  retained NN, SVR and kernel ridge) are verified and reused.

From the project root, with the pinned environment:

```bash
export PYTHONPATH="$PWD/work/bulkformer/packages"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/work/bulkformer/matplotlib_cache"
export PYTHONDONTWRITEBYTECODE=1
python outputs/erastin_multitask_comparison/validate_preflight.py
python outputs/erastin_multitask_comparison/run_benchmark.py
python outputs/erastin_multitask_comparison/review_benchmark.py
```

The runner uses six workers and resumes completed, checksummed repeats. Analysis
code hashes are saved separately from the frozen fitting specification. Inspect
the rendered figure, record `visual_qa.json`, then run `review_benchmark.py --seal`
to create the artifact manifest. Do not overwrite frozen fits after changing a
grid or model definition; use a new experiment directory.

The original CTRP and kernel benchmark files must remain available locally.
Downloaded data, arrays, fitted binaries and per-fold diagnostics remain outside
Git. These repeatedly inspected, overlapping splits provide exploratory evidence,
not independent replication or unseen-lineage/patient validation.
