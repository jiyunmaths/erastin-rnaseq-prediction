# Reproduce the TabPFN regression benchmark

Results and interpretation are in [REVIEW.md](REVIEW.md).

This uses **TabPFN-3.5-Fast**, package **9.0.0**, the official
`tabpfn-v3.5-fast-20260909.safetensors` checkpoint, eight ensemble passes,
float32 CPU inference and the unchanged 64-gene RNA panel. The checkpoint's
repository revision and SHA256 are pinned in `protocol.json`.

CTRP and PRISM keep their original response values, identity groups and 20 saved
lineage-stratified outer holdouts. Every evaluated lineage appears in both outer
sets. No culture/lineage predictors, lineage weights, new genes or assay pooling
are used. Native TabPFN preprocessing fits on each training set only.

The checkpoint was chosen using CPU timing alone. No pilot accuracy was computed.
The fixed eight-pass model has no study-specific hyperparameter search or
temperature calibration. Original inner folds are audited but are not used for
selection of this fixed model; the saved baselines retain their previous nested
tuning. These are exploratory comparisons on repeatedly inspected data.

There are two Python environments:

- `work/tabpfn/venv/`: isolated TabPFN dependencies, reusing the existing PyTorch
  installation. Exact installed versions and wheel hashes are documented by the
  local install report referenced in `source_manifest.json`.
- The original pinned analysis environment, including sklearn 1.5.2 via
  `work/bulkformer/packages`, evaluates saved numeric predictions and baselines.

From the project root, run TabPFN without the old dependency overlay:

```bash
env -u PYTHONPATH OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 \
  work/tabpfn/venv/bin/python outputs/erastin_tabpfn_comparison/validate_preflight.py
env -u PYTHONPATH OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 \
  work/tabpfn/venv/bin/python outputs/erastin_tabpfn_comparison/run_tabpfn.py
```

Use the original analysis environment for the review:

```bash
PYTHONPATH="$PWD/work/bulkformer/packages" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/sji/miniforge3/envs/deeplearning/bin/python \
  outputs/erastin_tabpfn_comparison/review_benchmark.py
```

Inference uses the local checkpoint with outbound connections blocked. The runner
uses two workers with three Torch threads each and resumes completed contexts
after checksum verification. Native `.tabpfn_fit` files save preprocessing and
training context without copying the large pretrained weights into every fold.
The checkpoint must remain at its recorded path when loading those contexts.

All archives receive CRC/checksum, row, seed, feature-coverage and native training
target-statistic checks. First and last contexts in each assay receive native
save/load prediction checks. Preflight also checks query-subset invariance and
invariance to changing other query rows. Other contexts are archive-checked, not
separately re-predicted after loading.

After inspecting the figure, record `visual_qa.json`, then run the review with
`--seal` to build the artifact manifest. Changes to checkpoint or preset require
a new experiment directory. Downloaded weights, environments, native contexts and
data remain outside Git. The retained model registry remains unchanged.

The checkpoint license is saved in `TABPFN_LICENSE.md`; this is a research/internal
benchmark. The original 2025 TabPFN paper concerns an earlier model generation.
Performance here does not establish results for other checkpoints, unseen cancer
lineages or patients. Pretraining-corpus overlap with these pharmacogenomic data
has not been independently audited.
