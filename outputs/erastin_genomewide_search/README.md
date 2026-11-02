# Genome-wide search beyond the retained 64-gene panel

Results, selected candidates and interpretation are in [REVIEW.md](REVIEW.md).
The user's reference to a 60-gene panel is interpreted as the existing 64-gene
panel. Every augmentation retains those genes in their original order.

The search includes all 19,176 unique measured protein-coding expression features
in the downloaded DepMap file: 19,112 candidates outside the core. Seventeen
duplicated Entrez identifiers are resolved by approved symbols in the frozen
HGNC release file. No pathway whitelist is applied. This file does not contain
all noncoding transcripts or DNA sequence/variant information.

The unchanged CTRP/PRISM cohorts have 744/386 lines and 23/20 lineages. All
evaluated lineages occur in both sides of each original 20 outer holdouts;
identity groups remain disjoint. The five saved inner folds are reused. Core
RNA values are byte-for-value identical; more than 14 million retained CTRP
expression entries were independently checked against the original matched data.

Two screens use only current-training observations:

- Absolute marginal Pearson correlation with erastin sensitivity.
- Absolute partial Pearson correlation after adjusting both gene and response
  for the core64 and training-lineage indicators, using an exact SVD projection.

Training-constant genes are excluded. The conditional screen also excludes
numerical aliases of the controls. No all-cohort variance/response filtering or
test-response ranking is used. Models use RNA only: lineage is a screening
control, not an additional model predictor. Culture covariates remain excluded.

Ridge's selector, 0/8/16/32/64/128 additions and penalty are jointly tuned inside
each outer-training set. Every inner-training fold recomputes its ranking before
validation. Rankings are refitted on outer training at the selected final fit.
A separate ridge model uses all genes with nested penalty tuning.

TabPFN uses the same fixed 3.5-Fast checkpoint and eight-pass preset as its
core64 baseline. It tests two training-defined panels: the ridge-selected screen
with additions capped at 32, and the prespecified conditional top32. TabPFN does
not tune selectors/counts using test outcomes. Exact core64 predictions are
reused when zero additions are chosen. Identical selected panels share a native
context. This is a CPU-scoped first search, not exhaustive interaction/feature
subset optimization or a test of all possible TabPFN gene counts.

From the project root, use the original analysis environment:

```bash
export PYTHONPATH="$PWD/work/bulkformer/packages"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLCONFIGDIR="$PWD/work/bulkformer/matplotlib_cache"
export PYTHONDONTWRITEBYTECODE=1
python outputs/erastin_genomewide_search/prepare_data.py
python outputs/erastin_genomewide_search/validate_preflight.py
python outputs/erastin_genomewide_search/run_ridge.py
```

Run TabPFN in its existing isolated environment:

```bash
env -u PYTHONPATH OMP_NUM_THREADS=3 MKL_NUM_THREADS=3 OPENBLAS_NUM_THREADS=1 \
  work/tabpfn/venv/bin/python outputs/erastin_genomewide_search/run_tabpfn.py
```

Then evaluate in the original environment:

```bash
PYTHONPATH="$PWD/work/bulkformer/packages" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  /home/sji/miniforge3/envs/deeplearning/bin/python \
  outputs/erastin_genomewide_search/review_search.py
```

The review can use `--wait` while the TabPFN stage is running. It audits an assay
once all 20 repeats of that assay finish. Inspect both figures, record
`visual_qa.json`, then run the review with `--seal` to create the artifact
manifest. Preserve frozen fitting code and source hashes when resuming.

`gene_selection_stability.tsv` reports training-only selection frequencies and
ridge coefficient stability. `cross_assay_candidate_genes.tsv` summarizes the
conditional top32 overlap. These overlapping folds and shared assay cell lines
are not independent confirmations. TabPFN selection frequencies are not feature
importance. A final consensus panel assembled after this search requires new
validation; reported accuracy belongs to the fold-specific selection procedure.

Nesting controls direct feature-selection leakage but does not undo prior
inspection of these cohorts. Results are exploratory. Candidate associations do
not establish ferroptosis mechanisms. The retained panel and model registry stay
unchanged. No patient or unseen-lineage prediction is validated here.
