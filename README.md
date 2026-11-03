# Erastin sensitivity prediction from bulk RNA-seq

Research project predicting cancer-cell-line erastin sensitivity from DepMap
bulk RNA-seq expression, using CTRP and PRISM response data.

The [project presentation](outputs/project_presentation/README.md) introduces
the methods, validation design, model comparisons and biological interpretation.

## Retained models

Decision recorded on **2026-11-02**: retain the **64-gene ferroptosis panel** with
**ridge, Elastic Net and the neural network**, using **unweighted training and
gene expression only**. Culture covariates and the expanded Hallmark EMT/DNA
repair panels are not part of the retained models.

This decision is reaffirmed after the latest neural-network benchmarks:
Elastic Net-selected additions, including stronger regularization, did not
establish lower prediction error than the matched 64-gene control.

The ensemble benchmark likewise supports retaining the individual models:
neither equal averaging nor learned mixing weights established a gain over
the retained neural network.

The nested top-10/top-20 comparison also supports retaining all 64 genes.
The full panel had the lowest mean test RMSE in both CTRP and PRISM over the
20 shared splits. Top 10 performed worse in both assays; top 20 was close in
CTRP, with its conditional interval spanning no difference, and modestly worse
in PRISM. Gene rankings and reduced-panel tuning were confined to nested
training data. The gene-importance rankings describe the retained models and
do not replace the panel with a fixed consensus subset.

The compact mutation/copy-number benchmark also supports keeping the RNA-only
models. Adding copy number reduced CTRP neural-network mean RMSE by 0.32%, but
its paired lineage-bootstrap interval included zero; PRISM showed no clear gain.
No tested DNA addition established lower primary pooled RMSE than its matched
RNA-only control. Some CTRP secondary MAE and equal-lineage metrics favored copy
number, making that model an exploratory candidate for independent validation.
All four arms used the same 20 nested splits and matched cohorts: 739 CTRP rows
and 386 PRISM rows. The CTRP RNA-only controls were refit on that 739-row cohort;
the canonical registry still points to the original retained models.

The existing CTRP and PRISM models are retained separately. These are saved
models from 20 shared, lineage-stratified holdouts for each response cohort;
they are not new models fitted on the entire cohort. Their response endpoints
and performance scales should remain distinct.

The canonical paths, gene order, input/split checksums, selected settings and
checksums for all **120 retained outer models** are recorded in
[the retained-model registry](outputs/retained_64_gene_models/model_registry.json).
The registry points to the existing files; no model binaries are duplicated.

| Response cohort | Existing model directory | Lines | Lineages | Outer train/test |
|---|---|---:|---:|---:|
| CTRP | `outputs/erastin_elastic_net_comparison/folds/` | 744 | 23 | 595 / 149 |
| PRISM | `outputs/erastin_prism_direct_comparison/folds/` | 386 | 20 | 308 / 78 |

Use the `__p64__r01` through `__p64__r20` files for each family. Each `.joblib`
model has companion `.json` fit metadata and `.npz` predictions/tuning arrays.
Keep the panel's recorded gene order and saved training scalers when predicting.

## Project layout

- `outputs/<experiment>/`: analysis code, reports, fixed protocols, definitions,
  validation evidence and locally saved results. The code remains at its
  original paths so existing path assumptions and frozen script checksums work.
- `outputs/retained_64_gene_models/`: the current model-selection decision and
  canonical registry.
- `scripts/register_retained_models.py`: verifies source manifests and every
  retained model/prediction/metadata checksum, then rebuilds the registry.
- `work/`: workflow helpers, downloaded sources, cached packages, checkpoints
  and scratch files. Only the small project-authored Python helpers are tracked.

Earlier comparisons, including culture covariates, alternative models and EMT/
DNA-repair additions, remain as the project's experimental history. Retaining
the 64-gene models does not delete those results or change their original reports.

## Reproduce and inspect

The experiment directories contain their own run/review scripts and protocols.
Read the relevant protocol and source manifest before rebuilding a study: some
experiments reuse earlier inputs, split arrays and fitted models. Do not overwrite
frozen results merely to regenerate a figure or inspect a saved fit.

The latest pinned analysis environment is documented in
`outputs/erastin_emt_ddr_comparison/requirements.txt`; the saved neural-network
adapter requires scikit-learn 1.5.2. To verify/rebuild the retained registry using
Python's standard library:

```bash
python scripts/register_retained_models.py
```

Useful study reports:

- [CTRP gene-only models](outputs/erastin_elastic_net_comparison/REVIEW.md)
- [Direct PRISM training](outputs/erastin_prism_direct_comparison/REVIEW.md)
- [EMT/DNA-repair comparison](outputs/erastin_emt_ddr_comparison/REVIEW.md)
- [PRISM versus CTRP gene importance](outputs/erastin_prism_ctrp_gene_importance/REVIEW.md)
- [Complete 64-gene neural-network importance and gradient rankings](outputs/retained_nn_gene_ranking/REVIEW.md)
- [Signed gene effects versus published ferroptosis mechanisms](outputs/erastin_gene_direction_literature/REVIEW.md)
- [Matched-cohort RNA plus mutation/copy-number benchmark](outputs/erastin_rna_dna_comparison/REVIEW.md)
- [Nested training selection of 10/20 genes versus the full 64-gene neural network](outputs/erastin_nn_topk_nested/REVIEW.md)
- [64-gene kernel and hierarchical lineage benchmark](outputs/erastin_kernel_lineage_comparison/REVIEW.md)
- [Multi-task learning across ferroptosis inducers](outputs/erastin_multitask_comparison/REVIEW.md)
- [Hierarchical lineage-specific gene effects](outputs/erastin_hierarchical_slopes/REVIEW.md)
- [TabPFN regression on the 64-gene panel](outputs/erastin_tabpfn_comparison/REVIEW.md)
- [Genome-wide search for additions to the core64 panel](outputs/erastin_genomewide_search/REVIEW.md)
- [Genome-wide Elastic Net selection beyond the core64 panel](outputs/erastin_genomewide_elastic_net/REVIEW.md)
- [Elastic Net-selected additions with the neural network](outputs/erastin_genomewide_enet_nn/REVIEW.md)
- [Stronger NN regularization for Elastic Net additions](outputs/erastin_enet_nn_stronger_regularization/REVIEW.md)
- [Ensemble of the retained 64-gene models](outputs/erastin_64gene_ensemble/REVIEW.md)

## Version-control scope

Git tracks the analysis code, reports/figures, fixed definitions and protocols,
source/checksum manifests, validation evidence and compact comparison summaries.
Downloaded data, environments, fitted model files, large intermediate arrays,
per-fold diagnostics and ZIP archives remain on disk and are ignored by Git.
The source manifests document the excluded files and their provenance.

The project remains an exploratory cell-line benchmark. Overlapping holdouts are
not independent biological cohorts, and the results do not establish patient or
unseen-lineage performance. The original harmonized CTRP AUC provenance limit
and each experiment's recorded interpretation limits still apply.
