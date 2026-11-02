Erastin sensitivity: matched 40-gene / BulkFormer comparison inputs
================================================================

Prepared for the approved 40-gene ferroptosis panel from Dixon and Olzmann,
The cell biology of ferroptosis (2024). No panel genes were selected or removed
using drug-response associations.

Result
------
745 cancer cell lines, 24 lineages, 734 identity groups.
All 40 genes have unique, concordant HGNC, Entrez and Ensembl identifiers in
the DepMap 24Q4 gene metadata, the downloaded current HGNC snapshot, and the
original BulkFormer-147M vocabulary. All 40 have finite expression values
and variation in the matched cohort. FSP1 is represented by AIFM2, SCD1 by
SCD, NRF2 by NFE2L2, and MESH1 by HDDC3.

The full BulkFormer input has 20,010 ordered gene positions. 19,085 (95.38%)
map uniquely to the expression matrix. The other 925 positions are -10:
605 lack a release Ensembl mapping, 300 lack expression, 17 have ambiguous
expression Entrez mappings, and 3 have conflicting approved gene symbols.
These are mapping/coverage categories, not evidence of biological absence.
Measured zero expression remains zero. Every mapping decision is retained
in gene_mapping.tsv.gz. No ambiguous expression columns were averaged or
duplicated across model tokens.

Sources and target
------------------
Expression and metadata: official DepMap 24Q4 Public v1, DOI
https://doi.org/10.25452/figshare.plus.27993248.v1
The 26Q3 portal/API served a human-verification page, so this preparation
deliberately pins the accessible 24Q4 archive. It does not claim 26Q3 coverage.
Release README: RSEM unstranded protein-coding expression, log2(TPM+1).
Model.csv, Gene.csv, RNA default profiles and profile metadata are from the
same release; each included RNA profile was checked against its ModelID.

Response: CTRPAUCMatrix.csv and CTRPResponseCurves.csv archived at
https://doi.org/10.5281/zenodo.22125121
These files are attributed to DepMap Harmonized CTD2 25Q2. The download
checksums match the Zenodo manifest. A freshly retrieved upstream DepMap
manifest was unavailable, so upstream byte identity is not independently
verified. The archive is identified explicitly rather than described as
a direct current DepMap download.

Compound identity is checked against official DepMap PortalCompounds.csv:
ERASTIN = DPC-002562 = CTRP:63578. Related compounds are not pooled.
PortalCompounds target/mechanism annotations are retained as historical
source metadata; the attached ferroptosis review defines biological scope.

Primary target: y_sensitivity = -erastin_auc_reported.
Higher y_sensitivity means greater sensitivity; lower original AUC means
greater sensitivity. Original AUC values are also retained unchanged.
The archived AUC values range from 0.3021 to 1.0 in the matched cohort.
The upstream integration interval, normalization and clipping recipe was
not independently established. These values must not be conflated with
older, unnormalized CTRP AUC releases. We do not divide AUC by maximum dose,
recalculate it from EC50, infer binary sensitive/resistant labels, or impute
missing response values.

There were 795 erastin-labelled lines. Fifty were excluded: 27 lacked both
24Q4 metadata and expression, 15 lacked expression, and 8 were explicitly
classified Non-Cancerous by DepMap. cohort_audit.tsv lists every decision.
All retained entries are Cell Line models with a cancer lineage. There is
one harmonized erastin curve per retained ModelID; no replicate averaging
was introduced here. Raw well-level QC, replicate uncertainty and exposure
duration are not present in these archived curve tables. This preparation
checks structural validity, not raw-screen biological QC.

Tested dose ranges vary. cohort.tsv retains MinimumDose, MaximumDose and
DoseUnit. The modal interval is 0.002-66 uM, with 655 matched lines. The
modal_dose_range_subset flag provides a predefined robustness analysis.
Do not present all 745 lines as measured over an identical dose interval.

Files and loading
-----------------
comparison_inputs.npz: load with numpy.load(path, allow_pickle=False).
  sample_ids                   shared row order; 745 ModelIDs
  X_panel_log2tpm1              float32, 745 x 40, unscaled
  panel_symbols                approved panel order
  X_bulkformer_lntpm1           float32, 745 x 20010, unscaled
  bulkformer_ensembl_ids        exact vocabulary order
  bulkformer_observed_gene_mask true for uniquely mapped genes
  panel_indices_0based          panel positions in BulkFormer vocabulary
  y_auc                        reported AUC, float64
  y_sensitivity                negative reported AUC, float64
  group_ids, lineage           validation grouping and tissue annotation
  outer_fold                   fixed 5-fold assignment
  modal_dose_range_subset       same-dose-range robustness subset

matched_depmap_expression.npz: all 19,193 original expression columns for
the same 745 lines, log2(TPM+1), with original column names and sample IDs.
This preserves source genes outside the BulkFormer mapping as well.

cohort.tsv: response, curve parameters, dose interval, RNA profile IDs,
cell-line metadata, patient IDs, groups, and outer/inner fold assignments.
panel_mapping.tsv / .json: inspectable 40-gene mapping and token indices.
gene_mapping.tsv.gz: complete 20,010-gene mapping, including exclusions.
erastin_all_795_labels.tsv: all available erastin labels, including excluded
models, to make cohort selection reviewable.
splits.json: explicit train/test IDs for outer folds, nested inner folds,
and leave-lineage-out checks.
preparation_report.json / validation_results.json: measured preparation
counts and results of independent artifact/source consistency checks.
source_manifest.json: URLs, hashes, release IDs and source limitations.
artifact_manifest.json: hashes of the final deliverables.

Three-model protocol
--------------------
1. Panel expression: the 40 raw-expression features -> standardized ridge.
2. Panel contextual embedding: full BulkFormer input -> frozen encoder ->
   mean output at the 40 panel indices -> standardized ridge.
3. All-gene contextual embedding: the same full input and frozen encoder ->
   mean output across all 20,010 positions -> standardized ridge.

BulkFormer is pinned to KangBoming/BulkFormer commit
6d7f21b513299c7803887a7954a11d2c95039f47, original 147M configuration.
It is not the separate bulkformer-latest model with a different vocabulary.
Source: https://github.com/KangBoming/BulkFormer/tree/6d7f21b513299c7803887a7954a11d2c95039f47

Input conversion is ln(TPM+1) = log2(TPM+1) * ln(2). Do not apply another
log transform, renormalize the protein-coding subset to sum to one million,
or standardize expression before feeding it into the frozen encoder.

The pinned model returns 640 contextual channels plus 3 global channels
(mask fraction, observed-expression mean and its implemented nonzero ratio),
so both embedding arms should produce 643 features. The panel embedding
still depends on the entire transcriptome and those global summaries.
It must not be described as using only 40 genes' information.

In this version, mask_prob is an auxiliary scalar feature, not a random
masking operation. comparison_config.json sets it to 925/20010, following
the notebook missing-gene-fraction convention. No extra inference masking
is applied. Use eval mode and frozen weights. All-gene primary pooling
includes missing-gene token outputs, matching the pinned notebook protocol.
Observed-gene-only pooling and 640-channel-only pooling are secondary
robustness checks, not silent changes to the primary comparison.

The encoder checkpoint, graph assets and pretrained embedding assets have
not been downloaded/loaded; their hashes therefore remain unverified.
BulkFormer embeddings have NOT been computed, and no models were trained.
This bundle completes the matched input-data preparation stage. Freeze and
record the actual checkpoint/resource hashes when feature extraction starts.
Foundation-model pretraining overlap with these cell lines remains unknown.

Validation design
-----------------
Seed 20261031. Five outer folds of sizes 147, 149, 151, 147 and 151, each
with four inner folds for tuning. Fold assignment uses lineage and identity
only, never response values or expression. Related identities are connected
using PatientID, RRID, CCLEName and StrippedCellLineName across the complete
Model table. Eleven retained groups contain two lines each; they stay intact.
This groups documented related lines, not every possible unannotated relation.

Every model uses exactly the same folds and targets. Fit standardization and
ridge hyperparameters only on the training part of each fold. The ridge
alpha grid and tuning objective are specified in comparison_config.json.
Do not choose genes, pooling protocols or ridge settings using outer-test
performance. Pair comparisons and uncertainty estimates by group_id.

Leave-lineage-out splits remove all training lines related to the held-out
lineage. Lineages with fewer than 10 test lines are flagged descriptive-only.
Some lineages are too small to occur in every random fold; inspect
fold_lineage_counts.tsv. A lineage-only baseline should accompany the three
models to quantify tissue effects. Baseline RNA and drug response may come
from different passages, assays and media. RNA prediction is not proof that
the fitted gene effects causally control ferroptosis.

Reproduction
------------
The build used Python 3.12.14, numpy 2.3.5 and pandas 2.2.3. In a Python
environment with the requirements installed, run from this bundle directory:

  python fetch_sources.py --raw-dir /path/to/raw_inputs
  python prepare_data.py --raw-dir /path/to/raw_inputs
  python verify_data.py --raw-dir /path/to/raw_inputs

fetch_sources.py verifies each input SHA256 and refuses changed sources.
Mutable HGNC and archive API metadata are preserved in source_snapshots/.
Source data already downloaded in this workspace are at work/erastin/raw.
That raw folder is not needed to load the prepared NPZ/TSV files.

The original paper panel curation is retained in panel_definition.json.
Review its mechanism-specific notes before interpreting any fitted weights.
