import json,time,shutil,hashlib
from pathlib import Path
out=Path('outputs/erastin_prism_validation');src=Path('outputs/erastin_elastic_net_comparison')
plan=dict(status='fixed_before_erastin_PRISM_response_matching_or_scoring',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
source='Official Broad DepMap PRISM Repurposing 20Q2 v1, DOI 10.6084/m9.figshare.20564034.v1. Secondary dose-response curves, NOT primary single-dose LFC. This is a pinned archive, not a claim of the latest release.',
compound=dict(name='erastin',broad_id='BRD-A25004090-001-08-4',portal_compound_id='DPC-002562',prior_ctrp_sample='CTRP:63578'),
models=dict(primary='64-gene unweighted ridge, neural network and Elastic Net; all 20 previously fitted split-specific models.',secondary='The corresponding five-gene models; exploratory because prior panel selection used this cohort.',training='None. No PRISM-label training, recalibration, feature selection, hyperparameter tuning, sign flipping, prediction clipping or model selection.'),
features='Exact original 64 genes and order, release-matched Entrez expression columns from DepMap 24Q4 log2(TPM+1); preserve original float32 source precision before predicting. Retain original saved training-only scalers.',
cohort='Require finite reported PRISM auc, explicitly passing STR profiling, an unambiguous ModelID, Cell Line and cancer metadata in the same 24Q4 release, finite expression, and one of the 23 represented training lineages. Cervix remains excluded. Keep an audit for all rows.',
replicates='Use erastin exact batch identity, prefer MTS010 where present as recommended by source, otherwise HTS002; median reported AUC for any repeated passing barcodes within a chosen screen/ModelID. Never choose a response value based on performance. No arbitrary R2 filtering of flat insensitive curves.',
groups='Reconstruct connected components over ALL 24Q4 release models using PatientID, RRID, CCLEName and StrippedCellLineName; verify exact agreement with saved training group IDs.',
primary_test='Novel identity groups: absent from all 744 original training-cohort identity groups; mean predictions of all 20 saved outer fits. These groups were not used in fitting or prior importance-based panel selection.',
secondary_test='Shared identity groups: predictions only from saved outer fits whose 595 training rows contain no member of that group; average available held-out predictions. Exclude groups never held out in the 20 original splits. Also report combined eligible set separately.',
primary_metric='Spearman correlation between predicted sensitivity and negative PRISM AUC. Continuous drug response, so no artificial binary accuracy threshold.',
secondary_metrics='Pearson correlation, per-lineage associations (Spearman only with at least 5 lines), split-specific Spearman/Pearson and diagnostic within-lineage-centered Pearson. Raw negative-AUC RMSE/MAE/R2 are explicitly uncalibrated cross-assay diagnostics, not interchangeable with original CTRP scores.',
scale_limitation='CTRP and PRISM have different assays/dose ranges; original CTRP AUC construction is not fully resolved. Correlations assess transfer of ranking. Raw predicted AUC and observed PRISM AUC need not be calibrated despite nominal AUC units.',
uncertainty=dict(count=5000,seed=2026111101,primary_unit='Whole identity-group pairs bootstrap on fixed ensemble predictions; shared across all six arms within each analysis set.',sensitivity_unit='Whole-lineage bootstrap using same paired predictions.',scope='Conditional on existing models, source data and panel choice. No refitting or multiple-comparison/model-selection adjustment. Overlapping repeats are not independent biological validations.'),
comparisons=[['elastic_net__64','ridge__64'],['elastic_net__64','neural_network__64'],['neural_network__64','ridge__64'],['elastic_net__5','elastic_net__64'],['ridge__5','ridge__64'],['neural_network__5','neural_network__64']],
assay_diagnostic='For exact cell-line overlap, compare measured original CTRP sensitivity with PRISM sensitivity; descriptive cross-assay agreement, not an achievable prediction ceiling.')
pp=out/'protocol.json'
if pp.exists():assert json.load(open(pp))==plan
else:pp.write_text(json.dumps(plan,indent=2)+'\n')
for name in ['weighted_mlp.py','metric_helpers.py']:
    dest=out/name
    if not dest.exists():shutil.copyfile(src/name,dest)
    assert dest.read_bytes()==(src/name).read_bytes()
print('PLAN_FROZEN',plan['created_utc'])
