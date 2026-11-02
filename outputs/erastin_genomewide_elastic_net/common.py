"""Frozen sources and shared nested splits for core-preserving EN discovery."""
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
import joblib
import models

OUT = Path(__file__).resolve().parent
WS = OUT.parent.parent
SOURCE = WS/'outputs/erastin_genomewide_search'
BASE = {'CTRP':WS/'outputs/erastin_elastic_net_comparison', 'PRISM':WS/'outputs/erastin_prism_direct_comparison'}
ASSAYS = list(BASE)
STATE = {}
EXPRESSION = INDEX = None
WORKERS = 4


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def clean(v):
    if isinstance(v,dict): return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)): return [clean(x) for x in v]
    if isinstance(v,np.generic): return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v): return None
    return v


def dump(p,v):
    temp=p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(clean(v),indent=2,allow_nan=False)+'\n')
    temp.replace(p)


def initialize():
    global EXPRESSION,INDEX
    EXPRESSION=np.load(SOURCE/'expression.npy',mmap_mode='r')
    a=np.load(SOURCE/'expression_index.npz'); INDEX={k:a[k] for k in a.files}
    for assay in ASSAYS:
        a=np.load(SOURCE/assay/'inputs.npz');s=np.load(SOURCE/assay/'shared_splits.npz')
        STATE[assay]=dict(rows=a['expression_rows'],xcore=a['x_64'],y=a['y'],ids=a['sample_ids'],groups=a['group_ids'],lineage=a['lineage'],test=s['outer_test'],inner=s['inner_fold'],seeds=s['outer_split_seeds'],inner_seeds=s['inner_split_seeds'])


def features(assay,rows,genes=None):
    ix=STATE[assay]['rows'][rows]
    return np.asarray(EXPRESSION[ix] if genes is None else EXPRESSION[ix[:,None],genes],dtype=float)


def prepare():
    OUT.mkdir(exist_ok=True)
    assert json.load(open(SOURCE/'validation.json'))['passed']
    previous={r['path']:r['sha256'] for r in json.load(open(SOURCE/'artifact_manifest.json'))['files']}
    names=['expression.npy','expression_index.npz','gene_mapping.tsv','protocol.json','validation.json','source_manifest.json','preparation_validation.json']
    names += [a+'/'+n for a in ASSAYS for n in ['inputs.npz','shared_splits.npz','predictions.npz']]
    names += [f'{a}/folds/screening__r{r:02d}.npz' for a in ASSAYS for r in range(1,21)]
    records=[]
    for n in names:
        assert sha(SOURCE/n)==previous[n]
        records.append(dict(path=str((SOURCE/n).relative_to(WS)),sha256=previous[n]))
    for assay in ASSAYS:
        (OUT/assay/'folds').mkdir(parents=True,exist_ok=True)
        known={r['path']:r['sha256'] for r in json.load(open(BASE[assay]/'artifact_manifest.json'))['files']}
        for n in ['predictions.npz','summary.tsv','protocol.json']:
            assert sha(BASE[assay]/n)==known[n]
            records.append(dict(path=str((BASE[assay]/n).relative_to(WS)),sha256=known[n]))
    reg=WS/'outputs/retained_64_gene_models/model_registry.json'
    records.append(dict(path=str(reg.relative_to(WS)),sha256=sha(reg)))
    p=dict(status='frozen_before_nested_fits',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        retained_core=64,universe=19176,candidates_outside_core=19112,assays=['CTRP','PRISM'],
        split_design='Unchanged 20 lineage-stratified identity-group 80/20 outer holdouts and all five saved inner folds. Every evaluated lineage in both outer sets; singleton inner-training lineage handling unchanged.',
        selection='All eligible outside genes jointly enter conditional Elastic Net. Regress each candidate and y on standardized core64 plus training-lineage indicators and intercept, using training-only SVD projection. Standardize residual predictors/target using current-training values only. Exclude only constants and numerical aliases; no marginal-response prefilter.',
        selector_l1_ratios=models.L1_RATIOS,alpha_fractions=models.FRACTIONS,
        alpha_definition='alpha=fraction*max(abs(Z.T@residual_standardized_y))/n/l1_ratio, computed anew on current training only.',
        objective='||target-Z beta||^2/(2n) + alpha*l1_ratio*||beta||_1 + alpha*(1-l1_ratio)*||beta||^2/2',
        nonzero_threshold=models.NONZERO,constant_filter='raw centered SS>1e-12 and residual SS>max(1e-12,raw SS*1e-8)',
        solver=dict(tol=models.TOL,max_iter=models.MAX_ITER,selection='cyclic',precompute='Working-set Gram only at <=2500 active columns; never allocate full-genome Gram.',kkt_limit=models.KKT_LIMIT,
            working_set='Warm-start restricted ElasticNet; expand by globally largest inactive KKT violations. Every gene checked after every solve; stop only when complete genome-wide KKT residual <= declared limit. No permanent feature exclusion.',
            refinement='On restricted convergence warning or restricted KKT excess, refine at tol=1e-10,max_iter=50000; abort if still invalid. Global dual gap also checked.'),
        ridge_alphas=models.ALPHAS,grid=models.GRID,
        tuning='Independently select genes inside every inner-training fold. Jointly tune Elastic Net mixing/alpha fraction and downstream ridge penalty using mean five-fold RMSE in original response units. First grid index breaks ties. Zero-addition ridge candidates come first. Refit selector on outer training only after hyperparameter choice.',
        retention='All original 64 genes are mandatory downstream ridge inputs, regardless of their selector associations. Selected outside genes are appended; no fixed top-K cap. Model predicts from RNA only; lineage is a selector control, with no culture predictors or sample weighting.',
        primary_comparisons=[['enet_selected_ridge','ridge64'],['enet_selected_ridge','enet64']],
        secondary_comparisons=[['enet_selected_ridge','nn64'],['enet_selected_ridge','tab64'],['enet_selected_ridge','ridge_genome'],['enet_selected_ridge','ridge_screened']],
        uncertainty=dict(count=5000,seed=2026112301,unit='Paired whole-lineage bootstrap retaining all overlapping repeats and all models together; conditional on fits, not retraining.'),
        scope='Elastic Net is the feature selector; downstream ridge isolates added-gene value against retained ridge. This is not a directly fitted genome-wide Elastic Net predictor or a TabPFN/NN refit.',
        limitations=['Previously inspected cohorts; exploratory discovery estimates despite nested selection controls.','Fold-specific selection procedure is evaluated. A consensus list assembled across folds is not a separately validated fixed panel.','Measured protein-coding RNA only; noncoding transcripts/DNA variants outside scope.','Associations and overlapping-fold selection frequencies do not establish causality, independent replication or each gene incremental value.','Original harmonized CTRP endpoint provenance limitations remain.'],
        references=['https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.enet_path.html','https://pmc.ncbi.nlm.nih.gov/articles/PMC124442/'],
        scripts={n:sha(OUT/n) for n in ['common.py','models.py','ridge_helpers.py','make_splits.py','run_comparison.py','validate_preflight.py']},
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__))
    if (OUT/'protocol.json').exists():
        old=json.load(open(OUT/'protocol.json'));p['created_utc']=old['created_utc'];assert old==clean(p)
    else: dump(OUT/'protocol.json',p)
    manifest=dict(files=records)
    if (OUT/'source_manifest.json').exists(): assert json.load(open(OUT/'source_manifest.json'))==manifest
    else: dump(OUT/'source_manifest.json',manifest)
    return sha(OUT/'protocol.json')


def verify():
    p=json.load(open(OUT/'protocol.json'))
    for n,h in p['scripts'].items(): assert sha(OUT/n)==h,n
    for r in json.load(open(OUT/'source_manifest.json'))['files']: assert sha(WS/r['path'])==r['sha256'],r['path']
