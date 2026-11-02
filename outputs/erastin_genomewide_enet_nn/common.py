"""Frozen data, fold-local EN caches and original NN settings."""
import hashlib
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn

OUT=Path(__file__).resolve().parent
WS=OUT.parent.parent
SOURCE=WS/'outputs/erastin_genomewide_search'
ENET=WS/'outputs/erastin_genomewide_elastic_net'
BASE={'CTRP':WS/'outputs/erastin_elastic_net_comparison','PRISM':WS/'outputs/erastin_prism_direct_comparison'}
ASSAYS=list(BASE)
STATE={}
EXPRESSION=INDEX=None
WORKERS=8


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v


def dump(p,v):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(clean(v),indent=2,allow_nan=False)+'\n');temp.replace(p)


def initialize():
    global EXPRESSION,INDEX
    EXPRESSION=np.load(SOURCE/'expression.npy',mmap_mode='r')
    a=np.load(SOURCE/'expression_index.npz');INDEX={k:a[k] for k in a.files}
    for assay in ASSAYS:
        a=np.load(SOURCE/assay/'inputs.npz');s=np.load(BASE[assay]/'shared_splits.npz');p=json.load(open(BASE[assay]/'protocol.json'))
        check=np.load(SOURCE/assay/'shared_splits.npz')
        for k in ['outer_test','inner_fold','outer_split_seeds','inner_split_seeds']:np.testing.assert_array_equal(s[k],check[k])
        for k in ['sample_ids','group_ids','lineage']:np.testing.assert_array_equal(a[k],s[k])
        STATE[assay]=dict(rows=a['expression_rows'],y=a['y'],ids=a['sample_ids'],groups=a['group_ids'],lineage=a['lineage'],
            test=s['outer_test'],inner=s['inner_fold'],seeds=s['outer_split_seeds'],inner_seeds=s['inner_split_seeds'],
            nn_seeds=s['neural_initialization_seeds'],nn_grid=p['neural_grid'])


def features(assay,indices):
    return np.asarray(EXPRESSION[STATE[assay]['rows'][:,None],indices],dtype=float)


def grid_for(assay):
    return [dict(selector=j,**p) for j in range(-1,20) for p in STATE[assay]['nn_grid']]


def cache(assay,r):
    return np.load(ENET/assay/'folds'/f'enet_selected_ridge__r{r+1:02d}.npz')


def prepare():
    assert json.load(open(ENET/'validation.json'))['passed']
    initialize();records=[]
    for source,names in [(SOURCE,['expression.npy','expression_index.npz','protocol.json','validation.json','gene_mapping.tsv']+
                            [a+'/'+n for a in ASSAYS for n in ['inputs.npz','shared_splits.npz','predictions.npz']]),
                         (ENET,['protocol.json','validation.json','models.py','ridge_helpers.py']+
                            [f'{a}/folds/enet_selected_ridge__r{r:02d}.{ext}' for a in ASSAYS for r in range(1,21) for ext in ['npz','json']]+
                            [a+'/predictions.npz' for a in ASSAYS])]:
        known={r['path']:r['sha256'] for r in json.load(open(source/'artifact_manifest.json'))['files']}
        for name in names:
            assert sha(source/name)==known[name],name
            records.append(dict(path=str((source/name).relative_to(WS)),sha256=known[name]))
    for assay in ASSAYS:
        (OUT/assay/'folds').mkdir(parents=True,exist_ok=True)
        source=BASE[assay];known={r['path']:r['sha256'] for r in json.load(open(source/'artifact_manifest.json'))['files']}
        names=['shared_splits.npz','protocol.json','predictions.npz','summary.tsv','model_helpers.py','weighted_mlp.py']
        names += [f'folds/neural_network__p64__r{r:02d}.{ext}' for r in range(1,21) for ext in ['npz','json','joblib']]
        for name in names:
            assert sha(source/name)==known[name],name
            records.append(dict(path=str((source/name).relative_to(WS)),sha256=known[name]))
    for local,original in [('selection_helpers.py',ENET/'models.py'),('ridge_helpers.py',ENET/'ridge_helpers.py'),
                           ('model_helpers.py',BASE['CTRP']/'model_helpers.py'),('weighted_mlp.py',BASE['CTRP']/'weighted_mlp.py')]:
        assert sha(OUT/local)==sha(original),local
    reg=WS/'outputs/retained_64_gene_models/model_registry.json'
    records.append(dict(path=str(reg.relative_to(WS)),sha256=sha(reg)))
    p=dict(status='frozen_before_neural_comparison',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        retained_core=64,universe=19176,candidate_genes=19112,assays=ASSAYS,
        split_design='Identical 20 lineage-stratified identity-group 80/20 outer holdouts and five inner folds per assay, including existing singleton inner-training handling. All evaluated lineages in outer training and test.',
        gene_selection='Reuse hash-verified fold-local Elastic Net paths from the completed genome-wide EN experiment. Each cached inner selection was fitted only on that inner-training subset; each outer selection only on its outer-training subset. No whole-cohort/consensus list. The 20 EN settings are considered independently by NN tuning, without reusing the ridge-selected setting.',
        selectors=json.load(open(ENET/'protocol.json'))['selector_l1_ratios'],alpha_fractions=json.load(open(ENET/'protocol.json'))['alpha_fractions'],
        selector_adjustment='Candidate/target residuals adjusted for current-training core64 plus lineage indicators, then standardized. Constants/aliases filtered on training only. Every eligible outside gene jointly considered by EN; cached paths are unchanged.',
        selection_cache_replay='Preflight replays first/last inner folds of first repeat in each assay and all outer paths in those repeats. Prior audit covers recorded inner and independently certified outer KKT solutions across all repeats.',
        neural_grids={a:STATE[a]['nn_grid'] for a in ASSAYS},grids={a:grid_for(a) for a in ASSAYS},
        neural_fixed=json.load(open(BASE['CTRP']/'protocol.json'))['neural_fixed'],
        neural_seeds='Exact same three initialization seeds in each repeat, reused across all candidates and both inner/outer stages as original NN benchmark.',
        tuning='Jointly choose 20 Elastic Net selector settings plus zero-addition fallback and the original assay-specific NN hyperparameters by mean five-fold RMSE. All scaling refit within current NN training subset. First grid index breaks ties. Core-only NN grid predictions and zero-addition outer models reused exactly.',
        retention='All core64 input genes always present, plus every EN-selected outside gene for that setting, without top-K truncation. RNA-only, unweighted, no culture/lineage model predictors.',
        ensembles='Mean of three independently initialized networks. New fits abort on a ConvergenceWarning; SciPy deprecation warnings are counted separately and do not alter the fit.',
        primary_comparisons=[['enet_selected_nn','nn64']],
        secondary_comparisons=[['enet_selected_nn','ridge64'],['enet_selected_nn','enet64'],['enet_selected_nn','tab64'],['enet_selected_nn','enet_selected_ridge'],['enet_selected_nn','ridge_genome']],
        uncertainty=dict(count=5000,seed=2026112401,unit='Whole-lineage paired bootstrap preserving models and all overlapping repeats; conditional on fitted models, no retraining/multiplicity correction.'),
        new_inner_ensembles=24000,new_inner_networks=72000,inner_candidate_fold_scores=25200,outer_ensembles=40,
        limitations=['Previously inspected cohorts; exploratory, not untouched confirmation.','Performance belongs to the fold-specific selector/NN tuning procedure. A consensus panel requires separate validation.','Selection frequencies across overlapping splits are not causal importance or independent replications.','Measured protein-coding RNA only; original endpoint provenance and NN-grid limitations remain.'],
        scripts={n:sha(OUT/n) for n in ['common.py','run_comparison.py','validate_preflight.py','selection_helpers.py','ridge_helpers.py','model_helpers.py','weighted_mlp.py','make_splits.py']},
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__),
        published_starting_snapshot='7aa826d7ace8702d777fafc8240f5b34dd4744b2')
    if (OUT/'protocol.json').exists():
        old=json.load(open(OUT/'protocol.json'));p['created_utc']=old['created_utc'];assert old==clean(p)
    else:dump(OUT/'protocol.json',p)
    m=dict(files=records)
    if (OUT/'source_manifest.json').exists():assert json.load(open(OUT/'source_manifest.json'))==m
    else:dump(OUT/'source_manifest.json',m)
    return sha(OUT/'protocol.json')


def verify():
    for n,h in json.load(open(OUT/'protocol.json'))['scripts'].items():assert sha(OUT/n)==h,n
    for r in json.load(open(OUT/'source_manifest.json'))['files']:assert sha(WS/r['path'])==r['sha256'],r['path']
