"""Frozen extended-regularization comparison with matched core-only control."""
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import scipy
import sklearn
import joblib

OUT=Path(__file__).resolve().parent
WS=OUT.parent.parent
OLD=WS/'outputs/erastin_genomewide_enet_nn'
SOURCE=WS/'outputs/erastin_genomewide_search'
ENET=WS/'outputs/erastin_genomewide_elastic_net'
BASE={'CTRP':WS/'outputs/erastin_elastic_net_comparison','PRISM':WS/'outputs/erastin_prism_direct_comparison'}
ASSAYS=list(BASE)
NEW_ALPHAS=[300.,1000.,3000.,10000.,30000.]
ARMS=['core_extended','optional_extended','forced_extended']
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
    z=np.load(SOURCE/'expression_index.npz');INDEX={k:z[k] for k in z.files}
    for assay in ASSAYS:
        a=np.load(SOURCE/assay/'inputs.npz');s=np.load(BASE[assay]/'shared_splits.npz');p=json.load(open(BASE[assay]/'protocol.json'))
        old=p['neural_grid']
        added=[dict(width=w,alpha=alpha) for w in [16,32] for alpha in NEW_ALPHAS if dict(width=w,alpha=alpha) not in old]
        check=np.load(SOURCE/assay/'shared_splits.npz')
        for k in ['outer_test','inner_fold','outer_split_seeds','inner_split_seeds']:np.testing.assert_array_equal(s[k],check[k])
        for k in ['sample_ids','group_ids','lineage']:np.testing.assert_array_equal(a[k],s[k])
        STATE[assay]=dict(rows=a['expression_rows'],y=a['y'],ids=a['sample_ids'],groups=a['group_ids'],lineage=a['lineage'],
            test=s['outer_test'],inner=s['inner_fold'],seeds=s['outer_split_seeds'],inner_seeds=s['inner_split_seeds'],nn_seeds=s['neural_initialization_seeds'],
            old_nn_grid=old,new_nn_grid=added,nn_grid=old+added)


def features(assay,indices):
    return np.asarray(EXPRESSION[STATE[assay]['rows'][:,None],indices],dtype=float)


def grid_for(assay):
    return [dict(selector=j,**p) for j in range(-1,20) for p in STATE[assay]['nn_grid']]


def cache(assay,r):return np.load(ENET/assay/'folds'/f'enet_selected_ridge__r{r+1:02d}.npz')


def old_columns(assay):
    s=STATE[assay];n,no=len(s['nn_grid']),len(s['old_nn_grid'])
    return np.array([block*n+j for block in range(21) for j in range(no)])


def prepare():
    initialize();assert json.load(open(OLD/'validation.json'))['passed'];records=[]
    names=['protocol.json','validation.json','model_helpers.py','weighted_mlp.py','selection_helpers.py','ridge_helpers.py']
    names += [f'{a}/folds/inner__r{r:02d}__f{k}.{ext}' for a in ASSAYS for r in range(1,21) for k in range(5) for ext in ['npz','json']]
    names += [f'{a}/folds/enet_selected_nn__r{r:02d}.{ext}' for a in ASSAYS for r in range(1,21) for ext in ['npz','json','joblib']]
    names += [f'forced_additions/{a}/folds/forced_nn__r{r:02d}.{ext}' for a in ASSAYS for r in range(1,21) for ext in ['npz','json','joblib']]
    names += [a+'/predictions.npz' for a in ASSAYS]+['forced_additions/'+a+'/predictions.npz' for a in ASSAYS]
    for source,files in [(OLD,names),(SOURCE,['expression.npy','expression_index.npz','protocol.json','validation.json']+
                [a+'/'+n for a in ASSAYS for n in ['inputs.npz','shared_splits.npz','predictions.npz']]),
            (ENET,['protocol.json','validation.json']+[f'{a}/folds/enet_selected_ridge__r{r:02d}.{ext}' for a in ASSAYS for r in range(1,21) for ext in ['npz','json']])]:
        known={r['path']:r['sha256'] for r in json.load(open(source/'artifact_manifest.json'))['files']}
        for name in files:
            assert sha(source/name)==known[name],name
            records.append(dict(path=str((source/name).relative_to(WS)),sha256=known[name]))
    for assay in ASSAYS:
        (OUT/assay/'folds').mkdir(parents=True,exist_ok=True)
        src=BASE[assay];known={r['path']:r['sha256'] for r in json.load(open(src/'artifact_manifest.json'))['files']}
        names=['shared_splits.npz','protocol.json','predictions.npz']+[f'folds/neural_network__p64__r{r:02d}.{ext}' for r in range(1,21) for ext in ['npz','json','joblib']]
        for name in names:
            assert sha(src/name)==known[name]
            records.append(dict(path=str((src/name).relative_to(WS)),sha256=known[name]))
    for name in ['model_helpers.py','weighted_mlp.py','selection_helpers.py','ridge_helpers.py','make_splits.py','evaluate.py','metric_helpers.py','analysis_helpers.py']:
        assert sha(OUT/name)==sha(OLD/name),name
    reg=WS/'outputs/retained_64_gene_models/model_registry.json';records.append(dict(path=str(reg.relative_to(WS)),sha256=sha(reg)))
    p=dict(status='frozen_before_stronger_regularization_fits',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        core_panel=64,assays=ASSAYS,universe=19176,candidates_outside_core=19112,new_penalties=NEW_ALPHAS,
        neural_grids={a:STATE[a]['nn_grid'] for a in ASSAYS},new_neural_settings={a:STATE[a]['new_nn_grid'] for a in ASSAYS},
        grids={a:grid_for(a) for a in ASSAYS},
        split_design='Same 20 identity-group/lineage-stratified 80/20 outer holdouts and five saved inner folds per assay. Every evaluated lineage in both outer sets; Cervix exclusion and protected inner-training singleton unchanged.',
        selector='Same 20 hash-verified, fold-local Elastic Net selections: adjusted for core64 + training lineage, constants/aliases filtered on training only. No whole-cohort or consensus list; all selected genes appended without a top-K cap.',
        reuse='Reuse every old NN candidate prediction by exact (selector,width,alpha) mapping from the verified previous benchmark. Fit only unseen penalties/settings, including core-only control. Original initialization triplets and NN solver/scaling are unchanged.',
        neural_fixed=json.load(open(OLD/'protocol.json'))['neural_fixed'],
        objective='On current-training-standardized X and y: 0.5*mean(squared error)+0.5*alpha*sum(weight matrices squared)/n_training; biases unpenalized. Unit sample weights.',
        selection='Mean five-fold inner RMSE in original response units. Core-only control tunes the extended NN grid; optional arm jointly tunes core-only or 20 EN settings and the same extended NN grid; forced arm tunes only the 20 nonempty settings. First grid index breaks ties. Outer refits use only outer training.',
        arms=ARMS,primary_comparisons=[['optional_extended','core_extended'],['forced_extended','core_extended']],
        secondary_comparisons=[['core_extended','nn64'],['optional_extended','nn64'],['optional_extended','old_optional'],['forced_extended','old_forced'],['optional_extended','tab64']],
        uncertainty=dict(count=5000,seed=2026112501,unit='Paired whole-lineage bootstrap retaining all models and overlapping repeats together; conditional on fits, no retraining or multiplicity adjustment.'),
        new_inner_ensembles=sum(len(STATE[a]['new_nn_grid'])*21*5*20 for a in ASSAYS),
        new_inner_networks=sum(len(STATE[a]['new_nn_grid'])*21*5*20*3 for a in ASSAYS),
        candidate_fold_scores=sum(len(grid_for(a))*5*20 for a in ASSAYS),logical_outer_ensembles=120,
        numerical_policy='Same tanh/L-BFGS, max_iter=2000,max_fun=40000,tol=0.001 and three-network seed triplets; abort on ConvergenceWarning. Record SciPy deprecation warnings separately.',
        retention='Core64 always included. RNA-only, unweighted, no culture/lineage prediction covariates. Retained model registry stays unchanged.',
        limitations=['Previously inspected cohorts: exploratory despite fold-local selection and tuning.','Finite extended grid; no post-result penalty expansion.','Performance concerns the selection/tuning procedure, not a consensus gene list or individual causal genes.','Original endpoint provenance and protein-coding RNA scope remain.'],
        references=['https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html'],
        scripts={n:sha(OUT/n) for n in ['common.py','run_comparison.py','validate_preflight.py','model_helpers.py','weighted_mlp.py','selection_helpers.py','ridge_helpers.py','make_splits.py']},
        versions=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,joblib=joblib.__version__))
    if (OUT/'protocol.json').exists():
        old=json.load(open(OUT/'protocol.json'));p['created_utc']=old['created_utc'];assert old==clean(p)
    else:dump(OUT/'protocol.json',p)
    m=dict(files=records)
    if (OUT/'source_manifest.json').exists():assert json.load(open(OUT/'source_manifest.json'))==m
    else:dump(OUT/'source_manifest.json',m)
    return sha(OUT/'protocol.json')


def verify():
    for name,h in json.load(open(OUT/'protocol.json'))['scripts'].items():assert sha(OUT/name)==h,name
    for row in json.load(open(OUT/'source_manifest.json'))['files']:assert sha(WS/row['path'])==row['sha256'],row['path']
