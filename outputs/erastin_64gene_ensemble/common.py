"""Retained-model provenance and unchanged nested validation splits."""
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
REGISTRY=WS/'outputs/retained_64_gene_models/model_registry.json'
ASSAYS=['CTRP','PRISM']
STATE={}
FAMILIES=['ridge','elastic_net','neural_network']
ARMS=['equal_blend','learned_blend','inner_best_single']


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
    registry=json.load(open(REGISTRY))
    for assay in ASSAYS:
        d=registry['cohorts'][assay];source=WS/d['source_experiment'];a=np.load(source/'panel_inputs.npz');s=np.load(source/'shared_splits.npz')
        np.testing.assert_array_equal(a['genes_64'],registry['genes_in_input_order'])
        for k in ['sample_ids','group_ids','lineage']:np.testing.assert_array_equal(a[k],s[k])
        STATE[assay]=dict(source=source,registry=d,x=a['x_64'],genes=a['genes_64'],y=a['y'],ids=a['sample_ids'],groups=a['group_ids'],lineage=a['lineage'],
            test=s['outer_test'],inner=s['inner_fold'],seeds=s['outer_split_seeds'],inner_seeds=s['inner_split_seeds'])


def source_root(assay,family,r):
    return STATE[assay]['source']/'folds'/f'{family}__p64__r{r+1:02d}'


def inputs(assay,r):
    """The final retained hyperparameter choice determines each base column."""
    s=STATE[assay];valid=s['inner'][r]>=0;inner=[];outer=[];parameters=[]
    for family in FAMILIES:
        root=source_root(assay,family,r);a=np.load(root.with_suffix('.npz'));m=json.load(open(root.with_suffix('.json')))
        inner.append(a['inner_validation_predictions'][m['selected_candidate'],valid])
        outer.append(a['test_prediction'])
        parameters.append(dict(family=family,selected_candidate=m['selected_candidate'],parameters=m['selected_parameters']))
    return np.column_stack(inner),np.column_stack(outer),valid,parameters


def prepare():
    initialize();registry=json.load(open(REGISTRY));assert registry['gene_count']==64 and not registry['culture_covariates'] and registry['training_weights']=='Unweighted'
    records=[dict(path=str(REGISTRY.relative_to(WS)),sha256=sha(REGISTRY))]
    for assay in ASSAYS:
        s=STATE[assay];d=s['registry'];source=s['source'];(OUT/assay/'folds').mkdir(parents=True,exist_ok=True)
        manifest=WS/d['source_artifact_manifest']['path'];assert sha(manifest)==d['source_artifact_manifest']['sha256']
        known={r['path']:r['sha256'] for r in json.load(open(manifest))['files']}
        assert json.load(open(source/'validation.json'))['passed']
        for name in ['panel_inputs.npz','shared_splits.npz','protocol.json','validation.json','summary.tsv','predictions.npz','weighted_mlp.py']:
            assert sha(source/name)==known[name],name
            records.append(dict(path=str((source/name).relative_to(WS)),sha256=known[name]))
        for family in FAMILIES:
            for r,item in enumerate(d['models'][family]['outer_models']):
                root=source_root(assay,family,r)
                for ext,row in item['files'].items():
                    assert WS/row['path']==root.with_suffix('.'+ext)
                    assert sha(root.with_suffix('.'+ext))==row['sha256']
                    records.append(dict(path=row['path'],sha256=row['sha256']))
        assert sha(OUT/'weighted_mlp.py')==sha(source/'weighted_mlp.py')
    p=dict(status='frozen_before_ensemble_evaluation',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        retained_panel=64,base_family_order=FAMILIES,assays=ASSAYS,outer_repeats_per_assay=20,
        split_design='Unchanged saved lineage-stratified identity-group 80/20 holdouts and five inner folds. Every evaluated lineage in outer train/test; protected inner-training singleton rows omitted from weight fitting because no inner validation prediction exists.',
        base_models='Use exactly the 120 retained outer models, their chosen hyperparameters and saved inner/outer predictions. No RNA model refit or hyperparameter change. NN is its original mean of three networks and counts as one family.',
        arms=dict(equal_blend='Fixed weights 1/3 for each family; no tuning.',
            learned_blend='Nonnegative family weights summing to one, no intercept/calibration/clipping. Fit on saved inner-validation predictions at retained base hyperparameters by equal-mean five-fold RMSE. Zero weights allowed; vertices and all edges included.',
            inner_best_single='Choose the family with lowest mean five-fold inner RMSE, as a model-selection-only control; family order breaks ties.'),
        fitting='Weight optimizer receives only rows with original inner assignment >=0. It never receives outer-test responses. Scaling of the meta training target/predictions is training-only and only improves numerical conditioning; weights are applied to base predictions in original response units.',
        numerical=dict(slsqp_ftol=1e-12,maxiter=1000,kkt_limit=2e-6,search='All three vertices, all three bounded scalar edge optima, uniform weights, two SLSQP starts. Accept best candidate only after global simplex KKT check.'),
        primary_comparisons=[[a,b] for a in ['equal_blend','learned_blend'] for b in FAMILIES],
        secondary_comparisons=[['learned_blend','equal_blend'],['equal_blend','inner_best_single'],['learned_blend','inner_best_single']],
        uncertainty=dict(count=5000,seed=2026112601,unit='Paired whole-lineage bootstrap preserving all overlapping repeats/models together; conditional on fitted models, no retraining/multiplicity adjustment.'),
        source_prediction_audit='Reproduce all retained outer predictions from their saved model binaries; verify exact data/splits, all base inner RMSEs/choices, registry hashes and ensemble arithmetic.',
        scope='RNA-only, unweighted, no culture/lineage predictors. Retained registry/model binaries unchanged. CTRP and PRISM trained/evaluated separately.',
        limitations=['These cohorts have informed prior experiments; results remain exploratory internal validation.',
            'The same inner labels originally choose base hyperparameters and now fit blend weights. Chosen-hyperparameter inner losses are optimistic training diagnostics, not independently cross-validated ensemble accuracy. Only outer-test scores assess the full ensemble procedure.',
            'No second layer of inner hyperparameter cross-fitting is added; this benchmark blends the retained models at their retained settings.',
            'Weights across overlapping splits are not independent replication or gene importance. Learned zero weights may omit a family.',
            'Original endpoint provenance, 64-gene panel selection history and within-represented-lineage scope remain.'],
        references=['https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingRegressor.html','https://docs.scipy.org/doc/scipy/reference/optimize.minimize-slsqp.html'],
        scripts={name:sha(OUT/name) for name in ['models.py','common.py','run_comparison.py','validate_preflight.py','make_splits.py','weighted_mlp.py']},
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
