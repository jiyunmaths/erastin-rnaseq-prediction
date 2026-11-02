"""Shared data/provenance operations; usable from both pinned Python environments."""
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
WS = OUT.parent.parent
WORK = WS/'work/tabpfn'
SOURCE = WS/'outputs/erastin_hierarchical_slopes'
CHECKPOINT = WORK/'checkpoints/tabpfn-v3.5-fast-20260909.safetensors'
ASSAYS = ['CTRP', 'PRISM']
BASELINES = ['ridge', 'elastic_net', 'neural_network', 'svr', 'kernel_ridge', 'hierarchical_ridge',
             'matched_intercept', 'hierarchical_slopes']
N_ESTIMATORS = 8
THREADS = 3
WORKERS = 2
STATE = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def clean(v):
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [clean(x) for x in v]
    if isinstance(v, np.generic):
        return clean(v.item())
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def dump(p, value):
    tmp = p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False)+'\n')
    tmp.replace(p)


def root_for(assay, r):
    return OUT/assay/'folds'/f'tabpfn35_fast__p64__r{r+1:02d}'


def initialize():
    for assay in ASSAYS:
        a = np.load(OUT/assay/'panel_inputs.npz')
        s = np.load(OUT/assay/'shared_splits.npz')
        STATE[assay] = dict(x=a['x_64'], y=a['y'], genes=a['genes_64'], ids=a['sample_ids'],
                            groups=a['group_ids'], lineage=a['lineage'], test=s['outer_test'],
                            inner=s['inner_fold'], seeds=s['outer_split_seeds'], inner_seeds=s['inner_split_seeds'])


def prepare():
    known = {r['path']: r['sha256'] for r in json.load(open(SOURCE/'artifact_manifest.json'))['files']}
    records = []
    for name in ['protocol.json', 'validation.json', 'summary.tsv', 'source_manifest.json']:
        assert sha(SOURCE/name) == known[name]
        records.append(dict(path=str((SOURCE/name).relative_to(WS)), sha256=known[name]))
    assert json.load(open(SOURCE/'validation.json'))['passed']
    for assay in ASSAYS:
        (OUT/assay/'folds').mkdir(parents=True, exist_ok=True)
        for name in ['panel_inputs.npz', 'shared_splits.npz', 'predictions.npz', 'validation.json']:
            original = SOURCE/assay/name
            assert sha(original) == known[f'{assay}/{name}']
            records.append(dict(path=str(original.relative_to(WS)), sha256=sha(original)))
            if name in ['panel_inputs.npz', 'shared_splits.npz']:
                dest = OUT/assay/name
                if dest.exists():
                    assert sha(dest) == sha(original)
                else:
                    shutil.copyfile(original, dest)
    # Preserve all baseline source model hashes established by the previous audit.
    for r in json.load(open(SOURCE/'source_manifest.json'))['files']:
        assert sha(WS/r['path']) == r['sha256']
        records.append(r)
    for assay in ASSAYS:
        for family in ['matched_intercept', 'hierarchical_slopes']:
            for r in range(1, 21):
                for ext in ['json', 'npz', 'joblib']:
                    name = f'{assay}/folds/{family}__p64__r{r:02d}.{ext}'
                    assert sha(SOURCE/name) == known[name]
                    records.append(dict(path=str((SOURCE/name).relative_to(WS)), sha256=known[name]))
    weights = json.load(open(CHECKPOINT.with_suffix(CHECKPOINT.suffix+'.json')))
    assert sha(CHECKPOINT) == weights['sha256'] == weights['expected_sha256']
    for path in [CHECKPOINT, CHECKPOINT.with_suffix(CHECKPOINT.suffix+'.json'), WORK/'install_report.json', WORK/'pilot_results.json']:
        records.append(dict(path=str(path.relative_to(WS)), sha256=sha(path)))
    old = json.load(open(SOURCE/'protocol.json'))
    p = dict(status='frozen_before_TabPFN_benchmark_fits', created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             cohorts=old['cohorts'], genes=64, culture_covariates=False, lineage_predictor=False,
             weighting='Unweighted training. Fixed pretrained TabPFN configuration; no local gradient training.',
             model='TabPFN-3.5-Fast regression, TabPFN package 9.0.0', checkpoint=weights,
             settings=dict(n_estimators=N_ESTIMATORS, device='cpu', inference_precision='torch.float32',
                           n_preprocessing_jobs=1, categorical_features_indices=[], fit_mode='fit_preprocessors',
                           memory_saving_mode='auto', softmax_temperature='auto', tuning_config=None,
                           predict_output='mean', random_state='Original outer-split seed for each repeat'),
             preprocessing='Input is unchanged 64-gene log RNA. No all-cohort scaling. Native feature/target preprocessing fit on outer-training rows only; predict transforms held-out features without fitting. Backbone train-context statistics exclude query rows, checked by query-neighbor invariance.',
             tuning='No study-specific hyperparameter, checkpoint, calibration or ensemble-weight tuning. Original inner-fold assignments are retained and audited but are not used for selection of this fixed model. Baseline nested tuning remains unchanged.',
             selection='CPU timing only selected 3.5-Fast over the slower v2 pilot. No pilot accuracy computed. Eight passes fixed before benchmark fits; no test-informed checkpoint or setting selection.',
             isolation='Separate venv for new dependencies; existing baseline analysis environment unchanged. Local checkpoint paths, offline HF mode, and network disabled during fit/predict.',
             primary_comparisons=[['tabpfn35_fast', 'ridge'], ['tabpfn35_fast', 'neural_network'],
                                  ['tabpfn35_fast', 'svr'], ['tabpfn35_fast', 'kernel_ridge']],
             secondary_comparisons=[['tabpfn35_fast', 'elastic_net'], ['tabpfn35_fast', 'hierarchical_ridge'],
                                    ['tabpfn35_fast', 'matched_intercept'], ['tabpfn35_fast', 'hierarchical_slopes']],
             primary_metric='pooled_rmse', secondary_metrics=['pooled_spearman', 'pooled_r2', 'macro_all_rmse'],
             uncertainty=dict(count=5000, seed=2026112101,
                              unit='Paired bootstrap of whole lineages, retaining all models and overlapping repeats together. Conditional exploratory intervals; no independent-repeat test.'),
             new_fitted_contexts=40, reused_baseline_models=320, workers=WORKERS, threads_per_worker=THREADS,
             limitations=['Repeatedly inspected cohorts/splits remain exploratory; confirmation on new data is required.',
                          'All evaluated lineages represented in training/testing; no unseen-lineage or patient claim.',
                          'Results apply to 3.5-Fast, this checkpoint and eight passes; not every TabPFN version or tuning strategy.',
                          'Pretraining-corpus overlap with these pharmacogenomic data has not been independently audited.',
                          'Original harmonized CTRP AUC generation provenance remains unresolved; no assay pooling or target changes.'],
             references=['https://github.com/PriorLabs/TabPFN', 'https://huggingface.co/Prior-Labs/tabpfn_3_5',
                         'https://www.nature.com/articles/s41586-024-08328-6'],
             data_sha256={f'{a}/{n}': sha(OUT/a/n) for a in ASSAYS for n in ['panel_inputs.npz', 'shared_splits.npz']},
             scripts={n: sha(OUT/n) for n in ['common.py', 'run_tabpfn.py', 'validate_preflight.py', 'make_splits.py']})
    if (OUT/'protocol.json').exists():
        prior = json.load(open(OUT/'protocol.json'))
        p['created_utc'] = prior['created_utc']
        assert clean(p) == prior
    else:
        dump(OUT/'protocol.json', p)
    dump(OUT/'source_manifest.json', dict(files=list({r['path']: r for r in records}.values())))
    return sha(OUT/'protocol.json')


def verify_sources():
    for row in json.load(open(OUT/'source_manifest.json'))['files']:
        assert sha(WS/row['path']) == row['sha256']
    p = json.load(open(OUT/'protocol.json'))
    for name, h in {**p['scripts'], **p['data_sha256']}.items():
        assert sha(OUT/name) == h
