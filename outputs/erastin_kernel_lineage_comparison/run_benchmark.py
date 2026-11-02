#!/usr/bin/env python3
"""Fixed, resumable 64-gene kernel/lineage comparison in CTRP and PRISM."""
import argparse
import concurrent.futures as futures
import hashlib
import json
import multiprocessing as mp
import platform
import shutil
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import scipy
import sklearn

from models import FAMILIES, GRIDS, path
from make_splits import inner_rows

OUT = Path(__file__).resolve().parent
WS = OUT.parent.parent
SOURCES = {'CTRP': WS/'outputs/erastin_elastic_net_comparison',
           'PRISM': WS/'outputs/erastin_prism_direct_comparison'}
BASELINES = ['ridge', 'elastic_net', 'neural_network']
WORKERS = 6
STATE = {}
sys.dont_write_bytecode = True


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


def root_for(assay, family, r):
    return OUT/assay/'folds'/f'{family}__p64__r{r+1:02d}'


def initialize():
    for assay in SOURCES:
        a = np.load(OUT/assay/'panel_inputs.npz')
        s = np.load(OUT/assay/'shared_splits.npz')
        STATE[assay] = dict(x=a['x_64'], y=a['y'], genes=a['genes_64'], ids=a['sample_ids'],
                            groups=a['group_ids'], lineage=a['lineage'], test=s['outer_test'],
                            inner=s['inner_fold'], seeds=s['outer_split_seeds'], inner_seeds=s['inner_split_seeds'])


def prepare():
    records = []
    cohorts = {}
    for assay, source in SOURCES.items():
        (OUT/assay/'folds').mkdir(parents=True, exist_ok=True)
        known = {v['path']: v['sha256'] for v in json.load(open(source/'artifact_manifest.json'))['files']}
        names = ['panel_inputs.npz', 'shared_splits.npz', 'protocol.json', 'validation.json',
                 'predictions.npz', 'summary.tsv', 'make_splits.py', 'evaluate.py', 'metric_helpers.py', 'weighted_mlp.py']
        names += [f'folds/{f}__p64__r{r:02d}.{ext}' for f in BASELINES
                  for r in range(1, 21) for ext in ['json', 'npz', 'joblib']]
        for name in names:
            assert sha(source/name) == known[name], (assay, name)
            records.append(dict(path=str((source/name).relative_to(WS)), sha256=known[name]))
        assert json.load(open(source/'validation.json'))['passed']
        for name in ['panel_inputs.npz', 'shared_splits.npz']:
            dest = OUT/assay/name
            if dest.exists():
                assert sha(dest) == known[name]
            else:
                shutil.copyfile(source/name, dest)
        old = json.load(open(source/'protocol.json'))
        cohorts[assay] = dict(samples=old['samples'], identity_groups=old['identity_groups'],
                             lineages=old['lineages'], outer_train_sizes=old['outer_train_sizes'],
                             outer_test_sizes=old['outer_test_sizes'], source=str(source.relative_to(WS)),
                             split_design=old['split_design'], inner_design=old['inner_design'],
                             target='Negative harmonized CTRP AUC' if assay == 'CTRP' else 'Negative reported PRISM erastin AUC',
                             input_sha256=known['panel_inputs.npz'], split_sha256=known['shared_splits.npz'])
    for name in ['make_splits.py', 'evaluate.py', 'metric_helpers.py', 'weighted_mlp.py']:
        assert sha(OUT/name) == sha(SOURCES['PRISM']/name)
    p = dict(status='frozen_before_new_model_fits', created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             cohorts=cohorts, gene_count=64, weighting='Unweighted training and equal-mean five inner-fold RMSE.',
             preprocessing='X and y StandardScaler fitted only on the current training subset. No feature selection.',
             culture_covariates=False, pooled_assays=False, grids=GRIDS,
             svr='RBF SVR on standardized X/y; tol=1e-6, max_iter=200000; precomputed kernels reused only within the same training fold.',
             kernel_ridge='Training-centered kernel, unpenalized intercept, K=(1-w)*ZZt/64+w*RBF; squared-error sum plus alpha RKHS penalty. Linear-only fallback included.',
             hierarchical_ridge='Unpenalized global intercept and shared standardized-gene slopes plus unscaled one-hot lineage offsets. Minimize SSE + gene_alpha*||beta||^2 + lineage_alpha*||b||^2. Gaussian random-intercept MAP / partial pooling, not lineage-specific slopes. None lineage_alpha sets all offsets to zero.',
             lineage_only='Same partially pooled lineage offsets, without gene effects; None lineage_alpha is training-mean prediction.',
             tuning='First candidate minimizing arithmetic mean of five inner-fold RMSEs in original response units. Existing inner assignments including -2 always-training CTRP Ampulla retained exactly.',
             primary_comparisons=[[f, b] for f in FAMILIES[:3] for b in ['ridge', 'neural_network']],
             secondary_comparisons=[[f, 'elastic_net'] for f in FAMILIES[:3]]+[['hierarchical_ridge', 'lineage_only'], ['lineage_only', 'training_mean']],
             primary_metric='pooled_rmse', secondary_metrics=['pooled_spearman', 'pooled_r2', 'macro_all_rmse', 'within_lineage_centered_spearman'],
             uncertainty=dict(count=5000, seed=2026111801,
                              unit='Paired resampling of entire lineage blocks, preserving all repeats and models together. Conditional on saved cohort/fits; no independent-repeat t test; unadjusted exploratory intervals.'),
             new_outer_models=160, reused_outer_models=120, workers=WORKERS,
             limitations=['Repeatedly inspected cohort and overlapping holdouts; exploratory model comparison, not confirmatory external validation.',
                          'All lineages represented in training and testing; does not test unseen-lineage generalization.',
                          'Original harmonized CTRP AUC generation provenance remains unresolved; CTRP/PRISM outcomes remain distinct.',
                          'Sparse lineage test subsets may have undefined correlations; macro RMSE retains every lineage.',
                          'No post-result grid expansion; boundary selections are reported. Retained model registry remains unchanged.'],
             versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__),
             scripts={n: sha(OUT/n) for n in ['models.py', 'run_benchmark.py', 'validate_preflight.py', 'review_benchmark.py',
                                             'make_splits.py', 'evaluate.py', 'metric_helpers.py', 'weighted_mlp.py']})
    pp = OUT/'protocol.json'
    if pp.exists():
        prev = json.load(open(pp))
        p['created_utc'] = prev['created_utc']
        assert clean(p) == prev
    else:
        dump(pp, p)
    dump(OUT/'source_manifest.json', dict(files=records))
    return sha(pp)


def fit_job(assay, family, r, ph):
    root = root_for(assay, family, r)
    if root.with_suffix('.json').exists():
        m = json.load(open(root.with_suffix('.json')))
        assert m['protocol_sha256'] == ph
        for ext in ['npz', 'joblib']:
            assert sha(root.with_suffix('.'+ext)) == m[ext+'_sha256']
        return m
    begin = time.monotonic()
    s = STATE[assay]
    tr, te = np.flatnonzero(~s['test'][r]), np.flatnonzero(s['test'][r])
    inner, grid = s['inner'][r], GRIDS[family]
    val = np.full((len(grid), len(s['y'])), np.nan)
    loss = np.empty((len(grid), 5))
    diagnostics = []
    for k in range(5):
        a, b = inner_rows(inner, k)
        assert set(a)|set(b) == set(tr)
        assert set(s['groups'][a]).isdisjoint(s['groups'][b])
        assert set(s['lineage'][a]) == set(s['lineage'])
        pred, _, diag = path(family, grid, s['x'], s['y'], s['lineage'], a, b)
        val[:, b] = pred
        loss[:, k] = np.sqrt(np.mean((pred-s['y'][b][None, :])**2, axis=1))
        diagnostics.append(dict(stage='inner', fold=k, **diag))
    assert np.isfinite(val[:, inner >= 0]).all() and np.isnan(val[:, inner < 0]).all()
    c = int(np.argmin(loss.mean(axis=1)))
    pred, model, diag = path(family, [grid[c]], s['x'], s['y'], s['lineage'], tr, te, retain=True)
    diagnostics.append(dict(stage='outer', **diag))
    joblib.dump(model, root.with_suffix('.joblib'), compress=3)
    np.savez_compressed(root.with_suffix('.npz'), train_rows=tr, test_rows=te,
                        test_prediction=pred[0], inner_validation_predictions=val, inner_fold_rmse=loss)
    m = dict(assay=assay, family=family, panel=64, repeat=r+1, train_n=len(tr), test_n=len(te),
             outer_split_seed=int(s['seeds'][r]), inner_split_seed=int(s['inner_seeds'][r]),
             selected_candidate=c, selected_parameters=grid[c], mean_inner_rmse=float(loss[c].mean()),
             fit_diagnostics=diagnostics, protocol_sha256=ph, seconds=time.monotonic()-begin,
             npz_sha256=sha(root.with_suffix('.npz')), joblib_sha256=sha(root.with_suffix('.joblib')))
    dump(root.with_suffix('.json'), m)
    return m


def verify_sources():
    for row in json.load(open(OUT/'source_manifest.json'))['files']:
        assert sha(WS/row['path']) == row['sha256']
    p = json.load(open(OUT/'protocol.json'))
    for name, h in p['scripts'].items():
        assert sha(OUT/name) == h
    for assay, c in p['cohorts'].items():
        assert sha(OUT/assay/'panel_inputs.npz') == c['input_sha256']
        assert sha(OUT/assay/'shared_splits.npz') == c['split_sha256']


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare', action='store_true')
    args = parser.parse_args()
    start = time.monotonic()
    ph = prepare()
    if args.prepare:
        print('PREPARED', ph)
        return
    initialize()
    pf = json.load(open(OUT/'preflight_validation.json'))
    assert pf['passed'] and pf['protocol_sha256'] == ph
    done = 0
    with futures.ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context('spawn'), initializer=initialize) as pool:
        jobs = [pool.submit(fit_job, a, f, r, ph) for a in SOURCES for r in range(20) for f in FAMILIES]
        for future in futures.as_completed(jobs):
            m = future.result()
            done += 1
            elapsed = time.monotonic()-start
            progress = dict(phase='fitting', completed=done, total=160, reused=120, elapsed_seconds=elapsed,
                            estimated_remaining_seconds=elapsed/done*(160-done), last_assay=m['assay'], last_family=m['family'])
            dump(OUT/'progress.json', progress)
            if done % 8 == 0:
                print(json.dumps(progress), flush=True)
    verify_sources()
    dump(OUT/'fit_completion.json', dict(complete=True, new_outer_models=160, reused_outer_models=120, seconds=time.monotonic()-start))
    dump(OUT/'progress.json', dict(phase='fits_complete', completed=160, total=160, reused=120))
    print('ALL_FITS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
