#!/usr/bin/env python3
"""Frozen random-intercept/random-slope comparison using saved assay splits."""
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

from models import GRID, ARMS, path
from make_splits import inner_rows

OUT = Path(__file__).resolve().parent
WS = OUT.parent.parent
SOURCE = WS/'outputs/erastin_kernel_lineage_comparison'
ORIGINAL = {'CTRP': WS/'outputs/erastin_elastic_net_comparison', 'PRISM': WS/'outputs/erastin_prism_direct_comparison'}
BASELINES = ['ridge', 'elastic_net', 'neural_network', 'svr', 'kernel_ridge', 'hierarchical_ridge', 'lineage_only']
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


def root_for(assay, arm, r):
    return OUT/assay/'folds'/f'{arm}__p64__r{r+1:02d}'


def initialize():
    for assay in ORIGINAL:
        a = np.load(OUT/assay/'panel_inputs.npz')
        s = np.load(OUT/assay/'shared_splits.npz')
        STATE[assay] = dict(x=a['x_64'], y=a['y'], genes=a['genes_64'], ids=a['sample_ids'], groups=a['group_ids'],
                            lineage=a['lineage'], test=s['outer_test'], inner=s['inner_fold'],
                            seeds=s['outer_split_seeds'], inner_seeds=s['inner_split_seeds'])


def eligible(arm):
    if arm == 'matched_intercept':
        return np.array([j for j, p in enumerate(GRID) if p['slope_weight'] == 0])
    if arm == 'positive_slopes':
        return np.array([j for j, p in enumerate(GRID) if p['slope_weight'] > 0])
    return np.arange(len(GRID))


def prepare():
    records, cohorts = [], {}
    known = {v['path']: v['sha256'] for v in json.load(open(SOURCE/'artifact_manifest.json'))['files']}
    for name in ['protocol.json', 'validation.json', 'summary.tsv', 'analysis_source_manifest.json']:
        assert sha(SOURCE/name) == known[name]
        records.append(dict(path=str((SOURCE/name).relative_to(WS)), sha256=known[name]))
    assert json.load(open(SOURCE/'validation.json'))['passed']
    for assay, original in ORIGINAL.items():
        (OUT/assay/'folds').mkdir(parents=True, exist_ok=True)
        names = [f'{assay}/{n}' for n in ['panel_inputs.npz', 'shared_splits.npz', 'predictions.npz', 'validation.json']]
        names += [f'{assay}/folds/{f}__p64__r{r:02d}.{ext}' for f in BASELINES[3:]
                  for r in range(1, 21) for ext in ['json', 'npz', 'joblib']]
        for name in names:
            assert sha(SOURCE/name) == known[name]
            records.append(dict(path=str((SOURCE/name).relative_to(WS)), sha256=known[name]))
        ok = {v['path']: v['sha256'] for v in json.load(open(original/'artifact_manifest.json'))['files']}
        names = ['protocol.json', 'validation.json', 'panel_inputs.npz', 'shared_splits.npz']
        names += [f'folds/{f}__p64__r{r:02d}.{ext}' for f in BASELINES[:3] for r in range(1, 21) for ext in ['json', 'npz', 'joblib']]
        for name in names:
            assert sha(original/name) == ok[name]
            records.append(dict(path=str((original/name).relative_to(WS)), sha256=ok[name]))
        for name in ['panel_inputs.npz', 'shared_splits.npz']:
            assert sha(SOURCE/assay/name) == ok[name]
            dest = OUT/assay/name
            if dest.exists():
                assert sha(dest) == ok[name]
            else:
                shutil.copyfile(SOURCE/assay/name, dest)
        old = json.load(open(original/'protocol.json'))
        cohorts[assay] = dict(samples=old['samples'], identity_groups=old['identity_groups'], lineages=old['lineages'],
                             outer_train_sizes=old['outer_train_sizes'], outer_test_sizes=old['outer_test_sizes'],
                             split_design=old['split_design'], inner_design=old['inner_design'],
                             target='Unchanged negative harmonized CTRP AUC' if assay == 'CTRP' else 'Unchanged negative reported PRISM erastin AUC')
    registry = WS/'outputs/retained_64_gene_models/model_registry.json'
    records.append(dict(path=str(registry.relative_to(WS)), sha256=sha(registry)))
    p = dict(status='frozen_before_random_slope_fits', created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             cohorts=cohorts, genes=64, weighting='Unweighted observations; equal-mean five inner-fold RMSE.',
             culture_covariates=False, expanded_genes=False, pooled_assays=False,
             model='y = global_intercept + Z beta + lineage_offset + (Z - training_lineage_mean_Z) u_lineage + error.',
             objective='SSE + alpha*||beta||^2 + alpha/intercept_weight*sum(a_lineage^2) + alpha/slope_weight*sum(||u_lineage||^2). Global intercept unpenalized. A zero variance weight removes that random term.',
             interpretation='Gaussian random-intercept and independent isotropic random-slope MAP estimates with inner-CV prior/noise ratios; not MCMC posterior sampling or estimated biological interaction significance.',
             centering='All X/y standardization and within-lineage X means use only the current training subset. Local slopes use the same global gene-SD scale; no per-lineage response scaling. Singleton training lineages have zero local design and exactly zero slope deviation.',
             kernel='K = ZZt + intercept_weight*same_lineage + slope_weight*(QQt)*same_lineage; Q is training-lineage-centered Z. Training-centered kernel plus alpha*I fits a free global intercept.',
             grid=GRID, grid_order='Slope variance first (zero first), intercept variance next (zero first), then ascending global penalty. First lowest mean inner RMSE wins ties.',
             arms=dict(matched_intercept='85 candidates with slope_weight=0 under the same variance-ratio parameterization.',
                       hierarchical_slopes='All 425 candidates, including zero-slope and global-ridge fallbacks.',
                       positive_slopes='340 candidates requiring positive slope variance; prespecified sensitivity analysis.'),
             baseline='Reuse original ridge/Elastic Net/NN and prior SVR/kernel-ridge/random-intercept/lineage-only models. The previous random-intercept model uses an independent intercept-penalty grid, so the newly matched intercept arm isolates slope flexibility.',
             tuning='Five erastin-only inner-fold RMSEs in original units. Same 20 outer splits/inner assignments; no new gene selection or test-informed grid expansion.',
             primary_comparisons=[['hierarchical_slopes', 'matched_intercept'], ['hierarchical_slopes', 'ridge'], ['hierarchical_slopes', 'neural_network']],
             secondary_comparisons=[['positive_slopes', 'matched_intercept'], ['positive_slopes', 'neural_network'],
                                    ['hierarchical_slopes', 'hierarchical_ridge'], ['hierarchical_slopes', 'kernel_ridge'],
                                    ['hierarchical_slopes', 'svr'], ['matched_intercept', 'hierarchical_ridge']],
             primary_metric='pooled_rmse', secondary_metrics=['pooled_spearman', 'pooled_r2', 'macro_all_rmse', 'within_lineage_centered_spearman'],
             uncertainty=dict(count=5000, seed=2026112001, unit='Paired whole-lineage bootstrap preserving all models and all overlapping repeats; conditional exploratory intervals, not independent seed tests.'),
             limitations=['All evaluated lineages are represented on both sides; no unseen-lineage performance claim.',
                          'These repeatedly inspected cohorts are exploratory and require new-data confirmation.',
                          'Tiny lineages have little information for local slopes; isotropic partial pooling does not establish gene-specific causal differences.',
                          'CTRP AUC recipe provenance remains incomplete; outcomes and assays are unchanged and separate.'],
             new_outer_bundles=120, reused_outer_models=280, workers=WORKERS,
             versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__),
             data_sha256={f'{a}/{n}': sha(OUT/a/n) for a in ORIGINAL for n in ['panel_inputs.npz', 'shared_splits.npz']},
             scripts={n: sha(OUT/n) for n in ['models.py', 'run_benchmark.py', 'validate_preflight.py', 'make_splits.py']})
    pp = OUT/'protocol.json'
    if pp.exists():
        old = json.load(open(pp))
        p['created_utc'] = old['created_utc']
        assert clean(p) == old
    else:
        dump(pp, p)
    dump(OUT/'source_manifest.json', dict(files=records))
    return sha(pp)


def fit_repeat(assay, r, ph):
    complete = OUT/assay/'folds'/f'repeat_{r+1:02d}_complete.json'
    if complete.exists():
        result = json.load(open(complete))
        assert result['protocol_sha256'] == ph
        for row in result['files']:
            assert sha(OUT/row['path']) == row['sha256']
        return result
    begin = time.monotonic()
    s = STATE[assay]
    x, y, lin, inner = s['x'], s['y'], s['lineage'], s['inner'][r]
    tr, te = np.flatnonzero(~s['test'][r]), np.flatnonzero(s['test'][r])
    val = np.full((len(GRID), len(y)), np.nan)
    loss = np.empty((len(GRID), 5))
    diagnostics = []
    for k in range(5):
        a, b = inner_rows(inner, k)
        assert set(a)|set(b) == set(tr) and set(s['groups'][a]).isdisjoint(s['groups'][b])
        assert set(lin[a]) == set(lin)
        pp, _, diag = path(x, y, lin, a, b)
        val[:, b] = pp
        loss[:, k] = np.sqrt(np.mean((pp-y[b][None, :])**2, axis=1))
        diagnostics.append(dict(fold=k, **diag))
    assert np.isfinite(val[:, inner >= 0]).all() and np.isnan(val[:, inner < 0]).all()
    selected = {arm: int(eligible(arm)[np.argmin(loss[eligible(arm)].mean(axis=1))]) for arm in ARMS}
    unique = sorted(set(selected.values()))
    pp, fitted, diag = path(x, y, lin, tr, te, [GRID[c] for c in unique], retain=True)
    fitted = {c: (pp[j], fitted[j]) for j, c in enumerate(unique)}
    tuning = OUT/assay/'folds'/f'tuning__r{r+1:02d}'
    np.savez_compressed(tuning.with_suffix('.npz'), validation_predictions=val, inner_fold_rmse=loss)
    dump(tuning.with_suffix('.json'), dict(protocol_sha256=ph, inner_diagnostics=diagnostics))
    files = [tuning.with_suffix('.'+ext) for ext in ['npz', 'json']]
    for arm, c in selected.items():
        pred, model = fitted[c]
        root = root_for(assay, arm, r)
        joblib.dump(model, root.with_suffix('.joblib'), compress=3)
        np.savez_compressed(root.with_suffix('.npz'), train_rows=tr, test_rows=te, test_prediction=pred)
        dump(root.with_suffix('.json'), dict(assay=assay, arm=arm, repeat=r+1, protocol_sha256=ph, selected_candidate=c,
                                             selected_parameters=GRID[c], mean_inner_rmse=float(loss[c].mean()),
                                             train_n=len(tr), test_n=len(te), outer_diagnostics=diag,
                                             effective_degrees_of_freedom=model['effective_degrees_of_freedom'],
                                             npz_sha256=sha(root.with_suffix('.npz')), joblib_sha256=sha(root.with_suffix('.joblib'))))
        files.extend(root.with_suffix('.'+ext) for ext in ['json', 'npz', 'joblib'])
    result = dict(assay=assay, repeat=r+1, protocol_sha256=ph, seconds=time.monotonic()-begin,
                  unique_selected_fits=len(unique), selected_candidates=selected,
                  files=[dict(path=str(p.relative_to(OUT)), sha256=sha(p)) for p in files])
    dump(complete, result)
    return result


def verify_sources():
    for row in json.load(open(OUT/'source_manifest.json'))['files']:
        assert sha(WS/row['path']) == row['sha256']
    p = json.load(open(OUT/'protocol.json'))
    for name, h in {**p['data_sha256'], **p['scripts']}.items():
        assert sha(OUT/name) == h


def main():
    begin = time.monotonic()
    ph = prepare()
    pf = json.load(open(OUT/'preflight_validation.json'))
    assert pf['passed'] and pf['protocol_sha256'] == ph
    with futures.ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context('spawn'), initializer=initialize) as pool:
        jobs = [pool.submit(fit_repeat, a, r, ph) for a in ORIGINAL for r in range(20)]
        for done, future in enumerate(futures.as_completed(jobs), 1):
            m = future.result()
            elapsed = time.monotonic()-begin
            progress = dict(phase='fitting', completed_assay_repeats=done, total_assay_repeats=40,
                            new_model_bundles=3*done, reused_models=280, elapsed_seconds=elapsed,
                            estimated_remaining_seconds=elapsed/done*(40-done), last_assay=m['assay'], last_repeat=m['repeat'])
            dump(OUT/'progress.json', progress)
            print(json.dumps(progress), flush=True)
    verify_sources()
    dump(OUT/'fit_completion.json', dict(complete=True, new_outer_bundles=120, reused_models=280, seconds=time.monotonic()-begin))
    print('ALL_FITS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
