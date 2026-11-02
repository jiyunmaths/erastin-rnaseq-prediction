#!/usr/bin/env python3
"""Inductive, masked four-drug training on frozen CTRP lineage splits."""
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
import pandas as pd
import scipy
import sklearn

from models import LINEAR_GRID, NN_GRID, ARMS, linear_path, neural_fit
from make_splits import inner_rows

OUT = Path(__file__).resolve().parent
WS = OUT.parent.parent
SOURCE = WS/'outputs/erastin_elastic_net_comparison'
KERNEL = WS/'outputs/erastin_kernel_lineage_comparison'
RAW = WS/'work/erastin/raw'
TASKS = [('erastin', 'DPC-002562'), ('RSL3', 'DPC-000040'), ('ML162', 'DPC-004290'), ('ML210', 'DPC-004294')]
BASELINES = ['ridge', 'elastic_net', 'neural_network', 'svr', 'kernel_ridge']
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


def root_for(arm, r):
    return OUT/'folds'/f'{arm}__p64__r{r+1:02d}'


def initialize():
    a = np.load(OUT/'multitask_inputs.npz')
    s = np.load(OUT/'shared_splits.npz')
    STATE.update(x=a['x'], y=a['y_tasks'], ids=a['sample_ids'], groups=a['group_ids'], lineage=a['lineage'],
                 test=s['outer_test'], inner=s['inner_fold'], seeds=s['outer_split_seeds'],
                 inner_seeds=s['inner_split_seeds'], nn_seeds=s['neural_initialization_seeds'])


def prepare():
    (OUT/'folds').mkdir(parents=True, exist_ok=True)
    records = []
    for source, names in [(SOURCE, ['panel_inputs.npz', 'shared_splits.npz', 'protocol.json', 'validation.json', 'summary.tsv']),
                          (KERNEL, ['CTRP/predictions.npz', 'protocol.json', 'validation.json', 'summary.tsv'])]:
        known = {v['path']: v['sha256'] for v in json.load(open(source/'artifact_manifest.json'))['files']}
        for name in names:
            assert sha(source/name) == known[name]
            records.append(dict(path=str((source/name).relative_to(WS)), sha256=known[name]))
        assert json.load(open(source/'validation.json'))['passed']
    for source, names in [(SOURCE, [f'folds/{f}__p64__r{r:02d}.{ext}' for f in BASELINES[:3]
                                   for r in range(1, 21) for ext in ['json', 'npz', 'joblib']]),
                          (KERNEL, [f'CTRP/folds/{f}__p64__r{r:02d}.{ext}' for f in BASELINES[3:]
                                    for r in range(1, 21) for ext in ['json', 'npz', 'joblib']])]:
        known = {v['path']: v['sha256'] for v in json.load(open(source/'artifact_manifest.json'))['files']}
        for name in names:
            assert sha(source/name) == known[name]
            records.append(dict(path=str((source/name).relative_to(WS)), sha256=known[name]))
    for name in ['PortalCompounds.csv', 'CTRPAUCMatrix.csv']:
        sidecar = json.load(open(RAW/(name+'.download.json')))
        assert sha(RAW/name) == sidecar['sha256']
        records.extend(dict(path=str((RAW/n).relative_to(WS)), sha256=sha(RAW/n)) for n in [name, name+'.download.json'])
    a = np.load(SOURCE/'panel_inputs.npz')
    m = pd.read_csv(RAW/'CTRPAUCMatrix.csv', index_col=0).reindex(a['sample_ids'])
    y = -m[[v[1] for v in TASKS]].to_numpy()
    np.testing.assert_array_equal(y[:, 0], a['y'])
    assert np.isfinite(a['x_64']).all() and not np.isinf(y).any()
    compounds = pd.read_csv(RAW/'PortalCompounds.csv').set_index('CompoundID')
    mapping = [dict(task=t, compound_id=c, original_name=compounds.loc[c, 'CompoundName'],
                    original_sample_ids=compounds.loc[c, 'SampleIDs'], observed_lines=int(np.isfinite(y[:, j]).sum()))
               for j, (t, c) in enumerate(TASKS)]
    expected_names = ['ERASTIN', '1S,3R-RSL-3', 'ML162', 'ML210']
    assert [r['original_name'] for r in mapping] == expected_names
    if (OUT/'multitask_inputs.npz').exists():
        old = np.load(OUT/'multitask_inputs.npz')
        for key, value in dict(x=a['x_64'], y_tasks=y, sample_ids=a['sample_ids'], group_ids=a['group_ids'], lineage=a['lineage'], genes=a['genes_64']).items():
            np.testing.assert_array_equal(old[key], value)
    else:
        np.savez_compressed(OUT/'multitask_inputs.npz', x=a['x_64'], y_tasks=y,
                            sample_ids=a['sample_ids'], group_ids=a['group_ids'], lineage=a['lineage'], genes=a['genes_64'])
    if (OUT/'shared_splits.npz').exists():
        assert sha(OUT/'shared_splits.npz') == sha(SOURCE/'shared_splits.npz')
    else:
        shutil.copyfile(SOURCE/'shared_splits.npz', OUT/'shared_splits.npz')
    dump(OUT/'task_mapping.json', mapping)
    old = json.load(open(SOURCE/'protocol.json'))
    p = dict(status='frozen_before_multitask_fits', created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             assay='CTRP', samples=744, identity_groups=733, lineages=23, gene_count=64, tasks=mapping,
             targets='Negative original harmonized CTRP AUC for every drug; higher means more sensitive. Erastin values exactly match the retained benchmark.',
             training_cohort='Same 744 erastin-labelled lines only; no additional auxiliary-only cells.',
             missing_values='Mask missing task labels, without response imputation. Each task center/scale uses observed current-training responses only.',
             split_design=old['split_design'], inner_design=old['inner_design'], ampulla_handling=old['ampulla_handling'],
             exclusion='All outer-test identity groups excluded from every task. All inner-validation identity groups excluded from every task in their inner fit. No auxiliary measurements are inputs at prediction time.',
             weighting='Unweighted cell contributions within each task; per-task losses normalize by observed label count. Neural task-loss weights are separately tuned, not lineage weights.',
             culture_covariates=False, lineage_covariates=False, gene_selection=False, pooled_assays=False,
             linear_grid=LINEAR_GRID, neural_grid=NN_GRID,
             linear_objective='0.5*sum_t[(n/n_t)*||Z_t beta_t + c_t - y_t||^2] + 0.5*alpha*sum_t||beta_t||^2 + 0.5*alpha*coupling_ratio*sum_t||beta_t-mean_s(beta_s)||^2. Intercepts unpenalized. Standardized X/y; n=current training line count.',
             neural_objective='0.5*sum_t[w_t/n_t*sum_observed(error_t^2)] + 0.5*alpha/n*(||W_hidden||^2+||W_output||^2). w=[1,aux/3,aux/3,aux/3]/(1+aux). One tanh hidden layer; separate linear output heads. Biases unpenalized.',
             neural_control='Auxiliary weight zero fits a one-head network with the same hidden/primary initialization, objective and optimizer. Primary and shared initialization identical across task-weight choices per seed.',
             neural_optimizer=dict(method='float64 analytic-gradient L-BFGS-B', max_iter=3000, max_fun=50000, gtol=1e-5, ftol=1e-12, maxls=40,
                                   ensemble='Average three independent initializations using original saved seed triplets.'),
             arms=dict(multitask_linear='All 68 coupled-linear candidates including zero coupling.',
                       matched_single_nn='Four auxiliary-weight-zero neural candidates.',
                       multitask_nn='All 12 neural candidates including zero-auxiliary fallback.',
                       positive_aux_nn='Eight strictly positive auxiliary-weight candidates; prespecified sensitivity analysis.'),
             tuning='Select by arithmetic mean of five erastin-only inner-fold RMSEs in original units; first fixed candidate wins ties. Auxiliary validation responses never tune settings.',
             primary_comparisons=[['multitask_linear', 'ridge'], ['multitask_nn', 'matched_single_nn'], ['multitask_nn', 'neural_network']],
             secondary_comparisons=[['positive_aux_nn', 'matched_single_nn'], ['positive_aux_nn', 'neural_network'],
                                    ['multitask_linear', 'neural_network'], ['matched_single_nn', 'neural_network'],
                                    ['multitask_nn', 'kernel_ridge'], ['multitask_nn', 'svr']],
             primary_metric='pooled_rmse', secondary_metrics=['pooled_spearman', 'pooled_r2', 'macro_all_rmse'],
             uncertainty=dict(count=5000, seed=2026111901, unit='Paired bootstrap of entire lineage blocks, preserving all repeats and models; conditional exploratory intervals, not independent repeat tests.'),
             limitations=['This repeatedly inspected cohort is exploratory and requires new-data confirmation.',
                          'Same-assay CTRP only. Downloaded PRISM secondary treatments contain none of the three auxiliary drugs.',
                          'GPX4-inhibitor vulnerabilities can differ from erastin; responses remain distinct tasks.',
                          'Original harmonized CTRP AUC recipe remains unverified; all existing target values remain unchanged.',
                          'All evaluated lineages appear in training and testing; no unseen-lineage claim.',
                          'Observed cross-drug correlations motivated task choice; they are not independent validation evidence.'],
             references=['https://academic.oup.com/bioinformatics/article/30/17/i556/200902', 'https://pubmed.ncbi.nlm.nih.gov/38552003/'],
             new_outer_model_bundles=80, reused_outer_models=100, workers=WORKERS,
             versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, sklearn=sklearn.__version__, joblib=joblib.__version__),
             data_sha256={n: sha(OUT/n) for n in ['multitask_inputs.npz', 'shared_splits.npz', 'task_mapping.json']},
             scripts={n: sha(OUT/n) for n in ['models.py', 'run_benchmark.py', 'validate_preflight.py', 'make_splits.py']})
    if (OUT/'protocol.json').exists():
        prior = json.load(open(OUT/'protocol.json'))
        p['created_utc'] = prior['created_utc']
        assert clean(p) == prior
    else:
        dump(OUT/'protocol.json', p)
    dump(OUT/'source_manifest.json', dict(files=records))
    return sha(OUT/'protocol.json')


def fit_repeat(r, ph):
    completed = OUT/'folds'/f'repeat_{r+1:02d}_complete.json'
    if completed.exists():
        m = json.load(open(completed))
        assert m['protocol_sha256'] == ph
        for f in m['files']:
            assert sha(OUT/f['path']) == f['sha256']
        return m
    begin = time.monotonic()
    s = STATE
    x, y, inner = s['x'], s['y'], s['inner'][r]
    tr, te = np.flatnonzero(~s['test'][r]), np.flatnonzero(s['test'][r])
    lv, nv = np.full((len(LINEAR_GRID), len(y)), np.nan), np.full((len(NN_GRID), len(y)), np.nan)
    ll, nl = np.empty((len(LINEAR_GRID), 5)), np.empty((len(NN_GRID), 5))
    diagnostics = []
    for k in range(5):
        a, b = inner_rows(inner, k)
        assert set(a)|set(b) == set(tr) and set(s['groups'][a]).isdisjoint(s['groups'][b])
        pp, _, ld = linear_path(x, y, a, b, LINEAR_GRID)
        lv[:, b] = pp[:, :, 0]
        ll[:, k] = np.sqrt(np.mean((pp[:, :, 0]-y[b, 0][None, :])**2, axis=1))
        for c, params in enumerate(NN_GRID):
            pp, _, diag = neural_fit(x, y, a, b, params, s['nn_seeds'][r])
            nv[c, b] = pp[:, 0]
            nl[c, k] = np.sqrt(np.mean((pp[:, 0]-y[b, 0])**2))
            diagnostics.append(dict(stage='inner', fold=k, candidate=c, members=diag))
    for values in [lv, nv]:
        assert np.isfinite(values[:, inner >= 0]).all() and np.isnan(values[:, inner < 0]).all()
    selections = {'multitask_linear': int(np.argmin(ll.mean(axis=1))),
                  'matched_single_nn': int(np.argmin(nl[:4].mean(axis=1))),
                  'multitask_nn': int(np.argmin(nl.mean(axis=1))),
                  'positive_aux_nn': int(np.argmin(nl[4:].mean(axis=1)))+4}
    unique_nn = {}
    for c in sorted(set(v for k, v in selections.items() if k != 'multitask_linear')):
        unique_nn[c] = neural_fit(x, y, tr, te, NN_GRID[c], s['nn_seeds'][r])
    c = selections['multitask_linear']
    lp, lm, ld = linear_path(x, y, tr, te, [LINEAR_GRID[c]])
    files = []
    tuning_root = OUT/'folds'/f'tuning__r{r+1:02d}'
    np.savez_compressed(tuning_root.with_suffix('.npz'), linear_validation_predictions=lv, neural_validation_predictions=nv,
                        linear_fold_rmse=ll, neural_fold_rmse=nl)
    dump(tuning_root.with_suffix('.json'), dict(repeat=r+1, protocol_sha256=ph, neural_diagnostics=diagnostics))
    files += [tuning_root.with_suffix('.'+ext) for ext in ['npz', 'json']]
    for arm, c in selections.items():
        if arm == 'multitask_linear':
            pp, model, diag, grid, loss = lp[0], lm[0], ld[0], LINEAR_GRID, ll
        else:
            pp, model, diag = unique_nn[c]
            grid, loss = NN_GRID, nl
        root = root_for(arm, r)
        joblib.dump(model, root.with_suffix('.joblib'), compress=3)
        np.savez_compressed(root.with_suffix('.npz'), train_rows=tr, test_rows=te, test_prediction=pp[:, 0], all_task_predictions=pp)
        dump(root.with_suffix('.json'), dict(arm=arm, repeat=r+1, panel=64, protocol_sha256=ph,
                                             selected_candidate=c, selected_parameters=grid[c], mean_inner_rmse=float(loss[c].mean()),
                                             train_n=len(tr), test_n=len(te), train_task_counts=np.isfinite(y[tr]).sum(axis=0),
                                             test_task_counts=np.isfinite(y[te]).sum(axis=0), neural_seeds=s['nn_seeds'][r], outer_diagnostics=diag,
                                             npz_sha256=sha(root.with_suffix('.npz')), joblib_sha256=sha(root.with_suffix('.joblib'))))
        files += [root.with_suffix('.'+ext) for ext in ['json', 'npz', 'joblib']]
    m = dict(repeat=r+1, protocol_sha256=ph, seconds=time.monotonic()-begin, selected_candidates=selections,
             unique_outer_neural_ensembles=len(unique_nn), files=[dict(path=str(p.relative_to(OUT)), sha256=sha(p)) for p in files])
    dump(completed, m)
    return m


def verify_sources():
    for row in json.load(open(OUT/'source_manifest.json'))['files']:
        assert sha(WS/row['path']) == row['sha256']
    p = json.load(open(OUT/'protocol.json'))
    for name, h in {**p['data_sha256'], **p['scripts']}.items():
        assert sha(OUT/name) == h


def main():
    begin = time.monotonic()
    ph = prepare()
    initialize()
    pf = json.load(open(OUT/'preflight_validation.json'))
    assert pf['passed'] and pf['protocol_sha256'] == ph
    with futures.ProcessPoolExecutor(max_workers=WORKERS, mp_context=mp.get_context('spawn'), initializer=initialize) as pool:
        jobs = [pool.submit(fit_repeat, r, ph) for r in range(20)]
        for done, future in enumerate(futures.as_completed(jobs), 1):
            m = future.result()
            elapsed = time.monotonic()-begin
            progress = dict(phase='fitting', completed_repeats=done, total_repeats=20, completed_model_bundles=4*done,
                            total_new_model_bundles=80, reused_models=100, elapsed_seconds=elapsed,
                            estimated_remaining_seconds=elapsed/done*(20-done), last_completed_repeat=m['repeat'])
            dump(OUT/'progress.json', progress)
            print(json.dumps(progress), flush=True)
    verify_sources()
    dump(OUT/'fit_completion.json', dict(complete=True, new_model_bundles=80, reused_models=100, seconds=time.monotonic()-begin))
    print('ALL_FITS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
