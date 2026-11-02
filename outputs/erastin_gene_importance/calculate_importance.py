#!/usr/bin/env python3
"""Exact within-held-out-lineage permutation importance of saved 20-repeat models.

For an additive squared-error loss, averaging every receiver/donor substitution
is exactly the expectation over all random within-lineage permutations. No model
is refitted, and no attribution Monte Carlo approximation is required.
"""
import concurrent.futures as futures
import gzip
import hashlib
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import sys
import time

import joblib
import numpy as np
from sklearn.preprocessing import StandardScaler

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
WS = OUT.parent.parent
STATE = {}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def initialize():
    with np.load(OUT / 'analysis_inputs.npz', allow_pickle=False) as d:
        STATE.update({k: d[k] for k in d.files})


def native_predict(saved, x):
    z = saved['x_scaler'].transform(x)
    return np.mean([saved['y_scaler'].inverse_transform(m.predict(z).reshape(-1, 1)).ravel()
                    for m in saved['members']], axis=0)


def one_holdout(repeat, lineage_index, protocol_hash):
    r, l = repeat, lineage_index
    token = f'repeat_{r + 1:02d}__lineage_{l:02d}'
    npth = OUT / 'folds' / (token + '.npz')
    jpth = OUT / 'folds' / (token + '.json')
    if jpth.exists():
        meta = json.loads(jpth.read_text())
        assert meta['protocol_sha256'] == protocol_hash and meta['array_sha256'] == sha(npth)
        return meta
    label = str(STATE['labels'][l])
    te = np.flatnonzero(STATE['lineage'] == label)
    tr = np.flatnonzero(STATE['lineage'] != label)
    assert set(STATE['groups'][te]).isdisjoint(STATE['groups'][tr])
    x, y = STATE['x'][te], STATE['y'][te]
    n, p = x.shape
    saved = joblib.load(WS / 'outputs/erastin_nn_repeated/folds' / (token + '.joblib'))
    assert saved['held_out_lineage'] == label and saved['repeat'] == r + 1
    assert set(saved['test_rows']) == set(te) and set(saved['train_rows']) == set(tr)
    sx = StandardScaler().fit(STATE['x'][tr])
    sy = StandardScaler().fit(STATE['y'][tr, None])
    for expected, observed in [(sx.mean_, saved['x_scaler'].mean_), (sx.scale_, saved['x_scaler'].scale_),
                               (sy.mean_, saved['y_scaler'].mean_), (sy.scale_, saved['y_scaler'].scale_),
                               (sx.mean_, STATE['ridge_mean'][r, l]), (sx.scale_, STATE['ridge_scale'][r, l])]:
        np.testing.assert_allclose(observed, expected, rtol=0, atol=1e-12)
    z = sx.transform(x)
    coef = STATE['ridge_coef'][r, l]
    pred_r = z @ coef + STATE['ridge_intercept'][r, l]
    pred_n = native_predict(saved, x)
    baseline_prediction = np.stack([pred_r, pred_n])
    prediction_error = float(np.max(np.abs(baseline_prediction - STATE['saved_predictions'][r][:, te])))
    assert prediction_error < 1e-11
    base_loss = (baseline_prediction - y) ** 2
    gradient = np.zeros((2, n, p))
    gradient[0] = coef
    members = []
    for network in saved['members']:
        assert network.activation == 'tanh' and len(network.coefs_) == 2 and network.out_activation_ == 'identity'
        w1, w2 = network.coefs_[0], network.coefs_[1].ravel()
        h = z @ w1 + network.intercepts_[0]
        members.append((h, w1, w2, float(network.intercepts_[1][0])))
        gradient[1] += (((1 - np.tanh(h) ** 2) * w2) @ w1.T) * sy.scale_[0] / len(saved['members'])
    per_cell_delta = np.empty((2, n, p))
    swap_error, gradient_error, exact_identity_error, permutation_error = 0., 0., 0., 0.
    # For every holdout, independently verify all 40 replacement predictions
    # for one receiver/donor pair with the sklearn prediction implementation.
    receiver, donor = 0, n - 1
    direct = np.repeat(x[[receiver]], p, axis=0)
    direct[np.arange(p), np.arange(p)] = x[donor]
    direct_nn = native_predict(saved, direct)
    # Independent finite differences for all genes and one observed cell.
    eps = 1e-5
    xp = np.repeat(x[[receiver]], p, axis=0)
    xm = xp.copy()
    xp[np.arange(p), np.arange(p)] += eps * sx.scale_
    xm[np.arange(p), np.arange(p)] -= eps * sx.scale_
    finite_difference = (native_predict(saved, xp) - native_predict(saved, xm)) / (2 * eps)
    gradient_error = float(np.max(np.abs(finite_difference - gradient[1, receiver])))
    assert gradient_error < 1e-8
    for j in range(p):
        difference = z[None, :, j] - z[:, None, j]
        shuffled = np.empty((2, n, n))
        shuffled[0] = pred_r[:, None] + difference * coef[j]
        shuffled[1] = 0.
        for h, w1, w2, bias in members:
            replacement = np.tanh(h[:, None, :] + difference[:, :, None] * w1[j]) @ w2 + bias
            shuffled[1] += (replacement * sy.scale_[0] + sy.mean_[0]) / len(members)
        error = float(np.max(np.abs(np.diagonal(shuffled, axis1=1, axis2=2) - baseline_prediction)))
        exact_identity_error = max(exact_identity_error, error)
        swap_error = max(swap_error, abs(float(shuffled[1, receiver, donor] - direct_nn[j])))
        loss = (shuffled - y[None, :, None]) ** 2
        per_cell_delta[:, :, j] = loss.mean(axis=2) - base_loss
        # Enumerate all permutations of the two-cell lineage to verify the
        # exact expectation, with complete replacement and native prediction.
        if n == 2 and r == 0:
            mse = []
            for perm in itertools.permutations(range(n)):
                xx = x.copy(); xx[:, j] = x[np.array(perm), j]
                pp = np.stack([sx.transform(xx) @ coef + STATE['ridge_intercept'][r, l], native_predict(saved, xx)])
                mse.append(np.mean((pp - y) ** 2, axis=1))
            permutation_error = max(permutation_error, float(np.max(np.abs(np.mean(mse, axis=0) - loss.mean(axis=(1, 2))))))
    assert max(swap_error, exact_identity_error, permutation_error) < 1e-11
    if n == 1:
        assert np.max(np.abs(per_cell_delta)) < 1e-12
    np.savez_compressed(npth, test_rows=te, per_cell_delta_mse=per_cell_delta,
                        standardized_gradient=gradient, baseline_prediction=baseline_prediction,
                        baseline_mse=base_loss.mean(axis=1), importance_mse=per_cell_delta.mean(axis=1))
    meta = {'repeat': r + 1, 'lineage_index': l, 'lineage': label, 'test_n': n,
            'n_distinct_identities': int(len(np.unique(STATE['groups'][te]))),
            'receiver_donor_pairs_per_gene': n * n, 'protocol_sha256': protocol_hash,
            'baseline_prediction_max_error': prediction_error, 'native_replacement_max_error': swap_error,
            'identity_replacement_max_error': exact_identity_error, 'gradient_finite_difference_max_error': gradient_error,
            'exhaustive_two_cell_permutation_max_error': permutation_error,
            'held_out_lineage_and_identities_excluded': True, 'training_scalers_verified': True,
            'array_sha256': sha(npth)}
    dump(jpth, meta)
    return meta


def main():
    start = time.time()
    (OUT / 'folds').mkdir(parents=True, exist_ok=True)
    sources = []
    def verify(directory, names):
        root = WS / 'outputs' / directory
        manifest = root / 'artifact_manifest.json'
        hashes = {v['path']: v['sha256'] for v in json.loads(manifest.read_text())['files']}
        for name in names:
            path = root / name
            h = sha(path); assert h == hashes[name], path
            sources.append({'path': str(path.relative_to(WS)), 'sha256': h})
    verify('erastin_40gene_comparison', ['comparison_inputs.npz'])
    verify('erastin_lineage_corrected', ['fitted_outer_models.json.gz', 'protocol.json'])
    nn_root = WS / 'outputs/erastin_nn_repeated'
    verify('erastin_nn_repeated', ['shared_splits.npz', 'predictions.npz', 'protocol.json'] +
           [str(p.relative_to(nn_root)) for p in sorted((nn_root / 'folds').glob('*.joblib'))])
    d = np.load(WS / 'outputs/erastin_40gene_comparison/comparison_inputs.npz', allow_pickle=False)
    s = np.load(nn_root / 'shared_splits.npz', allow_pickle=False)
    pr = np.load(nn_root / 'predictions.npz', allow_pickle=False)
    assert np.array_equal(d['sample_ids'], s['sample_ids']) and np.array_equal(d['sample_ids'], pr['sample_ids'])
    labels = s['holdout_labels']; assert len(labels) == 24
    assert len(np.unique(d['panel_symbols'])) == 40
    ridge = json.load(gzip.open(WS / 'outputs/erastin_lineage_corrected/fitted_outer_models.json.gz', 'rt'))
    lookup = {(v['repeat'], v['fold']): v for v in ridge if v['model'] == 'panel_expression' and v['repeat'] > 0}
    assert len(lookup) == 480
    arrays = {k: np.empty((20, 24, 40)) for k in ['mean', 'scale', 'coef']}
    intercept = np.empty((20, 24))
    for r in range(20):
        for l, label in enumerate(labels):
            saved = lookup[r + 1, label]
            assert saved['seed'] == s['inner_split_seeds'][r]
            assert set(saved['test_rows']) == set(np.flatnonzero(d['lineage'] == label))
            for k in arrays: arrays[k][r, l] = saved['fitted_model'][k]
            intercept[r, l] = saved['fitted_model']['intercept_after_scaling']
    protocol = {
        'status': 'fixed_before_gene_importance_results', 'created_utc': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        'script_sha256': sha(Path(__file__)), 'models': ['ridge', 'neural_network'], 'repeats': 20,
        'data': 'same 745 lines, 734 identities, 40 log2(TPM+1) genes, and 24 outer lineage holdouts',
        'saved_models': 'reuse the paired follow-up models from every repeat; no refitting or feature selection',
        'primary_importance': 'expected held-out MSE increase when replacing one gene by a uniformly selected donor from the SAME held-out lineage',
        'exact_expectation': 'all n*n receiver/donor pairs per gene; includes identity replacement; exactly equals expected random-bijection permutation MSE by linearity of additive loss',
        'ensemble': 'average three NN member predictions BEFORE computing squared error',
        'primary_scope': 'sample-weighted mean over the same 19 lineages with n>=10, 721 lines',
        'secondary_scope': 'equal-weight mean of the same 19 lineages; descriptive sample-weighted mean across all 23 non-singleton lineages (744 lines)',
        'singleton': 'Cervix n=1 has no within-lineage permutation information and is excluded from summary rankings',
        'direction': 'training-SD-standardized ridge coefficient; NN analytical local derivative at each held-out observed sample averaged over three members, in negative-AUC units per training SD',
        'lineage_consistency': 'mean over 20 repeats per lineage; positive importance counts, top-ten ranks, direction signs, and between-lineage rank correlations',
        'seed_consistency': 'per-repeat aggregate rank range and top-ten frequency; not independent biological replications',
        'uncertainty': {'draws': 5000, 'seed': 2026110801, 'unit': 'whole held-out lineage among the 19 eligible lineages',
                        'paired': 'same resamples for all genes, models and repeats; average repeat effects before resampling',
                        'scope': 'conditional on fitted models and observed donor distributions, no refitting; pointwise exploratory 95% intervals, no multiplicity correction'},
        'correlation': 'descriptive pairwise correlation after within-lineage centering in 721 eligible lines; no label-based gene regrouping',
        'limits': ['Unconditional gene shuffling disrupts correlated features and may create atypical expression combinations.',
                   'Small importance can reflect redundancy, limited within-lineage variation, regularization or low power, not biological irrelevance.',
                   'Positive/negative prediction slopes are associations, not causal responses to gene perturbation.',
                   'No new external cohort; related cell lines remain together in lineage-level uncertainty.'],
        'workers': 6}
    path = OUT / 'protocol.json'
    if path.exists():
        old = json.loads(path.read_text()); protocol['created_utc'] = old['created_utc']; assert old == protocol
    else: dump(path, protocol)
    ph = sha(path)
    dump(OUT / 'source_manifest.json', {'files': sources})
    np.savez_compressed(OUT / 'analysis_inputs.npz', x=d['X_panel_log2tpm1'].astype(float), y=d['y_sensitivity'].astype(float),
                        sample_ids=d['sample_ids'], groups=d['group_ids'], lineage=d['lineage'], genes=d['panel_symbols'], labels=labels,
                        inner_seeds=s['inner_split_seeds'], neural_seeds=s['neural_initialization_seeds'],
                        saved_predictions=np.stack([pr['ridge'], pr['neural_network']], axis=1),
                        ridge_intercept=intercept, **{'ridge_' + k: v for k, v in arrays.items()})
    print(json.dumps({'event': 'protocol_frozen', 'sha256': ph}), flush=True)
    rows = []
    with futures.ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context('spawn'), initializer=initialize) as pool:
        jobs = [pool.submit(one_holdout, r, l, ph) for r in range(20) for l in range(24)]
        for future in futures.as_completed(jobs):
            rows.append(future.result())
            if len(rows) % 24 == 0:
                progress = {'phase': 'calculating', 'completed_holdouts': len(rows), 'total_holdouts': 480, 'elapsed_seconds': time.time() - start}
                dump(OUT / 'progress.json', progress); print(json.dumps(progress), flush=True)
    for v in sources: assert sha(WS / v['path']) == v['sha256']
    assert sha(path) == ph and sha(Path(__file__)) == protocol['script_sha256']
    validation = {'passed': True, 'holdouts': len(rows), 'saved_models': 960, 'genes': 40, 'no_refitting': True,
                  'source_hashes_verified_before_and_after': True, 'checks': sorted(rows, key=lambda v: (v['repeat'], v['lineage_index'])),
                  'elapsed_seconds': time.time() - start}
    dump(OUT / 'calculation_validation.json', validation)
    dump(OUT / 'progress.json', {'phase': 'calculation_complete', 'completed_holdouts': 480, 'total_holdouts': 480,
                                'elapsed_seconds': time.time() - start})
    print(json.dumps({'event': 'calculation_complete', 'elapsed_seconds': time.time() - start}), flush=True)


if __name__ == '__main__':
    main()
