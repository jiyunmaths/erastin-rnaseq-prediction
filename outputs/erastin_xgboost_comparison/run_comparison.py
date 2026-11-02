#!/usr/bin/env python3
"""XGBoost nested CV on the frozen 40-gene panel and lineage holdouts."""
import concurrent.futures as futures
import hashlib
import itertools
import json
import multiprocessing as mp
from pathlib import Path
import platform
import shutil
import sys
import time
import warnings

import numpy as np
import scipy
import xgboost as xgb

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
WS = OUT.parent.parent
STRUCTURES = [dict(max_depth=d, min_child_weight=w, reg_lambda=l)
              for d, w, l in itertools.product([1, 2, 3], [5, 20], [1., 20.])]
ROUNDS = [100, 300, 600]
GRID = [dict(**p, num_boost_round=n) for p in STRUCTURES for n in ROUNDS]
FIXED = dict(objective='reg:squarederror', booster='gbtree', tree_method='hist',
             device='cpu', nthread=1, eta=.03, subsample=.8, colsample_bytree=.8,
             sampling_method='uniform', max_bin=64, reg_alpha=0., gamma=0.,
             base_score=0., seed_per_iteration=False, validate_parameters=True,
             verbosity=1)
STATE = {}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''): h.update(b)
    return h.hexdigest()


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, np.integer): return int(x)
    if isinstance(x, (float, np.floating)): return float(x) if np.isfinite(x) else None
    if isinstance(x, np.bool_): return bool(x)
    return x


def dump(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(clean(value), indent=2, allow_nan=False) + '\n')
    tmp.replace(path)


def initialize():
    d = np.load(WS/'outputs/erastin_40gene_comparison/comparison_inputs.npz')
    s = np.load(WS/'outputs/erastin_nn_repeated/shared_splits.npz')
    for key in ['sample_ids', 'group_ids', 'lineage']:
        np.testing.assert_array_equal(d[key], s[key])
    STATE.update(x=d['X_panel_log2tpm1'].astype(float), y=d['y_sensitivity'].astype(float),
                 ids=d['sample_ids'], groups=d['group_ids'], lineage=d['lineage'],
                 genes=d['panel_symbols'], labels=s['holdout_labels'],
                 inner=s['inner_fold_by_holdout'], seeds=s['inner_split_seeds'])
    STATE['model_seeds'] = np.array([int(v) % (2**31-1) for v in STATE['seeds']], dtype=np.int64)
    assert STATE['x'].shape == (745, 40) and np.isfinite(STATE['x']).all()
    assert STATE['inner'].shape == (20, 24, 745) and np.isfinite(STATE['y']).all()
    assert len(np.unique(STATE['model_seeds'])) == 20


def matrix(x, y=None):
    return xgb.DMatrix(x, label=y, feature_names=list(STATE['genes']), nthread=1)


def parameters(candidate, seed):
    return dict(FIXED, **{k: v for k, v in candidate.items() if k != 'num_boost_round'}, seed=int(seed))


def holdout(r, l, protocol_hash):
    token = f'repeat_{r+1:02d}__lineage_{l:02d}'
    paths = {ext: OUT/'folds'/(token+'.'+ext) for ext in ['json', 'npz', 'ubj']}
    if paths['json'].exists():
        meta = json.loads(paths['json'].read_text())
        assert meta['protocol_sha256'] == protocol_hash
        assert sha(paths['npz']) == meta['array_sha256'] and sha(paths['ubj']) == meta['model_sha256']
        return meta
    start = time.monotonic()
    x, y, g = STATE['x'], STATE['y'], STATE['groups']
    inner = STATE['inner'][r, l]; label = STATE['labels'][l]
    tr = np.flatnonzero(inner >= 0); te = np.flatnonzero(STATE['lineage'] == label)
    seed = int(STATE['model_seeds'][r])
    assert label not in set(STATE['lineage'][tr]) and set(g[tr]).isdisjoint(g[te])
    val = np.full((len(GRID), len(y)), np.nan)
    losses = np.empty((len(GRID), 5)); scaling = []; warning_records = []
    # Fit each 600-tree path once. Prefixes are identical to standalone fits
    # because all parameters and RNG seeds are fixed; independently audited.
    for k in range(5):
        a = tr[inner[tr] != k]; b = tr[inner[tr] == k]
        assert set(a) | set(b) == set(tr) and set(g[a]).isdisjoint(g[b])
        mu, sd = float(y[a].mean()), float(y[a].std())
        assert sd > 0
        scaling.append(dict(inner_fold=k, train_n=len(a), validation_n=len(b), y_mean=mu, y_scale=sd))
        da = matrix(x[a], (y[a]-mu)/sd); db = matrix(x[b])
        for j, structure in enumerate(STRUCTURES):
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model = xgb.train(parameters(structure, seed), da, num_boost_round=max(ROUNDS))
            warning_records.extend(dict(stage='inner', inner_fold=k, structure=j,
                                        category=w.category.__name__, message=str(w.message)) for w in caught)
            for q, n in enumerate(ROUNDS):
                c = j*len(ROUNDS)+q
                p = model.predict(db, iteration_range=(0, n)).astype(float)*sd+mu
                val[c, b] = p
                losses[c, k] = np.sqrt(np.mean((y[b]-p)**2))
    assert np.isfinite(val[:, tr]).all() and np.isnan(val[:, inner < 0]).all()
    chosen = int(np.argmin(losses.mean(axis=1))); candidate = GRID[chosen]
    mu, sd = float(y[tr].mean()), float(y[tr].std())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model = xgb.train(parameters(candidate, seed), matrix(x[tr], (y[tr]-mu)/sd),
                          num_boost_round=candidate['num_boost_round'])
    warning_records.extend(dict(stage='outer', category=w.category.__name__, message=str(w.message)) for w in caught)
    p = model.predict(matrix(x[te])).astype(float)*sd+mu
    assert np.isfinite(p).all()
    model.save_model(paths['ubj'])
    np.savez_compressed(paths['npz'], train_rows=tr, test_rows=te, inner_validation_predictions=val,
                        inner_fold_rmse=losses, test_prediction=p)
    meta = dict(repeat=r+1, lineage_index=l, lineage=str(label), inner_split_seed=int(STATE['seeds'][r]),
                model_seed=seed, train_n=len(tr), test_n=len(te), selected_candidate=chosen,
                selected_parameters=candidate, mean_inner_rmse=float(losses[chosen].mean()),
                y_mean=mu, y_scale=sd, inner_scaling=scaling, warnings=warning_records,
                protocol_sha256=protocol_hash, array_sha256=sha(paths['npz']), model_sha256=sha(paths['ubj']),
                seconds=time.monotonic()-start)
    dump(paths['json'], meta)
    return meta


def main():
    start = time.monotonic(); initialize(); (OUT/'folds').mkdir(exist_ok=True)
    sources = []
    for directory, names in [
        ('erastin_40gene_comparison', ['comparison_inputs.npz']),
        ('erastin_nn_repeated', ['shared_splits.npz', 'predictions.npz', 'per_repeat_metrics.tsv', 'protocol.json']),
        ('erastin_bayesian_comparison', ['review_bayesian.py']),
        ('erastin_repeated_cv', ['repeat_cv.py']),
        ('erastin_nonlinear_comparison', ['run_models.py'])]:
        root = WS/'outputs'/directory
        known = {v['path']: v['sha256'] for v in json.loads((root/'artifact_manifest.json').read_text())['files']}
        for name in names:
            assert sha(root/name) == known[name]
            sources.append(dict(path=str((root/name).relative_to(WS)), sha256=known[name]))
    wheel = next((WS/'work/erastin_xgboost/wheels').glob('xgboost_cpu-*.whl'))
    protocol = dict(status='frozen_before_xgboost_fitting', created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        script_sha256=sha(Path(__file__)), samples=745, identity_groups=734, outer_lineages=24, repeats=20,
        input='Same 40-gene log2(TPM+1) panel; target is negative reported CTRP erastin AUC.',
        outer_tests='Exact fixed leave-lineage-out tests from the repeated ridge/NN comparison; related identities excluded.',
        inner_folds='Exact saved five group-disjoint inner folds; represented training lineages are mixed within inner folds.',
        inner_split_seeds=STATE['seeds'], model_seeds=STATE['model_seeds'],
        seed_rule='model_seed = int(saved_inner_split_seed) % (2**31 - 1); same seed for every candidate and fit within a repeat',
        repeat_variation='Inner assignments and XGBoost row/column sampling seeds change jointly; outer tests fixed.',
        fixed_parameters=FIXED, grid=GRID, tuning='Minimum equal-weight mean inner-fold RMSE; tie uses first candidate.',
        prefixes='Train 600 trees per structural setting and score fixed 100/300/600 prefixes; selected model refit at chosen length.',
        early_stopping=False, preprocessing='Features unchanged; response standardized with population mean/SD fit on each current training subset; inverse transform predictions.',
        ensemble='One XGBoost model per outer holdout; saved NN uses its unchanged three-member ensemble.',
        baseline='Reuse exact matching saved ridge and NN predictions, with no averaging across repeats before scoring.',
        tuning_budget='36 XGBoost candidates versus 4 NN candidates; same folds and objective, not equal search budgets.',
        workers=6, inner_training_paths=28800, outer_refits=480,
        uncertainty=dict(count=5000, pooled_seed=2026110101, macro_seed=2026110102,
                         unit='Whole lineage; paired models/repeats; conditional on predictions without refitting.'),
        metrics=['spearman', 'rmse', 'r2'], macro='Equal-weight scores over the same 19 lineages with n>=10.',
        restrictions='No test-score-based grid changes, seed selection, or retry. Prior comparison outcomes already seen; exploratory cohort reuse.',
        versions=dict(python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__, xgboost=xgb.__version__),
        xgboost_build=xgb.build_info(), wheel=dict(name=wheel.name, sha256=sha(wheel)),
        documentation=['https://xgboost.readthedocs.io/en/stable/parameter.html',
                       'https://xgboost.readthedocs.io/en/stable/python/python_api.html'])
    pp = OUT/'protocol.json'
    if pp.exists():
        old = json.loads(pp.read_text()); protocol['created_utc'] = old['created_utc']; assert old == clean(protocol)
    else: dump(pp, protocol)
    ph = sha(pp); dump(OUT/'source_manifest.json', dict(files=sources))
    shutil.copyfile(WS/'outputs/erastin_nn_repeated/shared_splits.npz', OUT/'shared_splits.npz')
    # Small standalone input archive makes the delivered fitted models usable.
    np.savez_compressed(OUT/'panel_inputs.npz', x=STATE['x'], y=STATE['y'], sample_ids=STATE['ids'],
                        group_ids=STATE['groups'], lineage=STATE['lineage'], genes=STATE['genes'])
    print(json.dumps(dict(event='protocol_frozen', protocol_sha256=ph, candidates=len(GRID), outer_fits=480)), flush=True)
    completed = 0; counts = np.zeros(20, dtype=int)
    with futures.ProcessPoolExecutor(max_workers=6, mp_context=mp.get_context('spawn'), initializer=initialize) as pool:
        jobs = {pool.submit(holdout, r, l, ph): (r, l) for r in range(20) for l in range(24)}
        for future in futures.as_completed(jobs):
            result = future.result(); completed += 1; counts[result['repeat']-1] += 1
            elapsed = time.monotonic()-start
            progress = dict(phase='fitting', completed_outer_jobs=completed, total_outer_jobs=480,
                            complete_repeats=int(np.sum(counts==24)), elapsed_seconds=elapsed,
                            estimated_remaining_seconds=elapsed/completed*(480-completed))
            dump(OUT/'progress.json', progress)
            if completed % 12 == 0 or completed == 480:
                print(json.dumps(dict(event='fit_progress', **progress)), flush=True)
    assert sha(pp) == ph and sha(Path(__file__)) == protocol['script_sha256']
    for source in sources: assert sha(WS/source['path']) == source['sha256']
    dump(OUT/'fit_completion.json', dict(status='complete', outer_jobs=480, inner_paths=28800,
         elapsed_seconds=time.monotonic()-start, protocol_sha256=ph, sources_and_script_unchanged=True))
    dump(OUT/'progress.json', dict(phase='fits_complete', completed_outer_jobs=480))
    print(json.dumps(dict(event='all_fits_complete', seconds=time.monotonic()-start)), flush=True)


if __name__ == '__main__': main()
