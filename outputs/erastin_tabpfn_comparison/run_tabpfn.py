#!/usr/bin/env python3
"""Offline CPU TabPFN contexts, with the original 20 outer masks per assay."""
import concurrent.futures as futures
import importlib.metadata
import json
import multiprocessing as mp
import os
import resource
import socket
import time
import warnings
import zipfile

import common as c

for key, value in {'SKB_DATA_DIRECTORY': str(c.WORK/'skrub_data'), 'HF_HOME': str(c.WORK/'hf_cache'),
                   'HF_HUB_OFFLINE': '1', 'TABPFN_MODEL_CACHE_DIR': str(c.WORK/'checkpoints'),
                   'TABPFN_NO_BROWSER': '1', 'TABPFN_DISABLE_TELEMETRY': '1',
                   'MPLCONFIGDIR': str(c.WORK/'matplotlib_cache')}.items():
    os.environ[key] = value

import numpy as np
import torch
import tabpfn
from tabpfn import TabPFNRegressor, save_fitted_tabpfn_model, load_fitted_tabpfn_model


def offline():
    def blocked(*args, **kwargs):
        raise RuntimeError('Network connections disabled during this local benchmark.')
    socket.socket.connect = blocked
    socket.create_connection = blocked


def setup():
    torch.set_num_threads(c.THREADS)
    torch.set_num_interop_threads(1)
    offline()
    c.initialize()


def construct(seed):
    return TabPFNRegressor(model_path=str(c.CHECKPOINT), device='cpu', n_estimators=c.N_ESTIMATORS,
                           n_preprocessing_jobs=1, random_state=int(seed), categorical_features_indices=[],
                           fit_mode='fit_preprocessors', memory_saving_mode='auto',
                           inference_precision=torch.float32, tuning_config=None, show_progress_bar=False)


def fit_job(assay, r, ph):
    root = c.root_for(assay, r)
    if root.with_suffix('.json').exists():
        m = json.load(open(root.with_suffix('.json')))
        assert m['protocol_sha256'] == ph
        for ext in ['npz', 'tabpfn_fit']:
            assert c.sha(root.with_suffix('.'+ext)) == m[ext+'_sha256']
        return m
    start = time.monotonic()
    s = c.STATE[assay]
    tr, te = np.flatnonzero(~s['test'][r]), np.flatnonzero(s['test'][r])
    assert set(s['groups'][tr]).isdisjoint(s['groups'][te])
    assert set(s['lineage'][tr]) == set(s['lineage'][te]) == set(s['lineage'])
    model = construct(s['seeds'][r])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        model.fit(s['x'][tr].copy(), s['y'][tr].copy())
        fit_seconds = time.monotonic()-start
        moment = time.monotonic()
        pred = model.predict(s['x'][te].copy(), output_type='mean')
        predict_seconds = time.monotonic()-moment
    assert np.isfinite(pred).all() and pred.shape == (len(te),)
    assert model.n_features_in_ == 64 and model.n_estimators_ == c.N_ESTIMATORS
    assert model.ensemble_softmax_temperature_ == 1.
    subsets = model.ensemble_preprocessor_.subsample_feature_indices
    covered = set(range(64)) if any(ix is None for ix in subsets) else set(np.concatenate(subsets).tolist())
    assert covered == set(range(64))
    assert abs(model.y_train_mean_-s['y'][tr].mean()) < 1e-7
    assert abs(model.y_train_std_-s['y'][tr].std()) < 1e-7
    save_fitted_tabpfn_model(model, root.with_suffix('.tabpfn_fit'))
    with zipfile.ZipFile(root.with_suffix('.tabpfn_fit')) as z:
        assert z.testzip() is None
    np.savez_compressed(root.with_suffix('.npz'), train_rows=tr, test_rows=te, test_prediction=pred)
    reload_error = None
    if r in [0, 19]:
        restored = load_fitted_tabpfn_model(root.with_suffix('.tabpfn_fit'), device='cpu')
        again = restored.predict(s['x'][te].copy(), output_type='mean')
        reload_error = float(np.max(np.abs(again-pred)))
        assert reload_error < 2e-6
    m = dict(assay=assay, repeat=r+1, protocol_sha256=ph, model='tabpfn35_fast', gene_count=64,
             train_n=len(tr), test_n=len(te), train_lineages=len(set(s['lineage'][tr])), test_lineages=len(set(s['lineage'][te])),
             outer_split_seed=int(s['seeds'][r]), inner_split_seed=int(s['inner_seeds'][r]),
             n_estimators=c.N_ESTIMATORS, no_study_specific_tuning=True, offline_inference=True,
             feature_coverage=len(covered), native_training_target_mean=model.y_train_mean_, native_training_target_std=model.y_train_std_,
             fit_seconds=fit_seconds, predict_seconds=predict_seconds, elapsed_seconds=time.monotonic()-start,
             peak_process_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
             restored_prediction_max_error=reload_error,
             warnings=[dict(category=w.category.__name__, message=str(w.message)) for w in caught],
             versions={name: importlib.metadata.version(name) for name in ['tabpfn', 'torch', 'numpy', 'scipy', 'scikit-learn', 'pandas', 'joblib']},
             npz_sha256=c.sha(root.with_suffix('.npz')), tabpfn_fit_sha256=c.sha(root.with_suffix('.tabpfn_fit')))
    c.dump(root.with_suffix('.json'), m)
    return m


def main():
    begin = time.monotonic()
    ph = c.prepare()
    v = json.load(open(c.OUT/'preflight_validation.json'))
    assert v['passed'] and v['protocol_sha256'] == ph
    with futures.ProcessPoolExecutor(max_workers=c.WORKERS, mp_context=mp.get_context('spawn'), initializer=setup) as pool:
        jobs = [pool.submit(fit_job, a, r, ph) for a in c.ASSAYS for r in range(20)]
        for done, future in enumerate(futures.as_completed(jobs), 1):
            m = future.result()
            elapsed = time.monotonic()-begin
            progress = dict(phase='fitting', completed=done, total=40, elapsed_seconds=elapsed,
                            estimated_remaining_seconds=elapsed/done*(40-done), last_assay=m['assay'], last_repeat=m['repeat'])
            c.dump(c.OUT/'progress.json', progress)
            print(json.dumps(progress), flush=True)
    c.verify_sources()
    c.dump(c.OUT/'fit_completion.json', dict(complete=True, new_contexts=40, seconds=time.monotonic()-begin))
    print('ALL_CONTEXTS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
