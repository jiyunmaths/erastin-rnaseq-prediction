#!/usr/bin/env python3
"""Fixed TabPFN on outer-training-selected compact genome-wide additions."""
import concurrent.futures as futures
import hashlib
import json
import multiprocessing as mp
import os
import socket
import time
import warnings
import zipfile

import common as c
from models import TAB_MAX_ADDED

WORK = c.WS/'work/tabpfn'
for key, value in {'SKB_DATA_DIRECTORY': str(WORK/'skrub_data'), 'HF_HOME': str(WORK/'hf_cache'),
                   'HF_HUB_OFFLINE': '1', 'TABPFN_MODEL_CACHE_DIR': str(WORK/'checkpoints'),
                   'TABPFN_NO_BROWSER': '1', 'TABPFN_DISABLE_TELEMETRY': '1',
                   'MPLCONFIGDIR': str(WORK/'matplotlib_cache')}.items():
    os.environ[key] = value

import numpy as np
import torch
from tabpfn import TabPFNRegressor, save_fitted_tabpfn_model, load_fitted_tabpfn_model


def setup():
    torch.set_num_threads(3)
    torch.set_num_interop_threads(1)
    def blocked(*args, **kwargs):
        raise RuntimeError('Network disabled during local inference.')
    socket.socket.connect = blocked
    socket.create_connection = blocked
    c.initialize()


def fit_repeat(assay, r, ph):
    complete = c.OUT/assay/'folds'/f'tab_repeat_{r+1:02d}_complete.json'
    if complete.exists():
        m = json.load(open(complete))
        assert m['protocol_sha256'] == ph
        for f in m['files']:
            assert c.sha(c.OUT/f['path']) == f['sha256']
        return m
    begin = time.monotonic()
    s = c.STATE[assay]
    tr, te = np.flatnonzero(~s['test'][r]), np.flatnonzero(s['test'][r])
    assert set(s['groups'][tr]).isdisjoint(s['groups'][te])
    info = json.load(open(c.OUT/assay/'folds'/f'screening__r{r+1:02d}.json'))
    assert info['protocol_sha256'] == ph
    ranks = np.load(c.OUT/assay/'folds'/f'screening__r{r+1:02d}.npz')
    selected = info['selected_parameters']
    core = c.INDEX['core_indices']
    count = min(selected['added'], TAB_MAX_ADDED)
    adaptive = np.r_[core, ranks['ranked_'+selected['method']][:count]] if count else core.copy()
    panels = {'tab_ridge_screen32': adaptive, 'tab_conditional32': np.r_[core, ranks['ranked_conditional'][:32]]}
    saved = {}
    files = []
    for arm, panel in panels.items():
        assert np.array_equal(panel[:64], core) and len(set(panel)) == len(panel) <= 96
        digest = hashlib.sha256(panel.tobytes()).hexdigest()[:16]
        reused = len(panel) == 64
        if digest in saved:
            result = saved[digest]
        elif reused:
            old = np.load(c.TAB/assay/'predictions.npz')
            model_ref = c.TAB/assay/'folds'/f'tabpfn35_fast__p64__r{r+1:02d}.tabpfn_fit'
            result = dict(pred=old['tabpfn35_fast'][r, te], model_ref=str(model_ref.relative_to(c.WS)),
                          model_sha256=c.sha(model_ref), restored_prediction_max_error=None,
                          native_feature_coverage=64, warnings=[], fit_seconds=0., predict_seconds=0.)
        else:
            train, test = c.features(assay, tr, panel), c.features(assay, te, panel)
            model = TabPFNRegressor(model_path=str(c.CHECKPOINT), device='cpu', n_estimators=8,
                                    n_preprocessing_jobs=1, random_state=int(s['seeds'][r]), categorical_features_indices=[],
                                    fit_mode='fit_preprocessors', memory_saving_mode='auto',
                                    inference_precision=torch.float32, tuning_config=None, show_progress_bar=False)
            t = time.monotonic()
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter('always')
                model.fit(train, s['y'][tr].copy())
                fit_seconds = time.monotonic()-t
                t = time.monotonic()
                pred = model.predict(test, output_type='mean')
                predict_seconds = time.monotonic()-t
            assert np.isfinite(pred).all() and model.n_estimators_ == 8 and model.n_features_in_ == len(panel)
            subsets = model.ensemble_preprocessor_.subsample_feature_indices
            covered = set(range(len(panel))) if any(i is None for i in subsets) else set(np.concatenate(subsets).tolist())
            assert covered == set(range(len(panel)))
            assert abs(model.y_train_mean_-s['y'][tr].mean()) < 1e-7
            native = c.OUT/assay/'folds'/f'tab_context_{digest}__r{r+1:02d}.tabpfn_fit'
            save_fitted_tabpfn_model(model, native)
            files.append(native)
            restored_error = None
            if r in [0, 19]:
                again = load_fitted_tabpfn_model(native, device='cpu').predict(test, output_type='mean')
                restored_error = float(np.max(np.abs(again-pred)))
                assert restored_error < 2e-6
            result = dict(pred=pred, model_ref=str(native.relative_to(c.WS)), model_sha256=c.sha(native),
                          restored_prediction_max_error=restored_error, native_feature_coverage=len(covered),
                          warnings=[dict(category=v.category.__name__, message=str(v.message)) for v in caught],
                          fit_seconds=fit_seconds, predict_seconds=predict_seconds)
        saved[digest] = result
        root = c.root_for(assay, arm, r)
        np.savez_compressed(root.with_suffix('.npz'), train_rows=tr, test_rows=te,
                            test_prediction=result['pred'], feature_indices=panel)
        meta = {k: v for k, v in result.items() if k != 'pred'}
        meta.update(assay=assay, arm=arm, repeat=r+1, protocol_sha256=ph, gene_count=len(panel),
                    added=len(panel)-64, screening_method=selected['method'] if arm == 'tab_ridge_screen32' else 'conditional',
                    outer_split_seed=int(s['seeds'][r]), reused_core64_predictions=reused,
                    npz_sha256=c.sha(root.with_suffix('.npz')))
        c.dump(root.with_suffix('.json'), meta)
        files += [root.with_suffix('.'+e) for e in ['npz', 'json']]
    m = dict(assay=assay, repeat=r+1, protocol_sha256=ph, seconds=time.monotonic()-begin,
             unique_panels=len(saved), new_contexts=sum(len(v['pred']) > 0 and '/erastin_genomewide_search/' in v['model_ref'] for v in saved.values()),
             files=[dict(path=str(p.relative_to(c.OUT)), sha256=c.sha(p)) for p in files])
    c.dump(complete, m)
    return m


def main():
    begin = time.monotonic()
    ph = c.prepare()
    assert json.load(open(c.OUT/'ridge_completion.json'))['complete']
    with futures.ProcessPoolExecutor(max_workers=2, mp_context=mp.get_context('spawn'), initializer=setup) as pool:
        jobs = [pool.submit(fit_repeat, a, r, ph) for a in c.ASSAYS for r in range(20)]
        for done, future in enumerate(futures.as_completed(jobs), 1):
            m = future.result()
            elapsed = time.monotonic()-begin
            status = dict(phase='TabPFN_genomewide_additions', completed_assay_repeats=done, total_assay_repeats=40,
                          elapsed_seconds=elapsed, estimated_remaining_seconds=elapsed/done*(40-done),
                          last_assay=m['assay'], last_repeat=m['repeat'])
            c.dump(c.OUT/'progress.json', status)
            print(json.dumps(status), flush=True)
    c.verify_sources()
    c.dump(c.OUT/'tab_completion.json', dict(complete=True, outer_arm_bundles=80, seconds=time.monotonic()-begin))
    print('TABPFN_ADDITIONS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
