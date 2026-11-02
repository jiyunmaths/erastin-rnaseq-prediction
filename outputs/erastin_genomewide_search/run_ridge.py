#!/usr/bin/env python3
"""Nested selector/count/penalty tuning with core genes forced into every panel."""
import concurrent.futures as futures
import json
import multiprocessing as mp
import time

import joblib
import numpy as np

import common as c
import models
from make_splits import inner_rows


def fit_repeat(assay, r, ph):
    complete = c.OUT/assay/'folds'/f'ridge_repeat_{r+1:02d}_complete.json'
    if complete.exists():
        m = json.load(open(complete))
        assert m['protocol_sha256'] == ph
        for row in m['files']:
            assert c.sha(c.OUT/row['path']) == row['sha256']
        return m
    start = time.monotonic()
    s = c.STATE[assay]
    tr, te = np.flatnonzero(~s['test'][r]), np.flatnonzero(s['test'][r])
    core, candidates = c.INDEX['core_indices'], c.INDEX['candidate_indices']
    val = np.full((len(models.GRID), len(s['y'])), np.nan)
    fullval = np.full((len(models.FULL_ALPHAS), len(s['y'])), np.nan)
    loss, fullloss = np.empty((len(models.GRID), 5)), np.empty((len(models.FULL_ALPHAS), 5))
    top = np.empty((5, 2, 128), dtype=np.int32)
    diag = []
    for k in range(5):
        a, b = inner_rows(s['inner'][r], k)
        assert set(a)|set(b) == set(tr) and set(s['groups'][a]).isdisjoint(s['groups'][b])
        train, test = c.features(assay, a), c.features(assay, b)
        rank, scores, d = models.rank_genes(train, s['y'][a], s['lineage'][a], core, candidates)
        diag.append(dict(fold=k, **d))
        pp, _ = models.ridge_path(train[:, core], s['y'][a], test[:, core], models.ALPHAS)
        val[:17, b] = pp
        offset = 17
        for j, method in enumerate(models.METHODS):
            top[k, j] = rank[method][:128]
            for count in models.ADDED_COUNTS:
                indices = np.r_[core, rank[method][:count]]
                pp, _ = models.ridge_path(train[:, indices], s['y'][a], test[:, indices], models.ALPHAS)
                val[offset:offset+17, b] = pp
                offset += 17
        assert offset == len(models.GRID)
        pp, _ = models.full_ridge_path(train, s['y'][a], test, models.FULL_ALPHAS)
        fullval[:, b] = pp
        loss[:, k] = np.sqrt(np.mean((val[:, b]-s['y'][b][None, :])**2, axis=1))
        fullloss[:, k] = np.sqrt(np.mean((fullval[:, b]-s['y'][b][None, :])**2, axis=1))
    valid = s['inner'][r] >= 0
    for arr in [val, fullval]:
        assert np.isfinite(arr[:, valid]).all() and np.isnan(arr[:, ~valid]).all()
    old = np.load(c.BASE[assay]/'folds'/f'ridge__p64__r{r+1:02d}.npz')
    np.testing.assert_allclose(val[:17, valid], old['inner_validation_predictions'][:, valid], atol=1e-10, rtol=0)
    selected = int(np.argmin(loss.mean(axis=1)))
    selected_full = int(np.argmin(fullloss.mean(axis=1)))
    params = models.GRID[selected]
    train, test = c.features(assay, tr), c.features(assay, te)
    rank, scores, outer_diag = models.rank_genes(train, s['y'][tr], s['lineage'][tr], core, candidates)
    chosen = np.r_[core, rank[params['method']][:params['added']]] if params['added'] else core.copy()
    pp, aug = models.ridge_path(train[:, chosen], s['y'][tr], test[:, chosen], [params['alpha']], retain=True)
    fp, full = models.full_ridge_path(train, s['y'][tr], test, [models.FULL_ALPHAS[selected_full]], retain=True)
    files = []
    tuning = c.OUT/assay/'folds'/f'screening__r{r+1:02d}'
    np.savez_compressed(tuning.with_suffix('.npz'), validation_predictions=val, inner_fold_rmse=loss,
                        genome_validation_predictions=fullval, genome_inner_fold_rmse=fullloss, inner_top128=top,
                        ranked_marginal=rank['marginal'], ranked_conditional=rank['conditional'],
                        scores_marginal=scores['marginal'], scores_conditional=scores['conditional'])
    c.dump(tuning.with_suffix('.json'), dict(protocol_sha256=ph, assay=assay, repeat=r+1,
                                             selected_candidate=selected, selected_parameters=params,
                                             selected_full_candidate=selected_full, selected_full_alpha=models.FULL_ALPHAS[selected_full],
                                             inner_diagnostics=diag, outer_diagnostics=outer_diag))
    files += [tuning.with_suffix('.'+e) for e in ['npz', 'json']]
    for arm, pred, model, indices, setting, inner_loss in [
        ('ridge_augmented', pp[0], aug, chosen, params, float(loss[selected].mean())),
        ('ridge_genome', fp[0], full, np.arange(len(c.INDEX['genes'])), dict(alpha=models.FULL_ALPHAS[selected_full]), float(fullloss[selected_full].mean()))]:
        root = c.root_for(assay, arm, r)
        model.update(feature_indices=indices, gene_columns=c.INDEX['gene_columns'][indices])
        joblib.dump(model, root.with_suffix('.joblib'), compress=3)
        np.savez_compressed(root.with_suffix('.npz'), train_rows=tr, test_rows=te, test_prediction=pred, feature_indices=indices)
        c.dump(root.with_suffix('.json'), dict(assay=assay, arm=arm, repeat=r+1, protocol_sha256=ph,
                                               selected_parameters=setting, mean_inner_rmse=inner_loss,
                                               train_n=len(tr), test_n=len(te), gene_count=len(indices),
                                               npz_sha256=c.sha(root.with_suffix('.npz')), joblib_sha256=c.sha(root.with_suffix('.joblib'))))
        files += [root.with_suffix('.'+e) for e in ['json', 'npz', 'joblib']]
    result = dict(assay=assay, repeat=r+1, protocol_sha256=ph, seconds=time.monotonic()-start,
                  selected_parameters=params, files=[dict(path=str(p.relative_to(c.OUT)), sha256=c.sha(p)) for p in files])
    c.dump(complete, result)
    return result


def main():
    begin = time.monotonic()
    ph = c.prepare()
    pf = json.load(open(c.OUT/'preflight_validation.json'))
    assert pf['passed'] and pf['protocol_sha256'] == ph
    with futures.ProcessPoolExecutor(max_workers=c.WORKERS, mp_context=mp.get_context('spawn'), initializer=c.initialize) as pool:
        jobs = [pool.submit(fit_repeat, a, r, ph) for a in c.ASSAYS for r in range(20)]
        for done, future in enumerate(futures.as_completed(jobs), 1):
            m = future.result()
            elapsed = time.monotonic()-begin
            status = dict(phase='nested_ridge_screening', completed=done, total=40, elapsed_seconds=elapsed,
                          estimated_remaining_seconds=elapsed/done*(40-done), last_assay=m['assay'], last_repeat=m['repeat'])
            c.dump(c.OUT/'progress.json', status)
            print(json.dumps(status), flush=True)
    c.verify_sources()
    c.dump(c.OUT/'ridge_completion.json', dict(complete=True, outer_models=80, seconds=time.monotonic()-begin))
    print('RIDGE_SCREENING_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
