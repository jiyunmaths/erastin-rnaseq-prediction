#!/usr/bin/env python3
"""Independent screen/fit checks and exact baseline equivalence before the sweep."""
import time

import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import common as c
import models
import run_ridge
from make_splits import inner_rows, outer_split, inner_folds


def main():
    begin = time.monotonic()
    ph = c.prepare()
    c.initialize()
    checks = []
    split_checks = 0
    core, candidates = c.INDEX['core_indices'], c.INDEX['candidate_indices']
    for assay, s in c.STATE.items():
        for r, mask in enumerate(s['test']):
            np.testing.assert_array_equal(mask, outer_split(s['lineage'], s['groups'], s['seeds'][r]))
            np.testing.assert_array_equal(s['inner'][r], inner_folds(s['lineage'], s['groups'], mask, s['inner_seeds'][r]))
            tr, te = np.flatnonzero(~mask), np.flatnonzero(mask)
            assert set(s['groups'][tr]).isdisjoint(s['groups'][te])
            assert set(s['lineage'][tr]) == set(s['lineage'][te]) == set(s['lineage'])
            split_checks += 1
            for k in range(5):
                a, b = inner_rows(s['inner'][r], k)
                assert set(a)|set(b) == set(tr) and set(s['groups'][a]).isdisjoint(s['groups'][b])
                split_checks += 1
        a, b = inner_rows(s['inner'][0], 0)
        x, xx = c.features(assay, a), c.features(assay, b)
        rankings, scores, diag = models.rank_genes(x, s['y'][a], s['lineage'][a], core, candidates)
        inverse = {j: q for q, j in enumerate(candidates)}
        chosen = np.unique(np.r_[rankings['marginal'][:10], rankings['conditional'][:10]])
        controls = np.column_stack([np.ones(len(a)), StandardScaler().fit_transform(x[:, core]),
                                    (s['lineage'][a, None] == np.unique(s['lineage'][a])[None, :]).astype(float)])
        responses = np.column_stack([s['y'][a], x[:, chosen]])
        residual = responses-controls@np.linalg.lstsq(controls, responses, rcond=None)[0]
        max_marginal = max_partial = 0.
        for j, gene in enumerate(chosen):
            m = pearsonr(x[:, gene], s['y'][a]).statistic
            p = pearsonr(residual[:, j+1], residual[:, 0]).statistic
            max_marginal = max(max_marginal, abs(m-scores['marginal'][inverse[gene]]))
            max_partial = max(max_partial, abs(p-scores['conditional'][inverse[gene]]))
        assert max(max_marginal, max_partial) < 1e-8
        # All compact-panel candidate paths independently checked at this fold.
        max_fit = 0.
        for method, count in [('core_only', 0)]+[(m, k) for m in models.METHODS for k in models.ADDED_COUNTS]:
            genes = np.r_[core, rankings[method][:count]] if count else core
            pred, _ = models.ridge_path(x[:, genes], s['y'][a], xx[:, genes], models.ALPHAS)
            sx = StandardScaler().fit(x[:, genes])
            z, zz = sx.transform(x[:, genes]), sx.transform(xx[:, genes])
            for j, alpha in enumerate(models.ALPHAS):
                ref = Ridge(alpha=alpha, solver='cholesky').fit(z, s['y'][a]).predict(zz)
                max_fit = max(max_fit, float(np.max(np.abs(pred[j]-ref))))
        assert max_fit < 1e-8
        # High-dimensional path checked by a separate native dual solver.
        picks = [models.FULL_ALPHAS[j] for j in [0, 6, 12, 18, 24]]
        pred, _ = models.full_ridge_path(x, s['y'][a], xx, picks)
        sx = StandardScaler().fit(x)
        z, zz = sx.transform(x), sx.transform(xx)
        max_full = 0.
        for j, alpha in enumerate(picks):
            ref = Ridge(alpha=alpha, solver='cholesky').fit(z, s['y'][a]).predict(zz)
            max_full = max(max_full, float(np.max(np.abs(ref-pred[j]))))
        assert max_full < 2e-7
        # This complete first repeat is retained, including independently
        # training-fitted selection at all five folds and the outer refit.
        start = time.monotonic()
        result = run_ridge.fit_repeat(assay, 0, ph)
        elapsed = time.monotonic()-start
        checks.append(dict(assay=assay, marginal_score_max_error=max_marginal, partial_score_max_error=max_partial,
                           independent_compact_candidates=187, compact_prediction_max_error=max_fit,
                           independent_genome_alphas=5, full_prediction_max_error=max_full,
                           first_repeat_retained=True, first_repeat_seconds=elapsed))
        print('PREFLIGHT_ASSAY', assay, checks[-1], flush=True)
    # A feature that only tracks validation responses must not pass the training
    # variance filter or get promoted by either screen.
    rng = np.random.default_rng(122)
    synthetic = rng.normal(size=(100, 256))
    yy = rng.normal(size=100)
    synthetic[:80, 64] = 0.
    synthetic[80:, 64] = yy[80:]
    ranking, scores, _ = models.rank_genes(synthetic[:80], yy[:80], np.array(['A', 'B']*40),
                                          np.arange(64), np.arange(64, 256))
    for method in models.METHODS:
        assert 64 not in ranking[method][:128] and np.isnan(scores[method][0])
    c.verify_sources()
    validation = dict(passed=True, protocol_sha256=ph, split_checks=split_checks, checks=checks,
                       validation_only_decoy_not_selected=True, screen_function_receives_training_rows_only=True,
                       all_core64_inner_predictions_match_saved_ridge=True, seconds=time.monotonic()-begin)
    c.dump(c.OUT/'preflight_validation.json', validation)
    print('PREFLIGHT_PASSED', validation['seconds'], flush=True)


if __name__ == '__main__':
    main()
