#!/usr/bin/env python3
"""Explicit-feature reference for the mixed-model dual and leakage controls."""
import time

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import models
import run_benchmark as run
from make_splits import inner_rows, outer_split, inner_folds


def independent(x, y, lineage, tr, te, params):
    sx = StandardScaler().fit(x[tr])
    z, zz = sx.transform(x[tr]), sx.transform(x[te])
    levels = np.unique(lineage[tr])
    dt = (lineage[tr, None] == levels[None, :]).astype(float)
    dv = (lineage[te, None] == levels[None, :]).astype(float)
    centers = np.array([z[lineage[tr] == l].mean(axis=0) for l in levels])
    local_train, local_test = [], []
    for j, l in enumerate(levels):
        local_train.append((z-centers[j])*dt[:, j, None])
        local_test.append((zz-centers[j])*dv[:, j, None])
    a, b = [z], [zz]
    if params['intercept_weight'] > 0:
        scale = np.sqrt(params['intercept_weight'])
        a.append(scale*dt)
        b.append(scale*dv)
    if params['slope_weight'] > 0:
        scale = np.sqrt(params['slope_weight'])
        a.append(scale*np.column_stack(local_train))
        b.append(scale*np.column_stack(local_test))
    # Native primal/dual ridge on independently constructed explicit features.
    model = Ridge(alpha=params['alpha'], fit_intercept=True, solver='cholesky').fit(np.column_stack(a), y[tr])
    return model.predict(np.column_stack(b))


def main():
    begin = time.monotonic()
    ph = run.prepare()
    run.initialize()
    records = []
    split_checks = 0
    for assay, s in run.STATE.items():
        x, y, lin = s['x'], s['y'], s['lineage']
        for r, mask in enumerate(s['test']):
            np.testing.assert_array_equal(mask, outer_split(lin, s['groups'], s['seeds'][r]))
            np.testing.assert_array_equal(s['inner'][r], inner_folds(lin, s['groups'], mask, s['inner_seeds'][r]))
            tr, te = np.flatnonzero(~mask), np.flatnonzero(mask)
            assert set(s['groups'][tr]).isdisjoint(s['groups'][te])
            assert set(lin[tr]) == set(lin[te]) == set(lin)
            split_checks += 1
            for k in range(5):
                a, b = inner_rows(s['inner'][r], k)
                assert set(a)|set(b) == set(tr) and set(s['groups'][a]).isdisjoint(s['groups'][b])
                assert set(lin[a]) == set(lin)
                split_checks += 1
        a, b = inner_rows(s['inner'][0], 0)
        start = time.monotonic()
        pp, _, _ = models.path(x, y, lin, a, b)
        pilot = time.monotonic()-start
        error = 0.
        for c, params in enumerate(models.GRID):
            ref = independent(x, y, lin, a, b, params)
            error = max(error, float(np.max(np.abs(ref-pp[c]))))
        assert error < 2e-7, (assay, error)
        params = dict(alpha=100., intercept_weight=10., slope_weight=1.)
        p1, m1, _ = models.path(x, y, lin, a, b, [params], retain=True)
        poisoned = y.copy()
        poisoned[b] += 10000.
        p2, m2, _ = models.path(x, poisoned, lin, a, b, [params], retain=True)
        np.testing.assert_array_equal(p1, p2)
        changed = x.copy()
        changed[b] += 100.
        _, m2, _ = models.path(changed, y, lin, a, b, [params], retain=True)
        for key in ['global_coefficients', 'lineage_offsets', 'lineage_deviations', 'lineage_centers', 'dual']:
            np.testing.assert_array_equal(m1[0][key], m2[0][key])
        # The zero-local-variance corner is exactly the original ridge objective.
        z, zz = m1[0]['x_scaler'].transform(x[a]), m1[0]['x_scaler'].transform(x[b])
        zero = models.GRID[:17]
        p0, _, _ = models.path(x, y, lin, a, b, zero)
        ridge_error = 0.
        for c, p in enumerate(zero):
            ref = Ridge(alpha=p['alpha'], solver='svd').fit(z, y[a]).predict(zz)
            ridge_error = max(ridge_error, float(np.max(np.abs(ref-p0[c]))))
        assert ridge_error < 1e-8
        # A singleton training lineage has no within-lineage slope evidence.
        singleton = []
        for j, l in enumerate(m1[0]['levels']):
            if np.sum(lin[a] == l) == 1:
                np.testing.assert_array_equal(m1[0]['lineage_deviations'][j], np.zeros(64))
                singleton.append(str(l))
        records.append(dict(assay=assay, independent_explicit_feature_candidates=425,
                            prediction_max_error=error, ridge_corner_max_error=ridge_error,
                            heldout_response_invariance=True, heldout_feature_invariance_of_fitted_effects=True,
                            singleton_local_slopes_exactly_zero=singleton, one_inner_fold_pilot_seconds=pilot))
        print('PREFLIGHT_ASSAY', assay, records[-1], flush=True)
    np.testing.assert_array_equal(run.STATE['CTRP']['genes'], run.STATE['PRISM']['genes'])
    run.verify_sources()
    result = dict(passed=True, protocol_sha256=ph, split_checks=split_checks, checks=records, seconds=time.monotonic()-begin)
    run.dump(run.OUT/'preflight_validation.json', result)
    print('PREFLIGHT_PASSED', result['seconds'], flush=True)


if __name__ == '__main__':
    main()
