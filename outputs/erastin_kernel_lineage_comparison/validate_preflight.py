#!/usr/bin/env python3
"""Independent APIs and augmented least-squares checks before the full sweep."""
import time

import numpy as np
from scipy.linalg import lstsq
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler, KernelCenterer
from sklearn.svm import SVR

import run_benchmark as run
from models import FAMILIES, GRIDS, path, predict
from make_splits import inner_rows, outer_split, inner_folds


def independent(family, p, x, y, lineage, tr, te):
    sx = StandardScaler().fit(x[tr])
    sy = StandardScaler().fit(y[tr, None])
    z, zz = sx.transform(x[tr]), sx.transform(x[te])
    target = sy.transform(y[tr, None]).ravel()
    if family == 'svr':
        m = SVR(kernel='rbf', C=p['C'], epsilon=p['epsilon'], gamma=p['gamma'],
                tol=1e-6, max_iter=200000).fit(z, target)
        assert m.fit_status_ == 0
        pred = m.predict(zz)
    elif family == 'kernel_ridge':
        w = p['rbf_weight']
        k = (1-w)*(z@z.T)/64+w*rbf_kernel(z, gamma=p['gamma'])
        q = (1-w)*(zz@z.T)/64+w*rbf_kernel(zz, z, gamma=p['gamma'])
        center = KernelCenterer().fit(k)
        pred = KernelRidge(alpha=p['alpha'], kernel='precomputed').fit(center.transform(k), target).predict(center.transform(q))
    else:
        levels = np.unique(lineage[tr])
        d = (lineage[tr, None] == levels[None, :]).astype(float)
        dd = (lineage[te, None] == levels[None, :]).astype(float)
        if family == 'hierarchical_ridge':
            d, dd = np.column_stack([z, d]), np.column_stack([zz, dd])
            penalty = np.r_[np.full(64, p['gene_alpha']), np.full(len(levels), p['lineage_alpha'] or 0.)]
        else:
            penalty = np.full(len(levels), p['lineage_alpha'] or 0.)
        if p['lineage_alpha'] is None:
            keep = 64 if family == 'hierarchical_ridge' else 0
            d, dd, penalty = d[:, :keep], dd[:, :keep], penalty[:keep]
        # Intercept is explicit and unpenalized; this reference does not center D.
        d, dd = np.column_stack([np.ones(len(tr)), d]), np.column_stack([np.ones(len(te)), dd])
        ridge_rows = np.diag(np.sqrt(np.r_[0., penalty]))
        coef = lstsq(np.vstack([d, ridge_rows]), np.r_[target, np.zeros(d.shape[1])], lapack_driver='gelsd')[0]
        pred = dd@coef
    return sy.inverse_transform(pred[:, None]).ravel()


def main():
    begin = time.monotonic()
    ph = run.prepare()
    run.initialize()
    records = []
    split_checks = 0
    for assay, s in run.STATE.items():
        assert s['x'].shape[1] == 64 and np.isfinite(s['x']).all() and np.isfinite(s['y']).all()
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
                assert set(s['lineage'][a]) == set(s['lineage'])
                split_checks += 1
        a, b = inner_rows(s['inner'][0], 0)
        for family in FAMILIES:
            pred, _, _ = path(family, GRIDS[family], s['x'], s['y'], s['lineage'], a, b)
            error = 0.
            for c, p in enumerate(GRIDS[family]):
                ref = independent(family, p, s['x'], s['y'], s['lineage'], a, b)
                error = max(error, float(np.max(np.abs(pred[c]-ref))))
            assert error < 2e-6, (assay, family, error)
            m = run.fit_job(assay, family, 0, ph)
            data = np.load(run.root_for(assay, family, 0).with_suffix('.npz'))
            ref = independent(family, m['selected_parameters'], s['x'], s['y'], s['lineage'], data['train_rows'], data['test_rows'])
            outer_error = float(np.max(np.abs(ref-data['test_prediction'])))
            assert outer_error < 2e-6
            # Changing validation labels must not change fitted predictions.
            poisoned = s['y'].copy()
            poisoned[b] += 10000.
            original, model, _ = path(family, [GRIDS[family][0]], s['x'], s['y'], s['lineage'], a, b, retain=True)
            changed, _, _ = path(family, [GRIDS[family][0]], s['x'], poisoned, s['lineage'], a, b)
            np.testing.assert_array_equal(original, changed)
            np.testing.assert_allclose(predict(model, s['x'][b], s['lineage'][b]), original[0], atol=1e-10, rtol=0)
            records.append(dict(assay=assay, family=family, candidates_independently_checked=len(GRIDS[family]),
                                inner_max_error=error, selected_outer_max_error=outer_error, heldout_label_invariance=True))
            print('PREFLIGHT_ARM', assay, family, error, flush=True)
    np.testing.assert_array_equal(run.STATE['CTRP']['genes'], run.STATE['PRISM']['genes'])
    run.verify_sources()
    v = dict(passed=True, protocol_sha256=ph, split_checks=split_checks, independent_checks=records,
             outer_fits_retained=8, seconds=time.monotonic()-begin,
             split_arrays_reproduced=True, all_lineages_on_both_outer_sides=True)
    run.dump(run.OUT/'preflight_validation.json', v)
    print('PREFLIGHT_PASSED', v['seconds'], flush=True)


if __name__ == '__main__':
    main()
