"""Training-only preprocessing and fixed candidate paths for the new benchmark."""
import warnings

import numpy as np
from scipy.linalg import eigh, solve
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


GENE_ALPHAS = np.logspace(-4, 4, 17).tolist()
LINEAGE_ALPHAS = [0.1, 1., 10., 100., 1000., None]
GRIDS = {
    'svr': [dict(C=c, epsilon=e, gamma=g/64)
            for g in [0.1, 1., 10.] for c in [0.1, 1., 10., 100.]
            for e in [0.05, 0.2]],
    'kernel_ridge': [dict(alpha=a, rbf_weight=w, gamma=g/64)
                     for w, g in [(0., 1.), (0.5, 0.1), (0.5, 1.),
                                  (0.5, 10.), (1., 0.1), (1., 1.), (1., 10.)]
                     for a in np.logspace(-3, 3, 13).tolist()],
    'hierarchical_ridge': [dict(gene_alpha=a, lineage_alpha=b)
                           for b in LINEAGE_ALPHAS for a in GENE_ALPHAS],
    'lineage_only': [dict(gene_alpha=None, lineage_alpha=b) for b in LINEAGE_ALPHAS],
}
FAMILIES = list(GRIDS)


def kernel(a, b, p):
    w = p['rbf_weight']
    linear = a @ b.T / a.shape[1]
    return (1-w)*linear + w*rbf_kernel(a, b, gamma=p['gamma'])


def centered_kernel(k, q):
    mean = k.mean(axis=0)
    grand = float(k.mean())
    return (k-mean[None, :]-mean[:, None]+grand,
            q-mean[None, :]-q.mean(axis=1)[:, None]+grand, mean, grand)


def design(z, labels, levels, with_genes):
    d = (np.asarray(labels)[:, None] == np.asarray(levels)[None, :]).astype(float)
    return np.column_stack([z, d]) if with_genes else d


def path(family, params, x, y, lineage, tr, te, retain=False):
    """Return original-unit predictions for every candidate on a disjoint subset."""
    sx = StandardScaler().fit(x[tr])
    sy = StandardScaler().fit(y[tr, None])
    z, zz = sx.transform(x[tr]), sx.transform(x[te])
    target = sy.transform(y[tr, None]).ravel()
    preds = np.empty((len(params), len(te)))
    saved = None
    caught = []
    max_stationarity = 0.
    cache = {}
    for j, p in enumerate(params):
        model = dict(family=family, parameters=p, x_scaler=sx, y_scaler=sy)
        if family == 'svr':
            gamma = p['gamma']
            if gamma not in cache:
                cache[gamma] = (rbf_kernel(z, gamma=gamma), rbf_kernel(zz, z, gamma=gamma))
            k, q = cache[gamma]
            fitted = SVR(kernel='precomputed', C=p['C'], epsilon=p['epsilon'],
                         tol=1e-6, max_iter=200000, cache_size=256)
            with warnings.catch_warnings(record=True) as ws:
                warnings.simplefilter('always')
                fitted.fit(k, target)
            caught.extend(dict(candidate=j, category=v.category.__name__, message=str(v.message)) for v in ws)
            assert fitted.fit_status_ == 0
            pred = fitted.predict(q)
            model.update(train_z=z, estimator=fitted)
        elif family == 'kernel_ridge':
            key = (p['rbf_weight'], p['gamma'])
            if key not in cache:
                k, q, km, kg = centered_kernel(kernel(z, z, p), kernel(zz, z, p))
                eig, vec = eigh(k, check_finite=False)
                assert eig.min() > -1e-8
                eig = np.maximum(eig, 0.)
                cache[key] = (eig, vec, q@vec, vec.T@target, km, kg)
            eig, vec, qv, vy, km, kg = cache[key]
            weights = vy/(eig+p['alpha'])
            pred = qv@weights
            model.update(train_z=z, dual=vec@weights, kernel_mean=km, kernel_grand=kg)
        else:
            with_genes = family == 'hierarchical_ridge'
            if 'design' not in cache:
                levels = np.unique(lineage[tr])
                d, dd = design(z, lineage[tr], levels, with_genes), design(zz, lineage[te], levels, with_genes)
                dm = d.mean(axis=0)
                dc = d-dm
                cache['design'] = (levels, dm, dc.T@dc, dc.T@target, dd-dm)
            levels, dm, gram, xy, dd = cache['design']
            n_genes = x.shape[1] if with_genes else 0
            if p['lineage_alpha'] is None:
                beta = np.zeros(gram.shape[0])
                if with_genes:
                    beta[:n_genes] = solve(gram[:n_genes, :n_genes]+p['gene_alpha']*np.eye(n_genes),
                                           xy[:n_genes], assume_a='pos', check_finite=False)
            else:
                penalty = np.r_[np.full(n_genes, p['gene_alpha'] or 0., dtype=float),
                                np.full(len(levels), p['lineage_alpha'])]
                normal = gram+np.diag(penalty)
                beta = solve(normal, xy, assume_a='pos', check_finite=False)
                max_stationarity = max(max_stationarity, float(np.max(np.abs(normal@beta-xy))))
            pred = dd@beta
            model.update(levels=levels, design_mean=dm, coefficients=beta, with_genes=with_genes)
        preds[j] = sy.inverse_transform(pred[:, None]).ravel()
        if retain:
            assert len(params) == 1
            saved = model
    assert np.isfinite(preds).all()
    diagnostics = dict(warnings=caught, normal_equation_max_residual=max_stationarity,
                       x_mean=sx.mean_, x_scale=sx.scale_, y_mean=float(sy.mean_[0]), y_scale=float(sy.scale_[0]))
    return preds, saved, diagnostics


def predict(model, x, lineage):
    z = model['x_scaler'].transform(x)
    p = model['parameters']
    if model['family'] == 'svr':
        pred = model['estimator'].predict(rbf_kernel(z, model['train_z'], gamma=p['gamma']))
    elif model['family'] == 'kernel_ridge':
        q = kernel(z, model['train_z'], p)
        q = q-model['kernel_mean'][None, :]-q.mean(axis=1)[:, None]+model['kernel_grand']
        pred = q@model['dual']
    else:
        d = design(z, lineage, model['levels'], model['with_genes'])
        pred = (d-model['design_mean'])@model['coefficients']
    return model['y_scaler'].inverse_transform(pred[:, None]).ravel()
