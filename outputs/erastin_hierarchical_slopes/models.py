"""Gaussian partial pooling of lineage intercepts and all 64 gene slopes.

Primal: SSE + alpha*||beta||^2 + alpha/intercept_weight*||a||^2
        + alpha/slope_weight*sum_l ||u_l||^2.
Zero variance weights remove the respective random effects. Local deviations use
within-lineage training-centered expression. The dual fit has a free intercept.
"""
import numpy as np
from scipy.linalg import eigh
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-4, 4, 17).tolist()
INTERCEPT_WEIGHTS = [0., .1, 1., 10., 100.]
SLOPE_WEIGHTS = [0., .01, .1, 1., 10.]
GRID = [dict(alpha=a, intercept_weight=i, slope_weight=s)
        for s in SLOPE_WEIGHTS for i in INTERCEPT_WEIGHTS for a in ALPHAS]
ARMS = ['matched_intercept', 'hierarchical_slopes', 'positive_slopes']


def prepare(x, y, lineage, tr, te):
    sx, sy = StandardScaler().fit(x[tr]), StandardScaler().fit(y[tr, None])
    z, zz = sx.transform(x[tr]), sx.transform(x[te])
    target = sy.transform(y[tr, None]).ravel()
    target -= target.mean()
    levels = np.unique(lineage[tr])
    centers = np.array([z[lineage[tr] == l].mean(axis=0) for l in levels])
    li = np.searchsorted(levels, lineage[tr])
    # Unknown lineages receive zero random effects; evaluated lineages are known.
    lookup = {str(l): j for j, l in enumerate(levels)}
    lj = np.array([lookup.get(str(l), -1) for l in lineage[te]])
    q = z-centers[li]
    qq = zz.copy()
    known = lj >= 0
    qq[known] -= centers[lj[known]]
    qq[~known] = 0.
    same = li[:, None] == li[None, :]
    cross = (lj[:, None] == li[None, :]) & known[:, None]
    components = (z@z.T, same.astype(float), (q@q.T)*same)
    queries = (zz@z.T, cross.astype(float), (qq@q.T)*cross)
    return z, zz, q, qq, target, components, queries, dict(x_scaler=sx, y_scaler=sy, levels=levels,
                                                         lineage_centers=centers, train_lineage_index=li)


def path(x, y, lineage, tr, te, settings=GRID, retain=False):
    z, zz, q, qq, target, ks, qs, scaling = prepare(x, y, lineage, tr, te)
    predictions = np.empty((len(settings), len(te)))
    models, cache = [], {}
    min_eigenvalue = 0.
    for j, p in enumerate(settings):
        key = p['intercept_weight'], p['slope_weight']
        if key not in cache:
            ki, kl = key
            k = ks[0]+ki*ks[1]+kl*ks[2]
            query = qs[0]+ki*qs[1]+kl*qs[2]
            mean, grand = k.mean(axis=0), float(k.mean())
            kc = k-mean[None, :]-mean[:, None]+grand
            qc = query-mean[None, :]-query.mean(axis=1)[:, None]+grand
            eig, vec = eigh(kc, check_finite=False)
            min_eigenvalue = min(min_eigenvalue, float(eig.min()))
            assert eig.min() > -1e-7
            cache[key] = (np.maximum(eig, 0.), vec, qc@vec, vec.T@target, kc)
        eig, vec, projected_query, projected_y, kc = cache[key]
        weights = projected_y/(eig+p['alpha'])
        standardized = projected_query@weights
        predictions[j] = scaling['y_scaler'].inverse_transform(standardized[:, None]).ravel()
        if retain:
            dual = vec@weights
            dual -= dual.mean()
            beta = z.T@dual
            li = scaling['train_lineage_index']
            offsets = p['intercept_weight']*np.bincount(li, weights=dual, minlength=len(scaling['levels']))
            deviations = p['slope_weight']*np.array([q[li == l].T@dual[li == l] for l in range(len(scaling['levels']))])
            fitted = z@beta+offsets[li]+np.sum(q*deviations[li], axis=1)
            intercept = float(target.mean()-fitted.mean())
            error = float(np.max(np.abs((kc+p['alpha']*np.eye(len(tr)))@dual-target)))
            assert error < 1e-6
            model = dict(family='hierarchical_slopes', parameters=p, global_coefficients=beta,
                         lineage_offsets=offsets, lineage_deviations=deviations, global_intercept=intercept,
                         dual=dual, effective_degrees_of_freedom=float(1+np.sum(eig/(eig+p['alpha']))),
                         normal_equation_max_residual=error, **scaling)
            # The compact primal representation must reproduce dual predictions.
            np.testing.assert_allclose(predict(model, x[te], lineage[te]), predictions[j], atol=2e-8, rtol=0)
            models.append(model)
    assert np.isfinite(predictions).all()
    diagnostics = dict(x_mean=scaling['x_scaler'].mean_, x_scale=scaling['x_scaler'].scale_,
                       y_mean=float(scaling['y_scaler'].mean_[0]), y_scale=float(scaling['y_scaler'].scale_[0]),
                       min_centered_kernel_eigenvalue=min_eigenvalue,
                       lineage_centers=scaling['lineage_centers'], levels=scaling['levels'])
    return predictions, models, diagnostics


def predict(model, x, lineage):
    z = model['x_scaler'].transform(x)
    lookup = {str(l): j for j, l in enumerate(model['levels'])}
    pred = z@model['global_coefficients']+model['global_intercept']
    # Assign random terms by lineage, using only training-derived centers.
    for label in np.unique(lineage):
        rows = np.flatnonzero(lineage == label)
        if str(label) in lookup:
            j = lookup[str(label)]
            pred[rows] += model['lineage_offsets'][j]
            pred[rows] += (z[rows]-model['lineage_centers'][j])@model['lineage_deviations'][j]
    return model['y_scaler'].inverse_transform(pred[:, None]).ravel()
