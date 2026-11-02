"""Training-only genome-wide screening and ridge paths."""
import numpy as np
from scipy.linalg import eigh, svd
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-4, 4, 17).tolist()
FULL_ALPHAS = np.logspace(-4, 8, 25).tolist()
ADDED_COUNTS = [8, 16, 32, 64, 128]
METHODS = ['marginal', 'conditional']
GRID = [dict(method='core_only', added=0, alpha=a) for a in ALPHAS]
GRID += [dict(method=m, added=k, alpha=a) for m in METHODS for k in ADDED_COUNTS for a in ALPHAS]
TAB_MAX_ADDED = 32


def rank_genes(x, y, lineage, core, candidates):
    """x/y/lineage must contain only the current training observations."""
    yy = y-y.mean()
    assert np.sum(yy*yy) > 0
    xx = x[:, candidates].copy()
    xx -= xx.mean(axis=0)
    variance = np.einsum('ij,ij->j', xx, xx)
    valid = variance > 1e-12
    marginal = np.full(len(candidates), np.nan)
    marginal[valid] = (xx[:, valid].T@yy)/np.sqrt(variance[valid]*np.sum(yy*yy))
    z = StandardScaler().fit_transform(x[:, core])
    labels = np.unique(lineage)
    dummy = (lineage[:, None] == labels[None, :]).astype(float)
    controls = np.column_stack([z, dummy])
    controls -= controls.mean(axis=0)
    u, singular, _ = svd(controls, full_matrices=False, check_finite=False)
    keep = singular > singular[0]*max(controls.shape)*np.finfo(float).eps
    q = u[:, keep]
    residual_y = yy-q@(q.T@yy)
    projected = q.T@xx
    residual_variance = np.maximum(variance-np.einsum('ij,ij->j', projected, projected), 0.)
    incremental = valid & (residual_variance > np.maximum(1e-12, variance*1e-8))
    conditional = np.full(len(candidates), np.nan)
    conditional[incremental] = (xx[:, incremental].T@residual_y)/np.sqrt(residual_variance[incremental]*np.sum(residual_y**2))
    scores = {'marginal': marginal, 'conditional': conditional}
    rankings = {}
    for method, values in scores.items():
        finite = np.isfinite(values)
        assert finite.sum() >= 128 and np.max(np.abs(values[finite])) < 1+1e-6
        priority = np.where(finite, np.abs(values), -np.inf)
        rankings[method] = candidates[np.argsort(-priority, kind='stable')]
    return rankings, scores, dict(training_n=len(y), control_design_rank=int(keep.sum()),
                                  variable_candidates=int(valid.sum()), incremental_candidates=int(incremental.sum()),
                                  residual_y_sum_squares=float(np.sum(residual_y**2)))


def ridge_path(xtr, ytr, xte, alphas, retain=False):
    sx = StandardScaler().fit(xtr)
    z, zz = sx.transform(xtr), sx.transform(xte)
    mean = float(ytr.mean())
    u, s, vt = svd(z, full_matrices=False, check_finite=False)
    factors = (s[:, None]/(s[:, None]**2+np.asarray(alphas)[None, :]))*(u.T@(ytr-mean))[:, None]
    pred = (zz@vt.T)@factors+mean
    model = None
    if retain:
        assert len(alphas) == 1
        model = dict(x_scaler=sx, coef=vt.T@factors[:, 0], intercept=mean, alpha=alphas[0])
        np.testing.assert_allclose(predict(model, xte), pred[:, 0], atol=1e-10, rtol=0)
    return pred.T, model


def full_ridge_path(xtr, ytr, xte, alphas, retain=False):
    sx = StandardScaler().fit(xtr)
    z, zz = sx.transform(xtr), sx.transform(xte)
    mean = float(ytr.mean())
    kernel = z@z.T
    eig, vec = eigh(kernel, check_finite=False)
    assert eig.min() > -1e-6
    eig = np.maximum(eig, 0.)
    duals = (vec.T@(ytr-mean))[:, None]/(eig[:, None]+np.asarray(alphas)[None, :])
    pred = ((zz@z.T)@vec)@duals+mean
    model = None
    if retain:
        assert len(alphas) == 1
        model = dict(x_scaler=sx, coef=z.T@(vec@duals[:, 0]), intercept=mean, alpha=alphas[0])
        np.testing.assert_allclose(predict(model, xte), pred[:, 0], atol=1e-8, rtol=0)
    return pred.T, model


def predict(model, x):
    return model['x_scaler'].transform(x)@model['coef']+model['intercept']
