"""Masked multi-task regression; exact objectives and gradients in float64."""
import numpy as np
from scipy.linalg import block_diag, solve
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-4, 4, 17).tolist()
LINEAR_GRID = [dict(alpha=a, coupling_ratio=c) for c in [0., .1, 1., 10.] for a in ALPHAS]
NN_GRID = [dict(auxiliary_weight=w, width=h, alpha=a)
           for w in [0., .25, 1.] for h in [16, 32] for a in [10., 100.]]
ARMS = ['multitask_linear', 'matched_single_nn', 'multitask_nn', 'positive_aux_nn']
MAX_ITER = 3000


def prepare(x, y, tr, te):
    sx = StandardScaler().fit(x[tr])
    z, zz = sx.transform(x[tr]), sx.transform(x[te])
    mask = np.isfinite(y[tr])
    counts = mask.sum(axis=0)
    assert counts.min() > 1 and np.all(mask[:, 0])
    mean, scale = np.nanmean(y[tr], axis=0), np.nanstd(y[tr], axis=0)
    assert np.all(scale > 0)
    target = np.where(mask, (y[tr]-mean)/scale, 0.)
    return z, zz, target, mask, dict(x_scaler=sx, task_mean=mean, task_scale=scale, task_counts=counts)


def linear_path(x, y, tr, te, settings):
    z, zz, target, mask, scaling = prepare(x, y, tr, te)
    n, p = z.shape
    t = y.shape[1]
    means = np.array([z[mask[:, j]].mean(axis=0) for j in range(t)])
    grams, rhs = [], []
    for j in range(t):
        a = z[mask[:, j]]-means[j]
        factor = n/mask[:, j].sum()
        grams.append(factor*(a.T@a))
        rhs.append(factor*(a.T@target[mask[:, j], j]))
    gram, xy = block_diag(*grams), np.concatenate(rhs)
    couple = np.kron(np.eye(t)-np.ones((t, t))/t, np.eye(p))
    pred = np.empty((len(settings), len(te), t))
    models, diagnostics = [], []
    for c, params in enumerate(settings):
        penalty = params['alpha']*(np.eye(t*p)+params['coupling_ratio']*couple)
        normal = gram+penalty
        coef = solve(normal, xy, assume_a='pos', check_finite=False).reshape(t, p)
        intercept = -np.sum(means*coef, axis=1)
        pred[c] = (zz@coef.T+intercept)*scaling['task_scale']+scaling['task_mean']
        error = float(np.max(np.abs(normal@coef.ravel()-xy)))
        assert error < 1e-7
        models.append(dict(family='linear', parameters=params, coefficients=coef, intercept=intercept, **scaling))
        diagnostics.append(dict(normal_equation_max_residual=error))
    return pred, models, diagnostics


def shapes(p, h, t):
    return [(p, h), (h,), (h, t), (t,)]


def unpack(theta, p, h, t):
    values, offset = [], 0
    for shape in shapes(p, h, t):
        size = int(np.prod(shape))
        values.append(theta[offset:offset+size].reshape(shape))
        offset += size
    assert offset == len(theta)
    return values


def initialize(p, h, t, seed):
    # The shared hidden layer and primary head are identical at each seed for
    # single-task and multi-task fits. Auxiliary initialization uses another RNG.
    rng = np.random.RandomState(int(seed))
    bound = np.sqrt(6/(p+h))
    w = rng.uniform(-bound, bound, size=(p, h))
    b = rng.uniform(-bound, bound, size=h)
    bound = np.sqrt(6/(h+1))
    v0 = rng.uniform(-bound, bound, size=(h, 1))
    c0 = rng.uniform(-bound, bound, size=1)
    if t > 1:
        aux = np.random.RandomState(int(seed)^0x51A27D39)
        v = np.column_stack([v0, aux.uniform(-bound, bound, size=(h, t-1))])
        c = np.r_[c0, aux.uniform(-bound, bound, size=t-1)]
    else:
        v, c = v0, c0
    return np.concatenate([a.ravel() for a in [w, b, v, c]])


def objective(theta, z, target, mask, alpha, weights, h):
    n, p = z.shape
    t = target.shape[1]
    w, b, v, c = unpack(theta, p, h, t)
    hidden = np.tanh(z@w+b)
    residual = hidden@v+c-target
    factors = mask*weights[None, :]/mask.sum(axis=0)[None, :]
    loss = .5*np.sum(factors*residual**2)+.5*alpha/n*(np.sum(w*w)+np.sum(v*v))
    delta = factors*residual
    gv = hidden.T@delta+alpha/n*v
    gc = delta.sum(axis=0)
    back = (delta@v.T)*(1-hidden*hidden)
    gw = z.T@back+alpha/n*w
    gb = back.sum(axis=0)
    return float(loss), np.concatenate([a.ravel() for a in [gw, gb, gv, gc]])


def neural_fit(x, y, tr, te, params, seeds):
    z, zz, target, mask, scaling = prepare(x, y, tr, te)
    aux = params['auxiliary_weight']
    t = 1 if aux == 0 else y.shape[1]
    target, mask = target[:, :t], mask[:, :t]
    weights = np.r_[1., np.full(t-1, aux/(t-1))] if t > 1 else np.ones(1)
    weights /= weights.sum()
    members, preds, diagnostics = [], [], []
    h, p = params['width'], z.shape[1]
    for seed in seeds:
        theta = initialize(p, h, t, seed)
        args = (z, target, mask, params['alpha'], weights, h)
        fit = minimize(objective, theta, args=args, jac=True, method='L-BFGS-B',
                       options=dict(maxiter=MAX_ITER, maxfun=50000, gtol=1e-5, ftol=1e-12, maxls=40))
        value, gradient = objective(fit.x, *args)
        assert fit.success, (fit.message, np.max(np.abs(gradient)))
        w, b, v, c = unpack(fit.x, p, h, t)
        pp = ((np.tanh(zz@w+b)@v+c)*scaling['task_scale'][:t]+scaling['task_mean'][:t])
        preds.append(pp)
        members.append(dict(theta=fit.x, seed=int(seed)))
        diagnostics.append(dict(seed=int(seed), success=bool(fit.success), message=str(fit.message),
                                iterations=int(fit.nit), function_evaluations=int(fit.nfev),
                                loss=value, gradient_infinity_norm=float(np.max(np.abs(gradient)))))
    return np.mean(preds, axis=0), dict(family='neural', parameters=params, n_tasks=t,
                                      task_weights=weights, members=members, **scaling), diagnostics


def predict(model, x):
    z = model['x_scaler'].transform(x)
    if model['family'] == 'linear':
        return (z@model['coefficients'].T+model['intercept'])*model['task_scale']+model['task_mean']
    t, h = model['n_tasks'], model['parameters']['width']
    pp = []
    for m in model['members']:
        w, b, v, c = unpack(m['theta'], z.shape[1], h, t)
        pp.append((np.tanh(z@w+b)@v+c)*model['task_scale'][:t]+model['task_mean'][:t])
    return np.mean(pp, axis=0)
