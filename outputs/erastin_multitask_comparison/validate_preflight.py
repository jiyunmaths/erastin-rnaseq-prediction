#!/usr/bin/env python3
"""Independent linear algebra, autograd, finite-difference and leakage checks."""
import time

import numpy as np
from scipy.linalg import block_diag, lstsq
from sklearn.linear_model import Ridge

import models as models
import run_benchmark as run
from make_splits import inner_rows, outer_split, inner_folds


def independent_linear(x, y, tr, te, params):
    z, zz, target, mask, scaling = models.prepare(x, y, tr, te)
    n, p = z.shape
    t = y.shape[1]
    blocks, rhs = [], []
    for j in range(t):
        factor = np.sqrt(n/mask[:, j].sum())
        blocks.append(factor*np.column_stack([z[mask[:, j]], np.ones(mask[:, j].sum())]))
        rhs.append(factor*target[mask[:, j], j])
    penalty = np.diag(np.r_[np.ones(p), 0.])
    regularizer = np.sqrt(params['alpha'])*np.kron(np.eye(t), penalty)
    coupler = np.sqrt(params['alpha']*params['coupling_ratio'])*np.kron(np.eye(t)-np.ones((t, t))/t, penalty)
    design = np.vstack([block_diag(*blocks), regularizer, coupler])
    rhs = np.r_[np.concatenate(rhs), np.zeros(2*t*(p+1))]
    beta = lstsq(design, rhs, lapack_driver='gelsy')[0].reshape(t, p+1)
    return (np.column_stack([zz, np.ones(len(te))])@beta.T)*scaling['task_scale']+scaling['task_mean']


def autograd(theta, z, target, mask, alpha, weights, h):
    import torch
    torch.set_num_threads(1)
    v = torch.tensor(theta, dtype=torch.float64, requires_grad=True)
    arrays, offset = [], 0
    for shape in models.shapes(z.shape[1], h, target.shape[1]):
        count = int(np.prod(shape))
        arrays.append(v[offset:offset+count].reshape(shape))
        offset += count
    w, b, head, bias = arrays
    pred = torch.tanh(torch.as_tensor(z)@w+b)@head+bias
    factors = mask*weights[None, :]/mask.sum(axis=0)[None, :]
    loss = .5*(torch.as_tensor(factors)*(pred-torch.as_tensor(target))**2).sum()
    loss = loss+.5*alpha/len(z)*((w*w).sum()+(head*head).sum())
    loss.backward()
    return float(loss.detach()), v.grad.detach().numpy()


def main():
    begin = time.monotonic()
    ph = run.prepare()
    run.initialize()
    s = run.STATE
    x, y = s['x'], s['y']
    split_checks = 0
    for r, mask in enumerate(s['test']):
        np.testing.assert_array_equal(mask, outer_split(s['lineage'], s['groups'], s['seeds'][r]))
        np.testing.assert_array_equal(s['inner'][r], inner_folds(s['lineage'], s['groups'], mask, s['inner_seeds'][r]))
        tr, te = np.flatnonzero(~mask), np.flatnonzero(mask)
        assert (len(tr), len(te)) == (595, 149)
        assert set(s['groups'][tr]).isdisjoint(s['groups'][te])
        assert set(s['lineage'][tr]) == set(s['lineage'][te]) == set(s['lineage'])
        split_checks += 1
        for k in range(5):
            a, b = inner_rows(s['inner'][r], k)
            assert set(a)|set(b) == set(tr) and set(s['groups'][a]).isdisjoint(s['groups'][b])
            split_checks += 1
    a, b = inner_rows(s['inner'][0], 0)
    settings = [p for p in models.LINEAR_GRID if p['alpha'] in [models.ALPHAS[0], models.ALPHAS[8], models.ALPHAS[-1]]]
    pp, _, _ = models.linear_path(x, y, a, b, settings)
    lin_error = 0.
    for j, params in enumerate(settings):
        ref = independent_linear(x, y, a, b, params)
        lin_error = max(lin_error, float(np.max(np.abs(ref-pp[j]))))
    assert lin_error < 1e-8
    print('LINEAR_REFERENCE_PASSED', lin_error, flush=True)
    # Zero coupling reproduces the original ridge objective for every alpha.
    zero = models.LINEAR_GRID[:17]
    pp, _, _ = models.linear_path(x, y, a, b, zero)
    z, zz, target, mask, scale = models.prepare(x, y, a, b)
    ridge_error = 0.
    for j, params in enumerate(zero):
        ref = Ridge(alpha=params['alpha'], solver='svd').fit(z, y[a, 0]).predict(zz)
        ridge_error = max(ridge_error, float(np.max(np.abs(ref-pp[j, :, 0]))))
    assert ridge_error < 1e-8
    gradient_checks = []
    for c in [0, 3, 4, 7, 8, 11]:
        params = models.NN_GRID[c]
        t = 1 if params['auxiliary_weight'] == 0 else 4
        weights = np.r_[1., np.full(t-1, params['auxiliary_weight']/3)] if t > 1 else np.ones(1)
        weights /= weights.sum()
        theta = models.initialize(64, params['width'], t, s['nn_seeds'][0, 0])
        args = (z, target[:, :t], mask[:, :t], params['alpha'], weights, params['width'])
        loss, grad = models.objective(theta, *args)
        reference_loss, reference_grad = autograd(theta, *args)
        err = float(np.max(np.abs(grad-reference_grad)))
        assert abs(loss-reference_loss) < 1e-12 and err < 1e-12
        # Check a deterministic spread of parameters with central differences.
        ix = np.linspace(0, len(theta)-1, 25, dtype=int)
        numerical = []
        for j in ix:
            plus, minus = theta.copy(), theta.copy()
            plus[j] += 1e-5
            minus[j] -= 1e-5
            numerical.append((models.objective(plus, *args)[0]-models.objective(minus, *args)[0])/2e-5)
        finite_error = float(np.max(np.abs(np.array(numerical)-grad[ix])))
        assert finite_error < 1e-7
        gradient_checks.append(dict(candidate=c, autograd_max_error=err, finite_difference_max_error=finite_error))
    print('MASKED_NEURAL_GRADIENT_PASSED', flush=True)
    # Invariance under changing every validation task, not just erastin.
    poisoned = y.copy()
    poisoned[b] = 10000.
    p1, _, _ = models.linear_path(x, y, a, b, settings[:1])
    p2, _, _ = models.linear_path(x, poisoned, a, b, settings[:1])
    np.testing.assert_array_equal(p1, p2)
    p1, m1, _ = models.neural_fit(x, y, a, b, models.NN_GRID[8], s['nn_seeds'][0])
    p2, _, _ = models.neural_fit(x, poisoned, a, b, models.NN_GRID[8], s['nn_seeds'][0])
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_allclose(models.predict(m1, x[b]), p1, atol=1e-12, rtol=0)
    # Auxiliary training labels cannot affect the one-head matched control.
    poisoned = y.copy()
    poisoned[a, 1:] *= -3
    p1, _, _ = models.neural_fit(x, y, a, b, models.NN_GRID[0], s['nn_seeds'][0])
    p2, _, _ = models.neural_fit(x, poisoned, a, b, models.NN_GRID[0], s['nn_seeds'][0])
    np.testing.assert_array_equal(p1, p2)
    # Tiny CPU pilot measures cost without using outer-test performance.
    start = time.monotonic()
    for c, params in enumerate(models.NN_GRID):
        _, _, diagnostics = models.neural_fit(x, y, a, b, params, s['nn_seeds'][0])
        assert all(d['success'] for d in diagnostics)
    pilot = time.monotonic()-start
    run.verify_sources()
    v = dict(passed=True, protocol_sha256=ph, split_checks=split_checks, independent_linear_settings=len(settings),
             linear_reference_max_error=lin_error, zero_coupling_ridge_max_error=ridge_error,
             neural_gradient_checks=gradient_checks, all_validation_task_label_invariance=True,
             zero_auxiliary_control_invariance=True, one_inner_fold_neural_pilot_seconds=pilot,
             estimated_full_serial_neural_seconds=pilot*5*20, seconds=time.monotonic()-begin)
    run.dump(run.OUT/'preflight_validation.json', v)
    print('PREFLIGHT_PASSED', v, flush=True)


if __name__ == '__main__':
    main()
