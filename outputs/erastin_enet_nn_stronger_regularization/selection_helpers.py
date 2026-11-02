"""Train-only conditional genome-wide Elastic Net selection, followed by ridge."""
import time
import warnings

import numpy as np
from scipy.linalg import svd
from sklearn.linear_model import ElasticNet
from sklearn.exceptions import ConvergenceWarning
from sklearn.preprocessing import StandardScaler

from ridge_helpers import ALPHAS, ridge_path, predict

L1_RATIOS = [.1, .5, .9, .99]
FRACTIONS = [.5, .2, .1, .05, .02]
TOL = 1e-8
MAX_ITER = 20000
KKT_LIMIT = 2e-6
NONZERO = 1e-8
SELECTORS = [dict(l1_ratio=r, alpha_fraction=f) for r in L1_RATIOS for f in FRACTIONS]
GRID = [dict(selector=-1, ridge_alpha=a) for a in ALPHAS]
GRID += [dict(selector=j, ridge_alpha=a) for j in range(len(SELECTORS)) for a in ALPHAS]


def residual_design(x, y, lineage, core, candidates):
    """Accept only current-training observations; never validation/test data."""
    xx = x[:, candidates].copy()
    xx -= xx.mean(axis=0)
    raw_ss = np.einsum('ij,ij->j', xx, xx)
    z = StandardScaler().fit_transform(x[:, core])
    labels = np.unique(lineage)
    controls = np.column_stack([z, (lineage[:, None] == labels[None, :]).astype(float)])
    controls -= controls.mean(axis=0)
    u, singular, _ = svd(controls, full_matrices=False, check_finite=False)
    rank = singular > singular[0]*max(controls.shape)*np.finfo(float).eps
    q = u[:, rank]
    xx -= q@(q.T@xx)
    yy = y-y.mean()
    yy -= q@(q.T@yy)
    ss = np.einsum('ij,ij->j', xx, xx)
    valid = (raw_ss > 1e-12)&(ss > np.maximum(1e-12, raw_ss*1e-8))
    indices = candidates[valid]
    xx = xx[:, valid]
    xx -= xx.mean(axis=0)
    scale = np.sqrt(np.mean(xx**2, axis=0))
    xx /= scale
    yy -= yy.mean()
    yscale = float(np.sqrt(np.mean(yy**2)))
    yy /= yscale
    xx = np.asfortranarray(xx)
    return xx, yy, indices, dict(training_n=len(y), control_rank=int(rank.sum()), eligible_candidates=len(indices), residual_target_sd=yscale)


def kkt(z, y, beta, alpha, ratio):
    g = z.T@(z@beta-y)/len(y)+alpha*(1-ratio)*beta
    nz = beta != 0
    active = np.max(np.abs(g[nz]+alpha*ratio*np.sign(beta[nz]))) if nz.any() else 0.
    inactive = np.max(np.maximum(np.abs(g[~nz])-alpha*ratio, 0)) if (~nz).any() else 0.
    return float(max(active, inactive))


def working_path(z, target, alphas, ratio):
    """Exact-to-tolerance working sets, with every gene checked at every alpha.

    Restricted coordinate descent accelerates high-p fits. No gene is screened
    out permanently: global KKT violations expand the working set until the
    complete genome-wide problem is certified at the declared tolerance.
    """
    beta = np.zeros(z.shape[1])
    solutions, gaps, iterations, diagnostics = [], [], [], []
    for alpha in alphas:
        active = np.flatnonzero(beta != 0)
        total_iterations, warning_count, refinements = 0, 0, 0
        for step in range(50):
            g = z.T@(z@beta-target)/len(target)+alpha*(1-ratio)*beta
            violation = np.where(beta != 0, np.abs(g+alpha*ratio*np.sign(beta)), np.maximum(np.abs(g)-alpha*ratio,0))
            if violation.max() <= KKT_LIMIT:
                break
            new = np.flatnonzero((beta == 0)&(violation > KKT_LIMIT))
            if len(new):
                take = max(32,len(active))
                new = new[np.argsort(-violation[new],kind='stable')[:take]]
                active = np.union1d(active,new)
            assert len(active)
            zz = np.asfortranarray(z[:,active])
            # A small active Gram avoids a dense 19k-by-19k allocation.
            gram = np.ascontiguousarray(zz.T@zz) if len(active)<=2500 else False
            fitted = ElasticNet(alpha=alpha,l1_ratio=ratio,fit_intercept=False,precompute=gram,
                tol=TOL,max_iter=MAX_ITER,selection='cyclic',warm_start=True)
            fitted.coef_ = beta[active].copy()
            for attempt in range(2):
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter('always')
                    fitted.fit(zz,target)
                total_iterations += fitted.n_iter_
                warning_count += len(caught)
                error = kkt(zz,target,fitted.coef_,alpha,ratio)
                if not any(issubclass(w.category,ConvergenceWarning) for w in caught) and error<=KKT_LIMIT/4:
                    break
                fitted.tol,fitted.max_iter = 1e-10,50000
                refinements += 1
            assert error<=KKT_LIMIT/4, dict(alpha=alpha,ratio=ratio,error=error)
            beta[:] = 0.
            beta[active] = fitted.coef_
        else:
            raise RuntimeError('Working-set global KKT certification did not converge')
        error = kkt(z,target,beta,alpha,ratio)
        assert error<=KKT_LIMIT
        residual = target-z@beta
        # Global dual certificate from sklearn's Elastic Net dual objective.
        l1,l2 = len(target)*alpha*ratio,len(target)*alpha*(1-ratio)
        dual_norm = np.max(np.abs(z.T@residual-l2*beta))
        scale = min(1.,l1/dual_norm) if dual_norm else 1.
        gap = (.5*(residual@residual)*(1+scale**2)+l1*np.abs(beta).sum()
               -scale*(residual@target)+.5*l2*(1+scale**2)*(beta@beta))/len(target)
        assert gap>=-1e-10 and gap<2e-5, gap
        solutions.append(beta.copy());gaps.append(gap);iterations.append(total_iterations)
        diagnostics.append(dict(working_set_size=len(active),working_set_rounds=step,refinements=refinements,warning_count=warning_count))
    return np.array(solutions).T,np.array(gaps),iterations,diagnostics


def select_paths(x, y, lineage, core, candidates, selectors=None):
    """Run all candidate genes jointly; no marginal-response prefilter."""
    begin = time.monotonic()
    selectors = SELECTORS if selectors is None else selectors
    z, target, indices, d = residual_design(x, y, lineage, core, candidates)
    xymax = float(np.max(np.abs(z.T@target))/len(y))
    results, diagnostics = {}, []
    for ratio in sorted(set(v['l1_ratio'] for v in selectors)):
        settings = [(j,v) for j,v in enumerate(selectors) if v['l1_ratio'] == ratio]
        alphas = np.array([xymax/ratio*v['alpha_fraction'] for _,v in settings])
        assert np.all(np.diff(alphas) < 0) or len(alphas) == 1
        aa=alphas
        beta,gaps,iterations,working=working_path(z,target,aa,ratio)
        errors = [kkt(z, target, beta[:,k], aa[k], ratio) for k in range(len(aa))]
        for k, (j, setting) in enumerate(settings):
            nz = np.abs(beta[:,k]) > NONZERO
            results[j] = dict(feature_indices=indices[nz], coefficients=beta[nz,k], alpha=float(aa[k]), **setting)
            diagnostics.append(dict(selector=j, **setting, alpha=float(aa[k]), selected=int(nz.sum()),
                kkt=errors[k], dual_gap=float(gaps[k]), iterations=int(iterations[k]), tol=TOL, **working[k]))
    assert len(results) == len(selectors)
    return [results[j] for j in range(len(selectors))], dict(**d, seconds=time.monotonic()-begin, solutions=diagnostics)
