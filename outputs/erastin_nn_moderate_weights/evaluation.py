"""Evaluation routines copied from the verified earlier comparisons; see helper_sources.json."""
import json
import numpy as np
import pandas as pd
from scipy.stats import spearmanr,rankdata
METRICS=["spearman","rmse","r2"]


def metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    mse = np.mean((y-prediction)**2)
    den = np.sum((y-y.mean())**2)
    rho = float(spearmanr(y, prediction).statistic) if np.ptp(y) > 0 and np.ptp(prediction) > 0 else np.nan
    return {'n': len(y), 'spearman': rho, 'rmse': float(np.sqrt(mse)),
            'r2': float(1-len(y)*mse/den) if den > 0 else np.nan}

def weighted_ranker(values):
    """Exact midranks after integer-frequency replication, including ties."""
    order = np.argsort(values, kind='stable')
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1])+1]
    inverse = np.argsort(order)
    group_index = np.cumsum(np.r_[True, sorted_values[1:] != sorted_values[:-1]])-1
    def rank(weights):
        group_weight = np.add.reduceat(weights[:, order], starts, axis=1)
        mid = np.cumsum(group_weight, axis=1) - group_weight/2 + 0.5
        return mid[:, group_index][:, inverse]
    return rank

def repeated_group_bootstrap(y, preds, groups, *, count, seed, batch_size=100):
    """Bootstrap identities, keeping all paired models and repeats together.

    Statistics are means of per-repeat metrics, not scores after pooling the
    20 x 745 predictions as if they were independent observations.
    """
    names = list(preds)
    repeats = next(iter(preds.values())).shape[0]
    unique, group_index = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    draw = {name: {m: np.empty(count) for m in ['spearman','rmse','r2']} for name in names}
    yranker = weighted_ranker(y)
    prankers = {name: [weighted_ranker(row) for row in preds[name]] for name in names}
    for offset in range(0, count, batch_size):
        b = min(batch_size, count-offset)
        sampled = rng.integers(len(unique), size=(b, len(unique)))
        group_weights = np.array([np.bincount(row, minlength=len(unique)) for row in sampled], dtype=float)
        weights = group_weights[:, group_index]
        totals = weights.sum(axis=1)
        yr = yranker(weights)
        yc = yr-(totals[:,None]+1)/2
        y_rank_ss = (weights*yc**2).sum(axis=1)
        y_mean = (weights*y).sum(axis=1)/totals
        y_ss = (weights*(y[None,:]-y_mean[:,None])**2).sum(axis=1)
        for name in names:
            acc = {m: np.zeros(b) for m in ['spearman','rmse','r2']}
            for repeat in range(repeats):
                prediction = preds[name][repeat]
                pr = prankers[name][repeat](weights)
                pc = pr-(totals[:,None]+1)/2
                numerator = (weights*yc*pc).sum(axis=1)
                denominator = np.sqrt(y_rank_ss*(weights*pc**2).sum(axis=1))
                rho = np.divide(numerator, denominator, out=np.full(b,np.nan), where=denominator!=0)
                sse = weights @ ((y-prediction)**2)
                acc['spearman'] += rho/repeats
                acc['rmse'] += np.sqrt(sse/totals)/repeats
                acc['r2'] += (1-sse/y_ss)/repeats
            for metric in acc:
                draw[name][metric][offset:offset+b] = acc[metric]
        print(json.dumps({'event':'bootstrap_progress','completed':offset+b,'total':count}),flush=True)
    return draw

def validate_weighted_bootstrap(y, predictions, groups, metrics):
    # Independently expand sampled groups and compare rank statistics with scipy.
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(407)
    rank_error = 0.0
    for _ in range(5):
        wgroup = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
        w = wgroup[inverse]
        rows = np.repeat(np.arange(len(y)), w)
        for v in [y, predictions[0], np.round(predictions[0], 2)]:
            expected = rankdata(v[rows])
            actual = weighted_ranker(v)(w[None,:])[0][rows]
            rank_error = max(rank_error, float(np.max(np.abs(expected-actual))))
            assert np.array_equal(actual, expected)
    small = {'first':predictions[:2], 'identical':predictions[:2].copy()}
    result = repeated_group_bootstrap(y,small,groups,count=5,seed=908,batch_size=5)
    rng = np.random.default_rng(908)
    members = [np.flatnonzero(groups==g) for g in unique]
    max_error = 0.0
    for b in range(5):
        rows = np.concatenate([members[i] for i in rng.integers(len(unique),size=len(unique))])
        reference = [metrics(y[rows],p[rows]) for p in predictions[:2]]
        for metric in ['spearman','rmse','r2']:
            expected = np.mean([r[metric] for r in reference])
            err = abs(result['first'][metric][b]-expected)
            max_error=max(max_error,float(err));assert err<1e-12
            assert result['first'][metric][b]==result['identical'][metric][b]
    return {'weighted_midranks_exact_vs_expanded_scipy_ranks':True,
            'max_rank_error':rank_error,'bootstrap_max_metric_error_vs_expansion':max_error,
            'identical_predictions_have_exact_zero_paired_difference':True}

def bootstrap_macro(folds,names,repeats,seed):
    labels=sorted(folds[folds.test_n>=10].lineage.unique())
    draws=np.random.default_rng(seed).integers(len(labels),size=(5000,len(labels)))
    result={}
    for model in names:
        # Explicit array mean preserves undefined Spearman for any constant
        # predictor; never silently drop a difficult lineage from macro scores.
        v=np.array([[folds[(folds.model==model)&(folds.repeat==r)&(folds.lineage==l)][METRICS].iloc[0].to_numpy(float)
                     for l in labels] for r in repeats]).mean(axis=0)
        result[model]={metric:v[draws,j].mean(axis=1) for j,metric in enumerate(METRICS)}
    return result

def paired_comparisons(scores,macro,draws,macro_draws,pairs,repeats):
    rows=[];seed_rows=[]
    for scope,frame,boot in [('pooled_745',scores,draws),('macro_19',macro,macro_draws)]:
        for a,b in pairs:
            aa=frame[(frame.model==a)&frame.repeat.isin(repeats)].sort_values('repeat')
            bb=frame[(frame.model==b)&frame.repeat.isin(repeats)].sort_values('repeat')
            assert np.array_equal(aa.repeat.to_numpy(),bb.repeat.to_numpy())
            for metric in METRICS:
                sign=-1 if metric=='rmse' else 1
                delta=sign*(aa[metric].to_numpy()-bb[metric].to_numpy())
                lo,hi=np.percentile(sign*(boot[a][metric]-boot[b][metric]),[2.5,97.5])
                rows.append({'scope':scope,'first_model':a,'reference_model':b,'metric':metric,'improvement':np.mean(delta),
                    'favorable_repeats':int(np.sum(delta>0)),'repeats':len(repeats),'conditional_ci_low':lo,'conditional_ci_high':hi})
                seed_rows.extend({'scope':scope,'first_model':a,'reference_model':b,'metric':metric,'repeat':int(r),'improvement':v}
                                 for r,v in zip(aa.repeat,delta))
    return rows,seed_rows

def score_predictions(y,lineage,predictions,repeats):
    scores=[];folds=[];macro=[]
    for name,pp in predictions.items():
        assert len(pp)==len(repeats)
        for repeat,p in zip(repeats,pp):
            scores.append({'repeat':repeat,'model':name,**metrics(y,p)})
            per=[]
            for label in np.unique(lineage):
                ix=lineage==label
                met=metrics(y[ix],p[ix])
                folds.append({'repeat':repeat,'model':name,'lineage':label,'test_n':int(ix.sum()),**met})
                if ix.sum()>=10:per.append([met[m] for m in METRICS])
            assert len(per)==19
            macro.append({'repeat':repeat,'model':name,**dict(zip(METRICS,np.asarray(per).mean(axis=0))),
                          'lineages':19,'defined_spearman_lineages':int(np.isfinite(np.array(per)[:,0]).sum())})
    return pd.DataFrame(scores),pd.DataFrame(folds),pd.DataFrame(macro)
