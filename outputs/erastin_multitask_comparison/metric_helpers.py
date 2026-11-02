"""Metric and weighted midrank helpers copied unchanged from the previous comparison."""
import numpy as np
from scipy.stats import spearmanr,rankdata

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
