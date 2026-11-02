"""Unchanged XGBoost settings from the verified earlier comparison."""
import itertools

STRUCTURES = [dict(max_depth=d, min_child_weight=w, reg_lambda=l)
              for d, w, l in itertools.product([1, 2, 3], [5, 20], [1., 20.])]
ROUNDS = [100, 300, 600]
GRID = [dict(**p, num_boost_round=n) for p in STRUCTURES for n in ROUNDS]
FIXED = dict(objective='reg:squarederror', booster='gbtree', tree_method='hist',
             device='cpu', nthread=1, eta=.03, subsample=.8, colsample_bytree=.8,
             sampling_method='uniform', max_bin=64, reg_alpha=0., gamma=0.,
             base_score=0., seed_per_iteration=False, validate_parameters=True,
             verbosity=1)

def parameters(candidate, seed):
    return dict(FIXED, **{k: v for k, v in candidate.items() if k != 'num_boost_round'}, seed=int(seed))
