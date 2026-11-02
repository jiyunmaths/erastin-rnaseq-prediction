#!/usr/bin/env python3
"""Fit ensemble weights on inner-validation rows, never outer-test labels."""
import json
import time

import numpy as np

import common as c
from models import fit_weights,value_gradient


def fit_repeat(assay,r,ph):
    root=c.OUT/assay/'folds'/f'ensemble__r{r+1:02d}'
    if root.with_suffix('.json').exists():
        m=json.load(open(root.with_suffix('.json')));assert m['protocol_sha256']==ph and c.sha(root.with_suffix('.npz'))==m['npz_sha256'];return m
    start=time.monotonic();s=c.STATE[assay];inner,outer,valid,settings=c.inputs(assay,r)
    weights,diag=fit_weights(inner,s['y'][valid],s['inner'][r,valid])
    equal=np.ones(3)/3
    losses=np.array([value_gradient(w,inner,s['y'][valid],s['inner'][r,valid])[0] for w in np.eye(3)])
    best=int(np.argmin(losses));tr,te=np.flatnonzero(~s['test'][r]),np.flatnonzero(s['test'][r])
    assert set(np.flatnonzero(valid)).issubset(tr) and not np.any(valid&s['test'][r])
    pred=np.column_stack([outer@equal,outer@weights,outer[:,best]])
    np.savez_compressed(root.with_suffix('.npz'),train_rows=tr,test_rows=te,inner_rows=np.flatnonzero(valid),
        inner_base_predictions=inner,test_base_predictions=outer,weights=weights,equal_weights=equal,
        selected_single_family=np.array(best),ensemble_test_predictions=pred)
    meta=dict(assay=assay,repeat=r+1,protocol_sha256=ph,base_settings=settings,family_order=c.FAMILIES,
        ensemble_order=c.ARMS,learned_weights=weights,equal_weights=equal,selected_single_family=c.FAMILIES[best],
        base_inner_rmse=losses,inner_n=int(valid.sum()),protected_training_n=int(np.sum(s['inner'][r]==-2)),
        train_n=len(tr),test_n=len(te),weight_fit=diag,seconds=time.monotonic()-start,npz_sha256=c.sha(root.with_suffix('.npz')))
    c.dump(root.with_suffix('.json'),meta);return meta


def main():
    begin=time.monotonic();ph=c.prepare();c.verify();p=json.load(open(c.OUT/'preflight_validation.json'));assert p['passed'] and p['protocol_sha256']==ph
    for j,(assay,r) in enumerate([(a,r) for a in c.ASSAYS for r in range(20)],1):
        fit_repeat(assay,r,ph)
        c.dump(c.OUT/'progress.json',dict(phase='ensemble_weight_fitting',completed=j,total=40,elapsed_seconds=time.monotonic()-begin))
    c.verify();c.dump(c.OUT/'completion.json',dict(complete=True,retained_base_models=120,ensemble_recipes=40,ensemble_prediction_bundles=120,seconds=time.monotonic()-begin))
    print('ENSEMBLE_FITS_COMPLETE',time.monotonic()-begin,flush=True)


if __name__=='__main__':main()
