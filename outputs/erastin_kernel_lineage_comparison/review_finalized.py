#!/usr/bin/env python3
"""Run the frozen review with exact constant predictions for lineage-only controls.

The frozen fit/prediction implementation uses dense matrix multiplication. Equal
one-hot rows can differ by roughly 1e-16 across BLAS blocks, creating a spurious
correlation within an otherwise constant lineage. Evaluate that control through
one prediction per lineage. No fitted parameter, tuning choice or gene-model
prediction changes. The original frozen scripts remain byte-for-byte intact.
"""
import json
import time

import numpy as np

import review_benchmark as review
import run_benchmark as run

original_predict = review.predict
max_roundoff = 0.
corrected_batches = 0


def exact_control_predict(model, x, lineage):
    global max_roundoff, corrected_batches
    pred = original_predict(model, x, lineage)
    if model['family'] == 'lineage_only':
        before = pred.copy()
        for label in np.unique(lineage):
            rows = np.flatnonzero(lineage == label)
            pred[rows] = pred[rows[0]]
        error = float(np.max(np.abs(before-pred)))
        max_roundoff = max(max_roundoff, error)
        corrected_batches += int(error > 0)
        assert error < 1e-12
    return pred


if __name__ == '__main__':
    assert 'lineage_only' in review.FAMILIES
    review.predict = exact_control_predict
    review.main()
    p = run.OUT
    manifest = dict(created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                    original_review_sha256=run.sha(p/'review_benchmark.py'),
                    finalized_review_sha256=run.sha(__file__),
                    amendment='Analysis-only exact-constant evaluation of lineage-only controls. No refits, grid changes, split changes or gene-model prediction changes.',
                    max_lineage_control_roundoff_removed=max_roundoff, affected_control_outer_batches=corrected_batches,
                    reason='Dense BLAS evaluation produced sub-1e-15 differences between identical one-hot lineage rows. A control with no within-lineage variation must have undefined within-lineage correlations.',
                    fitted_artifacts_unchanged=True)
    run.dump(p/'analysis_source_manifest.json', manifest)
    v = json.load(open(p/'validation.json'))
    v['constant_lineage_control_validation'] = manifest
    run.dump(p/'validation.json', v)
    print('EXACT_CONTROL_REVIEW_COMPLETE', max_roundoff, flush=True)
