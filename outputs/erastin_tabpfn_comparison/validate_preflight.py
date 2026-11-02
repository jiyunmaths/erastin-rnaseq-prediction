#!/usr/bin/env python3
"""Fixed model, group-safe masks, native serialization and query-independence."""
import time
import zipfile

import numpy as np

import common as c
import run_tabpfn as run
from make_splits import inner_rows, outer_split, inner_folds


def main():
    begin = time.monotonic()
    ph = c.prepare()
    run.setup()
    split_checks, checks = 0, []
    for assay, s in c.STATE.items():
        for r, mask in enumerate(s['test']):
            np.testing.assert_array_equal(mask, outer_split(s['lineage'], s['groups'], s['seeds'][r]))
            np.testing.assert_array_equal(s['inner'][r], inner_folds(s['lineage'], s['groups'], mask, s['inner_seeds'][r]))
            tr, te = np.flatnonzero(~mask), np.flatnonzero(mask)
            assert set(s['groups'][tr]).isdisjoint(s['groups'][te])
            assert set(s['lineage'][tr]) == set(s['lineage'][te]) == set(s['lineage'])
            split_checks += 1
            for k in range(5):
                a, b = inner_rows(s['inner'][r], k)
                assert set(a)|set(b) == set(tr) and set(s['groups'][a]).isdisjoint(s['groups'][b])
                split_checks += 1
        # First-repeat contexts are retained as part of the 40-model experiment.
        m = run.fit_job(assay, 0, ph)
        root = c.root_for(assay, 0)
        archive = run.load_fitted_tabpfn_model(root.with_suffix('.tabpfn_fit'), device='cpu')
        te = np.flatnonzero(s['test'][0])
        saved = np.load(root.with_suffix('.npz'))['test_prediction']
        subset = archive.predict(s['x'][te[:5]].copy(), output_type='mean')
        chunk_error = float(np.max(np.abs(subset-saved[:5])))
        assert chunk_error < 5e-5, (assay, chunk_error)
        # Deliberately change other query rows; the first five predictions must
        # not use their values to fit preprocessing/context statistics.
        changed = s['x'][te].copy()
        changed[5:] = changed[5:]*-10.+100.
        other = archive.predict(changed, output_type='mean')
        neighbor_error = float(np.max(np.abs(other[:5]-saved[:5])))
        assert neighbor_error < 5e-5, (assay, neighbor_error)
        assert m['restored_prediction_max_error'] < 2e-6
        with zipfile.ZipFile(root.with_suffix('.tabpfn_fit')) as z:
            assert z.testzip() is None
        checks.append(dict(assay=assay, first_context_retained=True,
                           serialization_max_error=m['restored_prediction_max_error'],
                           query_subset_max_error=chunk_error, changed_query_neighbors_max_error=neighbor_error,
                           feature_coverage=m['feature_coverage'], predict_seconds=m['predict_seconds'],
                           peak_process_rss_kib=m['peak_process_rss_kib']))
        print('PREFLIGHT_ASSAY', assay, checks[-1], flush=True)
    c.verify_sources()
    result = dict(passed=True, protocol_sha256=ph, split_checks=split_checks, checks=checks,
                   fixed_config_no_study_specific_tuning=True, offline_cpu_inference=True,
                   no_validation_targets_passed_to_fit_or_predict=True,
                   training_only_native_preprocessing_source_checked=True,
                   code_pointers=['tabpfn.preprocessing.transform.fit_preprocessing_one',
                                  'tabpfn.preprocessing.ensemble.TabPFNEnsembleMember.transform_X_test'],
                   seconds=time.monotonic()-begin)
    c.dump(c.OUT/'preflight_validation.json', result)
    print('PREFLIGHT_PASSED', result['seconds'], flush=True)


if __name__ == '__main__':
    main()
