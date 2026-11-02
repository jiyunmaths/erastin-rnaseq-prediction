#!/usr/bin/env python3
"""Compare fixed TabPFN contexts with verified saved baseline predictions.

Run in the original analysis environment, not the isolated TabPFN environment.
Native TabPFN objects are restored/checked by run_tabpfn.py in their own env.
"""
import argparse
import json
import zipfile
from collections import Counter

import numpy as np
import pandas as pd

import common as c
from make_splits import inner_rows, outer_split, inner_folds
from evaluate import KEYS, score, bootstrap, validate_bootstrap, finite_mean, metrics

OUT = c.OUT
ARM = 'tabpfn35_fast'
LABELS = dict(ridge='Ridge', elastic_net='Elastic Net', neural_network='Neural network',
              svr='RBF SVR', kernel_ridge='Kernel ridge', hierarchical_ridge='Previous random-intercept model',
              matched_intercept='Matched intercept-only model', hierarchical_slopes='Hierarchical gene slopes',
              tabpfn35_fast='TabPFN-3.5-Fast (8 passes)', training_mean='Training mean')


def table(name, values):
    d = values if isinstance(values, pd.DataFrame) else pd.DataFrame(values)
    d.to_csv(OUT/name, sep='\t', index=False, float_format='%.12g')
    return d


def audit(assay):
    s = c.STATE[assay]
    x, y, lin, groups = s['x'], s['y'], s['lineage'], s['groups']
    old = np.load(c.SOURCE/assay/'predictions.npz')
    for key, value in [('sample_ids', s['ids']), ('group_ids', groups), ('lineage', lin), ('y', y), ('outer_test', s['test'])]:
        np.testing.assert_array_equal(old[key], value)
    preds = {f: old[f].copy() for f in c.BASELINES}
    preds.update({ARM: np.full(s['test'].shape, np.nan), 'training_mean': np.full(s['test'].shape, np.nan)})
    rows, warning_rows, reloads = [], [], []
    checks = 0
    for r, mask in enumerate(s['test']):
        tr, te = np.flatnonzero(~mask), np.flatnonzero(mask)
        np.testing.assert_array_equal(mask, outer_split(lin, groups, s['seeds'][r]))
        np.testing.assert_array_equal(s['inner'][r], inner_folds(lin, groups, mask, s['inner_seeds'][r]))
        assert set(groups[tr]).isdisjoint(groups[te]) and set(lin[tr]) == set(lin[te]) == set(lin)
        checks += 1
        for k in range(5):
            a, b = inner_rows(s['inner'][r], k)
            assert set(a)|set(b) == set(tr) and set(groups[a]).isdisjoint(groups[b])
            checks += 1
        root = c.root_for(assay, r)
        m = json.load(open(root.with_suffix('.json')))
        a = np.load(root.with_suffix('.npz'))
        assert m['protocol_sha256'] == c.sha(OUT/'protocol.json')
        assert (m['assay'], m['repeat'], m['model'], m['gene_count']) == (assay, r+1, ARM, 64)
        assert (m['train_n'], m['test_n']) == (len(tr), len(te))
        assert m['train_lineages'] == m['test_lineages'] == len(np.unique(lin))
        assert m['outer_split_seed'] == int(s['seeds'][r]) and m['inner_split_seed'] == int(s['inner_seeds'][r])
        assert m['n_estimators'] == 8 and m['feature_coverage'] == 64
        assert m['no_study_specific_tuning'] and m['offline_inference']
        assert abs(m['native_training_target_mean']-y[tr].mean()) < 1e-7
        assert abs(m['native_training_target_std']-y[tr].std()) < 1e-7
        assert m['versions']['tabpfn'] == '9.0.0'
        for ext in ['npz', 'tabpfn_fit']:
            assert c.sha(root.with_suffix('.'+ext)) == m[ext+'_sha256']
        with zipfile.ZipFile(root.with_suffix('.tabpfn_fit')) as z:
            assert z.testzip() is None
            config = json.loads(z.read('init_params.json'))
            assert config['model_path'] == str(c.CHECKPOINT) and config['device'] == 'cpu'
            assert config['n_estimators'] == 8 and config['tuning_config'] is None
            assert config['categorical_features_indices'] == [] and config['n_preprocessing_jobs'] == 1
        np.testing.assert_array_equal(a['train_rows'], tr)
        np.testing.assert_array_equal(a['test_rows'], te)
        assert a['test_prediction'].shape == (len(te),) and np.isfinite(a['test_prediction']).all()
        preds[ARM][r, te] = a['test_prediction']
        preds['training_mean'][r, te] = y[tr].mean()
        for f in c.BASELINES:
            assert np.isfinite(preds[f][r, te]).all() and np.isnan(preds[f][r, tr]).all()
        if r in [0, 19]:
            assert m['restored_prediction_max_error'] is not None and m['restored_prediction_max_error'] < 2e-6
            reloads.append(dict(repeat=r+1, error=m['restored_prediction_max_error']))
        else:
            assert m['restored_prediction_max_error'] is None
        rows.append(dict(assay=assay, repeat=r+1, train_n=len(tr), test_n=len(te), fit_seconds=m['fit_seconds'],
                         predict_seconds=m['predict_seconds'], elapsed_seconds=m['elapsed_seconds'],
                         peak_process_rss_kib=m['peak_process_rss_kib']))
        warning_rows.extend(dict(assay=assay, repeat=r+1, **w) for w in m['warnings'])
    bv = validate_bootstrap(y, preds, lin, s['test'], 2026112102)
    v = dict(assay=assay, passed=True, contexts=20, reused_baseline_models=160,
             split_checks=checks, all_lineages_on_both_outer_sides=True, fixed_config_no_inner_tuning=True,
             training_only_native_y_statistics_verified=True, all_64_input_features_covered=True,
             archive_checksums_and_crc_verified=20, native_first_last_serialization_checks=reloads,
             native_reload_checks_not_repeated_for_other_contexts=True,
             baseline_predictions_reused_from_verified_unchanged_source=True, bootstrap_arithmetic=bv,
             warning_summary=[dict(category=cat, message=msg, count=n) for (cat, msg), n in Counter((w['category'], w['message']) for w in warning_rows).items()])
    c.dump(OUT/assay/'validation.json', v)
    np.savez_compressed(OUT/assay/'predictions.npz', sample_ids=s['ids'], group_ids=groups, lineage=lin, y=y, outer_test=s['test'], **preds)
    return preds, rows, warning_rows, v


def evaluate(assay, preds, protocol):
    s = c.STATE[assay]
    y, lin, test = s['y'], s['lineage'], s['test']
    pooled, per, macro = score(y, preds, lin, test)
    for d in [pooled, per, macro]:
        d.insert(0, 'assay', assay)
    within = []
    for f, pp in preds.items():
        for r, mask in enumerate(test):
            yc, pc = y[mask].copy(), pp[r, mask].copy()
            for label in np.unique(lin):
                rows = lin[mask] == label
                yc[rows] -= yc[rows].mean()
                pc[rows] = pc[rows]-pc[rows].mean() if np.ptp(pc[rows]) > 0 else 0.
            m = metrics(yc, pc)
            within.append(dict(assay=assay, model=f, repeat=r+1, centered_spearman=m['spearman'], centered_pearson=m['pearson']))
    within = pd.DataFrame(within)
    values, summary = {}, []
    for f in preds:
        values[f] = {}
        for k in ['spearman', 'pearson', 'rmse', 'mae', 'r2']:
            values[f]['pooled_'+k] = pooled[pooled.model == f].sort_values('repeat')[k].to_numpy()
        for k in ['macro_all_rmse', 'macro_large_rmse']:
            values[f][k] = macro[macro.model == f].sort_values('repeat')[k].to_numpy()
        row = dict(assay=assay, model=f)
        for k, v in values[f].items():
            good = v[np.isfinite(v)]
            row.update({k+'_mean': float(finite_mean(v)), k+'_sd': float(good.std(ddof=1)) if len(good) > 1 else np.nan,
                        k+'_defined_repeats': len(good)})
        row['centered_spearman_mean'] = float(finite_mean(within[within.model == f].centered_spearman))
        summary.append(row)
    summary = pd.DataFrame(summary)
    old = pd.read_csv(c.SOURCE/'summary.tsv', sep='\t')
    for f in c.BASELINES:
        a = summary[summary.model == f].iloc[0]
        b = old[(old.assay == assay)&(old.model == f)].iloc[0]
        for k in ['pooled_rmse', 'pooled_spearman', 'pooled_r2', 'macro_all_rmse']:
            assert abs(a[k+'_mean']-b[k+'_mean']) < 1e-11
    c.dump(OUT/'progress.json', dict(phase='paired_bootstrap', assay=assay))
    draws, perdraw = bootstrap(y, preds, lin, test, protocol['uncertainty']['count'], protocol['uncertainty']['seed'])
    paired, effects, wins = [], [], []
    for first, ref in protocol['primary_comparisons']+protocol['secondary_comparisons']:
        for key in KEYS:
            sign = -1 if key.endswith('rmse') or key.endswith('mae') else 1
            delta = sign*(values[first][key]-values[ref][key])
            d = finite_mean(sign*(perdraw[first][key]-perdraw[ref][key]), axis=1)
            good = d[np.isfinite(d)]
            lo, hi = np.percentile(good, [2.5, 97.5]) if len(good) else (np.nan, np.nan)
            paired.append(dict(assay=assay, first_model=first, reference_model=ref, metric=key,
                               improvement=float(finite_mean(delta)), favorable_repeats=int(np.sum(delta > 0)),
                               tied_repeats=int(np.sum(np.abs(delta) <= 1e-12)), defined_repeats=int(np.isfinite(delta).sum()),
                               conditional_ci_low=lo, conditional_ci_high=hi))
        favorable = 0
        for label in np.unique(lin):
            a = per[(per.model == first)&(per.lineage == label)].sort_values('repeat')
            b = per[(per.model == ref)&(per.lineage == label)].sort_values('repeat')
            delta = b.rmse.to_numpy()-a.rmse.to_numpy()
            favorable += int(delta.mean() > 0)
            effects.append(dict(assay=assay, first_model=first, reference_model=ref, lineage=str(label), cohort_n=int(a.cohort_n.iloc[0]),
                                test_n=int(a.test_n.iloc[0]), rmse_improvement=float(delta.mean()), favorable_repeats=int(np.sum(delta > 0))))
        wins.append(dict(assay=assay, first_model=first, reference_model=ref, favorable_lineages=favorable, lineages=len(np.unique(lin))))
    np.savez_compressed(OUT/assay/'bootstrap_draws.npz', **{f+'__'+k: v for f, d in draws.items() for k, v in d.items()})
    return summary, pd.DataFrame(paired), pd.DataFrame(effects), pd.DataFrame(wins), pooled, per, macro, within


def render(summary, paired, wins):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    refs = ['ridge', 'neural_network', 'svr', 'kernel_ridge']
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5))
    for row, assay in enumerate(c.ASSAYS):
        for col, key in enumerate(['pooled_rmse', 'pooled_spearman']):
            ax = axes[row, col]
            for j, ref in enumerate(refs):
                d = paired[(paired.assay == assay)&(paired.reference_model == ref)&(paired.metric == key)].iloc[0]
                ax.plot([d.conditional_ci_low, d.conditional_ci_high], [j, j], color='#28768e', lw=2)
                ax.scatter([d.improvement], [j], color='#173e57', s=30, zorder=3)
            ax.axvline(0, color='#777777', ls='--', lw=1)
            ax.set_yticks(range(4), [LABELS[r] for r in refs] if col == 0 else ['']*4)
            ax.invert_yaxis()
            ax.set_title(assay+' | '+('RMSE reduction' if col == 0 else 'Spearman increase'))
            ax.set_xlabel('TabPFN improvement over reference (positive favors TabPFN)')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(axis='x', alpha=.15)
            ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('TabPFN-3.5-Fast, 64 genes: conditional 95% lineage-bootstrap intervals', fontsize=12)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.text(.5, .01, '20 overlapping holdouts per assay; fixed checkpoint and eight passes. Exploratory intervals preserve all repeats and models.', ha='center', fontsize=8)
    fig.savefig(OUT/'model_comparison.png', dpi=180)
    fig.savefig(OUT/'model_comparison.svg')
    plt.close(fig)
    lines = ['# TabPFN regression on the 64-gene panel', '',
             'Fixed TabPFN-3.5-Fast checkpoint, package 9.0.0, eight ensemble passes, CPU float32 inference. The same 20 lineage-stratified identity-group holdouts per assay are reused, with unweighted gene-only training and no culture or lineage predictors. No study-specific TabPFN hyperparameter tuning was performed; saved baseline nested tuning is unchanged.', '',
             '## Findings', '']
    for assay in c.ASSAYS:
        lines.append('**'+assay+'**')
        for ref in ['neural_network', 'svr', 'kernel_ridge', 'hierarchical_slopes']:
            d = paired[(paired.assay == assay)&(paired.reference_model == ref)&(paired.metric == 'pooled_rmse')].iloc[0]
            base = summary[(summary.assay == assay)&(summary.model == ref)].iloc[0].pooled_rmse_mean
            direction = 'lower' if d.improvement > 0 else 'higher'
            support = 'Conditional interval favors TabPFN.' if d.conditional_ci_low > 0 else 'Conditional interval favors the reference.' if d.conditional_ci_high < 0 else 'Conditional interval includes no difference.'
            w = wins[(wins.assay == assay)&(wins.reference_model == ref)].iloc[0]
            lines.append(f'- Versus {LABELS[ref]}: {abs(100*d.improvement/base):.2f}% {direction} mean RMSE; favorable in {d.favorable_repeats}/20 repeats and {w.favorable_lineages}/{w.lineages} lineages. {support}')
        lines.append('')
    lines += ['## Held-out performance', '', '| Assay | Model | RMSE | Spearman | R2 | Equal-lineage RMSE | Within-lineage centered Spearman |', '|---|---|---:|---:|---:|---:|---:|']
    for assay in c.ASSAYS:
        for f in c.BASELINES+[ARM, 'training_mean']:
            d = summary[(summary.assay == assay)&(summary.model == f)].iloc[0]
            rho = f'{d.pooled_spearman_mean:.4f}' if np.isfinite(d.pooled_spearman_mean) else 'undefined'
            within = f'{d.centered_spearman_mean:.4f}' if np.isfinite(d.centered_spearman_mean) else 'undefined'
            lines.append(f'| {assay} | {LABELS[f]} | {d.pooled_rmse_mean:.6f} | {rho} | {d.pooled_r2_mean:.4f} | {d.macro_all_rmse_mean:.6f} | {within} |')
    lines += ['', 'Values are mean per-repeat scores, not predictions averaged over overlapping outer fits.', '', '## Paired RMSE differences', '',
              '| Assay | Reference | TabPFN RMSE reduction | Conditional 95% interval | Favorable repeats |', '|---|---|---:|---|---:|']
    for _, d in paired[paired.metric == 'pooled_rmse'].iterrows():
        lines.append(f'| {d.assay} | {LABELS[d.reference_model]} | {d.improvement:+.6f} | [{d.conditional_ci_low:+.6f}, {d.conditional_ci_high:+.6f}] | {d.favorable_repeats}/20 |')
    lines += ['', '![TabPFN comparison](model_comparison.png)', '', '## Protocol and validation', '',
              '- CTRP: unchanged 744 lines, 733 identity groups and 23 lineages. PRISM: unchanged 386 lines, 379 groups and 20 lineages. Every evaluated lineage remains in training and testing; identities are disjoint. CTRP Cervix remains excluded and Ampulla handling unchanged.',
              '- Official checkpoint pinned by Hugging Face repository revision and SHA256. CPU-only timing, without accuracy, selected 3.5-Fast over the slower v2 pilot. The version and eight-pass setting were fixed before benchmarking.',
              '- Native feature and response preprocessing fits on outer-training rows only. No external all-cohort scaling. Query-subset and changed-neighbor checks verify that held-out rows do not fit context statistics. Only training responses are passed to fit; predict receives RNA only.',
              '- Fixed configuration has no inner search or temperature calibration. Original inner assignments are preserved and audited but are not used for this untuned model. This differs from the baselines, whose existing nested tuning is reused.',
              '- Forty native fitted-context archives are saved without duplicating the pretrained weights. All archive CRCs, checksums, train/test rows, native target-training statistics and 64-feature coverage were checked. First and last contexts in each assay were restored and their full test predictions reproduced; other archives were integrity-checked, not separately replayed.',
              '- TabPFN runs in an isolated dependency environment. Baseline evaluation uses the original pinned environment. Local weights and blocked network connections ensure inference does not call a hosted prediction service.',
              '- Three hundred twenty baseline model contexts are represented by verified, unchanged source predictions. Baseline summary metrics were reproduced. Whole-lineage paired bootstrap keeps all models and all 20 repeats together.',
              '- Intervals are exploratory, conditional and not multiplicity adjusted; overlapping repeats are not independent cohorts. The repeatedly inspected data need fresh confirmation. No patient or unseen-lineage claim is made.',
              '- Results concern this 3.5-Fast checkpoint, not every TabPFN version or tuning strategy. Pretraining-corpus overlap with these pharmacogenomic datasets has not been independently audited. Original harmonized CTRP AUC provenance remains unresolved. Retained registry unchanged.', '',
              'Sources: [official implementation](https://github.com/PriorLabs/TabPFN), [3.5 model card](https://huggingface.co/Prior-Labs/tabpfn_3_5), [original TabPFN small-data study](https://www.nature.com/articles/s41586-024-08328-6). The original study concerns an earlier model generation. Checkpoint license is saved separately.', '']
    (OUT/'REVIEW.md').write_text('\n'.join(lines))


def seal():
    assert json.load(open(OUT/'visual_qa.json'))['passed']
    c.verify_sources()
    for name, expected in json.load(open(OUT/'analysis_source_manifest.json'))['scripts'].items():
        assert c.sha(OUT/name) == expected
    c.dump(OUT/'progress.json', dict(phase='complete', validation_passed=True, contexts=40, baseline_models=320))
    files = [dict(path=str(p.relative_to(OUT)), sha256=c.sha(p), bytes=p.stat().st_size)
             for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'artifact_manifest.json' and '__pycache__' not in p.parts and p.suffix != '.log']
    c.dump(OUT/'artifact_manifest.json', dict(files=files))
    print('SEALED', len(files), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    if args.seal:
        seal()
        return
    c.initialize()
    c.verify_sources()
    assert json.load(open(OUT/'fit_completion.json'))['complete']
    c.dump(OUT/'analysis_source_manifest.json', dict(scripts={n: c.sha(OUT/n) for n in ['review_benchmark.py', 'evaluate.py', 'metric_helpers.py']}))
    protocol = json.load(open(OUT/'protocol.json'))
    results, runtime, warnings, validation = [], [], [], []
    for assay in c.ASSAYS:
        pred, rr, ww, vv = audit(assay)
        results.append(evaluate(assay, pred, protocol))
        runtime.extend(rr)
        warnings.extend(ww)
        validation.append(vv)
    names = ['summary.tsv', 'paired_comparisons.tsv', 'lineage_paired_effects.tsv', 'lineage_win_counts.tsv',
             'per_repeat_pooled_metrics.tsv', 'per_lineage_metrics.tsv', 'per_repeat_macro_metrics.tsv', 'within_lineage_metrics.tsv']
    frames = [table(n, pd.concat([r[j] for r in results], ignore_index=True)) for j, n in enumerate(names)]
    table('runtime_metrics.tsv', runtime)
    table('inference_warnings.tsv', warnings)
    c.dump(OUT/'validation.json', dict(passed=True, assays=validation, preflight=json.load(open(OUT/'preflight_validation.json')),
                                      contexts=40, baseline_models=320, checkpoint_sha256_verified=True,
                                      source_hashes_verified=True, fixed_configuration_no_tuning=True, retained_registry_unchanged=True))
    render(frames[0], frames[1], frames[3])
    c.verify_sources()
    c.dump(OUT/'final_summary.json', dict(validation_passed=True, summary=frames[0].to_dict('records')))
    c.dump(OUT/'progress.json', dict(phase='analysis_complete_pending_visual_qa', validation_passed=True))
    print('ANALYSIS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
