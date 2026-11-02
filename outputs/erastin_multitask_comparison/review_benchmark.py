#!/usr/bin/env python3
"""Audit inductive multi-task fitting and evaluate erastin-only improvements."""
import argparse
import json
from collections import Counter

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

import models
import run_benchmark as run
from make_splits import inner_rows, outer_split, inner_folds
from validate_preflight import independent_linear
from evaluate import KEYS, score, bootstrap, validate_bootstrap, finite_mean, metrics

OUT = run.OUT
LABELS = {'ridge': 'Retained ridge', 'elastic_net': 'Retained Elastic Net', 'neural_network': 'Retained neural network',
          'svr': 'Single-task RBF SVR', 'kernel_ridge': 'Single-task kernel ridge',
          'multitask_linear': 'Coupled multi-task linear', 'matched_single_nn': 'Matched single-task NN',
          'multitask_nn': 'Multi-task NN (zero-weight fallback allowed)', 'positive_aux_nn': 'Multi-task NN (positive auxiliary weight)',
          'training_mean': 'Training mean'}


def table(name, values):
    d = values if isinstance(values, pd.DataFrame) else pd.DataFrame(values)
    d.to_csv(OUT/name, sep='\t', index=False, float_format='%.12g')
    return d


def baseline_predict(model, f, x):
    if f == 'ridge':
        return ((x-model['mean'])/model['scale'])@model['coef']+model['intercept_after_scaling']
    z = model['x_scaler'].transform(x)
    sy = model['y_scaler']
    if f == 'elastic_net':
        return sy.inverse_transform(model['model'].predict(z)[:, None]).ravel()
    if f == 'neural_network':
        return np.mean([sy.inverse_transform(m.predict(z)[:, None]).ravel() for m in model['members']], axis=0)
    p = model['parameters']
    if f == 'svr':
        pred = model['estimator'].predict(rbf_kernel(z, model['train_z'], gamma=p['gamma']))
    else:
        w = p['rbf_weight']
        q = (1-w)*(z@model['train_z'].T)/64+w*rbf_kernel(z, model['train_z'], gamma=p['gamma'])
        q = q-model['kernel_mean'][None, :]-q.mean(axis=1)[:, None]+model['kernel_grand']
        pred = q@model['dual']
    return sy.inverse_transform(pred[:, None]).ravel()


def audit():
    run.initialize()
    run.verify_sources()
    assert json.load(open(OUT/'fit_completion.json'))['complete']
    protocol = json.load(open(OUT/'protocol.json'))
    s = run.STATE
    x, y, lin = s['x'], s['y'], s['lineage']
    old = np.load(run.KERNEL/'CTRP/predictions.npz')
    np.testing.assert_array_equal(old['y'], y[:, 0])
    for key, val in [('sample_ids', s['ids']), ('group_ids', s['groups']), ('lineage', lin), ('outer_test', s['test'])]:
        np.testing.assert_array_equal(old[key], val)
    preds = {f: np.full(s['test'].shape, np.nan) for f in run.BASELINES+models.ARMS+['training_mean']}
    params, tuning, aux_metrics, objective_checks, references, task_scalers = [], [], [], [], [], []
    all_convergence = []
    max_pred = max_loss = 0.
    score_checks = split_checks = 0
    for r, mask in enumerate(s['test']):
        tr, te = np.flatnonzero(~mask), np.flatnonzero(mask)
        np.testing.assert_array_equal(mask, outer_split(lin, s['groups'], s['seeds'][r]))
        np.testing.assert_array_equal(s['inner'][r], inner_folds(lin, s['groups'], mask, s['inner_seeds'][r]))
        assert set(s['groups'][tr]).isdisjoint(s['groups'][te]) and set(lin[tr]) == set(lin[te]) == set(lin)
        split_checks += 1
        completed = json.load(open(OUT/'folds'/f'repeat_{r+1:02d}_complete.json'))
        assert completed['protocol_sha256'] == run.sha(OUT/'protocol.json')
        for row in completed['files']:
            assert run.sha(OUT/row['path']) == row['sha256']
        preds['training_mean'][r, te] = y[tr, 0].mean()
        for f in run.BASELINES:
            root = (run.SOURCE/'folds' if f in run.BASELINES[:3] else run.KERNEL/'CTRP/folds')/f'{f}__p64__r{r+1:02d}'
            a = np.load(root.with_suffix('.npz'))
            np.testing.assert_array_equal(a['train_rows'], tr)
            np.testing.assert_array_equal(a['test_rows'], te)
            p = baseline_predict(joblib.load(root.with_suffix('.joblib')), f, x[te])
            np.testing.assert_allclose(p, a['test_prediction'], atol=1e-10, rtol=0)
            np.testing.assert_allclose(p, old[f][r, te], atol=1e-10, rtol=0)
            preds[f][r, te] = p
        t = np.load(OUT/'folds'/f'tuning__r{r+1:02d}.npz')
        td = json.load(open(OUT/'folds'/f'tuning__r{r+1:02d}.json'))
        assert td['protocol_sha256'] == run.sha(OUT/'protocol.json')
        for d in td['neural_diagnostics']:
            all_convergence.extend(d['members'])
        for family, grid in [('linear', models.LINEAR_GRID), ('neural', models.NN_GRID)]:
            val, loss = t[family+'_validation_predictions'], t[family+'_fold_rmse']
            assert np.isfinite(val[:, s['inner'][r] >= 0]).all() and np.isnan(val[:, s['inner'][r] < 0]).all()
            for k in range(5):
                a, b = inner_rows(s['inner'][r], k)
                assert set(a)|set(b) == set(tr) and set(s['groups'][a]).isdisjoint(s['groups'][b])
                assert set(lin[a]) == set(lin)
                split_checks += 1
                ref = np.sqrt(np.mean((val[:, b]-y[b, 0][None, :])**2, axis=1))
                err = float(np.max(np.abs(ref-loss[:, k])))
                assert err < 1e-12
                max_loss = max(max_loss, err)
                score_checks += len(grid)
            tuning.extend(dict(family=family, repeat=r+1, candidate=j, mean_erastin_inner_rmse=float(loss[j].mean()), **p) for j, p in enumerate(grid))
        choices = dict(multitask_linear=int(np.argmin(t['linear_fold_rmse'].mean(axis=1))),
                       matched_single_nn=int(np.argmin(t['neural_fold_rmse'][:4].mean(axis=1))),
                       multitask_nn=int(np.argmin(t['neural_fold_rmse'].mean(axis=1))),
                       positive_aux_nn=int(np.argmin(t['neural_fold_rmse'][4:].mean(axis=1)))+4)
        for arm in models.ARMS:
            root = run.root_for(arm, r)
            meta = json.load(open(root.with_suffix('.json')))
            a = np.load(root.with_suffix('.npz'))
            model = joblib.load(root.with_suffix('.joblib'))
            assert meta['protocol_sha256'] == run.sha(OUT/'protocol.json') and meta['repeat'] == r+1
            assert meta['arm'] == arm and meta['selected_candidate'] == choices[arm]
            grid = models.LINEAR_GRID if arm == 'multitask_linear' else models.NN_GRID
            loss = t['linear_fold_rmse'] if arm == 'multitask_linear' else t['neural_fold_rmse']
            assert meta['selected_parameters'] == grid[choices[arm]] == model['parameters']
            assert abs(meta['mean_inner_rmse']-loss[choices[arm]].mean()) < 1e-12
            np.testing.assert_array_equal(meta['neural_seeds'], s['nn_seeds'][r])
            np.testing.assert_array_equal(a['train_rows'], tr)
            np.testing.assert_array_equal(a['test_rows'], te)
            np.testing.assert_array_equal(meta['train_task_counts'], np.isfinite(y[tr]).sum(axis=0))
            sx = StandardScaler().fit(x[tr])
            np.testing.assert_allclose(model['x_scaler'].mean_, sx.mean_, atol=1e-12, rtol=0)
            np.testing.assert_allclose(model['x_scaler'].scale_, sx.scale_, atol=1e-12, rtol=0)
            for key, values in [('task_mean', np.nanmean(y[tr], axis=0)), ('task_scale', np.nanstd(y[tr], axis=0)),
                                ('task_counts', np.isfinite(y[tr]).sum(axis=0))]:
                np.testing.assert_allclose(model[key], values, atol=1e-12, rtol=0)
            p = models.predict(model, x[te])
            err = float(np.max(np.abs(p-a['all_task_predictions'])))
            max_pred = max(max_pred, err)
            assert err < 1e-10
            np.testing.assert_array_equal(p[:, 0], a['test_prediction'])
            preds[arm][r, te] = p[:, 0]
            if arm != 'multitask_linear':
                z, _, target, observed, scaling = models.prepare(x, y, tr, te)
                nt = model['n_tasks']
                for member, d in zip(model['members'], meta['outer_diagnostics']):
                    val, grad = models.objective(member['theta'], z, target[:, :nt], observed[:, :nt],
                                                  model['parameters']['alpha'], model['task_weights'], model['parameters']['width'])
                    assert abs(val-d['loss']) < 1e-12 and d['success']
                    objective_checks.append(dict(model=arm, repeat=r+1, seed=member['seed'], loss=val,
                                                  gradient_infinity_norm=float(np.max(np.abs(grad)))))
                    all_convergence.append(d)
            if r in [0, 19]:
                if arm == 'multitask_linear':
                    ref = independent_linear(x, y, tr, te, meta['selected_parameters'])
                else:
                    ref, _, _ = models.neural_fit(x, y, tr, te, meta['selected_parameters'], s['nn_seeds'][r])
                error = float(np.max(np.abs(ref-p)))
                inner_error = 0.
                for k in range(5):
                    aa, bb = inner_rows(s['inner'][r], k)
                    if arm == 'multitask_linear':
                        ref = independent_linear(x, y, aa, bb, meta['selected_parameters'])
                    else:
                        ref, _, _ = models.neural_fit(x, y, aa, bb, meta['selected_parameters'], s['nn_seeds'][r])
                    saved = t['linear_validation_predictions'] if arm == 'multitask_linear' else t['neural_validation_predictions']
                    inner_error = max(inner_error, float(np.max(np.abs(ref[:, 0]-saved[choices[arm], bb]))))
                assert max(error, inner_error) < 1e-8
                references.append(dict(model=arm, repeat=r+1, outer_error=error, inner_error=inner_error))
            for j in range(p.shape[1]):
                available = np.isfinite(y[te, j])
                aux_metrics.append(dict(model=arm, repeat=r+1, task=run.TASKS[j][0], **metrics(y[te[available], j], p[available, j])))
            params.append(dict(model=arm, repeat=r+1, candidate=choices[arm], mean_inner_rmse=meta['mean_inner_rmse'], **meta['selected_parameters']))
            task_scalers.extend(dict(model=arm, repeat=r+1, task=run.TASKS[j][0], training_n=int(model['task_counts'][j]),
                                     training_mean=float(model['task_mean'][j]), training_scale=float(model['task_scale'][j])) for j in range(4))
        # A selected zero-weight neural candidate must equal its matched control.
        if choices['multitask_nn'] < 4:
            np.testing.assert_array_equal(preds['multitask_nn'][r], preds['matched_single_nn'][r])
        if (r+1) % 5 == 0:
            print('AUDITED_REPEATS', r+1, flush=True)
    assert all(d['success'] for d in all_convergence)
    unique_outer_ensembles = sum(json.load(open(OUT/'folds'/f'repeat_{r+1:02d}_complete.json'))['unique_outer_neural_ensembles'] for r in range(20))
    bv = validate_bootstrap(y[:, 0], preds, lin, s['test'], 2026111902)
    v = dict(passed=True, new_outer_bundles=80, reused_outer_models=100, candidate_fold_scores_checked=score_checks,
             group_exclusion_checks=split_checks, all_tasks_excluded_on_heldout_identity_groups=True,
             training_only_task_standardization=True, no_response_imputation=True, shared_64_genes=True,
             unweighted_cells_within_each_task=True, culture_covariates=False,
             max_prediction_error=max_pred, max_inner_rmse_error=max_loss,
             neural_inner_member_fits=3600, all_neural_inner_and_outer_stopping_conditions_passed=True,
             unique_neural_outer_ensembles=unique_outer_ensembles, unique_neural_outer_member_fits=3*unique_outer_ensembles,
             reported_neural_outer_arm_bundles=60, stopping_message_counts_include_duplicate_reported_outer_bundles=True,
             max_recorded_gradient_infinity_norm=max(d['gradient_infinity_norm'] for d in all_convergence),
             stopping_messages=dict(Counter(d['message'] for d in all_convergence)), independent_or_replay_checks=references,
             bootstrap_arithmetic=bv, preflight=json.load(open(OUT/'preflight_validation.json')),
             baseline_models_and_predictions_reproduced=True, source_hashes_verified=True)
    table('selected_parameters.tsv', params)
    table('tuning_scores.tsv', tuning)
    table('task_prediction_metrics.tsv', aux_metrics)
    table('neural_objective_checks.tsv', objective_checks)
    table('task_scalers.tsv', task_scalers)
    run.dump(OUT/'validation.json', v)
    np.savez_compressed(OUT/'predictions.npz', sample_ids=s['ids'], group_ids=s['groups'], lineage=lin, y=y[:, 0], outer_test=s['test'], **preds)
    return preds, pd.DataFrame(params), v, protocol


def evaluate(preds, params, validation, protocol):
    s = run.STATE
    y, lin, test = s['y'][:, 0], s['lineage'], s['test']
    pooled, per, macro = score(y, preds, lin, test)
    table('per_repeat_pooled_metrics.tsv', pooled)
    table('per_lineage_metrics.tsv', per)
    table('per_repeat_macro_metrics.tsv', macro)
    values, summary = {}, []
    for f in preds:
        values[f] = {}
        for k in ['spearman', 'pearson', 'rmse', 'mae', 'r2']:
            values[f]['pooled_'+k] = pooled[pooled.model == f].sort_values('repeat')[k].to_numpy()
        for k in ['macro_all_rmse', 'macro_large_rmse']:
            values[f][k] = macro[macro.model == f].sort_values('repeat')[k].to_numpy()
        row = dict(model=f)
        for k, val in values[f].items():
            good = val[np.isfinite(val)]
            row.update({k+'_mean': float(finite_mean(val)), k+'_sd': float(good.std(ddof=1)) if len(good) > 1 else np.nan,
                        k+'_defined_repeats': len(good)})
        summary.append(row)
    summary = table('summary.tsv', summary)
    old = pd.read_csv(run.KERNEL/'summary.tsv', sep='\t')
    for f in run.BASELINES:
        a, b = summary[summary.model == f].iloc[0], old[(old.assay == 'CTRP')&(old.model == f)].iloc[0]
        for k in ['pooled_rmse', 'pooled_spearman', 'pooled_r2', 'macro_all_rmse']:
            assert abs(a[k+'_mean']-b[k+'_mean']) < 1e-11
    run.dump(OUT/'progress.json', dict(phase='paired_bootstrap', audited_models=180))
    draws, perdraw = bootstrap(y, preds, lin, test, protocol['uncertainty']['count'], protocol['uncertainty']['seed'])
    paired, effects, wins = [], [], []
    for first, ref in protocol['primary_comparisons']+protocol['secondary_comparisons']:
        for key in KEYS:
            sign = -1 if key.endswith('rmse') or key.endswith('mae') else 1
            delta = sign*(values[first][key]-values[ref][key])
            dd = finite_mean(sign*(perdraw[first][key]-perdraw[ref][key]), axis=1)
            good = dd[np.isfinite(dd)]
            lo, hi = np.percentile(good, [2.5, 97.5]) if len(good) else (np.nan, np.nan)
            paired.append(dict(first_model=first, reference_model=ref, metric=key, improvement=float(finite_mean(delta)),
                               favorable_repeats=int(np.sum(delta > 0)), defined_repeats=int(np.isfinite(delta).sum()),
                               tied_repeats=int(np.sum(np.abs(delta) <= 1e-12)), worse_repeats=int(np.sum(delta < -1e-12)),
                               conditional_ci_low=lo, conditional_ci_high=hi))
        favorable = 0
        for label in np.unique(lin):
            a = per[(per.model == first)&(per.lineage == label)].sort_values('repeat')
            b = per[(per.model == ref)&(per.lineage == label)].sort_values('repeat')
            delta = b.rmse.to_numpy()-a.rmse.to_numpy()
            favorable += int(delta.mean() > 0)
            effects.append(dict(first_model=first, reference_model=ref, lineage=str(label), cohort_n=int(a.cohort_n.iloc[0]),
                                test_n=int(a.test_n.iloc[0]), rmse_improvement=float(delta.mean()), favorable_repeats=int(np.sum(delta > 0))))
        wins.append(dict(first_model=first, reference_model=ref, favorable_lineages=favorable, lineages=23))
    paired = table('paired_comparisons.tsv', paired)
    table('lineage_paired_effects.tsv', effects)
    wins = table('lineage_win_counts.tsv', wins)
    counts = params.groupby(['model', 'candidate']).size().reset_index(name='outer_fits')
    table('parameter_counts.tsv', counts)
    selection = dict(neural_positive_auxiliary_selected_fits=int((params[(params.model == 'multitask_nn')].auxiliary_weight > 0).sum()),
                     linear_positive_coupling_selected_fits=int((params[(params.model == 'multitask_linear')].coupling_ratio > 0).sum()),
                     positive_aux_nn_auxiliary_weights=params[params.model == 'positive_aux_nn'].auxiliary_weight.value_counts().to_dict(),
                     post_result_grid_expansion=False)
    run.dump(OUT/'grid_boundary_audit.json', selection)
    np.savez_compressed(OUT/'bootstrap_draws.npz', **{f+'__'+k: v for f, d in draws.items() for k, v in d.items()})
    render(summary, paired, wins, selection)
    run.verify_sources()
    run.dump(OUT/'final_summary.json', dict(validation_passed=True, summary=summary.to_dict('records'), selection=selection,
                                           new_outer_bundles=80, reused_outer_models=100))
    run.dump(OUT/'progress.json', dict(phase='analysis_complete_pending_visual_qa', validation_passed=True))
    print('ANALYSIS_COMPLETE', flush=True)


def render(summary, paired, wins, selection):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    cases = [('multitask_linear', 'ridge', 'Coupled linear vs ridge'),
             ('multitask_nn', 'matched_single_nn', 'Joint NN vs matched single NN'),
             ('multitask_nn', 'neural_network', 'Joint NN vs retained NN'),
             ('positive_aux_nn', 'matched_single_nn', 'Positive-aux NN vs matched single NN')]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7))
    for col, key in enumerate(['pooled_rmse', 'pooled_spearman']):
        ax = axes[col]
        for j, (first, ref, label) in enumerate(cases):
            d = paired[(paired.first_model == first)&(paired.reference_model == ref)&(paired.metric == key)].iloc[0]
            ax.plot([d.conditional_ci_low, d.conditional_ci_high], [j, j], color='#28768e', lw=2)
            ax.scatter([d.improvement], [j], color='#173e57', s=32, zorder=3)
        ax.axvline(0, color='#777777', ls='--', lw=1)
        ax.set_yticks(range(len(cases)), [v[2] for v in cases] if col == 0 else ['']*len(cases))
        ax.invert_yaxis()
        ax.set_title('Erastin RMSE reduction' if col == 0 else 'Erastin Spearman increase')
        ax.set_xlabel('Improvement (positive favors multi-task model)')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.grid(axis='x', alpha=.15)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('CTRP 64-gene multi-task learning: mean effects and conditional 95% intervals', fontsize=12)
    fig.tight_layout(rect=(0, .05, 1, .94))
    fig.text(.5, .012, 'Paired bootstrap of whole lineages; all 20 overlapping holdouts preserved. Exploratory, conditional uncertainty.', ha='center', fontsize=8)
    fig.savefig(OUT/'model_comparison.png', dpi=180)
    fig.savefig(OUT/'model_comparison.svg')
    plt.close(fig)
    findings = []
    for first, ref, label in cases:
        d = paired[(paired.first_model == first)&(paired.reference_model == ref)&(paired.metric == 'pooled_rmse')].iloc[0]
        b = summary[summary.model == ref].iloc[0].pooled_rmse_mean
        direction = 'lower' if d.improvement > 0 else 'higher' if d.improvement < 0 else 'unchanged'
        support = 'The conditional interval favors improvement.' if d.conditional_ci_low > 0 else 'The conditional interval favors worse performance.' if d.conditional_ci_high < 0 else 'The conditional interval includes no difference.'
        ties = int(d.tied_repeats) if 'tied_repeats' in paired.columns else 0
        findings.append(f'- **{label}:** {abs(100*d.improvement/b):.2f}% {direction} mean RMSE; favorable in {d.favorable_repeats}/20 repeats, tied in {ties}/20. {support}')
    lines = ['# Multi-task learning across ferroptosis inducers', '',
             'Same 744 CTRP lines, 733 identity groups, 23 lineages, 64 genes, 20 saved lineage-stratified holdouts and five inner folds. All held-out identity groups are excluded from every drug-training task. No culture or lineage predictors; no new auxiliary-only cell lines.', '',
             'Targets are separate negative CTRP AUCs for erastin, 1S,3R-RSL-3, ML162 and ML210. Observed label counts are 744/721/721/690. Missing labels are masked; all task means and scales use only observed training labels. Only erastin inner-validation RMSE selects settings. Inference requires RNA only.', '',
             '## Findings', '', *findings, '',
             f"The neural selector used zero auxiliary weight in {20-selection['neural_positive_auxiliary_selected_fits']}/20 repeats. This tested configuration does not establish an erastin prediction gain from the related-drug responses. Keep the retained models unchanged; this does not rule out other multi-task formulations or additional training cohorts.", '',
             'The matched neural comparison isolates the effect of adding auxiliary losses within this implementation. Comparison with the retained NN also includes the effect of tighter optimizer settings. Statistical intervals remain conditional and exploratory.', '',
             '## Held-out erastin performance', '',
             '| Model | RMSE | Spearman | R2 | Equal-lineage RMSE |', '|---|---:|---:|---:|---:|']
    for f in run.BASELINES+models.ARMS+['training_mean']:
        d = summary[summary.model == f].iloc[0]
        rho = f'{d.pooled_spearman_mean:.4f}' if np.isfinite(d.pooled_spearman_mean) else 'undefined'
        lines.append(f'| {LABELS[f]} | {d.pooled_rmse_mean:.6f} | {rho} | {d.pooled_r2_mean:.4f} | {d.macro_all_rmse_mean:.6f} |')
    lines += ['', 'Values are means across holdouts, not predictions averaged across repeats.', '', '## Paired RMSE differences', '',
              '| Candidate | Reference | RMSE reduction | Relative reduction | Conditional 95% interval | Favorable repeats | Favorable lineages |',
              '|---|---|---:|---:|---|---:|---:|']
    for _, d in paired[paired.metric == 'pooled_rmse'].iterrows():
        base = summary[summary.model == d.reference_model].iloc[0].pooled_rmse_mean
        w = wins[(wins.first_model == d.first_model)&(wins.reference_model == d.reference_model)].iloc[0]
        lines.append(f'| {LABELS[d.first_model]} | {LABELS[d.reference_model]} | {d.improvement:+.6f} | {100*d.improvement/base:+.2f}% | [{d.conditional_ci_low:+.6f}, {d.conditional_ci_high:+.6f}] | {d.favorable_repeats}/20 | {w.favorable_lineages}/23 |')
    lines += ['', 'Positive improvement favors the candidate. Whole-lineage bootstrap intervals retain all repeats together, are conditional on this cohort/fits and are not multiplicity adjusted. Overlapping repeats are not independent cohorts.', '',
              '![Multi-task comparison](model_comparison.png)', '', '## Sharing selected by inner validation', '',
              f"- Neural model selected positive auxiliary weight in {selection['neural_positive_auxiliary_selected_fits']}/20 outer repeats; zero weight reproduces its matched single-task control.",
              f"- Coupled linear model selected positive coupling in {selection['linear_positive_coupling_selected_fits']}/20 repeats; zero coupling reproduces the original ridge objective.",
              '- The positive-auxiliary neural arm always shares, providing a prespecified sensitivity analysis independent of the fallback choice.', '',
              '## Model specification and validation', '',
              '- Linear model: separate drug intercepts and slopes, with a penalty shrinking drug slopes toward their shared mean. No false claim that independent multi-output ridge shares information.',
              '- Neural model: one shared tanh hidden layer (16/32 units), separate drug output heads, alpha 10/100, auxiliary weight 0/0.25/1 and three saved-seed initializations. Each observed response contributes equally within its task. Task weights and label-count normalization are explicit in protocol.json.',
              '- The matched neural control uses the same implementation, optimizer, hidden/primary initialization and erastin objective; the retained neural network is also shown because its previous optimizer settings differ.',
              '- All saved baseline/new predictions, scalers, selected settings and candidate/fold RMSEs were audited. Neural gradients were checked against PyTorch autograd and central differences. Coupled linear predictions were checked against independent augmented least squares; zero coupling against native ridge.',
              '- Poisoning validation responses for every drug leaves training predictions unchanged; poisoning auxiliary training labels leaves the matched single-task control unchanged.',
              '- Eighty new outer model bundles and 100 unchanged baseline models are included. Identical selected neural ensembles can be reused across reported arms; counts refer to saved arm/repeat bundles.',
              '- CTRP only: none of these three auxiliary compounds occurs in the downloaded PRISM secondary-treatment metadata. Cross-assay auxiliary learning would be a separate experiment.',
              '- These repeatedly inspected splits provide exploratory evidence, with no claim about patients or unseen lineages. The original harmonized CTRP AUC provenance limitation remains. The retained registry is unchanged.', '',
              'Multi-task rationale: [Gonen and Margolin](https://academic.oup.com/bioinformatics/article/30/17/i556/200902). Distinct erastin/GPX4-inhibitor vulnerabilities support separate outputs rather than merged labels: [Lei et al.](https://pubmed.ncbi.nlm.nih.gov/38552003/).', '']
    (OUT/'REVIEW.md').write_text('\n'.join(lines))


def seal():
    assert json.load(open(OUT/'visual_qa.json'))['passed']
    run.verify_sources()
    am = json.load(open(OUT/'analysis_source_manifest.json'))
    for name, expected in am['scripts'].items():
        assert run.sha(OUT/name) == expected
    run.dump(OUT/'progress.json', dict(phase='complete', validation_passed=True, new_outer_bundles=80, reused_outer_models=100))
    files = [dict(path=str(p.relative_to(OUT)), sha256=run.sha(p), bytes=p.stat().st_size)
             for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'artifact_manifest.json' and '__pycache__' not in p.parts and p.suffix != '.log']
    run.dump(OUT/'artifact_manifest.json', dict(files=files))
    print('SEALED', len(files), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    if args.seal:
        seal()
        return
    names = ['review_benchmark.py', 'evaluate.py', 'metric_helpers.py', 'weighted_mlp.py']
    run.dump(OUT/'analysis_source_manifest.json', dict(scripts={n: run.sha(OUT/n) for n in names}))
    preds, params, validation, protocol = audit()
    evaluate(preds, params, validation, protocol)


if __name__ == '__main__':
    main()
