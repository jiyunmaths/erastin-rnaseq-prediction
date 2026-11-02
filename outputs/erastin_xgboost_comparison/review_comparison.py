#!/usr/bin/env python3
"""Reconstruct XGBoost results, audit fitting, and compare matched predictions."""
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import StandardScaler

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent; WS = OUT.parent.parent
sys.path.insert(0, str(OUT))
import run_comparison as run


def module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m


old = module('previous_bn_review', WS/'outputs/erastin_bayesian_comparison/review_bayesian.py')
METRICS = ['spearman', 'rmse', 'r2']
NAMES = {'xgboost': 'XGBoost', 'ridge': 'Ridge', 'neural_network': 'Neural network'}


def table(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT/name, sep='\t', index=False, float_format='%.12g'); return frame


def independent_refit(candidate, seed, x, y, a, b):
    # Independent scaler and sklearn API; selected-prefix refits stop directly
    # at the selected number of trees, not at 600 followed by slicing.
    scaler = StandardScaler().fit(y[a, None])
    params = run.parameters(candidate, seed)
    params.pop('nthread'); params.pop('seed')
    model = xgb.XGBRegressor(**params, n_jobs=1, random_state=int(seed),
                            n_estimators=candidate['num_boost_round'])
    model.fit(x[a], scaler.transform(y[a, None]).ravel())
    return model.predict(x[b]).astype(float)*scaler.scale_[0]+scaler.mean_[0]


def main():
    run.initialize(); s = run.STATE
    x, y, lineage, groups = s['x'], s['y'], s['lineage'], s['groups']
    protocol = json.loads((OUT/'protocol.json').read_text()); ph = run.sha(OUT/'protocol.json')
    assert protocol['script_sha256'] == run.sha(OUT/'run_comparison.py')
    for source in json.loads((OUT/'source_manifest.json').read_text())['files']:
        assert run.sha(WS/source['path']) == source['sha256']
    assert run.sha(OUT/'shared_splits.npz') == run.sha(WS/'outputs/erastin_nn_repeated/shared_splits.npz')
    run.dump(OUT/'analysis_source_manifest.json', {'script_sha256': run.sha(Path(__file__))})
    predictions = np.full((20, 745), np.nan)
    parameters = []; tuning = []; warning_rows = []; reference_checks = []
    pred_error = loss_error = 0.; group_checks = 0
    for r in range(20):
        seen = np.zeros(745, dtype=int)
        for l, label in enumerate(s['labels']):
            token = f'repeat_{r+1:02d}__lineage_{l:02d}'
            root = OUT/'folds'/token
            meta = json.loads(root.with_suffix('.json').read_text())
            data = np.load(root.with_suffix('.npz'))
            assert meta['protocol_sha256'] == ph
            assert meta['array_sha256'] == run.sha(root.with_suffix('.npz'))
            assert meta['model_sha256'] == run.sha(root.with_suffix('.ubj'))
            tr, te = data['train_rows'], data['test_rows']; inner = s['inner'][r, l]
            np.testing.assert_array_equal(tr, np.flatnonzero(inner >= 0))
            np.testing.assert_array_equal(te, np.flatnonzero(lineage == label))
            assert set(groups[tr]).isdisjoint(groups[te]) and label not in set(lineage[tr])
            seen[te] += 1; group_checks += 1
            assert meta['repeat'] == r+1 and meta['lineage'] == label
            assert meta['inner_split_seed'] == int(s['seeds'][r]) and meta['model_seed'] == int(s['model_seeds'][r])
            assert meta['y_mean'] == y[tr].mean() and meta['y_scale'] == y[tr].std()
            candidate = meta['selected_candidate']; selected = run.GRID[candidate]
            assert selected == meta['selected_parameters']
            assert candidate == np.argmin(data['inner_fold_rmse'].mean(axis=1))
            assert np.isfinite(data['inner_validation_predictions'][:, tr]).all()
            assert np.isnan(data['inner_validation_predictions'][:, inner < 0]).all()
            for k in range(5):
                a = tr[inner[tr] != k]; b = tr[inner[tr] == k]
                assert set(a) | set(b) == set(tr) and set(groups[a]).isdisjoint(groups[b])
                group_checks += 1
                scale = meta['inner_scaling'][k]
                assert scale['y_mean'] == y[a].mean() and scale['y_scale'] == y[a].std()
                loss = np.sqrt(np.mean((y[b][None, :]-data['inner_validation_predictions'][:, b])**2, axis=1))
                loss_error = max(loss_error, float(np.max(np.abs(loss-data['inner_fold_rmse'][:, k]))))
            booster = xgb.Booster(); booster.load_model(root.with_suffix('.ubj')); booster.set_param({'nthread': 1})
            assert booster.num_boosted_rounds() == selected['num_boost_round']
            assert booster.num_features() == 40 and booster.feature_names == list(s['genes'])
            p = booster.predict(run.matrix(x[te])).astype(float)*meta['y_scale']+meta['y_mean']
            pred_error = max(pred_error, float(np.max(np.abs(p-data['test_prediction']))))
            predictions[r, te] = p
            parameters.append(dict(repeat=r+1, lineage=str(label), train_n=len(tr), test_n=len(te),
                                   model_seed=meta['model_seed'], candidate=candidate, **selected,
                                   mean_inner_rmse=meta['mean_inner_rmse']))
            tuning.extend(dict(repeat=r+1, lineage=str(label), candidate=c, **params,
                               mean_inner_rmse=float(data['inner_fold_rmse'][c].mean())) for c, params in enumerate(run.GRID))
            warning_rows.extend(dict(repeat=r+1, lineage=str(label), **v) for v in meta['warnings'])
            if r in [0, 19] and label in ['Lung', 'Thyroid']:
                ref = independent_refit(selected, meta['model_seed'], x, y, tr, te)
                outer_error = float(np.max(np.abs(ref-p))); inner_error = 0.
                for k in range(5):
                    a = tr[inner[tr] != k]; b = tr[inner[tr] == k]
                    ref = independent_refit(selected, meta['model_seed'], x, y, a, b)
                    inner_error = max(inner_error, float(np.max(np.abs(ref-data['inner_validation_predictions'][candidate, b]))))
                assert max(outer_error, inner_error) < 1e-7
                reference_checks.append(dict(repeat=r+1, lineage=str(label), candidate=candidate,
                                             outer_api_refit_error=outer_error, inner_prefix_refit_error=inner_error,
                                             inner_refits=5))
        assert np.all(seen == 1)
        if (r+1) % 5 == 0: print(json.dumps(dict(event='audit_progress', complete_repeats=r+1)), flush=True)
    assert pred_error < 1e-12 and loss_error < 1e-12 and np.isfinite(predictions).all()
    assert group_checks == 480*6 and len(reference_checks) == 4
    nn = np.load(WS/'outputs/erastin_nn_repeated/predictions.npz')
    for source_key, current in [('sample_ids', s['ids']), ('group_ids', groups), ('lineage', lineage),
                                ('observed_sensitivity', y), ('inner_split_seeds', s['seeds'])]:
        np.testing.assert_array_equal(nn[source_key], current)
    preds = dict(xgboost=predictions, ridge=nn['ridge'], neural_network=nn['neural_network'])
    scores, folds, macro = old.score_predictions(y, lineage, preds, list(range(1, 21)))
    saved = pd.read_csv(WS/'outputs/erastin_nn_repeated/per_repeat_metrics.tsv', sep='\t')
    for name in ['ridge', 'neural_network']:
        np.testing.assert_allclose(scores[scores.model==name].sort_values('repeat')[METRICS],
                                   saved[saved.model==name].sort_values('repeat')[METRICS], atol=1e-12, rtol=0)
    table('per_repeat_metrics.tsv', scores); table('per_lineage_metrics.tsv', folds)
    table('per_repeat_macro_metrics.tsv', macro); params = table('selected_parameters.tsv', parameters)
    table('tuning_scores.tsv', tuning)
    counts = table('parameter_counts.tsv', params.groupby(['max_depth', 'min_child_weight', 'reg_lambda', 'num_boost_round']).size().reset_index(name='outer_fits'))
    summaries = []
    for scope, frame in [('pooled_745', scores), ('macro_19', macro)]:
        for name in preds:
            row = dict(scope=scope, model=name)
            for metric in METRICS:
                a = frame[frame.model==name][metric].to_numpy()
                row.update({metric+'_mean': a.mean(), metric+'_sd': a.std(ddof=1), metric+'_min': a.min(), metric+'_max': a.max()})
            summaries.append(row)
    summary = table('summary.tsv', summaries)
    bootstrap_check = old.rc.validate_weighted_bootstrap(y, predictions[:2], lineage, old.prior.metrics)
    run.dump(OUT/'progress.json', dict(phase='paired_lineage_bootstrap', models_audited=480))
    draws = old.rc.repeated_group_bootstrap(y, preds, lineage, count=5000, seed=protocol['uncertainty']['pooled_seed'])
    macro_draws = old.bootstrap_macro(folds, list(preds), range(1, 21), protocol['uncertainty']['macro_seed'])
    pairs = [('xgboost', 'ridge'), ('xgboost', 'neural_network'), ('neural_network', 'ridge')]
    paired, seeds = old.paired_comparisons(scores, macro, draws, macro_draws, pairs, list(range(1, 21)))
    paired = table('paired_comparisons.tsv', paired); table('paired_seed_differences.tsv', seeds)
    effects = []; wins = []
    for reference in ['ridge', 'neural_network']:
        for label in s['labels']:
            a = folds[(folds.model=='xgboost') & (folds.lineage==label)].sort_values('repeat')
            b = folds[(folds.model==reference) & (folds.lineage==label)].sort_values('repeat')
            row = dict(reference=reference, lineage=str(label), n=int(np.sum(lineage==label)))
            for metric in METRICS:
                delta = (-1 if metric=='rmse' else 1)*(a[metric].to_numpy()-b[metric].to_numpy())
                row[metric+'_improvement'] = np.mean(delta)
                row[metric+'_favorable_repeats'] = int(np.sum(delta > 0))
            effects.append(row)
        eligible = [r for r in effects if r['reference']==reference and r['n']>=10]
        wins.append(dict(reference=reference, eligible_lineages=19,
                         **{m+'_favorable_lineages': sum(v[m+'_improvement']>0 for v in eligible) for m in METRICS}))
    effects = table('lineage_paired_effects.tsv', effects); table('lineage_win_counts.tsv', wins)
    validation = dict(passed=True, outer_models=480, group_exclusion_checks=group_checks,
        reconstructed_candidate_fold_scores=480*36*5, prediction_max_error=pred_error, inner_loss_max_error=loss_error,
        exact_shared_splits_verified=True, training_only_scaling_verified=True, baseline_predictions_and_scores_verified=True,
        independent_sklearn_api_and_prefix_refits=reference_checks, warning_count=len(warning_rows),
        warnings=warning_rows, bootstrap_checks=bootstrap_check)
    run.dump(OUT/'validation.json', validation)
    np.savez_compressed(OUT/'predictions.npz', sample_ids=s['ids'], group_ids=groups, lineage=lineage, y=y,
                        inner_split_seeds=s['seeds'], model_seeds=s['model_seeds'], **preds)
    np.savez_compressed(OUT/'bootstrap_draws.npz', **{scope+'__'+name+'__'+metric: v
        for scope, all_draws in [('pooled', draws), ('macro', macro_draws)]
        for name, metric_draws in all_draws.items() for metric, v in metric_draws.items()})
    final = dict(summary=summary.to_dict('records'), lineage_wins=wins, parameter_counts=counts.to_dict('records'),
                 validation_passed=True, distinct_prediction_vectors=len(np.unique(predictions, axis=0)),
                 boosting_length_counts=params.num_boost_round.value_counts().to_dict())
    run.dump(OUT/'final_summary.json', final)
    render(summary, paired, effects, counts, validation, final)
    for source in json.loads((OUT/'source_manifest.json').read_text())['files']:
        assert run.sha(WS/source['path']) == source['sha256']
    run.dump(OUT/'progress.json', dict(phase='complete', models_audited=480, validation_passed=True))
    archive = seal()
    print(json.dumps(dict(event='complete', archive_bytes=archive.stat().st_size, summary=run.clean(final)), indent=2), flush=True)


def seal():
    manifest = {'files': [dict(path=str(p.relative_to(OUT)), sha256=run.sha(p), bytes=p.stat().st_size)
                         for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts]}
    run.dump(OUT/'artifact_manifest.json', manifest)
    archive = OUT.parent/'erastin_xgboost_comparison_results.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for r in manifest['files']: z.write(OUT/r['path'], 'erastin_xgboost_comparison/'+r['path'])
        z.write(OUT/'artifact_manifest.json', 'erastin_xgboost_comparison/artifact_manifest.json')
    import hashlib
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for r in manifest['files']:
            assert hashlib.sha256(z.read('erastin_xgboost_comparison/'+r['path'])).hexdigest() == r['sha256']
    return archive


def render(summary, paired, effects, counts, validation, final):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator, StrMethodFormatter
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
    order = ['xgboost', 'neural_network', 'ridge']; colors = ['#bf6d22', '#8853a3', '#137d8b']
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.3), constrained_layout=True, sharex='col')
    for i, scope in enumerate(['pooled_745', 'macro_19']):
        v = summary[summary.scope==scope].set_index('model').loc[order]
        for j, metric in enumerate(['spearman', 'rmse']):
            ax = axes[i, j]
            ax.scatter(v[metric+'_mean'], np.arange(3), s=55, color=colors, zorder=3)
            ax.errorbar(v[metric+'_mean'], np.arange(3), xerr=v[metric+'_sd'], fmt='none', color='#29363b', capsize=3)
            ax.set_yticks(np.arange(3), [NAMES[k] for k in order]); ax.invert_yaxis()
            ax.grid(axis='x', alpha=.2); ax.spines[['top', 'right']].set_visible(False)
            ax.set_title(('Pooled 745 lines' if i==0 else 'Mean across 19 lineages')+' — '+('Spearman ↑' if j==0 else 'RMSE ↓'))
            ax.margins(y=.3)
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            if metric == 'rmse': ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.4f}'))
    fig.suptitle('XGBoost on the 40-gene panel\n20 matched repeats; fixed leave-lineage-out tests', fontsize=13)
    fig.supxlabel('Whiskers show seed SD; repeats reuse the same biological cohort.', fontsize=9)
    fig.savefig(OUT/'model_comparison.png', dpi=170); fig.savefig(OUT/'model_comparison.svg'); plt.close(fig)
    eligible = effects[effects.n>=10]
    labels = eligible.drop_duplicates('lineage').sort_values('n', ascending=False).lineage
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 8), constrained_layout=True, sharey=True)
    for ax, metric in zip(axes, ['spearman', 'rmse']):
        for offset, reference, color in [(-.17, 'ridge', '#137d8b'), (.17, 'neural_network', '#8853a3')]:
            v = eligible[eligible.reference==reference].set_index('lineage').loc[labels]
            ax.barh(np.arange(19)+offset, v[metric+'_improvement'], height=.3, color=color, label='Versus '+NAMES[reference])
        ax.axvline(0, color='#555', lw=1); ax.grid(axis='x', alpha=.2)
        ax.set_xlabel('Spearman increase' if metric=='spearman' else 'RMSE reduction')
        ax.spines[['top', 'right']].set_visible(False)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        if metric == 'rmse': ax.xaxis.set_major_formatter(StrMethodFormatter('{x:.3f}'))
    axes[0].set_yticks(np.arange(19), labels); axes[0].invert_yaxis()
    handles, names = axes[0].get_legend_handles_labels()
    fig.legend(handles, names, loc='outside lower center', ncol=2)
    fig.suptitle('XGBoost improvement within unseen lineages\nMean across 20 repeats; positive favors XGBoost', fontsize=13)
    fig.savefig(OUT/'lineage_comparison.png', dpi=170); fig.savefig(OUT/'lineage_comparison.svg'); plt.close(fig)
    pooled = summary[summary.scope=='pooled_745'].set_index('model')
    excess_ridge = 100*(pooled.loc['xgboost', 'rmse_mean']/pooled.loc['ridge', 'rmse_mean']-1)
    excess_nn = 100*(pooled.loc['xgboost', 'rmse_mean']/pooled.loc['neural_network', 'rmse_mean']-1)
    rows = ['# XGBoost comparison on the 40-gene panel', '',
        '**The tested XGBoost pipeline did not consistently improve on ridge or the neural network.** '
        f'Mean pooled RMSE was {excess_ridge:.2f}% higher than ridge and {excess_nn:.2f}% higher than the neural network. '
        'XGBoost improved pooled RMSE in 2/20 and 1/20 repeats, respectively, and pooled Spearman in 0/20 against either baseline. '
        'All paired lineage-bootstrap intervals include zero, so the observed performance differences remain uncertain. '
        'Equal-lineage Spearman was close to both baselines; equal-lineage RMSE was higher than both. '
        'These results do not justify replacing the existing leading models.', '',
        'Completed 20 matched repeats on 745 cancer cell lines, 734 identity groups, and the same 24 leave-lineage-out tests. '
        'Ridge and neural-network predictions were reused from the exact saved inner split seeds. Scores below are means of per-repeat scores, '
        'not scores after averaging predictions across repeats.', '',
        '| Model | Pooled Spearman ↑ | Pooled RMSE ↓ | Pooled R² ↑ | Macro Spearman ↑ | Macro RMSE ↓ |',
        '|---|---:|---:|---:|---:|---:|']
    for name in order:
        a = summary[(summary.scope=='pooled_745') & (summary.model==name)].iloc[0]
        b = summary[(summary.scope=='macro_19') & (summary.model==name)].iloc[0]
        rows.append(f'| {NAMES[name]} | {a.spearman_mean:.5f} | {a.rmse_mean:.6f} | {a.r2_mean:.5f} | {b.spearman_mean:.5f} | {b.rmse_mean:.6f} |')
    rows += ['', 'Macro scores weight the same 19 lineages with n≥10 equally. All 24 lineages remain in pooled scores.', '',
        f'![Model comparison]({OUT}/model_comparison.png)', '', '## Paired results', '',
        'Positive means XGBoost improves on the reference. Intervals are from 5,000 paired whole-lineage bootstrap resamples.', '',
        '| Scope | Reference | Metric | Mean improvement | Favorable repeats | Conditional 95% interval |',
        '|---|---|---|---:|---:|---|']
    for v in paired[(paired.first_model=='xgboost') & paired.metric.isin(['spearman', 'rmse'])].itertuples():
        rows.append(f'| {v.scope} | {NAMES[v.reference_model]} | {"Spearman increase" if v.metric=="spearman" else "RMSE reduction"} | '
                    f'{v.improvement:+.6f} | {v.favorable_repeats}/20 | [{v.conditional_ci_low:+.6f}, {v.conditional_ci_high:+.6f}] |')
    rows += ['', 'Among the 19 larger lineages, using effects averaged across repeats:', '']
    for v in final['lineage_wins']:
        rows.append(f'- Versus {NAMES[v["reference"]]}: higher Spearman in {v["spearman_favorable_lineages"]}/19; lower RMSE in {v["rmse_favorable_lineages"]}/19.')
    rows += ['', 'Compared with the neural network, Kidney, Liver, Myeloid and Bladder/Urinary Tract had lower RMSE in all 20 repeats. '
        'Bowel, CNS/Brain, Thyroid, Breast, Skin and Bone had higher RMSE in all 20 repeats. '
        'These are descriptive results on the same held-out cell lines, not independently replicated lineage-specific effects.']
    rows += ['', f'![Lineage comparison]({OUT}/lineage_comparison.png)', '', '## Fitting and tuning', '',
        'XGBoost CPU histogram trees used squared-error loss, learning rate 0.03, row and feature subsampling 0.8, '
        '64 histogram bins, and one CPU thread per fit. The frozen grid crossed depth 1/2/3, minimum child weight 5/20, '
        'L2 leaf regularization 1/20, and 100/300/600 boosting rounds (36 candidates). '
        'The minimum equal-weight mean RMSE across the five saved inner folds selected the configuration. '
        'Each 600-tree training path supplied fixed prefixes; the selected length was then refitted on all outer-training rows. '
        'No early stopping or held-out-lineage tuning was used.', '',
        'The 40 log2(TPM+1) features were used unchanged. Response scaling was fitted independently within each current training subset '
        'and predictions were returned to the original negative-AUC scale. Each repeat changes both the saved inner assignments and '
        'the recorded XGBoost row/column sampling seed. Outer holdouts are fixed. Each outer prediction uses one XGBoost model; '
        'the existing neural network uses its unchanged three-member ensemble.', '',
        'The implementation uses the official [XGBoost training API](https://xgboost.readthedocs.io/en/stable/python/python_api.html) '
        'and [tree parameters](https://xgboost.readthedocs.io/en/stable/parameter.html). Installed CPU package version and wheel hash are recorded in the protocol.', '',
        'Selected settings:', '', '| Depth | Minimum child weight | L2 penalty | Boosting rounds | Outer fits |', '|---:|---:|---:|---:|---:|']
    for v in counts.itertuples():
        rows.append(f'| {v.max_depth} | {v.min_child_weight} | {v.reg_lambda:g} | {v.num_boost_round} | {v.outer_fits} |')
    depth_boundary = int(counts.loc[counts.max_depth==3, 'outer_fits'].sum())
    length_boundary = int(counts.loc[counts.num_boost_round==600, 'outer_fits'].sum())
    rows += ['', f'The upper tested depth (3) was selected in {depth_boundary}/480 fits, and the upper boosting length (600) '
        f'in {length_boundary}/480. These counts describe the frozen search limits; deeper trees and other learning rates were not tested. '
        'The grid was not expanded after viewing results.', '', '## Validation', '',
        f'- All 480 saved outer models reproduced their predictions (maximum absolute error {validation["prediction_max_error"]:.3g}).',
        f'- All 86,400 candidate-fold scores were reconstructed; maximum RMSE difference {validation["inner_loss_max_error"]:.3g}.',
        '- Identity exclusion, held-out lineage exclusion, training-only target scaling, and exact split/baseline alignment passed.',
        '- Independent sklearn-API refits checked Lung and Thyroid in repeats 1 and 20, including all five selected inner fits stopped directly at the selected prefix length.',
        f'- Recorded training warnings: {validation["warning_count"]}.',
        '- Frequency-weighted bootstrap calculations were checked against explicit lineage replication and SciPy ranks.', '',
        '## Interpretation limits', '',
        '- This cohort has supported multiple exploratory comparisons. There is no independent drug-screen or clinical validation; intervals are not adjusted for multiple comparisons.',
        '- Seeds are not independent biological replications. Bootstrap intervals condition on saved predictions, do not refit models, and do not account fully for dependence from overlapping training sets.',
        '- Outer tests hold out lineages, while the preserved group-disjoint inner folds mix represented training lineages. Inner tuning is not itself leave-lineage-out.',
        '- Search budgets differ: 36 XGBoost candidates versus four neural-network candidates. The comparison evaluates these specified pipelines, not all configurations of each model family.',
        '- The target remains negative reported CTRP erastin AUC; upstream AUC processing provenance remains unresolved.', '',
        '## Reuse', '',
        'The archive contains the 40-gene input matrix, shared splits, saved UBJ boosters, prediction arrays, selected settings, all candidate validation predictions, and audit tables. '
        'For a saved booster, predict the gene columns in the order of `panel_inputs.npz:genes`, then multiply by the matching metadata `y_scale` and add `y_mean`. '
        'These are evaluation models trained without their respective held-out lineage, not a final model trained on the full cohort.', '',
        f'- [All scores]({OUT}/summary.tsv)', f'- [Paired comparisons]({OUT}/paired_comparisons.tsv)',
        f'- [Per-lineage effects]({OUT}/lineage_paired_effects.tsv)', f'- [Protocol]({OUT}/protocol.json)',
        f'- [Validation]({OUT}/validation.json)', '']
    (OUT/'REVIEW.md').write_text('\n'.join(rows))


if __name__ == '__main__':
    if '--render-only' in sys.argv:
        read = lambda name: pd.read_csv(OUT/name, sep='\t')
        render(read('summary.tsv'), read('paired_comparisons.tsv'), read('lineage_paired_effects.tsv'),
               read('parameter_counts.tsv'), json.loads((OUT/'validation.json').read_text()),
               json.loads((OUT/'final_summary.json').read_text()))
        print(json.dumps(dict(event='rendered_and_resealed', archive_bytes=seal().stat().st_size)))
    else: main()
