#!/usr/bin/env python3
"""Audit saved fits, compare paired holdouts, and render the benchmark report."""
import argparse
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import run_benchmark as run
from models import FAMILIES, GRIDS, predict
from make_splits import inner_rows, outer_split, inner_folds
from validate_preflight import independent
from evaluate import KEYS, score, bootstrap, validate_bootstrap, finite_mean, metrics

OUT = run.OUT
LABELS = {'ridge': 'Ridge', 'elastic_net': 'Elastic Net', 'neural_network': 'Neural network',
          'svr': 'RBF SVR', 'kernel_ridge': 'Kernel ridge', 'hierarchical_ridge': 'Hierarchical ridge',
          'lineage_only': 'Lineage only', 'training_mean': 'Training mean'}


def table(name, data):
    d = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    d.to_csv(OUT/name, sep='\t', index=False, float_format='%.12g')
    return d


def baseline_predict(m, family, x):
    if family == 'ridge':
        return ((x-m['mean'])/m['scale'])@m['coef']+m['intercept_after_scaling']
    z = m['x_scaler'].transform(x)
    sy = m['y_scaler']
    if family == 'elastic_net':
        return sy.inverse_transform(m['model'].predict(z)[:, None]).ravel()
    return np.mean([sy.inverse_transform(n.predict(z)[:, None]).ravel() for n in m['members']], axis=0)


def audit(assay, protocol):
    s = run.STATE[assay]
    source = run.SOURCES[assay]
    old = np.load(source/'predictions.npz')
    x, y, lineage, groups = s['x'], s['y'], s['lineage'], s['groups']
    for key, value in [('sample_ids', s['ids']), ('group_ids', groups), ('lineage', lineage), ('y', y), ('outer_test', s['test'])]:
        np.testing.assert_array_equal(old[key], value)
    preds = {f: np.full(s['test'].shape, np.nan) for f in run.BASELINES+FAMILIES+['training_mean']}
    params, tuning, independent_checks, warnings = [], [], [], []
    scores_checked = group_checks = 0
    max_prediction_error = max_score_error = max_normal = 0.
    for r, mask in enumerate(s['test']):
        np.testing.assert_array_equal(mask, outer_split(lineage, groups, s['seeds'][r]))
        np.testing.assert_array_equal(s['inner'][r], inner_folds(lineage, groups, mask, s['inner_seeds'][r]))
        tr, te = np.flatnonzero(~mask), np.flatnonzero(mask)
        assert set(groups[tr]).isdisjoint(groups[te]) and set(lineage[tr]) == set(lineage[te]) == set(lineage)
        group_checks += 1
        preds['training_mean'][r, te] = y[tr].mean()
        for f in run.BASELINES:
            root = source/'folds'/f'{f}__p64__r{r+1:02d}'
            a = np.load(root.with_suffix('.npz'))
            m = joblib.load(root.with_suffix('.joblib'))
            np.testing.assert_array_equal(a['train_rows'], tr)
            np.testing.assert_array_equal(a['test_rows'], te)
            p = baseline_predict(m, f, x[te])
            np.testing.assert_allclose(p, a['test_prediction'], atol=1e-10, rtol=0)
            np.testing.assert_allclose(p, old[f+'__64'][r, te], atol=1e-10, rtol=0)
            preds[f][r, te] = p
        for f in FAMILIES:
            root = run.root_for(assay, f, r)
            meta = json.load(open(root.with_suffix('.json')))
            a = np.load(root.with_suffix('.npz'))
            model = joblib.load(root.with_suffix('.joblib'))
            assert meta['protocol_sha256'] == run.sha(OUT/'protocol.json')
            assert (meta['assay'], meta['family'], meta['panel'], meta['repeat']) == (assay, f, 64, r+1)
            assert (meta['train_n'], meta['test_n']) == (len(tr), len(te))
            assert meta['outer_split_seed'] == int(s['seeds'][r]) and meta['inner_split_seed'] == int(s['inner_seeds'][r])
            for ext in ['npz', 'joblib']:
                assert run.sha(root.with_suffix('.'+ext)) == meta[ext+'_sha256']
            np.testing.assert_array_equal(a['train_rows'], tr)
            np.testing.assert_array_equal(a['test_rows'], te)
            c = int(np.argmin(a['inner_fold_rmse'].mean(axis=1)))
            assert c == meta['selected_candidate'] and GRIDS[f][c] == meta['selected_parameters'] == model['parameters']
            assert abs(meta['mean_inner_rmse']-a['inner_fold_rmse'][c].mean()) < 1e-12
            assert np.isfinite(a['inner_validation_predictions'][:, s['inner'][r] >= 0]).all()
            assert np.isnan(a['inner_validation_predictions'][:, s['inner'][r] < 0]).all()
            for k in range(5):
                aa, bb = inner_rows(s['inner'][r], k)
                assert set(aa)|set(bb) == set(tr) and set(groups[aa]).isdisjoint(groups[bb])
                assert set(lineage[aa]) == set(lineage)
                group_checks += 1
                loss = np.sqrt(np.mean((a['inner_validation_predictions'][:, bb]-y[bb][None, :])**2, axis=1))
                error = float(np.max(np.abs(loss-a['inner_fold_rmse'][:, k])))
                max_score_error = max(max_score_error, error)
                assert error < 1e-12
                scores_checked += len(GRIDS[f])
            for diag in meta['fit_diagnostics']:
                aa = tr if diag['stage'] == 'outer' else inner_rows(s['inner'][r], diag['fold'])[0]
                sx, sy = StandardScaler().fit(x[aa]), StandardScaler().fit(y[aa, None])
                np.testing.assert_allclose(diag['x_mean'], sx.mean_, atol=1e-12, rtol=0)
                np.testing.assert_allclose(diag['x_scale'], sx.scale_, atol=1e-12, rtol=0)
                assert diag['y_mean'] == sy.mean_[0] and diag['y_scale'] == sy.scale_[0]
                max_normal = max(max_normal, diag['normal_equation_max_residual'])
                warnings.extend(dict(assay=assay, family=f, repeat=r+1, stage=diag['stage'], **w) for w in diag['warnings'])
            for scaler, data in [('x_scaler', x[tr]), ('y_scaler', y[tr, None])]:
                reference = StandardScaler().fit(data)
                for attr in ['mean_', 'scale_']:
                    np.testing.assert_allclose(getattr(model[scaler], attr), getattr(reference, attr), atol=1e-12, rtol=0)
            p = predict(model, x[te], lineage[te])
            error = float(np.max(np.abs(p-a['test_prediction'])))
            max_prediction_error = max(max_prediction_error, error)
            assert error < 1e-10
            preds[f][r, te] = p
            if r in [0, 19]:
                ref = independent(f, meta['selected_parameters'], x, y, lineage, tr, te)
                outer_error = float(np.max(np.abs(ref-p)))
                inner_error = 0.
                for k in range(5):
                    aa, bb = inner_rows(s['inner'][r], k)
                    ref = independent(f, meta['selected_parameters'], x, y, lineage, aa, bb)
                    inner_error = max(inner_error, float(np.max(np.abs(ref-a['inner_validation_predictions'][c, bb]))))
                assert max(inner_error, outer_error) < 2e-6, (assay, f, r, inner_error, outer_error)
                independent_checks.append(dict(family=f, repeat=r+1, outer_error=outer_error, inner_error=inner_error))
            params.append(dict(assay=assay, model=f, repeat=r+1, selected_candidate=c, mean_inner_rmse=meta['mean_inner_rmse'], **meta['selected_parameters']))
            tuning.extend(dict(assay=assay, model=f, repeat=r+1, candidate=j, mean_inner_rmse=float(a['inner_fold_rmse'][j].mean()), **p) for j, p in enumerate(GRIDS[f]))
        if (r+1) % 5 == 0:
            print('AUDITED', assay, r+1, flush=True)
    assert max_normal < 1e-7
    bv = validate_bootstrap(y, preds, lineage, s['test'], 2026111802)
    validation = dict(assay=assay, passed=True, new_outer_models=80, reused_outer_models=60,
                      all_lineages_in_both_outer_sets=True, source_split_generator_reproduced=True,
                      group_exclusion_checks=group_checks, candidate_fold_scores_checked=scores_checked,
                      max_prediction_error=max_prediction_error, max_score_error=max_score_error,
                      normal_equation_max_residual=max_normal, independent_selected_refits=independent_checks,
                      warning_summary=[dict(category=c, message=m, count=n) for (c, m), n in Counter((w['category'], w['message']) for w in warnings).items()],
                      training_only_preprocessing_verified=True, baseline_models_and_predictions_reproduced=True,
                      bootstrap_arithmetic=bv)
    run.dump(OUT/assay/'validation.json', validation)
    np.savez_compressed(OUT/assay/'predictions.npz', sample_ids=s['ids'], group_ids=groups, lineage=lineage, y=y, outer_test=s['test'], **preds)
    return preds, params, tuning, validation


def evaluate(assay, preds, protocol):
    s = run.STATE[assay]
    y, lin, test = s['y'], s['lineage'], s['test']
    pooled, per, macro = score(y, preds, lin, test)
    for d in [pooled, per, macro]:
        d.insert(0, 'assay', assay)
    # Evaluation-only centering removes between-lineage means from both axes.
    centered = []
    for f, pp in preds.items():
        for r, mask in enumerate(test):
            yc, pc = y[mask].copy(), pp[r, mask].copy()
            for label in np.unique(lin):
                take = lin[mask] == label
                yc[take] -= yc[take].mean()
                pc[take] = pc[take]-pc[take].mean() if np.ptp(pc[take]) > 0 else 0.
            m = metrics(yc, pc)
            centered.append(dict(assay=assay, model=f, repeat=r+1, centered_spearman=m['spearman'], centered_pearson=m['pearson']))
    centered = pd.DataFrame(centered)
    values, summary = {}, []
    for f in preds:
        values[f] = {}
        for key in ['spearman', 'pearson', 'rmse', 'mae', 'r2']:
            values[f]['pooled_'+key] = pooled[pooled.model == f].sort_values('repeat')[key].to_numpy()
        for key in ['macro_all_rmse', 'macro_large_rmse']:
            values[f][key] = macro[macro.model == f].sort_values('repeat')[key].to_numpy()
        row = dict(assay=assay, model=f, gene_count=0 if f in ['lineage_only', 'training_mean'] else 64,
                   lineage_predictor=f in ['hierarchical_ridge', 'lineage_only'])
        for key, v in values[f].items():
            good = v[np.isfinite(v)]
            row.update({key+'_mean': float(finite_mean(v)), key+'_sd': float(good.std(ddof=1)) if len(good) > 1 else np.nan,
                        key+'_defined_repeats': len(good)})
        for key in ['centered_spearman', 'centered_pearson']:
            row[key+'_mean'] = float(finite_mean(centered[centered.model == f][key]))
        summary.append(row)
    summary = pd.DataFrame(summary)
    old = pd.read_csv(run.SOURCES[assay]/'summary.tsv', sep='\t').set_index('model')
    for f in run.BASELINES:
        a, b = summary[summary.model == f].iloc[0], old.loc[f+'__64']
        for key in ['pooled_spearman', 'pooled_rmse', 'pooled_r2', 'macro_all_rmse']:
            sourcekey = 'macro23_rmse' if key == 'macro_all_rmse' and assay == 'CTRP' else key
            assert abs(a[key+'_mean']-b[sourcekey+'_mean']) < 1e-11
    run.dump(OUT/'progress.json', dict(phase='paired_bootstrap', assay=assay))
    draws, perdraw = bootstrap(y, preds, lin, test, protocol['uncertainty']['count'], protocol['uncertainty']['seed'])
    pairs = protocol['primary_comparisons']+protocol['secondary_comparisons']
    paired, effects, wins = [], [], []
    for first, ref in pairs:
        for key in KEYS:
            sign = -1 if key.endswith('rmse') or key.endswith('mae') else 1
            delta = sign*(values[first][key]-values[ref][key])
            d = finite_mean(sign*(perdraw[first][key]-perdraw[ref][key]), axis=1)
            valid = d[np.isfinite(d)]
            low, high = np.percentile(valid, [2.5, 97.5]) if len(valid) else (np.nan, np.nan)
            paired.append(dict(assay=assay, first_model=first, reference_model=ref, metric=key,
                               improvement=float(finite_mean(delta)), favorable_repeats=int(np.sum(delta > 0)),
                               defined_repeats=int(np.isfinite(delta).sum()), conditional_ci_low=low, conditional_ci_high=high))
        favorable = 0
        for label in np.unique(lin):
            a = per[(per.model == first)&(per.lineage == label)].sort_values('repeat')
            b = per[(per.model == ref)&(per.lineage == label)].sort_values('repeat')
            delta = b.rmse.to_numpy()-a.rmse.to_numpy()
            favorable += int(delta.mean() > 0)
            effects.append(dict(assay=assay, first_model=first, reference_model=ref, lineage=str(label),
                                cohort_n=int(a.cohort_n.iloc[0]), test_n=int(a.test_n.iloc[0]), rmse_improvement=delta.mean(),
                                favorable_repeats=int(np.sum(delta > 0))))
        wins.append(dict(assay=assay, first_model=first, reference_model=ref, favorable_lineages=favorable, lineages=len(np.unique(lin))))
    np.savez_compressed(OUT/assay/'bootstrap_draws.npz', **{f+'__'+k: v for f, d in draws.items() for k, v in d.items()})
    return summary, pd.DataFrame(paired), pd.DataFrame(effects), pd.DataFrame(wins), pooled, per, macro, centered


def boundary_audit(params):
    records = []
    for assay in run.SOURCES:
        for f in FAMILIES:
            rows = params[(params.assay == assay)&(params.model == f)]
            for key in GRIDS[f][0]:
                levels = sorted({p[key] for p in GRIDS[f] if p[key] is not None})
                if not levels:
                    continue
                chosen = rows[key]
                records.append(dict(assay=assay, model=f, parameter=key, minimum=levels[0], maximum=levels[-1],
                                    lower_boundary_fits=int((chosen == levels[0]).sum()), upper_boundary_fits=int((chosen == levels[-1]).sum()),
                                    null_fallback_fits=int(chosen.isna().sum())))
    return table('grid_boundary_audit.tsv', records)


def render(summary, paired, wins, validation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    families = FAMILIES[:3]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 5.8))
    for row, assay in enumerate(run.SOURCES):
        for col, key in enumerate(['pooled_rmse', 'pooled_spearman']):
            ax = axes[row, col]
            for j, f in enumerate(families):
                d = paired[(paired.assay == assay)&(paired.first_model == f)&(paired.reference_model == 'ridge')&(paired.metric == key)].iloc[0]
                ax.plot([d.conditional_ci_low, d.conditional_ci_high], [j, j], color='#28768e', lw=2)
                ax.scatter([d.improvement], [j], color='#173e57', s=32, zorder=3)
            ax.axvline(0, color='#777777', ls='--', lw=1)
            ax.set_yticks(range(3), [LABELS[f] for f in families])
            ax.invert_yaxis()
            ax.set_title(assay+' | '+('RMSE reduction' if col == 0 else 'Spearman increase'))
            ax.set_xlabel('Improvement over ridge (positive favors candidate)')
            ax.grid(axis='x', alpha=.15)
            ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('64-gene benchmark: mean effects and conditional 95% lineage-bootstrap intervals', fontsize=12)
    fig.tight_layout(rect=(0, .03, 1, .95))
    fig.text(.5, .008, '20 overlapping holdouts per assay; intervals are exploratory and conditional on the saved cohort and fitted models.', ha='center', fontsize=8)
    fig.savefig(OUT/'model_comparison.png', dpi=180)
    fig.savefig(OUT/'model_comparison.svg')
    plt.close(fig)
    lines = ['# Kernel and lineage benchmark on the 64-gene panel', '',
             'Completed separate CTRP and PRISM comparisons using the exact saved 20 lineage-stratified identity-group holdouts and five inner folds. Training is unweighted; culture variables are excluded. All evaluated lineages occur in outer training and testing. No retained model or registry was replaced.', '',
             '## Held-out performance', '',
             'Values are means across repeats, not predictions averaged across fitted models. Lower RMSE and higher Spearman/R2 are better. The response scale differs between assays.', '',
             '| Assay | Model | RMSE | Spearman | R2 | Equal-lineage RMSE | Within-lineage centered Spearman |',
             '|---|---|---:|---:|---:|---:|---:|']
    for assay in run.SOURCES:
        for f in run.BASELINES+FAMILIES+['training_mean']:
            d = summary[(summary.assay == assay)&(summary.model == f)].iloc[0]
            cs = f'{d.centered_spearman_mean:.4f}' if np.isfinite(d.centered_spearman_mean) else 'undefined'
            lines.append(f'| {assay} | {LABELS[f]} | {d.pooled_rmse_mean:.6f} | {d.pooled_spearman_mean:.4f} | {d.pooled_r2_mean:.4f} | {d.macro_all_rmse_mean:.6f} | {cs} |')
    lines += ['', '## Paired RMSE comparison', '',
              'Positive reduction favors the new model. Intervals resample entire lineage blocks while keeping all models and repeats paired; they are not 20-independent-cohort confidence intervals and are not multiplicity adjusted.', '',
              '| Assay | Candidate | Reference | RMSE reduction | Relative reduction | Conditional 95% interval | Favorable repeats | Favorable lineages |',
              '|---|---|---|---:|---:|---|---:|---:|']
    for _, d in paired[paired.metric == 'pooled_rmse'].iterrows():
        base = summary[(summary.assay == d.assay)&(summary.model == d.reference_model)].iloc[0].pooled_rmse_mean
        w = wins[(wins.assay == d.assay)&(wins.first_model == d.first_model)&(wins.reference_model == d.reference_model)].iloc[0]
        lines.append(f'| {d.assay} | {LABELS[d.first_model]} | {LABELS[d.reference_model]} | {d.improvement:+.6f} | {100*d.improvement/base:+.2f}% | [{d.conditional_ci_low:+.6f}, {d.conditional_ci_high:+.6f}] | {d.favorable_repeats}/20 | {w.favorable_lineages}/{w.lineages} |')
    lines += ['', '![Paired model comparison](model_comparison.png)', '', '## Validation and interpretation', '',
              '- 160 new saved outer models (including 40 lineage-only controls) and 120 unchanged retained models were audited.',
              '- Every saved outer prediction was reproduced from its fitted model. Every new candidate/fold RMSE and selected candidate was recomputed.',
              '- Independent references use native RBF SVR, native kernel ridge with KernelCenterer, and augmented least squares for the hierarchical model. All first-fold candidates plus selected first/last-repeat inner and outer fits were checked.',
              '- All outer/inner splits were reproduced from the original generators; identity groups remain disjoint, including always-inner-training CTRP Ampulla.',
              '- Lineage offsets are jointly fitted Gaussian random-intercept MAP estimates with tuned penalties; gene slopes are shared across lineages. This comparison does not test lineage-specific gene slopes or unseen lineages.',
              '- The lineage-only model measures tissue-baseline information. Within-lineage centered correlation subtracts each held-out lineage mean from both observations and predictions for evaluation only; it is not used for fitting or tuning.',
              '- Full fixed grids, training-only scalers, source checksums and convergence checks are recorded. Boundary choices are reported without expanding grids after inspecting test results.',
              '- These overlapping splits have informed prior modelling choices. Improvements remain exploratory and need confirmation on new samples. The original harmonized CTRP AUC provenance limitation remains.', '',
              'See `summary.tsv`, `paired_comparisons.tsv`, `lineage_win_counts.tsv`, `per_lineage_metrics.tsv`, `selected_parameters.tsv`, `grid_boundary_audit.tsv`, `protocol.json` and `validation.json` for the numerical evidence.', '']
    (OUT/'REVIEW.md').write_text('\n'.join(lines))


def seal():
    assert json.load(open(OUT/'visual_qa.json'))['passed']
    run.verify_sources()
    run.dump(OUT/'progress.json', dict(phase='complete', validation_passed=True, new_outer_models=160, reused_outer_models=120))
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
    run.initialize()
    run.verify_sources()
    assert json.load(open(OUT/'fit_completion.json'))['complete']
    protocol = json.load(open(OUT/'protocol.json'))
    all_results, parameters, tuning, validations = [], [], [], []
    for assay in run.SOURCES:
        preds, pp, tt, vv = audit(assay, protocol)
        all_results.append(evaluate(assay, preds, protocol))
        parameters.extend(pp)
        tuning.extend(tt)
        validations.append(vv)
    names = ['summary.tsv', 'paired_comparisons.tsv', 'lineage_paired_effects.tsv', 'lineage_win_counts.tsv',
             'per_repeat_pooled_metrics.tsv', 'per_lineage_metrics.tsv', 'per_repeat_macro_metrics.tsv', 'within_lineage_metrics.tsv']
    frames = [table(name, pd.concat([result[j] for result in all_results], ignore_index=True)) for j, name in enumerate(names)]
    params = table('selected_parameters.tsv', parameters)
    table('tuning_scores.tsv', tuning)
    counts = params.fillna('disabled').groupby(['assay', 'model', 'selected_candidate'], sort=True).size().reset_index(name='outer_fits')
    table('parameter_counts.tsv', counts)
    boundary_audit(params)
    validation = dict(passed=True, assays=validations, preflight=json.load(open(OUT/'preflight_validation.json')),
                      new_outer_models=160, reused_outer_models=120, unweighted_training=True, culture_covariates=False,
                      source_hashes_verified=True, retained_model_registry_unchanged=True)
    run.dump(OUT/'validation.json', validation)
    render(frames[0], frames[1], frames[3], validation)
    run.verify_sources()
    run.dump(OUT/'final_summary.json', dict(validation_passed=True, summary=frames[0].to_dict('records'),
                                           new_outer_models=160, reused_outer_models=120))
    run.dump(OUT/'progress.json', dict(phase='analysis_complete_pending_visual_qa', validation_passed=True))
    print('ANALYSIS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
