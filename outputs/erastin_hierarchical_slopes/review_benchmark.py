#!/usr/bin/env python3
"""Audit and evaluate partially pooled lineage-specific gene effects."""
import argparse
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler

import models
import run_benchmark as run
from make_splits import inner_rows, outer_split, inner_folds
from validate_preflight import independent
from evaluate import KEYS, score, bootstrap, validate_bootstrap, finite_mean, metrics

OUT = run.OUT
LABELS = dict(ridge='Ridge', elastic_net='Elastic Net', neural_network='Neural network', svr='RBF SVR',
              kernel_ridge='Kernel ridge', hierarchical_ridge='Previous random-intercept model',
              lineage_only='Lineage only', matched_intercept='Matched intercept-only model',
              hierarchical_slopes='Hierarchical gene slopes (fallback allowed)',
              positive_slopes='Hierarchical gene slopes (positive variance)', training_mean='Training mean')


def table(name, values):
    d = values if isinstance(values, pd.DataFrame) else pd.DataFrame(values)
    d.to_csv(OUT/name, sep='\t', index=False, float_format='%.12g')
    return d


def baseline_predict(m, family, x, lineage):
    if family == 'ridge':
        return ((x-m['mean'])/m['scale'])@m['coef']+m['intercept_after_scaling']
    z = m['x_scaler'].transform(x)
    sy = m['y_scaler']
    if family == 'elastic_net':
        out = m['model'].predict(z)
    elif family == 'neural_network':
        return np.mean([sy.inverse_transform(n.predict(z)[:, None]).ravel() for n in m['members']], axis=0)
    elif family == 'svr':
        out = m['estimator'].predict(rbf_kernel(z, m['train_z'], gamma=m['parameters']['gamma']))
    elif family == 'kernel_ridge':
        p = m['parameters']
        w = p['rbf_weight']
        q = (1-w)*(z@m['train_z'].T)/64+w*rbf_kernel(z, m['train_z'], gamma=p['gamma'])
        q = q-m['kernel_mean'][None, :]-q.mean(axis=1)[:, None]+m['kernel_grand']
        out = q@m['dual']
    else:
        d = (lineage[:, None] == m['levels'][None, :]).astype(float)
        if m['with_genes']:
            d = np.column_stack([z, d])
        out = (d-m['design_mean'])@m['coefficients']
    return sy.inverse_transform(out[:, None]).ravel()


def audit(assay):
    s = run.STATE[assay]
    x, y, lin, groups = s['x'], s['y'], s['lineage'], s['groups']
    old = np.load(run.SOURCE/assay/'predictions.npz')
    for key, value in [('sample_ids', s['ids']), ('group_ids', groups), ('lineage', lin), ('y', y), ('outer_test', s['test'])]:
        np.testing.assert_array_equal(old[key], value)
    preds = {f: np.full(s['test'].shape, np.nan) for f in run.BASELINES+models.ARMS+['training_mean']}
    params, tuning, refs, deviations = [], [], [], []
    max_prediction = max_score = max_normal = 0.
    score_checks = split_checks = unique_fits = 0
    for r, mask in enumerate(s['test']):
        tr, te = np.flatnonzero(~mask), np.flatnonzero(mask)
        np.testing.assert_array_equal(mask, outer_split(lin, groups, s['seeds'][r]))
        np.testing.assert_array_equal(s['inner'][r], inner_folds(lin, groups, mask, s['inner_seeds'][r]))
        assert set(groups[tr]).isdisjoint(groups[te]) and set(lin[tr]) == set(lin[te]) == set(lin)
        split_checks += 1
        cm = json.load(open(OUT/assay/'folds'/f'repeat_{r+1:02d}_complete.json'))
        assert cm['protocol_sha256'] == run.sha(OUT/'protocol.json')
        unique_fits += cm['unique_selected_fits']
        for row in cm['files']:
            assert run.sha(OUT/row['path']) == row['sha256']
        preds['training_mean'][r, te] = y[tr].mean()
        for f in run.BASELINES:
            root = (run.ORIGINAL[assay]/'folds' if f in run.BASELINES[:3] else run.SOURCE/assay/'folds')/f'{f}__p64__r{r+1:02d}'
            a = np.load(root.with_suffix('.npz'))
            np.testing.assert_array_equal(a['train_rows'], tr)
            np.testing.assert_array_equal(a['test_rows'], te)
            pp = baseline_predict(joblib.load(root.with_suffix('.joblib')), f, x[te], lin[te])
            np.testing.assert_allclose(pp, a['test_prediction'], atol=1e-10, rtol=0)
            np.testing.assert_allclose(pp, old[f][r, te], atol=1e-10, rtol=0)
            # Preserve the previous exact-constant lineage-control correction.
            preds[f][r, te] = old[f][r, te]
        t = np.load(OUT/assay/'folds'/f'tuning__r{r+1:02d}.npz')
        td = json.load(open(OUT/assay/'folds'/f'tuning__r{r+1:02d}.json'))
        assert td['protocol_sha256'] == run.sha(OUT/'protocol.json')
        val, loss = t['validation_predictions'], t['inner_fold_rmse']
        assert np.isfinite(val[:, s['inner'][r] >= 0]).all() and np.isnan(val[:, s['inner'][r] < 0]).all()
        for k in range(5):
            a, b = inner_rows(s['inner'][r], k)
            assert set(a)|set(b) == set(tr) and set(groups[a]).isdisjoint(groups[b])
            assert set(lin[a]) == set(lin)
            split_checks += 1
            rmse = np.sqrt(np.mean((val[:, b]-y[b][None, :])**2, axis=1))
            error = float(np.max(np.abs(rmse-loss[:, k])))
            max_score = max(max_score, error)
            assert error < 1e-12
            score_checks += len(models.GRID)
            diag = td['inner_diagnostics'][k]
            sx, sy = StandardScaler().fit(x[a]), StandardScaler().fit(y[a, None])
            np.testing.assert_allclose(diag['x_mean'], sx.mean_, atol=1e-12, rtol=0)
            np.testing.assert_allclose(diag['x_scale'], sx.scale_, atol=1e-12, rtol=0)
            assert diag['y_mean'] == sy.mean_[0] and diag['y_scale'] == sy.scale_[0]
            np.testing.assert_array_equal(diag['levels'], np.unique(lin[a]))
            z = sx.transform(x[a])
            centers = np.array([z[lin[a] == l].mean(axis=0) for l in np.unique(lin[a])])
            np.testing.assert_allclose(diag['lineage_centers'], centers, atol=1e-12, rtol=0)
        tuning.extend(dict(assay=assay, repeat=r+1, candidate=c, mean_inner_rmse=float(loss[c].mean()), **p) for c, p in enumerate(models.GRID))
        for arm in models.ARMS:
            root = run.root_for(assay, arm, r)
            meta = json.load(open(root.with_suffix('.json')))
            a = np.load(root.with_suffix('.npz'))
            model = joblib.load(root.with_suffix('.joblib'))
            allowed = run.eligible(arm)
            selected = int(allowed[np.argmin(loss[allowed].mean(axis=1))])
            assert meta['selected_candidate'] == selected and meta['selected_parameters'] == models.GRID[selected] == model['parameters']
            assert (meta['assay'], meta['arm'], meta['repeat']) == (assay, arm, r+1)
            assert meta['protocol_sha256'] == run.sha(OUT/'protocol.json')
            assert abs(meta['mean_inner_rmse']-loss[selected].mean()) < 1e-12
            np.testing.assert_array_equal(a['train_rows'], tr)
            np.testing.assert_array_equal(a['test_rows'], te)
            for ext in ['npz', 'joblib']:
                assert run.sha(root.with_suffix('.'+ext)) == meta[ext+'_sha256']
            sx, sy = StandardScaler().fit(x[tr]), StandardScaler().fit(y[tr, None])
            for attr in ['mean_', 'scale_']:
                np.testing.assert_allclose(getattr(model['x_scaler'], attr), getattr(sx, attr), atol=1e-12, rtol=0)
                np.testing.assert_allclose(getattr(model['y_scaler'], attr), getattr(sy, attr), atol=1e-12, rtol=0)
            z = sx.transform(x[tr])
            for j, label in enumerate(model['levels']):
                take = lin[tr] == label
                np.testing.assert_allclose(model['lineage_centers'][j], z[take].mean(axis=0), atol=1e-12, rtol=0)
                if take.sum() == 1:
                    np.testing.assert_array_equal(model['lineage_deviations'][j], np.zeros(64))
                deviations.append(dict(assay=assay, model=arm, repeat=r+1, lineage=str(label), training_n=int(take.sum()),
                                       test_n=int(np.sum(lin[te] == label)), offset_original_units=float(model['lineage_offsets'][j]*sy.scale_[0]),
                                       local_slope_l2_original_units=float(np.linalg.norm(model['lineage_deviations'][j])*sy.scale_[0]),
                                       shared_slope_l2_original_units=float(np.linalg.norm(model['global_coefficients'])*sy.scale_[0])))
            pp = models.predict(model, x[te], lin[te])
            error = float(np.max(np.abs(pp-a['test_prediction'])))
            max_prediction = max(max_prediction, error)
            assert error < 2e-8
            max_normal = max(max_normal, model['normal_equation_max_residual'])
            assert model['normal_equation_max_residual'] < 1e-6
            preds[arm][r, te] = a['test_prediction']
            if r in [0, 19]:
                ref = independent(x, y, lin, tr, te, model['parameters'])
                outer_error = float(np.max(np.abs(ref-a['test_prediction'])))
                inner_error = 0.
                for k in range(5):
                    aa, bb = inner_rows(s['inner'][r], k)
                    ref = independent(x, y, lin, aa, bb, model['parameters'])
                    inner_error = max(inner_error, float(np.max(np.abs(ref-val[selected, bb]))))
                assert max(inner_error, outer_error) < 2e-7
                refs.append(dict(model=arm, repeat=r+1, outer_error=outer_error, inner_error=inner_error))
            params.append(dict(assay=assay, model=arm, repeat=r+1, candidate=selected, mean_inner_rmse=meta['mean_inner_rmse'],
                               effective_df=model['effective_degrees_of_freedom'], **model['parameters']))
        if models.GRID[cm['selected_candidates']['hierarchical_slopes']]['slope_weight'] == 0:
            np.testing.assert_array_equal(preds['hierarchical_slopes'][r], preds['matched_intercept'][r])
        if (r+1) % 5 == 0:
            print('AUDITED', assay, r+1, flush=True)
    bv = validate_bootstrap(y, preds, lin, s['test'], 2026112002)
    validation = dict(assay=assay, passed=True, new_arm_bundles=60, unique_selected_outer_fits=unique_fits,
                      reused_outer_models=140, group_exclusion_checks=split_checks, candidate_fold_scores_checked=score_checks,
                      all_outer_lineages_on_both_sides=True, same_identity_groups_and_splits=True,
                      training_only_scalers_and_lineage_centers=True, singleton_training_lineages_have_zero_slope_deviations=True,
                      prediction_max_error=max_prediction, inner_rmse_max_error=max_score, normal_equation_max_residual=max_normal,
                      independent_explicit_feature_selected_checks=refs, bootstrap_arithmetic=bv,
                      original_models_and_predictions_reproduced=True)
    run.dump(OUT/assay/'validation.json', validation)
    np.savez_compressed(OUT/assay/'predictions.npz', sample_ids=s['ids'], group_ids=groups, lineage=lin, y=y, outer_test=s['test'], **preds)
    return preds, params, tuning, deviations, validation


def evaluate(assay, preds, protocol):
    s = run.STATE[assay]
    y, lin, test = s['y'], s['lineage'], s['test']
    pooled, per, macro = score(y, preds, lin, test)
    for d in [pooled, per, macro]:
        d.insert(0, 'assay', assay)
    centered = []
    for f, pp in preds.items():
        for r, mask in enumerate(test):
            yc, pc = y[mask].copy(), pp[r, mask].copy()
            for label in np.unique(lin):
                rows = lin[mask] == label
                yc[rows] -= yc[rows].mean()
                pc[rows] = pc[rows]-pc[rows].mean() if np.ptp(pc[rows]) > 0 else 0.
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
        row = dict(assay=assay, model=f)
        for key, val in values[f].items():
            good = val[np.isfinite(val)]
            row.update({key+'_mean': float(finite_mean(val)), key+'_sd': float(good.std(ddof=1)) if len(good) > 1 else np.nan,
                        key+'_defined_repeats': len(good)})
        row['centered_spearman_mean'] = float(finite_mean(centered[centered.model == f].centered_spearman))
        row['centered_pearson_mean'] = float(finite_mean(centered[centered.model == f].centered_pearson))
        summary.append(row)
    summary = pd.DataFrame(summary)
    prior = pd.read_csv(run.SOURCE/'summary.tsv', sep='\t')
    for f in run.BASELINES:
        a = summary[summary.model == f].iloc[0]
        b = prior[(prior.assay == assay)&(prior.model == f)].iloc[0]
        for key in ['pooled_rmse', 'pooled_spearman', 'pooled_r2', 'macro_all_rmse']:
            if np.isfinite(b[key+'_mean']):
                assert abs(a[key+'_mean']-b[key+'_mean']) < 1e-11
    run.dump(OUT/'progress.json', dict(phase='paired_bootstrap', assay=assay))
    draws, perdraw = bootstrap(y, preds, lin, test, protocol['uncertainty']['count'], protocol['uncertainty']['seed'])
    paired, effects, wins = [], [], []
    for first, ref in protocol['primary_comparisons']+protocol['secondary_comparisons']:
        for key in KEYS:
            sign = -1 if key.endswith('rmse') or key.endswith('mae') else 1
            delta = sign*(values[first][key]-values[ref][key])
            dd = finite_mean(sign*(perdraw[first][key]-perdraw[ref][key]), axis=1)
            good = dd[np.isfinite(dd)]
            lo, hi = np.percentile(good, [2.5, 97.5]) if len(good) else (np.nan, np.nan)
            paired.append(dict(assay=assay, first_model=first, reference_model=ref, metric=key, improvement=float(finite_mean(delta)),
                               favorable_repeats=int(np.sum(delta > 0)), tied_repeats=int(np.sum(np.abs(delta) <= 1e-12)),
                               worse_repeats=int(np.sum(delta < -1e-12)), defined_repeats=int(np.isfinite(delta).sum()),
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
    return summary, pd.DataFrame(paired), pd.DataFrame(effects), pd.DataFrame(wins), pooled, per, macro, centered


def render(summary, paired, wins, selection):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    cases = [('hierarchical_slopes', 'matched_intercept', 'Slopes vs matched intercept-only'),
             ('hierarchical_slopes', 'neural_network', 'Slopes vs neural network'),
             ('positive_slopes', 'matched_intercept', 'Positive slopes vs intercept-only')]
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 6.2))
    for row, assay in enumerate(run.ORIGINAL):
        for col, key in enumerate(['pooled_rmse', 'pooled_spearman']):
            ax = axes[row, col]
            for j, (f, ref, label) in enumerate(cases):
                d = paired[(paired.assay == assay)&(paired.first_model == f)&(paired.reference_model == ref)&(paired.metric == key)].iloc[0]
                ax.plot([d.conditional_ci_low, d.conditional_ci_high], [j, j], color='#28768e', lw=2)
                ax.scatter([d.improvement], [j], color='#173e57', s=30, zorder=3)
            ax.axvline(0, color='#777777', ls='--', lw=1)
            ax.set_yticks(range(3), [v[2] for v in cases] if col == 0 else ['']*3)
            ax.invert_yaxis()
            ax.set_title(assay+' | '+('RMSE reduction' if col == 0 else 'Spearman increase'))
            ax.set_xlabel('Improvement (positive favors lineage-specific slopes)')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(axis='x', alpha=.15)
            ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Hierarchical lineage-specific gene effects: conditional 95% lineage-bootstrap intervals', fontsize=12)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.text(.5, .01, '20 overlapping holdouts per assay; conditional exploratory intervals preserve all repeats and models together.', ha='center', fontsize=8)
    fig.savefig(OUT/'model_comparison.png', dpi=180)
    fig.savefig(OUT/'model_comparison.svg')
    plt.close(fig)
    lines = ['# Hierarchical lineage-specific gene slopes', '',
             'This extends the previous random-intercept model by allowing all 64 gene effects to vary by lineage, with Gaussian shrinkage toward shared slopes. CTRP and PRISM remain separate. Both keep the original 20 lineage-stratified identity-group holdouts and five inner folds, with unweighted training and no culture covariates.', '',
             '## Findings', '']
    for assay in run.ORIGINAL:
        selected = selection[assay]['positive_slope_selected_fits']
        lines.append(f'**{assay}:** Positive lineage-slope variance was selected in {selected}/20 repeats.')
        for ref in ['matched_intercept', 'neural_network']:
            d = paired[(paired.assay == assay)&(paired.first_model == 'hierarchical_slopes')&(paired.reference_model == ref)&(paired.metric == 'pooled_rmse')].iloc[0]
            base = summary[(summary.assay == assay)&(summary.model == ref)].iloc[0].pooled_rmse_mean
            direction = 'lower' if d.improvement > 0 else 'higher' if d.improvement < 0 else 'unchanged'
            support = 'Conditional interval favors improvement.' if d.conditional_ci_low > 0 else 'Conditional interval favors worse performance.' if d.conditional_ci_high < 0 else 'Conditional interval includes no difference.'
            w = wins[(wins.assay == assay)&(wins.first_model == 'hierarchical_slopes')&(wins.reference_model == ref)].iloc[0]
            lines.append(f'- Versus {LABELS[ref]}: {abs(100*d.improvement/base):.2f}% {direction} mean RMSE; {d.favorable_repeats}/20 favorable, {d.tied_repeats}/20 tied; {w.favorable_lineages}/{w.lineages} favorable lineages. {support}')
        lines.append('')
    lines += ['## Held-out performance', '', '| Assay | Model | RMSE | Spearman | R2 | Equal-lineage RMSE | Within-lineage centered Spearman |', '|---|---|---:|---:|---:|---:|---:|']
    for assay in run.ORIGINAL:
        for f in run.BASELINES+models.ARMS+['training_mean']:
            d = summary[(summary.assay == assay)&(summary.model == f)].iloc[0]
            rho = f'{d.pooled_spearman_mean:.4f}' if np.isfinite(d.pooled_spearman_mean) else 'undefined'
            cs = f'{d.centered_spearman_mean:.4f}' if np.isfinite(d.centered_spearman_mean) else 'undefined'
            lines.append(f'| {assay} | {LABELS[f]} | {d.pooled_rmse_mean:.6f} | {rho} | {d.pooled_r2_mean:.4f} | {d.macro_all_rmse_mean:.6f} | {cs} |')
    lines += ['', 'Values are means across repeats, not predictions averaged across outer models.', '', '## Paired RMSE differences', '',
              '| Assay | Candidate | Reference | RMSE reduction | Conditional 95% interval | Favorable repeats | Tied repeats |', '|---|---|---|---:|---|---:|---:|']
    for _, d in paired[paired.metric == 'pooled_rmse'].iterrows():
        lines.append(f'| {d.assay} | {LABELS[d.first_model]} | {LABELS[d.reference_model]} | {d.improvement:+.6f} | [{d.conditional_ci_low:+.6f}, {d.conditional_ci_high:+.6f}] | {d.favorable_repeats}/20 | {d.tied_repeats}/20 |')
    lines += ['', '![Hierarchical model comparison](model_comparison.png)', '', '## Model and audit details', '',
              '- Shared globally standardized gene effects, partially pooled lineage offsets and partially pooled local gene-slope deviations. Local deviations act on expression centered by each lineage current-training mean; all 64 genes are included without outcome-driven selection.',
              '- Gaussian MAP mixed model with variance/penalty ratios selected in inner CV. This is not full Bayesian posterior sampling; bootstrap intervals concern prediction differences, not posterior gene-effect probabilities.',
              '- The matched intercept-only arm uses the same global/intercept variance-ratio grid. The older random-intercept baseline uses a different independent intercept-penalty grid and is shown separately.',
              '- Full grid: 17 global penalties, five intercept variance weights and five local-slope variance weights (425 candidates). Matched intercept-only uses 85 candidates; positive-slope sensitivity arm uses 340. Zero slopes can be selected in the full model.',
              '- All preprocessing and lineage centering fit only the current inner/outer training set. Singleton training lineages have zero local design and exactly zero slope deviation. Every evaluated lineage appears in outer training and testing; this is not an unseen-lineage benchmark.',
              '- Compact fitted coefficients reproduce the dual-kernel predictions. All 850 first-inner-fold assay/candidate combinations were independently checked using explicit lineage-by-gene features and native ridge. Selected first/last-repeat outer and inner fits were independently checked again.',
              '- Every saved baseline/new prediction, scaler, lineage center, selected candidate and inner-fold RMSE was audited. Source models and input/split hashes are unchanged. 120 new arm/repeat bundles and 280 saved baselines are included; identical selected fits are reused across arms.',
              '- Whole-lineage paired bootstrap preserves every model and all 20 overlapping repeats. Intervals are conditional on this cohort/fits, exploratory and unadjusted for multiplicity. Repeatedly inspected splits require confirmation on new data.',
              '- Within-lineage centered correlations remove held-out lineage means from both observations and predictions for evaluation only. They are never used to fit or tune.',
              '- Prior CTRP AUC provenance limitation remains; outcomes and assays are unchanged. No retained model registry changes.', '',
              'Detailed evidence: `summary.tsv`, `paired_comparisons.tsv`, `lineage_win_counts.tsv`, `selected_parameters.tsv`, `lineage_effect_norms.tsv`, `protocol.json` and `validation.json`.', '']
    (OUT/'REVIEW.md').write_text('\n'.join(lines))


def seal():
    assert json.load(open(OUT/'visual_qa.json'))['passed']
    run.verify_sources()
    for name, h in json.load(open(OUT/'analysis_source_manifest.json'))['scripts'].items():
        assert run.sha(OUT/name) == h
    run.dump(OUT/'progress.json', dict(phase='complete', validation_passed=True, new_outer_bundles=120, reused_outer_models=280))
    files = [dict(path=str(p.relative_to(OUT)), sha256=run.sha(p), bytes=p.stat().st_size)
             for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'artifact_manifest.json' and '__pycache__' not in p.parts and p.suffix != '.log']
    run.dump(OUT/'artifact_manifest.json', dict(files=files))
    print('SEALED', len(files), flush=True)


def selection_summary(params):
    result = {}
    for assay in run.ORIGINAL:
        full = params[(params.assay == assay)&(params.model == 'hierarchical_slopes')]
        positive = params[(params.assay == assay)&(params.model == 'positive_slopes')]
        boundaries = []
        for arm in models.ARMS:
            chosen = params[(params.assay == assay)&(params.model == arm)]
            allowed = [models.GRID[j] for j in run.eligible(arm)]
            for key in ['alpha', 'intercept_weight', 'slope_weight']:
                levels = sorted({p[key] for p in allowed})
                boundaries.append(dict(model=arm, parameter=key, lower=levels[0], upper=levels[-1],
                                       lower_boundary_fits=int((chosen[key] == levels[0]).sum()),
                                       upper_boundary_fits=int((chosen[key] == levels[-1]).sum())))
        result[assay] = dict(positive_slope_selected_fits=int((full.slope_weight > 0).sum()),
                             full_model_zero_intercept_fits=int((full.intercept_weight == 0).sum()),
                             positive_slope_arm_lower_boundary_fits=int((positive.slope_weight == .01).sum()),
                             post_result_grid_expansion=False, parameter_boundaries=boundaries)
    return result


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
    run.dump(OUT/'analysis_source_manifest.json', dict(scripts={n: run.sha(OUT/n) for n in ['review_benchmark.py', 'evaluate.py', 'metric_helpers.py', 'weighted_mlp.py']}))
    protocol = json.load(open(OUT/'protocol.json'))
    results, parameters, tuning, deviations, validation = [], [], [], [], []
    for assay in run.ORIGINAL:
        preds, pp, tt, dd, vv = audit(assay)
        results.append(evaluate(assay, preds, protocol))
        parameters.extend(pp)
        tuning.extend(tt)
        deviations.extend(dd)
        validation.append(vv)
    names = ['summary.tsv', 'paired_comparisons.tsv', 'lineage_paired_effects.tsv', 'lineage_win_counts.tsv',
             'per_repeat_pooled_metrics.tsv', 'per_lineage_metrics.tsv', 'per_repeat_macro_metrics.tsv', 'within_lineage_metrics.tsv']
    frames = [table(n, pd.concat([r[j] for r in results], ignore_index=True)) for j, n in enumerate(names)]
    params = table('selected_parameters.tsv', parameters)
    table('tuning_scores.tsv', tuning)
    table('lineage_effect_norms.tsv', deviations)
    table('parameter_counts.tsv', params.groupby(['assay', 'model', 'candidate']).size().reset_index(name='outer_fits'))
    selection = selection_summary(params)
    run.dump(OUT/'grid_boundary_audit.json', selection)
    run.dump(OUT/'validation.json', dict(passed=True, assays=validation, preflight=json.load(open(OUT/'preflight_validation.json')),
                                        new_outer_bundles=120, reused_outer_models=280, unweighted_training=True,
                                        culture_covariates=False, source_hashes_verified=True, retained_registry_unchanged=True))
    render(frames[0], frames[1], frames[3], selection)
    run.verify_sources()
    run.dump(OUT/'final_summary.json', dict(validation_passed=True, summary=frames[0].to_dict('records'), selection=selection))
    run.dump(OUT/'progress.json', dict(phase='analysis_complete_pending_visual_qa', validation_passed=True))
    print('ANALYSIS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
