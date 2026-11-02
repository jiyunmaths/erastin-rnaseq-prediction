#!/usr/bin/env python3
"""Audit nested discovery; report prediction effects and training-selected genes."""
import argparse
import json
import time
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import common as c
import models
from make_splits import inner_rows, outer_split, inner_folds
from evaluate import KEYS, score, bootstrap, validate_bootstrap, finite_mean

OUT = c.OUT
BASELINES = dict(ridge64='ridge', nn64='neural_network', svr64='svr', kernel64='kernel_ridge', tab64='tabpfn35_fast')
NEW = ['ridge_augmented', 'ridge_genome', 'tab_ridge_screen32', 'tab_conditional32']
LABELS = dict(ridge64='Core64 ridge', nn64='Core64 NN', svr64='Core64 SVR', kernel64='Core64 kernel ridge',
              tab64='Core64 TabPFN', ridge_augmented='Core64 + nested screened genes: ridge',
              ridge_genome='All 19,176 genes: ridge', tab_ridge_screen32='Core64 + ridge-screened additions: TabPFN',
              tab_conditional32='Core64 + conditional top32: TabPFN', training_mean='Training mean')


def table(name, data):
    d = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
    d.to_csv(OUT/name, sep='\t', index=False, float_format='%.12g')
    return d


def audit(assay):
    s = c.STATE[assay]
    y, lin, groups = s['y'], s['lineage'], s['groups']
    old = np.load(c.TAB/assay/'predictions.npz')
    for key, val in [('sample_ids', s['ids']), ('group_ids', groups), ('lineage', lin), ('y', y), ('outer_test', s['test'])]:
        np.testing.assert_array_equal(old[key], val)
    preds = {f: old[key].copy() for f, key in BASELINES.items()}
    preds.update({f: np.full(s['test'].shape, np.nan) for f in NEW+['training_mean']})
    core, candidates = c.INDEX['core_indices'], c.INDEX['candidate_indices']
    candidate_position = {j: q for q, j in enumerate(candidates)}
    params, occurrences, dense, checks = [], [], [], []
    score_checks = group_checks = 0
    max_pred = max_loss = 0.
    for r, mask in enumerate(s['test']):
        tr, te = np.flatnonzero(~mask), np.flatnonzero(mask)
        np.testing.assert_array_equal(mask, outer_split(lin, groups, s['seeds'][r]))
        np.testing.assert_array_equal(s['inner'][r], inner_folds(lin, groups, mask, s['inner_seeds'][r]))
        assert set(groups[tr]).isdisjoint(groups[te]) and set(lin[tr]) == set(lin[te]) == set(lin)
        group_checks += 1
        for prefix in ['ridge', 'tab']:
            m = json.load(open(OUT/assay/'folds'/f'{prefix}_repeat_{r+1:02d}_complete.json'))
            assert m['protocol_sha256'] == c.sha(OUT/'protocol.json')
            for row in m['files']:
                assert c.sha(OUT/row['path']) == row['sha256']
        t = np.load(OUT/assay/'folds'/f'screening__r{r+1:02d}.npz')
        meta = json.load(open(OUT/assay/'folds'/f'screening__r{r+1:02d}.json'))
        assert meta['protocol_sha256'] == c.sha(OUT/'protocol.json')
        val, loss = t['validation_predictions'], t['inner_fold_rmse']
        gv, gl = t['genome_validation_predictions'], t['genome_inner_fold_rmse']
        valid = s['inner'][r] >= 0
        for arr in [val, gv]:
            assert np.isfinite(arr[:, valid]).all() and np.isnan(arr[:, ~valid]).all()
        for k in range(5):
            aa, bb = inner_rows(s['inner'][r], k)
            assert set(aa)|set(bb) == set(tr) and set(groups[aa]).isdisjoint(groups[bb])
            group_checks += 1
            for pp, ll in [(val, loss), (gv, gl)]:
                ref = np.sqrt(np.mean((pp[:, bb]-y[bb][None, :])**2, axis=1))
                error = float(np.max(np.abs(ref-ll[:, k])))
                max_loss = max(max_loss, error)
                assert error < 1e-12
                score_checks += len(ll)
            if r in [0, 19]:
                rank, _, _ = models.rank_genes(c.features(assay, aa), y[aa], lin[aa], core, candidates)
                for j, method in enumerate(models.METHODS):
                    np.testing.assert_array_equal(rank[method][:128], t['inner_top128'][k, j])
        chosen = int(np.argmin(loss.mean(axis=1)))
        whole = int(np.argmin(gl.mean(axis=1)))
        assert meta['selected_candidate'] == chosen and meta['selected_parameters'] == models.GRID[chosen]
        assert meta['selected_full_candidate'] == whole and meta['selected_full_alpha'] == models.FULL_ALPHAS[whole]
        outer_x = c.features(assay, tr)
        rank, scores, _ = models.rank_genes(outer_x, y[tr], lin[tr], core, candidates)
        for method in models.METHODS:
            np.testing.assert_array_equal(rank[method], t['ranked_'+method])
            np.testing.assert_allclose(scores[method], t['scores_'+method], atol=1e-10, rtol=0, equal_nan=True)
        selected = models.GRID[chosen]
        aug_indices = np.r_[core, rank[selected['method']][:selected['added']]] if selected['added'] else core
        tab_count = min(selected['added'], models.TAB_MAX_ADDED)
        tab_indices = np.r_[core, rank[selected['method']][:tab_count]] if tab_count else core
        expected = dict(ridge_augmented=aug_indices, ridge_genome=np.arange(len(c.INDEX['genes'])),
                        tab_ridge_screen32=tab_indices, tab_conditional32=np.r_[core, rank['conditional'][:32]])
        for arm in NEW:
            root = c.root_for(assay, arm, r)
            m = json.load(open(root.with_suffix('.json')))
            a = np.load(root.with_suffix('.npz'))
            assert m['protocol_sha256'] == c.sha(OUT/'protocol.json')
            assert (m['assay'], m['arm'], m['repeat']) == (assay, arm, r+1)
            assert c.sha(root.with_suffix('.npz')) == m['npz_sha256']
            np.testing.assert_array_equal(a['train_rows'], tr)
            np.testing.assert_array_equal(a['test_rows'], te)
            np.testing.assert_array_equal(a['feature_indices'], expected[arm])
            features = expected[arm]
            assert len(set(features)) == len(features) and set(core) <= set(features)
            assert m['gene_count'] == len(features)
            beta = None
            if arm.startswith('ridge'):
                assert c.sha(root.with_suffix('.joblib')) == m['joblib_sha256']
                model = joblib.load(root.with_suffix('.joblib'))
                np.testing.assert_array_equal(model['feature_indices'], features)
                sx = StandardScaler().fit(outer_x[:, features])
                np.testing.assert_allclose(model['x_scaler'].mean_, sx.mean_, atol=1e-12, rtol=0)
                np.testing.assert_allclose(model['x_scaler'].scale_, sx.scale_, atol=1e-12, rtol=0)
                pp = models.predict(model, c.features(assay, te, features))
                error = float(np.max(np.abs(pp-a['test_prediction'])))
                max_pred = max(max_pred, error)
                assert error < 1e-8
                beta = model['coef']
                if arm == 'ridge_augmented':
                    assert m['selected_parameters'] == selected
                else:
                    assert m['selected_parameters']['alpha'] == models.FULL_ALPHAS[whole]
                    dense.append(beta.copy())
            else:
                ref = c.WS/m['model_ref']
                assert c.sha(ref) == m['model_sha256'] and m['native_feature_coverage'] == len(features)
                with zipfile.ZipFile(ref) as z:
                    assert z.testzip() is None
                    cfg = json.loads(z.read('init_params.json'))
                    assert cfg['device'] == 'cpu' and cfg['n_estimators'] == 8 and cfg['tuning_config'] is None
                if len(features) == 64:
                    assert m['reused_core64_predictions']
                    np.testing.assert_array_equal(a['test_prediction'], old['tabpfn35_fast'][r, te])
                elif r in [0, 19]:
                    assert m['restored_prediction_max_error'] is not None and m['restored_prediction_max_error'] < 2e-6
                    checks.append(dict(model=arm, repeat=r+1, restored_prediction_error=m['restored_prediction_max_error']))
            preds[arm][r, te] = a['test_prediction']
            params.append(dict(assay=assay, model=arm, repeat=r+1, gene_count=len(features),
                               method=selected['method'] if arm in ['ridge_augmented', 'tab_ridge_screen32'] else 'conditional' if arm == 'tab_conditional32' else 'all_genes',
                               added=len(features)-64, alpha=m['selected_parameters'].get('alpha') if arm.startswith('ridge') else np.nan))
            if arm != 'ridge_genome':
                for j, gene in enumerate(features[64:], 64):
                    q = candidate_position[gene]
                    occurrences.append(dict(assay=assay, model=arm, repeat=r+1, feature_index=int(gene),
                                            marginal_corr=float(scores['marginal'][q]), conditional_corr=float(scores['conditional'][q]),
                                            coefficient=float(beta[j]) if beta is not None else np.nan))
        if selected['added'] == 0:
            np.testing.assert_allclose(preds['ridge_augmented'][r, te], preds['ridge64'][r, te], atol=1e-10, rtol=0)
        preds['training_mean'][r, te] = y[tr].mean()
        if (r+1) % 5 == 0:
            print('AUDITED', assay, r+1, flush=True)
    dense = np.array(dense)
    np.savez_compressed(OUT/assay/'genome_ridge_coefficients.npz', coefficients=dense,
                        mean_absolute=np.mean(np.abs(dense), axis=0), mean_signed=np.mean(dense, axis=0),
                        sign_consistency=np.maximum((dense > 0).mean(axis=0), (dense < 0).mean(axis=0)))
    bv = validate_bootstrap(y, preds, lin, s['test'], 2026112202)
    validation = dict(assay=assay, passed=True, group_checks=group_checks, inner_candidate_fold_scores_checked=score_checks,
                      all_outer_training_rankings_reproduced=True, first_last_inner_rankings_reproduced=True,
                      every_added_gene_selected_without_its_outer_test_response=True,
                      ridge_prediction_max_error=max_pred, inner_rmse_max_error=max_loss,
                      baseline_core64_predictions_unchanged=True, native_context_archive_checks_passed=True,
                      native_first_last_reloads=checks, bootstrap_arithmetic=bv)
    c.dump(OUT/assay/'validation.json', validation)
    np.savez_compressed(OUT/assay/'predictions.npz', sample_ids=s['ids'], group_ids=groups, lineage=lin, y=y, outer_test=s['test'], **preds)
    return preds, params, occurrences, validation


def evaluate(assay, preds, protocol):
    s = c.STATE[assay]
    y, lin, test = s['y'], s['lineage'], s['test']
    pooled, per, macro = score(y, preds, lin, test)
    for frame in [pooled, per, macro]:
        frame.insert(0, 'assay', assay)
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
            row.update({k+'_mean': float(finite_mean(v)), k+'_sd': float(good.std(ddof=1)) if len(good) > 1 else np.nan})
        summary.append(row)
    old = pd.read_csv(c.TAB/'summary.tsv', sep='\t')
    for f, name in BASELINES.items():
        a = next(r for r in summary if r['model'] == f)
        b = old[(old.assay == assay)&(old.model == name)].iloc[0]
        for k in ['pooled_rmse', 'pooled_spearman', 'pooled_r2']:
            assert abs(a[k+'_mean']-b[k+'_mean']) < 1e-11
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
                               favorable_repeats=int(np.sum(delta > 1e-12)), tied_repeats=int(np.sum(np.abs(delta) <= 1e-12)),
                               defined_repeats=int(np.isfinite(delta).sum()), conditional_ci_low=lo, conditional_ci_high=hi))
        favorable = 0
        for label in np.unique(lin):
            a = per[(per.model == first)&(per.lineage == label)].sort_values('repeat')
            b = per[(per.model == ref)&(per.lineage == label)].sort_values('repeat')
            delta = b.rmse.to_numpy()-a.rmse.to_numpy()
            favorable += int(delta.mean() > 1e-12)
            effects.append(dict(assay=assay, first_model=first, reference_model=ref, lineage=str(label),
                                cohort_n=int(a.cohort_n.iloc[0]), test_n=int(a.test_n.iloc[0]), rmse_improvement=float(delta.mean())))
        wins.append(dict(assay=assay, first_model=first, reference_model=ref, favorable_lineages=favorable, lineages=len(np.unique(lin))))
    np.savez_compressed(OUT/assay/'bootstrap_draws.npz', **{f+'__'+k: v for f, d in draws.items() for k, v in d.items()})
    return pd.DataFrame(summary), pd.DataFrame(paired), pd.DataFrame(effects), pd.DataFrame(wins), pooled, per, macro


def gene_summary(occurrences):
    data = pd.DataFrame(occurrences)
    mapping = pd.read_csv(OUT/'gene_mapping.tsv', sep='\t', dtype={'entrez_id': str}).set_index('feature_index')
    rows = []
    for (assay, arm, gene), d in data.groupby(['assay', 'model', 'feature_index']):
        assert d.repeat.nunique() == len(d) <= 20 and not bool(mapping.loc[gene].in_core64)
        finite = d.coefficient.dropna().to_numpy()
        cond = d.conditional_corr.dropna().to_numpy()
        m = mapping.loc[gene]
        rows.append(dict(assay=assay, model=arm, feature_index=gene, gene=m.depmap_symbol, entrez_id=m.entrez_id,
                         release_symbols=m.current_symbols, selected_outer_fits=len(d), selection_fraction=len(d)/20,
                         mean_marginal_corr=d.marginal_corr.mean(), mean_conditional_corr=d.conditional_corr.mean(),
                         conditional_sign_consistency=max(np.mean(cond > 0), np.mean(cond < 0)) if len(cond) else np.nan,
                         mean_absolute_coefficient=float(np.abs(finite).mean()) if len(finite) else np.nan,
                         coefficient_sign_consistency=max(np.mean(finite > 0), np.mean(finite < 0)) if len(finite) else np.nan))
    summary = table('gene_selection_stability.tsv', rows)
    candidates = summary[summary.model == 'tab_conditional32'].copy()
    cross = []
    for gene, d in candidates.groupby('feature_index'):
        item = dict(feature_index=gene, gene=d.gene.iloc[0], entrez_id=d.entrez_id.iloc[0])
        counts = []
        for assay in c.ASSAYS:
            one = d[d.assay == assay]
            n = int(one.selected_outer_fits.iloc[0]) if len(one) else 0
            item[assay+'_selected_fits'] = n
            item[assay+'_mean_conditional_corr'] = float(one.mean_conditional_corr.iloc[0]) if len(one) else np.nan
            counts.append(n)
        item.update(min_assay_count=min(counts), total_count=sum(counts))
        cross.append(item)
    cross = pd.DataFrame(cross).sort_values(['min_assay_count', 'total_count', 'gene'], ascending=[False, False, True])
    table('cross_assay_candidate_genes.tsv', cross)
    dense = []
    for assay in c.ASSAYS:
        a = np.load(OUT/assay/'genome_ridge_coefficients.npz')
        chosen = np.argsort(-a['mean_absolute'])
        for gene in chosen[:100]:
            m = mapping.loc[gene]
            dense.append(dict(assay=assay, gene=m.depmap_symbol, entrez_id=m.entrez_id, in_core64=bool(m.in_core64),
                              mean_absolute_coefficient=float(a['mean_absolute'][gene]), mean_coefficient=float(a['mean_signed'][gene]),
                              sign_consistency=float(a['sign_consistency'][gene])))
    table('genome_ridge_top_coefficients.tsv', dense)
    return summary, cross


def render(summary, paired, wins, params, gene_stability, candidates):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    cases = [('ridge_augmented', 'ridge64', 'Screened additions: ridge'), ('ridge_genome', 'ridge64', 'All measured genes: ridge'),
             ('tab_ridge_screen32', 'tab64', 'Ridge-screened additions: TabPFN'), ('tab_conditional32', 'tab64', 'Conditional top32: TabPFN')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.7))
    for row, assay in enumerate(c.ASSAYS):
        for col, key in enumerate(['pooled_rmse', 'pooled_spearman']):
            ax = axes[row, col]
            for j, (first, ref, label) in enumerate(cases):
                d = paired[(paired.assay == assay)&(paired.first_model == first)&(paired.reference_model == ref)&(paired.metric == key)].iloc[0]
                ax.plot([d.conditional_ci_low, d.conditional_ci_high], [j, j], lw=2, color='#28768e')
                ax.scatter([d.improvement], [j], s=30, color='#173e57', zorder=3)
            ax.axvline(0, ls='--', lw=1, color='#777777')
            ax.set_yticks(range(4), [x[2] for x in cases] if col == 0 else ['']*4)
            ax.invert_yaxis()
            ax.set_title(assay+' | '+('RMSE reduction' if col == 0 else 'Spearman increase'))
            ax.set_xlabel('Improvement over matched core64 model')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(axis='x', alpha=.15)
            ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Genome-wide search beyond the retained 64 genes: conditional 95% intervals', fontsize=12)
    fig.tight_layout(rect=(0, .04, 1, .95))
    fig.text(.5, .01, 'Gene selection is inside training folds. Paired whole-lineage bootstrap retains all overlapping repeats. Exploratory evidence.', ha='center', fontsize=8)
    fig.savefig(OUT/'model_comparison.png', dpi=180)
    fig.savefig(OUT/'model_comparison.svg')
    plt.close(fig)
    top = candidates.head(20)
    image = top[['CTRP_selected_fits', 'PRISM_selected_fits']].to_numpy()
    fig, ax = plt.subplots(figsize=(5, max(4.5, len(top)*.25)))
    plot = ax.imshow(image, vmin=0, vmax=20, cmap='Blues', aspect='auto')
    ax.set_xticks([0, 1], ['CTRP', 'PRISM'])
    ax.set_yticks(np.arange(len(top)), top.gene.to_list())
    for i in range(len(top)):
        for j in range(2):
            ax.text(j, i, str(image[i, j]), ha='center', va='center', color='white' if image[i, j] > 12 else '#173e57', fontsize=8)
    ax.set_title('Conditional top32 selection frequency\n(training-only; out of 20 overlapping splits)', fontsize=10)
    fig.colorbar(plot, ax=ax, label='Selected training fits')
    fig.tight_layout()
    fig.savefig(OUT/'candidate_gene_stability.png', dpi=180)
    plt.close(fig)
    lines = ['# Genome-wide RNA search beyond the retained core64 panel', '',
             'The user referred to a 60-gene panel; this experiment retains the existing 64-gene panel unchanged. Search covers 19,176 unique measured protein-coding features, including 19,112 candidates outside the core. Seventeen duplicated Entrez IDs were resolved using approved release HGNC symbols. Noncoding transcripts and unmeasured DNA loci are outside this dataset.', '',
             '## Findings', '',
             '**Keep the core64 panel. This search did not establish a compact expansion that improves held-out accuracy.** Full-transcriptome ridge has a modest CTRP point improvement whose conditional interval includes zero, without improvement in PRISM. TabPFN additions do not improve CTRP and worsen PRISM. These results concern the tested screening procedures, not every possible gene combination.', '']
    for assay in c.ASSAYS:
        lines.append('**'+assay+'**')
        lines.append('')
        for first, ref, label in cases:
            d = paired[(paired.assay == assay)&(paired.first_model == first)&(paired.reference_model == ref)&(paired.metric == 'pooled_rmse')].iloc[0]
            base = summary[(summary.assay == assay)&(summary.model == ref)].iloc[0].pooled_rmse_mean
            support = 'Conditional interval favors improvement.' if d.conditional_ci_low > 0 else 'Conditional interval favors worse performance.' if d.conditional_ci_high < 0 else 'Conditional interval includes no difference.'
            direction = 'lower' if d.improvement > 0 else 'higher'
            lines.append(f'- {label}: {abs(100*d.improvement/base):.2f}% {direction} mean RMSE; {d.favorable_repeats}/20 favorable repeats, {d.tied_repeats}/20 ties. {support}')
        lines.append('')
    lines += ['| Assay | Procedure | RMSE reduction over matched core64 model | Conditional 95% interval |', '|---|---|---:|---:|']
    for assay in c.ASSAYS:
        for first, ref, label in cases:
            d = paired[(paired.assay == assay)&(paired.first_model == first)&(paired.reference_model == ref)&(paired.metric == 'pooled_rmse')].iloc[0]
            lines.append(f'| {assay} | {label} | {d.improvement:+.6f} | [{d.conditional_ci_low:+.6f}, {d.conditional_ci_high:+.6f}] |')
    lines += ['', 'Positive reductions favor additions. These are paired whole-lineage bootstrap intervals conditional on the fitted, overlapping splits; they do not include retraining or correct for the history of model searches.', '']
    lines += ['## Held-out performance', '', '| Assay | Model | Mean genes | RMSE | Spearman | R2 | Equal-lineage RMSE |', '|---|---|---:|---:|---:|---:|---:|']
    for assay in c.ASSAYS:
        for f in list(BASELINES)+NEW+['training_mean']:
            d = summary[(summary.assay == assay)&(summary.model == f)].iloc[0]
            n = params[(params.assay == assay)&(params.model == f)].gene_count.mean() if f in NEW else 0 if f == 'training_mean' else 64
            rho = f'{d.pooled_spearman_mean:.4f}' if np.isfinite(d.pooled_spearman_mean) else 'undefined'
            lines.append(f'| {assay} | {LABELS[f]} | {n:.1f} | {d.pooled_rmse_mean:.6f} | {rho} | {d.pooled_r2_mean:.4f} | {d.macro_all_rmse_mean:.6f} |')
    lines += ['', 'Scores are means across repeats. A compact panel can vary between training splits; the mean gene count is not a single locked final panel.', '',
              '![Genome-wide model comparison](model_comparison.png)', '', '## Candidate genes', '',
              'The strongest training-only conditional candidates differ between assays. Counts describe overlapping training sets and are not independent replications, causal evidence or validation of a consolidated gene panel. The assays also share many cell lines.', '',
              '| Assay | Examples among the most frequently selected conditional candidates (fits /20; ties possible) |', '|---|---|']
    for assay in c.ASSAYS:
        g = gene_stability[(gene_stability.assay == assay)&(gene_stability.model == 'tab_conditional32')].sort_values(['selected_outer_fits', 'gene'], ascending=[False, True]).head(5)
        lines.append('| '+assay+' | '+', '.join(f'{r.gene} ({r.selected_outer_fits}/20)' for _, r in g.iterrows())+' |')
    lines += ['', 'The table below puts genes selected in both assays first, then orders the remaining genes by total selection count. Mean conditional correlations are calculated only among training fits in which a gene was selected; a dash means it was never selected for that assay, not zero correlation.', '',
              '| Gene | Entrez ID | CTRP fits /20 | PRISM fits /20 | CTRP mean conditional correlation | PRISM mean conditional correlation |', '|---|---|---:|---:|---:|---:|']
    for _, r in candidates.head(15).iterrows():
        corr = [f'{r[a+"_mean_conditional_corr"]:+.3f}' if np.isfinite(r[a+'_mean_conditional_corr']) else '-' for a in c.ASSAYS]
        lines.append(f'| {r.gene} | {r.entrez_id} | {r.CTRP_selected_fits} | {r.PRISM_selected_fits} | {corr[0]} | {corr[1]} |')
    lines += ['', '![Candidate training-selection stability](candidate_gene_stability.png)', '',
              'Full training-only selection frequencies and coefficient summaries are in `gene_selection_stability.tsv`, `cross_assay_candidate_genes.tsv`, and `genome_ridge_top_coefficients.tsv`. Dense-ridge coefficients are regularized associations, not gene-selection probabilities. TabPFN selection counts are not TabPFN feature importance.', '',
              '## Leakage controls and interpretation', '',
              '- The core64 is mandatory. Outside genes are searched without a pathway whitelist. Only current-training constant features are filtered; the conditional method also removes numerical aliases of the control design.',
              '- Every inner-training fold independently computes marginal and conditional rankings. Conditional correlation adjusts both candidate and target for the core64 and training-lineage indicators using an SVD projection. Models themselves use RNA only, without lineage/culture covariates.',
              '- Selector, 0/8/16/32/64/128 additions and 17 ridge penalties are selected by five-fold erastin RMSE. Outer-training ranks are refitted only after that choice. A separate all-gene ridge control tunes 25 penalties. Zero additions recover the saved core64 ridge.',
              '- TabPFN uses the same fixed 3.5-Fast checkpoint, eight passes and outer seed as its saved core64 baseline. One arm uses min(ridge-selected additions,32); a second always uses training-only conditional top32. No TabPFN-based tuning of gene counts or selectors uses test data. Zero additions reuse exact baseline predictions.',
              '- Identity groups remain disjoint and every evaluated lineage is present in train/test. Both assays reuse their original 20 outer splits and inner assignments. All cells and response values are unchanged.',
              '- Source/mapping checks, independent screen/regression checks, a validation-only decoy control, repeated ranking checks, every inner RMSE, every ridge prediction, native context checks and all source hashes are audited. Native context reloads are performed for first/last repeats, not every archive.',
              '- Whole-lineage paired bootstrap preserves all models and repeats together. Intervals are conditional, exploratory and not multiplicity adjusted. Prior inspection of these data cannot be undone by nesting: these are selection-leakage-controlled discovery estimates, not a wholly untouched confirmatory benchmark.',
              '- Performance concerns the fold-specific selection procedure. Taking a final consensus list from all repetitions and claiming the same accuracy would introduce selection bias; that final fixed list needs separate validation.',
              '- Univariate screens can miss nonlinear interaction-only genes. TabPFN additions are CPU-capped at 32 and not specifically tuned for TabPFN. Neither negative results nor predictive associations establish ferroptosis mechanisms.',
              '- The ridge addition grid ends at 128. CTRP selects that upper limit in 12/20 fits, indicating that this search does not establish an optimal compact panel size. PRISM selects no additions in 9/20 fits. The grid is not expanded after viewing held-out results.',
              '- Original harmonized CTRP AUC provenance and TabPFN pretraining-overlap limitations remain. Retained core panel and model registry are unchanged.', '',
              'Methodological reference: [Ambroise and McLachlan, selection bias in gene extraction](https://pmc.ncbi.nlm.nih.gov/articles/PMC124442/).', '']
    (OUT/'REVIEW.md').write_text('\n'.join(lines))


def seal():
    assert json.load(open(OUT/'visual_qa.json'))['passed']
    c.verify_sources()
    for n, h in json.load(open(OUT/'analysis_source_manifest.json'))['scripts'].items():
        assert c.sha(OUT/n) == h
    c.dump(OUT/'progress.json', dict(phase='complete', validation_passed=True))
    files = [dict(path=str(p.relative_to(OUT)), sha256=c.sha(p), bytes=p.stat().st_size)
             for p in sorted(OUT.rglob('*')) if p.is_file() and p.name != 'artifact_manifest.json' and '__pycache__' not in p.parts and p.suffix != '.log']
    c.dump(OUT/'artifact_manifest.json', dict(files=files))
    print('SEALED', len(files), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wait', action='store_true')
    parser.add_argument('--seal', action='store_true')
    args = parser.parse_args()
    if args.seal:
        seal()
        return
    c.initialize()
    c.verify_sources()
    c.dump(OUT/'analysis_source_manifest.json', dict(scripts={n: c.sha(OUT/n) for n in ['review_search.py', 'evaluate.py', 'metric_helpers.py']}))
    protocol = json.load(open(OUT/'protocol.json'))
    results, params, occurrences, validations = [], [], [], []
    for assay in c.ASSAYS:
        if args.wait:
            print('WAITING_FOR_ASSAY', assay, flush=True)
            while len(list((OUT/assay/'folds').glob('tab_repeat_*_complete.json'))) < 20:
                time.sleep(5)
        pred, pp, oo, vv = audit(assay)
        results.append(evaluate(assay, pred, protocol))
        params.extend(pp)
        occurrences.extend(oo)
        validations.append(vv)
        print('ASSAY_ANALYSIS_COMPLETE', assay, flush=True)
    names = ['summary.tsv', 'paired_comparisons.tsv', 'lineage_paired_effects.tsv', 'lineage_win_counts.tsv',
             'per_repeat_pooled_metrics.tsv', 'per_lineage_metrics.tsv', 'per_repeat_macro_metrics.tsv']
    frames = [table(n, pd.concat([r[j] for r in results], ignore_index=True)) for j, n in enumerate(names)]
    params = table('selected_parameters.tsv', params)
    table('gene_selection_occurrences.tsv', occurrences)
    stability, candidates = gene_summary(occurrences)
    table('parameter_counts.tsv', params.groupby(['assay', 'model', 'method', 'added'], dropna=False).size().reset_index(name='outer_fits'))
    c.dump(OUT/'validation.json', dict(passed=True, assays=validations, preflight=json.load(open(OUT/'preflight_validation.json')),
                                      core64_retained=True, no_test_response_based_feature_selection=True,
                                      datasets_and_splits_unchanged=True, source_hashes_verified=True))
    render(frames[0], frames[1], frames[3], params, stability, candidates)
    c.verify_sources()
    c.dump(OUT/'final_summary.json', dict(validation_passed=True, summary=frames[0].to_dict('records'),
                                        candidate_genes=candidates.head(30).to_dict('records')))
    c.dump(OUT/'progress.json', dict(phase='analysis_complete_pending_visual_qa', validation_passed=True))
    print('ANALYSIS_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
