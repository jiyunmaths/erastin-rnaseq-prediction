#!/usr/bin/env python3
"""Audit, summarize and plot exact saved-model gene importance."""
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('calculate_importance', OUT / 'calculate_importance.py')
calc = importlib.util.module_from_spec(spec); spec.loader.exec_module(calc)
MODELS = ['ridge', 'neural_network']
TITLES = {'ridge': 'Ridge', 'neural_network': 'Neural network'}


def table(name, rows):
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT / name, sep='\t', index=False, float_format='%.12g')
    return frame


def correlation(a, b):
    return float(spearmanr(a, b).statistic)


def main():
    d = np.load(OUT / 'analysis_inputs.npz', allow_pickle=False)
    genes, labels, lineage = d['genes'], d['labels'], d['lineage']
    n = np.array([np.sum(lineage == v) for v in labels])
    eligible = n >= 10; nonsingleton = n >= 2
    assert eligible.sum() == 19 and n[eligible].sum() == 721 and n[nonsingleton].sum() == 744
    importance = np.empty((20, 24, 2, 40))
    baseline = np.empty((20, 24, 2))
    cell = np.empty((20, 2, 745, 40)); grad = np.empty_like(cell)
    protocol = json.loads((OUT / 'protocol.json').read_text())
    ph = calc.sha(OUT / 'protocol.json')
    assert calc.sha(OUT / 'calculate_importance.py') == protocol['script_sha256']
    validation = json.loads((OUT / 'calculation_validation.json').read_text())
    for r in range(20):
        seen = np.zeros(745, dtype=int)
        for l in range(24):
            path = OUT / 'folds' / f'repeat_{r+1:02d}__lineage_{l:02d}'
            meta = json.loads(path.with_suffix('.json').read_text())
            assert meta['protocol_sha256'] == ph and calc.sha(path.with_suffix('.npz')) == meta['array_sha256']
            a = np.load(path.with_suffix('.npz'), allow_pickle=False)
            te = a['test_rows']; assert set(te) == set(np.flatnonzero(lineage == labels[l]))
            seen[te] += 1
            cell[r][:, te, :] = a['per_cell_delta_mse']; grad[r][:, te, :] = a['standardized_gradient']
            importance[r, l] = a['importance_mse']; baseline[r, l] = a['baseline_mse']
            np.testing.assert_allclose(a['per_cell_delta_mse'].mean(axis=1), importance[r, l], rtol=0, atol=1e-14)
            np.testing.assert_allclose(((d['saved_predictions'][r][:, te] - d['y'][te])**2).mean(axis=1), baseline[r, l], rtol=0, atol=1e-14)
        assert np.all(seen == 1)
    assert np.isfinite(cell).all() and np.isfinite(grad).all()
    assert np.max(np.abs(importance[:, n == 1])) < 1e-12
    mu = importance.mean(axis=0)
    lineage_gradient = np.stack([grad[:, :, lineage == v].mean(axis=2) for v in labels], axis=1)
    scopes = {
        'weighted_19_lineages': (eligible, n[eligible] / n[eligible].sum()),
        'macro_19_lineages': (eligible, np.full(19, 1/19)),
        'weighted_23_nonsingleton_lineages': (nonsingleton, n[nonsingleton] / n[nonsingleton].sum())}
    aggregates = {scope: np.einsum('rlmg,l->rmg', importance[:, mask], weights) for scope, (mask, weights) in scopes.items()}
    base_aggregates = {scope: np.einsum('rlm,l->rm', baseline[:, mask], weights) for scope, (mask, weights) in scopes.items()}
    # Independent all-pairs ridge expectation from the closed-form squared loss.
    # E[(e_i + beta*(x_d-x_i))^2 - e_i^2] =
    # 2 e_i beta (mean_x-x_i) + beta^2 (var_x+(mean_x-x_i)^2).
    ridge_formula_error = 0.
    for r in range(20):
        for l in range(24):
            te = np.flatnonzero(lineage == labels[l])
            z = (d['x'][te] - d['ridge_mean'][r,l]) / d['ridge_scale'][r,l]
            shift = z.mean(axis=0) - z
            error = d['saved_predictions'][r, 0, te] - d['y'][te]
            b = d['ridge_coef'][r,l]
            expected = 2 * error[:, None] * b * shift + b*b*(z.var(axis=0) + shift*shift)
            ridge_formula_error = max(ridge_formula_error, float(np.max(np.abs(expected - cell[r, 0, te]))))
    assert ridge_formula_error < 1e-11
    # Pair every model, gene and repeat under the same lineage bootstrap draws.
    rng = np.random.default_rng(protocol['uncertainty']['seed'])
    picks = rng.integers(19, size=(protocol['uncertainty']['draws'], 19))
    counts = np.array([np.bincount(v, minlength=19) for v in picks])
    weighted = counts * n[eligible]
    weighted = weighted / weighted.sum(axis=1, keepdims=True)
    boot = {
        'weighted_19_lineages': np.einsum('bl,lmg->bmg', weighted, mu[eligible]),
        'macro_19_lineages': np.einsum('bl,lmg->bmg', counts / 19, mu[eligible])}
    # Check the weighted bootstrap using directly expanded lineage samples.
    bootstrap_error = 0.
    for i in range(25):
        expected = np.average(mu[eligible][picks[i]], axis=0, weights=n[eligible][picks[i]])
        bootstrap_error = max(bootstrap_error, float(np.max(np.abs(expected - boot['weighted_19_lineages'][i]))))
    assert bootstrap_error < 1e-14
    pooled_rows, repeat_rows, contrast_rows = [], [], []
    lineage_ranks = rankdata(-mu, axis=2, method='min')
    for scope, v in aggregates.items():
        mask, weights = scopes[scope]
        ranks = rankdata(-v, axis=2, method='min')
        ranks_mu = rankdata(-v.mean(axis=0), axis=1, method='min')
        direction = np.einsum('rlmg,l->rmg', lineage_gradient[:, mask], weights)
        for m, model in enumerate(MODELS):
            for j, gene in enumerate(genes):
                series = v[:, m, j]
                row = {'scope': scope, 'model': model, 'gene': gene, 'rank': int(ranks_mu[m,j]),
                       'delta_mse': series.mean(), 'repeat_sd': series.std(ddof=1), 'repeat_min': series.min(), 'repeat_max': series.max(),
                       'positive_repeats': int((series > 0).sum()), 'top10_repeats': int((ranks[:,m,j] <= 10).sum()),
                       'repeat_rank_min': int(ranks[:,m,j].min()), 'repeat_rank_max': int(ranks[:,m,j].max()),
                       'positive_lineages_of_19': int((mu[eligible,m,j] > 0).sum()),
                       'top10_lineages_of_19': int((lineage_ranks[eligible,m,j] <= 10).sum()),
                       'mean_standardized_slope': direction[:,m,j].mean(),
                       'positive_slope_lineages_of_19': int((lineage_gradient[:,eligible,m,j].mean(axis=0) > 0).sum()),
                       'negative_slope_lineages_of_19': int((lineage_gradient[:,eligible,m,j].mean(axis=0) < 0).sum()),
                       'mean_baseline_mse': base_aggregates[scope][:,m].mean(),
                       'relative_mse_increase_percent': 100*series.mean()/base_aggregates[scope][:,m].mean(),
                       'rmse_equivalent_increase': np.mean(np.sqrt(base_aggregates[scope][:,m]+series)-np.sqrt(base_aggregates[scope][:,m]))}
                if scope in boot:
                    ci = np.percentile(boot[scope][:,m,j], [2.5,97.5])
                    row.update(conditional_ci_low=ci[0], conditional_ci_high=ci[1],
                               bootstrap_top10_fraction=float(np.mean(rankdata(-boot[scope][:,m,:],axis=1,method='min')[:,j] <= 10)))
                pooled_rows.append(row)
                for r in range(20):
                    repeat_rows.append({'scope':scope,'model':model,'gene':gene,'repeat':r+1,
                                        'delta_mse':series[r],'rank':int(ranks[r,m,j]),'standardized_slope':direction[r,m,j]})
        for j,gene in enumerate(genes):
            row={'scope':scope,'gene':gene,'nn_minus_ridge_importance':(v[:,1,j]-v[:,0,j]).mean()}
            if scope in boot:
                ci=np.percentile(boot[scope][:,1,j]-boot[scope][:,0,j],[2.5,97.5])
                row.update(conditional_ci_low=ci[0],conditional_ci_high=ci[1])
            contrast_rows.append(row)
    global_table=table('gene_importance_summary.tsv',pooled_rows).sort_values(['scope','model','rank'])
    table('gene_importance_summary.tsv',global_table)
    table('per_repeat_importance.tsv',repeat_rows);table('importance_model_contrasts.tsv',contrast_rows)
    lineage_rows=[]
    for l,label in enumerate(labels):
        te=np.flatnonzero(lineage==label)
        for m,model in enumerate(MODELS):
            rr=rankdata(-importance[:,l,m],axis=1,method='min')
            for j,gene in enumerate(genes):
                g=grad[:,m,te,j]
                values=importance[:,l,m,j]
                lineage_rows.append({'lineage':label,'n':int(n[l]),'eligible_n_ge_10':bool(eligible[l]),
                    'model':model,'gene':gene,'delta_mse':mu[l,m,j] if n[l]>1 else np.nan,
                    'repeat_sd':values.std(ddof=1) if n[l]>1 else np.nan,
                    'rank':int(lineage_ranks[l,m,j]) if n[l]>1 else np.nan,
                    'repeat_rank_min':int(rr[:,j].min()) if n[l]>1 else np.nan,
                    'repeat_rank_max':int(rr[:,j].max()) if n[l]>1 else np.nan,
                    'positive_repeats':int((values>0).sum()) if n[l]>1 else np.nan,
                    'mean_standardized_slope':g.mean(),'mean_absolute_standardized_slope':np.abs(g).mean(),
                    'positive_local_slope_fraction':np.mean(g>0),'negative_local_slope_fraction':np.mean(g<0),
                    'slope_min':g.min(),'slope_max':g.max(),'baseline_mse':baseline[:,l,m].mean(),
                    'interpretation':'no within-lineage variability' if n[l]==1 else ('primary' if eligible[l] else 'small_n_descriptive_only')})
    line_table=table('per_lineage_importance.tsv',lineage_rows)
    # Feature correlation describes potential shared signal; it does not alter importance.
    centered=d['x'].copy()
    for label in labels:
        ix=lineage==label;centered[ix]-=centered[ix].mean(axis=0)
    ix=np.isin(lineage,labels[eligible])
    corr=np.corrcoef(centered[ix],rowvar=False)
    table('within_lineage_gene_correlations.tsv',pd.DataFrame(corr,index=genes,columns=genes).rename_axis('gene').reset_index())
    correlations=[]
    for j,gene in enumerate(genes):
        others=np.arange(40)!=j;k=np.arange(40)[others][np.argmax(np.abs(corr[j,others]))]
        correlations.append({'gene':gene,'strongest_correlated_panel_gene':genes[k],'within_lineage_centered_pearson':corr[j,k]})
    table('strongest_gene_correlations.tsv',correlations)
    stability=[];cross=[];eligible_indices=np.flatnonzero(eligible)
    for m,model in enumerate(MODELS):
        for ai,a in enumerate(eligible_indices):
            for b in eligible_indices[ai+1:]:
                stability.append({'model':model,'lineage_a':labels[a],'lineage_b':labels[b],
                                  'gene_importance_rank_correlation':correlation(mu[a,m],mu[b,m])})
    for l in eligible_indices:
        cross.append({'lineage':labels[l],'n':int(n[l]),'ridge_nn_gene_rank_correlation':correlation(mu[l,0],mu[l,1]),
                      'top5_overlap':len(set(np.argsort(-mu[l,0])[:5]) & set(np.argsort(-mu[l,1])[:5]))})
    stability=table('between_lineage_rank_correlations.tsv',stability)
    table('within_lineage_model_agreement.tsv',cross)
    primary=global_table[global_table.scope=='weighted_19_lineages']
    weighted_point=aggregates['weighted_19_lineages'].mean(axis=0)
    top10=[list(genes[np.argsort(-weighted_point[m])[:10]]) for m in range(2)]
    stability_stats={model:{'median':float(v.gene_importance_rank_correlation.median()),
                              'q25':float(v.gene_importance_rank_correlation.quantile(.25)),
                              'q75':float(v.gene_importance_rank_correlation.quantile(.75)),
                              'min':float(v.gene_importance_rank_correlation.min()),'max':float(v.gene_importance_rank_correlation.max())}
                     for model in MODELS for v in [stability[stability.model==model]]}
    summary={'models':MODELS,'primary_lineages':19,'primary_lines':721,'repeats':20,
        'ridge_nn_primary_gene_rank_correlation':correlation(weighted_point[0],weighted_point[1]),
        'top10_overlap':len(set(top10[0])&set(top10[1])),'top10_genes':dict(zip(MODELS,top10)),
        'between_lineage_rank_correlation':stability_stats,
        'within_lineage_model_agreement_median':float(np.median([v['ridge_nn_gene_rank_correlation'] for v in cross])),
        'positive_pointwise_intervals':{m:primary[(primary.model==m)&(primary.conditional_ci_low>0)].gene.tolist() for m in MODELS},
        'top5_seed_rank_ranges':{m:primary[(primary.model==m)&(primary['rank']<=5)][['gene','repeat_rank_min','repeat_rank_max']].to_dict('records') for m in MODELS}}
    calc.dump(OUT/'summary.json',summary)
    validation_summary={'passed':True,'source_models':960,'exact_all_donor_receiver_substitutions':True,
        'ridge_closed_form_expectation_max_error':ridge_formula_error,'bootstrap_expansion_max_error':bootstrap_error,
        'one_prediction_per_sample_per_model_repeat':True,'all_input_and_model_hashes_verified':True,
        'max_baseline_prediction_error':max(v['baseline_prediction_max_error'] for v in validation['checks']),
        'max_native_replacement_error':max(v['native_replacement_max_error'] for v in validation['checks']),
        'max_gradient_finite_difference_error':max(v['gradient_finite_difference_max_error'] for v in validation['checks']),
        'max_two_cell_exhaustive_permutation_error':max(v['exhaustive_two_cell_permutation_max_error'] for v in validation['checks']),
        'calculation_elapsed_seconds':validation['elapsed_seconds']}
    calc.dump(OUT/'validation.json',validation_summary)
    np.savez_compressed(OUT/'importance_arrays.npz',genes=genes,labels=labels,n=n,model_names=np.array(MODELS),
        importance_mse=importance,baseline_mse=baseline,mean_standardized_gradient=lineage_gradient,
        sample_ids=d['sample_ids'],lineage=lineage,groups=d['groups'],per_cell_delta_mse=cell,standardized_gradient=grad)
    np.savez_compressed(OUT/'bootstrap_draws.npz',lineage_picks=picks,eligible_lineages=labels[eligible],**boot)
    render(primary,global_table,line_table,mu,labels,n,eligible,genes,summary,validation_summary)
    for row in json.loads((OUT/'source_manifest.json').read_text())['files']:
        assert calc.sha(calc.WS/row['path'])==row['sha256']
    calc.dump(OUT/'progress.json',{'phase':'complete','model_holdouts':960,'repeats':20,'genes':40,'validation_passed':True})
    manifest={'files':[{'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':calc.sha(p)}
                       for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts]}
    calc.dump(OUT/'artifact_manifest.json',manifest)
    archive=OUT.parent/'erastin_gene_importance_results.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for row in manifest['files']:z.write(OUT/row['path'],'erastin_gene_importance/'+row['path'])
        z.write(OUT/'artifact_manifest.json','erastin_gene_importance/artifact_manifest.json')
    import hashlib
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in manifest['files']:
            assert hashlib.sha256(z.read('erastin_gene_importance/'+row['path'])).hexdigest()==row['sha256']
    print(json.dumps({'summary':summary,'validation':validation_summary,'archive_bytes':archive.stat().st_size},indent=2))


def render(primary,all_summary,line_table,mu,labels,n,eligible,genes,summary,validation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    rank_mean=primary.groupby('gene')['rank'].mean().reindex(genes).to_numpy()
    order=np.argsort(rank_mean,kind='stable')[:15]
    fig,axes=plt.subplots(1,2,figsize=(11.5,7.8),constrained_layout=True,sharey=True,sharex=True)
    for ax,model,color in zip(axes,MODELS,['#007f8b','#9257a7']):
        v=primary[primary.model==model].set_index('gene').loc[genes[order]]
        point=v.delta_mse.to_numpy()*1000
        lo=v.conditional_ci_low.to_numpy()*1000;hi=v.conditional_ci_high.to_numpy()*1000
        ax.barh(np.arange(15),point,color=color,alpha=.8,height=.65)
        ax.errorbar(point,np.arange(15),xerr=np.stack([point-lo,hi-point]),fmt='none',ecolor='#253340',capsize=2,lw=1)
        ax.axvline(0,color='#56636a',lw=.8);ax.grid(axis='x',alpha=.18)
        ax.set(title=TITLES[model],xlabel='Expected held-out MSE increase × 1,000')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(np.arange(15),genes[order]);axes[0].invert_yaxis()
    fig.suptitle('Gene importance in unseen cancer lineages\nMean of 20 paired repeats; 721 cells in 19 lineages',fontsize=13)
    fig.supxlabel('Whiskers: pointwise conditional 95% whole-lineage bootstrap intervals. Positive = predictive benefit.',fontsize=9)
    fig.savefig(OUT/'gene_importance.png',dpi=180);fig.savefig(OUT/'gene_importance.svg');plt.close(fig)
    # Display all forty genes, ordered by average model rank, and all 19 eligible lineages.
    all_order=np.argsort(rank_mean,kind='stable')
    li=np.flatnonzero(eligible);li=li[np.argsort(-n[li],kind='stable')]
    values=mu[li][:,:,all_order]*1000
    bound=float(np.max(np.abs(values)))
    fig,axes=plt.subplots(1,2,figsize=(17.5,13),constrained_layout=True,sharey=True)
    for m,ax in enumerate(axes):
        im=ax.imshow(values[:,m,:].T,cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound),aspect='auto')
        ax.set_xticks(np.arange(len(li)),[f'{labels[l]} ({n[l]})' for l in li],rotation=65,ha='right',fontsize=8)
        ax.set_yticks(np.arange(40),genes[all_order],fontsize=9)
        ax.set_title(TITLES[MODELS[m]],fontsize=12)
    fig.colorbar(im,ax=axes,shrink=.65,label='Expected MSE increase × 1,000 (same scale in both panels)')
    fig.suptitle('All 40 genes: importance varies across held-out lineages\nRed: shuffling worsens prediction; blue: shuffling improves prediction; mean of 20 repeats',fontsize=13)
    fig.savefig(OUT/'lineage_gene_heatmap.png',dpi=180);fig.savefig(OUT/'lineage_gene_heatmap.svg');plt.close(fig)
    # Ranked-lineage heatmap allows low-magnitude lineages to remain legible.
    fig,axes=plt.subplots(1,2,figsize=(17.5,9.5),constrained_layout=True,sharey=True)
    for m,ax in enumerate(axes):
        ranks=rankdata(-mu[li,m],axis=1,method='min')[:,order].T
        im=ax.imshow(ranks,cmap='viridis_r',vmin=1,vmax=40,aspect='auto')
        for a in range(15):
            for b,l in enumerate(li):
                if mu[l,m,order[a]]<=0:ax.text(b,a,'×',ha='center',va='center',fontsize=10,color='white' if ranks[a,b]>20 else '#15212b')
        ax.set_xticks(np.arange(len(li)),[f'{labels[l]} ({n[l]})' for l in li],rotation=65,ha='right',fontsize=8)
        ax.set_yticks(np.arange(15),genes[order]);ax.set_title(TITLES[MODELS[m]])
    fig.colorbar(im,ax=axes,shrink=.65,label='Within-lineage rank among 40 genes (1 = largest benefit)')
    fig.suptitle('Top 15 overall genes: lineage-specific ranks\n× marks a nonpositive importance; all values averaged over 20 repeats',fontsize=13)
    fig.savefig(OUT/'lineage_gene_ranks.png',dpi=180);fig.savefig(OUT/'lineage_gene_ranks.svg');plt.close(fig)
    p=primary.set_index(['model','gene'])
    top=list(genes[order[:10]])
    rows=['# Gene importance for erastin sensitivity prediction','',
        'Both models prioritize SLC7A11, ACSL4 and AIFM2 in that order. GPX4 and CYB5R1 complete the same top-five set. '
        f'The full 40-gene importance rankings agree strongly between models (Spearman {summary["ridge_nn_primary_gene_rank_correlation"]:.3f}; '
        f'{summary["top10_overlap"]}/10 genes shared in their top tens). Rankings are much less uniform across lineages.','',
        'This analysis reuses every saved ridge fit and three-member neural ensemble from the 20 matched repeats. '
        'The 24 outer leave-lineage-out tests, fitted weights, hyperparameters, and expression panel are unchanged. '
        'No models were retrained, and no genes were selected using these importance results.','',
        '## How importance was measured','',
        'For each saved model, replace one gene at a time using expression values from donors in the same held-out lineage. '
        'Calculate the increase in mean squared prediction error relative to the unchanged predictions. '
        'All receiver/donor pairs are evaluated, including self-replacement. For additive squared error, this is exactly '
        'the expected loss over all random within-lineage permutations; it has no permutation Monte Carlo noise. '
        'Neural member predictions are averaged before scoring the ensemble.','',
        'Positive importance means shuffling the gene worsens prediction, so that gene helps the fitted model on those held-out cells. '
        'Negative importance means shuffling improves prediction. This measures model reliance and usefulness, not causal gene effects. '
        'The perturbation preserves each lineage’s marginal expression distribution, so the primary importance measures within-lineage predictive value '
        'and does not attribute predictions arising solely from differences between lineage means.','',
        'Primary rankings weight cells equally across the same 19 lineages with n≥10 (721 lines), then average effects across all 20 repeats. '
        'Equal-weight lineage rankings and a descriptive analysis of all 23 non-singleton lineages (744 lines) are also supplied. '
        'The singleton Cervix lineage has no within-lineage permutation information.','',
        '## Leading genes','',
        'Importance is expected MSE increase ×1,000. “Helpful lineages” counts positive mean importance across 20 repeats; '
        'it is a descriptive count, not a significance threshold. Directions use ridge coefficients and mean local neural slopes, '
        'each standardized to a one-training-SD increase in that gene. The target is negative erastin AUC: higher means greater sensitivity.','',
        '| Gene | Ridge rank | Ridge importance | NN rank | NN importance | Helpful lineages, ridge / NN | Higher expression predicts |',
        '|---|---:|---:|---:|---:|---:|---|']
    for gene in top:
        a=p.loc['ridge',gene];b=p.loc['neural_network',gene]
        direction='Greater sensitivity' if a.mean_standardized_slope>0 and b.mean_standardized_slope>0 else ('Lower sensitivity' if a.mean_standardized_slope<0 and b.mean_standardized_slope<0 else 'Different directions')
        rows.append(f'| {gene} | {int(a["rank"])} | {a.delta_mse*1000:.3f} | {int(b["rank"])} | {b.delta_mse*1000:.3f} | {int(a.positive_lineages_of_19)}/19 / {int(b.positive_lineages_of_19)}/19 | {direction} |')
    rows+=['',f'![Importance and conditional intervals]({OUT}/gene_importance.png)','',
           '## Consistency across seeds and lineages','']
    for model in MODELS:
        v=primary[primary.model==model].sort_values('rank').head(5)
        detail=', '.join(f'{r.gene} #{int(r.repeat_rank_min)}–{int(r.repeat_rank_max)}' for r in v.itertuples())
        rows.append(f'- **{TITLES[model]} across seeds:** top-five gene rank ranges: {detail}.')
    for model in MODELS:
        v=summary['between_lineage_rank_correlation'][model]
        rows.append(f'- **{TITLES[model]} across lineages:** median pairwise correlation of the 40-gene importance rankings is {v["median"]:.3f} '
                    f'(interquartile range {v["q25"]:.3f}–{v["q75"]:.3f}). These 171 lineage-pair correlations are descriptive and dependent.')
    rows+=['',f'Within a given lineage, ridge and neural gene rankings agree much more closely: median rank correlation '
           f'{summary["within_lineage_model_agreement_median"]:.3f}. Model agreement does not imply a universal ranking across cancer types.','',
           '| Gene | Ridge slope sign across 19 lineage means | NN slope sign across 19 lineage means |',
           '|---|---:|---:|']
    for gene in top[:5]:
        a=p.loc['ridge',gene];b=p.loc['neural_network',gene]
        def sign_text(v):return f'{int(v.positive_slope_lineages_of_19)} positive / {int(v.negative_slope_lineages_of_19)} negative'
        rows.append(f'| {gene} | {sign_text(a)} | {sign_text(b)} |')
    rows+=['','For example, SLC7A11 ranks first in Lung and CNS/Brain in both models, but has negative importance in Myeloid, Bone and Soft Tissue. '
           'GPX4 has a smaller overall effect but positive importance in 17/19 lineages in both models. '
           'Differences between lineage rankings can reflect biological differences, limited sample size, measurement noise, and model error; this analysis does not separate those sources.','',
           '| Held-out lineage | n | Ridge top three | Neural-network top three |',
           '|---|---:|---|---|']
    leaders=[]
    for l in li:
        gene_lists=[list(genes[np.argsort(-mu[l,m])[:3]]) for m in range(2)]
        rows.append(f'| {labels[l]} | {n[l]} | {", ".join(gene_lists[0])} | {", ".join(gene_lists[1])} |')
        leaders.append({'lineage':labels[l],'n':int(n[l]),'ridge_top3':', '.join(gene_lists[0]),'neural_network_top3':', '.join(gene_lists[1])})
    table('lineage_top_genes.tsv',leaders)
    rows+=['',f'![Lineage ranks]({OUT}/lineage_gene_ranks.png)','',
           f'[All 40 genes, absolute importance heatmap]({OUT}/lineage_gene_heatmap.png)','',
           '## Weighting and uncertainty','',
           '| Summary | Ridge top five | Neural-network top five |','|---|---|---|']
    for scope in ['weighted_19_lineages','macro_19_lineages','weighted_23_nonsingleton_lineages']:
        lists=[', '.join(all_summary[(all_summary.scope==scope)&(all_summary.model==m)].sort_values('rank').head(5).gene) for m in MODELS]
        rows.append(f'| {scope} | {lists[0]} | {lists[1]} |')
    rows+=['','Intervals use 5,000 paired resamples of the 19 complete held-out lineages, preserving every gene, model and repeat together. '
           'They condition on the saved models and observed donor distributions; models are not refitted. The intervals are pointwise and exploratory, '
           'with no adjustment for examining 40 genes. Training sets overlap between holdouts, and neither repeats nor positive-lineage counts are independent biological replications.','']
    for model in MODELS:
        rows.append(f'- {TITLES[model]} genes with pointwise primary intervals entirely above zero: '+', '.join(summary['positive_pointwise_intervals'][model])+'.')
    rows+=['','## Interpretation limits','',
           '- The ranking describes these two predictive models, the 40 measured genes, and this archived erastin-response cohort. It does not prove a gene causes erastin resistance or sensitivity.',
           '- Shuffling a gene breaks its correlation with other genes and may generate atypical expression combinations. Correlated genes may share predictive information, so small importance is not evidence of biological irrelevance. The full within-lineage-centered expression correlation matrix is supplied.',
           '- Direction is a partial model response with other features fixed. A consistent coefficient or local-gradient sign can coexist with negative held-out importance in some lineages because predictive calibration and gene–response relationships vary.',
           '- Small lineages provide imprecise individual assessments. The n≥10 cutoff and the same 19 primary lineages are inherited from the preceding comparisons, not chosen after seeing gene scores.',
           '- Gene importance is not additive across features and cannot be read as a percentage of explained erastin sensitivity. No knockout, causal mediation, external drug screen, or clinical validation was performed.',
           '- This analysis uses the 40-gene expression panel only, not BulkFormer embeddings.','',
           '## Validation and files','',
           f'All 960 saved model/holdout combinations reproduced their stored predictions (maximum absolute error {validation["max_baseline_prediction_error"]:.2g}). '
           f'Native sklearn replacement predictions were checked for all 40 genes in every holdout (maximum error {validation["max_native_replacement_error"]:.2g}). '
           f'Neural analytical gradients matched independent finite differences (maximum error {validation["max_gradient_finite_difference_error"]:.2g}). '
           'All ridge exact-permutation results additionally matched a closed-form loss calculation; exhaustive permutations of the two-cell lineage validated the all-donor identity. '
           'Training-only scalers, held-out identity exclusion, every model hash, and source-data hashes were checked.','',
           f'- [All-gene summary, ranks and intervals]({OUT}/gene_importance_summary.tsv)',
           f'- [Per-lineage gene importance and direction]({OUT}/per_lineage_importance.tsv)',
           f'- [Per-repeat importance and ranks]({OUT}/per_repeat_importance.tsv)',
           f'- [Within-lineage gene correlations]({OUT}/within_lineage_gene_correlations.tsv)',
           f'- [Registered analysis specification]({OUT}/protocol.json)',
           f'- [Validation]({OUT}/validation.json)','']
    (OUT/'REVIEW.md').write_text('\n'.join(rows))


if __name__=='__main__':main()
