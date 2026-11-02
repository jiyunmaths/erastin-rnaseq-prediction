#!/usr/bin/env python3
"""Summarize exact held-out gene influence, predictive benefit and consistency."""
import json
from pathlib import Path
import sys
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import rankdata,spearmanr

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT))
import calculate_importance as calc

def table(name,rows):
    f=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    f.to_csv(OUT/name,sep='\t',index=False,float_format='%.12g');return f

def main():
    d=np.load(OUT/'analysis_inputs.npz');labels=d['labels'];genes=d['added_genes'];names=np.concatenate([genes,np.array(['ALL_24_ADDED'])])
    n=np.array([np.sum(d['lineage']==l) for l in labels]);eligible=n>=10;li=np.flatnonzero(eligible)
    assert eligible.sum()==19 and n[eligible].sum()==721
    importance=np.empty((20,24,25));shift=np.empty_like(importance);gradient=np.empty((20,24,24));base=np.empty((20,24))
    protocol=json.loads((OUT/'protocol.json').read_text());ph=calc.sha(OUT/'protocol.json')
    assert calc.sha(OUT/'calculate_importance.py')==protocol['script_sha256']
    calc.dump(OUT/'analysis_source_manifest.json',dict(script_sha256=calc.sha(Path(__file__))))
    validation=json.loads((OUT/'calculation_validation.json').read_text());max_cell_error=0.
    for r in range(20):
        seen=np.zeros(745,int)
        for l in range(24):
            root=OUT/'folds'/f'r{r+1:02d}__l{l:02d}'
            meta=json.loads(root.with_suffix('.json').read_text());a=np.load(root.with_suffix('.npz'))
            assert meta['protocol_sha256']==ph and calc.sha(root.with_suffix('.npz'))==meta['array_sha256']
            te=a['test_rows'];seen[te]+=1;np.testing.assert_array_equal(te,np.flatnonzero(d['lineage']==labels[l]))
            np.testing.assert_allclose(a['baseline_prediction'],d['predictions'][r,te],rtol=0,atol=1e-11)
            importance[r,l]=a['importance_mse'];shift[r,l]=a['mean_squared_prediction_shift'];base[r,l]=a['baseline_mse']
            gradient[r,l]=a['standardized_gradient'].mean(axis=0)
            max_cell_error=max(max_cell_error,float(np.max(np.abs(a['per_cell_delta_mse'].mean(axis=0)-importance[r,l]))))
            np.testing.assert_allclose(a['per_cell_mean_squared_prediction_shift'].mean(axis=0),shift[r,l],rtol=0,atol=1e-14)
            np.testing.assert_allclose(base[r,l],np.mean((d['predictions'][r,te]-d['y'][te])**2),rtol=0,atol=1e-14)
        assert np.all(seen==1)
    assert np.isfinite(importance).all() and np.isfinite(shift).all() and np.isfinite(gradient).all()
    assert np.min(shift)>-1e-15 and np.max(np.abs(importance[:,n==1]))<1e-12
    mu=importance.mean(axis=0);sm=shift.mean(axis=0);gm=gradient.mean(axis=0)
    picks=np.random.default_rng(protocol['uncertainty']['seed']).integers(19,size=(5000,19))
    counts=np.array([np.bincount(v,minlength=19) for v in picks])
    weighted=counts*n[eligible];weighted=weighted/weighted.sum(axis=1,keepdims=True)
    boots={'weighted_19':np.einsum('bl,lg->bg',weighted,mu[eligible]),'macro_19':np.einsum('bl,lg->bg',counts/19,mu[eligible])}
    bootstrap_error=0.
    for k in range(25):
        expected=np.average(mu[eligible][picks[k]],axis=0,weights=n[eligible][picks[k]])
        bootstrap_error=max(bootstrap_error,float(np.max(np.abs(expected-boots['weighted_19'][k]))))
    assert bootstrap_error<1e-14
    rows=[];repeat_rows=[];lineage_rows=[]
    scopes={'weighted_19':n[eligible]/n[eligible].sum(),'macro_19':np.full(19,1/19)}
    lineage_ranks=rankdata(-mu[:,:24],axis=1,method='min')
    lineage_influence_ranks=rankdata(-sm[:,:24],axis=1,method='min')
    for scope,w in scopes.items():
        v=np.einsum('rlg,l->rg',importance[:,eligible],w);changes=np.einsum('rlg,l->rg',shift[:,eligible],w)
        grad=np.einsum('rlg,l->rg',gradient[:,eligible],w);baseline=base[:,eligible]@w
        ranks=rankdata(-v[:,:24],axis=1,method='min');overall=rankdata(-v.mean(axis=0)[:24],method='min')
        influence=rankdata(-changes.mean(axis=0)[:24],method='min')
        boot_ranks=rankdata(-boots[scope][:,:24],axis=1,method='min')
        for j,g in enumerate(names):
            lo,hi=np.percentile(boots[scope][:,j],[2.5,97.5])
            row=dict(scope=scope,gene=str(g),rank=int(overall[j]) if j<24 else None,
                delta_mse=float(v[:,j].mean()),relative_mse_increase_percent=float(100*v[:,j].mean()/baseline.mean()),
                rmse_equivalent_increase=float(np.mean(np.sqrt(baseline+v[:,j])-np.sqrt(baseline))),
                rms_prediction_change=float(np.sqrt(changes[:,j].mean())),influence_rank=int(influence[j]) if j<24 else None,
                repeat_sd=float(v[:,j].std(ddof=1)),positive_repeats=int(np.sum(v[:,j]>0)),
                rank_min=int(ranks[:,j].min()) if j<24 else None,rank_max=int(ranks[:,j].max()) if j<24 else None,
                top5_repeats=int(np.sum(ranks[:,j]<=5)) if j<24 else None,
                positive_lineages_of_19=int(np.sum(mu[eligible,j]>0)),
                positive_all20_lineages_of_19=int(np.sum(np.all(importance[:,eligible,j]>0,axis=0))),
                negative_all20_lineages_of_19=int(np.sum(np.all(importance[:,eligible,j]<0,axis=0))),
                top5_lineages_of_19=int(np.sum(lineage_ranks[eligible,j]<=5)) if j<24 else None,
                mean_standardized_slope=float(grad[:,j].mean()) if j<24 else None,
                positive_slope_lineages=int(np.sum(gm[eligible,j]>0)) if j<24 else None,
                negative_slope_lineages=int(np.sum(gm[eligible,j]<0)) if j<24 else None,
                conditional_ci_low=float(lo),conditional_ci_high=float(hi),
                bootstrap_top5_fraction=float(np.mean(boot_ranks[:,j]<=5)) if j<24 else None,
                baseline_mse=float(baseline.mean()))
            rows.append(row)
            repeat_rows.extend(dict(scope=scope,repeat=r+1,gene=str(g),delta_mse=float(v[r,j]),rms_prediction_change=float(np.sqrt(changes[r,j])),
                                    rank=int(ranks[r,j]) if j<24 else None,standardized_slope=float(grad[r,j]) if j<24 else None) for r in range(20))
    summary=table('gene_importance_summary.tsv',pd.DataFrame(rows).sort_values(['scope','rank'],na_position='last'))
    table('per_repeat_importance.tsv',repeat_rows)
    for l,label in enumerate(labels):
        ranks=rankdata(-importance[:,l,:24],axis=1,method='min')
        for j,g in enumerate(names):
            valid=n[l]>1
            lineage_rows.append(dict(lineage=str(label),n=int(n[l]),gene=str(g),eligible=bool(eligible[l]),
                delta_mse=float(mu[l,j]) if valid else None,rms_prediction_change=float(np.sqrt(sm[l,j])) if valid else None,
                relative_mse_increase_percent=float(100*mu[l,j]/base[:,l].mean()) if valid else None,
                rank=int(lineage_ranks[l,j]) if valid and j<24 else None,
                influence_rank=int(lineage_influence_ranks[l,j]) if valid and j<24 else None,
                positive_repeats=int(np.sum(importance[:,l,j]>0)) if valid else None,
                rank_min=int(ranks[:,j].min()) if valid and j<24 else None,rank_max=int(ranks[:,j].max()) if valid and j<24 else None,
                standardized_slope=float(gm[l,j]) if j<24 else None,
                baseline_mse=float(base[:,l].mean()),scope='primary' if eligible[l] else 'small_n_descriptive' if valid else 'singleton_no_permutation_information'))
    per_lineage=table('per_lineage_importance.tsv',lineage_rows)
    pairs=[]
    for ai,a in enumerate(li):
        for b in li[ai+1:]:pairs.append(dict(lineage_a=str(labels[a]),lineage_b=str(labels[b]),
            importance_rank_correlation=float(spearmanr(mu[a,:24],mu[b,:24]).statistic),
            influence_rank_correlation=float(spearmanr(sm[a,:24],sm[b,:24]).statistic)))
    pairs=table('between_lineage_rank_correlations.tsv',pairs)
    seed_pairs=[];w=n[eligible]/n[eligible].sum();seed_importance=np.einsum('rlg,l->rg',importance[:,eligible,:24],w)
    for a in range(20):
        for b in range(a+1,20):seed_pairs.append(dict(repeat_a=a+1,repeat_b=b+1,rank_correlation=float(spearmanr(seed_importance[a],seed_importance[b]).statistic)))
    seed_pairs=table('between_seed_rank_correlations.tsv',seed_pairs)
    # All 64 genes are included as potential correlated partners; no response
    # labels or importance scores determine feature group definitions.
    centered=d['x'].copy()
    for label in labels:
        ix=d['lineage']==label;centered[ix]-=centered[ix].mean(axis=0)
    corr=np.corrcoef(centered[np.isin(d['lineage'],labels[eligible])],rowvar=False)
    correlation_rows=[]
    for j,g in zip(d['added_indices'],genes):
        others=np.delete(np.arange(64),j);k=others[np.argmax(np.abs(corr[j,others]))]
        correlation_rows.append(dict(gene=str(g),strongest_panel_partner=str(d['genes'][k]),
                                     partner_added=bool(k>=40),within_lineage_centered_pearson=float(corr[j,k])))
    table('strongest_gene_correlations.tsv',correlation_rows)
    top=summary[(summary.scope=='weighted_19') & (summary.gene!='ALL_24_ADDED')].sort_values('rank')
    top_influence=top.sort_values('influence_rank')
    result=dict(primary_genes=24,primary_lineages=19,primary_samples=721,repeats=20,
        top5_predictive_genes=top.head(5).gene.tolist(),top5_influence_genes=top_influence.head(5).gene.tolist(),
        median_between_lineage_importance_rank_correlation=float(pairs.importance_rank_correlation.median()),
        median_between_lineage_influence_rank_correlation=float(pairs.influence_rank_correlation.median()),
        median_between_seed_rank_correlation=float(seed_pairs.rank_correlation.median()),
        positive_pointwise_intervals=top[top.conditional_ci_low>0].gene.tolist(),
        top5_rank_ranges=top.head(5)[['gene','rank_min','rank_max','top5_repeats']].to_dict('records'))
    calc.dump(OUT/'summary.json',result)
    check=dict(passed=True,saved_models=480,added_genes=24,joint_block=True,no_refitting=True,
        exact_within_lineage_permutation_expectation=True,model_and_input_source_hashes_verified=True,
        max_cell_aggregation_error=max_cell_error,bootstrap_expansion_error=bootstrap_error,
        **{key:max(row[key] for row in validation['checks']) for key in ['baseline_prediction_error','native_swap_error','identity_swap_error','gradient_finite_difference_error','exhaustive_two_cell_error']})
    calc.dump(OUT/'validation.json',check)
    np.savez_compressed(OUT/'importance_arrays.npz',genes=names,labels=labels,n=n,importance_mse=importance,mean_squared_prediction_shift=shift,
                        standardized_gradient=gradient,baseline_mse=base)
    np.savez_compressed(OUT/'bootstrap_draws.npz',lineage_picks=picks,**boots)
    render(summary,per_lineage,pairs,result,check)
    for source in json.loads((OUT/'source_manifest.json').read_text())['files']:assert calc.sha(calc.WS/source['path'])==source['sha256']
    assert calc.sha(Path(__file__))==json.loads((OUT/'analysis_source_manifest.json').read_text())['script_sha256']
    calc.dump(OUT/'progress.json',dict(phase='complete',saved_models=480,added_genes=24,validation_passed=True))
    archive=seal();print(json.dumps(dict(summary=result,validation=check,archive_bytes=archive.stat().st_size),indent=2))

def seal():
    manifest=dict(files=[dict(path=str(p.relative_to(OUT)),sha256=calc.sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*'))
                         if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts])
    calc.dump(OUT/'artifact_manifest.json',manifest);archive=OUT.parent/'erastin_added_gene_importance_results.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for r in manifest['files']:z.write(OUT/r['path'],'erastin_added_gene_importance/'+r['path'])
        z.write(OUT/'artifact_manifest.json','erastin_added_gene_importance/artifact_manifest.json')
    import hashlib
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for r in manifest['files']:assert hashlib.sha256(z.read('erastin_added_gene_importance/'+r['path'])).hexdigest()==r['sha256']
    return archive

def render(summary,lineages,pairs,result,validation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    primary=summary[(summary.scope=='weighted_19') & (summary.gene!='ALL_24_ADDED')].sort_values('rank')
    order=primary.gene.to_numpy();y=np.arange(24)
    fig,axes=plt.subplots(1,2,figsize=(11,9),constrained_layout=True,sharey=True)
    a=primary.set_index('gene').loc[order]
    axes[0].barh(y,a.delta_mse*1000,color='#7b5299',alpha=.85)
    axes[0].errorbar(a.delta_mse*1000,y,xerr=np.stack([(a.delta_mse-a.conditional_ci_low)*1000,(a.conditional_ci_high-a.delta_mse)*1000]),fmt='none',ecolor='#26343d',capsize=2)
    axes[0].axvline(0,color='#555',lw=.8);axes[0].set_xlabel('Expected MSE increase × 1,000')
    axes[0].set_title('Held-out predictive benefit\nWhiskers: conditional 95% lineage bootstrap')
    axes[1].barh(y,a.rms_prediction_change,color='#17828d',alpha=.85)
    axes[1].set_xlabel('RMS change in predicted negative AUC');axes[1].set_title('Influence on predictions\nIndependent of response error')
    for ax in axes:ax.grid(axis='x',alpha=.18);ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(y,order);axes[0].invert_yaxis()
    fig.suptitle('The 24 added genes in the 64-gene neural network\nExact within-lineage permutations; 20 repeats, 721 cells in 19 lineages',fontsize=13)
    fig.savefig(OUT/'added_gene_importance.png',dpi=170);fig.savefig(OUT/'added_gene_importance.svg');plt.close(fig)
    eligible=lineages[lineages.eligible & (lineages.gene!='ALL_24_ADDED')]
    labels=eligible.drop_duplicates('lineage').sort_values('n',ascending=False).lineage.tolist()
    values=eligible.pivot(index='gene',columns='lineage',values='delta_mse').loc[order,labels].to_numpy()*1000
    ranks=eligible.pivot(index='gene',columns='lineage',values='rank').loc[order,labels].to_numpy()
    fig,axes=plt.subplots(1,2,figsize=(17,10.5),constrained_layout=True,sharey=True)
    bound=float(np.max(np.abs(values)))
    im=axes[0].imshow(values,aspect='auto',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-bound,vcenter=0,vmax=bound))
    fig.colorbar(im,ax=axes[0],shrink=.6,label='Expected MSE increase × 1,000')
    im=axes[1].imshow(ranks,aspect='auto',cmap='viridis_r',vmin=1,vmax=24)
    for i in range(24):
        for j in range(19):
            if values[i,j]<=0:axes[1].text(j,i,'×',ha='center',va='center',fontsize=8,color='white' if ranks[i,j]>12 else '#17212b')
    fig.colorbar(im,ax=axes[1],shrink=.6,label='Rank among 24 additions (1 = highest)')
    for ax in axes:ax.set_xticks(range(19),labels,rotation=65,ha='right',fontsize=8)
    axes[0].set_yticks(range(24),order);axes[0].set_title('Importance magnitude: red = beneficial signal')
    axes[1].set_title('Within-lineage rank; × = nonpositive importance')
    fig.suptitle('Added-gene importance varies across unseen lineages\nEach cell averages the same 20 saved models; columns ordered by cell count',fontsize=13)
    fig.savefig(OUT/'lineage_gene_importance.png',dpi=170);fig.savefig(OUT/'lineage_gene_importance.svg');plt.close(fig)
    rows=['# Importance of the 24 added genes in the 64-gene neural network','',
        '**Top genes by held-out predictive benefit: '+', '.join(result['top5_predictive_genes'])+'.** '
        'Top genes by prediction influence: '+', '.join(result['top5_influence_genes'])+'. '
        'All rankings here are among the 24 additions, not among the complete 64-gene panel.','',
        'We reused all 480 saved outer neural-network ensembles (20 repeats × 24 lineage holdouts). '
        'Each ensemble averages its three member predictions before squared error is evaluated. No models were retrained. '
        'For each gene and holdout, every receiver/donor pair within that lineage was evaluated, including self replacements. '
        'This exactly computes the expectation of additive squared-error loss over random within-lineage permutations, without permutation Monte Carlo noise.','',
        'Primary rankings weight the 721 cells in the same 19 lineages with n≥10. Equal-lineage rankings are provided as a sensitivity analysis. '
        'The five smaller lineages are descriptive only, and the singleton has no within-lineage permutation information.','',
        '| Gene | Rank | MSE increase (%) | Positive lineages | Top-five seeds | Seed rank range | Prediction slope |',
        '|---|---:|---:|---:|---:|---|---|']
    for v in primary.itertuples():
        rows.append(f'| {v.gene} | {int(v.rank)} | {v.relative_mse_increase_percent:+.3f} | {v.positive_lineages_of_19}/19 | '
                    f'{int(v.top5_repeats)}/20 | {int(v.rank_min)}–{int(v.rank_max)} | {"higher sensitivity" if v.mean_standardized_slope>0 else "lower sensitivity"} |')
    rows+=['','MSE increase (%) means the expected MSE change after shuffling divided by the unshuffled baseline MSE. '
        'Positive importance means disrupting the gene worsens held-out accuracy. Negative importance means the disruption improves accuracy on this cohort. '
        'Influence is the RMS prediction change and can be large even for a gene with unhelpful error contributions. '
        'Direction is the mean local derivative per training SD of expression; the response is negative erastin AUC, so higher predicted response means greater sensitivity.','',
        f'![Importance]({OUT}/added_gene_importance.png)','','## Consistency across seeds and lineages','',
        f'- Median pairwise rank correlation between seed-level aggregate rankings: {result["median_between_seed_rank_correlation"]:.3f}.',
        f'- Median pairwise rank correlation between lineage-specific importance rankings: {result["median_between_lineage_importance_rank_correlation"]:.3f}.',
        f'- Median lineage-to-lineage rank correlation for prediction influence: {result["median_between_lineage_influence_rank_correlation"]:.3f}.',
        '- Pointwise conditional 95% intervals entirely above zero: '+(', '.join(result['positive_pointwise_intervals']) if result['positive_pointwise_intervals'] else 'none')+'.', '',
        'Stable aggregate ranks across seeds reflect stability of these fitted models on the same cohort. They do not imply stable importance across cancer lineages or independent biological replication. '
        'The per-lineage table includes positive-repeat counts, rank ranges, direction and influence for every gene.','',
        f'![Lineage importance]({OUT}/lineage_gene_importance.png)','','## Joint contribution of the added block','']
    macro=summary[(summary.scope=='macro_19') & (summary.gene!='ALL_24_ADDED')].sort_values('rank')
    leading=str(primary.iloc[0].gene)
    negative=lineages[(lineages.gene==leading) & lineages.eligible & (lineages.delta_mse<0)].lineage.tolist()
    context=['## Weighting and lineage exceptions','',
        'Giving each lineage equal weight changes the top five to '+', '.join(macro.head(5).gene)+'. '
        'The first two genes remain '+', '.join(macro.head(2).gene)+', while the rest of the ranking depends more on lineage weighting.','',
        f'The leading gene, {leading}, has negative predictive importance in '+(', '.join(negative) if negative else 'none of the eligible lineages')+'. '
        'Negative importance means its shuffled predictions are more accurate on those held-out cells; it does not establish an opposite biological gene effect.','']
    insert=rows.index('## Joint contribution of the added block')
    rows[insert:insert]=context
    for scope in ['weighted_19','macro_19']:
        v=summary[(summary.scope==scope)&(summary.gene=='ALL_24_ADDED')].iloc[0]
        rows.append(f'- {scope}: joint permutation of the 24 additions changes MSE by {v.delta_mse:+.6f} '
                    f'({v.relative_mse_increase_percent:+.2f}%; conditional 95% interval [{v.conditional_ci_low:+.6f}, {v.conditional_ci_high:+.6f}]); '
                    f'positive importance in {v.positive_lineages_of_19}/19 lineages.')
    rows+=['','The joint permutation preserves dependence among the added genes while disrupting their links to the original 40 genes and response. '
        'It measures reliance of the fitted 64-gene model on this block. It does not decompose the improvement over a separately trained 40-gene model, '
        'and individual importance values must not be summed as independent contributions.','',
        '## Limits and validation','',
        '- All within-lineage substitutions preserve the marginal expression distribution of the shuffled gene. They can disrupt correlations with other genes and produce atypical combinations. The strongest within-lineage correlated partner of each addition is reported separately.',
        '- Low or negative importance may reflect redundant information, regularization, limited within-lineage variation or estimation error. It is not evidence of biological irrelevance or a causal gene effect.',
        '- Bootstrap intervals resample whole eligible lineages, condition on the saved models and donor distributions, and do not refit models. Intervals and rankings are exploratory and not adjusted for multiple comparisons.',
        '- The same cohort has supported several model comparisons; no external screen or clinical validation was added.',
        f'- All 480 baseline predictions reconstructed (maximum error {validation["baseline_prediction_error"]:.3g}). Native replacement checks, finite-difference gradients and exhaustive two-cell permutations passed.',
        f'- Maximum native swap prediction discrepancy: {validation["native_swap_error"]:.3g}; maximum gradient discrepancy: {validation["gradient_finite_difference_error"]:.3g}.','',
        '## Files','',f'- [All gene rankings and intervals]({OUT}/gene_importance_summary.tsv)',
        f'- [Per-lineage results]({OUT}/per_lineage_importance.tsv)',f'- [Per-seed results]({OUT}/per_repeat_importance.tsv)',
        f'- [Correlated genes]({OUT}/strongest_gene_correlations.tsv)',f'- [Protocol]({OUT}/protocol.json)',f'- [Validation]({OUT}/validation.json)','']
    (OUT/'REVIEW.md').write_text('\n'.join(rows))

if __name__=='__main__':main()
