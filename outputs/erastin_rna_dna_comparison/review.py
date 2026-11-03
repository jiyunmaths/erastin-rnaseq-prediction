#!/usr/bin/env python3
"""Audit matched-cohort nested fits, then evaluate paired held-out predictions."""
import json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator,FuncFormatter
import numpy as np
import pandas as pd
import run_benchmark as b
import modeling as m
import evaluate as ev
from make_splits import inner_rows

OUT=b.OUT
LABELS={'ridge':'Ridge','elastic_net':'Elastic Net','neural_network':'NN'}
ARM_LABELS={'rna':'RNA64','rna_mutation':'RNA + mutation','rna_cnv':'RNA + copy number','rna_mutation_cnv':'RNA + both'}
COLORS=['#61666d','#d58528','#238c9a','#865fac']


def table(name,data):
    frame=data if isinstance(data,pd.DataFrame) else pd.DataFrame(data)
    frame.to_csv(OUT/name,sep='\t',index=False,float_format='%.12g')
    return frame


def old_predict(model,x,family):
    if family=='ridge':return ((x-model['mean'])/model['scale'])@model['coef']+model['intercept_after_scaling']
    z=model['x_scaler'].transform(x)
    if family=='elastic_net':return model['y_scaler'].inverse_transform(model['model'].predict(z)[:,None]).ravel()
    return np.mean([model['y_scaler'].inverse_transform(a.predict(z)[:,None]).ravel() for a in model['members']],axis=0)


def audit(ds,r,ph):
    s=b.STATE[ds];root=OUT/ds/'folds';complete=json.load(open(root/f'repeat__r{r+1:02d}_complete.json'))
    assert complete['protocol_sha256']==ph and complete['models']==12
    for f in complete['files']:assert b.sha(OUT/f['path'])==f['sha256']
    tr,te=np.flatnonzero(~s['test'][r]),np.flatnonzero(s['test'][r]);original=np.load(b.BASE[ds]/'shared_splits.npz')
    np.testing.assert_array_equal(s['test'][r],original['outer_test'][r,s['original_rows']])
    np.testing.assert_array_equal(s['inner'][r],original['inner_fold'][r,s['original_rows']])
    assert set(s['group_ids'][tr]).isdisjoint(s['group_ids'][te])
    assert set(s['lineage'][tr])==set(s['lineage'][te])==set(s['lineage'])
    preds={};params=[];features=[];checks=[]
    for family in b.FAMILIES:
        for arm in m.ARMS:
            name=b.arm_name(family,arm);rr=root/f'{name}__r{r+1:02d}';md=json.load(open(rr.with_suffix('.json')));a=np.load(rr.with_suffix('.npz'))
            assert md['protocol_sha256']==ph and md['npz_sha256']==b.sha(rr.with_suffix('.npz'))
            np.testing.assert_array_equal(a['train_rows'],tr);np.testing.assert_array_equal(a['test_rows'],te)
            grid=m.grid(family,arm,s['source']);expectedloss=np.empty((len(grid),5));maxscale=0.;maxkkt=0.;maxiter=0;extended=0
            for fold in range(5):
                train,val=inner_rows(s['inner'][r],fold);inner=root/(rr.name+f'__inner{fold}')
                aa=np.load(inner.with_suffix('.npz'));mm=json.load(open(inner.with_suffix('.json')))
                assert mm['protocol_sha256']==ph and mm['npz_sha256']==b.sha(inner.with_suffix('.npz'))
                np.testing.assert_array_equal(aa['training_rows'],train);np.testing.assert_array_equal(aa['validation_rows'],val)
                assert set(train).isdisjoint(val) and set(train)|set(val)==set(tr)
                assert set(s['group_ids'][train]).isdisjoint(s['group_ids'][val])
                assert set(s['lineage'][train])==set(s['lineage'])
                pp=m.OmicsPreprocessor(arm).fit(*(s[k][train] for k in ['rna','mutation','cnv']))
                dd=mm['preprocessing'];assert dd['training_n']==len(train)
                for key in ['mutation_keep','cnv_keep']:np.testing.assert_array_equal(dd[key],getattr(pp,key))
                np.testing.assert_array_equal(dd['feature_names'],pp.names(s))
                for block in ['mutation','cnv']:
                    names=s['mutation_features'] if block=='mutation' else s['cnv_features']
                    for j,g in enumerate(names):features.append(dict(assay=ds,repeat=r+1,family=family,arm=arm,context='inner',fold=fold,feature=str(g),included=bool(j in getattr(pp,block+'_keep'))))
                if not md['reused']:
                    for key,v in [('rna_mean',pp.rna_scaler.mean_),('rna_scale',pp.rna_scaler.scale_)]:
                        error=float(np.max(np.abs(np.array(dd[key])-v)));maxscale=max(maxscale,error);assert error<1e-12
                    for d in dd['diagnostics']:
                        if family=='elastic_net':
                            assert all(v['max_kkt_residual']<1e-6 for v in d['fit'])
                            maxkkt=max(maxkkt,max(v['max_kkt_residual'] for v in d['fit']))
                            maxiter=max(maxiter,max(v['max_iterations'] for v in d['fit']))
                            extended+=sum(v['iteration_limit']>500000 for v in d['fit'])
                np.testing.assert_array_equal(aa['validation_predictions'],a['inner_validation_predictions'][:,val])
                expectedloss[:,fold]=np.sqrt(np.mean((aa['validation_predictions']-s['y'][val][None,:])**2,axis=1))
            np.testing.assert_allclose(expectedloss,a['inner_fold_rmse'],rtol=0,atol=1e-12)
            eligible=s['inner'][r]>=0
            assert np.isfinite(a['inner_validation_predictions'][:,eligible]).all() and np.isnan(a['inner_validation_predictions'][:,~eligible]).all()
            chosen=int(np.argmin(expectedloss.mean(axis=1)));assert md['selected_candidate']==chosen and md['selected_parameters']==grid[chosen]
            assert abs(md['mean_inner_rmse']-expectedloss[chosen].mean())<1e-12
            pp=m.OmicsPreprocessor(arm).fit(*(s[k][tr] for k in ['rna','mutation','cnv']))
            np.testing.assert_array_equal(a['feature_names'],pp.names(s));np.testing.assert_array_equal(a['feature_names'][:64],s['genes'])
            for block in ['mutation','cnv']:
                names=s['mutation_features'] if block=='mutation' else s['cnv_features']
                for j,g in enumerate(names):features.append(dict(assay=ds,repeat=r+1,family=family,arm=arm,context='outer',fold=-1,feature=str(g),included=bool(j in getattr(pp,block+'_keep'))))
            if md['reused']:
                assert ds=='PRISM' and arm=='rna'
                source=b.WS/md['source_model'];orig=np.load(source.with_suffix('.npz'));model=joblib.load(source.with_suffix('.joblib'))
                np.testing.assert_array_equal(orig['test_prediction'],a['test_prediction'])
                np.testing.assert_array_equal(orig['inner_validation_predictions'],a['inner_validation_predictions'])
                p=old_predict(model,s['rna'][te],family)
            else:
                assert b.sha(rr.with_suffix('.joblib'))==md['joblib_sha256'];model=joblib.load(rr.with_suffix('.joblib'))
                assert model['parameters']==grid[chosen] and model['family']==family and model['arm']==arm
                assert joblib.hash(pp)==joblib.hash(model['preprocessor'])
                if family=='elastic_net':assert md['fit_diagnostics']['kkt_residual']<1e-6
                if family=='neural_network':
                    for member,seed in zip(model['members'],s['nn_seeds'][r]):
                        assert member.random_state==int(seed) and member.alpha==grid[chosen]['alpha']
                        assert member.hidden_layer_sizes==(grid[chosen]['width'],)
                p=m.predict(model,*(s[k][te] for k in ['rna','mutation','cnv']))
            error=float(np.max(np.abs(p-a['test_prediction'])));assert error<1e-11
            preds[name]=a['test_prediction'];params.append(dict(assay=ds,repeat=r+1,family=family,arm=arm,reused=md['reused'],features=len(a['feature_names']),**grid[chosen]))
            checks.append(dict(model=name,prediction_max_error=error,training_scale_error=maxscale,inner_folds=5,reused=md['reused'],
                max_elastic_net_path_kkt=maxkkt,max_elastic_net_path_iterations=maxiter,extended_elastic_net_paths=extended))
    return preds,params,features,dict(assay=ds,repeat=r+1,models=checks)


def analyze(ds,preds,protocol):
    s=b.STATE[ds];pooled,lineage,macro=ev.score(s['y'],preds,s['lineage'],s['test']);values={};summary=[]
    for arm in preds:
        v={'pooled_'+k:pooled[pooled.model==arm].sort_values('repeat')[k].to_numpy() for k in ['rmse','spearman','pearson','mae','r2']}
        v.update({k:macro[macro.model==arm].sort_values('repeat')[k].to_numpy() for k in ['macro_all_rmse','macro_large_rmse']});values[arm]=v
        family,aa=arm.split('__');row=dict(assay=ds,model=arm,family=family,arm=aa,samples=len(s['y']))
        for key,vv in v.items():row[key+'_mean']=float(ev.finite_mean(vv));row[key+'_sd']=float(np.std(vv[np.isfinite(vv)],ddof=1))
        summary.append(row)
    bootstrap_check=ev.validate_bootstrap(s['y'],preds,s['lineage'],s['test'],protocol['uncertainty']['seed'])
    draw,perdraw=ev.bootstrap(s['y'],preds,s['lineage'],s['test'],protocol['uncertainty']['count'],protocol['uncertainty']['seed'])
    pairs=[];effects=[]
    for first,ref in protocol['primary_comparisons']+protocol['secondary_comparisons']:
        for metric in ev.KEYS:
            sign=-1 if metric.endswith(('rmse','mae')) else 1
            change=sign*(values[first][metric]-values[ref][metric]);boot=ev.finite_mean(sign*(perdraw[first][metric]-perdraw[ref][metric]),axis=1)
            lo,hi=np.percentile(boot[np.isfinite(boot)],[2.5,97.5])
            pairs.append(dict(assay=ds,first_model=first,reference_model=ref,metric=metric,improvement=float(ev.finite_mean(change)),conditional_ci_low=float(lo),conditional_ci_high=float(hi),favorable_repeats=int(np.sum(change>1e-12)),tied_repeats=int(np.sum(np.abs(change)<=1e-12)),defined_repeats=int(np.isfinite(change).sum())))
        for label in np.unique(s['lineage']):
            a=lineage[(lineage.model==first)&(lineage.lineage==label)].sort_values('repeat');r=lineage[(lineage.model==ref)&(lineage.lineage==label)].sort_values('repeat')
            effects.append(dict(assay=ds,first_model=first,reference_model=ref,lineage=str(label),cohort_n=int(a.cohort_n.iloc[0]),rmse_improvement=float(np.mean(r.rmse.to_numpy()-a.rmse.to_numpy())),favorable_repeats=int(np.sum(a.rmse.to_numpy()<r.rmse.to_numpy()-1e-12))))
    for d in [pooled,lineage,macro]:d.insert(0,'assay',ds)
    np.savez_compressed(OUT/ds/'predictions.npz',**preds,sample_ids=s['sample_ids'],group_ids=s['group_ids'],lineage=s['lineage'],outer_test=s['test'])
    np.savez_compressed(OUT/ds/'bootstrap_draws.npz',**{arm+'__'+k:v for arm,d in draw.items() for k,v in d.items()})
    return summary,pairs,effects,pooled,lineage,macro,bootstrap_check


def figures(pooled,paired):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    fig,axes=plt.subplots(2,3,figsize=(13,7.7),sharey='row')
    for row,ds in enumerate(b.BASE):
        for col,f in enumerate(b.FAMILIES):
            ax=axes[row,col];d=pooled[pooled.assay==ds];v=np.stack([d[d.model==b.arm_name(f,a)].sort_values('repeat').rmse.to_numpy() for a in m.ARMS])
            for r in range(20):ax.plot(range(4),v[:,r],color='#aaaaaa',alpha=.3,lw=.7)
            for j,vv in enumerate(v):
                ax.scatter(np.full(20,j),vv,color=COLORS[j],s=12,alpha=.5)
                ax.scatter(j,np.mean(vv),color=COLORS[j],edgecolor='black',marker='D',s=55,zorder=4)
            ax.set_xticks(range(4),['RNA64','+ Mut','+ CN','+ Both']);ax.set_xlim(-.35,3.35);ax.grid(axis='y',alpha=.2)
            ax.set_title(ds+' · '+LABELS[f],loc='left',weight='bold');ax.set_ylabel('Test RMSE (lower is better)')
    fig.suptitle('Does adding DNA improve erastin prediction?',fontsize=16,weight='bold',y=.99)
    fig.text(.5,.02,'20 shared lineage-stratified holdouts · CTRP n=739; PRISM n=386 · same cohort in all four arms\nDiamonds: mean RMSE; lines: paired splits. Mutation = 6 annotated features; copy number = 66 genes.',ha='center',fontsize=9)
    fig.subplots_adjust(top=.91,bottom=.14,left=.07,right=.99,hspace=.35,wspace=.3)
    for ext in ['png','svg']:fig.savefig(OUT/('performance_comparison.'+ext),dpi=160,facecolor='white')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(12.5,6.5),sharex=True)
    for ax,ds in zip(axes,b.BASE):
        d=paired[(paired.assay==ds)&(paired.metric=='pooled_rmse')]
        d=d[d.apply(lambda r:r.first_model.split('__')[0]==r.reference_model.split('__')[0],axis=1)]
        for y,r in enumerate(d.itertuples()):
            arm=r.first_model.split('__')[1];c=COLORS[m.ARMS.index(arm)]
            ax.hlines(y,r.conditional_ci_low,r.conditional_ci_high,color=c,lw=2);ax.scatter(r.improvement,y,color=c,s=40)
        ax.set_yticks(range(len(d)),[LABELS[r.first_model.split('__')[0]]+' · '+ARM_LABELS[r.first_model.split('__')[1]].replace('RNA + ','') for r in d.itertuples()]);ax.invert_yaxis()
        ax.axvline(0,color='#555555',ls='--',lw=.8);ax.grid(axis='x',alpha=.2);ax.set_title(ds,weight='bold');ax.set_xlabel('RMSE improvement versus RNA64 (×10⁻³)')
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda value,position:f'{value*1000:g}'))
    fig.suptitle('DNA additions versus the matched RNA-only control',weight='bold',fontsize=15,y=.98)
    fig.text(.5,.025,'Positive favors DNA additions. Conditional pointwise 95% paired whole-lineage bootstrap intervals.\nIntervals condition on fixed fits and overlapping repeats; no multiple-comparison adjustment.',ha='center',fontsize=9)
    fig.subplots_adjust(left=.18,right=.99,top=.88,bottom=.14,wspace=.6)
    for ext in ['png','svg']:fig.savefig(OUT/('paired_rmse_improvement.'+ext),dpi=160,facecolor='white')
    plt.close(fig)


def report(summary,paired,freq,protocol):
    primary=paired[(paired.metric=='pooled_rmse')&paired.apply(lambda r:r.first_model.split('__')[0]==r.reference_model.split('__')[0],axis=1)]
    recommendation=(f'**Recommendation:** retain the individual RNA-only 64-gene models as the default, and keep the CTRP copy-number model as an exploratory candidate. None of the {len(primary)} primary DNA-versus-RNA pooled-RMSE comparisons has a conditional 95% interval entirely above zero. Some CTRP secondary metrics favor copy-number additions; no clear PRISM primary-metric gain is established.'
        if not (primary.conditional_ci_low>0).any() else
        '**Recommendation:** treat favorable DNA arms as exploratory candidates for independent validation; the current RNA-only registry remains unchanged.')
    lines=['# RNA64 plus mutation and copy-number benchmark','',
        'Four arms were evaluated for ridge, Elastic Net and the retained small neural-network family on all 20 shared nested splits, separately for CTRP and PRISM. All arms retain the original 64 RNA genes. Training is unweighted and excludes culture and lineage prediction covariates. The canonical RNA-only model registry is unchanged.','',
        recommendation,'',
        '## Matched cohorts and validation','',
        '- **CTRP:** 739 of the original 744 model rows (728 identity groups), with all 23 lineages retained. Five rows lacked mutation matrix coverage. RNA-only models were refit on these same 739 rows. Their performance must not be compared directly with the historical 744-row baseline to infer a DNA effect.',
        '- **PRISM:** all 386 original model rows (379 identity groups) and 20 lineages. The 60 original RNA-only fits and their inner/outer predictions were reused exactly.',
        '- Original outer test masks and five inner-fold assignments were projected onto the matched cohorts without reassignment. Every evaluated lineage occurs in both outer train and test; identity groups never cross folds. This evaluates new cell lines in represented lineages, not unseen-lineage prediction.',
        '- Scalers, mutation frequency filters, copy-number medians and constant-feature filters are fitted only on the current training fold. Hyperparameters and block scales are selected by mean five-fold inner RMSE. Outer outcomes do not participate in preprocessing or tuning.','',
        '## Held-out performance','',
        'Values are means across 20 overlapping holdouts, not independent biological replicates. RMSE is in the original assay-specific negative-AUC units; lower is better. Response values are oriented so larger means greater sensitivity.','',
        '| Assay | Family | Inputs | RMSE | Spearman | R2 | Equal-lineage RMSE |','|---|---|---|---:|---:|---:|---:|']
    for r in summary.itertuples():lines.append(f'| {r.assay} | {LABELS[r.family]} | {ARM_LABELS[r.arm]} | {r.pooled_rmse_mean:.6f} | {r.pooled_spearman_mean:.4f} | {r.pooled_r2_mean:.4f} | {r.macro_all_rmse_mean:.6f} |')
    primary=paired[(paired.metric=='pooled_rmse')&paired.apply(lambda r:r.first_model.split('__')[0]==r.reference_model.split('__')[0],axis=1)]
    lines+=['','## Paired effect of adding DNA','',
        'Positive improvement means lower RMSE than the same-family RNA-only control. Intervals are conditional pointwise 95% paired whole-lineage bootstrap intervals, with 5,000 draws. Each draw preserves all models and overlapping repeats together. They exclude retraining uncertainty and are not corrected for the multiple comparisons.','',
        '| Assay | Family | Addition | RMSE improvement [95% interval] | Relative improvement | Better splits /20 |','|---|---|---|---:|---:|---:|']
    for r in primary.itertuples():
        family,arm=r.first_model.split('__');base=summary[(summary.assay==r.assay)&(summary.model==r.reference_model)].pooled_rmse_mean.iloc[0]
        lines.append(f'| {r.assay} | {LABELS[family]} | {ARM_LABELS[arm].replace("RNA + ","")} | {r.improvement:.6f} [{r.conditional_ci_low:.6f}, {r.conditional_ci_high:.6f}] | {100*r.improvement/base:.2f}% | {r.favorable_repeats} |')
    lines+=['','![Test RMSE on shared splits](performance_comparison.png)','','![Paired improvement intervals](paired_rmse_improvement.png)','','## Interpretation','']
    for ds in b.BASE:
        d=summary[summary.assay==ds].sort_values('pooled_rmse_mean');best=d.iloc[0]
        lines.append(f'- **{ds}:** the lowest observed mean RMSE is {best.pooled_rmse_mean:.6f}, from {LABELS[best.family]} with {ARM_LABELS[best.arm]}. This is an exploratory descriptive ranking across the tested arms.')
        for r in primary[primary.assay==ds].itertuples():
            family,arm=r.first_model.split('__');conclusion='conditional interval above zero' if r.conditional_ci_low>0 else 'conditional interval below zero' if r.conditional_ci_high<0 else 'interval spans zero'
            lines.append(f'  - {LABELS[family]}, {ARM_LABELS[arm]}: {conclusion}; improved in {r.favorable_repeats}/20 splits.')
    lines+=['','Secondary metrics provide a more favorable CTRP signal than pooled RMSE. For the NN copy-number arm, the MAE improvement is 0.000691 (conditional 95% interval 0.000095 to 0.001250), and RMSE averaged equally across the larger lineages improves by 0.000595 (0.000117 to 0.001116). The NN arm with both DNA blocks improves RMSE averaged equally across all lineages by 0.000517 (0.000057 to 0.001008). These are secondary, pointwise comparisons without multiplicity adjustment. They support further evaluation of the CTRP copy-number candidate, not an established primary-metric gain.',
        'The NN copy-number arm has lower mean per-lineage RMSE in 15/23 CTRP lineages and 10/20 PRISM lineages. Thus 19/20 favorable CTRP split results do not imply a uniform benefit across cancer types. See `lineage_effects.tsv` for effect variation and `lineage_win_counts.tsv` for compact counts. Small lineages have few test cells; their directional counts are descriptive.','',
        '## DNA definitions and coverage','',
        'All DNA data come from the official [DepMap 24Q4 Public v1 release](https://plus.figshare.com/articles/dataset/DepMap_24Q4_Public/27993248), matching the pinned RNA release. Gene symbol and Entrez columns were verified against the RNA header. Files were verified against official MD5 values and saved SHA256 checksums.','',
        '**Mutation:** six prespecified features: KEAP1 LikelyLoF, NFE2L2 hotspot, BAP1 LikelyLoF, TP53 LikelyLoF, TP53 hotspot and KRAS hotspot. Official 1/2 calls become carrier=1; finite zero remains zero. Missing data never become wild type. Genotype 2 denotes summed allele frequency above 0.95 and is not called biallelic loss. LikelyLoF/hotspot are annotation categories, not experimentally established function; TP53 categories can overlap.',
        'Each fit requires at least five carriers and five non-carriers. Binary features are centered without variance scaling, to avoid amplifying rare carriers, and their block multiplier (1 or 2) is tuned inside inner CV. Rare classes have limited power, particularly BAP1.','',
        '| Assay | Feature | Matched-cohort carriers | Included in outer training fits /20 |','|---|---|---:|---:|']
    cov=json.load(open(OUT/'coverage_summary.json'))
    for ds in b.BASE:
        for feat,count in cov['assays'][ds]['mutation_positive_counts'].items():
            ff=freq[(freq.assay==ds)&(freq.family=='ridge')&(freq.arm=='rna_mutation_cnv')&(freq.feature==feat)]
            lines.append(f'| {ds} | {feat} | {count} | {int(ff.outer_included_fits.iloc[0])} |')
    lines+=['','**Copy number:** the 63 available RNA genes plus KEAP1, TP53 and KRAS, for 66 features. CBS is absent from the official copy-number header and remains RNA-only; no alias is substituted. Relative copy ratios are transformed as `log2(x+1)`. No amplification, deep-deletion or biallelic state is inferred. Training medians fill missing values, training constants are removed, and standardized copy-number features receive an inner-tuned block multiplier of 0.25 or 1.',
        'The primary cohort requires all six genotypes and at least 80% available copy-number values per sample. A copy-number feature also requires at least 80% training-fold coverage. In these matched cohorts, all selected copy-number entries are finite, so no values actually require imputation. The retained RNA64 block is always present.','',
        '## Numerical verification','',
        'A training-only preflight initially found Elastic Net convergence warnings at 50,000 iterations and in a cold native refit at 500,000. Before any production fit or outer-result inspection, fixed convergence-only retries up to 5 million iterations were added, preserving the candidate grid, objective and tolerance. The earlier protocols are archived locally; the amendment is recorded in the frozen protocol and `grid_amendment.json`. Final solver warnings still cause failure. Baseline first-fold candidate predictions and the first outer RNA-only refits were checked against the original PRISM fits. Every production outer prediction is reproduced from its saved model; all inner-fold losses, selected candidates, current-training preprocessors and file checksums are audited.','',
        '## Limits and next decision','',
        'The cohorts have been inspected repeatedly in earlier experiments, so correct nesting does not make this a fresh confirmatory study. A favorable arm needs validation on new response data or a prospective, previously untouched cohort. The two drug-screen cohorts overlap in cell lines and are not independent biological replication. Effect intervals condition on fitted models and omit training variability; 20 seed repeats are not independent cohorts.',
        'This compact mutation panel is not a genome-wide SNV/indel search. Gene-level annotations omit allele-specific effects, zygosity and functional context. Binary mutations are centered without variance standardization and only block multipliers 1 and 2 are tested; alternative encodings and substantially larger mutation weights are not evaluated. Copy number and RNA may overlap in information. A null result applies to these feature definitions and tuning grids, not to all multi-omics models. DNA and RNA can come from different collection profiles despite release matching. Original CTRP endpoint provenance and PRISM response-processing limitations remain.','',
        '## Artifacts','',
        '- [Summary](summary.tsv)','- [Paired comparisons](paired_comparisons.tsv)','- [Coverage](coverage_summary.json)','- [Excluded lines](excluded_samples.tsv)','- [Gene mapping](gene_mapping.tsv)','- [Official DNA download links and checksums](references.json)','- [Mutation/CN feature inclusion](feature_inclusion_frequency.tsv)','- [Lineage improvement counts](lineage_win_counts.tsv)','- [Chosen settings](selected_parameters.tsv)','- [Validation](validation.json)','- [Frozen protocol](protocol.json)','- [Source checksums](source_manifest.json)','',
        '## Reproduction','',
        'Run from the repository root in the pinned scikit-learn 1.5.2 environment. The commands below verify/reuse existing checkpoints and rebuild the report on this completed workspace. Raw matrices, prepared arrays, fitted binaries and fold-level prediction arrays are ignored by Git. The source manifest records their checksums.','',
        'Initial acquisition and preparation are implemented in `download_data.py` and `prepare_data.py`. They belong before protocol freezing for a fresh experiment; do not rerun them over the frozen download manifests or prepared inputs. A fresh-environment reproduction also requires reconstructing the pinned source RNA cohorts and their saved splits/models, as documented by the retained-model registry and source manifests.','',
        '```bash','export PYTHONPATH="$PWD/work/bulkformer/packages"','export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1','export MPLCONFIGDIR="$PWD/work/bulkformer/matplotlib_cache"','export PYTHONDONTWRITEBYTECODE=1','PY=/home/sji/miniforge3/envs/deeplearning/bin/python','$PY outputs/erastin_rna_dna_comparison/run_benchmark.py','$PY outputs/erastin_rna_dna_comparison/review.py','```','']
    (OUT/'REVIEW.md').write_text('\n'.join(lines))


def main():
    b.initialize();b.verify();protocol=json.load(open(OUT/'protocol.json'));ph=b.sha(OUT/'protocol.json')
    assert json.load(open(OUT/'completion.json'))['complete']
    summaries=[];paired=[];effects=[];pooled=[];lineages=[];macros=[];checks=[];features=[];parameters=[];bootchecks=[]
    registry=json.load(open(b.WS/'outputs/retained_64_gene_models/model_registry.json'))
    for ds,s in b.STATE.items():
        preds={b.arm_name(f,a):np.full(s['test'].shape,np.nan) for f in b.FAMILIES for a in m.ARMS}
        for r in range(20):
            pp,pa,ff,cc=audit(ds,r,ph);parameters.extend(pa);features.extend(ff);checks.append(cc)
            for name,p in pp.items():preds[name][r,s['test'][r]]=p
        su,pa,ef,po,li,ma,bc=analyze(ds,preds,protocol)
        if ds=='PRISM':
            for family in b.FAMILIES:
                ours=next(v for v in su if v['model']==b.arm_name(family,'rna'));original=registry['cohorts'][ds]['models'][family]['mean_heldout_metrics']
                for our,their in [('pooled_rmse_mean','rmse'),('pooled_spearman_mean','spearman'),('pooled_r2_mean','r2'),('macro_all_rmse_mean','equal_lineage_rmse')]:assert abs(ours[our]-original[their])<1e-10
        summaries.extend(su);paired.extend(pa);effects.extend(ef);pooled.append(po);lineages.append(li);macros.append(ma);bootchecks.append(dict(assay=ds,**bc))
        print(ds+': all 240 model arms audited; metrics and bootstrap complete.',flush=True)
    summary=table('summary.tsv',summaries);pair=table('paired_comparisons.tsv',paired);eff=table('lineage_effects.tsv',effects)
    pooled=table('pooled_metrics.tsv',pd.concat(pooled));table('lineage_metrics.tsv',pd.concat(lineages));table('macro_metrics.tsv',pd.concat(macros));params=table('selected_parameters.tsv',parameters)
    parameter_counts=params.groupby(['assay','family','arm','mutation_scale','cnv_scale','alpha','width','l1_ratio'],dropna=False).size().reset_index(name='fits');table('parameter_counts.tsv',parameter_counts)
    feature=pd.DataFrame(features);freq=[]
    for keys,d in feature.groupby(['assay','family','arm','feature'],sort=False):
        ff=d[d.context=='outer'];ii=d[d.context=='inner']
        freq.append(dict(zip(['assay','family','arm','feature'],keys),outer_included_fits=int(ff.included.sum()),outer_fits=len(ff),inner_included_contexts=int(ii.included.sum()),inner_contexts=len(ii)))
    freq=table('feature_inclusion_frequency.tsv',freq)
    wins=[]
    for keys,d in eff.groupby(['assay','first_model','reference_model'],sort=False):wins.append(dict(zip(['assay','first_model','reference_model'],keys),lineages=len(d),favorable_lineages=int((d.rmse_improvement>1e-12).sum()),unfavorable_lineages=int((d.rmse_improvement< -1e-12).sum()),median_lineage_rmse_improvement=float(d.rmse_improvement.median())))
    table('lineage_win_counts.tsv',wins);figures(pooled,pair);report(summary,pair,freq,protocol);b.verify()
    source_manifest=json.load(open(OUT/'source_manifest.json'))
    for source in source_manifest['files']:assert b.sha(b.WS/source['path'])==source['sha256']
    validation=dict(passed=True,protocol_sha256=ph,repeat_comparisons=40,outer_model_arms=480,new_outer_models=420,reused_PRISM_RNA_models=60,
        identical_cells_and_splits_across_all_arms=True,original_masks_projected_exactly=True,all_inner_assignments_preserved=True,
        no_identity_group_leakage=True,every_evaluated_lineage_in_both_train_and_test=True,all_RNA64_genes_retained=True,
        all_inner_losses_and_selected_candidates_verified=True,all_current_training_preprocessors_verified=True,
        mutation_missing_never_wild_type=True,PRISM_original_RNA_predictions_reused_exactly=True,PRISM_RNA_metrics_match_registry=True,
        CTRP_RNA_controls_refit_on_matched_739=True,all_source_hashes_reverified_after_analysis=True,
        matched_CNV_entries_all_finite=all(np.isfinite(s['cnv']).all() for s in b.STATE.values()),bootstrap_checks=bootchecks,checks=checks,analysis_script_sha256=b.sha(Path(__file__)))
    b.dump(OUT/'validation.json',validation)
    b.dump(OUT/'final_summary.json',dict(summary=summary.to_dict('records'),paired_rmse=pair[pair.metric=='pooled_rmse'].to_dict('records')))
    b.dump(OUT/'progress.json',dict(phase='complete',validation_passed=True,repeat_comparisons=40,outer_model_arms=480))
    files=[dict(path=str(p.relative_to(OUT)),sha256=b.sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts]
    b.dump(OUT/'artifact_manifest.json',dict(files=files));print(summary[['assay','family','arm','pooled_rmse_mean','pooled_spearman_mean','pooled_r2_mean']].to_string(index=False),flush=True)
    print(pair[pair.metric=='pooled_rmse'].to_string(index=False),flush=True);print('RNA_DNA_REVIEW_COMPLETE',flush=True)


if __name__=='__main__':main()
