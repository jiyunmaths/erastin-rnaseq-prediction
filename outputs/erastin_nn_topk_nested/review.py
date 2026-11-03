#!/usr/bin/env python3
"""Audit every nested selection, then compare paired outer-test predictions."""
import itertools
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import benchmark as b
import evaluate as ev
from make_splits import inner_folds, inner_rows

OUT=b.OUT
ARMS=['top10','top20','full64']


def table(name,rows):
    frame=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    frame.to_csv(OUT/name,sep='\t',index=False,float_format='%.12g')
    return frame


def audit(ds,r,ph):
    s=b.STATE[ds];root=OUT/ds/'folds';complete=json.load(open(root/f'repeat__r{r+1:02d}_complete.json'))
    assert complete['protocol_sha256']==ph
    for f in complete['files']:assert b.sha(OUT/f['path'])==f['sha256']
    tr,te=np.flatnonzero(~s['test'][r]),np.flatnonzero(s['test'][r])
    a=np.load(root/f'comparison__r{r+1:02d}.npz');np.testing.assert_array_equal(a['train_rows'],tr);np.testing.assert_array_equal(a['test_rows'],te)
    maxpred=maxscale=0.;selector_checks=[];selections=[]
    for context in range(6):
        rows=inner_rows(s['inner'][r],context)[0] if context<5 else tr
        rr=root/f'selector__r{r+1:02d}__c{context}';z=np.load(rr.with_suffix('.npz'));meta=json.load(open(rr.with_suffix('.json')))
        np.testing.assert_array_equal(z['context_training_rows'],rows)
        assert set(rows).issubset(tr) and set(s['groups'][rows]).isdisjoint(s['groups'][te])
        if context<5:
            va=inner_rows(s['inner'][r],context)[1]
            assert set(s['groups'][rows]).isdisjoint(s['groups'][va])
        xx,yy,lin,groups=s['x'][rows],s['y'][rows],s['lineage'][rows],s['groups'][rows]
        seed=int(np.random.SeedSequence([b.MASTER_SEED,list(b.BASE).index(ds),r,context]).generate_state(1)[0])
        assert meta['selector_seed']==seed
        assignment=inner_folds(lin,groups,np.zeros(len(rows),bool),seed,k=3)
        np.testing.assert_array_equal(assignment,z['selector_assignment'])
        valid=assignment>=0;delta=z['receiver_delta_mse'];assert np.isfinite(delta[valid]).all() and np.isnan(delta[~valid]).all()
        importance=delta[valid].mean(axis=0);np.testing.assert_array_equal(importance,z['importance'])
        order=np.argsort(-importance,kind='stable');np.testing.assert_array_equal(order,z['ranking'])
        chosen=int(np.argmin(z['selector_fold_rmse'].mean(axis=1)));assert chosen==int(z['selected_candidate'])
        assert s['grid'][chosen]==meta['selected_parameters']
        models=joblib.load(rr.with_suffix('.joblib'))
        for fold,model in enumerate(models):
            fit,recv=inner_rows(assignment,fold)
            assert set(groups[fit]).isdisjoint(groups[recv])
            assert set(lin[fit])==set(lin)
            expected_scale=np.std(xx[fit],axis=0);expected_scale[expected_scale==0]=1.
            for actual,expected in [(model['x_scaler'].mean_,xx[fit].mean(axis=0)),
                                    (model['x_scaler'].scale_,expected_scale),
                                    (model['y_scaler'].mean_,yy[fit].mean())]:
                error=float(np.max(np.abs(actual-expected)));maxscale=max(maxscale,error);assert error<1e-11
            assert model['x_scaler'].n_samples_seen_==len(fit)
            assert model['y_scaler'].n_samples_seen_==len(fit)
            for member,seed0 in zip(model['members'],s['nn_seeds'][r]):
                assert member.random_state==int(seed0) and member.alpha==s['grid'][chosen]['alpha']
                assert member.hidden_layer_sizes==(s['grid'][chosen]['width'],)
            p=b.prediction(model,xx[recv]);expected=z['candidate_predictions'][chosen,recv]
            error=float(np.max(np.abs(p-expected)));maxpred=max(maxpred,error);assert error<1e-11
            losses=np.sqrt(np.mean((z['candidate_predictions'][:,recv]-yy[recv][None,:])**2,axis=1))
            np.testing.assert_allclose(losses,z['selector_fold_rmse'][:,fold],rtol=0,atol=1e-12)
        mask=a['inner_feature_masks'][context] if context<5 else a['feature_masks']
        for ki,k in enumerate(b.TOPK):
            np.testing.assert_array_equal(np.flatnonzero(mask[ki]),np.sort(order[:k]))
            assert mask[ki].sum()==k
            for j,gene in enumerate(s['genes']):
                selections.append(dict(assay=ds,repeat=r+1,context='outer' if context==5 else 'inner',
                    fold=context if context<5 else -1,panel=k,gene=str(gene),selected=bool(mask[ki,j])))
        assert np.all(mask[0]<=mask[1])
        selector_checks.append(dict(context=context,training_n=len(rows),ranking_receiver_n=int(valid.sum()),
            protected_n=int((~valid).sum()),donor_checks=sum(len(v['lineage_checks']) for v in meta['audit']),
            direct_replacement_max_error=max(v['max_direct_prediction_error'] for v in meta['audit'])))
    for fold in range(5):
        va=inner_rows(s['inner'][r],fold)[1]
        scores=np.sqrt(np.mean((a['inner_validation_predictions'][:,:,va]-s['y'][va][None,None,:])**2,axis=2))
        np.testing.assert_allclose(scores,a['inner_fold_rmse'][:,:,fold],rtol=0,atol=1e-12)
    params=[]
    for ki,k in enumerate(b.TOPK):
        rr=root/f'top{k}__r{r+1:02d}';z=np.load(rr.with_suffix('.npz'));meta=json.load(open(rr.with_suffix('.json')));model=joblib.load(rr.with_suffix('.joblib'))
        indices=np.flatnonzero(a['feature_masks'][ki]);np.testing.assert_array_equal(indices,z['feature_indices'])
        np.testing.assert_array_equal(model['genes'],s['genes'][indices]);np.testing.assert_array_equal(model['feature_indices'],indices)
        chosen=int(np.argmin(a['inner_fold_rmse'][ki].mean(axis=1)))
        assert chosen==meta['selected_candidate'] and s['grid'][chosen]==meta['selected_parameters']
        p=b.prediction(model,s['x'][te][:,indices]);error=float(np.max(np.abs(p-a['test_predictions'][ki])));maxpred=max(maxpred,error);assert error<1e-11
        np.testing.assert_allclose(model['x_scaler'].mean_,s['x'][tr][:,indices].mean(axis=0),rtol=0,atol=1e-12)
        np.testing.assert_allclose(model['y_scaler'].mean_,s['y'][tr].mean(),rtol=0,atol=1e-12)
        params.append(dict(assay=ds,repeat=r+1,model=f'top{k}',**meta['selected_parameters']))
    original=np.load(b.BASE[ds]/'folds'/f'neural_network__p64__r{r+1:02d}.npz')
    np.testing.assert_array_equal(original['test_prediction'],a['test_predictions'][2])
    return a['test_predictions'],selections,params,dict(assay=ds,repeat=r+1,selector_checks=selector_checks,
        model_prediction_max_error=maxpred,training_scaler_max_error=maxscale)


def analyze(ds,preds,protocol):
    s=b.STATE[ds];pooled,lineage,macro=ev.score(s['y'],preds,s['lineage'],s['test'])
    values,summary={},[]
    for arm in ARMS:
        v={'pooled_'+k:pooled.query('model == @arm').sort_values('repeat')[k].to_numpy() for k in ['rmse','spearman','pearson','mae','r2']}
        v.update({k:macro.query('model == @arm').sort_values('repeat')[k].to_numpy() for k in ['macro_all_rmse','macro_large_rmse']});values[arm]=v
        row=dict(assay=ds,model=arm)
        for key,val in v.items():row[key+'_mean']=float(ev.finite_mean(val));row[key+'_sd']=float(np.std(val[np.isfinite(val)],ddof=1))
        summary.append(row)
    draw,perdraw=ev.bootstrap(s['y'],preds,s['lineage'],s['test'],protocol['uncertainty']['count'],protocol['uncertainty']['seed'])
    bootstrap_check=ev.validate_bootstrap(s['y'],preds,s['lineage'],s['test'],protocol['uncertainty']['seed'])
    pairs,effects=[],[]
    for first,ref in protocol['primary_comparisons']+protocol['secondary_comparisons']:
        for metric in ev.KEYS:
            sign=-1 if metric.endswith(('rmse','mae')) else 1
            change=sign*(values[first][metric]-values[ref][metric]);boot=ev.finite_mean(sign*(perdraw[first][metric]-perdraw[ref][metric]),axis=1)
            lo,hi=np.percentile(boot[np.isfinite(boot)],[2.5,97.5])
            pairs.append(dict(assay=ds,first_model=first,reference_model=ref,metric=metric,
                improvement=float(ev.finite_mean(change)),conditional_ci_low=float(lo),conditional_ci_high=float(hi),
                favorable_repeats=int(np.sum(change>1e-12)),tied_repeats=int(np.sum(np.abs(change)<=1e-12)),defined_repeats=int(np.isfinite(change).sum())))
        for label in np.unique(s['lineage']):
            a=lineage[(lineage.model==first)&(lineage.lineage==label)].sort_values('repeat')
            r=lineage[(lineage.model==ref)&(lineage.lineage==label)].sort_values('repeat')
            effects.append(dict(assay=ds,first_model=first,reference_model=ref,lineage=str(label),
                cohort_n=int(a.cohort_n.iloc[0]),rmse_improvement=float(np.mean(r.rmse.to_numpy()-a.rmse.to_numpy()))))
    for d in [pooled,lineage,macro]:d.insert(0,'assay',ds)
    np.savez_compressed(OUT/ds/'predictions.npz',**preds,sample_ids=s['ids'],group_ids=s['groups'],lineage=s['lineage'],outer_test=s['test'])
    np.savez_compressed(OUT/ds/'bootstrap_draws.npz',**{arm+'__'+k:v for arm,d in draw.items() for k,v in d.items()})
    return summary,pairs,effects,pooled,lineage,macro,bootstrap_check


def figures(pooled,paired):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none'})
    colors=['#147d92','#d28b23','#666666'];labels=['Top 10','Top 20','Full 64']
    fig,axes=plt.subplots(2,2,figsize=(10,7.8))
    for col,ds in enumerate(b.BASE):
        for row,metric in enumerate(['rmse','spearman']):
            ax=axes[row,col];d=pooled[pooled.assay==ds]
            a=np.stack([d[d.model==arm].sort_values('repeat')[metric].to_numpy() for arm in ARMS])
            for r in range(20):ax.plot(range(3),a[:,r],color='#aaaaaa',alpha=.28,lw=.7,zorder=1)
            for j,v in enumerate(a):
                ax.scatter(np.full(20,j),v,c=colors[j],s=15,alpha=.55,zorder=2)
                ax.scatter(j,np.mean(v),c=colors[j],edgecolor='black',marker='D',s=70,zorder=3)
            ax.set_xticks(range(3),labels);ax.set_xlim(-.4,2.4);ax.grid(axis='y',alpha=.2)
            ax.set_title(ds+(' · RMSE (lower is better)' if metric=='rmse' else ' · Spearman (higher is better)'),loc='left',weight='bold')
            ax.set_ylabel('Negative-AUC units' if metric=='rmse' else 'Correlation')
    fig.suptitle('Training-selected gene panels versus full 64-gene NN',weight='bold',fontsize=15,y=.98)
    fig.text(.5,.02,'20 shared lineage-stratified holdouts per assay · lines pair the same split · diamonds show means\nAll selection and tuning confined to nested training data',ha='center',fontsize=9)
    fig.subplots_adjust(top=.9,bottom=.12,hspace=.35,wspace=.3,left=.09,right=.98)
    for ext in ['png','svg']:fig.savefig(OUT/('performance_comparison.'+ext),dpi=160,facecolor='white')
    plt.close(fig)
    d=paired[(paired.metric=='pooled_rmse')&(paired.reference_model=='full64')]
    fig,ax=plt.subplots(figsize=(8.2,4.7))
    for y,r in enumerate(d.itertuples()):
        ax.hlines(y,r.conditional_ci_low,r.conditional_ci_high,color=colors[0 if r.first_model=='top10' else 1],lw=2)
        ax.scatter(r.improvement,y,color=colors[0 if r.first_model=='top10' else 1],s=45)
    ax.set_yticks(range(len(d)),[f'{r.assay}  ·  {r.first_model.replace("top","Top ")}' for r in d.itertuples()]);ax.invert_yaxis()
    ax.axvline(0,color='#555555',linestyle='--',lw=.8);ax.grid(axis='x',alpha=.2)
    ax.set_xlabel('RMSE improvement over Full 64 (positive favors the smaller panel)')
    ax.set_title('Paired comparison on the same outer test cells',weight='bold',pad=15)
    fig.text(.5,.025,'Conditional pointwise 95% whole-lineage bootstrap intervals; fixed fits, no multiplicity adjustment',ha='center',fontsize=8)
    fig.subplots_adjust(left=.22,right=.96,top=.85,bottom=.18)
    for ext in ['png','svg']:fig.savefig(OUT/('paired_rmse_improvement.'+ext),dpi=160,facecolor='white')
    plt.close(fig)


def report(summary,paired,freq,stability,protocol):
    lines=['# Nested top-10, top-20 and full-64 neural-network comparison','',
        'The two reduced panels are selected **within training data**, separately for CTRP and PRISM. Each procedure is evaluated on all 20 original lineage-stratified 80/20 outer splits. The full64 comparator is the exact retained three-network ensemble for each split. Unweighted RNA-only fitting, identity groups, original NN tuning grids, seed triplets and scaling rules are preserved.','',
        '## Performance','',
        'RMSE is in original negative-AUC units; smaller is better. Correlations are against negative AUC, so larger is better. Means describe 20 overlapping holdouts, not 20 independent cohorts.','',
        '| Assay | Panel | Mean RMSE | Spearman | R2 | Equal-lineage RMSE |','|---|---|---:|---:|---:|---:|']
    for r in summary.itertuples():lines.append(f'| {r.assay} | {r.model} | {r.pooled_rmse_mean:.6f} | {r.pooled_spearman_mean:.4f} | {r.pooled_r2_mean:.4f} | {r.macro_all_rmse_mean:.6f} |')
    lines+=['','Positive improvement means lower RMSE than full64. Intervals are conditional pointwise 95% paired whole-lineage bootstrap intervals.','',
        '| Assay | Panel vs full64 | RMSE improvement [95% interval] | Relative RMSE improvement | Lower-error splits /20 |','|---|---|---:|---:|---:|']
    primary=paired[(paired.metric=='pooled_rmse')&(paired.reference_model=='full64')]
    for r in primary.itertuples():
        base=summary[(summary.assay==r.assay)&(summary.model=='full64')].pooled_rmse_mean.iloc[0]
        lines.append(f'| {r.assay} | {r.first_model} | {r.improvement:.6f} [{r.conditional_ci_low:.6f}, {r.conditional_ci_high:.6f}] | {100*r.improvement/base:.2f}% | {r.favorable_repeats} |')
    lines+=['','![Performance over the shared splits](performance_comparison.png)','','![Paired RMSE changes](paired_rmse_improvement.png)','','## What the comparison supports','']
    for r in primary.itertuples():
        outcome='lower RMSE, with its conditional interval above zero' if r.conditional_ci_low>0 else ('higher RMSE, with its conditional interval below zero' if r.conditional_ci_high<0 else 'an interval spanning zero; no clear RMSE gain is established')
        lines.append(f'- **{r.assay}, {r.first_model}:** {outcome}. Lower held-out RMSE in {r.favorable_repeats}/20 splits.')
    lines+=['','The existing model registry is unchanged. Reduced-panel results evaluate this fold-local selection procedure, not one fixed top-10/top-20 list that has received independent validation.','',
        '## Selection and leakage controls','',
        '1. Keep each original outer test set completely outside selection, scaling and hyperparameter tuning.','2. Within each saved inner training set, make three additional response-blind identity-group/lineage folds. Tune a full64 selector NN on these local folds using the original assay grid.','3. Rank genes by the selected selector candidate\'s pooled out-of-fold expected increase in MSE after replacing each gene with every same-lineage selector-training donor value. Donor and receiver identity groups are disjoint. These receivers belong entirely to the current inner training set.','4. Fit each 10/20-gene NN candidate on its current inner training set and score on the untouched saved inner validation fold. Select NN parameters by mean five-fold RMSE, separately for each K.','5. Recompute the selector on the complete outer training set, choose its top K genes, refit the chosen reduced NN and predict the original outer test set.','',
        'Gene ranking breaks ties by canonical gene order and takes exactly K genes. The 10-gene set is nested inside the 20-gene set in each selection context. Downstream inputs preserve canonical order. Protected singleton training-lineage groups remain training and do not become ranking or tuning validation receivers. All evaluated lineages still occur in both outer train and test.','',
        'Preflight refits exactly reproduced full64 predictions for the first split and all inner NN candidates in one fold per assay. Mutating excluded validation/test expression and outcomes did not change the local selection. The completed audit checks all 240 selector contexts, selected models, fold assignments, donor exclusions, scalers, selected masks, tuning losses, outer predictions, source hashes and exact full64 prediction reuse.','',
        '## Frequently selected genes','',
        'Counts below are selections among the 20 outer-training fits. They describe stability; they do not define a separately evaluated consensus panel. Complete counts for all 64 genes and 100 inner contexts per assay are in the selection-frequency table.','']
    for ds in b.BASE:
        for k in b.TOPK:
            d=freq[(freq.assay==ds)&(freq.panel==k)].sort_values(['outer_selected_fits','inner_selected_contexts','gene'],ascending=[False,False,True]).head(15)
            lines.append(f'**{ds}, top{k}:** '+', '.join(f'{r.gene} ({r.outer_selected_fits}/20)' for r in d.itertuples())+'.')
            st=next(v for v in stability if v['assay']==ds and v['panel']==k)
            lines+=['',f'Median between-repeat panel Jaccard similarity: {st["median_jaccard"]:.3f}.','']
    lines+=['## Interpretation limits','',
        'These cohorts have been inspected repeatedly in earlier model and gene analyses. Training-local selection prevents direct use of outer responses, but this remains exploratory rather than a new independent validation study. Bootstrap intervals condition on these selected fits, retain overlapping repeats together and are not adjusted for multiple comparisons or retraining uncertainty.','',
        'Within-lineage donor replacement can disrupt gene correlations; redundant genes can share importance. Selection frequency is not a significance, causal or knockout measure. The benchmark concerns cell lines within represented lineages. CTRP endpoint provenance limitations and the separate official PRISM response processing remain as documented in their source experiments. The assays overlap in cell lines, so agreement is not independent biological replication.','',
        '## Artifacts and reproduction','',
        '- [Metric summary](summary.tsv)','- [Paired differences and intervals](paired_comparisons.tsv)','- [All 64 genes\' selection frequencies](gene_selection_frequency.tsv)','- [Outer selected panels](outer_selected_panels.tsv)','- [Per-lineage error changes](lineage_effects.tsv)','- [Validation](validation.json)','- [Frozen protocol](protocol.json)','- [Source checksums](source_manifest.json)','',
        'From the project root, using the original pinned environment:','','```bash',
        'PYTHONPATH="$PWD/work/bulkformer/packages" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\',
        'PYTHONDONTWRITEBYTECODE=1 /home/sji/miniforge3/envs/deeplearning/bin/python \\',
        '  outputs/erastin_nn_topk_nested/benchmark.py',
        'PYTHONPATH="$PWD/work/bulkformer/packages" OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \\',
        'MPLCONFIGDIR="$PWD/work/bulkformer/matplotlib_cache" PYTHONDONTWRITEBYTECODE=1 \\',
        '/home/sji/miniforge3/envs/deeplearning/bin/python outputs/erastin_nn_topk_nested/review.py','```','']
    (OUT/'REVIEW.md').write_text('\n'.join(lines))


def main():
    b.initialize();b.verify();protocol=json.load(open(OUT/'protocol.json'));ph=b.sha(OUT/'protocol.json')
    assert json.load(open(OUT/'completion.json'))['complete']
    summaries=[];paired=[];effects=[];pooled=[];lineages=[];macros=[];checks=[];sels=[];params=[];bootstrapchecks=[]
    registry=json.load(open(b.REGISTRY))
    for ds,s in b.STATE.items():
        preds={arm:np.full(s['test'].shape,np.nan) for arm in ARMS}
        for r in range(20):
            pp,se,pa,check=audit(ds,r,ph);sels.extend(se);params.extend(pa);checks.append(check)
            for j,arm in enumerate(ARMS):preds[arm][r,s['test'][r]]=pp[j]
        su,pa,ef,po,li,ma,bc=analyze(ds,preds,protocol)
        full=next(v for v in su if v['model']=='full64');reg=registry['cohorts'][ds]['models']['neural_network']['mean_heldout_metrics']
        for our,their in [('pooled_rmse_mean','rmse'),('pooled_spearman_mean','spearman'),('pooled_r2_mean','r2'),('macro_all_rmse_mean','equal_lineage_rmse')]:
            assert abs(full[our]-reg[their])<1e-10
        summaries.extend(su);paired.extend(pa);effects.extend(ef);pooled.append(po);lineages.append(li);macros.append(ma);bootstrapchecks.append(dict(assay=ds,**bc))
        print(f'{ds}: all 20 repeats and 120 nested selector contexts audited and scored.',flush=True)
    summary=table('summary.tsv',summaries);pair=table('paired_comparisons.tsv',paired);eff=table('lineage_effects.tsv',effects)
    pooled=table('pooled_metrics.tsv',pd.concat(pooled));table('lineage_metrics.tsv',pd.concat(lineages));table('macro_metrics.tsv',pd.concat(macros))
    selections=pd.DataFrame(sels);table('all_selection_masks.tsv',selections)
    freq=[];outerpanels=[];stability=[]
    for ds,s in b.STATE.items():
        for k in b.TOPK:
            d=selections[(selections.assay==ds)&(selections.panel==k)]
            outer=d[d.context=='outer'];inner=d[d.context=='inner']
            for gene in s['genes']:
                freq.append(dict(assay=ds,panel=k,gene=str(gene),outer_selected_fits=int(outer[outer.gene==gene].selected.sum()),
                    outer_fits=20,inner_selected_contexts=int(inner[inner.gene==gene].selected.sum()),inner_contexts=100))
            sets=[]
            for r in range(1,21):
                genes=outer[(outer['repeat']==r)&outer.selected].gene.tolist();assert len(genes)==k
                sets.append(set(genes));outerpanels.append(dict(assay=ds,repeat=r,panel=k,genes=', '.join(genes)))
            jac=[len(a&c)/len(a|c) for a,c in itertools.combinations(sets,2)]
            stability.append(dict(assay=ds,panel=k,pairs=len(jac),median_jaccard=float(np.median(jac)),min_jaccard=float(np.min(jac)),max_jaccard=float(np.max(jac))))
    freq=table('gene_selection_frequency.tsv',freq);table('outer_selected_panels.tsv',outerpanels);table('selection_stability.tsv',stability);table('selected_parameters.tsv',params)
    for ds in b.BASE:
        for k in b.TOPK:
            d=freq[(freq.assay==ds)&(freq.panel==k)];assert d.outer_selected_fits.sum()==20*k and d.inner_selected_contexts.sum()==100*k
    figures(pooled,pair);report(summary,pair,freq,stability,protocol)
    b.verify()
    validation=dict(passed=True,protocol_sha256=ph,repeat_comparisons=40,new_outer_ensembles=80,
        selector_contexts=240,selector_crossfit_models=720,original_full64_predictions_exactly_reused=True,
        original_full64_metrics_match_registry=True,all_scales_and_tuning_losses_verified=True,
        training_local_selection_only=True,all_top10_panels_nested_in_top20=True,bootstrap_checks=bootstrapchecks,
        max_prediction_error=max(v['model_prediction_max_error'] for v in checks),
        max_training_scaler_error=max(v['training_scaler_max_error'] for v in checks),checks=checks,
        analysis_script_sha256=b.sha(Path(__file__)))
    b.dump(OUT/'validation.json',validation)
    b.dump(OUT/'final_summary.json',dict(summary=summary.to_dict('records'),primary_rmse=pair[(pair.metric=='pooled_rmse')&(pair.reference_model=='full64')].to_dict('records'),selection_stability=stability))
    files=[dict(path=str(p.relative_to(OUT)),sha256=b.sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts]
    b.dump(OUT/'artifact_manifest.json',dict(files=files));b.dump(OUT/'progress.json',dict(phase='complete',validation_passed=True,repeat_comparisons=40))
    # Progress is part of the sealed result, so refresh its entry after final status.
    files=[dict(path=str(p.relative_to(OUT)),sha256=b.sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts]
    b.dump(OUT/'artifact_manifest.json',dict(files=files))
    print(summary.to_string(index=False),flush=True)
    print(pair[(pair.metric=='pooled_rmse')&(pair.reference_model=='full64')].to_string(index=False),flush=True)
    print('TOPK_REVIEW_COMPLETE',flush=True)


if __name__=='__main__':main()
