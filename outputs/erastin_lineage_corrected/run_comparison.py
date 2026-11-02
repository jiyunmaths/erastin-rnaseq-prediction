#!/usr/bin/env python3
"""Corrected leave-lineage-out ridge evaluation using frozen saved embeddings.

Run from workspace with work/bulkformer/packages on PYTHONPATH and BLAS threads=1.
The original outer lineage holdouts are preserved. Five corrected global group
folds are restricted to outer-training rows, exactly as in the original design.
"""
import concurrent.futures as futures
import gzip
import html
import importlib.util
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import pandas as pd

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
WS=OUT.parent.parent
PRIMARY=['panel_expression','panel_contextual_embedding','all_gene_contextual_embedding']
METRICS=['spearman','rmse','r2']


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    return m


def make_holdouts(spec,ids,groups,lineages,inner_assignment):
    ix={s:i for i,s in enumerate(ids)}
    folds=[]
    for f in spec['leave_lineage_out']:
        tr=np.array([ix[s] for s in f['train_ModelIDs']]);te=np.array([ix[s] for s in f['test_ModelIDs']])
        assert set(lineages[te])=={f['lineage']} and f['lineage'] not in set(lineages[tr])
        assert set(groups[tr]).isdisjoint(groups[te])
        assert set(te)==set(np.flatnonzero(lineages==f['lineage']))
        assert set(tr)==set(np.flatnonzero(~np.isin(groups,groups[te])))
        inner=[]
        for k in range(5):
            a=tr[inner_assignment[tr]!=k];b=tr[inner_assignment[tr]==k]
            assert len(a)>0 and len(b)>0 and set(a)|set(b)==set(tr)
            assert set(a).isdisjoint(b) and set(groups[a]).isdisjoint(groups[b])
            assert set(a).isdisjoint(te) and set(b).isdisjoint(te)
            inner.append({'train':a,'test':b})
        folds.append({'label':f['lineage'],'train':tr,'test':te,'inner':inner})
    test=np.concatenate([f['test'] for f in folds])
    assert len(test)==len(ids) and len(np.unique(test))==len(ids)
    return folds


def macro_table(fold_scores):
    eligible=fold_scores[fold_scores.test_n>=10]
    assert eligible.groupby(['repeat','model']).size().eq(19).all()
    assert np.isfinite(eligible[eligible.model.isin(PRIMARY)][METRICS]).all().all()
    return eligible.groupby(['repeat','seed','model'],as_index=False)[METRICS].mean()


def paired_rows(point,draw,scope,n_repeats):
    rows=[]
    for a,b in [(PRIMARY[1],PRIMARY[0]),(PRIMARY[2],PRIMARY[0]),(PRIMARY[1],PRIMARY[2])]:
        for metric in METRICS:
            sign=-1 if metric=='rmse' else 1
            effect=sign*(point.loc[a,metric]-point.loc[b,metric])
            delta=sign*(draw[a][metric]-draw[b][metric]);lo,hi=np.percentile(delta,[2.5,97.5])
            rows.append({'scope':scope,'repeats':n_repeats,'first_model':a,'reference_model':b,'metric':metric,
                'improvement':effect,'conditional_lineage_bootstrap_ci_low':lo,'conditional_lineage_bootstrap_ci_high':hi})
    return rows


def macro_bootstrap(fold_scores,repeats,seed,count=5000):
    # Equal weight for each of the 19 eligible lineages, after averaging scores
    # over fixed repeat seeds. The same resampled lineages are used for all arms.
    subset=fold_scores[(fold_scores.repeat.isin(repeats))&(fold_scores.test_n>=10)]
    labels=sorted(subset.fold.unique());assert len(labels)==19
    picks=np.random.default_rng(seed).integers(19,size=(count,19))
    draw={}
    for name in PRIMARY:
        v=subset[subset.model==name].groupby('fold')[METRICS].mean().loc[labels]
        draw[name]={metric:v[metric].to_numpy()[picks].mean(axis=1) for metric in METRICS}
    return draw


def render(summary,macro_summary,paired,seed_differences,lineage_differences,legacy,validation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    labels={'panel_expression':'40-gene expression','panel_contextual_embedding':'Panel-focused BulkFormer',
            'all_gene_contextual_embedding':'All-gene BulkFormer','lineage_only':'Lineage-only control','training_mean':'Training-mean control'}
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    rows=lineage_differences[lineage_differences.scope=='mean_20_inner_seeds'].copy()
    eligible=rows[rows.test_n>=10].sort_values(['test_n','lineage'],ascending=[False,True])
    fig,axes=plt.subplots(1,2,figsize=(11,8),constrained_layout=True,sharey=True)
    positions=np.arange(len(eligible))
    for ax,metric,title in zip(axes,['spearman','rmse'],['Spearman increase','RMSE reduction']):
        v=eligible[metric+'_improvement'].to_numpy()
        ax.barh(positions,v,color=np.where(v>0,'#167d8d','#c56747'),height=.65)
        ax.axvline(0,color='#37474f',lw=1)
        ax.set_xlabel(title+' (positive favors BulkFormer)');ax.grid(axis='x',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(positions,[f'{r.lineage} (n={r.test_n})' for r in eligible.itertuples()]);axes[0].invert_yaxis()
    fig.suptitle('Unseen lineages: panel-focused BulkFormer versus 40-gene expression\nMean effect over 20 fixed inner-fold seeds; 19 lineages with n ≥ 10',fontsize=12)
    fig.savefig(OUT/'lineage_effects.png',dpi=160);fig.savefig(OUT/'lineage_effects.svg');plt.close(fig)
    focus=seed_differences[(seed_differences.first_model==PRIMARY[1])&(seed_differences.reference_model==PRIMARY[0])]
    fig,axes=plt.subplots(1,2,figsize=(10,4.6),constrained_layout=True)
    colors={'pooled_745':'#126e75','macro_19_lineages':'#9c558a'}
    for ax,metric,title in zip(axes,['spearman','rmse'],['Spearman increase','RMSE reduction']):
        for scope,color in colors.items():
            v=focus[(focus.scope==scope)&(focus.metric==metric)].sort_values('repeat')
            ax.scatter(v.improvement,v.repeat,s=25,color=color,label='Pooled cells' if scope=='pooled_745' else 'Equal-weight lineage mean')
        ax.axvline(0,color='#555',ls='--',lw=1);ax.grid(axis='x',alpha=.2)
        ax.set(xlabel=title+' (positive favors BulkFormer)',ylabel='Inner-fold seed',yticks=np.arange(1,21,2))
        ax.spines[['top','right']].set_visible(False)
    handles,legend_labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='outside lower center',ncol=2)
    fig.suptitle('Leave-lineage-out sensitivity to inner-fold assignment',fontsize=12)
    fig.savefig(OUT/'inner_seed_effects.png',dpi=160);fig.savefig(OUT/'inner_seed_effects.svg');plt.close(fig)
    lines=['Corrected leave-lineage-out comparison','='*43,'',
        '745 cancer cell lines; 734 identity groups; 24 held-out lineages.',
        'Original lineage holdouts preserved. Five corrected global group folds are restricted to non-held-out lineages for inner tuning.',
        'Direct corrected rerun uses the original seed 20261031. Sensitivity analysis uses the same 20 fixed seeds as the preceding repeated-CV study.',
        'The 20 seeds vary only inner tuning folds; they do not create new outer lineage holdouts or new biological samples.',
        'All arms use shared folds, train-only standardization and the unchanged 17-value ridge grid. Target: negative reported erastin AUC.',
        'Frozen BulkFormer embeddings reused without new inference or feature selection. Panel pooling still sees the full transcriptome.', '',
        'Pooled held-out performance (every cell line receives one prediction per run):',
        summary.to_string(index=False,float_format=lambda x:f'{x:.6f}'),'',
        'Equal-weight mean of within-lineage metrics (19 lineages with n >= 10):',
        macro_summary.to_string(index=False,float_format=lambda x:f'{x:.6f}'),'',
        'Primary contrast: panel-focused BulkFormer minus expression; positive is better:',
        paired[(paired.first_model==PRIMARY[1])&(paired.reference_model==PRIMARY[0])].to_string(index=False,float_format=lambda x:f'{x:.6f}'),'',
        'Interpretation:',
        '- Pooled Spearman includes differences between lineage means. Equal-weight within-lineage Spearman assesses ranking within each unseen lineage.',
        '- Five small lineages (n < 10) remain in pooled metrics and detailed tables, but are excluded from macro summaries and lineage win counts.',
        '- Uncertainty uses 5,000 paired resamples of whole held-out lineages (24 for pooled metrics; 19 for macro metrics). All models and seeds remain paired.',
        '- Intervals condition on these predictions. Models are not retrained within bootstrap draws, and training sets overlap across lineage holdouts.',
        '- Neither seed counts nor lineage win counts are used as independent-study significance tests. No seed, model variant, penalty range or metric was selected by held-out performance.',
        '- These are lineages withheld from supervised ridge fitting, not a verified foundation-pretraining-disjoint benchmark. Pretraining overlap remains unresolved.',
        '- The same archived CTRP labels and DepMap 24Q4 inputs are reused. No independent drug screen or clinical cohort was tested.', '',
        'Legacy versus corrected original-seed pooled results:',
        legacy.to_string(index=False,float_format=lambda x:f'{x:.6f}'),'',
        'Audit: protocol.json, source_manifest.json, shared_splits.npz, outer_holdouts.json, split_audit.tsv,',
        'per_run_metrics.tsv, per_lineage_metrics.tsv, per_run_predictions.npz, tuning_scores.tsv.gz,',
        'fitted_outer_models.json.gz, validation.json and artifact_manifest.json.', '']
    (OUT/'RESULTS.txt').write_text('\n'.join(lines))
    show=summary.copy();show['model']=show.model.map(labels)
    page='<!doctype html><html><head><meta charset="utf-8"><title>Corrected lineage holdouts</title><style>body{font:16px system-ui;max-width:1200px;margin:35px auto;padding:0 22px;color:#19323c}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border-bottom:1px solid #ddd}img{max-width:100%}pre{white-space:pre-wrap;font:14px system-ui;line-height:1.5}</style></head><body><h1>Corrected leave-lineage-out comparison</h1>'
    page+=show.to_html(index=False,border=0,float_format=lambda x:f'{x:.5f}')
    page+='<img src="lineage_effects.png" alt="Paired effects in held-out lineages"><img src="inner_seed_effects.png" alt="Paired effects across inner-fold seeds">'
    page+='<pre>'+html.escape('\n'.join(lines))+'</pre></body></html>'
    (OUT/'RESULTS.html').write_text(page)


def main():
    started=time.time()
    inputs=WS/'outputs/erastin_40gene_comparison';prior=WS/'outputs/erastin_model_comparison';repeat=WS/'outputs/erastin_repeated_cv'
    cm=module('comparison',prior/'compare_models.py');rc=module('corrected_splits',repeat/'repeat_cv.py')
    source_groups=[(inputs,'artifact_manifest.json',['comparison_inputs.npz','cohort.tsv','comparison_config.json','splits.json']),
        (prior,'result_artifact_manifest.json',['compare_models.py','bulkformer_embeddings.npz','embedding_metadata.json','leave_lineage_out_predictions.tsv','leave_lineage_out_summary.tsv']),
        (repeat,'artifact_manifest.json',['repeat_cv.py','protocol.json','shared_splits.npz'])]
    sources=[]
    for directory,manifest,names in source_groups:
        known={f['path']:f['sha256'] for f in json.loads((directory/manifest).read_text())['files']}
        for name in names:
            path=directory/name;digest=cm.sha(path);assert digest==known[name],path
            sources.append({'path':str(path.relative_to(WS)),'sha256':digest})
    data=np.load(inputs/'comparison_inputs.npz',allow_pickle=False);emb=np.load(prior/'bulkformer_embeddings.npz',allow_pickle=False)
    old_splits=np.load(repeat/'shared_splits.npz',allow_pickle=False)
    ids=data['sample_ids'];groups=data['group_ids'];lineages=data['lineage'];y=data['y_sensitivity'].astype(float)
    assert np.array_equal(ids,emb['sample_ids']) and np.array_equal(ids,old_splits['sample_ids'])
    frame=pd.read_csv(inputs/'cohort.tsv',sep='\t',keep_default_na=False).set_index('ModelID').loc[ids]
    assert np.array_equal(frame.group_id.to_numpy(),groups) and np.array_equal(frame.OncotreeLineage.to_numpy(),lineages)
    assert len(ids)==745 and len(np.unique(groups))==734 and len(np.unique(lineages))==24
    assert frame.groupby('group_id').OncotreeLineage.nunique().max()==1
    spec=json.loads((inputs/'splits.json').read_text());config=json.loads((inputs/'comparison_config.json').read_text())
    alphas=np.array(config['ridge_alphas']);seeds=[20261031]+[int(s) for s in old_splits['seeds']]
    features={PRIMARY[0]:data['X_panel_log2tpm1'],PRIMARY[1]:emb['panel_embedding'],PRIMARY[2]:emb['all_gene_embedding'],
        'lineage_only':lineages,'training_mean':None}
    for name in PRIMARY:assert len(features[name])==745 and np.isfinite(features[name]).all()
    protocol={'status':'frozen_before_fitting','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'script_sha256':cm.sha(Path(__file__)),'outer_holdouts':'original 24 lineage tests and identity-purged training sets unchanged',
        'inner_cv':'five corrected global grouped lineage-balanced folds restricted to each outer training set; same design as original',
        'original_seed_primary':20261031,'inner_seed_sensitivity':seeds[1:],
        'outer_lineage_tests_do_not_change_with_seed':True,'primary_contrast':'panel_contextual_embedding versus panel_expression',
        'primary_models':PRIMARY,'primary_seed_controls':['lineage_only','training_mean'],'alpha_grid':alphas.tolist(),
        'selection':'minimum equal-weight mean inner-fold RMSE; train-only standardization',
        'pooled_metrics':'Spearman, RMSE, R2 on all 745 held-out predictions',
        'macro_metrics':'equal-weight mean of within-lineage metrics over the same 19 lineages with n>=10',
        'small_lineages':'all five retained in pooled scores; descriptive only individually',
        'bootstrap':'5000 paired whole-lineage resamples; fixed predictions; no refitting; no independent-seed tests',
        'bootstrap_seeds':{'primary_pooled':2026110301,'repeat_pooled':2026110302,'primary_macro':2026110303,'repeat_macro':2026110304},
        'original_results_already_seen':True,'all_seeds_and_metrics_fixed_before_fitting':True,'workers':6}
    path=OUT/'protocol.json'
    if path.exists():
        old=json.loads(path.read_text());protocol['created_utc']=old['created_utc'];assert protocol==old
    else:cm.dump(path,protocol)
    protocol_hash=cm.sha(path);cm.dump(OUT/'source_manifest.json',{'files':sources})
    all_cv=[];assignment=[];audit=[];inner_masks=[]
    for run,seed in enumerate(seeds):
        inner=rc.corrected_make_folds(frame,5,seed).to_numpy()
        if run:assert np.array_equal(inner,old_splits['outer_fold'][run-1])
        cv=make_holdouts(spec,ids,groups,lineages,inner);all_cv.append(cv);assignment.append(inner)
        mask=np.full((24,len(ids)),-1,dtype=np.int8)
        for i,f in enumerate(cv):
            mask[i,f['train']]=inner[f['train']]
            for j,v in enumerate(f['inner']):
                audit.append({'repeat':run,'seed':seed,'held_out_lineage':f['label'],'outer_train_n':len(f['train']),
                    'outer_test_n':len(f['test']),'inner_fold':j,'inner_train_n':len(v['train']),'inner_validation_n':len(v['test']),
                    'group_disjoint':True,'held_out_lineage_absent':True})
        inner_masks.append(mask)
    np.savez_compressed(OUT/'shared_splits.npz',sample_ids=ids,group_ids=groups,lineage=lineages,seeds=np.array(seeds,dtype=np.uint32),
        global_inner_assignment=np.array(assignment,dtype=np.int8),inner_fold_by_holdout=np.array(inner_masks,dtype=np.int8),
        holdout_labels=np.array([f['label'] for f in all_cv[0]]))
    cm.dump(OUT/'outer_holdouts.json',spec['leave_lineage_out']);cm.table(OUT/'split_audit.tsv',pd.DataFrame(audit))
    print(json.dumps({'event':'protocol_and_21_inner_split_sets_frozen','protocol_sha256':protocol_hash}),flush=True)
    # Reproduce original predictions under original splits before correcting them.
    legacy_cv=make_holdouts(spec,ids,groups,lineages,data['outer_fold'])
    old_pred=pd.read_csv(prior/'leave_lineage_out_predictions.tsv',sep='\t').set_index('ModelID').loc[ids]
    reproduction={}
    with futures.ThreadPoolExecutor(max_workers=6) as pool:
        jobs={name:pool.submit(cm.nested_fit,name,x,y,legacy_cv,alphas,len(ids),groups) for name,x in features.items()}
        for name,job in jobs.items():
            p,_,_,_=job.result();error=float(np.max(np.abs(p-old_pred[name].to_numpy())))
            assert error<1e-10;reproduction[name]=error
    print(json.dumps({'event':'legacy_predictions_reproduced','max_abs_errors':reproduction}),flush=True)
    predictions={name:np.full((21,len(ids)),np.nan) for name in PRIMARY}
    controls={};metric_rows=[];fold_rows=[];tuning=[];models=[];completed=0
    with futures.ThreadPoolExecutor(max_workers=6) as pool:
        jobs={}
        for run,cv in enumerate(all_cv):
            for name in (list(features) if run==0 else PRIMARY):
                jobs[pool.submit(cm.nested_fit,name,features[name],y,cv,alphas,len(ids),groups)]=(run,name)
        for future in futures.as_completed(jobs):
            run,name=jobs[future];p,fr,tu,mo=future.result();assert np.isfinite(p).all()
            if name in PRIMARY:predictions[name][run]=p
            else:controls[name]=p
            metric_rows.append({'repeat':run,'seed':seeds[run],'model':name,**cm.metrics(y,p)})
            fold_rows += [{'repeat':run,'seed':seeds[run],**r} for r in fr]
            tuning += [{'repeat':run,'seed':seeds[run],**r} for r in tu]
            models += [{'repeat':run,'seed':seeds[run],**r} for r in mo]
            completed+=1
            cm.dump(OUT/'progress.json',{'phase':'fitting','completed_model_runs':completed,'total_model_runs':len(jobs),
                'elapsed_seconds':time.time()-started})
            cm.table(OUT/'per_run_metrics.tsv',pd.DataFrame(metric_rows).sort_values(['repeat','model']))
            print(json.dumps({'event':'model_run_complete','repeat':run,'seed':seeds[run],'model':name,
                'completed':completed,'total':len(jobs),'elapsed_seconds':time.time()-started}),flush=True)
    assert all(np.isfinite(p).all() for p in predictions.values())
    scores=pd.DataFrame(metric_rows).sort_values(['repeat','model'])
    fold_scores=pd.DataFrame(fold_rows).sort_values(['repeat','model','fold'])
    fold_scores['interpretation']=np.where(fold_scores.test_n>=10,'eligible','descriptive_only_small_n')
    cm.table(OUT/'per_lineage_metrics.tsv',fold_scores);cm.table(OUT/'tuning_scores.tsv.gz',pd.DataFrame(tuning))
    (OUT/'fitted_outer_models.json.gz').write_bytes(gzip.compress(json.dumps(cm.clean(models)).encode(),mtime=0))
    np.savez_compressed(OUT/'per_run_predictions.npz',sample_ids=ids,group_ids=groups,lineage=lineages,observed_sensitivity=y,
        seeds=np.array(seeds,dtype=np.uint32),**predictions,**controls)
    macro=macro_table(fold_scores);cm.table(OUT/'per_run_macro_metrics.tsv',macro)
    validation={'passed':True,'source_hashes_verified':True,'original_predictions_max_abs_errors':reproduction,
        'original_outer_lineage_holdouts_preserved':True,'all_outer_and_inner_groups_disjoint':True,
        'all_test_lineages_absent_from_supervised_training_and_tuning':True,'same_corrected_20_seed_assignments_as_prior_run':True,
        'all_745_predictions_present_per_run':True,'protocol_sha256':protocol_hash,
        'model_count':len(models),'primary_seed':20261031,'sensitivity_repeats':20}
    validation.update(rc.validate_weighted_bootstrap(y,predictions[PRIMARY[1]][:2],lineages,cm.metrics))
    cm.dump(OUT/'progress.json',{'phase':'lineage_bootstrap','completed_model_runs':65,'total_model_runs':65})
    draws_primary=rc.repeated_group_bootstrap(y,{n:p[:1] for n,p in predictions.items()},lineages,count=5000,seed=2026110301)
    draws_repeated=rc.repeated_group_bootstrap(y,{n:p[1:] for n,p in predictions.items()},lineages,count=5000,seed=2026110302)
    macro_primary=macro_bootstrap(fold_scores,[0],2026110303)
    macro_repeated=macro_bootstrap(fold_scores,list(range(1,21)),2026110304)
    np.savez_compressed(OUT/'bootstrap_draws.npz',**{scope+'__'+n+'__'+m:a for scope,d in
        [('primary_pooled',draws_primary),('repeated_pooled',draws_repeated),('primary_macro',macro_primary),('repeated_macro',macro_repeated)]
        for n,values in d.items() for m,a in values.items()})
    primary_point=scores[scores.repeat==0].set_index('model')[METRICS]
    repeated_point=scores[scores.repeat>0].groupby('model')[METRICS].mean()
    primary_macro=macro[macro.repeat==0].set_index('model')[METRICS]
    repeated_macro=macro[macro.repeat>0].groupby('model')[METRICS].mean()
    summary=pd.concat([primary_point.reset_index().assign(scope='corrected_original_seed'),
        repeated_point.reset_index().assign(scope='mean_20_inner_seeds')],ignore_index=True)
    macro_summary=pd.concat([primary_macro.reset_index().assign(scope='corrected_original_seed'),
        repeated_macro.reset_index().assign(scope='mean_20_inner_seeds')],ignore_index=True)
    paired=pd.DataFrame(paired_rows(primary_point,draws_primary,'original_seed_pooled_745',1)+
        paired_rows(repeated_point,draws_repeated,'mean_20_seeds_pooled_745',20)+
        paired_rows(primary_macro,macro_primary,'original_seed_macro_19',1)+
        paired_rows(repeated_macro,macro_repeated,'mean_20_seeds_macro_19',20))
    cm.table(OUT/'pooled_summary.tsv',summary);cm.table(OUT/'macro_summary.tsv',macro_summary)
    cm.table(OUT/'paired_comparisons.tsv',paired)
    seed_rows=[]
    for scope,table in [('pooled_745',scores),('macro_19_lineages',macro)]:
        for run in range(1,21):
            p=table[table.repeat==run].set_index('model')
            for a,b in [(PRIMARY[1],PRIMARY[0]),(PRIMARY[2],PRIMARY[0]),(PRIMARY[1],PRIMARY[2])]:
                for metric in METRICS:
                    seed_rows.append({'scope':scope,'repeat':run,'seed':seeds[run],'first_model':a,'reference_model':b,'metric':metric,
                        'improvement':(-1 if metric=='rmse' else 1)*(p.loc[a,metric]-p.loc[b,metric])})
    seed_differences=pd.DataFrame(seed_rows);cm.table(OUT/'per_seed_differences.tsv',seed_differences)
    seed_summary=seed_differences.groupby(['scope','first_model','reference_model','metric']).improvement.agg(
        mean='mean',std='std',min='min',max='max',favorable_seeds=lambda v:int((v>0).sum()),seeds='size').reset_index()
    cm.table(OUT/'seed_stability_summary.tsv',seed_summary)
    lineage_rows=[]
    for scope,runs in [('corrected_original_seed',[0]),('mean_20_inner_seeds',list(range(1,21)))]:
        t=fold_scores[fold_scores.repeat.isin(runs)].groupby(['fold','model'])[[*METRICS,'test_n']].mean()
        for label in sorted(np.unique(lineages)):
            a=t.loc[(label,PRIMARY[1])];b=t.loc[(label,PRIMARY[0])]
            lineage_rows.append({'scope':scope,'lineage':label,'test_n':int(a.test_n),
                **{metric+'_improvement':(-1 if metric=='rmse' else 1)*(a[metric]-b[metric]) for metric in METRICS}})
    lineage_differences=pd.DataFrame(lineage_rows);cm.table(OUT/'lineage_paired_effects.tsv',lineage_differences)
    wins=[]
    for scope,rows in lineage_differences[lineage_differences.test_n>=10].groupby('scope'):
        for metric in METRICS:
            v=rows[metric+'_improvement'];wins.append({'scope':scope,'metric':metric,'favorable_lineages':int((v>0).sum()),'eligible_lineages':len(v)})
    cm.table(OUT/'lineage_win_counts.tsv',pd.DataFrame(wins))
    old_summary=pd.read_csv(prior/'leave_lineage_out_summary.tsv',sep='\t')
    legacy=pd.concat([old_summary[['model',*METRICS]].assign(scope='legacy_original_seed'),
        summary[summary.scope=='corrected_original_seed']],ignore_index=True)
    cm.table(OUT/'legacy_vs_corrected.tsv',legacy)
    validation['alpha_grid_boundary_counts']=fold_scores.groupby('model').alpha.apply(lambda x:int(x.isin([alphas[0],alphas[-1]]).sum())).to_dict()
    assert cm.sha(OUT/'protocol.json')==protocol_hash and cm.sha(Path(__file__))==protocol['script_sha256']
    for source in sources:assert cm.sha(WS/source['path'])==source['sha256']
    validation['protocol_script_and_sources_unchanged_after_fitting']=True
    validation['elapsed_seconds']=time.time()-started;cm.dump(OUT/'validation.json',validation)
    render(summary,macro_summary,paired,seed_differences,lineage_differences,legacy,validation)
    cm.dump(OUT/'final_summary.json',{'status':'complete','primary_seed':20261031,'inner_seed_repeats':20,
        'pooled_results':summary.to_dict('records'),'macro_results':macro_summary.to_dict('records'),
        'primary_contrast':paired[(paired.first_model==PRIMARY[1])&(paired.reference_model==PRIMARY[0])].to_dict('records'),
        'lineage_win_counts':wins,'elapsed_seconds':time.time()-started})
    cm.dump(OUT/'progress.json',{'phase':'complete','completed_model_runs':65,'total_model_runs':65,'elapsed_seconds':time.time()-started})
    files=[{'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':cm.sha(p)} for p in sorted(OUT.rglob('*'))
        if p.is_file() and p.name not in ['artifact_manifest.json','progress.json']]
    cm.dump(OUT/'artifact_manifest.json',{'files':files})
    archive=OUT.parent/(OUT.name+'_results.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.name!='progress.json':z.write(p,arcname=str(Path(OUT.name)/p.relative_to(OUT)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({'event':'complete','elapsed_seconds':time.time()-started}),flush=True)


if __name__=='__main__':main()
