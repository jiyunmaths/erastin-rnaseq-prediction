#!/usr/bin/env python3
"""Repeated grouped nested CV on the fixed erastin cohort and saved embeddings."""
import argparse
import gzip
import hashlib
import html
import importlib.util
import json
from pathlib import Path
import sys
import time
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import rankdata


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def hash_file(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def weighted_ranker(values):
    """Exact midranks after integer-frequency replication, including ties."""
    order = np.argsort(values, kind='stable')
    sorted_values = values[order]
    starts = np.r_[0, np.flatnonzero(sorted_values[1:] != sorted_values[:-1])+1]
    inverse = np.argsort(order)
    group_index = np.cumsum(np.r_[True, sorted_values[1:] != sorted_values[:-1]])-1
    def rank(weights):
        group_weight = np.add.reduceat(weights[:, order], starts, axis=1)
        mid = np.cumsum(group_weight, axis=1) - group_weight/2 + 0.5
        return mid[:, group_index][:, inverse]
    return rank


def repeated_group_bootstrap(y, preds, groups, *, count, seed, batch_size=100):
    """Bootstrap identities, keeping all paired models and repeats together.

    Statistics are means of per-repeat metrics, not scores after pooling the
    20 x 745 predictions as if they were independent observations.
    """
    names = list(preds)
    repeats = next(iter(preds.values())).shape[0]
    unique, group_index = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(seed)
    draw = {name: {m: np.empty(count) for m in ['spearman','rmse','r2']} for name in names}
    yranker = weighted_ranker(y)
    prankers = {name: [weighted_ranker(row) for row in preds[name]] for name in names}
    for offset in range(0, count, batch_size):
        b = min(batch_size, count-offset)
        sampled = rng.integers(len(unique), size=(b, len(unique)))
        group_weights = np.array([np.bincount(row, minlength=len(unique)) for row in sampled], dtype=float)
        weights = group_weights[:, group_index]
        totals = weights.sum(axis=1)
        yr = yranker(weights)
        yc = yr-(totals[:,None]+1)/2
        y_rank_ss = (weights*yc**2).sum(axis=1)
        y_mean = (weights*y).sum(axis=1)/totals
        y_ss = (weights*(y[None,:]-y_mean[:,None])**2).sum(axis=1)
        for name in names:
            acc = {m: np.zeros(b) for m in ['spearman','rmse','r2']}
            for repeat in range(repeats):
                prediction = preds[name][repeat]
                pr = prankers[name][repeat](weights)
                pc = pr-(totals[:,None]+1)/2
                numerator = (weights*yc*pc).sum(axis=1)
                denominator = np.sqrt(y_rank_ss*(weights*pc**2).sum(axis=1))
                rho = np.divide(numerator, denominator, out=np.full(b,np.nan), where=denominator!=0)
                sse = weights @ ((y-prediction)**2)
                acc['spearman'] += rho/repeats
                acc['rmse'] += np.sqrt(sse/totals)/repeats
                acc['r2'] += (1-sse/y_ss)/repeats
            for metric in acc:
                draw[name][metric][offset:offset+b] = acc[metric]
        print(json.dumps({'event':'bootstrap_progress','completed':offset+b,'total':count}),flush=True)
    return draw


def validate_weighted_bootstrap(y, predictions, groups, metrics):
    # Independently expand sampled groups and compare rank statistics with scipy.
    unique, inverse = np.unique(groups, return_inverse=True)
    rng = np.random.default_rng(407)
    rank_error = 0.0
    for _ in range(5):
        wgroup = np.bincount(rng.integers(len(unique), size=len(unique)), minlength=len(unique))
        w = wgroup[inverse]
        rows = np.repeat(np.arange(len(y)), w)
        for v in [y, predictions[0], np.round(predictions[0], 2)]:
            expected = rankdata(v[rows])
            actual = weighted_ranker(v)(w[None,:])[0][rows]
            rank_error = max(rank_error, float(np.max(np.abs(expected-actual))))
            assert np.array_equal(actual, expected)
    small = {'first':predictions[:2], 'identical':predictions[:2].copy()}
    result = repeated_group_bootstrap(y,small,groups,count=5,seed=908,batch_size=5)
    rng = np.random.default_rng(908)
    members = [np.flatnonzero(groups==g) for g in unique]
    max_error = 0.0
    for b in range(5):
        rows = np.concatenate([members[i] for i in rng.integers(len(unique),size=len(unique))])
        reference = [metrics(y[rows],p[rows]) for p in predictions[:2]]
        for metric in ['spearman','rmse','r2']:
            expected = np.mean([r[metric] for r in reference])
            err = abs(result['first'][metric][b]-expected)
            max_error=max(max_error,float(err));assert err<1e-12
            assert result['first'][metric][b]==result['identical'][metric][b]
    return {'weighted_midranks_exact_vs_expanded_scipy_ranks':True,
            'max_rank_error':rank_error,'bootstrap_max_metric_error_vs_expansion':max_error,
            'identical_predictions_have_exact_zero_paired_difference':True}


def make_cv(frame, seed, make_folds):
    outer = make_folds(frame,5,seed).to_numpy()
    inner = np.full((5,len(frame)),-1,dtype=np.int16)
    folds=[]
    for fold in range(5):
        tr=np.flatnonzero(outer!=fold);te=np.flatnonzero(outer==fold)
        inner[fold,tr]=make_folds(frame.iloc[tr],4,seed+fold+1).to_numpy()
        folds.append({'label':str(fold),'train':tr,'test':te,'inner':[
            {'train':tr[inner[fold,tr]!=j],'test':tr[inner[fold,tr]==j]} for j in range(4)]})
    return outer,inner,folds


def render(out, overall, paired, per_seed, validations, cm):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    focus=per_seed[(per_seed.first_model=='panel_contextual_embedding') & (per_seed.reference_model=='panel_expression')]
    fig, axes=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
    for ax, metric, title in zip(axes,['spearman','rmse'],['Spearman improvement','RMSE reduction']):
        values=focus[focus.metric==metric].improvement.to_numpy()
        ax.scatter(values,np.arange(1,len(values)+1),s=30,color='#136f77',label='One new split seed')
        ax.axvline(0,color='#a24d4d',lw=1.3,linestyle='--',label='No improvement')
        ax.axvline(values.mean(),color='#162e38',lw=1.3,label='Mean across seeds')
        ax.set(xlabel=title+' (positive favors BulkFormer)',ylabel='Repeat',yticks=np.arange(1,len(values)+1,2))
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.2)
    axes[0].legend(loc='best',fontsize=8)
    fig.suptitle('Panel-focused BulkFormer versus 40-gene expression: 20 new grouped splits',fontsize=12)
    fig.savefig(out/'paired_seed_improvements.png',dpi=180);fig.savefig(out/'paired_seed_improvements.svg');plt.close(fig)
    summary=overall[['model','metric','mean','std_across_seeds','min','max']].copy()
    main=paired[(paired.first_model=='panel_contextual_embedding') & (paired.reference_model=='panel_expression')]
    lines=['Repeated grouped nested cross-validation — completed','='*58,'',
      'Fixed follow-up: 20 new seeds, 745 cancer lines, 734 related-identity groups.',
      'Each repeat uses 5 lineage-balanced outer folds and 4 grouped inner folds.',
      'Same saved embeddings, labels, 40 genes, ridge grid and train-only scaling.',
      'Primary follow-up contrast: panel-focused BulkFormer minus the 40-gene expression model.',
      'Seeds were fixed and all splits saved before fitting any model in this follow-up.',
      'This follow-up was designed after seeing the original single-seed results. It is not an independent cohort.', '',
      'Performance across repeats (mean and SD describe split sensitivity):',
      summary.to_string(index=False,float_format=lambda x:f'{x:.6f}'),'',
      'Primary paired improvements; positive favors panel-focused BulkFormer:',
      main.to_string(index=False,float_format=lambda x:f'{x:.6f}'),'',
      'Interpretation of uncertainty:',
      '- The number of favorable seeds describes consistency under repartitioning of this same cohort.',
      '- Repeats share cell lines and training data. They are not independent studies, and no independent-seed t-test or binomial significance test was used.',
      '- The 5,000 bootstrap resamples draw identity groups and keep all model/repeat predictions paired. Each statistic is the mean of the per-repeat scores.',
      '- Bootstrap intervals condition on these fitted out-of-fold predictions and 20 seeds. Models are not retrained within each bootstrap.',
      '- Scores are never calculated by treating 20 x 745 predictions as independent biological samples.',
      '- No seed, model variant, or hyperparameter range was selected using the outer-test scores.', '',
      'Scope and limitations:',
      '- This evaluates group-held-out cell lines with lineage balancing. It does not repeat the leave-lineage-out or external-screen evaluations.',
      '- The previous leave-lineage-out result favored the expression baseline; repeated grouped CV cannot overturn that separate result.',
      '- Panel-focused embeddings see the entire transcriptome and three global channels. They are not a 40-gene-only input model.',
      '- DepMap 24Q4 and archived CTRP AUC inputs remain fixed; AUC source-processing details and foundation-model pretraining overlap remain unresolved.',
      '- No new embeddings, gene selection, checkpoint training, or deployment model fitting was performed.', '',
      'Audit files: protocol.json, split_assignments.tsv.gz, shared_splits.npz, per_repeat_metrics.tsv,',
      'per_repeat_predictions.npz, paired_seed_differences.tsv, outer_fold_metrics.tsv, tuning_scores.tsv.gz,',
      'fitted_outer_models.json.gz, validation.json, source_manifest.json, artifact_manifest.json.', '']
    (out/'RESULTS.txt').write_text('\n'.join(lines))
    display=summary.copy();display['model']=display.model.map(cm.LABELS)
    page='<!doctype html><html><head><meta charset="utf-8"><title>Repeated erastin CV</title><style>body{font:16px system-ui;max-width:1180px;margin:35px auto;padding:0 24px;color:#19323c}table{border-collapse:collapse;font-size:13px}td,th{padding:6px;border-bottom:1px solid #dce4e7}img{max-width:100%}pre{white-space:pre-wrap;font:14px system-ui;line-height:1.45}</style></head><body>'
    page+='<h1>Repeated erastin model comparison</h1><p>20 new seeds · same 745 cell lines · frozen saved embeddings</p>'
    page+=display.to_html(index=False,float_format=lambda x:f'{x:.5f}',border=0)
    page+='<img src="paired_seed_improvements.png" alt="Per-seed improvement of panel-focused BulkFormer over expression">'
    page+='<h2>Paired effects and interpretation</h2><pre>'+html.escape('\n'.join(lines[11:]))+'</pre></body></html>'
    (out/'RESULTS.html').write_text(page)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--workspace',type=Path,required=True)
    ap.add_argument('--output-dir',type=Path,default=Path(__file__).resolve().parent)
    args=ap.parse_args();ws=args.workspace.resolve();out=args.output_dir.resolve();out.mkdir(parents=True,exist_ok=True)
    started=time.time()
    inp=ws/'outputs/erastin_40gene_comparison';prior=ws/'outputs/erastin_model_comparison'
    cm=load_module('fixed_comparison',prior/'compare_models.py')
    prep=load_module('fixed_preparation',inp/'prepare_data.py')
    files=[inp/'comparison_inputs.npz',inp/'cohort.tsv',inp/'comparison_config.json',inp/'prepare_data.py',
           inp/'splits.json',prior/'bulkformer_embeddings.npz',prior/'embedding_metadata.json',
           prior/'compare_models.py',prior/'oof_predictions.tsv']
    sources=[{'path':str(p.relative_to(ws)),'sha256':hash_file(p)} for p in files]
    for directory,mname,needed in [(inp,'artifact_manifest.json',[p.name for p in files if p.parent==inp]),
                                  (prior,'result_artifact_manifest.json',[p.name for p in files if p.parent==prior])]:
        manifest=json.load(open(directory/mname));by_name={f['path']:f['sha256'] for f in manifest['files']}
        for name in needed:assert hash_file(directory/name)==by_name[name],name
    data=np.load(inp/'comparison_inputs.npz',allow_pickle=False);emb=np.load(prior/'bulkformer_embeddings.npz',allow_pickle=False)
    ids=data['sample_ids'];y=data['y_sensitivity'];groups=data['group_ids'];n=len(ids)
    frame=pd.read_csv(inp/'cohort.tsv',sep='\t',keep_default_na=False).set_index('ModelID').loc[ids]
    assert np.array_equal(ids,emb['sample_ids']) and np.array_equal(frame.group_id.to_numpy(),groups)
    assert n==745 and len(np.unique(groups))==734 and frame.OncotreeLineage.nunique()==24
    config=json.load(open(inp/'comparison_config.json'));alphas=np.array(config['ridge_alphas'])
    original_seed=20261031;master_seed=20261101
    seeds=[int(s) for s in np.random.SeedSequence(master_seed).generate_state(20)]
    assert len(set(seeds))==20 and original_seed not in seeds
    names=['panel_expression','panel_contextual_embedding','all_gene_contextual_embedding','lineage_only']
    features={names[0]:data['X_panel_log2tpm1'],names[1]:emb['panel_embedding'],
              names[2]:emb['all_gene_embedding'],names[3]:data['lineage']}
    for name,f in features.items():
        if name!='lineage_only':assert len(f)==n and np.isfinite(f).all()
    protocol={'status':'frozen_before_followup_fitting','created_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        'original_seed_excluded_from_followup':original_seed,'seed_generator':'numpy.random.SeedSequence(master_seed).generate_state(20)',
        'master_seed':master_seed,'new_seeds':seeds,'repeats':20,'outer_folds':5,'inner_folds':4,
        'groups':734,'samples':745,'lineages':24,'split_algorithm':'unchanged prepare_data.make_folds; inner seed = outer seed + outer fold + 1',
        'models':names,'primary_contrast':'panel_contextual_embedding versus panel_expression',
        'metrics':['spearman','rmse','r2'],'alpha_grid':alphas.tolist(),
        'selection':'minimum equal-weight mean inner-fold RMSE; no outer-test tuning',
        'features':'saved unmodified embeddings and baseline expression; no new feature selection',
        'reporting':'all 20 repeats; mean, SD, range and paired win count; no seed selection',
        'bootstrap_count':5000,'bootstrap_seed':20261102,
        'bootstrap_unit':'identity group; same resampled groups for every model and repeat',
        'bootstrap_estimand':'difference of mean per-repeat metrics, conditional on fixed OOF predictions',
        'seed_pvalues':'not computed; repeats are not independent studies',
        'followup_scope':'split stability in the existing cohort; no external validation or repeated lineage holdout',
        'original_results_already_seen':True}
    protocol_path=out/'protocol.json'
    if protocol_path.exists():
        old=json.load(open(protocol_path))
        protocol['created_utc']=old['created_utc']
        assert old==protocol,'Do not overwrite a different registered follow-up'
    else:cm.dump(protocol_path,protocol)
    cm.dump(out/'source_manifest.json',{'files':sources})
    original_outer,original_inner,original_cv=make_cv(frame,original_seed,prep.make_folds)
    assert np.array_equal(original_outer,data['outer_fold'])
    for f in range(5):assert np.array_equal(original_inner[f],frame[f'inner_fold_when_outer_{f}'].to_numpy())
    all_outer=[];all_inner=[];all_cv=[];assignments=[];split_audit=[];partitions=set()
    for repeat,seed in enumerate(seeds):
        outer,inner,cv=make_cv(frame,seed,prep.make_folds)
        key=tuple(sorted(tuple(np.flatnonzero(outer==f)) for f in range(5)))
        assert key not in partitions;partitions.add(key)
        for fold in cv:
            tr,te=fold['train'],fold['test'];assert set(groups[tr]).isdisjoint(groups[te])
            for j,g in enumerate(fold['inner']):
                a,b=g['train'],g['test'];assert set(a)|set(b)==set(tr) and set(a).isdisjoint(b)
                assert set(groups[a]).isdisjoint(groups[b])
                split_audit.append({'repeat':repeat+1,'seed':seed,'outer_fold':fold['label'],'inner_fold':j,
                                    'outer_train_n':len(tr),'outer_test_n':len(te),'inner_train_n':len(a),'inner_validation_n':len(b)})
        all_outer.append(outer);all_inner.append(inner);all_cv.append(cv)
        assignments.append(pd.DataFrame({'repeat':repeat+1,'seed':seed,'ModelID':ids,'group_id':groups,
            'lineage':data['lineage'],'outer_fold':outer,**{f'inner_when_outer_{f}':inner[f] for f in range(5)}}))
    np.savez_compressed(out/'shared_splits.npz',seeds=np.array(seeds,dtype=np.uint32),sample_ids=ids,
                        group_ids=groups,outer_fold=np.array(all_outer),inner_fold=np.array(all_inner))
    cm.table(out/'split_assignments.tsv.gz',pd.concat(assignments,ignore_index=True))
    cm.table(out/'split_audit.tsv',pd.DataFrame(split_audit))
    protocol_hash=hash_file(protocol_path)
    print(json.dumps({'event':'protocol_and_all_splits_frozen','seeds':seeds,'protocol_sha256':protocol_hash}),flush=True)
    # Replay the original experiment to check equivalence before the new runs.
    old_oof=pd.read_csv(prior/'oof_predictions.tsv',sep='\t').set_index('ModelID').loc[ids]
    reproduction={}
    for name in names:
        p,_,_,_=cm.nested_fit(name,features[name],y,original_cv,alphas,n,groups)
        error=float(np.max(np.abs(p-old_oof[name].to_numpy())))
        assert error<1e-10;reproduction[name]=error
    predictions={name:np.empty((20,n)) for name in names}
    metrics_rows=[];fold_rows=[];tuning_rows=[];model_rows=[];reference_checks={}
    for repeat,(seed,cv) in enumerate(zip(seeds,all_cv)):
        for name in names:
            p,fr,tu,mo=cm.nested_fit(name,features[name],y,cv,alphas,n,groups)
            assert np.isfinite(p).all()
            predictions[name][repeat]=p
            metrics_rows.append({'repeat':repeat+1,'seed':seed,'model':name,**cm.metrics(y,p)})
            fold_rows += [{'repeat':repeat+1,'seed':seed,**r} for r in fr]
            tuning_rows += [{'repeat':repeat+1,'seed':seed,**r} for r in tu]
            model_rows += [{'repeat':repeat+1,'seed':seed,**r} for r in mo]
            if repeat in [0,19]:
                reference_checks[f'{repeat+1}:{name}']=cm.test_ridge_math(features[name],y,cv[0],fr[0]['alpha'])
        cm.table(out/'per_repeat_metrics.tsv',pd.DataFrame(metrics_rows))
        cm.dump(out/'progress.json',{'phase':'fitting','completed_repeats':repeat+1,'total_repeats':20,
            'elapsed_seconds':time.time()-started,'updated_utc':time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())})
        print(json.dumps({'event':'repeat_complete','repeat':repeat+1,'seed':seed,
            'elapsed_seconds':time.time()-started,
            'panel_spearman':metrics_rows[-3]['spearman'],'expression_spearman':metrics_rows[-4]['spearman']}),flush=True)
    assert hash_file(protocol_path)==protocol_hash
    scores=pd.DataFrame(metrics_rows)
    cm.table(out/'outer_fold_metrics.tsv',pd.DataFrame(fold_rows));cm.table(out/'tuning_scores.tsv.gz',pd.DataFrame(tuning_rows))
    (out/'fitted_outer_models.json.gz').write_bytes(gzip.compress(json.dumps(cm.clean(model_rows)).encode(),mtime=0))
    np.savez_compressed(out/'per_repeat_predictions.npz',sample_ids=ids,seeds=np.array(seeds,dtype=np.uint32),
        observed_sensitivity=y,group_ids=groups,**predictions)
    validations={'passed':True,'original_seed_outer_and_inner_splits_reproduced':True,
        'original_oof_prediction_max_abs_errors':reproduction,'all_20_new_partitions_distinct':True,
        'all_outer_and_inner_identity_groups_disjoint':True,'all_745_oof_predictions_present_per_model_per_seed':True,
        'independent_sklearn_reference_max_abs_errors':reference_checks,
        'frozen_protocol_sha256':protocol_hash,'input_and_embedding_hashes_verified':True}
    validations.update(validate_weighted_bootstrap(y,predictions[names[1]],groups,cm.metrics))
    cm.dump(out/'progress.json',{'phase':'paired_group_bootstrap','completed_repeats':20,'total_repeats':20})
    draw=repeated_group_bootstrap(y,predictions,groups,count=5000,seed=20261102)
    np.savez_compressed(out/'bootstrap_mean_metric_draws.npz',**{name+'__'+m:v for name,a in draw.items() for m,v in a.items()})
    overall=[]
    for name in names:
        s=scores[scores.model==name]
        for metric in ['spearman','rmse','r2']:
            values=s[metric].to_numpy();lo,hi=np.percentile(draw[name][metric],[2.5,97.5])
            overall.append({'model':name,'metric':metric,'mean':values.mean(),'std_across_seeds':values.std(ddof=1),
                'min':values.min(),'max':values.max(),'conditional_group_bootstrap_ci_low':lo,'conditional_group_bootstrap_ci_high':hi})
    differences=[];paired=[]
    for a,b in [(names[1],names[0]),(names[2],names[0]),(names[1],names[2])]:
        sa=scores[scores.model==a].set_index('seed').loc[seeds];sb=scores[scores.model==b].set_index('seed').loc[seeds]
        for metric in ['spearman','rmse','r2']:
            direction=-1 if metric=='rmse' else 1
            effect=direction*(sa[metric]-sb[metric]).to_numpy()
            uncertainty=direction*(draw[a][metric]-draw[b][metric]);lo,hi=np.percentile(uncertainty,[2.5,97.5])
            paired.append({'first_model':a,'reference_model':b,'metric':metric,'mean_improvement':effect.mean(),
                'sd_across_seeds':effect.std(ddof=1),'min_improvement':effect.min(),'max_improvement':effect.max(),
                'favorable_seeds':int((effect>0).sum()),'total_seeds':20,'conditional_group_bootstrap_ci_low':lo,
                'conditional_group_bootstrap_ci_high':hi})
            differences += [{'repeat':r+1,'seed':seed,'first_model':a,'reference_model':b,'metric':metric,'improvement':v}
                            for r,(seed,v) in enumerate(zip(seeds,effect))]
    overall=pd.DataFrame(overall);paired=pd.DataFrame(paired);differences=pd.DataFrame(differences)
    cm.table(out/'mean_performance.tsv',overall);cm.table(out/'paired_summary.tsv',paired)
    cm.table(out/'paired_seed_differences.tsv',differences)
    validations['protocol_unchanged_after_results']=hash_file(protocol_path)==protocol_hash
    validations['alpha_boundary_selection_counts']=pd.DataFrame(fold_rows).groupby('model').alpha.apply(lambda a:int(a.isin([alphas[0],alphas[-1]]).sum())).to_dict()
    validations['elapsed_seconds']=time.time()-started
    cm.dump(out/'validation.json',validations)
    render(out,overall,paired,differences,validations,cm)
    primary=paired[(paired.first_model==names[1]) & (paired.reference_model==names[0])]
    cm.dump(out/'final_summary.json',{'status':'complete','repeats':20,'samples':n,'groups':734,
        'primary_paired_results':primary.to_dict('records'),'mean_performance':overall.to_dict('records'),
        'uncertainty_note':'Repeated splits reuse the same cohort. Bootstrap intervals condition on all fitted OOF predictions; no independent-seed significance test.',
        'elapsed_seconds':time.time()-started})
    artifact=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name not in ['artifact_manifest.json','progress.json']:
            artifact.append({'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':hash_file(p)})
    cm.dump(out/'artifact_manifest.json',{'files':artifact})
    archive=out.parent/(out.name+'_results.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='progress.json':z.write(p,arcname=str(Path(out.name)/p.relative_to(out)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    cm.dump(out/'progress.json',{'phase':'complete','completed_repeats':20,'total_repeats':20,'elapsed_seconds':time.time()-started})
    print(json.dumps({'event':'complete','elapsed_seconds':time.time()-started,'primary_results':cm.clean(primary.to_dict('records'))}),flush=True)


if __name__=='__main__':main()
