#!/usr/bin/env python3
"""Independent reconstruction, sklearn reference, readable review and archive."""
import concurrent.futures as futures
import gzip
import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT=Path(__file__).resolve().parent
WS=OUT.parent.parent
PRIMARY=['panel_expression','panel_contextual_embedding','all_gene_contextual_embedding']
METRICS=['spearman','rmse','r2']
LABELS=dict(zip(PRIMARY,['40-gene expression','Panel-focused BulkFormer','All-gene BulkFormer']))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def refit(name,label,x,y,tr,te,inner,alphas,expected,selected):
    cv=[(np.flatnonzero(inner!=k),np.flatnonzero(inner==k)) for k in range(5)]
    fit=GridSearchCV(make_pipeline(StandardScaler(),Ridge(solver='svd')),
        {'ridge__alpha':alphas},scoring='neg_root_mean_squared_error',cv=cv,n_jobs=1)
    fit.fit(x[tr].astype(float),y[tr])
    error=float(np.max(np.abs(fit.predict(x[te].astype(float))-expected)))
    assert error<1e-10 and fit.best_params_['ridge__alpha']==selected
    return {'model':name,'held_out_lineage':label,'selected_alpha':selected,'prediction_max_abs_error':error}


def review():
    pooled=pd.read_csv(OUT/'pooled_summary.tsv',sep='\t')
    macro=pd.read_csv(OUT/'macro_summary.tsv',sep='\t')
    paired=pd.read_csv(OUT/'paired_comparisons.tsv',sep='\t')
    stability=pd.read_csv(OUT/'seed_stability_summary.tsv',sep='\t')
    wins=pd.read_csv(OUT/'lineage_win_counts.tsv',sep='\t')
    focus=paired[(paired.first_model==PRIMARY[1])&(paired.reference_model==PRIMARY[0])]
    seed_focus=stability[(stability.first_model==PRIMARY[1])&(stability.reference_model==PRIMARY[0])]
    effect=focus[focus.scope=='mean_20_seeds_pooled_745'].set_index('metric')
    if (effect.improvement>0).all():
        conclusion='Panel-focused BulkFormer has higher mean pooled performance on this leave-lineage-out benchmark; the paired intervals and within-lineage results below determine how much confidence to place in that advantage.'
    else:
        conclusion='The corrected leave-lineage-out results do not support consistent overall superiority of panel-focused BulkFormer over the 40-gene expression baseline.'
    lines=['# Corrected leave-lineage-out comparison','',conclusion,'',
        'Reused the saved embeddings for 745 cancer cell lines in 734 identity groups. Each of 24 lineages was completely excluded from supervised training and tuning when it was evaluated. Original outer holdouts, labels, ridge penalty grid and train-only standardization were preserved.','',
        'The direct corrected rerun uses the original seed (20261031). The sensitivity analysis uses the same 20 fixed seeds as the preceding grouped-CV study. Each seed changes only the five inner folds; the 24 outer lineage tests stay fixed.','',
        '## Pooled predictions across all 24 held-out lineages','',
        'Means across the 20 fixed inner-fold seeds; 745 held-out predictions per seed. These are means of metrics, not metrics of an averaged-prediction ensemble.','',
        '| Model | Spearman ↑ | RMSE ↓ | R² ↑ |','|---|---:|---:|---:|']
    for name in PRIMARY:
        r=pooled[(pooled.scope=='mean_20_inner_seeds')&(pooled.model==name)].iloc[0]
        lines.append(f'| {LABELS[name]} | {r.spearman:.4f} | {r.rmse:.5f} | {r.r2:.4f} |')
    lines+=['','## Equal weight for each of 19 larger lineages','',
        'Mean within-lineage scores, then averaged over the 20 seeds. The established n ≥ 10 threshold excludes five small lineages from this summary; all 24 remain in pooled metrics. Pooled correlation can reflect between-lineage differences, so it is not a substitute for within-lineage ranking.','',
        '| Model | Mean within-lineage Spearman ↑ | Mean within-lineage RMSE ↓ | Mean within-lineage R² ↑ |',
        '|---|---:|---:|---:|']
    for name in PRIMARY:
        r=macro[(macro.scope=='mean_20_inner_seeds')&(macro.model==name)].iloc[0]
        lines.append(f'| {LABELS[name]} | {r.spearman:.4f} | {r.rmse:.5f} | {r.r2:.4f} |')
    lines+=['','## Panel-focused BulkFormer versus expression','',
        'Positive changes favor panel-focused BulkFormer. Intervals are paired, conditional 95% whole-lineage bootstrap intervals (5,000 draws).','',
        '| Summary | Metric | Mean improvement | 95% interval | Favorable inner-fold seeds |',
        '|---|---|---:|---:|---:|']
    for scope,short,seed_scope in [('mean_20_seeds_pooled_745','Pooled','pooled_745'),('mean_20_seeds_macro_19','Equal-weight lineages','macro_19_lineages')]:
        for metric,description in [('spearman','Spearman increase'),('rmse','RMSE reduction'),('r2','R² increase')]:
            r=focus[(focus.scope==scope)&(focus.metric==metric)].iloc[0]
            s=seed_focus[(seed_focus.scope==seed_scope)&(seed_focus.metric==metric)].iloc[0]
            lines.append(f'| {short} | {description} | {r.improvement:+.5f} | [{r.conditional_lineage_bootstrap_ci_low:+.5f}, {r.conditional_lineage_bootstrap_ci_high:+.5f}] | {int(s.favorable_seeds)}/20 |')
    lines+=['','Among the 19 lineages with at least 10 samples, using the mean effect across 20 seeds:','']
    for metric,description in [('spearman','higher Spearman'),('rmse','lower RMSE'),('r2','higher R²')]:
        r=wins[(wins.scope=='mean_20_inner_seeds')&(wins.metric==metric)].iloc[0]
        lines.append(f'- Panel-focused BulkFormer has {description} in {int(r.favorable_lineages)}/19 lineages.')
    lines+=['',f'![Per-lineage effects]({OUT / "lineage_effects.png"})','',f'![Inner-seed sensitivity]({OUT / "inner_seed_effects.png"})','',
        '## Direct original-seed correction','',
        '| Model | Legacy Spearman | Corrected Spearman | Legacy RMSE | Corrected RMSE |',
        '|---|---:|---:|---:|---:|']
    legacy=pd.read_csv(OUT/'legacy_vs_corrected.tsv',sep='\t')
    for name in PRIMARY:
        a=legacy[(legacy.model==name)&(legacy.scope=='legacy_original_seed')].iloc[0]
        b=legacy[(legacy.model==name)&(legacy.scope=='corrected_original_seed')].iloc[0]
        lines.append(f'| {LABELS[name]} | {a.spearman:.4f} | {b.spearman:.4f} | {a.rmse:.5f} | {b.rmse:.5f} |')
    lines+=['',
        'The original predictions were reproduced before the corrected evaluation. The prior bug shuffled a view of a pandas index and corrupted lineage-count lookup. The corrected splitter copies group IDs and verifies tracked counts against actual fold membership. Related-identity groups remained intact in both versions. No original artifacts were overwritten.','',
        '## Interpretation and checks','',
        '- These are unseen lineages for the supervised ridge head. BulkFormer pretraining overlap is unresolved; no pretraining-disjoint claim is made.',
        '- Panel-focused embeddings pool the selected 40 gene tokens after processing the full transcriptome. This is not an assay requiring only 40 measured genes.',
        '- The 20 inner-fold seeds reuse the same data and outer tests. They quantify tuning-split sensitivity, not 20 independent experiments.',
        '- The bootstrap resamples complete held-out lineages, keeping models and seeds paired. Models are not retrained in each draw, and training data overlap across outer holdouts. Intervals are conditional descriptive uncertainty, not a complete estimate of future-lineage risk.',
        '- Results retain the archived CTRP AUC and DepMap 24Q4 limitations. No external screen or clinical cohort was added.',
        '- All saved outer models were independently reconstructed. Selected penalties match recorded inner-fold minima. Independent sklearn Pipeline/GridSearchCV fits were checked for the Lung and Thyroid holdouts across all three primary models.',
        '- See `protocol.json`, `validation.json`, `independent_review.json`, `shared_splits.npz`, prediction/model files and `artifact_manifest.json` for the audit trail.','']
    (OUT/'REVIEW.md').write_text('\n'.join(lines))


def main():
    assert json.loads((OUT/'progress.json').read_text())['phase']=='complete'
    for item in json.loads((OUT/'source_manifest.json').read_text())['files']:
        assert digest(WS/item['path'])==item['sha256']
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert digest(OUT/'run_comparison.py')==protocol['script_sha256']
    d=np.load(WS/'outputs/erastin_40gene_comparison/comparison_inputs.npz',allow_pickle=False)
    e=np.load(WS/'outputs/erastin_model_comparison/bulkformer_embeddings.npz',allow_pickle=False)
    splits=np.load(OUT/'shared_splits.npz',allow_pickle=False)
    preds=np.load(OUT/'per_run_predictions.npz',allow_pickle=False)
    ids=d['sample_ids'];groups=d['group_ids'];lineages=d['lineage'];y=d['y_sensitivity']
    for bundle in [e,splits,preds]:assert np.array_equal(bundle['sample_ids'],ids)
    assert np.array_equal(splits['seeds'],preds['seeds'])
    labels=list(splits['holdout_labels']);split_checks=0
    for repeat in range(21):
        for i,label in enumerate(labels):
            te=np.flatnonzero(lineages==label);tr=np.flatnonzero(~np.isin(groups,groups[te]))
            inner=splits['inner_fold_by_holdout'][repeat,i]
            assert np.all(inner[te]==-1) and np.all(inner[tr]>=0)
            assert not np.any(lineages[tr]==label) and set(groups[tr]).isdisjoint(groups[te])
            for k in range(5):
                a=tr[inner[tr]!=k];b=tr[inner[tr]==k]
                assert set(a)|set(b)==set(tr) and set(groups[a]).isdisjoint(groups[b])
                assert set(a).isdisjoint(te) and set(b).isdisjoint(te)
                split_checks+=1
    scores=pd.read_csv(OUT/'per_run_metrics.tsv',sep='\t')
    metric_error=0.
    for row in scores.itertuples():
        p=preds[row.model][row.repeat] if row.model in PRIMARY else preds[row.model]
        values={'spearman':spearmanr(y,p).statistic,'rmse':np.sqrt(np.mean((y-p)**2)),
                'r2':1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)}
        metric_error=max(metric_error,max(abs(v-getattr(row,m)) for m,v in values.items()))
    assert metric_error<1e-12 and len(scores)==65
    features={PRIMARY[0]:d['X_panel_log2tpm1'],PRIMARY[1]:e['panel_embedding'],PRIMARY[2]:e['all_gene_embedding'],'lineage_only':lineages}
    models=json.loads(gzip.decompress((OUT/'fitted_outer_models.json.gz').read_bytes()))
    assert len(models)==1560
    tuning=pd.read_csv(OUT/'tuning_scores.tsv.gz',sep='\t')
    tuned={(int(r),n,f):v for (r,n,f),v in tuning.groupby(['repeat','model','fold'])}
    max_error=0.;selection_checks=0
    for model in models:
        name=model['model'];r=model['repeat'];label=model['fold'];test=np.array(model['test_rows']);fit=model['fitted_model']
        assert np.array_equal(test,np.flatnonzero(lineages==label))
        if name=='training_mean':p=np.full(len(test),fit['intercept'])
        else:
            x=features[name][test]
            if name=='lineage_only':x=(x[:,None]==np.array(fit['categories'])[None,:]).astype(float)
            else:x=x.astype(float)
            p=((x-np.array(fit['mean']))/np.array(fit['scale']))@np.array(fit['coef'])+fit['intercept_after_scaling']
            v=tuned[(r,name,label)];assert len(v)==17
            assert np.isclose(v.loc[v.mean_inner_rmse.idxmin(),'alpha'],fit['alpha'],rtol=1e-13,atol=0)
            selection_checks+=1
        expected=preds[name][r,test] if name in PRIMARY else preds[name][test]
        max_error=max(max_error,float(np.max(np.abs(p-expected))))
    assert max_error<1e-12 and selection_checks==1536
    alphas=protocol['alpha_grid'];reference=[]
    with futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[]
        for label in ['Lung','Thyroid']:
            te=np.flatnonzero(lineages==label);tr=np.flatnonzero(lineages!=label)
            inner=splits['inner_fold_by_holdout'][0,labels.index(label),tr]
            for name in PRIMARY:
                selected=next(m['fitted_model']['alpha'] for m in models if m['repeat']==0 and m['model']==name and m['fold']==label)
                jobs.append(pool.submit(refit,name,label,features[name],y,tr,te,inner,alphas,preds[name][0,te],selected))
        reference=[job.result() for job in jobs]
    audit={'passed':True,'source_hashes_and_frozen_run_script_unchanged':True,'independent_inner_split_checks':split_checks,
        'outer_fitted_models_reconstructed':len(models),'selected_penalties_checked':selection_checks,
        'recomputed_metric_rows':len(scores),'max_metric_error':metric_error,'max_exported_model_prediction_error':max_error,
        'independent_sklearn_nested_refits':reference}
    (OUT/'independent_review.json').write_text(json.dumps(audit,indent=2)+'\n')
    review()
    files=[{'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(OUT.rglob('*'))
        if p.is_file() and p.name not in ['artifact_manifest.json','progress.json']]
    (OUT/'artifact_manifest.json').write_text(json.dumps({'files':files},indent=2)+'\n')
    archive=OUT.parent/(OUT.name+'_results.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.name!='progress.json':z.write(p,arcname=str(Path(OUT.name)/p.relative_to(OUT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for item in files:assert hashlib.sha256(z.read(str(Path(OUT.name)/item['path']))).hexdigest()==item['sha256']
    print(json.dumps(audit,indent=2),flush=True)


if __name__=='__main__':main()
