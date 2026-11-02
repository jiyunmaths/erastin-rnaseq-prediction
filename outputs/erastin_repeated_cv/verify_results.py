#!/usr/bin/env python3
"""Independent readback, nested sklearn reference and packaging of CV results."""
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


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    out=Path(__file__).resolve().parent
    ws=out.parent.parent
    inputs=ws/'outputs/erastin_40gene_comparison'
    prior=ws/'outputs/erastin_model_comparison'
    assert json.loads((out/'progress.json').read_text())['phase']=='complete'
    for item in json.loads((out/'source_manifest.json').read_text())['files']:
        assert digest(ws/item['path'])==item['sha256']
    data=np.load(inputs/'comparison_inputs.npz',allow_pickle=False)
    emb=np.load(prior/'bulkformer_embeddings.npz',allow_pickle=False)
    split=np.load(out/'shared_splits.npz',allow_pickle=False)
    pred=np.load(out/'per_repeat_predictions.npz',allow_pickle=False)
    ids=data['sample_ids'];y=data['y_sensitivity'];groups=data['group_ids']
    for bundle in [emb,split,pred]:assert np.array_equal(bundle['sample_ids'],ids)
    assert np.array_equal(split['seeds'],pred['seeds'])
    features={'panel_expression':data['X_panel_log2tpm1'],
              'panel_contextual_embedding':emb['panel_embedding'],
              'all_gene_contextual_embedding':emb['all_gene_embedding'],
              'lineage_only':data['lineage']}
    # Independently check memberships rather than relying on pipeline assertions.
    for repeat in range(20):
        outer=split['outer_fold'][repeat]
        assert len(outer)==745 and set(outer)==set(range(5))
        for fold in range(5):
            tr=np.flatnonzero(outer!=fold);te=np.flatnonzero(outer==fold)
            assert set(groups[tr]).isdisjoint(groups[te])
            inner=split['inner_fold'][repeat,fold]
            assert np.all(inner[te]==-1) and set(inner[tr])==set(range(4))
            for j in range(4):
                a=tr[inner[tr]!=j];b=tr[inner[tr]==j]
                assert set(groups[a]).isdisjoint(groups[b])
                assert set(a)|set(b)==set(tr)
    metrics=pd.read_csv(out/'per_repeat_metrics.tsv',sep='\t')
    metric_error=0.
    for row in metrics.itertuples():
        p=pred[row.model][row.repeat-1]
        calc={'spearman':spearmanr(y,p).statistic,
              'rmse':np.sqrt(np.mean((y-p)**2)),
              'r2':1-np.sum((y-p)**2)/np.sum((y-y.mean())**2)}
        metric_error=max(metric_error,max(abs(v-getattr(row,k)) for k,v in calc.items()))
    assert metric_error<1e-12
    models=json.loads(gzip.decompress((out/'fitted_outer_models.json.gz').read_bytes()))
    tune=pd.read_csv(out/'tuning_scores.tsv.gz',sep='\t')
    assert len(models)==400
    reconstruction_error=0.
    for saved in models:
        r=saved['repeat']-1;fold=int(saved['fold']);name=saved['model']
        test=np.array(saved['test_rows']);fit=saved['fitted_model']
        assert np.array_equal(test,np.flatnonzero(split['outer_fold'][r]==fold))
        x=features[name][test]
        if name=='lineage_only':x=(x[:,None]==np.array(fit['categories'])[None,:]).astype(float)
        else:x=x.astype(float)
        p=((x-np.array(fit['mean']))/np.array(fit['scale']))@np.array(fit['coef'])+fit['intercept_after_scaling']
        reconstruction_error=max(reconstruction_error,float(np.max(np.abs(p-pred[name][r,test]))))
        rows=tune[(tune.repeat==r+1)&(tune.model==name)&(tune.fold==fold)]
        assert len(rows)==17
        assert np.isclose(float(rows.loc[rows.mean_inner_rmse.idxmin(),'alpha']),fit['alpha'],rtol=1e-13,atol=0)
    assert reconstruction_error<1e-12
    # An independent full nested-tuning calculation for the first outer fold
    # of the first repeat, using sklearn's Pipeline and GridSearchCV directly.
    tr=np.flatnonzero(split['outer_fold'][0]!=0);te=np.flatnonzero(split['outer_fold'][0]==0)
    inner=split['inner_fold'][0,0,tr]
    cv=[(np.flatnonzero(inner!=j),np.flatnonzero(inner==j)) for j in range(4)]
    alphas=json.loads((inputs/'comparison_config.json').read_text())['ridge_alphas']
    reference={}
    for name,x in features.items():
        if name=='lineage_only':continue
        search=GridSearchCV(make_pipeline(StandardScaler(),Ridge(solver='svd')),
            {'ridge__alpha':alphas},scoring='neg_root_mean_squared_error',cv=cv,n_jobs=1)
        search.fit(x[tr].astype(float),y[tr])
        error=float(np.max(np.abs(search.predict(x[te].astype(float))-pred[name][0,te])))
        saved=next(m for m in models if m['repeat']==1 and int(m['fold'])==0 and m['model']==name)
        assert search.best_params_['ridge__alpha']==saved['fitted_model']['alpha']
        assert error<1e-10
        reference[name]={'selected_alpha':search.best_params_['ridge__alpha'],'max_abs_oof_error':error}
    audit={'passed':True,'all_source_hashes_unchanged':True,'independent_split_group_audit_passed':True,
        'n_repeats':20,'n_outer_fitted_models':400,'n_recomputed_oof_metric_rows':len(metrics),
        'max_recomputed_metric_error':metric_error,'max_exported_model_reconstruction_error':reconstruction_error,
        'all_400_selected_alphas_match_minimum_inner_rmse':True,
        'independent_nested_sklearn_pipeline_reference':reference}
    (out/'independent_review.json').write_text(json.dumps(audit,indent=2)+'\n')
    overall=pd.read_csv(out/'mean_performance.tsv',sep='\t')
    paired=pd.read_csv(out/'paired_summary.tsv',sep='\t')
    primary=paired[(paired.first_model=='panel_contextual_embedding')&(paired.reference_model=='panel_expression')].set_index('metric')
    labels={'panel_expression':'40-gene expression','panel_contextual_embedding':'Panel-focused BulkFormer',
            'all_gene_contextual_embedding':'All-gene BulkFormer','lineage_only':'Lineage-only control'}
    lines=['# Repeated erastin sensitivity comparison','',
        'Completed 20 fixed seeds on the same 745 cancer cell lines (734 identity groups), reusing saved embeddings. Each repeat used five outer folds and four inner folds, shared across models; scaling and ridge selection used training data only. Target: negative reported erastin AUC, so larger values mean greater sensitivity.','',
        '| Model | Mean Spearman ± SD | Mean RMSE ± SD | Mean R² ± SD |',
        '|---|---:|---:|---:|']
    for name,label in labels.items():
        rows=overall[overall.model==name].set_index('metric')
        values=[f"{rows.loc[m,'mean']:.4f} ± {rows.loc[m,'std_across_seeds']:.4f}" for m in ['spearman','rmse','r2']]
        lines.append('| '+label+' | '+' | '.join(values)+' |')
    lines+=['','SD describes variation under repartitioning, not the standard error of independent datasets.','',
        '| Panel-focused improvement over expression | Mean change | Favorable seeds | Conditional 95% group-bootstrap interval |',
        '|---|---:|---:|---:|']
    for m,label in [('spearman','Spearman increase'),('rmse','RMSE reduction'),('r2','R² increase')]:
        r=primary.loc[m]
        lines.append(f"| {label} | {r.mean_improvement:.5f} | {int(r.favorable_seeds)}/20 | [{r.conditional_group_bootstrap_ci_low:.5f}, {r.conditional_group_bootstrap_ci_high:.5f}] |")
    raw_rmse=overall[(overall.model=='panel_expression')&(overall.metric=='rmse')]['mean'].iloc[0]
    reduction=100*primary.loc['rmse','mean_improvement']/raw_rmse
    crosses=all(primary.conditional_group_bootstrap_ci_low<=0)&all(primary.conditional_group_bootstrap_ci_high>=0)
    interval_note='All three paired intervals include zero.' if crosses else 'See the paired intervals above for uncertainty.'
    lines+=['',f"The mean RMSE reduction is {reduction:.2f}%. {interval_note}",'',
        'The win counts measure split stability in this cohort. The 20 repeats reuse the same cells and are not independent studies. The 5,000 bootstrap draws resample identity groups and retain all models and repeats together; intervals condition on fitted out-of-fold predictions and do not include bootstrap refitting.','',
        '## Split-generation correction','',
        'Validation uncovered a bug in the earlier split generator: shuffling a NumPy view also shuffled the pandas index used for lineage counts. Identity groups stayed intact, but lineage balancing was corrupted. The corrected generator copies IDs before shuffling and verifies actual lineage counts for every outer and inner split. Across the original seed and the 20 follow-up seeds, maximum outer-fold lineage deviation from the ideal count fell from up to 9 cells to 0.8 cells.','',
        'The first repeated run is preserved in `../erastin_repeated_cv_legacy_splits/`. Its results were visible before the corrected rerun. The correction was motivated by the reproduced indexing bug; seeds, features, models, ridge grid, metrics and bootstrap design were unchanged. `legacy_vs_corrected.tsv` reports both versions. Original input data, embeddings and single-seed results were preserved.','',
        '## Scope and verification','',
        'This supports a within-cohort stability assessment, not an external validation claim. Panel-focused BulkFormer pools the 40 selected gene tokens after processing the full transcriptome; it is not a model requiring only 40 measured genes. Checkpoint pretraining overlap and original AUC processing details remain unresolved.','',
        'The prior pooled leave-lineage-out analysis favored raw expression. Its inner folds used the legacy generator, and that analysis was not rerun here; these repeated grouped splits do not establish improvement on unseen lineages.','',
        'All 400 saved outer models reproduce their out-of-fold predictions, all selected penalties match the minimum inner-fold RMSE, and all source hashes remain unchanged. An independent sklearn Pipeline/GridSearchCV refit reproduces the selected penalties and predictions for all three primary models on repeat 1, outer fold 0. See `independent_review.json` and `validation.json`.','',
        'Detailed tables, shared splits, predictions, fitted models, frozen protocol and audit files are included in `../erastin_repeated_cv_results.zip`.','']
    (out/'REVIEW.md').write_text('\n'.join(lines))
    files=[{'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p)} for p in sorted(out.rglob('*')) if p.is_file() and p.name not in ['artifact_manifest.json','progress.json']]
    (out/'artifact_manifest.json').write_text(json.dumps({'files':files},indent=2)+'\n')
    archive=out.parent/(out.name+'_results.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='progress.json':z.write(p,arcname=str(Path(out.name)/p.relative_to(out)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps(audit,indent=2))


if __name__=='__main__':main()
