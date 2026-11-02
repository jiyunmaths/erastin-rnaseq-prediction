#!/usr/bin/env python3
"""Shared, grouped nested-CV ridge comparisons. No feature selection or test tuning."""
import argparse
import hashlib
import html
import json
from pathlib import Path
import time
import zipfile

import numpy as np
import pandas as pd
from scipy.linalg import svd
from scipy.stats import rankdata, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import OneHotEncoder, StandardScaler

SEED = 20261031
PRIMARY = ['panel_expression', 'panel_contextual_embedding', 'all_gene_contextual_embedding']
LABELS = {'panel_expression': '40-gene expression', 'panel_contextual_embedding': 'BulkFormer: 40-gene pooling',
          'all_gene_contextual_embedding': 'BulkFormer: all-gene pooling', 'lineage_only': 'Lineage-only control',
          'training_mean': 'Training-mean control', 'panel_trunk640': 'Panel pooling: 640 channels',
          'all_gene_trunk640': 'All-gene pooling: 640 channels', 'observed_gene_pooling': 'Observed-gene pooling'}


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def clean(x):
    if isinstance(x, dict): return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)): return [clean(v) for v in x]
    if isinstance(x, np.ndarray): return clean(x.tolist())
    if isinstance(x, (np.integer,)): return int(x)
    if isinstance(x, (np.floating, float)): return float(x) if np.isfinite(x) else None
    if isinstance(x, np.bool_): return bool(x)
    return x


def dump(path, x):
    path.write_text(json.dumps(clean(x), indent=2, allow_nan=False) + '\n')


def table(path, df):
    df.to_csv(path, sep='\t', index=False, float_format='%.17g')


def metrics(y, prediction):
    y, prediction = np.asarray(y), np.asarray(prediction)
    mse = np.mean((y-prediction)**2)
    den = np.sum((y-y.mean())**2)
    rho = float(spearmanr(y, prediction).statistic) if np.ptp(y) > 0 and np.ptp(prediction) > 0 else np.nan
    return {'n': len(y), 'spearman': rho, 'rmse': float(np.sqrt(mse)),
            'r2': float(1-len(y)*mse/den) if den > 0 else np.nan}


def prepare_features(features, tr, te):
    categories = None
    if features.dtype.kind in 'OUS':
        encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False, dtype=np.float64)
        a = encoder.fit_transform(features[tr].reshape(-1, 1))
        b = encoder.transform(features[te].reshape(-1, 1))
        categories = encoder.categories_[0].tolist()
    else:
        a, b = np.asarray(features[tr], dtype=np.float64), np.asarray(features[te], dtype=np.float64)
    scaler = StandardScaler().fit(a)
    return scaler.transform(a), scaler.transform(b), scaler, categories


def ridge_path(features, y, tr, te, alphas, retain=False):
    a, b, scaler, categories = prepare_features(features, tr, te)
    mean_y = float(y[tr].mean())
    u, s, vt = svd(a, full_matrices=False, check_finite=False, lapack_driver='gesdd')
    # One SVD for all alphas, algebraically the same standardized ridge fit.
    factors = (s[:, None] / (s[:, None]**2 + np.asarray(alphas)[None, :])) * (u.T @ (y[tr]-mean_y))[:, None]
    prediction = (b @ vt.T) @ factors + mean_y
    model = None
    if retain:
        coef = vt.T @ factors[:, 0]
        model = {'mean': scaler.mean_, 'scale': scaler.scale_, 'coef': coef,
                 'intercept_after_scaling': mean_y, 'alpha': float(alphas[0]), 'categories': categories}
        # Check exported model reproduces prediction with no hidden state.
        assert np.allclose(b @ coef + mean_y, prediction[:, 0], atol=1e-12, rtol=1e-10)
    return prediction, model


def nested_fit(name, features, y, folds, alphas, n, groups):
    prediction = np.full(n, np.nan)
    fold_results, tuning, models = [], [], []
    expected_test = []
    for fold in folds:
        tr, te = fold['train'], fold['test']
        assert len(tr) > 0 and len(te) > 0
        assert set(groups[tr]).isdisjoint(groups[te])
        if name == 'training_mean':
            pred = np.full(len(te), y[tr].mean())
            selected, score = None, None
            model = {'intercept': y[tr].mean()}
        else:
            losses = []
            for inner in fold['inner']:
                a, b = inner['train'], inner['test']
                assert set(a).isdisjoint(b) and set(a) | set(b) == set(tr)
                assert set(groups[a]).isdisjoint(groups[b])
                assert set(a).isdisjoint(te) and set(b).isdisjoint(te)
                val, _ = ridge_path(features, y, a, b, alphas)
                losses.append(np.sqrt(np.mean((val-y[b, None])**2, axis=0)))
            # Equal-weight mean of validation-fold RMSE; deterministic tie order.
            mean_loss = np.mean(losses, axis=0)
            best = int(np.argmin(mean_loss)); selected = float(alphas[best]); score = float(mean_loss[best])
            for alpha, loss in zip(alphas, mean_loss):
                tuning.append({'model': name, 'fold': fold['label'], 'alpha': alpha, 'mean_inner_rmse': loss})
            p, model = ridge_path(features, y, tr, te, [selected], retain=True)
            pred = p[:, 0]
        assert np.isfinite(pred).all() and np.isnan(prediction[te]).all()
        prediction[te] = pred
        expected_test.extend(te)
        fold_results.append({'model': name, 'fold': fold['label'], 'train_n': len(tr), 'test_n': len(te),
                             'alpha': selected, 'mean_inner_rmse': score, **metrics(y[te], pred)})
        models.append({'model': name, 'fold': fold['label'], 'test_rows': te, 'fitted_model': model})
    assert len(expected_test) == len(set(expected_test))
    assert np.isfinite(prediction[expected_test]).all()
    return prediction, fold_results, tuning, models


def bootstrap(y, predictions, groups, count=5000):
    rng = np.random.default_rng(SEED + 100)
    unique = np.unique(groups)
    members = [np.flatnonzero(groups == g) for g in unique]
    draws = {name: {metric: np.empty(count) for metric in ['spearman','rmse','r2']} for name in predictions}
    for b in range(count):
        ix = np.concatenate([members[i] for i in rng.integers(len(unique), size=len(unique))])
        yy = y[ix]; ry = rankdata(yy); cy = ry - ry.mean()
        sy = np.sqrt(np.sum(cy**2)); den = np.sum((yy-yy.mean())**2)
        for name, p in predictions.items():
            pp = p[ix]; rp = rankdata(pp); cp = rp-rp.mean()
            rho_den = sy*np.sqrt(np.sum(cp**2))
            mse = np.mean((yy-pp)**2)
            draws[name]['spearman'][b] = np.sum(cy*cp)/rho_den if rho_den else np.nan
            draws[name]['rmse'][b] = np.sqrt(mse)
            draws[name]['r2'][b] = 1-len(ix)*mse/den if den else np.nan
    return draws


def test_ridge_math(features, y, fold, alpha):
    tr, te = fold['train'], fold['test']
    own, _ = ridge_path(features, y, tr, te, [alpha])
    a, b, _, _ = prepare_features(features, tr, te)
    ref = Ridge(alpha=alpha, fit_intercept=True, solver='svd').fit(a, y[tr]).predict(b)
    error = float(np.max(np.abs(own[:, 0] - ref)))
    assert error < 1e-9, error
    return error


def write_report(out, summary, comparisons, modal, loo, metadata, validations):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'svg.fonttype': 'none'})
    main = summary[summary.model.isin(PRIMARY + ['lineage_only'])].copy()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.1), constrained_layout=True)
    for ax, metric, title in zip(axes, ['spearman', 'rmse', 'r2'], ['Spearman correlation (higher better)', 'RMSE (lower better)', 'R² (higher better)']):
        x = main[metric].to_numpy(); lo = main[metric+'_ci_low'].to_numpy(); hi = main[metric+'_ci_high'].to_numpy()
        for j, (point, lower, upper) in enumerate(zip(x, lo, hi)):
            ax.plot([lower, upper], [j, j], color='#526875', lw=2)
            ax.scatter(point, j, color='#126E75', s=45, zorder=3)
        ax.set_yticks(range(len(main)), [LABELS[n] for n in main.model] if ax is axes[0] else [])
        ax.invert_yaxis();ax.set_title(title, fontsize=11);ax.grid(axis='x', alpha=.2)
        if metric != 'rmse': ax.axvline(0, color='grey', lw=.7)
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Erastin sensitivity: shared grouped out-of-fold predictions', fontsize=13)
    fig.savefig(out/'comparison_metrics.png', dpi=180);fig.savefig(out/'comparison_metrics.svg');plt.close(fig)
    pred = pd.read_csv(out/'oof_predictions.tsv', sep='\t')
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, name in zip(axes, PRIMARY):
        observed_auc = -pred.observed_sensitivity; predicted_auc = -pred[name]
        ax.scatter(observed_auc, predicted_auc, s=12, alpha=.38, color='#126E75', edgecolors='none')
        lower = min(observed_auc.min(), predicted_auc.min())-.02
        upper = max(observed_auc.max(), predicted_auc.max())+.02
        ax.plot([lower, upper], [lower, upper], '--', lw=1, color='grey')
        ax.set(xlim=(lower, upper), ylim=(lower, upper), title=LABELS[name], xlabel='Reported erastin AUC')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_ylabel('Predicted erastin AUC');fig.suptitle('Lower AUC indicates greater sensitivity; predictions are not clipped')
    fig.savefig(out/'oof_scatter.png', dpi=180);fig.savefig(out/'oof_scatter.svg');plt.close(fig)
    fmt = lambda x: 'NA' if pd.isna(x) else f'{x:.3f}'
    display = summary[['model','spearman','spearman_ci_low','spearman_ci_high','rmse','rmse_ci_low','rmse_ci_high','r2']].copy()
    display['model'] = display.model.map(lambda x: LABELS.get(x, x))
    text = ['Erastin three-model comparison — completed', '='*45, '',
        '745 cancer cell lines; 734 grouped identities; 24 lineages.',
        'Primary: five shared outer folds, four inner folds; train-only standardization; ridge tuning by mean inner-fold RMSE.',
        'Target = negative reported erastin AUC. Spearman/R2: higher better. RMSE: lower better.', '',
        display.to_string(index=False, float_format=lambda x:f'{x:.4f}'), '',
        'Paired differences (positive means first model is better):',
        comparisons.to_string(index=False, float_format=lambda x:f'{x:.4f}'), '',
        '95% intervals use 5,000 paired bootstrap resamples of identity groups and fixed out-of-fold predictions.',
        'These conditional intervals do not include refitting, split-selection or foundation-pretraining uncertainty.', '',
        'Same-dose-range sensitivity analysis: 655 lines, 0.002–66 uM, models retuned/refit using the original folds restricted to that subset.',
        modal.to_string(index=False, float_format=lambda x:f'{x:.4f}'), '',
        'Leave-lineage-out: outer test lineage completely excluded, including related training lines.',
        'Hyperparameters selected using the five pre-existing group folds restricted to the remaining lineages.',
        loo.to_string(index=False, float_format=lambda x:f'{x:.4f}'), '',
        'Interpretation limits:',
        '- The panel expression arm uses 40 genes. Both embedding arms see the entire transcriptome.',
        '- BulkFormer outputs include 640 contextual channels plus 3 global summary channels; trunk-only and observed-only pooling are secondary checks.',
        '- DepMap 24Q4 and an archived harmonized CTRP mirror are pinned. Upstream AUC integration/clipping details and direct upstream byte identity remain unresolved.',
        '- Dose range, medium and passage differences can affect drug response. Raw assay replicates and raw-well QC were unavailable.',
        '- BulkFormer pretraining overlap with these lines is unresolved. This is a downstream label-held-out comparison, not a verified pretraining-disjoint test.',
        '- No independent external drug screen was tested. Lineage holdouts and small subgroups are exploratory robustness checks.',
        '- Ridge coefficients are correlated predictive associations, not causal ferroptosis effects.',
        '- Bootstrap comparisons are descriptive and are not adjusted for multiple comparisons.', '',
        'Model resources: authors BulkFormer-147M checkpoint; actual instantiated parameter count '+str(metadata['actual_parameter_count'])+'.',
        'CPU float32 inference; strict checkpoint loading; no extra random masking; unchanged fixed attention projections.',
        'See embedding_metadata.json, model_resource_manifest.json, comparison_validation.json and execution_protocol.json for the audit trail.',
        'See oof_predictions.tsv, outer_fold_metrics.tsv, tuning_scores.tsv and fitted_outer_models.json for reproducible predictions and fits.',
        'No final all-data deployment model was fitted; the saved models are evaluation-fold models.', '']
    (out/'RESULTS.txt').write_text('\n'.join(text))
    body = ('<!doctype html><html><head><meta charset="utf-8"><title>Erastin model comparison</title>'
        '<style>body{font:16px system-ui;max-width:1150px;margin:40px auto;padding:0 24px;color:#18313b}table{border-collapse:collapse;font-size:13px}th,td{padding:7px;border-bottom:1px solid #d8e0e3;text-align:right}th:first-child,td:first-child{text-align:left}img{max-width:100%}pre{white-space:pre-wrap;line-height:1.5;font:14px system-ui}a{color:#126e75}</style></head><body>'
        '<h1>Erastin model comparison</h1><p>745 cancer cell lines · 40-gene panel · frozen BulkFormer · shared grouped validation</p>'
        +display.to_html(index=False,float_format=fmt,border=0)
        +'<img src="comparison_metrics.png" alt="Out-of-fold performance and group bootstrap intervals">'
        +'<img src="oof_scatter.png" alt="Observed versus predicted erastin AUC">'
        +'<h2>Protocol, robustness checks and limits</h2><pre>'+html.escape('\n'.join(text[7:]))+'</pre></body></html>')
    (out/'RESULTS.html').write_text(body)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input-dir', type=Path, required=True)
    ap.add_argument('--output-dir', type=Path, required=True)
    ap.add_argument('--baselines-only', action='store_true')
    ap.add_argument('--bootstrap-count', type=int, default=5000)
    args = ap.parse_args(); out=args.output_dir;out.mkdir(parents=True, exist_ok=True)
    started=time.time()
    cfg=json.loads((args.input_dir/'comparison_config.json').read_text())
    for item in json.loads((args.input_dir/'artifact_manifest.json').read_text())['files']:
        if item['path'] in ['comparison_inputs.npz','comparison_config.json','splits.json']:
            assert sha(args.input_dir/item['path']) == item['sha256']
    d=np.load(args.input_dir/'comparison_inputs.npz',allow_pickle=False)
    ids=d['sample_ids'];y=d['y_sensitivity'].astype(float);groups=d['group_ids'];n=len(ids)
    ix={v:i for i,v in enumerate(ids)}
    spec=json.loads((args.input_dir/'splits.json').read_text())
    cv=[]
    for fold in spec['outer_folds']:
        cv.append({'label':str(fold['fold']), 'train':np.array([ix[m] for m in fold['train_ModelIDs']]),
            'test':np.array([ix[m] for m in fold['test_ModelIDs']]),
            'inner':[{'train':np.array([ix[m] for m in f['train_ModelIDs']]),
                      'test':np.array([ix[m] for m in f['validation_ModelIDs']])} for f in fold['inner_folds']]})
    alphas=np.array(cfg['ridge_alphas'])
    features={'panel_expression':d['X_panel_log2tpm1'], 'lineage_only':d['lineage'], 'training_mean':None}
    if not args.baselines_only:
        metadata=json.loads((out/'embedding_metadata.json').read_text())
        assert metadata['status']=='complete' and metadata['fingerprint']['input_sha256']==sha(args.input_dir/'comparison_inputs.npz')
        emb=np.load(out/'bulkformer_embeddings.npz',allow_pickle=False)
        assert np.array_equal(emb['sample_ids'],ids)
        embedding_qc={}
        expected_aux=np.column_stack([
            np.full(n,cfg['bulkformer']['mask_prob_argument']),
            np.sum(np.where(d['X_bulkformer_lntpm1']==-10,0,d['X_bulkformer_lntpm1']),axis=1)/d['bulkformer_observed_gene_mask'].sum(),
            np.mean(d['X_bulkformer_lntpm1']!=0,axis=1)])
        for key in ['panel_embedding','all_gene_embedding','observed_gene_embedding']:
            a=emb[key]
            assert a.shape==(n,643) and np.isfinite(a).all()
            assert np.allclose(a[:,-3:],expected_aux,rtol=5e-4,atol=5e-5)
            sd=a.astype(float).std(axis=0)
            assert np.median(sd[:640])>1e-8,'Collapsed contextual representation'
            embedding_qc[key]={'shape':a.shape,'constant_feature_count':int((sd==0).sum()),
                'contextual_feature_std_min_median_max':[float(sd[:640].min()),float(np.median(sd[:640])),float(sd[:640].max())],
                'global_channel_max_abs_error_from_direct_input_summary':float(np.max(np.abs(a[:,-3:]-expected_aux)))}
        dump(out/'embedding_feature_qc.json',embedding_qc)
        features={'panel_expression':features['panel_expression'],
            'panel_contextual_embedding':emb['panel_embedding'], 'all_gene_contextual_embedding':emb['all_gene_embedding'],
            'lineage_only':features['lineage_only'], 'training_mean':None,
            'panel_trunk640':emb['panel_embedding'][:,:640], 'all_gene_trunk640':emb['all_gene_embedding'][:,:640],
            'observed_gene_pooling':emb['observed_gene_embedding']}
        for name,f in features.items():
            if f is not None and f.dtype.kind not in 'OUS':assert len(f)==n and np.isfinite(f).all()
    protocol={'seed':SEED,'input_sha256':sha(args.input_dir/'comparison_inputs.npz'),
        'splits_sha256':sha(args.input_dir/'splits.json'),'alpha_grid':alphas.tolist(),
        'alpha_selection':'minimum equal-weight mean inner-fold RMSE; ties use first/smallest grid alpha',
        'standardization':'StandardScaler fit independently on each training subset; no feature selection',
        'lineage_encoder':'OneHotEncoder fit on each training subset; unknown lineage maps to all-zero vector',
        'primary_outer_folds':5,'primary_inner_folds':4,'bootstrap_group_resamples':args.bootstrap_count,
        'bootstrap_scope':'fixed OOF predictions; does not account for model-refitting or split-selection uncertainty',
        'modal_dose_range':'refit and retune on included subset; original inner/outer folds restricted',
        'leave_lineage_out_tuning':'five saved group folds restricted to non-held-out lineages',
        'primary_models':PRIMARY,'secondary_models':['lineage_only','training_mean','panel_trunk640','all_gene_trunk640','observed_gene_pooling'],
        'no_posthoc_grid_expansion':True}
    dump(out/'execution_protocol.json',protocol)
    math_checks={name:test_ridge_math(f,y,cv[0],alphas[8]) for name,f in features.items() if f is not None and name not in ['panel_trunk640','all_gene_trunk640','observed_gene_pooling']}
    predictions={};fold_results=[];tuning=[];models=[]
    for name,f in features.items():
        p,fr,tu,mo=nested_fit(name,f,y,cv,alphas,n,groups)
        assert np.isfinite(p).all()
        predictions[name]=p;fold_results+=fr;tuning+=tu;models+=mo
        print(json.dumps({'event':'nested_model_complete','model':name,**clean(metrics(y,p))}),flush=True)
    frame=pd.DataFrame({'ModelID':ids,'group_id':groups,'lineage':d['lineage'],'outer_fold':d['outer_fold'],
                        'observed_sensitivity':y,'reported_auc':d['y_auc'],**predictions})
    table(out/'oof_predictions.tsv',frame);table(out/'outer_fold_metrics.tsv',pd.DataFrame(fold_results))
    table(out/'tuning_scores.tsv',pd.DataFrame(tuning));dump(out/'fitted_outer_models.json',models)
    coef={m['fold']:m['fitted_model']['coef'] for m in models if m['model']=='panel_expression'}
    coef_table=pd.DataFrame({'gene':d['panel_symbols'],**{'fold_'+k:v for k,v in coef.items()}})
    coef_table['mean_standardized_coefficient']=np.stack(list(coef.values())).mean(axis=0)
    table(out/'panel_ridge_coefficients.tsv',coef_table)
    if args.baselines_only:
        table(out/'baseline_metrics.tsv',pd.DataFrame([{'model':k,**metrics(y,v)} for k,v in predictions.items()]))
        dump(out/'baseline_validation.json',{'passed':True,'ridge_vs_sklearn_max_abs_error':math_checks,
            'all_745_rows_have_exactly_one_oof_prediction':True,'shared_group_splits_verified':True})
        return
    boot_names=PRIMARY+['lineage_only','training_mean']
    draws=bootstrap(y,{k:predictions[k] for k in boot_names},groups,args.bootstrap_count)
    summary=[]
    for name,p in predictions.items():
        row={'model':name,**metrics(y,p)}
        for metric in ['spearman','rmse','r2']:
            lo,hi=np.nanpercentile(draws[name][metric],[2.5,97.5]) if name in draws else (np.nan,np.nan)
            row.update({metric+'_ci_low':lo,metric+'_ci_high':hi})
        summary.append(row)
    summary=pd.DataFrame(summary);table(out/'performance_summary.tsv',summary)
    paired=[]
    for a,b in [(PRIMARY[1],PRIMARY[0]),(PRIMARY[2],PRIMARY[0]),(PRIMARY[1],PRIMARY[2])]:
        for metric in ['spearman','rmse','r2']:
            direction=-1 if metric=='rmse' else 1
            delta=direction*(draws[a][metric]-draws[b][metric]);lo,hi=np.nanpercentile(delta,[2.5,97.5])
            point=direction*(metrics(y,predictions[a])[metric]-metrics(y,predictions[b])[metric])
            paired.append({'first_model':a,'reference_model':b,'metric':metric,'improvement':point,
                           'ci_low':lo,'ci_high':hi,'bootstrap_fraction_improved':np.mean(delta>0)})
    paired=pd.DataFrame(paired);table(out/'paired_comparisons.tsv',paired)
    np.savez_compressed(out/'bootstrap_draws.npz',**{k+'__'+m:v for k,x in draws.items() for m,v in x.items()})
    # Preserve model-assessment outputs before secondary analyses begin.
    subset=d['modal_dose_range_subset'];subcv=[]
    for f in cv:
        subcv.append({'label':f['label'],'train':f['train'][subset[f['train']]],'test':f['test'][subset[f['test']]],
            'inner':[{'train':g['train'][subset[g['train']]],'test':g['test'][subset[g['test']]]} for g in f['inner']]})
    modal=[];modal_pred={}
    for name in PRIMARY+['lineage_only','training_mean']:
        p,fr,tu,mo=nested_fit(name,features[name],y,subcv,alphas,n,groups)
        modal.append({'model':name,**metrics(y[subset],p[subset])});modal_pred[name]=p[subset]
    modal=pd.DataFrame(modal);table(out/'modal_dose_metrics.tsv',modal)
    table(out/'modal_dose_predictions.tsv',pd.DataFrame({'ModelID':ids[subset],'observed_sensitivity':y[subset],**modal_pred}))
    loocv=[]
    for f in spec['leave_lineage_out']:
        tr=np.array([ix[m] for m in f['train_ModelIDs']]);te=np.array([ix[m] for m in f['test_ModelIDs']])
        loocv.append({'label':f['lineage'],'train':tr,'test':te,'inner':[
            {'train':tr[d['outer_fold'][tr]!=k],'test':tr[d['outer_fold'][tr]==k]} for k in range(5)]})
    loo=[];loo_folds=[];loo_pred={}
    for name in PRIMARY+['lineage_only','training_mean']:
        p,fr,tu,mo=nested_fit(name,features[name],y,loocv,alphas,n,groups)
        assert np.isfinite(p).all();loo_pred[name]=p
        loo.append({'model':name,**metrics(y,p)});loo_folds+=fr
        print(json.dumps({'event':'leave_lineage_complete','model':name,**clean(metrics(y,p))}),flush=True)
    loo=pd.DataFrame(loo);table(out/'leave_lineage_out_summary.tsv',loo)
    lf=pd.DataFrame(loo_folds);lf['interpretation']=np.where(lf.test_n<10,'descriptive_only_small_n','eligible')
    table(out/'leave_lineage_out_by_lineage.tsv',lf)
    table(out/'leave_lineage_out_predictions.tsv',pd.DataFrame({'ModelID':ids,'lineage':d['lineage'],'observed_sensitivity':y,**loo_pred}))
    by_lineage=[]
    for lineage in np.unique(d['lineage']):
        mask=d['lineage']==lineage
        for name in PRIMARY:
            by_lineage.append({'lineage':lineage,'model':name,**metrics(y[mask],predictions[name][mask])})
    table(out/'primary_oof_by_lineage.tsv',pd.DataFrame(by_lineage))
    validation={'passed':True,'sample_count':n,'identity_groups':len(np.unique(groups)),
        'ridge_vs_sklearn_max_abs_error':math_checks,'all_primary_oof_rows_complete':True,
        'group_disjoint_outer_inner_modal_and_lineage_splits':True,'pretrained_features_sample_order_verified':True,
        'no_test_rows_used_for_scaling_or_hyperparameter_selection':True,'finite_predictions':True,
        'fold_model_predictions_reconstruct_from_exported_parameters':True,
        'elapsed_comparison_seconds':time.time()-started,
        'selected_alphas_at_grid_boundary':pd.DataFrame(fold_results).groupby('model').alpha.apply(lambda x:int(x.isin([alphas[0],alphas[-1]]).sum())).to_dict()}
    dump(out/'comparison_validation.json',validation)
    write_report(out,summary,paired,modal,loo,metadata,validation)
    dump(out/'final_summary.json',{'status':'complete','sample_count':n,
        'primary_metrics':summary[summary.model.isin(PRIMARY+['lineage_only'])].to_dict('records'),
        'paired_comparisons':paired.to_dict('records'),
        'limits':['conditional group-bootstrap intervals','pretraining overlap unresolved',
                  'archived CTRP AUC provenance limitations','no external-screen validation']})
    outputs=[]
    for p in sorted(out.rglob('*')):
        if p.is_file() and p.name not in ['result_artifact_manifest.json','pipeline_status.json','extraction_progress.json']:
            outputs.append({'path':p.relative_to(out).as_posix(),'bytes':p.stat().st_size,'sha256':sha(p)})
    dump(out/'result_artifact_manifest.json',{'files':outputs})
    archive=out.parent/(out.name+'_results.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name not in ['pipeline_status.json','extraction_progress.json']:
                z.write(p,arcname=str(Path(out.name)/p.relative_to(out)))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    print(json.dumps({'event':'comparison_complete','elapsed_seconds':time.time()-started}),flush=True)


if __name__=='__main__':
    main()
