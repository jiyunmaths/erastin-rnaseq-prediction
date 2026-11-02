#!/usr/bin/env python3
"""Audit and summarize the fixed nonlinear-head comparison."""
import concurrent.futures as futures
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
WS=OUT.parent.parent


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


run=module('nonlinear_methods',OUT/'run_models.py')
REPS=run.REPS
METRICS=['spearman','rmse','r2']
REP_LABELS=dict(zip(REPS,['40-gene expression','Panel-focused BulkFormer','All-gene BulkFormer']))
FAMILY_LABELS={'ridge':'Ridge','random_forest':'Random forest','neural_network':'Neural network (3-member mean)'}
ARMS=[f+'__'+r for f in run.FAMILIES for r in REPS]


def table(path,frame):
    frame.to_csv(path,sep='\t',index=False,float_format='%.17g')


def reference_fit(family,rep,params,x,y,tr,te,expected,protocol):
    if family=='random_forest':
        model=RandomForestRegressor(**protocol['rf_fixed'],**params)
        model.fit(x[tr],y[tr]);pred=model.predict(x[te])
    else:
        predictions=[]
        for seed in protocol['nn_fixed']['initialization_seeds']:
            nn=MLPRegressor(hidden_layer_sizes=(params['width'],),alpha=params['alpha'],activation='tanh',
                solver='lbfgs',max_iter=2000,max_fun=40000,tol=1e-3,early_stopping=False,random_state=seed)
            model=TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(),nn),transformer=StandardScaler())
            model.fit(x[tr],y[tr]);predictions.append(model.predict(x[te]))
        pred=np.mean(predictions,axis=0)
    err=float(np.max(np.abs(pred-expected)));assert err<1e-8,(family,rep,err)
    return {'family':family,'representation':rep,'lineage':'Lung','max_abs_prediction_error':err}


def render(summary,paired,wins,selection,convergence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,3,figsize=(13,4.7),constrained_layout=True)
    for ax,metric,title,cmap in zip(axes,['spearman','rmse','macro_spearman'],
        ['Pooled Spearman ↑','Pooled RMSE ↓','Mean within-lineage Spearman ↑'],['YlGnBu','YlGnBu_r','YlGnBu']):
        values=np.array([[summary.set_index('arm').loc[f+'__'+r,metric] for r in REPS] for f in run.FAMILIES])
        im=ax.imshow(values,cmap=cmap,aspect='auto')
        ax.set_xticks(range(3),['Expression','Panel FM','All-gene FM'],rotation=25,ha='right')
        ax.set_yticks(range(3),['Ridge','Random forest','Neural network'] if ax is axes[0] else [])
        for i in range(3):
            for j in range(3):
                normalized=im.norm(values[i,j]);dark=(normalized>.65 if cmap!='YlGnBu_r' else normalized<.35)
                ax.text(j,i,f'{values[i,j]:.4f}',ha='center',va='center',color='white' if dark else '#182f3a',fontsize=11)
        ax.set_title(title,fontsize=11)
    fig.suptitle('New prediction heads on the same corrected leave-lineage-out splits\n745 cell lines · 24 held-out lineages · fixed inner split seed',fontsize=13)
    fig.savefig(OUT/'model_comparison.png',dpi=170);fig.savefig(OUT/'model_comparison.svg');plt.close(fig)
    lines=['# Random-forest and neural-network comparison','',
        'The small neural-network ensemble on 40-gene expression had the best pooled point estimates, with only a marginal gain over expression ridge. Its paired Spearman and RMSE improvement intervals include zero. Neither nonlinear prediction head established an overall BulkFormer advantage on unseen lineages in this bounded search.','',
        'Completed nested leave-lineage-out evaluation on 745 cell lines (734 identity groups, 24 lineages), reusing the saved expression and BulkFormer representations. Ridge, random forest and a small neural-network ensemble use the same corrected outer holdouts and five inner folds.','',
        '**This is a model-family comparison on one fixed inner split scheme (seed 20261031), not a repeat over the previous 20 split seeds.** The three neural-network initializations are ensemble members, not independent validation datasets.','',
        '| Prediction head | Representation | Pooled Spearman ↑ | Pooled RMSE ↓ | Pooled R² ↑ | Mean within-lineage Spearman ↑ |',
        '|---|---|---:|---:|---:|---:|']
    indexed=summary.set_index('arm')
    for arm in ARMS:
        r=indexed.loc[arm]
        lines.append(f'| {FAMILY_LABELS[r.family]} | {REP_LABELS[r.representation]} | {r.spearman:.4f} | {r.rmse:.5f} | {r.r2:.4f} | {r.macro_spearman:.4f} |')
    chosen=indexed.loc['nested_selector'];base=indexed.loc['ridge__panel_expression']
    best=summary[summary.arm.isin(ARMS)].sort_values('rmse').iloc[0]
    lines+=['',f'The lowest observed pooled RMSE among the nine fixed arms was {best.rmse:.5f} for {FAMILY_LABELS[best.family]} on {REP_LABELS[best.representation]}. This is an exploratory ranking using held-out results, not a separately validated model-selection claim.','',
        'Panel-focused BulkFormer with a neural network had the highest mean within-lineage Spearman (0.2636), but its pooled RMSE was worse (0.12896). Improvements in within-lineage ranking did not translate into better overall response prediction.','',
        f'The prespecified nested selector chose the model and representation using only inner-validation RMSE in each outer fold. Its pooled Spearman was {chosen.spearman:.4f}, RMSE {chosen.rmse:.5f}, and R² {chosen.r2:.4f}; the expression ridge reference was {base.spearman:.4f}, {base.rmse:.5f}, and {base.r2:.4f}.','',
        f'![Model-family comparison]({OUT / "model_comparison.png"})','',
        '## Paired comparisons against expression ridge','',
        'Positive improvement favors the first model. Intervals are conditional 95% whole-lineage bootstrap intervals; all models remain paired.','',
        '| Model | Spearman improvement [95% interval] | RMSE reduction [95% interval] |',
        '|---|---:|---:|']
    for arm in [a for a in ARMS if a!='ridge__panel_expression']+['nested_selector']:
        p=paired[(paired.scope=='pooled_745')&(paired.first_model==arm)&(paired.reference_model=='ridge__panel_expression')].set_index('metric')
        label='Nested model/representation selector' if arm=='nested_selector' else FAMILY_LABELS[indexed.loc[arm,'family']]+' / '+REP_LABELS[indexed.loc[arm,'representation']]
        strings=[f"{p.loc[m,'improvement']:+.5f} [{p.loc[m,'ci_low']:+.5f}, {p.loc[m,'ci_high']:+.5f}]" for m in ['spearman','rmse']]
        lines.append('| '+label+' | '+' | '.join(strings)+' |')
    lines+=['','## Does panel-focused BulkFormer help within each prediction head?','',
        '| Head | Pooled Spearman change | Pooled RMSE reduction | Lineages with higher Spearman | Lineages with lower RMSE |',
        '|---|---:|---:|---:|---:|']
    for family in run.FAMILIES:
        a=family+'__'+REPS[1];b=family+'__'+REPS[0]
        p=paired[(paired.scope=='pooled_745')&(paired.first_model==a)&(paired.reference_model==b)].set_index('metric')
        w=wins[(wins.first_model==a)&(wins.reference_model==b)].set_index('metric')
        lines.append(f"| {FAMILY_LABELS[family]} | {p.loc['spearman','improvement']:+.5f} | {p.loc['rmse','improvement']:+.5f} | {int(w.loc['spearman','favorable_lineages'])}/19 | {int(w.loc['rmse','favorable_lineages'])}/19 |")
    lines+=['','## Fixed search and validation','',
        '- Random forest: 128 trees; leaf sizes 5 or 15; per-split feature sampling sqrt(p) or 30% of features. Four candidates per outer fold; fixed forest seed.',
        '- Neural network: one tanh hidden layer with 16 or 32 units; L2 alpha 10 or 100. Four candidates; each prediction averages three fixed initializations. L-BFGS, max_iter 2000, max_fun 40000, tolerance 0.001. No random early-stopping validation split.',
        '- Neural-network X and target scaling are fitted on each current training subset only. Predictions are returned to the original negative-AUC scale before tuning and scoring.',
        '- Both new heads select the minimum equal-weight mean inner-fold RMSE. Ridge predictions and tuning losses are reused from the exact matching original-seed corrected run.',
        '- The nested selector chooses among all nine independently tuned head/representation arms using inner RMSE only. It selected random forest on panel-focused embeddings in 17 holdouts and ridge on panel-focused embeddings in seven, yet performed worse than expression ridge on the outer holdouts. Its choices are saved in nested_selection.tsv.',
        '- Inner validation folds contain lineages also represented in their training folds. The selector result suggests that this tuning design may be a poor guide to unseen-lineage transfer; inner lineage-held-out tuning is a separate, untested alternative.',
        '- All 72 neural outer fits selected alpha=100, the stronger regularization value in the fixed search. The expression networks selected 16 hidden units in all 24 holdouts. This does not establish that the tested regularization range is optimal.',
        '- Related identity groups are kept together in every inner and outer split. Each held-out lineage is absent from supervised fitting and tuning.',
        '- Macro summaries and lineage win counts use the 19 lineages with at least 10 samples. Five smaller lineages remain in pooled scores and detailed tables.',
        '- All candidate validation predictions, losses, selected outer models and optimizer diagnostics are saved. An independent audit recomputes tuning decisions, reconstructs predictions, verifies train-only scaling, and refits the Lung holdout with separate sklearn code.',
        f"- Neural-network member fits with convergence warnings: {convergence['all_member_fits_with_convergence_warning']}/{convergence['all_member_fits']}; selected outer members: {convergence['outer_member_fits_with_convergence_warning']}/{convergence['outer_member_fits']}. No candidate was removed because of an outer-test score.", '',
        '## Interpretation limits','',
        '- This follow-up was designed after the ridge results were known. The search is bounded and is not an exhaustive comparison of all forests or neural networks.',
        '- Bootstrap draws resample complete held-out lineages (24 pooled, 19 macro), with fixed predictions. Models are not retrained within draws, and training sets overlap across holdouts.',
        '- Comparisons and intervals are descriptive; there is no multiplicity-adjusted claim of a best model. The nested selector provides a separate, training-only rule for selecting an arm.',
        '- BulkFormer embeddings see the full transcriptome. Panel-focused pooling is not a 40-gene-only input model. Checkpoint pretraining overlap remains unresolved.',
        '- The same archived CTRP AUC and DepMap 24Q4 data are used; no external drug screen or clinical cohort was added.', '',
        'Audit and reproduction files: protocol.json, source_manifest.json, per_lineage_metrics.tsv, tuning_scores.tsv, predictions.npz, paired_comparisons.tsv, validation.json, and folds/ (selected fitted models, candidate validation predictions, diagnostics and checksums).','']
    (OUT/'REVIEW.md').write_text('\n'.join(lines))


def main():
    completion=json.loads((OUT/'fit_completion.json').read_text());assert completion['status']=='complete'
    protocol=json.loads((OUT/'protocol.json').read_text());ph=run.digest(OUT/'protocol.json')
    assert run.digest(OUT/'run_models.py')==protocol['script_sha256']
    for s in json.loads((OUT/'source_manifest.json').read_text())['files']:assert run.digest(WS/s['path'])==s['sha256']
    run.load_state();state=run.STATE;ids=state['ids'];y=state['y'];groups=state['groups'];lineage=state['lineage']
    rc=module('bootstrap_methods',WS/'outputs/erastin_repeated_cv/repeat_cv.py')
    ridge=np.load(WS/'outputs/erastin_lineage_corrected/per_run_predictions.npz',allow_pickle=False)
    assert np.array_equal(ids,ridge['sample_ids'])
    old_folds=pd.read_csv(WS/'outputs/erastin_lineage_corrected/per_lineage_metrics.tsv',sep='\t')
    old_folds=old_folds[(old_folds.repeat==0)&old_folds.model.isin(REPS)]
    pred={arm:np.full(len(y),np.nan) for arm in ARMS}
    member_pred={rep:np.full((3,len(y)),np.nan) for rep in REPS}
    inner_losses={};fold_rows=[];tuning_rows=[];max_error=0.;max_loss_error=0.;records=[]
    convergence={'all_member_fits':0,'all_member_fits_with_convergence_warning':0,'outer_member_fits':0,'outer_member_fits_with_convergence_warning':0}
    for rep in REPS:
        arm='ridge__'+rep;pred[arm]=ridge[rep][0].copy()
        for r in old_folds[old_folds.model==rep].itertuples():
            te=np.flatnonzero(lineage==r.fold);values=run.metrics(y[te],pred[arm][te])
            inner_losses[(arm,r.fold)]=r.mean_inner_rmse
            fold_rows.append({'arm':arm,'family':'ridge','representation':rep,'lineage':r.fold,'test_n':len(te),
                'mean_inner_rmse':r.mean_inner_rmse,**values})
    paths=sorted((OUT/'folds').glob('*.json'));assert len(paths)==144
    references=[]
    for path in paths:
        meta=json.loads(path.read_text());assert meta['protocol_sha256']==ph
        model_path=path.with_suffix('.joblib');npz_path=path.with_suffix('.npz')
        assert run.digest(model_path)==meta['model_sha256'] and run.digest(npz_path)==meta['predictions_sha256']
        saved=joblib.load(model_path);p=np.load(npz_path,allow_pickle=False)
        family=meta['family'];rep=meta['representation'];arm=meta['arm'];label=meta['lineage'];i=meta['holdout_index']
        tr=p['train_rows'];te=p['test_rows'];inner=state['inner'][i]
        assert np.array_equal(tr,np.flatnonzero(inner>=0)) and np.array_equal(te,np.flatnonzero(lineage==label))
        assert set(groups[tr]).isdisjoint(groups[te]) and label not in set(lineage[tr])
        recomputed=[]
        for candidate,r in enumerate(meta['tuning']):
            val=p['inner_validation_predictions'][candidate];assert np.isfinite(val[tr]).all() and np.isnan(val[te]).all()
            losses=[]
            for k in range(5):
                a=tr[inner[tr]!=k];b=tr[inner[tr]==k]
                assert set(groups[a]).isdisjoint(groups[b]) and set(a)|set(b)==set(tr)
                losses.append(float(np.sqrt(np.mean((y[b]-val[b])**2))))
            error=max(abs(np.array(losses)-np.array(r['fold_rmse'])))
            error=max(error,abs(np.mean(losses)-r['mean_inner_rmse']));max_loss_error=max(max_loss_error,float(error))
            recomputed.append(float(np.mean(losses)))
            tuning_rows.append({'arm':arm,'lineage':label,'candidate':candidate,'parameters':json.dumps(r['parameters'],sort_keys=True),'mean_inner_rmse':np.mean(losses)})
        assert np.argmin(recomputed)==meta['selected_candidate'] and max_loss_error<1e-12
        inner_losses[(arm,label)]=recomputed[meta['selected_candidate']]
        x=state['features'][rep]
        if family=='random_forest':
            # Reconstruct predictions directly from the saved trees' leaf means.
            values=[tree.tree_.value[tree.apply(x[te].astype(np.float32)),0,0] for tree in saved['model'].estimators_]
            reconstructed=np.mean(values,axis=0)
        else:
            sx=saved['x_scaler'];sy=saved['y_scaler']
            expected_sx=StandardScaler().fit(x[tr]);expected_sy=StandardScaler().fit(y[tr,None])
            assert np.allclose(sx.mean_,expected_sx.mean_,rtol=0,atol=1e-12) and np.allclose(sx.scale_,expected_sx.scale_,rtol=0,atol=1e-12)
            assert np.array_equal(sy.mean_,expected_sy.mean_) and np.array_equal(sy.scale_,expected_sy.scale_)
            members=[]
            for network in saved['members']:
                z=(x[te]-sx.mean_)/sx.scale_
                for layer,(coef,bias) in enumerate(zip(network.coefs_,network.intercepts_)):
                    z=z@coef+bias
                    if layer<len(network.coefs_)-1:z=np.tanh(z)
                members.append(z.ravel()*sy.scale_[0]+sy.mean_[0])
            reconstructed=np.mean(members,axis=0)
            assert np.allclose(members,p['member_test_predictions'],rtol=0,atol=1e-12)
            member_pred[rep][:,te]=p['member_test_predictions']
            for diag in meta['fit_diagnostics']:
                for member in diag['members']:
                    bad=any(w['category']=='ConvergenceWarning' for w in member['warnings'])
                    convergence['all_member_fits']+=1;convergence['all_member_fits_with_convergence_warning']+=int(bad)
                    if diag.get('stage')=='outer_refit':
                        convergence['outer_member_fits']+=1;convergence['outer_member_fits_with_convergence_warning']+=int(bad)
        max_error=max(max_error,float(np.max(np.abs(reconstructed-p['test_prediction']))));assert max_error<1e-10
        assert np.isnan(pred[arm][te]).all();pred[arm][te]=p['test_prediction']
        values=run.metrics(y[te],p['test_prediction'])
        fold_rows.append({'arm':arm,'family':family,'representation':rep,'lineage':label,'test_n':len(te),
            'mean_inner_rmse':meta['mean_inner_rmse'],**values})
        records.append(meta)
        if label=='Lung':references.append((family,rep,meta['selected_parameters'],x,y,tr,te,p['test_prediction'],protocol))
    assert all(np.isfinite(v).all() for v in pred.values())
    assert convergence['all_member_fits']==4536 and convergence['outer_member_fits']==216
    with futures.ThreadPoolExecutor(max_workers=3) as pool:
        refs=list(pool.map(lambda args:reference_fit(*args),references))
    assert len(refs)==6
    # This selector's choices are made entirely from the retained inner losses.
    pred['nested_selector']=np.full(len(y),np.nan);selection=[]
    for label in state['labels']:
        selected=min(ARMS,key=lambda a:inner_losses[(a,label)])
        te=np.flatnonzero(lineage==label);pred['nested_selector'][te]=pred[selected][te]
        selection.append({'lineage':label,'test_n':len(te),'selected_arm':selected,
            'mean_inner_rmse':inner_losses[(selected,label)]})
        fold_rows.append({'arm':'nested_selector','family':'nested_selector','representation':'selected_by_inner_cv',
            'lineage':label,'test_n':len(te),'mean_inner_rmse':inner_losses[(selected,label)],**run.metrics(y[te],pred[selected][te])})
    folds=pd.DataFrame(fold_rows);table(OUT/'per_lineage_metrics.tsv',folds);table(OUT/'tuning_scores.tsv',pd.DataFrame(tuning_rows))
    table(OUT/'nested_selection.tsv',pd.DataFrame(selection))
    np.savez_compressed(OUT/'predictions.npz',sample_ids=ids,group_ids=groups,lineage=lineage,observed_sensitivity=y,**pred)
    np.savez_compressed(OUT/'neural_member_predictions.npz',sample_ids=ids,**member_pred)
    member_scores=[{'representation':rep,'seed':seed,'note':'member of inner-tuned ensemble; not independently tuned',**run.metrics(y,p)}
        for rep,a in member_pred.items() for seed,p in zip(run.SEEDS,a)]
    table(OUT/'neural_member_metrics.tsv',pd.DataFrame(member_scores))
    eligible=folds[folds.test_n>=10];assert eligible.groupby('arm').size().eq(19).all()
    macro=eligible.groupby('arm')[METRICS].mean()
    assert np.isfinite(macro).all().all()
    rc.validate_weighted_bootstrap(y,np.array([pred[ARMS[0]],pred[ARMS[-1]]]),lineage,run.metrics)
    draws=rc.repeated_group_bootstrap(y,{a:p[None,:] for a,p in pred.items()},lineage,count=5000,seed=2026110401)
    labels=sorted(eligible.lineage.unique());picks=np.random.default_rng(2026110402).integers(19,size=(5000,19))
    macro_draws={}
    for arm in pred:
        values=eligible[eligible.arm==arm].set_index('lineage').loc[labels]
        macro_draws[arm]={metric:values[metric].to_numpy()[picks].mean(axis=1) for metric in METRICS}
    summary=[]
    for arm,p in pred.items():
        row={'arm':arm,'family':arm.split('__')[0],'representation':arm.split('__')[1] if '__' in arm else 'selected_by_inner_cv',**run.metrics(y,p)}
        for metric in METRICS:
            row['macro_'+metric]=macro.loc[arm,metric]
            row[metric+'_ci_low'],row[metric+'_ci_high']=np.percentile(draws[arm][metric],[2.5,97.5])
            row['macro_'+metric+'_ci_low'],row['macro_'+metric+'_ci_high']=np.percentile(macro_draws[arm][metric],[2.5,97.5])
        summary.append(row)
    summary=pd.DataFrame(summary);table(OUT/'summary.tsv',summary)
    contrasts=[(a,'ridge__panel_expression') for a in pred if a!='ridge__panel_expression']
    contrasts += [(f+'__'+REPS[1],f+'__'+REPS[0]) for f in run.FAMILIES]
    contrasts += [(f+'__'+REPS[2],f+'__'+REPS[0]) for f in run.FAMILIES]
    contrasts += [(f+'__'+r,'ridge__'+r) for f in ['random_forest','neural_network'] for r in REPS]
    contrasts=list(dict.fromkeys(contrasts));paired=[];wins=[]
    points=summary.set_index('arm')
    for a,b in contrasts:
        for metric in METRICS:
            sign=-1 if metric=='rmse' else 1
            va=eligible[eligible.arm==a].set_index('lineage').loc[labels,metric]
            vb=eligible[eligible.arm==b].set_index('lineage').loc[labels,metric]
            wins.append({'first_model':a,'reference_model':b,'metric':metric,
                'favorable_lineages':int((sign*(va-vb)>0).sum()),'eligible_lineages':19})
            for scope,bootstrap,prefix in [('pooled_745',draws,''),('macro_19_lineages',macro_draws,'macro_')]:
                delta=sign*(bootstrap[a][metric]-bootstrap[b][metric]);lo,hi=np.percentile(delta,[2.5,97.5])
                paired.append({'scope':scope,'first_model':a,'reference_model':b,'metric':metric,
                    'improvement':sign*(points.loc[a,prefix+metric]-points.loc[b,prefix+metric]),'ci_low':lo,'ci_high':hi})
    paired=pd.DataFrame(paired);wins=pd.DataFrame(wins);table(OUT/'paired_comparisons.tsv',paired);table(OUT/'lineage_win_counts.tsv',wins)
    np.savez_compressed(OUT/'bootstrap_draws.npz',**{scope+'__'+arm+'__'+m:v for scope,d in [('pooled',draws),('macro',macro_draws)] for arm,a in d.items() for m,v in a.items()})
    validation={'passed':True,'all_source_and_frozen_script_hashes_verified':True,'new_outer_models_audited':144,
        'candidate_inner_cv_prediction_sets_audited':576,'group_and_lineage_exclusion_verified':True,
        'train_only_neural_feature_and_target_scaling_verified':True,'max_recomputed_inner_loss_error':max_loss_error,
        'max_independent_model_reconstruction_error':max_error,'independent_lung_refits':refs,
        'nested_selector_uses_only_inner_validation_losses':True,'neural_convergence':convergence}
    run.dump(OUT/'validation.json',validation)
    table(OUT/'selected_parameters.tsv',pd.DataFrame([{'arm':r['arm'],'lineage':r['lineage'],'selected_candidate':r['selected_candidate'],
        'parameters':json.dumps(r['selected_parameters'],sort_keys=True),'mean_inner_rmse':r['mean_inner_rmse']} for r in records]))
    render(summary,paired,wins,pd.DataFrame(selection),convergence)
    run.dump(OUT/'final_summary.json',{'status':'complete','samples':745,'heldout_lineages':24,'fixed_inner_split_seed':20261031,
        'metrics':summary.to_dict('records'),'neural_convergence':convergence,'nested_selection_counts':pd.Series([r['selected_arm'] for r in selection]).value_counts().to_dict()})
    files=[{'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':run.digest(p)} for p in sorted(OUT.rglob('*')) if p.is_file() and p.name not in ['artifact_manifest.json','progress.json']]
    run.dump(OUT/'artifact_manifest.json',{'files':files})
    archive=OUT.parent/(OUT.name+'_results.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.name!='progress.json':z.write(p,arcname=str(Path(OUT.name)/p.relative_to(OUT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for item in files:
            import hashlib
            assert hashlib.sha256(z.read(str(Path(OUT.name)/item['path']))).hexdigest()==item['sha256']
    run.dump(OUT/'progress.json',{'phase':'complete','completed_outer_jobs':144,'total_outer_jobs':144})
    print(json.dumps(run.clean({'status':'complete','validation':validation,'metrics':summary.to_dict('records')})),flush=True)


if __name__=='__main__':main()
