#!/usr/bin/env python3
"""Audit moderate-weight fits as they finish, then perform paired evaluation."""
import json
from pathlib import Path
import sys
import time
import zipfile
from collections import Counter
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import run_moderate as run
import evaluation as ev
from lineage_weights import SCHEMES,weights_for,weight_audit

OUT=run.OUT;WS=run.WS;METRICS=ev.METRICS
ARMS=['unweighted','inverse_sqrt','capped_inverse_5','inverse_frequency']
LABELS={'unweighted':'Unweighted','inverse_sqrt':'Inverse square root','capped_inverse_5':'Inverse frequency, cap 5','inverse_frequency':'Full inverse frequency'}

def table(name,rows):
    a=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows);a.to_csv(OUT/name,sep='\t',index=False,float_format='%.12g');return a

def wait_for(path):
    while not path.exists():time.sleep(.5)

def record_weights(rows,groups,context,audit):
    rows.append({**context,**{k:v for k,v in audit.items() if k!='groups'}});groups.extend(dict(**context,**v) for v in audit['groups'])

def audit():
    run.initialize();s=run.STATE;x=s['x'];y=s['y'];lineage=s['lineage'];g=s['groups'];ph=run.sha(OUT/'protocol.json');protocol=json.loads((OUT/'protocol.json').read_text())
    for name,h in protocol['scripts'].items():assert run.sha(OUT/name)==h
    for a in json.loads((OUT/'source_manifest.json').read_text())['files']:assert run.sha(WS/a['path'])==a['sha256']
    run.dump(OUT/'analysis_source_manifest.json',dict(script_sha256=run.sha(Path(__file__)),report_sha256=run.sha(OUT/'report_moderate.py')))
    np.testing.assert_array_equal(np.load(OUT/'shared_splits.npz')['inner_fold_by_holdout'],np.load(WS/'outputs/erastin_nn_repeated/shared_splits.npz')['inner_fold_by_holdout'])
    base=np.load(WS/'outputs/erastin_panel_comparison/predictions.npz');full=np.load(WS/'outputs/erastin_lineage_weighted/predictions.npz')
    for a in [base,full]:
        for k,v in [('sample_ids',s['ids']),('group_ids',g),('lineage',lineage),('y',y),('inner_split_seeds',s['seeds'])]:np.testing.assert_array_equal(a[k],v)
    preds={scheme:np.full((20,len(y)),np.nan) for scheme in SCHEMES};preds.update(unweighted=base['neural_network__64'].copy(),inverse_frequency=full['weighted__neural_network__64'].copy())
    parameters=[];tuning=[];warnings=[];wrows=[];wgroups=[];references=[];objectives=[];max_pred=max_rmse=max_objective=0.;scores_checked=groups_checked=weights_checked=0
    for r in range(20):
        for scheme in SCHEMES:
            seen=np.zeros(len(y),int)
            for l,label in enumerate(s['labels']):
                root=OUT/'folds'/f'{scheme}__r{r+1:02d}__l{l:02d}';wait_for(root.with_suffix('.json'))
                m=json.loads(root.with_suffix('.json').read_text());a=np.load(root.with_suffix('.npz'));inner=s['inner'][r,l];tr=a['train_rows'];te=a['test_rows']
                np.testing.assert_array_equal(tr,np.flatnonzero(inner>=0));np.testing.assert_array_equal(te,np.flatnonzero(lineage==label))
                assert set(g[tr]).isdisjoint(g[te]) and label not in set(lineage[tr]);groups_checked+=1;seen[te]+=1
                assert m['scheme']==scheme and m['repeat']==r+1 and m['lineage']==label and m['lineage_index']==l and m['panel']==64 and m['protocol_sha256']==ph
                assert m['inner_split_seed']==int(s['seeds'][r]);np.testing.assert_array_equal(m['neural_seeds'],s['nn_seeds'][r])
                for ext in ['joblib','npz']:assert run.sha(root.with_suffix('.'+ext))==m[ext+'_sha256']
                c=m['selected_candidate'];params=s['grid'][c]
                assert params==m['selected_parameters'] and c==np.argmin(a['inner_fold_rmse'].mean(axis=1))
                assert np.isfinite(a['inner_validation_predictions'][:,tr]).all() and np.isnan(a['inner_validation_predictions'][:,inner<0]).all()
                w=weights_for(lineage[tr],scheme);np.testing.assert_array_equal(w,a['outer_training_weights']);assert m['outer_weights']==weight_audit(lineage[tr],scheme);weights_checked+=1
                context=dict(scheme=scheme,repeat=r+1,holdout_lineage=str(label),stage='outer',inner_fold=-1)
                record_weights(wrows,wgroups,context,m['outer_weights'])
                for k in range(5):
                    aa=tr[inner[tr]!=k];bb=tr[inner[tr]==k];assert set(aa)|set(bb)==set(tr) and set(g[aa]).isdisjoint(g[bb]);groups_checked+=1
                    iw=weights_for(lineage[aa],scheme);np.testing.assert_array_equal(iw,a['inner_training_weights'][k,aa]);assert np.isnan(a['inner_training_weights'][k,np.setdiff1d(np.arange(len(y)),aa)]).all()
                    assert m['inner_scaling'][k]['weights']==weight_audit(lineage[aa],scheme);weights_checked+=1
                    assert m['inner_scaling'][k]['y_mean']==y[aa].mean() and m['inner_scaling'][k]['y_scale']==y[aa].std()
                    record_weights(wrows,wgroups,dict(**{q:v for q,v in context.items() if q not in ['stage','inner_fold']},stage='inner',inner_fold=k),m['inner_scaling'][k]['weights'])
                    loss=np.sqrt(np.mean((a['inner_validation_predictions'][:,bb]-y[bb][None,:])**2,axis=1))
                    max_rmse=max(max_rmse,float(np.max(np.abs(loss-a['inner_fold_rmse'][:,k]))));scores_checked+=len(s['grid'])
                model=joblib.load(root.with_suffix('.joblib'));sx=StandardScaler().fit(x[tr]);sy=StandardScaler().fit(y[tr,None]);z=sx.transform(x[tr]);target=sy.transform(y[tr,None]).ravel()
                for key,expected in [('mean_',sx.mean_),('scale_',sx.scale_)]:np.testing.assert_allclose(getattr(model['x_scaler'],key),expected,rtol=0,atol=1e-12)
                for key,expected in [('mean_',sy.mean_),('scale_',sy.scale_)]:np.testing.assert_allclose(getattr(model['y_scaler'],key),expected,rtol=0,atol=1e-12)
                assert [v.random_state for v in model['members']]==m['neural_seeds']
                members=np.array([sy.inverse_transform(v.predict(sx.transform(x[te]))[:,None]).ravel() for v in model['members']])
                np.testing.assert_allclose(members,a['member_test_predictions'],rtol=0,atol=1e-12);p=members.mean(axis=0)
                max_pred=max(max_pred,float(np.max(np.abs(p-a['test_prediction']))));preds[scheme][r,te]=p
                for j,v in enumerate(model['members']):
                    np.testing.assert_array_equal(v.training_sample_weight_,w)
                    assert v.hidden_layer_sizes==(params['width'],) and v.alpha==params['alpha'] and v.activation=='tanh' and v.solver=='lbfgs'
                    value=.5*np.sum(w*(v.predict(z)-target)**2)/len(tr)+.5*v.alpha*sum(np.sum(t*t) for t in v.coefs_)/len(tr)
                    error=float(abs(value-v.loss_));max_objective=max(max_objective,error);assert error<1e-10
                    objectives.append(dict(scheme=scheme,repeat=r+1,lineage=str(label),member=j,loss_recalculation_error=error,weighted_training_loss=value,iterations=v.n_iter_))
                parameters.append(dict(model=scheme,scheme=scheme,repeat=r+1,lineage=str(label),candidate=c,**params,mean_inner_rmse=m['mean_inner_rmse'],train_n=len(tr),test_n=len(te)))
                tuning.extend(dict(model=scheme,repeat=r+1,lineage=str(label),candidate=ci,**par,mean_inner_rmse=float(a['inner_fold_rmse'][ci].mean())) for ci,par in enumerate(s['grid']))
                for d in m['fit_diagnostics']:
                    for member in d['members']:
                        warnings.extend(dict(scheme=scheme,repeat=r+1,lineage=str(label),stage=d['stage'],**v) for v in member['warnings'])
                if r in [0,19] and label in ['Lung','Thyroid']:
                    ref,_,_,_=run.fitted_neural(params,x,y,tr,te,s['nn_seeds'][r],w);error=float(np.max(np.abs(ref-p)));inner_error=0.
                    for k in range(5):
                        aa=tr[inner[tr]!=k];bb=tr[inner[tr]==k]
                        ref,_,_,_=run.fitted_neural(params,x,y,aa,bb,s['nn_seeds'][r],weights_for(lineage[aa],scheme))
                        inner_error=max(inner_error,float(np.max(np.abs(ref-a['inner_validation_predictions'][c,bb]))))
                    assert max(error,inner_error)<1e-10;references.append(dict(scheme=scheme,repeat=r+1,lineage=str(label),outer_prediction_error=error,inner_prediction_error=inner_error,inner_refits=5))
                run.dump(OUT/'audit_progress.json',dict(audited_models=len(parameters),total=960,last_repeat=r+1))
            assert np.all(seen==1)
        print(json.dumps(dict(event='audit_progress',complete_repeats=r+1)),flush=True)
    wait_for(OUT/'fit_completion.json');assert json.loads((OUT/'fit_completion.json').read_text())['complete']
    assert groups_checked==weights_checked==5760 and scores_checked==19200 and len(references)==8 and max_pred<1e-10 and max_rmse<1e-12
    assert all(np.isfinite(v).all() for v in preds.values())
    # Reference schemes use the same training memberships for descriptive weight audits.
    for scheme in ['unweighted','inverse_frequency']:
        for r in range(20):
            for l,label in enumerate(s['labels']):
                inner=s['inner'][r,l];tr=np.flatnonzero(inner>=0)
                record_weights(wrows,wgroups,dict(scheme=scheme,repeat=r+1,holdout_lineage=str(label),stage='outer',inner_fold=-1),weight_audit(lineage[tr],scheme))
    table('training_subset_weight_audit.tsv',wrows);table('training_lineage_weights.tsv',wgroups);table('model_training_objective_checks.tsv',objectives)
    params=table('selected_parameters.tsv',parameters);table('tuning_scores.tsv',tuning)
    warnings=Counter((v['category'],v['message']) for v in warnings);warning_summary=[dict(category=k[0],message=k[1],count=n) for k,n in warnings.items()]
    bootstrap_check=ev.validate_weighted_bootstrap(y,preds['inverse_sqrt'],lineage,ev.metrics)
    validation=dict(passed=True,new_outer_models=960,reused_reference_arms=2,group_exclusion_checks=groups_checked,training_weight_vector_checks=weights_checked,
        reconstructed_candidate_fold_scores=scores_checked,prediction_max_error=max_pred,inner_rmse_max_error=max_rmse,
        direct_weighted_objectives_checked=len(objectives),weighted_objective_max_error=max_objective,selected_setting_replays=references,
        warning_summary=warning_summary,warning_count=sum(warnings.values()),convergence_warning_count=sum(v['count'] for v in warning_summary if v['category']=='ConvergenceWarning'),
        preflight=json.loads((OUT/'preflight_validation.json').read_text()),bootstrap_validation=bootstrap_check,
        exact_saved_splits_verified=True,training_only_unweighted_scaling_verified=True,source_hashes_verified=True,
        final_capped_weights_bounded_by_5=True,weight_normalization_mean_one_verified=True,
        replay_note='Selected weighted NN refits use the same verified adapter; independent formula, finite-difference gradient and objective checks are separate.')
    run.dump(OUT/'validation.json',validation);return {v:preds[v] for v in ARMS},params,validation,protocol

def evaluate(preds,params,validation,protocol):
    s=run.STATE;y=s['y'];lineage=s['lineage'];repeats=list(range(1,21));scores,folds,macro=ev.score_predictions(y,lineage,preds,repeats)
    original=pd.read_csv(WS/'outputs/erastin_panel_comparison/per_repeat_metrics.tsv',sep='\t')
    np.testing.assert_allclose(scores[scores.model=='unweighted'].sort_values('repeat')[METRICS],original[original.model=='neural_network__64'].sort_values('repeat')[METRICS],rtol=0,atol=1e-12)
    table('per_repeat_metrics.tsv',scores);table('per_lineage_metrics.tsv',folds);table('per_repeat_macro_metrics.tsv',macro)
    rows=[]
    for scope,frame in [('pooled_745',scores),('macro_19',macro)]:
        for arm in ARMS:
            row=dict(scope=scope,model=arm)
            for metric in METRICS:
                v=frame[frame.model==arm][metric].to_numpy();assert np.isfinite(v).all()
                row.update({metric+'_mean':v.mean(),metric+'_sd':v.std(ddof=1),metric+'_min':v.min(),metric+'_max':v.max()})
            rows.append(row)
    summary=table('summary.tsv',rows);run.dump(OUT/'progress.json',dict(phase='paired_lineage_bootstrap',new_models_audited=960))
    draws=ev.repeated_group_bootstrap(y,preds,lineage,count=5000,seed=protocol['uncertainty']['pooled_seed'])
    mdraws=ev.bootstrap_macro(folds,ARMS,repeats,protocol['uncertainty']['macro_seed'])
    pairs=protocol['primary_comparisons']+protocol['secondary_comparisons'];paired,seedrows=ev.paired_comparisons(scores,macro,draws,mdraws,pairs,repeats)
    paired=table('paired_comparisons.tsv',paired);table('paired_seed_differences.tsv',seedrows);effects=[];wins=[]
    for first,reference in pairs:
        for heldout in s['labels']:
            a=folds[(folds.model==first)&(folds.lineage==heldout)].sort_values('repeat');b=folds[(folds.model==reference)&(folds.lineage==heldout)].sort_values('repeat')
            row=dict(first_model=first,reference_model=reference,lineage=str(heldout),n=int(a.test_n.iloc[0]))
            for metric in METRICS:
                delta=(-1 if metric=='rmse' else 1)*(a[metric].to_numpy()-b[metric].to_numpy())
                row[metric+'_improvement']=float(delta.mean());row[metric+'_favorable_repeats']=int(np.sum(delta>0))
            effects.append(row)
        eligible=[v for v in effects if v['first_model']==first and v['reference_model']==reference and v['n']>=10]
        wins.append(dict(first_model=first,reference_model=reference,eligible_lineages=19,**{metric+'_favorable_lineages':sum(v[metric+'_improvement']>0 for v in eligible) for metric in METRICS}))
    effects=table('lineage_paired_effects.tsv',effects);wins=table('lineage_win_counts.tsv',wins)
    counts=table('parameter_counts.tsv',params.groupby(['model','width','alpha']).size().reset_index(name='outer_fits'))
    np.savez_compressed(OUT/'predictions.npz',sample_ids=s['ids'],group_ids=s['groups'],lineage=lineage,y=y,inner_split_seeds=s['seeds'],**preds)
    np.savez_compressed(OUT/'bootstrap_draws.npz',**{scope+'__'+arm+'__'+metric:v for scope,d in [('pooled',draws),('macro',mdraws)] for arm,values in d.items() for metric,v in values.items()})
    from report_moderate import render
    render(summary,paired,effects,wins,counts,validation)
    for a in json.loads((OUT/'source_manifest.json').read_text())['files']:assert run.sha(WS/a['path'])==a['sha256']
    a=json.loads((OUT/'analysis_source_manifest.json').read_text());assert run.sha(Path(__file__))==a['script_sha256'] and run.sha(OUT/'report_moderate.py')==a['report_sha256']
    final=dict(validation_passed=True,summary=summary.to_dict('records'),lineage_wins=wins.to_dict('records'),new_outer_models=960,warning_count=validation['warning_count'])
    run.dump(OUT/'final_summary.json',final);run.dump(OUT/'progress.json',dict(phase='complete',new_models_audited=960,validation_passed=True))
    archive=seal();print(json.dumps(dict(event='complete',archive_bytes=archive.stat().st_size,summary=run.clean(final)),indent=2),flush=True)

def seal():
    manifest=dict(files=[dict(path=str(p.relative_to(OUT)),sha256=run.sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts]);run.dump(OUT/'artifact_manifest.json',manifest)
    archive=OUT.parent/'erastin_nn_moderate_weights_results.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for row in manifest['files']:z.write(OUT/row['path'],'erastin_nn_moderate_weights/'+row['path'])
        z.write(OUT/'artifact_manifest.json','erastin_nn_moderate_weights/artifact_manifest.json')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in manifest['files']:assert run.hashlib.sha256(z.read('erastin_nn_moderate_weights/'+row['path'])).hexdigest()==row['sha256']
    return archive

if __name__=='__main__':
    predictions,parameters,validation,protocol=audit();evaluate(predictions,parameters,validation,protocol)
