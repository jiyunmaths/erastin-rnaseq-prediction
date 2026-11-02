#!/usr/bin/env python3
"""Independent audit and paired repeated-seed review of expression-only models."""
import concurrent.futures as futures
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
WS=OUT.parent.parent


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


run=module('repeat_methods',OUT/'run_repeats.py')
METRICS=['spearman','rmse','r2']
NAMES=['ridge','neural_network']


def table(name,data):
    data.to_csv(OUT/name,sep='\t',index=False,float_format='%.17g')


def independent_refit(x,y,tr,te,params,seeds,expected_nn,alpha,expected_ridge,label,repeat):
    member_predictions=[]
    for seed in seeds:
        nn=MLPRegressor(hidden_layer_sizes=(params['width'],),alpha=params['alpha'],activation='tanh',solver='lbfgs',
            max_iter=2000,max_fun=40000,tol=1e-3,early_stopping=False,random_state=int(seed))
        model=TransformedTargetRegressor(regressor=make_pipeline(StandardScaler(),nn),transformer=StandardScaler())
        model.fit(x[tr],y[tr]);member_predictions.append(model.predict(x[te]))
    nn_error=float(np.max(np.abs(np.mean(member_predictions,axis=0)-expected_nn)))
    baseline=make_pipeline(StandardScaler(),Ridge(alpha=alpha,solver='svd'))
    baseline.fit(x[tr],y[tr]);ridge_error=float(np.max(np.abs(baseline.predict(x[te])-expected_ridge)))
    assert nn_error<1e-8 and ridge_error<1e-10
    return {'repeat':repeat,'lineage':label,'neural_max_abs_error':nn_error,'ridge_max_abs_error':ridge_error}


def render(summary,paired,differences,lineage_effects,wins,convergence,parameters):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    fig,axes=plt.subplots(1,2,figsize=(10,4.8),constrained_layout=True)
    for ax,metric,title in zip(axes,['spearman','rmse'],['Spearman increase','RMSE reduction']):
        for scope,label,color in [('pooled_745','Pooled cells','#136f77'),('macro_19_lineages','Equal-weight lineage mean','#97518e')]:
            s=differences[(differences.scope==scope)&(differences.metric==metric)].sort_values('repeat')
            ax.scatter(s.improvement,s.repeat,label=label,color=color,s=26)
        ax.axvline(0,color='#5b5b5b',ls='--',lw=1)
        ax.set(xlabel=title+' (positive favors neural network)',ylabel='Repeat',yticks=np.arange(1,21,2))
        ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='outside lower center',ncol=2)
    fig.suptitle('40-gene neural network versus ridge: 20 paired repeats\nSame 24 lineage holdouts; inner folds and network initializations vary',fontsize=12)
    fig.savefig(OUT/'paired_seed_effects.png',dpi=170);fig.savefig(OUT/'paired_seed_effects.svg');plt.close(fig)
    s=lineage_effects[lineage_effects.test_n>=10].sort_values(['test_n','lineage'],ascending=[False,True]);pos=np.arange(len(s))
    fig,axes=plt.subplots(1,2,figsize=(11,8),constrained_layout=True,sharey=True)
    for ax,metric,title in zip(axes,['spearman','rmse'],['Spearman increase','RMSE reduction']):
        v=s[metric+'_mean_improvement'].to_numpy()
        ax.barh(pos,v,color=np.where(v>0,'#136f77','#bd6544'));ax.axvline(0,color='#444',lw=1)
        ax.set_xlabel(title+' (positive favors neural network)');ax.grid(axis='x',alpha=.2)
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(pos,[f'{r.lineage} (n={r.test_n})' for r in s.itertuples()]);axes[0].invert_yaxis()
    fig.suptitle('Mean neural-network improvement over ridge across 20 repeats\n19 held-out lineages with at least 10 cell lines',fontsize=12)
    fig.savefig(OUT/'lineage_effects.png',dpi=160);fig.savefig(OUT/'lineage_effects.svg');plt.close(fig)
    indexed=summary.set_index(['scope','model','metric'])
    primary=paired[paired.scope=='pooled_745'].set_index('metric')
    rho=primary.loc['spearman'];rmse=primary.loc['rmse']
    overlap=(rho.ci_low<=0<=rho.ci_high) and (rmse.ci_low<=0<=rmse.ci_high)
    uncertainty='Both paired pooled improvement intervals include zero. The error reduction was consistent across these seeds, but a clear advantage across lineages is not established.' if overlap else 'The paired intervals describe conditional uncertainty across held-out lineages.'
    lines=['# Repeated neural-network versus ridge comparison','',
        f"The neural network improved pooled Spearman in {int(rho.favorable_repeats)}/20 repeats and reduced pooled RMSE in {int(rmse.favorable_repeats)}/20. The mean improvements were {rho.mean_improvement:+.5f} in Spearman and {rmse.mean_improvement:+.6f} in RMSE reduction. Paired uncertainty and lineage-level behavior are reported below.",'',
        'Both models use the same 40-gene expression panel, 745 cancer cell lines, 734 identity groups, and 24 fixed leave-lineage-out tests. Each repeat pairs neural-network predictions with ridge predictions from the exact same saved inner-fold assignment.','',
        'Twenty inner-fold seeds and twenty fresh, fixed neural initialization trios were registered before fitting. Each neural prediction averages three members. Inner splits and initializations change jointly, so this study assesses their combined sensitivity and does not separate their contributions. The original single-split result is excluded from the repeat averages.','',
        '## Mean held-out performance','',
        'Means of per-repeat metrics; SD describes variation across repeats. These are not scores of an averaged-prediction ensemble.','',
        '| Scope | Model | Spearman mean ± SD ↑ | RMSE mean ± SD ↓ | R² mean ± SD ↑ |',
        '|---|---|---:|---:|---:|']
    for scope,label in [('pooled_745','Pooled, 745 cells'),('macro_19_lineages','Equal-weight mean, 19 lineages')]:
        for name,display in [('ridge','Ridge'),('neural_network','Neural network')]:
            cells=[f"{indexed.loc[(scope,name,m),'mean']:.5f} ± {indexed.loc[(scope,name,m),'sd_across_repeats']:.5f}" for m in METRICS]
            lines.append('| '+label+' | '+display+' | '+' | '.join(cells)+' |')
    lines+=['','## Paired neural-network improvement','',
        'Positive means better than the matched ridge model. Intervals use 5,000 paired resamples of whole held-out lineages.','',
        '| Scope | Metric | Mean improvement | Favorable repeats | Conditional 95% interval |',
        '|---|---|---:|---:|---:|']
    for r in paired.itertuples():
        scope='Pooled cells' if r.scope=='pooled_745' else 'Equal-weight lineages'
        desc={'spearman':'Spearman increase','rmse':'RMSE reduction','r2':'R² increase'}[r.metric]
        lines.append(f'| {scope} | {desc} | {r.mean_improvement:+.6f} | {r.favorable_repeats}/20 | [{r.ci_low:+.6f}, {r.ci_high:+.6f}] |')
    base=indexed.loc[('pooled_745','ridge','rmse'),'mean'];pct=100*rmse.mean_improvement/base
    lines+=['',f'The mean pooled RMSE reduction relative to ridge is {pct:.2f}%. '+uncertainty,'',
        f'![Paired repeat effects]({OUT / "paired_seed_effects.png"})','',
        '## Behavior by held-out lineage','',
        'Using mean effects across all 20 repeats, among the 19 lineages with n ≥ 10:','']
    for r in wins.itertuples():
        wording={'spearman':'higher Spearman','rmse':'lower RMSE','r2':'higher R²'}[r.metric]
        lines.append(f'- Neural network has {wording} in {r.favorable_lineages}/19 lineages.')
    lines+=['',f'![Lineage effects]({OUT / "lineage_effects.png"})','',
        '## Protocol and audit','',
        '- The neural grid is unchanged: one tanh layer with 16 or 32 units, L2 alpha 10 or 100; four candidates. Each candidate is a mean of three fixed initialization seeds.',
        '- X and target scalers are fitted only on each current training subset. L-BFGS uses max_iter=2000, max_fun=40000, tol=0.001, without a random early-stopping validation split.',
        '- Each outer fold retunes candidates by equal-weight mean five-fold inner RMSE. Held-out lineages and related identities never enter supervised training or tuning.',
        '- The unchanged neural pipeline exactly replays the earlier Lung and Thyroid tests, including every candidate loss and selected setting.',
        '- The 480 new selected ensemble models and all 1,920 candidate validation-prediction sets are audited. Independent sklearn pipelines refit both neural network and ridge on the first and last repeats for Lung and Thyroid.',
        f"- Neural member fits with convergence warnings: {convergence['all_member_fits_with_convergence_warning']}/{convergence['all_member_fits']}; selected outer members: {convergence['outer_member_fits_with_convergence_warning']}/{convergence['outer_member_fits']}.",
        '- Selected settings, seed triplets, predictions, inner losses, fitted models, hashes and numerical audits are saved alongside this review.','',
        '## Limits','',
        '- The repeats reuse the same cohort and outer lineage tests. Win counts describe stability, not independent replication; no independent-seed t-test or binomial significance test is used.',
        '- Pooled bootstrap draws resample 24 complete lineages, preserving all model and repeat predictions together. Macro draws resample the 19 lineages with n≥10. Models are not retrained within draws, and training sets overlap across holdouts.',
        '- Five smaller lineages remain in pooled scores and detailed tables but are excluded from macro scores and lineage win counts.',
        '- This follow-up was motivated by the earlier marginal neural-network gain. No external screen or independent cohort was added.',
        '- Inner validation mixes represented training lineages; the same established design was preserved for a matched comparison. This study does not evaluate lineage-held-out inner tuning.',
        '- Only the 40-gene expression panel is used. The archived CTRP AUC and DepMap 24Q4 data limitations remain unchanged.','']
    (OUT/'REVIEW.md').write_text('\n'.join(lines))


def main():
    assert json.loads((OUT/'fit_completion.json').read_text())['status']=='complete'
    protocol=json.loads((OUT/'protocol.json').read_text());ph=run.prior.digest(OUT/'protocol.json')
    assert run.prior.digest(OUT/'run_repeats.py')==protocol['script_sha256']
    for s in json.loads((OUT/'source_manifest.json').read_text())['files']:assert run.prior.digest(WS/s['path'])==s['sha256']
    run.load_state();state=run.STATE;x=state['x'];y=state['y'];ids=state['ids'];lineage=state['lineage'];groups=state['groups']
    shared=np.load(OUT/'shared_splits.npz',allow_pickle=False)
    ridge_source=np.load(WS/'outputs/erastin_lineage_corrected/per_run_predictions.npz',allow_pickle=False)
    assert np.array_equal(ids,shared['sample_ids']) and np.array_equal(ids,ridge_source['sample_ids'])
    assert np.array_equal(shared['inner_split_seeds'],ridge_source['seeds'][1:])
    assert np.array_equal(shared['inner_fold_by_holdout'],state['inner'][1:])
    assert np.array_equal(shared['neural_initialization_seeds'],state['nn_seeds'][1:])
    old_folds=pd.read_csv(WS/'outputs/erastin_lineage_corrected/per_lineage_metrics.tsv',sep='\t')
    old_folds=old_folds[(old_folds.repeat>0)&old_folds.model.eq('panel_expression')]
    pred={'ridge':ridge_source['panel_expression'][1:].copy(),'neural_network':np.full((20,len(y)),np.nan)}
    members=np.full((20,3,len(y)),np.nan);tuning=[];parameters=[];references=[]
    max_error=0.;max_loss_error=0.;group_checks=0
    convergence={'all_member_fits':0,'all_member_fits_with_convergence_warning':0,'outer_member_fits':0,'outer_member_fits_with_convergence_warning':0}
    paths=sorted((OUT/'folds').glob('*.json'));assert len(paths)==480
    for path in paths:
        meta=json.loads(path.read_text());assert meta['protocol_sha256']==ph
        assert run.prior.digest(path.with_suffix('.joblib'))==meta['model_sha256']
        assert run.prior.digest(path.with_suffix('.npz'))==meta['predictions_sha256']
        model=joblib.load(path.with_suffix('.joblib'));p=np.load(path.with_suffix('.npz'),allow_pickle=False)
        repeat=meta['repeat'];r=repeat-1;i=meta['holdout_index'];label=meta['lineage']
        tr=p['train_rows'];te=p['test_rows'];inner=shared['inner_fold_by_holdout'][r,i]
        assert np.array_equal(tr,np.flatnonzero(inner>=0)) and np.array_equal(te,np.flatnonzero(lineage==label))
        assert set(groups[tr]).isdisjoint(groups[te]) and label not in set(lineage[tr])
        assert np.array_equal(meta['initialization_seeds'],shared['neural_initialization_seeds'][r])
        losses=[]
        for c,record in enumerate(meta['tuning']):
            val=p['inner_validation_predictions'][c];assert np.isfinite(val[tr]).all() and np.isnan(val[te]).all()
            fold_losses=[]
            for k in range(5):
                a=tr[inner[tr]!=k];b=tr[inner[tr]==k]
                assert set(a)|set(b)==set(tr) and set(groups[a]).isdisjoint(groups[b])
                assert set(a).isdisjoint(te) and set(b).isdisjoint(te);group_checks+=1
                fold_losses.append(float(np.sqrt(np.mean((y[b]-val[b])**2))))
            err=max(abs(np.array(fold_losses)-np.array(record['fold_rmse'])))
            err=max(err,abs(np.mean(fold_losses)-record['mean_inner_rmse']));max_loss_error=max(max_loss_error,float(err))
            losses.append(float(np.mean(fold_losses)))
            tuning.append({'repeat':repeat,'lineage':label,'candidate':c,'parameters':json.dumps(record['parameters'],sort_keys=True),'mean_inner_rmse':np.mean(fold_losses)})
        assert np.argmin(losses)==meta['selected_candidate'] and max_loss_error<1e-12
        sx=model['x_scaler'];sy=model['y_scaler'];expected_sx=StandardScaler().fit(x[tr]);expected_sy=StandardScaler().fit(y[tr,None])
        assert np.array_equal(sx.mean_,expected_sx.mean_) and np.array_equal(sx.scale_,expected_sx.scale_)
        assert np.array_equal(sy.mean_,expected_sy.mean_) and np.array_equal(sy.scale_,expected_sy.scale_)
        reconstructed=[]
        for network,seed in zip(model['members'],meta['initialization_seeds']):
            assert network.random_state==seed
            z=(x[te]-sx.mean_)/sx.scale_
            for layer,(coef,bias) in enumerate(zip(network.coefs_,network.intercepts_)):
                z=z@coef+bias
                if layer<len(network.coefs_)-1:z=np.tanh(z)
            reconstructed.append(z.ravel()*sy.scale_[0]+sy.mean_[0])
        max_error=max(max_error,float(np.max(np.abs(np.mean(reconstructed,axis=0)-p['test_prediction']))))
        assert np.allclose(reconstructed,p['member_test_predictions'],rtol=0,atol=1e-12) and max_error<1e-10
        assert np.isnan(pred['neural_network'][r,te]).all();pred['neural_network'][r,te]=p['test_prediction'];members[r][:,te]=p['member_test_predictions']
        for diag in meta['fit_diagnostics']:
            assert [m['seed'] for m in diag['members']]==meta['initialization_seeds']
            for member in diag['members']:
                bad=any(w['category']=='ConvergenceWarning' for w in member['warnings'])
                convergence['all_member_fits']+=1;convergence['all_member_fits_with_convergence_warning']+=int(bad)
                if diag.get('stage')=='outer_refit':
                    convergence['outer_member_fits']+=1;convergence['outer_member_fits_with_convergence_warning']+=int(bad)
        parameters.append({'repeat':repeat,'lineage':label,'candidate':meta['selected_candidate'],
            'width':meta['selected_parameters']['width'],'alpha':meta['selected_parameters']['alpha'],'mean_inner_rmse':meta['mean_inner_rmse']})
        if repeat in [1,20] and label in ['Lung','Thyroid']:
            alpha=float(old_folds[(old_folds.repeat==repeat)&old_folds.fold.eq(label)].iloc[0].alpha)
            references.append((x,y,tr,te,meta['selected_parameters'],meta['initialization_seeds'],p['test_prediction'],alpha,pred['ridge'][r,te],label,repeat))
    assert all(np.isfinite(p).all() for p in pred.values()) and np.isfinite(members).all()
    assert convergence['all_member_fits']==30240 and convergence['outer_member_fits']==1440
    with futures.ThreadPoolExecutor(max_workers=3) as pool:reference_results=list(pool.map(lambda args:independent_refit(*args),references))
    assert len(reference_results)==4
    scores=[];fold_scores=[];member_scores=[]
    for r in range(20):
        for name,p in pred.items():
            scores.append({'repeat':r+1,'inner_split_seed':int(shared['inner_split_seeds'][r]),'model':name,**run.prior.metrics(y,p[r])})
            for label in state['labels']:
                te=np.flatnonzero(lineage==label);values=run.prior.metrics(y[te],p[r,te])
                if name=='ridge':
                    old=old_folds[(old_folds.repeat==r+1)&old_folds.fold.eq(label)].iloc[0]
                    np.testing.assert_allclose([values[m] for m in METRICS],[old[m] for m in METRICS],atol=1e-12,rtol=0,equal_nan=True)
                fold_scores.append({'repeat':r+1,'model':name,'lineage':label,'test_n':len(te),**values})
        for j,p in enumerate(members[r]):
            member_scores.append({'repeat':r+1,'member':j,'initialization_seed':int(shared['neural_initialization_seeds'][r,j]),**run.prior.metrics(y,p)})
    scores=pd.DataFrame(scores);folds=pd.DataFrame(fold_scores);eligible=folds[folds.test_n>=10]
    assert eligible.groupby(['repeat','model']).size().eq(19).all()
    macro=eligible.groupby(['repeat','model'],as_index=False)[METRICS].mean();assert np.isfinite(macro[METRICS]).all().all()
    table('per_repeat_metrics.tsv',scores);table('per_lineage_metrics.tsv',folds);table('per_repeat_macro_metrics.tsv',macro)
    table('neural_member_metrics.tsv',pd.DataFrame(member_scores));table('tuning_scores.tsv',pd.DataFrame(tuning));table('selected_parameters.tsv',pd.DataFrame(parameters))
    table('seed_assignments.tsv',pd.DataFrame({'repeat':range(1,21),'inner_split_seed':shared['inner_split_seeds'],
        **{f'neural_seed_{j+1}':shared['neural_initialization_seeds'][:,j] for j in range(3)}}))
    np.savez_compressed(OUT/'predictions.npz',sample_ids=ids,lineage=lineage,group_ids=groups,observed_sensitivity=y,
        inner_split_seeds=shared['inner_split_seeds'],neural_initialization_seeds=shared['neural_initialization_seeds'],**pred)
    np.savez_compressed(OUT/'neural_member_predictions.npz',sample_ids=ids,predictions=members)
    rc=module('bootstrap_methods',WS/'outputs/erastin_repeated_cv/repeat_cv.py')
    boot_check=rc.validate_weighted_bootstrap(y,pred['neural_network'][:2],lineage,run.prior.metrics)
    draws=rc.repeated_group_bootstrap(y,pred,lineage,count=5000,seed=2026110701)
    labels=sorted(eligible.lineage.unique());picks=np.random.default_rng(2026110702).integers(19,size=(5000,19));macro_draws={}
    for name in NAMES:
        values=eligible[eligible.model==name].groupby('lineage')[METRICS].mean().loc[labels]
        macro_draws[name]={m:values[m].to_numpy()[picks].mean(axis=1) for m in METRICS}
    summary=[];paired=[];differences=[]
    for scope,data,boot in [('pooled_745',scores,draws),('macro_19_lineages',macro,macro_draws)]:
        for name in NAMES:
            s=data[data.model==name].sort_values('repeat')
            for metric in METRICS:
                v=s[metric].to_numpy();lo,hi=np.percentile(boot[name][metric],[2.5,97.5])
                summary.append({'scope':scope,'model':name,'metric':metric,'mean':v.mean(),'sd_across_repeats':v.std(ddof=1),
                    'min':v.min(),'max':v.max(),'ci_low':lo,'ci_high':hi})
        a=data[data.model=='neural_network'].set_index('repeat').loc[range(1,21)]
        b=data[data.model=='ridge'].set_index('repeat').loc[range(1,21)]
        for metric in METRICS:
            sign=-1 if metric=='rmse' else 1;v=sign*(a[metric]-b[metric]).to_numpy()
            delta=sign*(boot['neural_network'][metric]-boot['ridge'][metric]);lo,hi=np.percentile(delta,[2.5,97.5])
            paired.append({'scope':scope,'metric':metric,'mean_improvement':v.mean(),'sd_across_repeats':v.std(ddof=1),
                'min_improvement':v.min(),'max_improvement':v.max(),'favorable_repeats':int((v>0).sum()),'total_repeats':20,'ci_low':lo,'ci_high':hi})
            differences += [{'scope':scope,'metric':metric,'repeat':i+1,'improvement':value} for i,value in enumerate(v)]
    summary=pd.DataFrame(summary);paired=pd.DataFrame(paired);differences=pd.DataFrame(differences)
    table('performance_summary.tsv',summary);table('paired_summary.tsv',paired);table('paired_seed_differences.tsv',differences)
    lineage_effects=[]
    for label in state['labels']:
        a=folds[(folds.model=='neural_network')&folds.lineage.eq(label)].set_index('repeat').loc[range(1,21)]
        b=folds[(folds.model=='ridge')&folds.lineage.eq(label)].set_index('repeat').loc[range(1,21)]
        row={'lineage':label,'test_n':int(a.test_n.iloc[0])}
        for metric in METRICS:
            v=(-1 if metric=='rmse' else 1)*(a[metric]-b[metric]);valid=v.dropna()
            row.update({metric+'_mean_improvement':valid.mean(),metric+'_sd_across_repeats':valid.std(ddof=1),
                metric+'_favorable_repeats':int((valid>0).sum()),metric+'_valid_repeats':len(valid)})
        lineage_effects.append(row)
    lineage_effects=pd.DataFrame(lineage_effects);table('lineage_paired_effects.tsv',lineage_effects)
    larger=lineage_effects[lineage_effects.test_n>=10]
    wins=pd.DataFrame([{'metric':m,'favorable_lineages':int((larger[m+'_mean_improvement']>0).sum()),'eligible_lineages':19} for m in METRICS])
    table('lineage_win_counts.tsv',wins)
    np.savez_compressed(OUT/'bootstrap_draws.npz',**{scope+'__'+name+'__'+m:v for scope,d in [('pooled',draws),('macro',macro_draws)] for name,a in d.items() for m,v in a.items()})
    validation={'passed':True,'source_and_frozen_script_hashes_verified':True,'matched_ridge_seeds_verified':True,
        'matched_fixed_outer_lineages_and_identity_exclusion_verified':True,'independent_inner_membership_checks':group_checks,
        'new_ensemble_models_reconstructed':480,'candidate_validation_sets_audited':1920,
        'max_prediction_reconstruction_error':max_error,'max_recomputed_inner_loss_error':max_loss_error,
        'train_only_feature_and_target_scaling_verified':True,'independent_sklearn_refits':reference_results,
        'ridge_per_lineage_metrics_reproduced':True,'neural_convergence':convergence,**boot_check}
    run.prior.dump(OUT/'validation.json',validation)
    render(summary,paired,differences,lineage_effects,wins,convergence,pd.DataFrame(parameters))
    run.prior.dump(OUT/'final_summary.json',{'status':'complete','repeats':20,'samples':745,'heldout_lineages':24,
        'performance':summary.to_dict('records'),'paired_effects':paired.to_dict('records'),'lineage_win_counts':wins.to_dict('records'),
        'neural_convergence':convergence})
    files=[{'path':p.relative_to(OUT).as_posix(),'bytes':p.stat().st_size,'sha256':run.prior.digest(p)} for p in sorted(OUT.rglob('*'))
        if p.is_file() and p.name not in ['artifact_manifest.json','progress.json']]
    run.prior.dump(OUT/'artifact_manifest.json',{'files':files})
    archive=OUT.parent/(OUT.name+'_results.zip')
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in sorted(OUT.rglob('*')):
            if p.is_file() and p.name!='progress.json':z.write(p,arcname=str(Path(OUT.name)/p.relative_to(OUT)))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for item in files:assert hashlib.sha256(z.read(str(Path(OUT.name)/item['path']))).hexdigest()==item['sha256']
    run.prior.dump(OUT/'progress.json',{'phase':'complete','completed_outer_jobs':480,'total_outer_jobs':480,'complete_repeats':20})
    print(json.dumps(run.prior.clean({'status':'complete','validation':validation,'paired_effects':paired.to_dict('records')})),flush=True)


if __name__=='__main__':main()
