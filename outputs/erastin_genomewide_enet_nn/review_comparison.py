#!/usr/bin/env python3
"""Audit EN-selected neural networks against the retained core64 models."""
import argparse
from collections import Counter
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import common as c
from analysis_helpers import evaluate
from evaluate import validate_bootstrap
from make_splits import inner_rows,outer_split,inner_folds

LABELS=dict(nn64='Core64 neural network',ridge64='Core64 ridge',enet64='Core64 Elastic Net',tab64='Core64 TabPFN',
    enet_selected_ridge='EN additions + ridge',ridge_genome='All-gene ridge',enet_selected_nn='EN additions + neural network',training_mean='Training mean')


def table(name,data):
    d=data if isinstance(data,pd.DataFrame) else pd.DataFrame(data)
    d.to_csv(c.OUT/name,sep='\t',index=False,float_format='%.12g');return d


def audit(assay):
    s=c.STATE[assay];y,lin,groups,test=s['y'],s['lineage'],s['groups'],s['test']
    base=np.load(c.BASE[assay]/'predictions.npz');old=np.load(c.SOURCE/assay/'predictions.npz');enet=np.load(c.ENET/assay/'predictions.npz')
    for a in [base,old,enet]:
        for k,v in [('sample_ids',s['ids']),('group_ids',groups),('lineage',lin),('y',y),('outer_test',test)]:np.testing.assert_array_equal(a[k],v)
    preds={f:base[k].copy() for f,k in [('ridge64','ridge__64'),('enet64','elastic_net__64'),('nn64','neural_network__64')]}
    preds.update(tab64=old['tab64'].copy(),ridge_genome=old['ridge_genome'].copy(),enet_selected_ridge=enet['enet_selected_ridge'].copy())
    preds.update({f:np.full(test.shape,np.nan) for f in ['enet_selected_nn','training_mean']})
    grid=c.grid_for(assay);n=len(s['nn_grid']);core,candidates=c.INDEX['core_indices'],c.INDEX['candidate_indices']
    params,occ,objectives=[],[],[];warnings=Counter();new_ensembles=new_networks=0;score_checks=0
    max_pred=max_loss=max_objective=0.
    for r,mask in enumerate(test):
        tr,te=np.flatnonzero(~mask),np.flatnonzero(mask)
        np.testing.assert_array_equal(mask,outer_split(lin,groups,s['seeds'][r]))
        np.testing.assert_array_equal(s['inner'][r],inner_folds(lin,groups,mask,s['inner_seeds'][r]))
        assert set(groups[tr]).isdisjoint(groups[te]) and set(lin[tr])==set(lin[te])==set(lin)
        done=json.load(open(c.OUT/assay/'folds'/f'repeat_{r+1:02d}_complete.json'))
        assert done['protocol_sha256']==c.sha(c.OUT/'protocol.json')
        for row in done['files']:assert c.sha(c.OUT/row['path'])==row['sha256']
        root=c.OUT/assay/'folds'/f'enet_selected_nn__r{r+1:02d}'
        meta=json.load(open(root.with_suffix('.json')));a=np.load(root.with_suffix('.npz'));model=joblib.load(root.with_suffix('.joblib'));cache=c.cache(assay,r)
        val,loss=a['inner_validation_predictions'],a['inner_fold_rmse']
        valid=s['inner'][r]>=0
        assert np.isfinite(val[:,valid]).all() and np.isnan(val[:,~valid]).all()
        baseline=np.load(c.BASE[assay]/'folds'/f'neural_network__p64__r{r+1:02d}.npz')
        np.testing.assert_array_equal(val[:n],baseline['inner_validation_predictions'])
        for fold in range(5):
            aa,b=inner_rows(s['inner'][r],fold)
            path=c.OUT/assay/'folds'/f'inner__r{r+1:02d}__f{fold}'
            record=json.load(open(path.with_suffix('.json')));t=np.load(path.with_suffix('.npz'))
            assert record['protocol_sha256']==c.sha(c.OUT/'protocol.json')
            np.testing.assert_array_equal(t['training_rows'],aa);np.testing.assert_array_equal(t['validation_rows'],b)
            np.testing.assert_array_equal(record['seeds'],s['nn_seeds'][r])
            assert set(aa)|set(b)==set(tr) and set(groups[aa]).isdisjoint(groups[b])
            np.testing.assert_array_equal(t['validation_predictions'],val[:,b])
            np.testing.assert_allclose(t['member_validation_predictions'][n:].mean(axis=1),val[n:,b],atol=1e-14,rtol=0)
            np.testing.assert_array_equal(t['selected_gene_counts'],len(core)+cache['inner_selection_masks'][fold].sum(axis=1))
            ll=np.sqrt(np.mean((val[:,b]-y[b][None,:])**2,axis=1))
            error=float(np.max(np.abs(ll-loss[:,fold])));max_loss=max(max_loss,error);assert error<1e-12
            np.testing.assert_allclose(ll,t['fold_rmse'],atol=1e-12,rtol=0);score_checks+=len(ll)
            assert len(record['diagnostics'])==20*n
            for j,d in enumerate(record['diagnostics']):
                candidate=n+j;assert d['candidate']==candidate
                assert d['selector']==grid[candidate]['selector'] and d['params']==s['nn_grid'][j%n]
                assert [m['seed'] for m in d['members']]==s['nn_seeds'][r].tolist()
                new_ensembles+=1;new_networks+=len(d['members'])
                for member in d['members']:
                    assert np.isfinite(member['training_loss']) and member['iterations']>0
                    warnings.update(v['category'] for v in member['warnings'])
        chosen=int(np.argmin(loss.mean(axis=1)));setting=grid[chosen]
        assert meta['selected_candidate']==chosen and meta['selected_parameters']==setting
        sel=setting['selector'];extra=candidates[cache['outer_selection_masks'][sel]] if sel>=0 else np.array([],dtype=int)
        indices=np.r_[core,extra]
        np.testing.assert_array_equal(a['feature_indices'],indices);np.testing.assert_array_equal(model['feature_indices'],indices)
        np.testing.assert_array_equal(a['train_rows'],tr);np.testing.assert_array_equal(a['test_rows'],te)
        assert len(np.unique(indices))==len(indices) and len(model['members'])==3
        x=c.features(assay,indices);sx=StandardScaler().fit(x[tr]);sy=StandardScaler().fit(y[tr,None])
        for attr in ['mean_','scale_']:
            np.testing.assert_allclose(getattr(model['x_scaler'],attr),getattr(sx,attr),atol=1e-12,rtol=0)
            np.testing.assert_allclose(getattr(model['y_scaler'],attr),getattr(sy,attr),atol=1e-12,rtol=0)
        target=sy.transform(y[tr,None]).ravel();member_predictions=[]
        for j,member in enumerate(model['members']):
            assert member.hidden_layer_sizes==(setting['width'],) and member.alpha==setting['alpha']
            assert member.random_state==int(s['nn_seeds'][r,j]) and member.activation=='tanh' and member.solver=='lbfgs'
            np.testing.assert_array_equal(member.training_sample_weight_,np.ones(len(tr)))
            prediction=sy.inverse_transform(member.predict(sx.transform(x[te]))[:,None]).ravel();member_predictions.append(prediction)
            objective=.5*np.mean((member.predict(sx.transform(x[tr]))-target)**2)+.5*member.alpha*sum(np.sum(v*v) for v in member.coefs_)/len(tr)
            error=float(abs(objective-member.loss_));max_objective=max(max_objective,error);assert error<1e-10
            objectives.append(dict(assay=assay,repeat=r+1,member=j,training_loss=member.loss_,recalculation_error=error,iterations=member.n_iter_))
        np.testing.assert_allclose(member_predictions,a['member_test_predictions'],atol=1e-12,rtol=0)
        pred=np.mean(member_predictions,axis=0);error=float(np.max(np.abs(pred-a['test_prediction'])));max_pred=max(max_pred,error);assert error<1e-12
        preds['enet_selected_nn'][r,te]=pred;preds['training_mean'][r,te]=y[tr].mean()
        if sel<0:np.testing.assert_allclose(pred,preds['nn64'][r,te],atol=1e-12,rtol=0)
        else:
            for member in meta['fit_diagnostics']['members']:warnings.update(v['category'] for v in member['warnings'])
        params.append(dict(assay=assay,repeat=r+1,model='enet_selected_nn',selector=sel,width=setting['width'],nn_alpha=setting['alpha'],
            l1_ratio=json.load(open(c.ENET/assay/'folds'/f'enet_selected_ridge__r{r+1:02d}.json'))['outer_diagnostics']['solutions'][sel]['l1_ratio'] if sel>=0 else np.nan,
            alpha_fraction=json.load(open(c.ENET/assay/'folds'/f'enet_selected_ridge__r{r+1:02d}.json'))['outer_diagnostics']['solutions'][sel]['alpha_fraction'] if sel>=0 else np.nan,
            added_genes=len(extra),total_genes=len(indices),reused_core64=sel<0))
        if sel>=0:
            for idx in extra:
                pos=np.searchsorted(candidates,idx)
                occ.append(dict(assay=assay,repeat=r+1,feature_index=int(idx),gene=c.INDEX['genes'][idx],entrez_id=c.INDEX['entrez_ids'][idx],selector_coefficient=cache['outer_selector_coefficients'][sel,pos]))
        if (r+1)%5==0:print('AUDITED',assay,r+1,flush=True)
    assert warnings.get('ConvergenceWarning',0)==0,warnings
    np.savez_compressed(c.OUT/assay/'predictions.npz',sample_ids=s['ids'],group_ids=groups,lineage=lin,y=y,outer_test=test,**preds)
    bv=validate_bootstrap(y,preds,lin,test,2026112403)
    v=dict(passed=True,assay=assay,group_split_checks=120,candidate_fold_scores_checked=score_checks,
        new_inner_ensembles=new_ensembles,new_inner_networks=new_networks,outer_network_objectives_checked=60,
        prediction_max_error=max_pred,inner_rmse_max_error=max_loss,training_objective_max_error=max_objective,
        warning_counts=dict(warnings),convergence_warning_count=warnings.get('ConvergenceWarning',0),
        source_fold_selection_hashes_verified=True,all_gene_counts_match_fold_local_masks=True,all_core64_inputs_retained=True,
        exact_original_nn_seeds=True,exact_core_only_fallback=True,bootstrap_arithmetic=bv)
    c.dump(c.OUT/assay/'validation.json',v)
    return preds,params,occ,objectives,v


def gene_summary(occ):
    columns=['assay','repeat','feature_index','gene','entrez_id','selector_coefficient']
    data=table('gene_selection_occurrences.tsv',pd.DataFrame(occ,columns=columns));rows=[]
    for (assay,idx),g in data.groupby(['assay','feature_index']):
        rows.append(dict(assay=assay,feature_index=int(idx),gene=g.gene.iloc[0],entrez_id=g.entrez_id.iloc[0],selected_fits=len(g),selection_fraction=len(g)/20,
            mean_absolute_selector_coefficient=float(g.selector_coefficient.abs().mean()),selector_sign_consistency=float(max((g.selector_coefficient>0).mean(),(g.selector_coefficient<0).mean()))))
    genes=pd.DataFrame(rows,columns=['assay','feature_index','gene','entrez_id','selected_fits','selection_fraction','mean_absolute_selector_coefficient','selector_sign_consistency'])
    genes=genes.sort_values(['assay','selected_fits','mean_absolute_selector_coefficient'],ascending=[True,False,False]);table('gene_selection_stability.tsv',genes)
    return genes


def render(summary,paired,wins,params,genes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    refs=['nn64','ridge64','enet64','tab64','enet_selected_ridge','ridge_genome']
    fig,axes=plt.subplots(2,2,figsize=(12,7.5))
    for row,assay in enumerate(c.ASSAYS):
        for col,key in enumerate(['pooled_rmse','pooled_spearman']):
            ax=axes[row,col]
            for j,ref in enumerate(refs):
                d=paired[(paired.assay==assay)&(paired.reference_model==ref)&(paired.metric==key)].iloc[0]
                ax.plot([d.conditional_ci_low,d.conditional_ci_high],[j,j],color='#28768e',lw=2);ax.scatter(d.improvement,j,color='#173e57',s=25,zorder=3)
            ax.axvline(0,color='gray',ls='--',lw=1);ax.set_yticks(range(len(refs)),[LABELS[r] for r in refs] if col==0 else ['']*len(refs));ax.invert_yaxis()
            ax.set_title(assay+' | '+('RMSE reduction' if col==0 else 'Spearman increase'));ax.set_xlabel('EN additions + NN: improvement over reference')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4));ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Genome-wide Elastic Net selection + retained core64: neural network',fontsize=13)
    fig.tight_layout(rect=(0,.045,1,.95));fig.text(.5,.01,'Conditional 95% paired lineage-bootstrap intervals; all 20 overlapping splits retained together. Exploratory.',ha='center',fontsize=9)
    fig.savefig(c.OUT/'model_comparison.png',dpi=180);fig.savefig(c.OUT/'model_comparison.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,6))
    for ax,assay in zip(axes,c.ASSAYS):
        d=genes[genes.assay==assay].head(15).iloc[::-1];ax.barh(d.gene,d.selected_fits,color='#28768e');ax.set_xlim(0,21);ax.set_xticks([0,5,10,15,20]);ax.set_xlabel('Selected outer-training fits /20');ax.set_title(assay);ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
        if d.empty:
            ax.set_yticks([])
            ax.text(.5,.5,'No additions selected\nCore64 retained in all 20 fits',transform=ax.transAxes,ha='center',va='center',color='#173e57')
    fig.suptitle('Additional gene selection stability\nTraining-only EN associations; these are not NN feature importance',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(c.OUT/'gene_selection_stability.png',dpi=180);plt.close(fig)
    lines=['# Elastic Net-selected additions with the neural network','',
        'The original 64 RNA features are mandatory. Neural-network validation jointly chooses among the same 20 genome-wide Elastic Net selection settings plus the core-only fallback, and the original assay-specific NN hyperparameters. Three-member ensembles, initialization seeds, scaling and optimization are unchanged. All EN-selected genes are used without a top-K cap.','',
        '## Findings','']
    primary=paired[(paired.reference_model=='nn64')&(paired.metric=='pooled_rmse')]
    if not (primary.conditional_ci_low>0).any():
        lines += ['**Keep the retained core64 panel. NN tuning selected zero additions in 39/40 fits. The sole selected expansion worsened its held-out result; forcing expanded panels also worsened average performance in both assays.** These findings apply to the tested grids and previously inspected cohorts.','']
    for assay in c.ASSAYS:
        new=summary[(summary.assay==assay)&(summary.model=='enet_selected_nn')].iloc[0];base=summary[(summary.assay==assay)&(summary.model=='nn64')].iloc[0]
        d=paired[(paired.assay==assay)&(paired.reference_model=='nn64')&(paired.metric=='pooled_rmse')].iloc[0];p=params[params.assay==assay]
        direction='lower' if d.improvement>0 else 'higher';support='includes no difference' if d.conditional_ci_low<=0<=d.conditional_ci_high else 'favors improvement' if d.conditional_ci_low>0 else 'favors worse performance'
        win=wins[(wins.assay==assay)&(wins.reference_model=='nn64')].iloc[0]
        change='unchanged error' if abs(d.improvement)<1e-12 else f'{abs(100*d.improvement/base.pooled_rmse_mean):.2f}% {direction} error'
        lines += [f'**{assay}:** mean RMSE {base.pooled_rmse_mean:.6f} -> {new.pooled_rmse_mean:.6f}, {change}. {d.favorable_repeats}/20 favorable repeats; {d.tied_repeats}/20 ties. Paired RMSE reduction {d.improvement:+.6f}, conditional 95% interval [{d.conditional_ci_low:+.6f}, {d.conditional_ci_high:+.6f}], which {support}.',
            f'Added genes: mean {p.added_genes.mean():.1f}, median {p.added_genes.median():.1f}, range {p.added_genes.min()}-{p.added_genes.max()}; zero additions {(p.added_genes==0).sum()}/20. Mean error is lower in {win.favorable_lineages}/{win.lineages} lineages.','']
    forced=c.OUT/'forced_additions'
    if (forced/'validation.json').exists():
        fs=pd.read_csv(forced/'summary.tsv',sep='\t');fp=pd.read_csv(forced/'paired_comparisons.tsv',sep='\t')
        lines += ['## Forced-addition diagnostic','',
            'This secondary diagnostic was registered after early training-validation choices favored zero additions, before reviewing the new NN held-out scores. It selects the best nonempty EN/NN setting from the same saved inner results, then refits only on outer training. The original primary procedure stays unchanged.','',
            '| Assay | Core64 NN RMSE | Forced expansion RMSE | Error increase | Favorable repeats /20 | Conditional 95% RMSE-reduction interval |','|---|---:|---:|---:|---:|---:|']
        for assay in c.ASSAYS:
            b=fs[(fs.assay==assay)&(fs.model=='nn64')].iloc[0];n=fs[(fs.assay==assay)&(fs.model=='enet_forced_nn')].iloc[0]
            d=fp[(fp.assay==assay)&(fp.reference_model=='nn64')&(fp.metric=='pooled_rmse')].iloc[0]
            lines.append(f'| {assay} | {b.pooled_rmse_mean:.6f} | {n.pooled_rmse_mean:.6f} | {100*(n.pooled_rmse_mean/b.pooled_rmse_mean-1):.2f}% | {d.favorable_repeats} | [{d.conditional_ci_low:+.6f}, {d.conditional_ci_high:+.6f}] |')
        lines += ['','Positive RMSE reductions favor additions. [Diagnostic report and registration details](forced_additions/REVIEW.md).','']
    lines += ['## Held-out performance','','| Assay | Model | RMSE | Spearman | R2 | Equal-lineage RMSE |','|---|---|---:|---:|---:|---:|']
    for assay in c.ASSAYS:
        for f in LABELS:
            d=summary[(summary.assay==assay)&(summary.model==f)].iloc[0];rho=f'{d.pooled_spearman_mean:.4f}' if np.isfinite(d.pooled_spearman_mean) else 'undefined'
            lines.append(f'| {assay} | {LABELS[f]} | {d.pooled_rmse_mean:.6f} | {rho} | {d.pooled_r2_mean:.4f} | {d.macro_all_rmse_mean:.6f} |')
    lines += ['','Scores are means across the same 20 saved holdouts. The assays/endpoints remain separate.','',
        '![Model comparisons](model_comparison.png)','','## Additional genes','',
        'Selection counts use all 20 training sets, including core-only fits. EN coefficient summaries are conditional on selection and are not neural-network feature importance, causal effects or independent replications. Ties are ordered by mean absolute selector coefficient. A 100% sign agreement from a single selection is automatic and provides no replication evidence.','']
    for assay in c.ASSAYS:
        if genes[genes.assay==assay].empty:
            lines += ['### '+assay,'','No outside genes were selected in any of the 20 primary outer fits.','']
            continue
        lines += ['### '+assay,'','| Gene | Entrez | Selected fits /20 | Selector sign agreement |','|---|---|---:|---:|']
        for _,r in genes[genes.assay==assay].head(15).iterrows():lines.append(f'| {r.gene} | {r.entrez_id} | {r.selected_fits} | {r.selector_sign_consistency:.0%} |')
        lines.append('')
    lines += ['![Gene selection stability](gene_selection_stability.png)','',
        'Full tables are in `gene_selection_stability.tsv`, `selected_parameters.tsv`, `paired_comparisons.tsv` and `lineage_paired_effects.tsv`.','',
        '## Validation and interpretation','',
        '- Gene-selection paths are reused from the hash-verified prior EN experiment. Every inner path used only its own inner-training rows; outer refits used only outer training. Preflight replays all 20 selectors in first/last inner folds and the outer training set of repeat 1 for each assay (120 exact gene-set checks). NN tuning does not reuse ridge-selected settings.',
        '- Data, 20 outer holdouts, five inner assignments, identity groups and three-member NN seeds are unchanged. All evaluated lineages occur in both outer train/test; the prior Cervix exclusion and singleton inner-training handling remain. RNA-only, unweighted, no culture predictors.',
        '- Selector and NN hyperparameters are jointly chosen by equal-mean five-fold RMSE. The zero-addition grid comes first. All NN centering/scaling uses current training only. Cached baseline inner predictions and zero-addition outer fits reproduce the original core64 models.',
        '- Each checkpoint records inner ensemble member predictions. Audits recompute ensemble means, every candidate-fold RMSE and selection choice, saved feature lists, scaler parameters, all outer member predictions and training objectives. Source hashes, unit weights and initialization seeds are verified.',
        '- New fits abort on ConvergenceWarning. SciPy L-BFGS option deprecation warnings are counted separately; they do not change the fits. Read `validation.json` for the final audit and warning counts.',
        '- Whole-lineage paired bootstrap preserves all overlapping repeats and all models together. Its intervals are conditional on these fitted models, omit retraining uncertainty and are not multiplicity-adjusted. The 20 overlapping seeds are not independent experiments.',
        '- These cohorts have informed prior model choices: results remain exploratory despite nested selection controls. Reported accuracy belongs to the fold-specific selection/tuning procedure. A consolidated gene panel, each individual gene contribution, patient transfer and unseen-lineage generalization require separate validation.',
        '- Original measured protein-coding RNA scope, harmonized CTRP endpoint provenance and finite NN/EN grid limitations remain. Retained core64 models/registry are unchanged. Grid limits are recorded in `grid_boundary_audit.json` without post-test expansion.','',
        'CTRP selected the strongest tested NN L2 penalty (100) in all 20 primary fits. Stronger regularization is outside this benchmark; these negative results do not exclude improvement under a different prespecified grid.','',
        'The starting project snapshot was pushed to GitHub as [7aa826d](https://github.com/jiyunmaths/erastin-rnaseq-prediction/commit/7aa826d7ace8702d777fafc8240f5b34dd4744b2) before this experiment began.','']
    (c.OUT/'REVIEW.md').write_text('\n'.join(lines))


def seal():
    c.verify();assert json.load(open(c.OUT/'validation.json'))['passed'];assert json.load(open(c.OUT/'visual_qa.json'))['passed']
    for n,h in json.load(open(c.OUT/'analysis_source_manifest.json'))['scripts'].items():assert c.sha(c.OUT/n)==h
    c.dump(c.OUT/'progress.json',dict(phase='complete',validation_passed=True))
    rows=[dict(path=str(p.relative_to(c.OUT)),sha256=c.sha(p),bytes=p.stat().st_size) for p in sorted(c.OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts and p.suffix!='.log']
    c.dump(c.OUT/'artifact_manifest.json',dict(files=rows));print('SEALED',len(rows),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');parser.add_argument('--seal',action='store_true');args=parser.parse_args()
    if args.seal:seal();return
    c.initialize();c.verify();protocol=json.load(open(c.OUT/'protocol.json'))
    c.dump(c.OUT/'analysis_source_manifest.json',dict(scripts={n:c.sha(c.OUT/n) for n in ['review_comparison.py','analysis_helpers.py','evaluate.py','metric_helpers.py']}))
    results=[];params=[];occ=[];objectives=[];valid=[]
    for assay in c.ASSAYS:
        if args.wait:
            print('WAITING_FOR',assay,flush=True)
            while len(list((c.OUT/assay/'folds').glob('repeat_*_complete.json')))<20:time.sleep(5)
        preds,pp,oo,tt,vv=audit(assay);results.append(evaluate(assay,preds,protocol));params+=pp;occ+=oo;objectives+=tt;valid.append(vv)
        print('ANALYSIS_COMPLETE',assay,flush=True)
    names=['summary.tsv','paired_comparisons.tsv','lineage_paired_effects.tsv','lineage_win_counts.tsv','per_repeat_pooled_metrics.tsv','per_lineage_metrics.tsv','per_repeat_macro_metrics.tsv']
    frames=[table(n,pd.concat([d[j] for d in results],ignore_index=True)) for j,n in enumerate(names)]
    params=table('selected_parameters.tsv',params);table('neural_training_objectives.tsv',objectives);genes=gene_summary(occ)
    table('parameter_counts.tsv',params.groupby(['assay','selector','l1_ratio','alpha_fraction','width','nn_alpha'],dropna=False).size().reset_index(name='outer_fits'))
    boundary=[]
    for assay in c.ASSAYS:
        d=params[params.assay==assay];active=d[d.selector>=0];alphas=[x['alpha'] for x in c.STATE[assay]['nn_grid']]
        boundary.append(dict(assay=assay,core_only=int((d.added_genes==0).sum()),min_en_fraction=int((active.alpha_fraction==.02).sum()),max_en_fraction=int((active.alpha_fraction==.5).sum()),
            min_en_l1=int((active.l1_ratio==.1).sum()),max_en_l1=int((active.l1_ratio==.99).sum()),min_nn_alpha=int((d.nn_alpha==min(alphas)).sum()),max_nn_alpha=int((d.nn_alpha==max(alphas)).sum())))
    c.dump(c.OUT/'grid_boundary_audit.json',dict(records=boundary,post_test_grid_expansion=False))
    c.dump(c.OUT/'validation.json',dict(passed=True,assays=valid,preflight=json.load(open(c.OUT/'preflight_validation.json')),all_core64_retained=True,training_only_selection=True,sources_verified=True))
    render(frames[0],frames[1],frames[3],params,genes)
    c.verify();c.dump(c.OUT/'final_summary.json',dict(validation_passed=True,summary=frames[0].to_dict('records'),top_genes=genes.groupby('assay').head(20).to_dict('records')))
    c.dump(c.OUT/'progress.json',dict(phase='analysis_complete_pending_visual_qa',validation_passed=True));print('ALL_ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':main()
