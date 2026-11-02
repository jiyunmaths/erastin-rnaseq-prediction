#!/usr/bin/env python3
"""Audit saved nested selection, quantify paired effects and candidate stability."""
import argparse
import json
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import common as c
import models
from make_splits import inner_rows,outer_split,inner_folds
from evaluate import KEYS,score,bootstrap,validate_bootstrap,finite_mean

BASELINES=dict(ridge64='ridge__64',enet64='elastic_net__64',nn64='neural_network__64')
PREVIOUS=['tab64','ridge_genome','ridge_screened']
LABELS=dict(ridge64='Core64 ridge',enet64='Core64 Elastic Net',nn64='Core64 neural network',
    tab64='Core64 TabPFN',ridge_genome='All-gene ridge',ridge_screened='Previous screened additions: ridge',
    enet_selected_ridge='Core64 + Elastic Net additions: ridge',training_mean='Training mean')


def table(name,data):
    d=data if isinstance(data,pd.DataFrame) else pd.DataFrame(data)
    d.to_csv(c.OUT/name,sep='\t',index=False,float_format='%.12g')
    return d


def audit(assay):
    s=c.STATE[assay];y,lin,groups,test=s['y'],s['lineage'],s['groups'],s['test']
    base=np.load(c.BASE[assay]/'predictions.npz');old=np.load(c.SOURCE/assay/'predictions.npz')
    for a in [base,old]:
        for k,v in [('sample_ids',s['ids']),('group_ids',groups),('lineage',lin),('y',y),('outer_test',test)]:np.testing.assert_array_equal(a[k],v)
    preds={f:base[key].copy() for f,key in BASELINES.items()}
    preds.update(tab64=old['tab64'].copy(),ridge_genome=old['ridge_genome'].copy(),ridge_screened=old['ridge_augmented'].copy())
    preds.update({f:np.full(test.shape,np.nan) for f in ['enet_selected_ridge','training_mean']})
    params,occurrences,solver=[],[],[];pred_error=rmse_error=max_kkt=0.;replays=[]
    core,candidates=c.INDEX['core_indices'],c.INDEX['candidate_indices']
    for r,mask in enumerate(test):
        tr,te=np.flatnonzero(~mask),np.flatnonzero(mask)
        np.testing.assert_array_equal(mask,outer_split(lin,groups,s['seeds'][r]))
        np.testing.assert_array_equal(s['inner'][r],inner_folds(lin,groups,mask,s['inner_seeds'][r]))
        assert set(groups[tr]).isdisjoint(groups[te]) and set(lin[tr])==set(lin[te])==set(lin)
        done=json.load(open(c.OUT/assay/'folds'/f'repeat_{r+1:02d}_complete.json'))
        assert done['protocol_sha256']==c.sha(c.OUT/'protocol.json')
        for row in done['files']:assert c.sha(c.OUT/row['path'])==row['sha256']
        root=c.OUT/assay/'folds'/f'enet_selected_ridge__r{r+1:02d}'
        m=json.load(open(root.with_suffix('.json')));a=np.load(root.with_suffix('.npz'))
        model=joblib.load(root.with_suffix('.joblib'))
        np.testing.assert_array_equal(a['train_rows'],tr);np.testing.assert_array_equal(a['test_rows'],te)
        val,loss=a['inner_validation_predictions'],a['inner_fold_rmse']
        for k in range(5):
            aa,b=inner_rows(s['inner'][r],k)
            ll=np.sqrt(np.mean((val[:,b]-y[b][None,:])**2,axis=1))
            err=float(np.max(np.abs(loss[:,k]-ll)));rmse_error=max(rmse_error,err);assert err<1e-12
            assert set(aa)|set(b)==set(tr) and set(groups[aa]).isdisjoint(groups[b])
            for d in m['inner_diagnostics'][k]['solutions']:
                assert d['kkt']<=models.KKT_LIMIT and d['dual_gap']<2e-5
                solver.append(dict(assay=assay,repeat=r+1,stage='inner',fold=k,**d))
        chosen=int(np.argmin(loss.mean(axis=1)));setting=models.GRID[chosen]
        assert chosen==m['selected_candidate'] and setting==m['selected_parameters']
        sel=setting['selector']
        extra=candidates[a['outer_selection_masks'][sel]] if sel>=0 else np.array([],dtype=int)
        indices=np.r_[core,extra]
        np.testing.assert_array_equal(a['feature_indices'],indices)
        np.testing.assert_array_equal(model['feature_indices'],indices)
        assert len(np.unique(indices))==len(indices) and set(core).issubset(indices)
        scale=StandardScaler().fit(c.features(assay,tr,indices))
        np.testing.assert_allclose(model['x_scaler'].mean_,scale.mean_,atol=1e-12,rtol=0)
        np.testing.assert_allclose(model['x_scaler'].scale_,scale.scale_,atol=1e-12,rtol=0)
        pred=models.predict(model,c.features(assay,te,indices))
        err=float(np.max(np.abs(pred-a['test_prediction'])));pred_error=max(pred_error,err);assert err<1e-10
        preds['enet_selected_ridge'][r,te]=pred;preds['training_mean'][r,te]=y[tr].mean()
        if sel<0:np.testing.assert_allclose(pred,preds['ridge64'][r,te],atol=1e-10,rtol=0)
        # Independently certify every outer selector solution from saved weights.
        x=c.features(assay,tr)
        z,target,ix,diag=models.residual_design(x,y[tr],lin[tr],core,candidates)
        positions=np.searchsorted(candidates,ix)
        beta=a['outer_selector_coefficients'][:,positions].T
        residual=z@beta-target[:,None]
        gradient=z.T@residual/len(tr)
        for j,d in enumerate(m['outer_diagnostics']['solutions']):
            ratio,alpha=d['l1_ratio'],d['alpha']
            g=gradient[:,j]+alpha*(1-ratio)*beta[:,j]
            nz=beta[:,j]!=0
            err=float(np.max(np.where(nz,np.abs(g+alpha*ratio*np.sign(beta[:,j])),np.maximum(np.abs(g)-alpha*ratio,0))))
            # Thresholding literal tiny coefficients for stored selection can add negligible error.
            assert err<models.KKT_LIMIT+1e-7,err
            max_kkt=max(max_kkt,err)
            np.testing.assert_array_equal(a['outer_selection_masks'][j],np.abs(a['outer_selector_coefficients'][j])>models.NONZERO)
            solver.append(dict(assay=assay,repeat=r+1,stage='outer',fold=-1,independent_kkt=err,**d))
        # Replay both end inner folds of the first repeat without validation labels.
        if r==0:
            for k in [0,4]:
                aa,b=inner_rows(s['inner'][r],k)
                fits,dd=models.select_paths(c.features(assay,aa),y[aa],lin[aa],core,candidates)
                for j,fit in enumerate(fits):
                    np.testing.assert_array_equal(candidates[a['inner_selection_masks'][k,j]],fit['feature_indices'])
                replays.append(dict(repeat=1,fold=k,selector_gene_sets_checked=len(fits),identical=True))
        params.append(dict(assay=assay,repeat=r+1,model='enet_selected_ridge',selector=sel,ridge_alpha=setting['ridge_alpha'],
            l1_ratio=models.SELECTORS[sel]['l1_ratio'] if sel>=0 else np.nan,
            alpha_fraction=models.SELECTORS[sel]['alpha_fraction'] if sel>=0 else np.nan,
            enet_alpha=m['outer_diagnostics']['solutions'][sel]['alpha'] if sel>=0 else np.nan,
            added_genes=len(extra),total_genes=len(indices)))
        if sel>=0:
            selected_coef=a['outer_selector_coefficients'][sel]
            for j,gene_index in enumerate(extra):
                pos=np.searchsorted(candidates,gene_index)
                occurrences.append(dict(assay=assay,repeat=r+1,feature_index=int(gene_index),gene=c.INDEX['genes'][gene_index],
                    entrez_id=c.INDEX['entrez_ids'][gene_index],selector_coefficient=selected_coef[pos],ridge_coefficient=model['coef'][len(core)+j]))
        if (r+1)%5==0:print('AUDITED',assay,r+1,flush=True)
    np.savez_compressed(c.OUT/assay/'predictions.npz',sample_ids=s['ids'],group_ids=groups,lineage=lin,y=y,outer_test=test,**preds)
    bv=validate_bootstrap(y,preds,lin,test,2026112303)
    v=dict(passed=True,assay=assay,group_split_checks=120,inner_candidate_fold_scores_checked=len(models.GRID)*5*20,
        recorded_inner_enet_solutions_checked=20*5*len(models.SELECTORS),independent_outer_enet_kkt_checks=20*len(models.SELECTORS),
        solver_solutions=len(solver),solver_refinements=sum(d['refinements'] for d in solver),
        recovered_solver_warnings=sum(d['warning_count'] for d in solver),max_recorded_dual_gap=max(d['dual_gap'] for d in solver),
        outer_kkt_max=max_kkt,inner_rmse_max_error=rmse_error,ridge_prediction_max_error=pred_error,
        replayed_training_only_inner_selections=replays,all_core64_inputs_retained=True,all_baseline_predictions_unchanged=True,
        bootstrap_arithmetic=bv)
    c.dump(c.OUT/assay/'validation.json',v)
    return preds,params,occurrences,solver,v


def evaluate(assay,preds,protocol):
    s=c.STATE[assay];y,lin,test=s['y'],s['lineage'],s['test']
    pooled,per,macro=score(y,preds,lin,test)
    for d in [pooled,per,macro]:d.insert(0,'assay',assay)
    values,summary={},[]
    for f in preds:
        values[f]={'pooled_'+k:pooled[pooled.model==f].sort_values('repeat')[k].to_numpy() for k in ['spearman','pearson','rmse','mae','r2']}
        values[f].update({k:macro[macro.model==f].sort_values('repeat')[k].to_numpy() for k in ['macro_all_rmse','macro_large_rmse']})
        row=dict(assay=assay,model=f)
        for k,v in values[f].items():
            good=v[np.isfinite(v)];row.update({k+'_mean':float(finite_mean(v)),k+'_sd':float(good.std(ddof=1)) if len(good)>1 else np.nan})
        summary.append(row)
    draws,perdraw=bootstrap(y,preds,lin,test,protocol['uncertainty']['count'],protocol['uncertainty']['seed'])
    paired,effects,wins=[],[],[]
    for first,ref in protocol['primary_comparisons']+protocol['secondary_comparisons']:
        for key in KEYS:
            sign=-1 if key.endswith('rmse') or key.endswith('mae') else 1
            delta=sign*(values[first][key]-values[ref][key]);dd=finite_mean(sign*(perdraw[first][key]-perdraw[ref][key]),axis=1)
            good=dd[np.isfinite(dd)];lo,hi=np.percentile(good,[2.5,97.5]) if len(good) else (np.nan,np.nan)
            paired.append(dict(assay=assay,first_model=first,reference_model=ref,metric=key,improvement=float(finite_mean(delta)),
                favorable_repeats=int(np.sum(delta>1e-12)),tied_repeats=int(np.sum(np.abs(delta)<=1e-12)),defined_repeats=int(np.isfinite(delta).sum()),conditional_ci_low=lo,conditional_ci_high=hi))
        favorable=0
        for label in np.unique(lin):
            a=per[(per.model==first)&(per.lineage==label)].sort_values('repeat');b=per[(per.model==ref)&(per.lineage==label)].sort_values('repeat')
            delta=b.rmse.to_numpy()-a.rmse.to_numpy();favorable+=int(delta.mean()>1e-12)
            effects.append(dict(assay=assay,first_model=first,reference_model=ref,lineage=str(label),cohort_n=int(a.cohort_n.iloc[0]),rmse_improvement=float(delta.mean())))
        wins.append(dict(assay=assay,first_model=first,reference_model=ref,favorable_lineages=favorable,lineages=len(np.unique(lin))))
    np.savez_compressed(c.OUT/assay/'bootstrap_draws.npz',**{f+'__'+k:v for f,d in draws.items() for k,v in d.items()})
    return [pd.DataFrame(summary),pd.DataFrame(paired),pd.DataFrame(effects),pd.DataFrame(wins),pooled,per,macro]


def gene_summary(occurrences):
    # Include an empty table safely if every fit chooses the core-only fallback.
    occ=pd.DataFrame(occurrences,columns=['assay','repeat','feature_index','gene','entrez_id','selector_coefficient','ridge_coefficient'])
    table('gene_selection_occurrences.tsv',occ)
    rows=[]
    for (assay,idx),g in occ.groupby(['assay','feature_index']):
        rows.append(dict(assay=assay,feature_index=int(idx),gene=g.gene.iloc[0],entrez_id=g.entrez_id.iloc[0],selected_fits=len(g),selection_fraction=len(g)/20,
            mean_abs_selector_coefficient=float(g.selector_coefficient.abs().mean()),selector_sign_consistency=float(max((g.selector_coefficient>0).mean(),(g.selector_coefficient<0).mean())),
            mean_abs_ridge_coefficient=float(g.ridge_coefficient.abs().mean()),ridge_sign_consistency=float(max((g.ridge_coefficient>0).mean(),(g.ridge_coefficient<0).mean()))))
    genes=pd.DataFrame(rows,columns=['assay','feature_index','gene','entrez_id','selected_fits','selection_fraction','mean_abs_selector_coefficient','selector_sign_consistency','mean_abs_ridge_coefficient','ridge_sign_consistency'])
    genes=genes.sort_values(['assay','selected_fits','mean_abs_selector_coefficient'],ascending=[True,False,False])
    table('gene_selection_stability.tsv',genes)
    cross=[]
    for idx,g in genes.groupby('feature_index'):
        d=dict(feature_index=int(idx),gene=g.gene.iloc[0],entrez_id=g.entrez_id.iloc[0])
        for a in c.ASSAYS:
            t=g[g.assay==a];d[a+'_selected_fits']=int(t.selected_fits.iloc[0]) if len(t) else 0
        d['min_assay_count']=min(d[a+'_selected_fits'] for a in c.ASSAYS);d['total_count']=sum(d[a+'_selected_fits'] for a in c.ASSAYS);cross.append(d)
    cross=pd.DataFrame(cross,columns=['feature_index','gene','entrez_id','CTRP_selected_fits','PRISM_selected_fits','min_assay_count','total_count']).sort_values(['min_assay_count','total_count','gene'],ascending=[False,False,True])
    table('cross_assay_candidate_genes.tsv',cross)
    return genes,cross


def render(summary,paired,wins,params,genes,cross):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    refs=['ridge64','enet64','nn64','tab64','ridge_genome','ridge_screened']
    fig,axes=plt.subplots(2,2,figsize=(12,7.5))
    for row,assay in enumerate(c.ASSAYS):
        for col,metric in enumerate(['pooled_rmse','pooled_spearman']):
            ax=axes[row,col]
            for j,ref in enumerate(refs):
                d=paired[(paired.assay==assay)&(paired.reference_model==ref)&(paired.metric==metric)].iloc[0]
                ax.plot([d.conditional_ci_low,d.conditional_ci_high],[j,j],color='#28768e',lw=2);ax.scatter(d.improvement,j,color='#173e57',s=25,zorder=3)
            ax.axvline(0,color='gray',ls='--',lw=1);ax.set_yticks(range(len(refs)),[LABELS[r] for r in refs] if col==0 else ['']*len(refs));ax.invert_yaxis()
            ax.set_title(assay+' | '+('RMSE reduction' if col==0 else 'Spearman increase'));ax.set_xlabel('Elastic Net additions: improvement over reference')
            ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
            ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Genome-wide Elastic Net selection + retained core64: ridge prediction',fontsize=13)
    fig.tight_layout(rect=(0,.045,1,.95));fig.text(.5,.01,'Conditional 95% paired lineage-bootstrap intervals; all 20 overlapping splits retained together. Exploratory.',ha='center',fontsize=9)
    fig.savefig(c.OUT/'model_comparison.png',dpi=180);fig.savefig(c.OUT/'model_comparison.svg');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,6))
    for ax,assay in zip(axes,c.ASSAYS):
        d=genes[genes.assay==assay].head(15).iloc[::-1]
        ax.barh(d.gene,d.selected_fits,color='#28768e');ax.set_xlim(0,21);ax.set_xticks([0,5,10,15,20]);ax.set_xlabel('Selected outer-training fits /20');ax.set_title(assay);ax.grid(axis='x',alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Additional gene selection stability\nTraining-only associations; overlapping fits are not independent replications',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.92));fig.savefig(c.OUT/'gene_selection_stability.png',dpi=180);plt.close(fig)
    lines=['# Genome-wide Elastic Net additions to the retained 64-gene panel','',
        'Elastic Net selects outside genes jointly after adjusting for the core64 and training lineage. Ridge then fits the mandatory core64 plus selected additions. All 19,112 outside protein-coding genes are considered before current-training constants/aliases are excluded. There is no preliminary response-correlation shortlist. This experiment evaluates a gene-selection procedure, not a direct genome-wide Elastic Net predictor.','',
        '## Findings','']
    primary=paired[(paired.reference_model=='ridge64')&(paired.metric=='pooled_rmse')]
    if not (primary.conditional_ci_low>0).any():
        lines += ['**Retain the original core64 panel. This Elastic Net selection procedure did not establish improved pooled accuracy in either assay.** Selected genes remain discovery associations; the experiment does not demonstrate that any individual added gene improves prediction.','']
    for assay in c.ASSAYS:
        new=summary[(summary.assay==assay)&(summary.model=='enet_selected_ridge')].iloc[0]
        ref=summary[(summary.assay==assay)&(summary.model=='ridge64')].iloc[0]
        d=paired[(paired.assay==assay)&(paired.reference_model=='ridge64')&(paired.metric=='pooled_rmse')].iloc[0]
        t=params[params.assay==assay]
        support='includes no difference' if d.conditional_ci_low<=0<=d.conditional_ci_high else 'favors improvement' if d.conditional_ci_low>0 else 'favors worse performance'
        direction='lower' if d.improvement>0 else 'higher'
        lines += [f'**{assay}:** mean RMSE {ref.pooled_rmse_mean:.6f} -> {new.pooled_rmse_mean:.6f}; {abs(100*d.improvement/ref.pooled_rmse_mean):.2f}% {direction} error. {d.favorable_repeats}/20 favorable repeats; {d.tied_repeats}/20 ties. Paired RMSE reduction {d.improvement:+.6f}, conditional 95% interval [{d.conditional_ci_low:+.6f}, {d.conditional_ci_high:+.6f}], which {support}.',
            f'Added genes: mean {t.added_genes.mean():.1f}, median {t.added_genes.median():.1f}, range {t.added_genes.min()}-{t.added_genes.max()}; core-only fallback {(t.added_genes==0).sum()}/20 fits.','']
        macro=paired[(paired.assay==assay)&(paired.reference_model=='ridge64')&(paired.metric=='macro_all_rmse')].iloc[0]
        win=wins[(wins.assay==assay)&(wins.reference_model=='ridge64')].iloc[0]
        lines += [f'Equal-lineage RMSE: {ref.macro_all_rmse_mean:.6f} -> {new.macro_all_rmse_mean:.6f}; paired reduction interval [{macro.conditional_ci_low:+.6f}, {macro.conditional_ci_high:+.6f}]. Mean RMSE is lower in {win.favorable_lineages}/{win.lineages} lineages. These secondary results do not establish broad improvement.','']
    lines += ['## Held-out performance','','| Assay | Model | RMSE | Spearman | R2 | Equal-lineage RMSE |','|---|---|---:|---:|---:|---:|']
    for assay in c.ASSAYS:
        for f in LABELS:
            d=summary[(summary.assay==assay)&(summary.model==f)].iloc[0]
            rho=f'{d.pooled_spearman_mean:.4f}' if np.isfinite(d.pooled_spearman_mean) else 'undefined'
            lines.append(f'| {assay} | {LABELS[f]} | {d.pooled_rmse_mean:.6f} | {rho} | {d.pooled_r2_mean:.4f} | {d.macro_all_rmse_mean:.6f} |')
    lines += ['','Scores are means across the saved 20 holdouts. CTRP and PRISM endpoints remain separate.','',
        '![Paired model comparisons](model_comparison.png)','','## Additional genes','','The following examples rank by selection frequency, then mean absolute selector coefficient. Coefficients and sign consistency are conditional on selection; these are not causal importance or independent replications. The assays share cell lines.','']
    for assay in c.ASSAYS:
        lines += ['### '+assay,'','| Gene | Entrez | Selected fits /20 | Selector sign agreement | Ridge sign agreement |','|---|---|---:|---:|---:|']
        for _,d in genes[genes.assay==assay].head(15).iterrows():
            lines.append(f'| {d.gene} | {d.entrez_id} | {d.selected_fits} | {d.selector_sign_consistency:.0%} | {d.ridge_sign_consistency:.0%} |')
        lines.append('')
    lines += ['![Additional gene stability](gene_selection_stability.png)','','Full tables: `gene_selection_stability.tsv`, `cross_assay_candidate_genes.tsv`, `selected_parameters.tsv`, `lineage_paired_effects.tsv`, and `paired_comparisons.tsv`.','',
        '## Validation and interpretation','',
        '- Same cohorts, response values, identity groups, 20 outer holdouts and five inner folds as the retained benchmark; all evaluated lineages appear in both outer train/test. Cervix exclusion and singleton inner-training handling are unchanged.',
        '- All centering, scaling, core/lineage adjustment, alpha maxima and multigene feature selection are recomputed inside each current training fold. Selector L1 ratio, alpha fraction and ridge penalty are selected by mean five-fold inner RMSE. A zero-addition candidate family prevents forced expansion.',
        '- Every feature is considered by the multivariate objective. Working sets only accelerate coordinate descent; complete genome-wide KKT checks prevent permanent exclusion by a preliminary screen. Native full-genome solver comparisons, dual certificates, augmented-Lasso equivalence and independent ridge checks passed.',
        '- Saved outer predictions, all inner RMSEs/choices, all outer selector KKT conditions, core retention, training scalers, recorded inner solver certificates, selected-fold replay and source hashes are audited. The solver table distinguishes independently recomputed outer certificates from recorded inner certificates.',
        '- Some restricted solves require the prespecified tighter-tolerance retry. Recovered warnings and refinement counts are recorded in `validation.json`; every accepted final solution passes its declared global KKT/dual-gap limits.',
        '- Whole-lineage paired bootstrap resamples lineages while keeping models/repeats together. Intervals are conditional on fitted models, do not include retraining and are not multiplicity-adjusted. Twenty overlapping repeats are not 20 independent experiments.',
        '- These previously inspected cohorts support exploratory discovery, not untouched confirmatory accuracy. Reported scores belong to the fold-specific selection procedure; a final consensus panel needs separate validation. No individual gene is proven to add predictive value merely because it is often selected.',
        '- Selection can still miss nonlinear interaction-only effects. The finite penalty grid, RNA-only scope and original harmonized CTRP endpoint provenance limitations remain. The original core64 and retained-model registry are unchanged.',
        '- Read `grid_boundary_audit.json` for selected-grid limits. The frozen grid is not expanded after seeing test results.','',
        'Methods: [scikit-learn Elastic Net path](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.enet_path.html); [Ambroise and McLachlan on gene-selection bias](https://pmc.ncbi.nlm.nih.gov/articles/PMC124442/).','']
    (c.OUT/'REVIEW.md').write_text('\n'.join(lines))


def seal():
    c.verify();assert json.load(open(c.OUT/'validation.json'))['passed'];assert json.load(open(c.OUT/'visual_qa.json'))['passed']
    for n,h in json.load(open(c.OUT/'analysis_source_manifest.json'))['scripts'].items():assert c.sha(c.OUT/n)==h
    c.dump(c.OUT/'progress.json',dict(phase='complete',validation_passed=True))
    files=[dict(path=str(p.relative_to(c.OUT)),sha256=c.sha(p),bytes=p.stat().st_size) for p in sorted(c.OUT.rglob('*'))
        if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts and p.suffix!='.log']
    c.dump(c.OUT/'artifact_manifest.json',dict(files=files));print('SEALED',len(files),flush=True)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--wait',action='store_true');parser.add_argument('--seal',action='store_true');args=parser.parse_args()
    if args.seal:seal();return
    c.initialize();c.verify();protocol=json.load(open(c.OUT/'protocol.json'))
    c.dump(c.OUT/'analysis_source_manifest.json',dict(scripts={n:c.sha(c.OUT/n) for n in ['review_comparison.py','evaluate.py','metric_helpers.py']}))
    results=[];params=[];occ=[];solver=[];valid=[]
    for assay in c.ASSAYS:
        if args.wait:
            print('WAITING_FOR',assay,flush=True)
            while len(list((c.OUT/assay/'folds').glob('repeat_*_complete.json')))<20:time.sleep(5)
        preds,pp,oo,ss,vv=audit(assay);results.append(evaluate(assay,preds,protocol));params+=pp;occ+=oo;solver+=ss;valid.append(vv)
        print('ANALYSIS_COMPLETE',assay,flush=True)
    names=['summary.tsv','paired_comparisons.tsv','lineage_paired_effects.tsv','lineage_win_counts.tsv','per_repeat_pooled_metrics.tsv','per_lineage_metrics.tsv','per_repeat_macro_metrics.tsv']
    frames=[table(n,pd.concat([d[j] for d in results],ignore_index=True)) for j,n in enumerate(names)]
    params=table('selected_parameters.tsv',params);table('elastic_net_solver_checks.tsv',solver)
    table('parameter_counts.tsv',params.groupby(['assay','selector','l1_ratio','alpha_fraction','ridge_alpha'],dropna=False).size().reset_index(name='outer_fits'))
    genes,cross=gene_summary(occ)
    boundary=[]
    for assay in c.ASSAYS:
        d=params[params.assay==assay];active=d[d.selector>=0]
        boundary.append(dict(assay=assay,core_only=int((d.added_genes==0).sum()),min_fraction=int((active.alpha_fraction==min(models.FRACTIONS)).sum()),
            max_fraction=int((active.alpha_fraction==max(models.FRACTIONS)).sum()),min_l1=int((active.l1_ratio==min(models.L1_RATIOS)).sum()),max_l1=int((active.l1_ratio==max(models.L1_RATIOS)).sum()),
            min_ridge_alpha=int((d.ridge_alpha==min(models.ALPHAS)).sum()),max_ridge_alpha=int((d.ridge_alpha==max(models.ALPHAS)).sum())))
    c.dump(c.OUT/'grid_boundary_audit.json',dict(records=boundary,post_test_grid_expansion=False))
    c.dump(c.OUT/'validation.json',dict(passed=True,assays=valid,preflight=json.load(open(c.OUT/'preflight_validation.json')),core64_retained=True,training_only_selection=True,sources_verified=True))
    render(frames[0],frames[1],frames[3],params,genes,cross)
    c.verify();c.dump(c.OUT/'final_summary.json',dict(validation_passed=True,summary=frames[0].to_dict('records'),candidate_genes=cross.head(30).to_dict('records')))
    c.dump(c.OUT/'progress.json',dict(phase='analysis_complete_pending_visual_qa',validation_passed=True));print('ALL_ANALYSIS_COMPLETE',flush=True)


if __name__=='__main__':main()
