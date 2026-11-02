from pathlib import Path
out=Path('outputs/erastin_lineage_weighted')
s=Path('outputs/erastin_panel_comparison/review_panels.py').read_text()
s=s.replace('"""Audit all expanded-panel fits and evaluate the nine matched prediction arms."""','"""Audit all weighted fits and compare with the nine original unweighted arms."""')
s=s.replace('from sklearn.neural_network import MLPRegressor','from weighted_mlp import WeightedMLPRegressor as MLPRegressor, inverse_lineage_weights')
s=s.replace('import run_panels as run','import run_weighted as run')
s=s.replace("def name(f,p):return f+'__'+str(p)","def name(f,p,weighted=True):return ('weighted__' if weighted else 'unweighted__')+f+'__'+str(p)")
a=s.index('def label(arm):');b=s.index('\ndef table',a)
s=s[:a]+'''def label(arm):
    weight,f,p=arm.split('__');return LABELS[f]+' / '+p+' genes / '+weight
''' + s[b:]
s=s.replace("    if f=='ridge':\n        sx", "    w=inverse_lineage_weights(run.STATE['lineage'][a])\n    if f=='ridge':\n        sx")
s=s.replace('.fit(sx.transform(x[a]),y[a])','.fit(sx.transform(x[a]),y[a],sample_weight=w)')
s=s.replace('.fit(x[a],target)', '.fit(x[a],target,sample_weight=w)')
s=s.replace('m.fit(sx.transform(x[a]),target)','m.fit(sx.transform(x[a]),target,sample_weight=w)')
a=s.index('def planned_pairs():');b=s.index('\ndef audit():',a)
s=s[:a]+'''def planned_pairs():
    pairs=[(name(f,p),name(f,p,False)) for f in FAMILIES for p in SIZES]
    for f in FAMILIES:pairs.extend([(name(f,64),name(f,40)),(name(f,80),name(f,40)),(name(f,80),name(f,64))])
    for p in SIZES:pairs.extend([(name('neural_network',p),name('ridge',p)),(name('xgboost',p),name('ridge',p)),(name('xgboost',p),name('neural_network',p))])
    assert len(pairs)==len(set(pairs))==27
    return pairs
''' + s[b:]
s=s.replace("run.sha(OUT/'run_panels.py')", "run.sha(OUT/'run_weighted.py')")
s=s.replace("    for source in json.loads((OUT/'source_manifest.json').read_text())['files']:assert", "    assert run.sha(OUT/'weighted_mlp.py')==protocol['weighted_mlp_sha256']\n    for source in json.loads((OUT/'source_manifest.json').read_text())['files']:assert",1)
a=s.index("    base=np.load(");b=s.index('    parameters=[]',a)
s=s[:a]+'''    base=np.load(WS/'outputs/erastin_panel_comparison/predictions.npz')
    for key,current in [('sample_ids',s['ids']),('group_ids',groups),('lineage',lineage),('y',y),('inner_split_seeds',s['seeds'])]:np.testing.assert_array_equal(base[key],current)
    preds={name(f,p):np.full((20,745),np.nan) for f in FAMILIES for p in SIZES}
    preds.update({name(f,p,False):base[f+'__'+str(p)].copy() for f in FAMILIES for p in SIZES})
    weight_rows=[];weight_group_rows=[]
''' + s[b:]
s=s.replace('for panel in [64,80]:','for panel in SIZES:')
s=s.replace("                    assert m['protocol_sha256']==ph", "                    outer_w=inverse_lineage_weights(lineage[tr])\n                    assert m['outer_weights']==run.weight_audit(lineage[tr])\n                    if panel==40 and f=='ridge':record_weights(weight_rows,weight_group_rows,r+1,str(heldout),'outer',-1,m['outer_weights'])\n                    assert m['protocol_sha256']==ph")
s=s.replace("                        loss=np.sqrt", "                        assert m['inner_scaling'][k]['weights']==run.weight_audit(lineage[aa])\n                        if panel==40 and f=='ridge':record_weights(weight_rows,weight_group_rows,r+1,str(heldout),'inner',k,m['inner_scaling'][k]['weights'])\n                        loss=np.sqrt")
s=s.replace("                            assert model['alpha']==params['alpha'] and model['intercept_after_scaling']==y[tr].mean()", "                            assert model['alpha']==params['alpha']\n                            np.testing.assert_array_equal(model['training_sample_weight'],outer_w)\n                            np.testing.assert_allclose(model['weighted_y_mean'],np.average(y[tr],weights=outer_w),rtol=0,atol=1e-14)\n                            np.testing.assert_allclose(model['weighted_x_mean_after_scaling'],np.average(sx.transform(x[tr]),axis=0,weights=outer_w),rtol=0,atol=1e-14)")
s=s.replace("                            assert [v.random_state", "                            for v in model['members']:np.testing.assert_array_equal(v.training_sample_weight_,outer_w)\n                            assert [v.random_state")
s=s.replace("assert len(references)==24", "assert len(references)==36")
s=s.replace('480*2*(17+4+36)*5','480*3*(17+4+36)*5').replace('2880*6','4320*6').replace('2880','4320')
s=s.replace("    parameters=table('selected_parameters.tsv',parameters);", "    table('training_subset_weight_audit.tsv',weight_rows);table('training_lineage_weights.tsv',weight_group_rows)\n    parameters=table('selected_parameters.tsv',parameters);")
s=s.replace('reused_40_gene_models=1440','reused_unweighted_prediction_arms=9')
s=s.replace("baseline_replay=json.loads((OUT/'baseline_replay_validation.json').read_text()),", "preflight=json.loads((OUT/'preflight_validation.json').read_text()),\n                    training_subset_weight_checks=4320*6,all_training_lineages_equal_total_weight=True,\n                    neural_refit_note='NN deterministic refits use the same weighted adapter; independent finite-difference gradients and direct-objective checks are in preflight. Ridge/XGBoost reference refits use separate sklearn APIs.',")
a=s.index("    original=pd.read_csv(");b=s.index("    table('per_repeat_metrics",a)
s=s[:a]+'''    original=pd.read_csv(WS/'outputs/erastin_panel_comparison/per_repeat_metrics.tsv',sep='\\t')
    for f in FAMILIES:
        for p in SIZES:
            np.testing.assert_allclose(scores[scores.model==name(f,p,False)].sort_values('repeat')[METRICS],
                original[original.model==f+'__'+str(p)].sort_values('repeat')[METRICS],rtol=0,atol=1e-12)
''' + s[b:]
s=s.replace("            f,p=arm.rsplit('__',1);row=dict(scope=scope,model=arm,family=f,panel=int(p))", "            weighting,f,p=arm.split('__');row=dict(scope=scope,model=arm,weighting=weighting,family=f,panel=int(p))")
a=s.index('def render(')
s=s[:a]+'''def render(summary,paired,effects,wins,counts,validation):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import StrMethodFormatter
    colors={'ridge':'#137d8b','neural_network':'#8853a3','xgboost':'#bf6d22'}
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True)
    for i,scope in enumerate(['pooled_745','macro_19']):
        for j,metric in enumerate(['spearman','rmse']):
            ax=axes[i,j]
            for f in FAMILIES:
                for weight,style in [('unweighted','--'),('weighted','-')]:
                    v=summary[(summary.scope==scope)&(summary.family==f)&(summary.weighting==weight)].sort_values('panel')
                    ax.errorbar(v.panel,v[metric+'_mean'],yerr=v[metric+'_sd'],marker='o',capsize=2,color=colors[f],linestyle=style,label=LABELS[f]+' / '+weight)
            ax.set_xticks(SIZES);ax.set_xlabel('Number of genes');ax.set_ylabel('Spearman ↑' if metric=='spearman' else 'RMSE ↓')
            ax.set_title('Pooled 745 cell lines' if i==0 else 'Equal mean across 19 lineages (n≥10)')
            ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
            if metric=='rmse':ax.yaxis.set_major_formatter(StrMethodFormatter('{x:.4f}'))
    handles,names=axes[0,0].get_legend_handles_labels();fig.legend(handles,names,loc='outside lower center',ncol=3,fontsize=9)
    fig.suptitle('Inverse-lineage-frequency training weights: 20 matched repeats\\nWhiskers show seed SD on the same biological cohort.',fontsize=13)
    fig.savefig(OUT/'weighted_comparison.png',dpi=170);fig.savefig(OUT/'weighted_comparison.svg');plt.close(fig)
    selected=effects[effects.n>=10];lineages=selected.drop_duplicates('lineage').sort_values('n',ascending=False).lineage.tolist()
    fig,axes=plt.subplots(1,2,figsize=(14,9),constrained_layout=True,sharey=True)
    contrasts=[(name(f,p),name(f,p,False)) for f in FAMILIES for p in SIZES]
    for ax,metric in zip(axes,['spearman','rmse']):
        a=np.column_stack([selected[(selected.first_model==first)&(selected.reference_model==reference)].set_index('lineage').loc[lineages,metric+'_improvement'] for first,reference in contrasts])
        limit=float(np.max(np.abs(a)));im=ax.imshow(a,aspect='auto',cmap='RdBu',vmin=-limit,vmax=limit)
        ax.set_xticks(range(9),[LABELS[f]+'\\n'+str(p) for f in FAMILIES for p in SIZES],rotation=55,ha='right',fontsize=8)
        ax.set_title('Spearman increase' if metric=='spearman' else 'RMSE reduction');fig.colorbar(im,ax=ax,shrink=.75,pad=.02)
    axes[0].set_yticks(range(19),lineages)
    fig.suptitle('Weighted minus original unweighted performance on unseen lineages\\nMean paired effect across 20 repeats; blue favors weighting',fontsize=13)
    fig.savefig(OUT/'lineage_weighting_effects.png',dpi=170);fig.savefig(OUT/'lineage_weighting_effects.svg');plt.close(fig)
    write_report(summary,paired,effects,wins,counts,validation)

def record_weights(rows,group_rows,repeat,holdout,stage,inner_fold,audit):
    context=dict(repeat=repeat,holdout_lineage=holdout,stage=stage,inner_fold=inner_fold)
    rows.append(dict(**context,**{k:v for k,v in audit.items() if k!='groups'}))
    group_rows.extend(dict(**context,**v) for v in audit['groups'])

def write_report(summary,paired,effects,wins,counts,validation):
    from report_weighted import write_report as writer
    writer(summary,paired,effects,wins,counts,validation)

if __name__=='__main__':
    predictions,parameters,validation,protocol=audit();evaluate(predictions,parameters,validation,protocol)
'''
s=s.replace("archive=OUT.parent/'erastin_panel_comparison_results.zip'","archive=OUT.parent/'erastin_lineage_weighted_results.zip'").replace("'erastin_panel_comparison/'+row['path']","'erastin_lineage_weighted/'+row['path']").replace("'erastin_panel_comparison/artifact_manifest.json'","'erastin_lineage_weighted/artifact_manifest.json'")
(out/'review_weighted.py').write_text(s)
