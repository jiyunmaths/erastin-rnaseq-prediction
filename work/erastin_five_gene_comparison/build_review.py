from pathlib import Path
s=Path('outputs/erastin_stratified_panel_comparison/review_panels.py').read_text()
s=s.replace('Audit all nine unweighted arms, including exact 64-gene NN reuse.','Audit fixed five-gene and 64-gene ridge/NN models on shared splits.')
s=s.replace('import joblib,numpy as np,pandas as pd,xgboost as xgb','import joblib,numpy as np,pandas as pd')
s=s.replace('import run_panels as run','import run_comparison as run')
s=s.replace('from validate_preflight import native_nn','from validate_preflight import independent')
a=s.index('def pairs():');b=s.index('\ndef wait(',a)
s=s[:a]+'''def pairs():
    protocol=json.loads((OUT/'protocol.json').read_text());v=protocol['primary_comparisons']+protocol['secondary_comparisons'];assert len(v)==len({tuple(x) for x in v})==6;return v
''' + s[b:]
s=s.replace("assert m['reused']==(f=='neural_network' and panel==64)","assert m['reused']==(panel==64)")
s=s.replace("ext='ubj' if f=='xgboost' else 'joblib'","ext='joblib'")
a=s.index("                if f=='xgboost':");b=s.index('                maxp=max(',a)
s=s[:a]+'''                model=joblib.load(root.with_suffix('.joblib'));sx=StandardScaler().fit(x[tr])
                if f=='ridge':
                    np.testing.assert_allclose(model['mean'],sx.mean_,atol=1e-12,rtol=0);np.testing.assert_allclose(model['scale'],sx.scale_,atol=1e-12,rtol=0);assert model['alpha']==p['alpha'] and model['intercept_after_scaling']==y[tr].mean();pred=sx.transform(x[te])@model['coef']+model['intercept_after_scaling']
                else:
                    sy=StandardScaler().fit(y[tr,None]);target=sy.transform(y[tr,None]).ravel();z=sx.transform(x[tr])
                    for key in ['mean_','scale_']:
                        np.testing.assert_allclose(getattr(model['x_scaler'],key),getattr(sx,key),atol=1e-12,rtol=0);np.testing.assert_allclose(getattr(model['y_scaler'],key),getattr(sy,key),atol=1e-12,rtol=0)
                    assert [v.random_state for v in model['members']]==m['neural_seeds']
                    members=np.array([sy.inverse_transform(v.predict(sx.transform(x[te]))[:,None]).ravel() for v in model['members']]);np.testing.assert_allclose(members,a['member_test_predictions'],atol=1e-12,rtol=0);pred=members.mean(axis=0)
                    for j,v in enumerate(model['members']):
                        np.testing.assert_array_equal(v.training_sample_weight_,np.ones(len(tr)));assert v.alpha==p['alpha'] and v.hidden_layer_sizes==(p['width'],)
                        obj=.5*np.mean((v.predict(z)-target)**2)+.5*v.alpha*sum(np.sum(t*t) for t in v.coefs_)/len(tr);err=float(abs(obj-v.loss_));maxobj=max(maxobj,err);assert err<1e-10;objectives.append(dict(model=arm,repeat=r+1,member=j,loss_recalculation_error=err,iterations=v.n_iter_))
''' + s[b:]
s=s.replace("if 'warning_category' in d:warnings.append(dict(model=arm,repeat=r+1,category=d['warning_category'],message=d['warning']))\n                    ","")
s=s.replace(",int(s['model_seeds'][r]))", ")")
s=s.replace('total=180','total=80').replace('reused==20 and groups_checked==1080 and scores_checked==17100 and len(references)==18','reused==40 and groups_checked==480 and scores_checked==4200 and len(references)==8')
s=s.replace("previous=np.load(WS/'outputs/erastin_lineage_stratified_nn/predictions.npz');np.testing.assert_array_equal(preds['neural_network__64'],previous['unweighted'])", "previous=np.load(WS/'outputs/erastin_stratified_panel_comparison/predictions.npz')\n    for f in run.FAMILIES:np.testing.assert_array_equal(preds[run.arm_name(f,64)],previous[run.arm_name(f,64)])")
s=s.replace('seed=2026110704','seed=2026110904')
s=s.replace('total_outer_models=180,new_outer_models=160,reused_nn64_outer_models=reused','total_outer_models=80,new_outer_models=40,reused_64gene_outer_models=reused')
s=s.replace("byte_identical_shared_splits_verified=True)","byte_identical_shared_splits_verified=True,exploratory_post_selection_comparison=True)")
s=s.replace("original=pd.read_csv(WS/'outputs/erastin_lineage_stratified_nn/per_repeat_pooled_metrics.tsv',sep='\\t');np.testing.assert_allclose(pooled[pooled.model=='neural_network__64'].sort_values('repeat')[['spearman','rmse','r2']],original[original.model=='unweighted'].sort_values('repeat')[['spearman','rmse','r2']],atol=1e-12,rtol=0)","original=pd.read_csv(WS/'outputs/erastin_stratified_panel_comparison/per_repeat_pooled_metrics.tsv',sep='\\t')\n    for f in run.FAMILIES:\n        arm=run.arm_name(f,64);np.testing.assert_allclose(pooled[pooled.model==arm].sort_values('repeat')[['spearman','rmse','r2']],original[original.model==arm].sort_values('repeat')[['spearman','rmse','r2']],atol=1e-12,rtol=0)")
s=s.replace("audited_models=180","audited_models=80")
s=s.replace("from report_panels import render","from report_comparison import render")
s=s.replace("report_panels.py","report_comparison.py")
s=s.replace('new_outer_models=160,reused_outer_models=20,total_outer_models=180','new_outer_models=40,reused_outer_models=40,total_outer_models=80').replace('total_outer_models=180','total_outer_models=80')
s=s.replace('erastin_stratified_panel_comparison_results.zip','erastin_five_gene_comparison_results.zip').replace("'erastin_stratified_panel_comparison/'+row['path']","'erastin_five_gene_comparison/'+row['path']").replace("'erastin_stratified_panel_comparison/artifact_manifest.json'","'erastin_five_gene_comparison/artifact_manifest.json'")
s=s.replace("    pcounts=[]",'''    agreement=[]
    from scipy.stats import spearmanr
    for f in run.FAMILIES:
        for r,mask in enumerate(s['test']):
            a=preds[run.arm_name(f,5)][r,mask];b=preds[run.arm_name(f,64)][r,mask]
            agreement.append(dict(family=f,repeat=r+1,pearson=float(np.corrcoef(a,b)[0,1]),spearman=float(spearmanr(a,b).statistic),prediction_rmse=float(np.sqrt(np.mean((a-b)**2)))))
    table('prediction_agreement.tsv',agreement)
    pcounts=[]''')
Path('outputs/erastin_five_gene_comparison/review_comparison.py').write_text(s)
