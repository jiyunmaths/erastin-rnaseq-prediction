from pathlib import Path
out=Path('outputs/erastin_lineage_weighted')
s=Path('outputs/erastin_panel_comparison/run_panels.py').read_text()
s=s.replace('"""Frozen 40/64/80-gene comparison using the exact prior nested splits."""','"""Inverse-lineage-frequency training-loss comparison on frozen nested splits."""')
s=s.replace('import xgboost as xgb','import xgboost as xgb\nfrom scipy.linalg import svd\nfrom sklearn.preprocessing import StandardScaler\nfrom weighted_mlp import WeightedMLPRegressor, inverse_lineage_weights')
a=s.index('def matrix(');b=s.index('\ndef fit_job',a)
s=s[:a]+'''def weight_audit(lineages):
    w=inverse_lineage_weights(lineages);labels,counts=np.unique(lineages,return_counts=True)
    return dict(n=len(w),represented_lineages=len(labels),weight_min=float(w.min()),weight_max=float(w.max()),
        weight_sum=float(w.sum()),effective_sample_size=float(w.sum()**2/np.dot(w,w)),
        groups=[dict(lineage=str(label),n=int(n),weight_per_cell=float(len(w)/(len(labels)*n)),
            total_weight=float(w[lineages==label].sum())) for label,n in zip(labels,counts)])

def matrix(x,genes,y=None,w=None):
    return xgb.DMatrix(x,label=y,weight=w,feature_names=list(genes),nthread=1)

def ridge_path(x,y,tr,te,alphas,retain=False,weights=None):
    w=inverse_lineage_weights(STATE['lineage'][tr]) if weights is None else weights
    sx=StandardScaler().fit(x[tr]);a=sx.transform(x[tr]);b=sx.transform(x[te])
    mean_x=np.average(a,axis=0,weights=w);mean_y=float(np.average(y[tr],weights=w));sw=np.sqrt(w)
    u,s,vt=svd((a-mean_x)*sw[:,None],full_matrices=False,check_finite=False,lapack_driver='gesdd')
    factors=(s[:,None]/(s[:,None]**2+np.asarray(alphas)[None,:]))*(u.T@((y[tr]-mean_y)*sw))[:,None]
    p=((b-mean_x)@vt.T)@factors+mean_y;saved=None
    if retain:
        coef=vt.T@factors[:,0];intercept=mean_y-mean_x@coef
        saved=dict(mean=sx.mean_,scale=sx.scale_,coef=coef,intercept_after_scaling=intercept,alpha=float(alphas[0]),
            weighted_x_mean_after_scaling=mean_x,weighted_y_mean=mean_y,training_sample_weight=w)
        np.testing.assert_allclose(b@coef+intercept,p[:,0],atol=1e-12,rtol=0)
    return p,saved

def fitted_neural(params,x,y,tr,te,seeds,retain=False,weights=None):
    begin=time.perf_counter();w=inverse_lineage_weights(STATE['lineage'][tr]) if weights is None else weights
    sx=StandardScaler().fit(x[tr]);sy=StandardScaler().fit(y[tr,None])
    a=sx.transform(x[tr]);b=sx.transform(x[te]);target=sy.transform(y[tr,None]).ravel()
    members=[];predictions=[];diagnostics=[]
    for seed in seeds:
        model=WeightedMLPRegressor(hidden_layer_sizes=(params['width'],),activation='tanh',solver='lbfgs',
            alpha=params['alpha'],max_iter=2000,max_fun=40000,tol=.001,early_stopping=False,random_state=int(seed))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always');model.fit(a,target,sample_weight=w)
        prediction=sy.inverse_transform(model.predict(b)[:,None]).ravel()
        assert np.isfinite(prediction).all();predictions.append(prediction)
        diagnostics.append(dict(seed=int(seed),iterations=model.n_iter_,training_loss=model.loss_,
            warnings=[dict(category=v.category.__name__,message=str(v.message)) for v in caught]))
        if retain:members.append(model)
    pp=np.array(predictions);saved=dict(family='neural_network',x_scaler=sx,y_scaler=sy,members=members) if retain else None
    return pp.mean(axis=0),saved,dict(seconds=time.perf_counter()-begin,members=diagnostics),pp
''' + s[b:]
s=s.replace('ph,replay=False','ph')
s=s.replace("folder=OUT/('replay' if replay else 'folds');folder.mkdir(exist_ok=True)","folder=OUT/'folds';folder.mkdir(exist_ok=True)")
s=s.replace('scaling.append(dict(inner_fold=k,y_mean=float(y[a].mean()),y_scale=float(y[a].std())))',"scaling.append(dict(inner_fold=k,y_mean=float(y[a].mean()),y_scale=float(y[a].std()),weights=weight_audit(STATE['lineage'][a])))")
s=s.replace('cm.ridge_path','ridge_path')
s=s.replace("nn.fitted_prediction(family,params,x,y,a,b)","fitted_neural(params,x,y,a,b,nn.SEEDS)")
s=s.replace("nn.fitted_prediction(family,params,x,y,tr,te,retain=True)","fitted_neural(params,x,y,tr,te,nn.SEEDS,retain=True)")
s=s.replace('matrix(x[a],genes,(y[a]-mu)/sd)',"matrix(x[a],genes,(y[a]-mu)/sd,inverse_lineage_weights(STATE['lineage'][a]))")
s=s.replace('matrix(x[tr],genes,(y[tr]-mu)/sd)',"matrix(x[tr],genes,(y[tr]-mu)/sd,inverse_lineage_weights(STATE['lineage'][tr]))")
s=s.replace('mean_inner_rmse=float(loss[c].mean()),inner_scaling=scaling,fit_diagnostics=diagnostics,',"mean_inner_rmse=float(loss[c].mean()),inner_scaling=scaling,outer_weights=weight_audit(STATE['lineage'][tr]),fit_diagnostics=diagnostics,")
a=s.index('def prepare():');b=s.index('\ndef main():',a)
s=s[:a]+'''def prepare():
    OUT.mkdir(exist_ok=True);base=WS/'outputs/erastin_panel_comparison'
    known={r['path']:r['sha256'] for r in json.loads((base/'artifact_manifest.json').read_text())['files']}
    sources=[]
    for name in ['panel_inputs.npz','shared_splits.npz','protocol.json','predictions.npz','summary.tsv']:
        assert sha(base/name)==known[name]
        sources.append(dict(path=str((base/name).relative_to(WS)),sha256=known[name]))
    for path in [WS/'outputs/erastin_xgboost_comparison/run_comparison.py',WS/'outputs/erastin_nonlinear_comparison/run_models.py',
                 WS/'outputs/erastin_model_comparison/compare_models.py',WS/'outputs/erastin_bayesian_comparison/review_bayesian.py',
                 WS/'outputs/erastin_repeated_cv/repeat_cv.py']:
        sources.append(dict(path=str(path.relative_to(WS)),sha256=sha(path)))
    prior=json.loads((base/'protocol.json').read_text())
    protocol={k:prior[k] for k in ['panels','samples','identity_groups','repeats','lineage_holdouts','inner_folds','split_seeds',
        'neural_seed_triplets','xgboost_seeds','ridge_alphas','neural_grid','neural_fixed','xgboost_grid','xgboost_fixed','versions']}
    protocol.update(status='frozen_before_weighted_fits',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
        script_sha256=sha(Path(__file__)),weighted_mlp_sha256=sha(OUT/'weighted_mlp.py'),
        weighting='Every current fitting subset: w_i=N_train/(K_represented_lineages*n_lineage_in_that_subset). Mean one; each represented lineage has equal total weight. Recomputed independently in each inner fold and outer refit. All represented lineages included.',
        neural_implementation='Pinned sklearn 1.5.2 MLPRegressor with weighted squared-error data loss/output delta; unchanged architecture, optimizer, initialization and L2 scaling. Unit-weight replay and finite-difference gradient checks precede fitting.',
        tuning='Unchanged minimum equal-weight mean of five UNWEIGHTED inner-fold RMSE values; first grid member breaks ties. This isolates training-loss reweighting.',
        scaling='UNCHANGED unweighted train-only X scaling for ridge/NN and unweighted train-only y scaling for NN/XGBoost. Ridge uses weighted centering to fit an unpenalized intercept.',
        controls='Exact saved samples, targets, inner/outer splits, initialization/sampling seeds, grids and settings across all weighted/unweighted arms. No test-based tuning changes.',
        baseline='Reuse nine verified unweighted prediction arms from the completed panel comparison. All nine weighted arms are newly refit.',
        comparisons='Nine matched weighted-versus-unweighted contrasts; within weighted families 64 vs 40, 80 vs 40, 80 vs 64; within weighted panels cross-family contrasts.',
        uncertainty=dict(count=5000,pooled_seed=2026110401,macro_seed=2026110402,unit='Paired whole held-out lineage across all arms/repeats. Conditional on fitted predictions, no refitting.'),
        reporting='Pooled 745-cell scores and equal-weight macro scores over the same 19 lineages with n>=10. All 20 repeats retained. Unweighted test scoring remains unchanged.',
        limitations='Exploratory reuse of this cohort; no independent validation or multiplicity adjustment. Very small represented training lineages receive large per-cell weights. Inner tuning folds mix lineages; outer folds alone test unseen lineage transfer.',
        new_outer_models=4320,workers=WORKERS)
    pp=OUT/'protocol.json'
    if pp.exists():
        previous=json.loads(pp.read_text());protocol['created_utc']=previous['created_utc'];assert previous==xp.clean(protocol)
    else:dump(pp,protocol)
    dump(OUT/'source_manifest.json',dict(files=sources))
    for name in ['panel_inputs.npz','shared_splits.npz']:shutil.copyfile(base/name,OUT/name)
    return sha(pp)
''' + s[b:]
a=s.index('def main():')
s=s[:a]+'''def main():
    start=time.monotonic();ph=prepare();initialize()
    preflight=json.loads((OUT/'preflight_validation.json').read_text());assert preflight['passed']
    assert preflight['protocol_sha256']==ph
    print(json.dumps(dict(event='protocol_frozen',sha256=ph,new_outer_models=4320)),flush=True)
    with futures.ProcessPoolExecutor(max_workers=WORKERS,mp_context=mp.get_context('spawn'),initializer=initialize) as pool:
        jobs=[(p,f,r,l) for r in range(20) for l in range(24) for p in [40,64,80] for f in FAMILIES]
        pending={pool.submit(fit_job,*args,ph):args for args in jobs};completed=0;counts={}
        for future in futures.as_completed(pending):
            m=future.result();completed+=1;key=f'{m["family"]}__{m["panel"]}';counts[key]=counts.get(key,0)+1
            elapsed=time.monotonic()-start
            progress=dict(phase='fitting',completed=completed,total=4320,by_arm=counts,elapsed_seconds=elapsed,
                estimated_remaining_seconds=elapsed/completed*(4320-completed))
            dump(OUT/'progress.json',progress)
            if completed%72==0 or completed==4320:print(json.dumps(dict(event='fit_progress',**progress)),flush=True)
    assert sha(Path(__file__))==json.loads((OUT/'protocol.json').read_text())['script_sha256']
    assert sha(OUT/'weighted_mlp.py')==json.loads((OUT/'protocol.json').read_text())['weighted_mlp_sha256']
    assert sha(OUT/'protocol.json')==ph
    for source in json.loads((OUT/'source_manifest.json').read_text())['files']:assert sha(WS/source['path'])==source['sha256']
    dump(OUT/'fit_completion.json',dict(complete=True,new_outer_models=4320,elapsed_seconds=time.monotonic()-start,sources_unchanged=True))
    dump(OUT/'progress.json',dict(phase='fits_complete',completed=4320,total=4320))
    print(json.dumps(dict(event='all_fits_complete',seconds=time.monotonic()-start)),flush=True)

if __name__=='__main__':main()
'''
(out/'run_weighted.py').write_text(s)
