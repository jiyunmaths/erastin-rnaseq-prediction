#!/usr/bin/env python3
"""Audit the Bayesian networks and compare matched held-out predictions."""
import importlib.util
import json
from pathlib import Path
import sys
import zipfile

import networkx as nx
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler

sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent
WS=OUT.parent.parent
sys.path.insert(0,str(OUT))
import run_bayesian as run
import gaussian_network as gn


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


prior=module('prior_models',WS/'outputs/erastin_nonlinear_comparison/run_models.py')
rc=module('prior_bootstrap',WS/'outputs/erastin_repeated_cv/repeat_cv.py')
METRICS=['spearman','rmse','r2']
NAMES={'ridge':'Ridge, 40 genes','neural_network':'Neural network, 40 genes',
    'gaussian_naive':'Naive Gaussian BN','gaussian_tan':'Tree-augmented Gaussian BN',
    'ridge_panel_bulkformer':'Ridge, panel BulkFormer','ridge_all_bulkformer':'Ridge, all-gene BulkFormer'}


def table(name,rows):
    f=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows)
    f.to_csv(OUT/name,sep='\t',index=False,float_format='%.12g');return f


def independent_fit(x,y,xt,family,shrink):
    # Independent construction uses networkx for tree learning, least squares
    # on covariance-equivalent augmented rows for CPDs, and covariance rather
    # than precision for response conditioning.
    sx=StandardScaler().fit(x);sy=StandardScaler().fit(y[:,None])
    z=np.column_stack([sy.transform(y[:,None]).ravel(),sx.transform(x)])
    n,p=z.shape
    augmented=np.vstack([np.sqrt(1-shrink)*z,np.sqrt(n*shrink)*np.eye(p)])
    s=augmented.T@augmented/n
    residual=s[1:,1:]-np.outer(s[1:,0],s[0,1:])/s[0,0]
    corr=residual/np.sqrt(np.outer(np.diag(residual),np.diag(residual)))
    weights=-.5*np.log1p(-np.minimum(corr*corr,1-1e-12));np.fill_diagonal(weights,0)
    parents={i:None for i in range(p-1)}
    if family=='gaussian_tan':
        graph=nx.Graph();graph.add_nodes_from(range(p-1))
        graph.add_weighted_edges_from((i,j,float(weights[i,j])) for i in range(p-1) for j in range(i+1,p-1))
        tree=nx.maximum_spanning_tree(graph)
        for a,b in nx.bfs_edges(tree,source=0):parents[b]=a
    bmat=np.zeros((p,p));noise=np.empty(p);noise[0]=s[0,0]
    for j in range(p-1):
        par=[0]+([] if parents[j] is None else [parents[j]+1])
        beta=np.linalg.lstsq(augmented[:,par],augmented[:,j+1],rcond=None)[0]
        bmat[j+1,par]=beta
        noise[j+1]=np.sum((augmented[:,j+1]-augmented[:,par]@beta)**2)/n
    transform=np.linalg.inv(np.eye(p)-bmat)
    covariance=transform@np.diag(noise)@transform.T
    coefficient=np.linalg.solve(covariance[1:,1:],covariance[1:,0])
    pred=sy.mean_[0]+sy.scale_[0]*(sx.transform(xt)@coefficient)
    variance=sy.scale_[0]**2*(covariance[0,0]-covariance[0,1:]@coefficient)
    return pred,float(variance)


def bootstrap_macro(folds,names,repeats,seed):
    labels=sorted(folds[folds.test_n>=10].lineage.unique())
    draws=np.random.default_rng(seed).integers(len(labels),size=(5000,len(labels)))
    result={}
    for model in names:
        # Explicit array mean preserves undefined Spearman for any constant
        # predictor; never silently drop a difficult lineage from macro scores.
        v=np.array([[folds[(folds.model==model)&(folds.repeat==r)&(folds.lineage==l)][METRICS].iloc[0].to_numpy(float)
                     for l in labels] for r in repeats]).mean(axis=0)
        result[model]={metric:v[draws,j].mean(axis=1) for j,metric in enumerate(METRICS)}
    return result


def paired_comparisons(scores,macro,draws,macro_draws,pairs,repeats):
    rows=[];seed_rows=[]
    for scope,frame,boot in [('pooled_745',scores,draws),('macro_19',macro,macro_draws)]:
        for a,b in pairs:
            aa=frame[(frame.model==a)&frame.repeat.isin(repeats)].sort_values('repeat')
            bb=frame[(frame.model==b)&frame.repeat.isin(repeats)].sort_values('repeat')
            assert np.array_equal(aa.repeat.to_numpy(),bb.repeat.to_numpy())
            for metric in METRICS:
                sign=-1 if metric=='rmse' else 1
                delta=sign*(aa[metric].to_numpy()-bb[metric].to_numpy())
                lo,hi=np.percentile(sign*(boot[a][metric]-boot[b][metric]),[2.5,97.5])
                rows.append({'scope':scope,'first_model':a,'reference_model':b,'metric':metric,'improvement':np.mean(delta),
                    'favorable_repeats':int(np.sum(delta>0)),'repeats':len(repeats),'conditional_ci_low':lo,'conditional_ci_high':hi})
                seed_rows.extend({'scope':scope,'first_model':a,'reference_model':b,'metric':metric,'repeat':int(r),'improvement':v}
                                 for r,v in zip(aa.repeat,delta))
    return rows,seed_rows


def score_predictions(y,lineage,predictions,repeats):
    scores=[];folds=[];macro=[]
    for name,pp in predictions.items():
        assert len(pp)==len(repeats)
        for repeat,p in zip(repeats,pp):
            scores.append({'repeat':repeat,'model':name,**prior.metrics(y,p)})
            per=[]
            for label in np.unique(lineage):
                ix=lineage==label
                met=prior.metrics(y[ix],p[ix])
                folds.append({'repeat':repeat,'model':name,'lineage':label,'test_n':int(ix.sum()),**met})
                if ix.sum()>=10:per.append([met[m] for m in METRICS])
            assert len(per)==19
            macro.append({'repeat':repeat,'model':name,**dict(zip(METRICS,np.asarray(per).mean(axis=0))),
                          'lineages':19,'defined_spearman_lineages':int(np.isfinite(np.array(per)[:,0]).sum())})
    return pd.DataFrame(scores),pd.DataFrame(folds),pd.DataFrame(macro)


def main():
    run.initialize();s=run.STATE;x,y=s['x'],s['y'];lineage=s['lineage'];labels=s['labels'];genes=s['genes']
    protocol=json.loads((OUT/'protocol.json').read_text());ph=run.sha(OUT/'protocol.json')
    for name,h in protocol['scripts'].items():assert run.sha(OUT/name)==h
    pred=np.full((21,2,745),np.nan);variance=np.full_like(pred,np.nan)
    all_edges=np.zeros((21,24,40,40));coefficients=np.empty((21,24,2,40))
    parameters=[];tuning=[];edge_rows=[];reference=[]
    max_prediction_error=0.;max_variance_error=0.;max_covariance_error=0.;max_tuning_error=0.;graph_checks=0
    oldroot=WS/'outputs/erastin_bayesian_comparison_initial_grid'
    old_hashes={v['path']:v['sha256'] for v in json.loads((oldroot/'artifact_manifest.json').read_text())['files']}
    old_predictions=np.full_like(pred,np.nan);initial_prefixes_unchanged=0
    for r in range(21):
        seen=np.zeros(745,dtype=int)
        for l,label in enumerate(labels):
            token=f'repeat_{r:02d}__lineage_{l:02d}'
            meta=json.loads((OUT/'folds'/(token+'.json')).read_text())
            a=np.load(OUT/'folds'/(token+'.npz'),allow_pickle=False)
            assert run.sha(OUT/'folds'/(token+'.npz'))==meta['array_sha256'] and meta['protocol_sha256']==ph
            tr,te=a['train_rows'],a['test_rows'];inner=s['inner'][r,l]
            assert set(te)==set(np.flatnonzero(lineage==label)) and set(tr)==set(np.flatnonzero(inner>=0))
            assert set(s['groups'][tr]).isdisjoint(s['groups'][te]);seen[te]+=1
            for m,family in enumerate(run.FAMILIES):
                model=meta['selected_models'][m]
                chosen=int(a['selected_candidates'][m]);assert chosen==np.argmin(a['inner_fold_rmse'][m].mean(axis=1))
                assert model['shrinkage']==run.GRID[chosen]
                joint=StandardScaler().fit(np.column_stack([y[tr],x[tr]]))
                np.testing.assert_allclose(model['mean'],joint.mean_,atol=1e-12,rtol=0)
                np.testing.assert_allclose(model['scale'],joint.scale_,atol=1e-12,rtol=0)
                model={k:np.array(v) if isinstance(v,list) else v for k,v in model.items()}
                pp,var=gn.predict(model,x[te]);err=float(np.max(np.abs(pp-a['test_predictions'][m])))
                max_prediction_error=max(max_prediction_error,err);max_variance_error=max(max_variance_error,abs(var-a['test_variances'][m]))
                bmat=model['structural_coefficients'];noise=model['noise_variance']
                graph=nx.from_numpy_array(bmat.T,create_using=nx.DiGraph);assert nx.is_directed_acyclic_graph(graph)
                inv=np.linalg.inv(np.eye(41)-bmat);cov=inv@np.diag(noise)@inv.T
                np.linalg.cholesky(cov)
                weights=np.linalg.solve(cov[1:,1:],cov[1:,0])
                p2=model['mean'][0]+model['scale'][0]*(((x[te]-model['mean'][1:])/model['scale'][1:])@weights)
                max_covariance_error=max(max_covariance_error,float(np.max(np.abs(p2-pp))))
                v2=(cov[0,0]-cov[0,1:]@weights)*model['scale'][0]**2
                max_variance_error=max(max_variance_error,abs(v2-var))
                assert err<1e-11 and var>0;graph_checks+=1
                coefficients[r,l,m]=model['conditional_coefficient']*model['scale'][0]
                pred[r,m,te]=pp;variance[r,m,te]=var
                parameters.append({'repeat':r,'model':family,'lineage':label,'shrinkage':model['shrinkage'],
                    'selected_candidate':chosen,'mean_inner_rmse':a['inner_fold_rmse'][m,chosen].mean(),
                    'conditional_predictive_sd':np.sqrt(var),'train_n':len(tr),'test_n':len(te)})
                for c,shrink in enumerate(run.GRID):
                    losses=[]
                    for k in range(5):
                        va=tr[inner[tr]==k];train=tr[inner[tr]!=k]
                        assert set(s['groups'][train]).isdisjoint(s['groups'][va])
                        losses.append(np.sqrt(np.mean((y[va]-a['inner_validation_predictions'][m,c,va])**2)))
                    max_tuning_error=max(max_tuning_error,float(np.max(np.abs(np.array(losses)-a['inner_fold_rmse'][m,c]))))
                    tuning.append({'repeat':r,'model':family,'lineage':label,'candidate':c,'shrinkage':shrink,'mean_inner_rmse':np.mean(losses)})
                if family=='gaussian_tan':
                    assert len(model['gene_edges'])==39
                    tree=nx.Graph();tree.add_nodes_from(range(40));tree.add_edges_from(model['gene_edges']);assert nx.is_tree(tree)
                    for i,j in model['gene_edges']:
                        active=model['conditional_mutual_information'][i,j]>1e-12
                        if active:all_edges[r,l,i,j]=all_edges[r,l,j,i]=1
                        edge_rows.append({'repeat':r,'lineage':label,'gene_a':genes[i],'gene_b':genes[j],
                            'active_dependency':bool(active),'conditional_mutual_information':model['conditional_mutual_information'][i,j]})
                # Verify every candidate for six holdouts with independently
                # learned graphs/CPDs and a separate inference formula.
                if r in [0,1,20] and label in ['Lung','Thyroid']:
                    outer_p,outer_v=independent_fit(x[tr],y[tr],x[te],family,model['shrinkage'])
                    check={'repeat':r,'lineage':label,'model':family,'outer_prediction_error':float(np.max(np.abs(outer_p-pp))),
                           'outer_variance_error':float(abs(outer_v-var)),'inner_prediction_max_error':0.}
                    for c,shrink in enumerate(run.GRID):
                        for k in range(5):
                            aa=tr[inner[tr]!=k];bb=tr[inner[tr]==k]
                            pv,_=independent_fit(x[aa],y[aa],x[bb],family,shrink)
                            e=float(np.max(np.abs(pv-a['inner_validation_predictions'][m,c,bb])))
                            check['inner_prediction_max_error']=max(check['inner_prediction_max_error'],e)
                    assert max(check['outer_prediction_error'],check['outer_variance_error'],check['inner_prediction_max_error'])<1e-10
                    reference.append(check)
            oldfile=oldroot/'folds'/(token+'.npz');assert run.sha(oldfile)==old_hashes[str(oldfile.relative_to(oldroot))]
            old=np.load(oldfile,allow_pickle=False)
            assert np.array_equal(old['inner_validation_predictions'],a['inner_validation_predictions'][:,:8],equal_nan=True)
            old_predictions[r][:,te]=old['test_predictions'];initial_prefixes_unchanged+=1
        assert np.all(seen==1)
    assert np.isfinite(pred).all() and np.isfinite(variance).all()
    assert max(max_prediction_error,max_variance_error,max_covariance_error,max_tuning_error)<1e-10
    parameter_table=table('selected_parameters.tsv',parameters);table('tuning_scores.tsv',tuning);table('selected_gene_edges.tsv',edge_rows)
    old=np.load(WS/'outputs/erastin_nonlinear_comparison/predictions.npz',allow_pickle=False)
    repeated=np.load(WS/'outputs/erastin_nn_repeated/predictions.npz',allow_pickle=False)
    ridge=np.load(WS/'outputs/erastin_lineage_corrected/per_run_predictions.npz',allow_pickle=False)
    assert np.array_equal(old['sample_ids'],s['ids']) and np.array_equal(repeated['sample_ids'],s['ids']) and np.array_equal(ridge['sample_ids'],s['ids'])
    repeat_predictions={'gaussian_naive':pred[1:,0],'gaussian_tan':pred[1:,1],
        'ridge':repeated['ridge'],'neural_network':repeated['neural_network'],
        'ridge_panel_bulkformer':ridge['panel_contextual_embedding'][1:],
        'ridge_all_bulkformer':ridge['all_gene_contextual_embedding'][1:]}
    original_predictions={key:old[key][None,:] for key in old.files if '__' in key}
    assert len(original_predictions)==9
    original_predictions.update({family:pred[[0],m] for m,family in enumerate(run.FAMILIES)})
    scores,folds,macro=score_predictions(y,lineage,repeat_predictions,list(range(1,21)))
    original_scores,original_folds,original_macro=score_predictions(y,lineage,original_predictions,[0])
    # Verify reused reference metrics independently against the original files.
    original_known=pd.read_csv(WS/'outputs/erastin_nonlinear_comparison/summary.tsv',sep='\t').set_index('arm')
    for row in original_scores.itertuples():
        if row.model in original_known.index:
            np.testing.assert_allclose([getattr(row,m) for m in METRICS],original_known.loc[row.model,METRICS].to_numpy(float),rtol=0,atol=1e-12)
    table('per_repeat_metrics.tsv',scores);table('per_lineage_metrics.tsv',folds);table('per_repeat_macro_metrics.tsv',macro)
    table('original_seed_metrics.tsv',original_scores);table('original_seed_lineage_metrics.tsv',original_folds)
    summary=[]
    for scope,f in [('pooled_745',scores),('macro_19',macro)]:
        for name in repeat_predictions:
            v=f[f.model==name]
            row={'scope':scope,'model':name}
            for metric in METRICS:
                a=v[metric].to_numpy();row.update({metric+'_mean':np.mean(a),metric+'_sd':np.std(a,ddof=1),metric+'_min':np.min(a),metric+'_max':np.max(a)})
            summary.append(row)
    summary=table('repeated_summary.tsv',summary)
    print(json.dumps({'event':'audit_complete_starting_bootstrap','independent_refits':len(reference)}),flush=True)
    run.dump(OUT/'progress.json',{'phase':'paired_lineage_bootstrap','models_audited':1008})
    boot_check=rc.validate_weighted_bootstrap(y,pred[1:3,1],lineage,prior.metrics)
    boot=rc.repeated_group_bootstrap(y,repeat_predictions,lineage,count=5000,seed=protocol['uncertainty']['repeat_pooled_seed'])
    macro_boot=bootstrap_macro(folds,list(repeat_predictions),range(1,21),protocol['uncertainty']['repeat_macro_seed'])
    pairs=[('gaussian_tan',name) for name in ['ridge','neural_network','gaussian_naive','ridge_panel_bulkformer','ridge_all_bulkformer']]
    pairs.extend([('gaussian_naive','ridge'),('gaussian_naive','neural_network')])
    contrasts,seed_delta=paired_comparisons(scores,macro,boot,macro_boot,pairs,list(range(1,21)))
    contrasts=table('paired_repeated_comparisons.tsv',contrasts);table('paired_seed_differences.tsv',seed_delta)
    original_boot=rc.repeated_group_bootstrap(y,original_predictions,lineage,count=5000,seed=protocol['uncertainty']['original_pooled_seed'])
    original_macro_boot=bootstrap_macro(original_folds,list(original_predictions),[0],protocol['uncertainty']['original_macro_seed'])
    pairs0=[('gaussian_tan',name) for name in original_predictions if name!='gaussian_tan']
    contrasts0,_=paired_comparisons(original_scores,original_macro,original_boot,original_macro_boot,pairs0,[0])
    table('paired_original_comparisons.tsv',contrasts0)
    # All reported single-split arms use exactly the original inner split seed.
    orig=original_scores.merge(original_macro[['model']+METRICS],on='model',suffixes=('','_macro'))
    table('original_seed_summary.tsv',orig)
    delta=[]
    for label in labels:
        for reference_model in ['ridge','neural_network','gaussian_naive']:
            a=folds[(folds.lineage==label)&(folds.model=='gaussian_tan')].sort_values('repeat')
            b=folds[(folds.lineage==label)&(folds.model==reference_model)].sort_values('repeat')
            row={'lineage':label,'n':int(np.sum(lineage==label)),'reference_model':reference_model}
            for metric in METRICS:
                sign=-1 if metric=='rmse' else 1
                row[metric+'_improvement']=float(np.mean(sign*(a[metric].to_numpy()-b[metric].to_numpy())))
            delta.append(row)
    delta=table('lineage_paired_effects.tsv',delta)
    winrows=[]
    for ref in ['ridge','neural_network','gaussian_naive']:
        v=delta[(delta.n>=10)&(delta.reference_model==ref)]
        winrows.append({'reference_model':ref,**{m+'_favorable_lineages':int((v[m+'_improvement']>0).sum()) for m in METRICS},'eligible_lineages':19})
    table('lineage_win_counts.tsv',winrows)
    # Gaussian predictive uncertainty is model-based and uncalibrated.
    prob=[]
    z90=float(norm.ppf(.95))
    for r in range(21):
        for m,family in enumerate(run.FAMILIES):
            for label in ['pooled_745']+list(labels):
                ix=np.ones(745,dtype=bool) if label=='pooled_745' else lineage==label
                err=y[ix]-pred[r,m,ix];var=variance[r,m,ix];sd=np.sqrt(var)
                prob.append({'repeat':r,'model':family,'lineage':label,'n':int(ix.sum()),
                    'gaussian_nll':np.mean(.5*(np.log(2*np.pi*var)+err*err/var)),
                    'coverage_90':np.mean(np.abs(err)<=z90*sd),'mean_interval_width_90':np.mean(2*z90*sd)})
    prob=table('predictive_uncertainty.tsv',prob)
    prob_summary=table('predictive_uncertainty_summary.tsv',prob[(prob.repeat>0)&(prob.lineage=='pooled_745')].groupby('model',as_index=False)[['gaussian_nll','coverage_90','mean_interval_width_90']].mean())
    edge_frequency=all_edges[1:].mean(axis=(0,1));edges=[]
    for i in range(40):
        for j in range(i+1,40):
            edges.append({'gene_a':genes[i],'gene_b':genes[j],'selected_fits':int(all_edges[1:,:,i,j].sum()),
                'total_fits':480,'selection_frequency':edge_frequency[i,j],
                'heldout_lineages_ever_selected':int((all_edges[1:,:,i,j].sum(axis=0)>0).sum())})
    edges=table('gene_edge_stability.tsv',pd.DataFrame(edges).sort_values(['selection_frequency','gene_a','gene_b'],ascending=[False,True,True]))
    edge_by_lineage=[]
    for l,label in enumerate(labels):
        for i,j in zip(*np.triu_indices(40,1)):
            if all_edges[1:,l,i,j].sum()>0:
                edge_by_lineage.append({'lineage':label,'gene_a':genes[i],'gene_b':genes[j],'selection_frequency':all_edges[1:,l,i,j].mean()})
    table('gene_edges_by_lineage.tsv',edge_by_lineage)
    initial_scores,_,_=score_predictions(y,lineage,{f:old_predictions[1:,m] for m,f in enumerate(run.FAMILIES)},range(1,21))
    table('initial_grid_metrics.tsv',initial_scores)
    initial_change=[]
    for family in run.FAMILIES:
        aa=scores[scores.model==family];bb=initial_scores[initial_scores.model==family]
        initial_change.append({'model':family,**{'initial_'+m:bb[m].mean() for m in METRICS},**{'extended_'+m:aa[m].mean() for m in METRICS}})
    table('grid_extension_comparison.tsv',initial_change)
    counts=parameter_table[parameter_table.repeat>0].groupby(['model','shrinkage']).size().reset_index(name='outer_fits')
    table('parameter_selection_counts.tsv',counts)
    validation={'passed':True,'selected_models_audited':graph_checks,'candidate_loss_sets_audited':21*24*2*len(run.GRID),
        'prediction_reconstruction_max_error':max_prediction_error,'independent_covariance_conditioning_max_error':max_covariance_error,
        'conditional_variance_max_error':max_variance_error,'inner_fold_loss_max_error':max_tuning_error,
        'acyclic_graphs_and_positive_definite_covariances':True,'source_scalers_and_group_exclusion_verified':True,
        'independent_full_candidate_refits':reference,'original_grid_predictions_unchanged_in_extension':initial_prefixes_unchanged,
        'reused_original_reference_metrics_match':True,'bootstrap_validation':boot_check,
        'mathematical_tests':json.loads((OUT/'mathematical_validation.json').read_text())}
    run.dump(OUT/'validation.json',validation)
    np.savez_compressed(OUT/'predictions.npz',sample_ids=s['ids'],lineage=lineage,groups=s['groups'],y=y,seeds=s['seeds'],
        **{f:pred[:,m] for m,f in enumerate(run.FAMILIES)},**{f+'_variance':variance[:,m] for m,f in enumerate(run.FAMILIES)},
        **{'repeat_reference__'+k:v for k,v in repeat_predictions.items() if k not in run.FAMILIES})
    np.savez_compressed(OUT/'graph_arrays.npz',genes=genes,labels=labels,active_gene_edges=all_edges,standardized_response_coefficients=coefficients)
    np.savez_compressed(OUT/'bootstrap_draws.npz',**{scope+'__'+name+'__'+metric:v for scope,b in
        [('repeat_pooled',boot),('repeat_macro',macro_boot),('original_pooled',original_boot),('original_macro',original_macro_boot)]
        for name,md in b.items() for metric,v in md.items()})
    final={'repeated_summary':summary.to_dict('records'),'original_summary':orig.to_dict('records'),
        'parameter_counts':counts.to_dict('records'),'lineage_wins':winrows,'predictive_uncertainty':prob_summary.to_dict('records'),
        'distinct_prediction_vectors_per_BN':{f:len(np.unique(pred[1:,m],axis=0)) for m,f in enumerate(run.FAMILIES)},
        'grid_extension':initial_change,'validation_passed':True}
    run.dump(OUT/'final_summary.json',final)
    render(summary,orig,contrasts,delta,genes,edge_frequency,edges,counts,prob_summary,validation,final)
    for source in json.loads((OUT/'source_manifest.json').read_text())['files']:assert run.sha(WS/source['path'])==source['sha256']
    run.dump(OUT/'progress.json',{'phase':'complete','models_audited':1008,'validation_passed':True})
    archive=seal_artifacts()
    print(json.dumps({'event':'review_complete','archive_bytes':archive.stat().st_size,'final':run.clean(final)},indent=2),flush=True)


def seal_artifacts():
    oldroot=WS/'outputs/erastin_bayesian_comparison_initial_grid'
    manifest={'files':[{'path':str(p.relative_to(OUT)),'bytes':p.stat().st_size,'sha256':run.sha(p)} for p in sorted(OUT.rglob('*'))
                       if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts]}
    run.dump(OUT/'artifact_manifest.json',manifest)
    archive=OUT.parent/'erastin_bayesian_comparison_results.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for row in manifest['files']:z.write(OUT/row['path'],'erastin_bayesian_comparison/'+row['path'])
        z.write(OUT/'artifact_manifest.json','erastin_bayesian_comparison/artifact_manifest.json')
        # Preserve the full original-grid run with its own manifest.
        for p in sorted(oldroot.rglob('*')):
            if p.is_file() and '__pycache__' not in p.parts:z.write(p,'erastin_bayesian_comparison_initial_grid/'+str(p.relative_to(oldroot)))
    import hashlib
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for root_name,records in [('erastin_bayesian_comparison',manifest['files']),('erastin_bayesian_comparison_initial_grid',json.loads((oldroot/'artifact_manifest.json').read_text())['files'])]:
            for row in records:assert hashlib.sha256(z.read(root_name+'/'+row['path'])).hexdigest()==row['sha256']
    return archive


def render(summary,orig,contrasts,delta,genes,edge_frequency,edges,counts,prob,validation,final):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none'})
    order=['neural_network','ridge','gaussian_tan','gaussian_naive','ridge_panel_bulkformer','ridge_all_bulkformer']
    colors=['#8654a2','#167b8d','#c97d20','#b49b65','#509e87','#7c9a99']
    fig,axes=plt.subplots(2,2,figsize=(12,8),constrained_layout=True,sharex='col')
    for i,scope in enumerate(['pooled_745','macro_19']):
        v=summary[summary.scope==scope].set_index('model').loc[order]
        for j,metric in enumerate(['spearman','rmse']):
            ax=axes[i,j]
            ax.scatter(v[metric+'_mean'],np.arange(len(order)),color=colors,s=55,zorder=3)
            ax.errorbar(v[metric+'_mean'],np.arange(len(order)),xerr=v[metric+'_sd'],fmt='none',color='#27333d',capsize=2)
            ax.set_yticks(np.arange(len(order)),[NAMES[k] for k in order]);ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
            ax.set_title(('Pooled 745 lines' if i==0 else 'Equal-weight mean of 19 lineages')+' — '+('Spearman ↑' if j==0 else 'RMSE ↓'))
            ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Probabilistic Gaussian networks versus saved models\n20 matched inner-fold seeds; fixed leave-lineage-out tests',fontsize=13)
    fig.supxlabel('Whiskers show seed SD, not uncertainty from independent biological samples.',fontsize=9)
    fig.savefig(OUT/'model_comparison.png',dpi=160);fig.savefig(OUT/'model_comparison.svg');plt.close(fig)
    v=delta[delta.n>=10];lineages=v.drop_duplicates('lineage').sort_values('n',ascending=False).lineage
    fig,axes=plt.subplots(1,2,figsize=(12,8),constrained_layout=True,sharey=True)
    for ax,metric in zip(axes,['spearman','rmse']):
        for off,ref,color in [(-.17,'ridge','#167b8d'),(.17,'neural_network','#8654a2')]:
            vv=v[v.reference_model==ref].set_index('lineage').loc[lineages]
            ax.barh(np.arange(19)+off,vv[metric+'_improvement'],height=.3,label=NAMES[ref],color=color)
        ax.axvline(0,color='#555',lw=1);ax.grid(axis='x',alpha=.2)
        ax.set_xlabel('Spearman increase' if metric=='spearman' else 'RMSE reduction')
        ax.spines[['top','right']].set_visible(False)
    axes[0].set_yticks(np.arange(19),lineages);axes[0].invert_yaxis()
    handles,legend_labels=axes[1].get_legend_handles_labels()
    fig.legend(handles,legend_labels,loc='outside lower center',ncol=2)
    fig.suptitle('Tree-augmented Bayesian network by held-out lineage\nMean paired change across 20 repeats; positive favors the Bayesian network',fontsize=12)
    fig.savefig(OUT/'lineage_comparison.png',dpi=160);fig.savefig(OUT/'lineage_comparison.svg');plt.close(fig)
    fig,ax=plt.subplots(figsize=(10.5,10),constrained_layout=True)
    im=ax.imshow(edge_frequency,cmap='Blues',vmin=0,vmax=1)
    ax.set_xticks(np.arange(40),genes,rotation=90,fontsize=7);ax.set_yticks(np.arange(40),genes,fontsize=7)
    ax.set_title('Gene dependency selection frequency\n480 selected TAN fits; dependencies conditional on erastin sensitivity')
    fig.colorbar(im,ax=ax,shrink=.7,label='Fraction of selected outer fits containing the active edge')
    fig.savefig(OUT/'gene_dependency_stability.png',dpi=180);fig.savefig(OUT/'gene_dependency_stability.svg');plt.close(fig)
    main=summary[summary.scope=='pooled_745'].set_index('model')
    macro=summary[summary.scope=='macro_19'].set_index('model')
    rows=['# Probabilistic graphical models for erastin sensitivity','',
        '**Neither Gaussian Bayesian network improves on the 40-gene ridge or neural-network models.** '
        f'The tree-augmented network has {100*(main.loc["gaussian_tan","rmse_mean"]/main.loc["ridge","rmse_mean"]-1):.2f}% higher pooled RMSE than ridge and '
        f'{100*(main.loc["gaussian_tan","rmse_mean"]/main.loc["neural_network","rmse_mean"]-1):.2f}% higher RMSE than the neural network. '
        'It loses on pooled RMSE and Spearman in all 20 matched repeats; the corresponding paired 95% lineage-bootstrap intervals favor the existing models. '
        'Its small pooled improvement over the naive Bayesian network has intervals crossing zero, and its equal-weight lineage RMSE is slightly worse than naive.','',
        'Two continuous Gaussian Bayesian networks were tested on the same 40-gene expression panel and corrected leave-lineage-out splits. '
        'All 24 lineage holdouts were preserved, including 745 cancer cell lines and 734 identity groups. '
        'The network structure, expression/response scaling and distribution parameters were learned separately inside every training fold.','',
        'The tree-augmented model (TAN) adds 39 learned gene–gene dependencies to the naive network. '
        'The naive model assumes genes are independent given continuous sensitivity. Both predict the conditional mean of negative erastin AUC. '
        'No response discretization, BulkFormer refitting, gene filtering or prediction clipping was used.','',
        '## Matched comparison across 20 repeats','',
        'These are means of per-repeat held-out scores. Outer lineage tests stay fixed; seeds vary the inner tuning assignments. '
        'Ridge, neural-network and BulkFormer ridge predictions are reused from the exact matching saved seeds.','',
        '| Model | Pooled Spearman ↑ | Pooled RMSE ↓ | Pooled R² ↑ | Macro Spearman ↑ | Macro RMSE ↓ |',
        '|---|---:|---:|---:|---:|---:|']
    for name in order:
        a=main.loc[name];b=macro.loc[name]
        rows.append(f'| {NAMES[name]} | {a.spearman_mean:.5f} | {a.rmse_mean:.6f} | {a.r2_mean:.5f} | {b.spearman_mean:.5f} | {b.rmse_mean:.6f} |')
    rows+=['',f'![Repeated comparison]({OUT}/model_comparison.png)','',
        '## Paired change for the tree-augmented Bayesian network','',
        'Positive values mean improvement. Intervals use 5,000 paired resamples of whole held-out lineages, conditional on the saved predictions.','',
        '| Scope | Compared with | Metric | Mean improvement | Favorable repeats | Conditional 95% interval |',
        '|---|---|---|---:|---:|---|']
    for row in contrasts[(contrasts.first_model=='gaussian_tan')&contrasts.reference_model.isin(['ridge','neural_network','gaussian_naive'])&contrasts.metric.isin(['spearman','rmse'])].itertuples():
        rows.append(f'| {row.scope} | {NAMES[row.reference_model]} | {"Spearman increase" if row.metric=="spearman" else "RMSE reduction"} | '
                    f'{row.improvement:+.6f} | {row.favorable_repeats}/20 | [{row.conditional_ci_low:+.6f}, {row.conditional_ci_high:+.6f}] |')
    rows+=['','Among the 19 lineages with n≥10:','']
    for item in final['lineage_wins']:
        rows.append(f'- TAN versus {NAMES[item["reference_model"]]}: higher Spearman in {item["spearman_favorable_lineages"]}/19 and lower RMSE in {item["rmse_favorable_lineages"]}/19.')
    rows+=['',f'![Lineage comparison]({OUT}/lineage_comparison.png)','',
        '## Original-seed comparison including random forest','',
        'Random forest was previously fitted for inner seed 20261031 only. This table compares every arm at that same original seed; '
        'it does not compare a one-seed forest with an averaged 20-seed predictor. The original seed is excluded from the repeated means above.','',
        '| Model / representation | Spearman ↑ | RMSE ↓ | R² ↑ | Macro Spearman ↑ |','|---|---:|---:|---:|---:|']
    for row in orig.sort_values('rmse').itertuples():
        label=NAMES.get(row.model,row.model.replace('__',' / ').replace('_',' '))
        rows.append(f'| {label} | {row.spearman:.5f} | {row.rmse:.6f} | {row.r2:.5f} | {row.spearman_macro:.5f} |')
    rows+=['','## Regularization and the preserved initial run','',
        'The initial grid was [0, .01, .03, .1, .2, .4, .6, .8]. All 504 naive fits and 483/504 TAN fits selected the upper boundary by inner validation. '
        'After seeing that tuning-boundary problem (and the initial pooled test scores), a documented amendment added [.9, .95, .99, 1]. '
        'The reason was the inner-selection boundary, not a target held-out score. The revised grid was fixed before its results were scored. '
        'The full original run is preserved in the results archive; all original candidate validation predictions were reproduced exactly. '
        'The extension left every selected model and held-out prediction unchanged; the final selected values are inside the extended range. '
        'This remains an exploratory follow-up on an already examined cohort.','',
        '| Model | Selected shrinkage | Number of follow-up outer fits |','|---|---:|---:|']
    for row in counts.itertuples():rows.append(f'| {NAMES[row.model]} | {row.shrinkage:g} | {row.outer_fits}/480 |')
    rows+=['',f'Distinct complete held-out prediction vectors among the 20 repeats: naive = {final["distinct_prediction_vectors_per_BN"]["gaussian_naive"]}; '
        f'TAN = {final["distinct_prediction_vectors_per_BN"]["gaussian_tan"]}. Deterministic fits can repeat exactly when inner tuning selects the same settings. '
        'This is stability under tuning assignments, not twenty independently trained stochastic models or independent biological replications.','',
        '## Predictive uncertainty','',
        'The Bayesian networks also return a Gaussian conditional variance. These intervals use fitted distribution parameters; '
        'they do not integrate parameter/graph uncertainty and were not recalibrated on the held-out responses.','',
        '| Model | Mean Gaussian NLL ↓ | Coverage of nominal 90% interval | Mean interval width |','|---|---:|---:|---:|']
    for row in prob.itertuples():rows.append(f'| {NAMES[row.model]} | {row.gaussian_nll:.4f} | {row.coverage_90:.1%} | {row.mean_interval_width_90:.4f} |')
    rows+=['','## Learned gene dependencies','',
        'The network is learned from the training expression and sensitivity values, not imported from a protein-interaction database. '
        'The response is a root in the probabilistic factorization; this arrow choice does not claim that drug sensitivity causes gene expression. '
        'Likewise, the orientation of the gene tree is arbitrary for Gaussian inference. Gene pairs below are therefore reported without causal direction.','',
        '| Gene pair | Selection frequency across 480 TAN fits |','|---|---:|']
    for row in edges.head(15).itertuples():rows.append(f'| {row.gene_a} — {row.gene_b} | {row.selection_frequency:.1%} |')
    rows+=['',f'![Gene dependency stability]({OUT}/gene_dependency_stability.png)','',
        'Edge counts reflect overlapping training cohorts. They are descriptive stability measures, not independent confirmation of physical or regulatory interactions.','',
        '## Method and interpretation limits','',
        '- TAN uses a maximum spanning tree of Gaussian conditional mutual information between genes given response, then linear Gaussian conditional distributions at the nodes. '
        'This is a continuous-response adaptation of the [TAN factorization](https://dang.cs.technion.ac.il/journal_papers/friedman1997Bayesian.pdf); the original paper evaluates classification.',
        '- Gaussian networks have a joint Gaussian representation and permit exact response conditioning; their conditional means are linear in the observed expression. '
        'See the [linear Gaussian network documentation](https://pgmpy.org/api/generated/models/pgmpy.models.LinearGaussianBayesianNetwork.html). '
        'The implementation is saved here and uses NumPy/scikit-learn; pgmpy was not installed or used.',
        '- Covariance shrinkage uses S=(1−λ)C+λI after training-only standardization. The [diagonal shrinkage form](https://scikit-learn.org/stable/modules/generated/sklearn.covariance.ledoit_wolf.html) '
        'is familiar, but λ here is chosen by inner RMSE, not by the Ledoit–Wolf estimator.',
        '- These are Bayesian-network factorizations with estimated parameters, not a full Bayesian posterior over structures and parameters. '
        'The comparison does not test nonlinear, mixture, discrete or biologically constrained Bayesian networks.',
        '- Each outer holdout tunes λ using the same five grouped inner folds as prior models. These inner folds mix represented training lineages; this study does not add lineage-held-out inner tuning.',
        '- Pooled metrics include differences between lineage means. Macro metrics weight the same 19 larger lineages equally. '
        'A constant within-lineage predictor has undefined Spearman; if it occurs, macro Spearman remains undefined rather than dropping that lineage.',
        '- Confidence intervals resample 24 lineages for pooled metrics and 19 for macro metrics. All arms and repeats remain paired; models are not refitted. '
        'Training sets overlap, the repeated samples are not independent datasets, and exploratory comparisons are not multiplicity adjusted.',
        '- The same archived CTRP labels and DepMap 24Q4 inputs are reused. No external drug screen or clinical cohort has been added. '
        'BulkFormer reference arms retain the earlier unresolved pretraining-overlap limitation.','',
        '## Audit and files','',
        f'All {validation["selected_models_audited"]} selected Bayesian-network fits were audited, including training-only scalers, identity exclusion, acyclic graphs, '
        'positive-definite covariances, exact prediction reconstruction and independent covariance-based conditioning. '
        f'All {validation["candidate_loss_sets_audited"]} candidate loss sets were reconstructed. '
        'For original/first/last seeds in Lung and Thyroid, every candidate was independently refitted with networkx tree learning, augmented-data least squares CPDs and separate covariance conditioning. '
        'Known-network synthetic checks verify the exact conditional mean/variance, tree-root invariance and naive-network formula.','',
        f'- [Repeated score table]({OUT}/repeated_summary.tsv)',f'- [Original-seed comparison including random forest]({OUT}/original_seed_summary.tsv)',
        f'- [Paired repeated comparisons]({OUT}/paired_repeated_comparisons.tsv)',f'- [Gene edge stability]({OUT}/gene_edge_stability.tsv)',
        f'- [Protocol]({OUT}/protocol.json)',f'- [Validation]({OUT}/validation.json)','']
    (OUT/'REVIEW.md').write_text('\n'.join(rows))


if __name__=='__main__':
    if '--render-only' in sys.argv:
        def read(name):return pd.read_csv(OUT/name,sep='\t')
        graph=np.load(OUT/'graph_arrays.npz',allow_pickle=False)
        render(read('repeated_summary.tsv'),read('original_seed_summary.tsv'),read('paired_repeated_comparisons.tsv'),
            read('lineage_paired_effects.tsv'),graph['genes'],graph['active_gene_edges'][1:].mean(axis=(0,1)),
            read('gene_edge_stability.tsv'),read('parameter_selection_counts.tsv'),read('predictive_uncertainty_summary.tsv'),
            json.loads((OUT/'validation.json').read_text()),json.loads((OUT/'final_summary.json').read_text()))
        for source in json.loads((OUT/'source_manifest.json').read_text())['files']:assert run.sha(WS/source['path'])==source['sha256']
        archive=seal_artifacts()
        print(json.dumps({'event':'rendered_and_resealed','archive_bytes':archive.stat().st_size}),flush=True)
    else:main()
