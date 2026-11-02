#!/usr/bin/env python3
"""Summarize exact held-out gene replacement, weights, gradients and stability."""
import json,zipfile,hashlib
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata,spearmanr
import calculate_importance as calc
OUT=calc.OUT;FAMILIES=calc.FAMILIES

def table(name,rows):
    a=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows);a.to_csv(OUT/name,sep='\t',index=False,float_format='%.12g');return a

def ranks(v):return rankdata(-np.asarray(v),axis=-1,method='average')

def corr(a,b):
    return float(spearmanr(a,b).statistic) if np.ptp(a)>0 and np.ptp(b)>0 else np.nan

def pair_stats(v,model,kind,unit,names,rows):
    values=[]
    for i in range(len(v)):
        for j in range(i+1,len(v)):
            r=corr(v[i],v[j]);rows.append(dict(model=model,measure=kind,unit=unit,first=str(names[i]),second=str(names[j]),spearman=r));values.append(r)
    v=np.asarray(values);v=v[np.isfinite(v)]
    return dict(model=model,measure=kind,unit=unit,pairs=len(v),median=float(np.median(v)),q25=float(np.percentile(v,25)),q75=float(np.percentile(v,75)))

def main():
    calc.initialize();s=calc.STATE;genes=s['genes'];labels=s['labels'];x=s['x'];lineage=s['lineage'];p=len(genes);R=20;L=len(labels);ph=calc.sha(OUT/'protocol.json');protocol=json.load(open(OUT/'protocol.json'));cv=json.load(open(OUT/'calculation_validation.json'));assert cv['passed']
    assert calc.sha(OUT/'calculate_importance.py')==protocol['script_sha256']
    calc.dump(OUT/'analysis_source_manifest.json',dict(review_sha256=calc.sha(Path(__file__)),report_sha256=calc.sha(OUT/'report_importance.py')))
    D=np.empty((2,R,L,p));S=np.empty_like(D);T=np.empty_like(D);TS=np.empty_like(D);G=np.empty_like(D);GA=np.empty_like(D);G2=np.empty_like(D);GP=np.empty_like(D);B=np.empty((2,R,L));N=np.empty((R,L),int);ridge_coefs=[];max_aggregate=0.
    distinct=[]
    for r in range(R):
        root=OUT/'repeats'/f'r{r+1:02d}';meta=json.load(open(root.with_suffix('.json')));assert meta['protocol_sha256']==ph and calc.sha(root.with_suffix('.npz'))==meta['array_sha256'];a=np.load(root.with_suffix('.npz'));te=a['test_rows'];np.testing.assert_array_equal(te,np.flatnonzero(s['outer_test'][r]))
        pp=a['baseline_prediction'];np.testing.assert_allclose(pp,s['saved_predictions'][:,r,te],atol=1e-11,rtol=0);np.testing.assert_allclose(a['baseline_squared_error'],(pp-s['y'][te][None,:])**2,atol=1e-14,rtol=0)
        np.testing.assert_array_equal(a['standardized_gradient'][0],np.repeat(a['ridge_coefficients'][None,:],len(te),axis=0));ridge_coefs.append(a['ridge_coefficients'])
        for l,label in enumerate(labels):
            pos=np.flatnonzero(lineage[te]==label);N[r,l]=len(pos);assert len(pos)>0
            D[:,r,l]=a['train_reference_delta_mse'][:,pos].mean(axis=1);S[:,r,l]=a['train_reference_mean_squared_prediction_shift'][:,pos].mean(axis=1)
            T[:,r,l]=a['test_reference_delta_mse'][:,pos].mean(axis=1);TS[:,r,l]=a['test_reference_mean_squared_prediction_shift'][:,pos].mean(axis=1)
            gg=a['standardized_gradient'][:,pos];G[:,r,l]=gg.mean(axis=1);GA[:,r,l]=np.abs(gg).mean(axis=1);G2[:,r,l]=(gg**2).mean(axis=1);GP[:,r,l]=(gg>0).mean(axis=1);B[:,r,l]=a['baseline_squared_error'][:,pos].mean(axis=1)
        reconstructed=(D[:,r]*N[r][None,:,None]).sum(axis=1)/N[r].sum();actual=a['train_reference_delta_mse'].mean(axis=1);max_aggregate=max(max_aggregate,float(np.max(np.abs(reconstructed-actual))));assert max_aggregate<1e-12
        for j,gene in enumerate(genes):ridge_coefs[-1][j]=a['ridge_coefficients'][j]
    assert np.all(N==N[0]) and N[0].sum()==149
    n=N[0];cohort=np.array([np.sum(lineage==l) for l in labels]);eligible=cohort>=10;strict=n>=5;testeligible=n>=2
    assert eligible.sum()==19 and strict.sum()==12 and testeligible.sum()==20
    for l,label in enumerate(labels):
        ever=np.any(s['outer_test'],axis=0)&(lineage==label);distinct.append(dict(lineage=str(label),cohort_n=int(cohort[l]),test_n=int(n[l]),unique_test_cells=int(ever.sum()),unique_test_identity_groups=len(np.unique(s['groups'][ever])),eligible19=bool(eligible[l]),strict12=bool(strict[l]),test_permutation_estimable=bool(testeligible[l])))
    table('lineage_information.tsv',distinct)
    scopes={'pooled23':n.astype(float),'macro23':np.ones(L),'pooled19':n*eligible,'macro19':eligible.astype(float)}
    rng23=np.random.default_rng(protocol['uncertainty']['seed']);rng19=np.random.default_rng(protocol['uncertainty']['seed']+1);draws23=rng23.integers(L,size=(5000,L));draws19=rng19.integers(19,size=(5000,19));summaries=[];repeat_rows=[];bootstrap_arrays={};pair_rows=[];rank_stats=[];lineage_rows=[]
    dranks=ranks(D);mean_lineage_D=D.mean(axis=1);lineage_ranks=ranks(mean_lineage_D);influence_lineage=np.sqrt(S.mean(axis=1));gradient_lineage=np.sqrt(G2.mean(axis=1));global_values={}
    for scope,w in scopes.items():
        keep=np.flatnonzero(w>0);pick=draws23 if len(keep)==23 else draws19;weights=w[keep][pick];den=weights.sum(axis=1)
        selected=D.mean(axis=1)[:,keep,:];sh=S.mean(axis=1)[:,keep,:];base=B.mean(axis=1)[:,keep]
        for m,model in enumerate(FAMILIES):
            perseed=np.einsum('rlg,l->rg',D[m],w)/w.sum();base_seed=B[m]@w/w.sum();shift_seed=np.einsum('rlg,l->rg',S[m],w)/w.sum();g_seed=np.einsum('rlg,l->rg',G[m],w)/w.sum();ga_seed=np.einsum('rlg,l->rg',GA[m],w)/w.sum();g2_seed=np.einsum('rlg,l->rg',G2[m],w)/w.sum();gp_seed=np.einsum('rlg,l->rg',GP[m],w)/w.sum()
            point=perseed.mean(axis=0);prank=ranks(point);seedrank=ranks(perseed);influence=np.sqrt(shift_seed.mean(axis=0));irank=ranks(influence);gradient_rms=np.sqrt(g2_seed.mean(axis=0));grank=ranks(gradient_rms)
            boot=(selected[m][pick]*weights[:,:,None]).sum(axis=1)/den[:,None];bbase=(base[m][pick]*weights).sum(axis=1)/den;relboot=100*boot/bbase[:,None];lo,hi=np.percentile(boot,[2.5,97.5],axis=0);rlo,rhi=np.percentile(relboot,[2.5,97.5],axis=0);branks=ranks(boot)
            bootstrap_arrays[model+'__'+scope+'__delta_mse']=boot;bootstrap_arrays[model+'__'+scope+'__baseline_mse']=bbase
            for j,gene in enumerate(genes):
                summaries.append(dict(model=model,scope=scope,gene=str(gene),added_24=bool(j>=40),rank=float(prank[j]),delta_mse=float(point[j]),baseline_mse=float(base_seed.mean()),relative_mse_increase_percent=float(100*point[j]/base_seed.mean()),conditional_ci_low=float(lo[j]),conditional_ci_high=float(hi[j]),relative_ci_low=float(rlo[j]),relative_ci_high=float(rhi[j]),bootstrap_top10_frequency=float(np.mean((branks[:,j]<=10)&(boot[:,j]>0))),
                    rms_prediction_change=float(influence[j]),influence_rank=float(irank[j]),mean_standardized_slope=float(g_seed[:,j].mean()),mean_absolute_standardized_slope=float(ga_seed[:,j].mean()),rms_standardized_slope=float(gradient_rms[j]),slope_rms_rank=float(grank[j]),positive_gradient_fraction=float(gp_seed[:,j].mean()),positive_seed_slopes=int(np.sum(g_seed[:,j]>0)),
                    positive_repeats=int(np.sum(perseed[:,j]>0)),top10_repeats=int(np.sum((seedrank[:,j]<=10)&(perseed[:,j]>0))),rank_min=float(seedrank[:,j].min()),rank_max=float(seedrank[:,j].max()),rank_median=float(np.median(seedrank[:,j])),
                    positive_lineages_23=int(np.sum(mean_lineage_D[m,:,j]>0)),positive_lineages_19=int(np.sum(mean_lineage_D[m,eligible,j]>0)),positive_lineages_12=int(np.sum(mean_lineage_D[m,strict,j]>0)),top10_lineages_19=int(np.sum((lineage_ranks[m,eligible,j]<=10)&(mean_lineage_D[m,eligible,j]>0)))))
                for r in range(R):repeat_rows.append(dict(model=model,scope=scope,repeat=r+1,gene=str(gene),delta_mse=float(perseed[r,j]),relative_mse_increase_percent=float(100*perseed[r,j]/base_seed[r]),rank=float(seedrank[r,j]),rms_prediction_change=float(np.sqrt(shift_seed[r,j])),mean_standardized_slope=float(g_seed[r,j])))
            if scope=='pooled23':
                global_values[model]=dict(delta=point,influence=influence,gradient=gradient_rms,perseed=perseed,seedrank=seedrank)
                for kind,v in [('predictive_importance',perseed),('prediction_influence',np.sqrt(shift_seed)),('standardized_sensitivity',np.sqrt(g2_seed))]:rank_stats.append(pair_stats(v,model,kind,'between_splits',list(range(1,21)),pair_rows))
    for m,model in enumerate(FAMILIES):
        for l,label in enumerate(labels):
            rnk=ranks(D[m,:,l]);value=mean_lineage_D[m,l];br=B[m,:,l].mean();influence=influence_lineage[m,l];gr=gradient_lineage[m,l]
            for j,gene in enumerate(genes):
                lineage_rows.append(dict(model=model,lineage=str(label),gene=str(gene),cohort_n=int(cohort[l]),test_n=int(n[l]),eligible19=bool(eligible[l]),strict12=bool(strict[l]),delta_mse=float(value[j]),relative_mse_increase_percent=float(100*value[j]/br),baseline_mse=float(br),rank=float(lineage_ranks[m,l,j]),positive_repeats=int(np.sum(D[m,:,l,j]>0)),top10_repeats=int(np.sum((rnk[:,j]<=10)&(D[m,:,l,j]>0))),rank_min=float(rnk[:,j].min()),rank_max=float(rnk[:,j].max()),rms_prediction_change=float(influence[j]),influence_rank=float(ranks(influence)[j]),mean_standardized_slope=float(G[m,:,l,j].mean()),rms_standardized_slope=float(gr[j]),positive_gradient_fraction=float(GP[m,:,l,j].mean())))
        for unit,mask in [('between_lineages19',eligible),('between_lineages12',strict)]:
            for kind,v in [('predictive_importance',mean_lineage_D[m,mask]),('prediction_influence',influence_lineage[m,mask]),('standardized_sensitivity',gradient_lineage[m,mask])]:rank_stats.append(pair_stats(v,model,kind,unit,labels[mask],pair_rows))
    summary=table('gene_importance_summary.tsv',summaries);perlineage=table('per_lineage_importance.tsv',lineage_rows);table('per_repeat_importance.tsv',repeat_rows);table('rank_correlations.tsv',pair_rows);table('rank_correlation_summary.tsv',rank_stats)
    # Same test-lineage population for both donor reference definitions.
    sensitivity=[];sensitivity_summary=[];w=n*testeligible
    for m,model in enumerate(FAMILIES):
        train=(D[m].mean(axis=0)*w[:,None]).sum(axis=0)/w.sum();test=(T[m].mean(axis=0)*w[:,None]).sum(axis=0)/w.sum();rt=ranks(train);rp=ranks(test)
        for j,gene in enumerate(genes):sensitivity.append(dict(model=model,gene=str(gene),train_reference_delta_mse=float(train[j]),test_permutation_delta_mse=float(test[j]),train_reference_rank=float(rt[j]),test_permutation_rank=float(rp[j])))
        sensitivity_summary.append(dict(model=model,lineages=20,rank_correlation=corr(train,test),top10_overlap=len(set(genes[rt<=10])&set(genes[rp<=10]))))
    table('donor_reference_sensitivity.tsv',sensitivity);table('donor_reference_sensitivity_summary.tsv',sensitivity_summary)
    cross=[]
    for measure,key in [('predictive_importance','delta'),('prediction_influence','influence'),('standardized_sensitivity','gradient')]:cross.append(dict(scope='pooled23',measure=measure,spearman=corr(global_values['ridge'][key],global_values['neural_network'][key])))
    for l,label in enumerate(labels):
        for measure,v in [('predictive_importance',mean_lineage_D),('prediction_influence',influence_lineage),('standardized_sensitivity',gradient_lineage)]:cross.append(dict(scope=str(label),measure=measure,spearman=corr(v[0,l],v[1,l])))
    table('cross_model_rank_correlations.tsv',cross)
    # Correlations describe redundancy in the measured expression panel, without responses.
    centered=x.copy()
    for label in labels:centered[lineage==label]-=centered[lineage==label].mean(axis=0)
    correlations=[]
    for scope,xx in [('pooled',x),('within_lineage_centered',centered)]:
        cc=np.corrcoef(xx,rowvar=False);np.fill_diagonal(cc,0)
        for j,gene in enumerate(genes):
            k=int(np.argmax(np.abs(cc[j])));correlations.append(dict(scope=scope,gene=str(gene),strongest_partner=str(genes[k]),pearson=float(cc[j,k])))
    table('strongest_gene_correlations.tsv',correlations)
    coeff=np.array(ridge_coefs);cr=ranks(np.abs(coeff));table('ridge_coefficients_by_split.tsv',[dict(repeat=r+1,gene=str(gene),standardized_coefficient=float(coeff[r,j]),absolute_coefficient_rank=float(cr[r,j])) for r in range(R) for j,gene in enumerate(genes)])
    gradient_summary=summary[summary.scope=='pooled23'][['model','gene','mean_standardized_slope','mean_absolute_standardized_slope','rms_standardized_slope','slope_rms_rank','positive_gradient_fraction','positive_seed_slopes']].copy();table('standardized_sensitivity_summary.tsv',gradient_summary)
    # Independently expand lineage bootstrap picks to audit weighted aggregation.
    be=0.
    for model,m in [('ridge',0),('neural_network',1)]:
        for b in range(5):
            picks=draws23[b];expected=np.array([np.average(D[m,r,picks],axis=0,weights=n[picks]) for r in range(R)]).mean(axis=0);actual=bootstrap_arrays[model+'__pooled23__delta_mse'][b];be=max(be,float(np.max(np.abs(expected-actual))));assert be<1e-12
    maxima={k:max(v[k] for v in cv['checks']) for k in ['saved_prediction_error','native_replacement_error','identity_replacement_error','finite_difference_gradient_error','exhaustive_small_test_permutation_error','full_donor_matrix_error']}
    validation=dict(passed=True,no_refitting=True,saved_models=40,genes=64,repeats=20,lineages=23,eligible_consistency_lineages=19,strict_consistency_lineages=12,secondary_test_permutation_lineages=20,
        max_cell_lineage_aggregation_error=max_aggregate,bootstrap_expansion_error=be,source_hashes_verified=True,all_replacements_within_lineage=True,training_donor_and_test_identity_groups_disjoint=True,**maxima)
    calc.dump(OUT/'validation.json',validation)
    np.savez_compressed(OUT/'importance_arrays.npz',genes=genes,labels=labels,test_n=n,cohort_n=cohort,train_reference_delta_mse=D,train_reference_mean_squared_prediction_shift=S,test_reference_delta_mse=T,
        test_reference_mean_squared_prediction_shift=TS,standardized_gradient_mean=G,standardized_gradient_rms=np.sqrt(G2),baseline_mse=B,ridge_coefficients=coeff)
    np.savez_compressed(OUT/'bootstrap_draws.npz',lineage_picks23=draws23,lineage_picks19=draws19,**bootstrap_arrays)
    primary=summary[summary.scope=='pooled23'];result=dict(top10_predictive={model:primary[primary.model==model].sort_values('rank').head(10).gene.tolist() for model in FAMILIES},
        top10_sensitivity={model:primary[primary.model==model].sort_values('slope_rms_rank').head(10).gene.tolist() for model in FAMILIES},rank_stability=rank_stats,cross_model_correlations=cross[:3],donor_reference_sensitivity=sensitivity_summary,
        positive_pointwise_intervals={model:primary[(primary.model==model)&(primary.conditional_ci_low>0)].sort_values('rank').gene.tolist() for model in FAMILIES})
    calc.dump(OUT/'summary.json',result)
    from report_importance import render
    render(summary,perlineage,pd.DataFrame(rank_stats),pd.DataFrame(cross),pd.DataFrame(sensitivity_summary),result,validation)
    for a in json.load(open(OUT/'source_manifest.json'))['files']:assert calc.sha(calc.WS/a['path'])==a['sha256']
    am=json.load(open(OUT/'analysis_source_manifest.json'));assert calc.sha(Path(__file__))==am['review_sha256'] and calc.sha(OUT/'report_importance.py')==am['report_sha256']
    calc.dump(OUT/'progress.json',dict(phase='complete',validation_passed=True,saved_models=40,genes=64))
    manifest=dict(files=[dict(path=str(p.relative_to(OUT)),sha256=calc.sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts]);calc.dump(OUT/'artifact_manifest.json',manifest)
    archive=OUT.parent/'erastin_stratified_gene_importance_results.zip'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for row in manifest['files']:z.write(OUT/row['path'],'erastin_stratified_gene_importance/'+row['path'])
        z.write(OUT/'artifact_manifest.json','erastin_stratified_gene_importance/artifact_manifest.json')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in manifest['files']:assert hashlib.sha256(z.read('erastin_stratified_gene_importance/'+row['path'])).hexdigest()==row['sha256']
    print(json.dumps(dict(event='complete',archive_bytes=archive.stat().st_size,summary=result),indent=2),flush=True)
if __name__=='__main__':main()
