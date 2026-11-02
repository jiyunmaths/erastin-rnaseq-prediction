#!/usr/bin/env python3
"""Compare predictive gene reliance, signed sensitivities and lineage consistency."""
import json,itertools,hashlib,zipfile
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import rankdata,spearmanr
import calculate_importance as calc
OUT=calc.OUT;MODELS=calc.MODELS;DATASETS=['CTRP','PRISM']
def ranks(v):return rankdata(-np.asarray(v),axis=-1,method='average')
def corr(a,b):return float(spearmanr(a,b).statistic) if np.ptp(a)>0 and np.ptp(b)>0 else np.nan
def table(name,rows):return calc.table(name,rows)
def stat_pairs(values,dataset,model,unit,rows):
    vv=[]
    for i,j in itertools.combinations(range(len(values)),2):
        v=corr(values[i],values[j]);rows.append(dict(dataset=dataset,model=model,unit=unit,first=i,second=j,spearman=v))
        if np.isfinite(v):vv.append(v)
    return dict(dataset=dataset,model=model,unit=unit,pairs=len(vv),median=float(np.median(vv)),q25=float(np.percentile(vv,25)),q75=float(np.percentile(vv,75)))

def aggregate(ds,protocol):
    s=calc.STATE[ds];labels=s['labels'];lineage=s['lineage'];nrow=len(s['y']);R=20;L=len(labels);P=64
    out={k:np.empty((3,R,L,P)) for k in ['D','S','T','TS','G','GA','G2','GP']};out['B']=np.empty((3,R,L));N=np.empty((R,L),int);cell={k:np.zeros((3,nrow,P)) for k in ['D','S','G','G2']};cell['B']=np.zeros((3,nrow));celln=np.zeros(nrow,int);coef=np.empty((2,R,P));maxagg=0.;checks=[]
    for r in range(R):
        root=OUT/'repeats'/f'{ds.lower()}__r{r+1:02d}';meta=json.load(open(root.with_suffix('.json')));assert meta['protocol_sha256']==calc.sha(OUT/'protocol.json') and calc.sha(root.with_suffix('.npz'))==meta['array_sha256'];a=np.load(root.with_suffix('.npz'));te=a['test_rows'];np.testing.assert_array_equal(te,np.flatnonzero(s['test'][r]));np.testing.assert_array_equal(a['baseline_prediction'],s['saved_predictions'][:,r,te]);celln[te]+=1
        raw=dict(D=a['train_reference_delta_mse'],S=a['train_reference_mean_squared_prediction_shift'],T=a['test_reference_delta_mse'],TS=a['test_reference_mean_squared_prediction_shift'],G=a['standardized_gradient']);raw['GA']=np.abs(raw['G']);raw['G2']=raw['G']**2;raw['GP']=(raw['G']>0).astype(float)
        for k in ['D','S','G','G2']:cell[k][:,te,:]+=raw[k]
        cell['B'][:,te]+=a['baseline_squared_error'];coef[0,r]=a['ridge_coefficients'];coef[1,r]=a['elastic_net_coefficients']
        for l,label in enumerate(labels):
            pos=np.flatnonzero(lineage[te]==label);N[r,l]=len(pos);assert len(pos)>0
            for k in out:
                if k!='B':out[k][:,r,l]=raw[k][:,pos].mean(axis=1)
            out['B'][:,r,l]=a['baseline_squared_error'][:,pos].mean(axis=1)
        reconstructed=(out['D'][:,r]*N[r][None,:,None]).sum(axis=1)/N[r].sum();maxagg=max(maxagg,float(np.max(np.abs(reconstructed-raw['D'].mean(axis=1)))));assert maxagg<1e-12;checks.append(meta)
    assert np.all(N==N[0]);np.testing.assert_array_equal(celln,s['test'].sum(axis=0));n=N[0];cohort=np.array([np.sum(lineage==l) for l in labels])
    for k,v in cell.items():
        den=celln[None,:,None] if v.ndim==3 else celln[None,:];cell[k]=np.divide(v,den,out=np.full(v.shape,np.nan),where=den>0)
    out.update(labels=labels,test_n=n,cohort_n=cohort,cell=cell,celln=celln,coefficients=coef,checks=checks,max_aggregation_error=maxagg)
    np.savez_compressed(OUT/(ds.lower()+'_importance_arrays.npz'),**{k:v for k,v in out.items() if k not in ['cell','checks']},**{'cell_mean_'+k:v for k,v in cell.items()})
    return out

def main():
    calc.initialize();protocol=json.load(open(OUT/'protocol.json'));assert calc.sha(OUT/'calculate_importance.py')==protocol['script_sha256'];cv=json.load(open(OUT/'calculation_validation.json'));assert cv['passed'] and cv['saved_models']==120
    calc.dump(OUT/'analysis_source_manifest.json',dict(review_sha256=calc.sha(Path(__file__))));data={ds:aggregate(ds,protocol) for ds in DATASETS};genes=calc.STATE['PRISM']['genes'];native_old=np.load(calc.WS/'outputs/erastin_elastic_net_gene_agreement/importance_arrays.npz');previous=pd.read_csv(calc.WS/'outputs/erastin_elastic_net_gene_agreement/gene_importance_summary.tsv',sep='\t');maxold=0.
    for m,f in enumerate(MODELS):
        oldm=list(native_old['models']).index(f)
        for key,oldkey in [('D','train_reference_delta_mse'),('S','train_reference_mean_squared_prediction_shift'),('G','standardized_gradient_mean'),('B','baseline_mse')]:
            err=float(np.max(np.abs(data['CTRP'][key][m]-native_old[oldkey][oldm])));maxold=max(maxold,err);assert err<1e-12
    common=np.array(protocol['common_large_lineages']);strict=np.array(protocol['common_strict_lineages']);matched=np.array(protocol['matched_cell_ids']);old_draws=np.load(calc.WS/'outputs/erastin_stratified_gene_importance/bootstrap_draws.npz');draws_common=np.random.default_rng(protocol['uncertainty']['seed']+2).integers(13,size=(5000,13));summaries=[];repeats=[];perlineage=[];selection=[];coeffrows=[];bootstrap_arrays={};vectors={};stability=[];pairrows=[];booterror=0.;matched_meta=[];reference_sensitivity=[]
    for ds in DATASETS:
        d=data[ds];s=calc.STATE[ds];labels=d['labels'];n=d['test_n'];cohort=d['cohort_n'];D=d['D'];S=d['S'];G=d['G'];G2=d['G2'];B=d['B'];eligible=cohort>=10;cm=np.isin(labels,common);sm=np.isin(labels,strict);assert np.sum(cm)==13 and np.sum(sm)==7;np.testing.assert_array_equal(labels[cm],common)
        scopes={'native_pooled':n.astype(float),'native_macro':np.ones(len(labels)),'common13_macro':cm.astype(float)};native_picks=old_draws['lineage_picks23'] if ds=='CTRP' else np.random.default_rng(protocol['uncertainty']['seed']).integers(len(labels),size=(5000,len(labels)));vectors[ds]={};meanL=D.mean(axis=1);rankL=ranks(meanL)
        for scope,w in scopes.items():
            keep=np.flatnonzero(w>0);picks=draws_common if scope=='common13_macro' else native_picks;weights=w[keep][picks] if scope!='native_macro' else None
            for m,f in enumerate(MODELS):
                value=np.einsum('rlg,l->rg',D[m],w)/w.sum();point=value.mean(axis=0);rank=ranks(point);seedrank=ranks(value);baseline=B[m]@w/w.sum();influence=np.sqrt(np.einsum('rlg,l->rg',S[m],w).mean(axis=0)/w.sum());signed=np.einsum('rlg,l->rg',G[m],w).mean(axis=0)/w.sum();absolute=np.einsum('rlg,l->rg',d['GA'][m],w).mean(axis=0)/w.sum();rms=np.sqrt(np.einsum('rlg,l->rg',G2[m],w).mean(axis=0)/w.sum());gp=np.einsum('rlg,l->rg',d['GP'][m],w).mean(axis=0)/w.sum()
                lo=hi=rlo=rhi=np.full(64,np.nan)
                if scope!='native_macro':
                    den=weights.sum(axis=1);selected=D[m].mean(axis=0)[keep];bb=B[m].mean(axis=0)[keep];boot=(selected[picks]*weights[:,:,None]).sum(axis=1)/den[:,None];bbase=(bb[picks]*weights).sum(axis=1)/den;lo,hi=np.percentile(boot,[2.5,97.5],axis=0);rlo,rhi=np.percentile(100*boot/bbase[:,None],[2.5,97.5],axis=0);bootstrap_arrays[ds+'__'+f+'__'+scope+'__delta_mse']=boot;bootstrap_arrays[ds+'__'+f+'__'+scope+'__baseline_mse']=bbase
                    for b in range(5):
                        ix=keep[picks[b]];expected=np.array([np.average(D[m,r,ix],axis=0,weights=w[ix]) for r in range(20)]).mean(axis=0);booterror=max(booterror,float(np.max(np.abs(expected-boot[b]))));assert booterror<1e-12
                vectors[ds].setdefault(scope,{})[f]=dict(delta=point,influence=influence,sensitivity=rms,signed=signed)
                if ds=='CTRP' and scope=='native_pooled':
                    ref=previous[(previous.model==f)&(previous.scope=='pooled23')].set_index('gene').loc[genes];np.testing.assert_array_equal(rank,ref['rank']);maxold=max(maxold,float(np.max(np.abs(point-ref.delta_mse.to_numpy()))));assert maxold<1e-12
                for j,gene in enumerate(genes):summaries.append(dict(dataset=ds,model=f,scope=scope,gene=str(gene),rank=float(rank[j]),delta_mse=float(point[j]),baseline_mse=float(baseline.mean()),relative_mse_increase_percent=float(100*point[j]/baseline.mean()),conditional_ci_low=float(lo[j]),conditional_ci_high=float(hi[j]),relative_ci_low=float(rlo[j]),relative_ci_high=float(rhi[j]),positive_repeats=int(np.sum(value[:,j]>0)),top10_repeats=int(np.sum((seedrank[:,j]<=10)&(value[:,j]>0))),rank_median=float(np.median(seedrank[:,j])),rank_min=float(seedrank[:,j].min()),rank_max=float(seedrank[:,j].max()),positive_native_lineages=int(np.sum(meanL[m,:,j]>0)),positive_native_large_lineages=int(np.sum(meanL[m,eligible,j]>0)),native_large_lineages=int(eligible.sum()),positive_common13=int(np.sum(meanL[m,cm,j]>0)),top10_common13=int(np.sum((rankL[m,cm,j]<=10)&(meanL[m,cm,j]>0))),positive_strict7=int(np.sum(meanL[m,sm,j]>0)),broad_common13_support=bool(np.sum(meanL[m,cm,j]>0)>=11),rms_prediction_change=float(influence[j]),mean_standardized_slope=float(signed[j]),mean_absolute_standardized_slope=float(absolute[j]),rms_standardized_slope=float(rms[j]),sensitivity_rank=float(ranks(rms)[j]),positive_gradient_fraction=float(gp[j])))
                if scope=='native_pooled':
                    for r in range(20):
                        for j,gene in enumerate(genes):repeats.append(dict(dataset=ds,model=f,repeat=r+1,gene=str(gene),delta_mse=float(value[r,j]),rank=float(seedrank[r,j])))
                    stability.append(stat_pairs(value,ds,f,'between_splits',pairrows))
        for m,f in enumerate(MODELS):
            for l,label in enumerate(labels):
                sr=ranks(D[m,:,l]);signed=G[m,:,l].mean(axis=0);rms=np.sqrt(G2[m,:,l].mean(axis=0));influence=np.sqrt(S[m,:,l].mean(axis=0))
                for j,gene in enumerate(genes):perlineage.append(dict(dataset=ds,model=f,lineage=str(label),gene=str(gene),cohort_n=int(cohort[l]),test_n=int(n[l]),common13=bool(cm[l]),strict7=bool(sm[l]),delta_mse=float(meanL[m,l,j]),relative_mse_increase_percent=float(100*meanL[m,l,j]/B[m,:,l].mean()),rank=float(rankL[m,l,j]),positive_repeats=int(np.sum(D[m,:,l,j]>0)),top10_repeats=int(np.sum((sr[:,j]<=10)&(D[m,:,l,j]>0))),mean_standardized_slope=float(signed[j]),rms_standardized_slope=float(rms[j]),rms_prediction_change=float(influence[j])))
            for unit,mask in [('between_native_large_lineages',eligible),('between_common13_lineages',cm),('between_strict7_lineages',sm)]:
                for measure,values in [('predictive_importance',meanL[m,mask]),('standardized_sensitivity',np.sqrt(G2[m].mean(axis=0)[mask]))]:
                    row=stat_pairs(values,ds,f,unit+'__'+measure,pairrows);row['measure']=measure;stability.append(row)
            if f in ['ridge','elastic_net']:
                co=d['coefficients'][['ridge','elastic_net'].index(f)]
                for j,gene in enumerate(genes):
                    v=co[:,j];pos=int(np.sum(v>0));neg=int(np.sum(v<0));nz=pos+neg;selection.append(dict(dataset=ds,model=f,gene=str(gene),selected_fits=nz,positive_fits=pos,negative_fits=neg,sign_agreement_when_selected=max(pos,neg)/nz if nz else np.nan,mean_coefficient=float(v.mean()),mean_absolute_coefficient=float(np.abs(v).mean())))
            for r in range(20):
                val=np.einsum('lg,l->g',G[m,r],n)/n.sum();rms=np.sqrt(np.einsum('lg,l->g',G2[m,r],n)/n.sum())
                for j,gene in enumerate(genes):coeffrows.append(dict(dataset=ds,model=f,repeat=r+1,gene=str(gene),mean_standardized_slope=float(val[j]),rms_standardized_slope=float(rms[j])))
            # Compare donor definitions on exactly the common 13 lineages; none is a singleton test lineage.
            train=meanL[m,cm].mean(axis=0);test=d['T'][m].mean(axis=0)[cm].mean(axis=0);reference_sensitivity.append(dict(dataset=ds,model=f,scope='common13_macro',rank_correlation=corr(train,test),top10_overlap=len(set(genes[(ranks(train)<=10)&(train>0)])&set(genes[(ranks(test)<=10)&(test>0)]))))
        # Same receiver cell IDs, separate valid test appearances, equal weights within lineage then across lineages.
        idx=np.array([int(np.flatnonzero(s['ids']==mid)[0]) for mid in matched]);cl=s['lineage'][idx];assert set(cl)==set(common) and np.all(d['celln'][idx]>0);matched_L={k:np.stack([v[:,idx[cl==l]].mean(axis=1) for l in common],axis=1) for k,v in d['cell'].items()};vectors[ds]['matched_cells_common13_macro']={}
        for j,mid in enumerate(matched):matched_meta.append(dict(dataset=ds,ModelID=str(mid),lineage=str(cl[j]),test_appearances=int(d['celln'][idx[j]]),group_id=str(s['groups'][idx[j]])))
        for m,f in enumerate(MODELS):
            point=matched_L['D'][m].mean(axis=0);base=matched_L['B'][m].mean();signed=matched_L['G'][m].mean(axis=0);rms=np.sqrt(matched_L['G2'][m].mean(axis=0));influence=np.sqrt(matched_L['S'][m].mean(axis=0));boot=matched_L['D'][m][draws_common].mean(axis=1);bb=matched_L['B'][m][draws_common].mean(axis=1);lo,hi=np.percentile(boot,[2.5,97.5],axis=0);rlo,rhi=np.percentile(100*boot/bb[:,None],[2.5,97.5],axis=0);vectors[ds]['matched_cells_common13_macro'][f]=dict(delta=point,influence=influence,sensitivity=rms,signed=signed)
            for j,gene in enumerate(genes):summaries.append(dict(dataset=ds,model=f,scope='matched_cells_common13_macro',gene=str(gene),rank=float(ranks(point)[j]),delta_mse=float(point[j]),baseline_mse=float(base),relative_mse_increase_percent=float(100*point[j]/base),conditional_ci_low=float(lo[j]),conditional_ci_high=float(hi[j]),relative_ci_low=float(rlo[j]),relative_ci_high=float(rhi[j]),mean_standardized_slope=float(signed[j]),rms_standardized_slope=float(rms[j]),rms_prediction_change=float(influence[j])))
    summary=table('gene_importance_summary.tsv',summaries);perlineage=table('per_lineage_importance.tsv',perlineage);table('per_repeat_importance.tsv',repeats);selection=table('coefficient_selection.tsv',selection);table('coefficient_sensitivity_by_repeat.tsv',coeffrows);table('matched_cell_control.tsv',matched_meta);table('donor_reference_sensitivity.tsv',reference_sensitivity);stability=table('rank_stability.tsv',stability);table('rank_correlations.tsv',pairrows)
    # Independently verify new Elastic Net coefficient selections against the completed fit audits.
    for ds in DATASETS:
        ref=pd.read_csv(calc.SOURCES[ds]/'elastic_net_gene_selection.tsv',sep='\t').query('panel==64').set_index('gene');found=selection[(selection.dataset==ds)&(selection.model=='elastic_net')].set_index('gene').loc[genes]
        for key in ['selected_fits','positive_fits','negative_fits']:np.testing.assert_array_equal(found[key],ref.loc[genes,key])
    cross=[];within=[];overlap=[]
    for scope in ['native_pooled','native_macro','common13_macro','matched_cells_common13_macro']:
        for f in MODELS:
            c=vectors['CTRP'][scope][f];p=vectors['PRISM'][scope][f]
            for measure,key in [('predictive_importance','delta'),('prediction_influence','influence'),('standardized_sensitivity','sensitivity')]:cross.append(dict(model=f,scope=scope,measure=measure,spearman=corr(c[key],p[key])))
            for k in [5,10,20]:
                cg=set(genes[(ranks(c['delta'])<=k)&(c['delta']>0)]);pg=set(genes[(ranks(p['delta'])<=k)&(p['delta']>0)]);overlap.append(dict(model=f,scope=scope,top_k=k,shared=len(cg&pg),CTRP_positive_top_k=len(cg),PRISM_positive_top_k=len(pg),shared_genes=', '.join(sorted(cg&pg))))
        for ds in DATASETS:
            for first,ref in itertools.combinations(MODELS,2):within.append(dict(dataset=ds,scope=scope,first_model=first,reference_model=ref,spearman=corr(vectors[ds][scope][first]['delta'],vectors[ds][scope][ref]['delta'])))
    lineagecross=[];lineageprofile=[]
    for f in MODELS:
        m=MODELS.index(f)
        for label in protocol['common_lineages']:
            ci=list(data['CTRP']['labels']).index(label);pi=list(data['PRISM']['labels']).index(label);lineagecross.append(dict(model=f,lineage=label,common13=label in protocol['common_large_lineages'],strict7=label in protocol['common_strict_lineages'],predictive_rank_spearman=corr(data['CTRP']['D'][m,:,ci].mean(axis=0),data['PRISM']['D'][m,:,pi].mean(axis=0))))
        c=data['CTRP']['D'][m].mean(axis=0)[np.isin(data['CTRP']['labels'],common)];p=data['PRISM']['D'][m].mean(axis=0)[np.isin(data['PRISM']['labels'],common)]
        for j,gene in enumerate(genes):lineageprofile.append(dict(model=f,gene=str(gene),common13_lineage_effect_correlation=corr(c[:,j],p[:,j]),positive_in_both_common13=int(np.sum((c[:,j]>0)&(p[:,j]>0))),positive_in_CTRP_only=int(np.sum((c[:,j]>0)&(p[:,j]<=0))),positive_in_PRISM_only=int(np.sum((c[:,j]<=0)&(p[:,j]>0)))))
    cross=table('cross_assay_rank_correlations.tsv',cross);within=table('within_assay_model_agreement.tsv',within);overlap=table('cross_assay_top_gene_overlap.tsv',overlap);lineagecross=table('lineage_cross_assay_rank_agreement.tsv',lineagecross);table('gene_lineage_effect_agreement.tsv',lineageprofile)
    combined=pd.DataFrame(dict(gene=genes)).set_index('gene');primary=summary[summary.scope=='native_pooled']
    for ds in DATASETS:
        for f in MODELS:
            d=primary[(primary.dataset==ds)&(primary.model==f)].set_index('gene');fields=['rank','relative_mse_increase_percent','conditional_ci_low','conditional_ci_high','positive_repeats','top10_repeats','positive_common13','top10_common13','positive_strict7','mean_standardized_slope'];combined=combined.join(d[fields].add_prefix(ds+'__'+f+'__'))
            for scope in ['common13_macro','matched_cells_common13_macro']:
                d=summary[(summary.dataset==ds)&(summary.model==f)&(summary.scope==scope)].set_index('gene');combined=combined.join(d[['rank','relative_mse_increase_percent']].add_prefix(ds+'__'+f+'__'+scope+'__'))
        d=selection[(selection.dataset==ds)&(selection.model=='elastic_net')].set_index('gene');combined=combined.join(d[['selected_fits','positive_fits','negative_fits']].add_prefix(ds+'__elastic_net__'))
    combined=table('gene_comparison.tsv',combined.reset_index());np.savez_compressed(OUT/'bootstrap_draws.npz',common13_picks=draws_common,**bootstrap_arrays)
    maxima={key:max(v[key] for v in cv['checks']) for key in ['saved_prediction_error','native_replacement_error','identity_replacement_error','finite_difference_gradient_error','full_donor_matrix_error']};validation=dict(passed=True,no_refitting=True,saved_models=120,genes=64,datasets=2,repeats_each=20,common_large_lineages=13,common_strict_lineages=7,matched_receiver_cells=290,CTRP_previous_importance_max_error=maxold,previous_CTRP_native_ranks_exactly_reproduced=True,bootstrap_expansion_max_error=booterror,train_test_and_donor_identity_checks=sum(1+len(v['lineage_checks']) for v in cv['checks']),coefficient_selection_counts_and_signs_verified=True,source_hashes_verified=True,**maxima);calc.dump(OUT/'validation.json',validation)
    result=dict(top10_native={ds:{f:primary[(primary.dataset==ds)&(primary.model==f)].sort_values('rank').head(10).gene.tolist() for f in MODELS} for ds in DATASETS},positive_pointwise_native_intervals={ds:{f:primary[(primary.dataset==ds)&(primary.model==f)&(primary.conditional_ci_low>0)].sort_values('rank').gene.tolist() for f in MODELS} for ds in DATASETS},always_selected_EN={ds:selection[(selection.dataset==ds)&(selection.model=='elastic_net')&(selection.selected_fits==20)].sort_values('mean_absolute_coefficient',ascending=False).gene.tolist() for ds in DATASETS},cross_assay_rank_correlations=cross[cross.measure=='predictive_importance'].to_dict('records'),native_model_agreement=within[within.scope=='native_pooled'].to_dict('records'),broad_common13_genes={ds:{f:primary[(primary.dataset==ds)&(primary.model==f)&(primary.positive_common13>=11)].sort_values('rank').gene.tolist() for f in MODELS} for ds in DATASETS})
    calc.dump(OUT/'summary.json',result)
    for row in json.load(open(OUT/'source_manifest.json'))['files']:assert calc.sha(calc.WS/row['path'])==row['sha256']
    assert calc.sha(Path(__file__))==json.load(open(OUT/'analysis_source_manifest.json'))['review_sha256'];calc.dump(OUT/'progress.json',dict(phase='analysis_complete',validation_passed=True,saved_models=120));print(json.dumps(calc.clean(result),indent=2),flush=True)

def seal():
    manifest=dict(files=[dict(path=str(p.relative_to(OUT)),sha256=calc.sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts]);calc.dump(OUT/'artifact_manifest.json',manifest);archive=OUT.parent/'erastin_prism_ctrp_gene_importance_results.zip';prefix=OUT.name+'/'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for row in manifest['files']:z.write(OUT/row['path'],prefix+row['path'])
        z.write(OUT/'artifact_manifest.json',prefix+'artifact_manifest.json')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in manifest['files']:assert hashlib.sha256(z.read(prefix+row['path'])).hexdigest()==row['sha256']
    print(json.dumps(dict(archive_bytes=archive.stat().st_size,files=len(manifest['files'])+1)),flush=True)
if __name__=='__main__':main()
