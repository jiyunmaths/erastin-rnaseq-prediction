#!/usr/bin/env python3
"""Matched held-out importance and selection stability for 64-gene saved models."""
import hashlib,json,itertools,sys,time,zipfile
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.stats import rankdata,spearmanr
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;WS=OUT.parent.parent
EN=WS/'outputs/erastin_elastic_net_comparison';OLD=WS/'outputs/erastin_stratified_gene_importance'
MODELS=['elastic_net','ridge','neural_network']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
    if isinstance(v,np.generic):return v.item()
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def dump(p,v):p.write_text(json.dumps(clean(v),indent=2,allow_nan=False)+'\n')
def table(name,rows):
    a=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows);a.to_csv(OUT/name,sep='\t',index=False,float_format='%.12g');return a
def ranks(v):return rankdata(-np.asarray(v),axis=-1,method='average')
def corr(a,b):return float(spearmanr(a,b).statistic) if np.ptp(a)>0 and np.ptp(b)>0 else np.nan

def prepare():
    source=[]
    for root,names in [(EN,['panel_inputs.npz','shared_splits.npz','predictions.npz','protocol.json','validation.json','elastic_net_gene_selection.tsv','elastic_net_coefficients.tsv','selected_parameters.tsv']),
                       (OLD,['analysis_inputs.npz','importance_arrays.npz','gene_importance_summary.tsv','standardized_sensitivity_summary.tsv','per_lineage_importance.tsv','bootstrap_draws.npz','protocol.json','validation.json','source_manifest.json'])]:
        known={v['path']:v['sha256'] for v in json.load(open(root/'artifact_manifest.json'))['files']}
        if root==EN:names+= [f'folds/{m}__p64__r{r+1:02d}.{ext}' for m in MODELS for r in range(20) for ext in ['json','npz','joblib']]
        for name in names:
            assert sha(root/name)==known[name];source.append(dict(path=str((root/name).relative_to(WS)),sha256=known[name]))
    p=dict(status='fixed_before_new_importance_calculation',created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),models=MODELS,panel=64,repeats=20,no_refitting=True,
        split_design='Reuse exact 20 lineage-stratified grouped 80/20 splits: 595 train, 149 test, all 23 lineages in both; Cervix excluded.',
        selection='Nonzero coefficient in each saved outer-training fit. Always selected=20/20; frequent=at least 18/20, descriptive thresholds only. Check signed stability separately.',
        primary='For each held-out receiver and each gene, replace that gene with every same-lineage training donor value; mean increase in held-out squared error, then cell-weight over 149 receivers and average over 20 splits. Analytic donor moments for linear Elastic Net; verify against exhaustive donor-pair enumeration.',
        secondary='Equal-lineage and >=10-cohort-line sensitivity summaries; RMS prediction change and RMS local derivative per training expression SD are separate from predictive importance.',
        agreement='Compare identical held-out replacement ranks across all 64 genes using Spearman, top5/top10 overlap, split and lineage agreement. Compare signed mean coefficients/local derivatives only on matched standardized expression units.',
        uncertainty='Reuse the exact existing 5,000 whole-lineage bootstrap draws for all three models. Conditional pointwise intervals; fixed fits, donors and panels, no refitting or selection/multiplicity correction.',
        limitations=['Correlated genes may share coefficients; replacement disrupts gene correlations and can produce atypical expression combinations. It is within-lineage marginal replacement, not full conditional permutation.',
            'Selection across overlapping random splits is descriptive stability, not statistical significance, independent biological replication or causal evidence.',
            'A linear coefficient vector is common to all lineages in a split. Similar signed slopes do not establish uniform held-out prediction benefit across lineages.',
            'The five-gene candidate panel and prior model choices were informed by this cohort. No new reduced-panel model is fitted or independently validated here.'],
        scripts={name:sha(OUT/name) for name in ['analyze_agreement.py','report_agreement.py']})
    pp=OUT/'protocol.json'
    if pp.exists():old=json.load(open(pp));p['created_utc']=old['created_utc'];assert old==clean(p)
    else:dump(pp,p)
    dump(OUT/'source_manifest.json',dict(files=source));return p

def calculate():
    a=np.load(EN/'panel_inputs.npz');sp=np.load(EN/'shared_splits.npz');oldinputs=np.load(OLD/'analysis_inputs.npz');old=np.load(OLD/'importance_arrays.npz');preds=np.load(EN/'predictions.npz')
    x=a['x_64'];y=a['y'];genes=a['genes_64'];lin=a['lineage'];groups=a['group_ids'];labels=old['labels'];R=20;L=len(labels);P=len(genes)
    for key,oldkey in [('x_64','x'),('genes_64','genes'),('y','y'),('sample_ids','sample_ids'),('group_ids','groups'),('lineage','lineage')]:np.testing.assert_array_equal(a[key],oldinputs[oldkey])
    np.testing.assert_array_equal(sp['outer_test'],oldinputs['outer_test']);np.testing.assert_array_equal(genes,old['genes']);assert (R,L,P,len(y))==(20,23,64,744)
    source_old={v['path']:v['sha256'] for v in json.load(open(OLD/'source_manifest.json'))['files']}
    for k,m in enumerate(['ridge','neural_network']):
        np.testing.assert_array_equal(preds[m+'__64'],oldinputs['saved_predictions'][k])
        for r in range(R):
            rel=f'outputs/erastin_stratified_panel_comparison/folds/{m}__p64__r{r+1:02d}.joblib';assert sha(EN/'folds'/Path(rel).name)==source_old[rel]
    D=np.empty((3,R,L,P));S=np.empty_like(D);G=np.empty_like(D);G2=np.empty_like(D);B=np.empty((3,R,L));N=np.empty((R,L),int);coef=[];checks=[];pair_checks=0
    D[1:]=old['train_reference_delta_mse'];S[1:]=old['train_reference_mean_squared_prediction_shift'];G[1:]=old['standardized_gradient_mean'];G2[1:]=old['standardized_gradient_rms']**2;B[1:]=old['baseline_mse']
    for r in range(R):
        tr=np.flatnonzero(~sp['outer_test'][r]);te=np.flatnonzero(sp['outer_test'][r]);assert set(groups[tr]).isdisjoint(groups[te]);assert set(lin[tr])==set(lin[te])==set(labels);pair_checks+=1
        root=EN/'folds'/f'elastic_net__p64__r{r+1:02d}';saved=joblib.load(root.with_suffix('.joblib'));arr=np.load(root.with_suffix('.npz'));meta=json.load(open(root.with_suffix('.json')))
        np.testing.assert_array_equal(arr['train_rows'],tr);np.testing.assert_array_equal(arr['test_rows'],te);sx=saved['x_scaler'];sy=saved['y_scaler'];model=saved['model'];z=sx.transform(x);beta=model.coef_*sy.scale_[0];coef.append(beta)
        np.testing.assert_allclose(sx.mean_,x[tr].mean(axis=0),rtol=0,atol=1e-12);np.testing.assert_allclose(sx.scale_,x[tr].std(axis=0),rtol=0,atol=1e-12);np.testing.assert_allclose(sy.mean_,[y[tr].mean()],rtol=0,atol=1e-12)
        prediction=sy.inverse_transform(model.predict(z[te])[:,None]).ravel();err=float(np.max(np.abs(prediction-preds['elastic_net__64'][r,te])));assert err<1e-12
        error=prediction-y[te];G[0,r]=beta;G2[0,r]=beta**2;en_error=native_error=shift_error=0.;native_checks=0
        for l,label in enumerate(labels):
            recv=np.flatnonzero(lin[te]==label);t=te[recv];d=tr[lin[tr]==label];assert len(t)>0 and len(d)>0 and set(groups[t]).isdisjoint(groups[d]);pair_checks+=1;N[r,l]=len(t)
            md=z[d].mean(axis=0)-z[t];second=md**2+z[d].var(axis=0)
            changes=2*error[recv,None]*beta[None,:]*md+beta[None,:]**2*second
            D[0,r,l]=changes.mean(axis=0);S[0,r,l]=(beta[None,:]**2*second).mean(axis=0);B[0,r,l]=np.mean(error[recv]**2)
            diff=z[d][None,:,:]-z[t][:,None,:];replacement=prediction[recv,None,None]+diff*beta[None,None,:];delta=((replacement-y[t,None,None])**2-error[recv,None,None]**2).mean(axis=(0,1));shift=((replacement-prediction[recv,None,None])**2).mean(axis=(0,1))
            en_error=max(en_error,float(np.max(np.abs(delta-D[0,r,l]))));shift_error=max(shift_error,float(np.max(np.abs(shift-S[0,r,l]))));assert max(en_error,shift_error)<1e-12
            if r in [0,19] and label in ['Lung','Thyroid','Ampulla of Vater']:
                for j in [0,9,17,63]:
                    xx=np.repeat(x[t],len(d),axis=0);xx[:,j]=np.tile(x[d,j],len(t));ref=sy.inverse_transform(model.predict(sx.transform(xx))[:,None]).ravel().reshape(len(t),len(d));native_error=max(native_error,float(np.max(np.abs(ref-replacement[:,:,j]))));native_checks+=1
            assert native_error<1e-12
        first=x[te[0]];eps=1e-5;plus=np.repeat(first[None,:],P,axis=0);minus=plus.copy();plus[np.arange(P),np.arange(P)]+=eps*sx.scale_;minus[np.arange(P),np.arange(P)]-=eps*sx.scale_
        fd=(model.predict(sx.transform(plus))-model.predict(sx.transform(minus)))*sy.scale_[0]/(2*eps);graderr=float(np.max(np.abs(fd-beta)));assert graderr<1e-8
        checks.append(dict(repeat=r+1,saved_prediction_error=err,all_gene_donor_pair_enumeration_error=en_error,prediction_shift_enumeration_error=shift_error,native_replacement_error=native_error,native_full_matrix_checks=native_checks,finite_difference_gradient_error=graderr,selected_genes=int(np.count_nonzero(beta)),selected_parameters=meta['selected_parameters']))
    assert np.all(N==N[0]);np.testing.assert_array_equal(N[0],old['test_n']);assert pair_checks==480
    coef=np.array(coef);records=[]
    original_selection=pd.read_csv(EN/'elastic_net_gene_selection.tsv',sep='\t').query('panel==64').set_index('gene')
    for j,gene in enumerate(genes):
        v=coef[:,j];positive=int(np.sum(v>0));negative=int(np.sum(v<0));n=positive+negative;ref=original_selection.loc[gene]
        assert n==ref.selected_fits and positive==ref.positive_fits and negative==ref.negative_fits;np.testing.assert_allclose(np.mean(np.abs(v)),ref.mean_absolute_coefficient_original_y_per_training_sd,rtol=0,atol=1e-12)
        records.append(dict(gene=str(gene),selected_fits=n,total_fits=20,selection_frequency=n/20,positive_fits=positive,negative_fits=negative,sign_agreement_when_selected=max(positive,negative)/n if n else np.nan,mean_coefficient=v.mean(),mean_absolute_coefficient=np.abs(v).mean(),rms_coefficient=np.sqrt(np.mean(v*v)),mean_abs_coefficient_rank=float(ranks(np.abs(coef).mean(axis=0))[j])))
    selection=table('elastic_net_selection.tsv',pd.DataFrame(records).sort_values(['selected_fits','mean_absolute_coefficient'],ascending=False))
    np.savez_compressed(OUT/'importance_arrays.npz',models=MODELS,genes=genes,labels=labels,test_n=N[0],cohort_n=old['cohort_n'],train_reference_delta_mse=D,train_reference_mean_squared_prediction_shift=S,standardized_gradient_mean=G,standardized_gradient_rms=np.sqrt(G2),baseline_mse=B,elastic_net_coefficients=coef)
    validation=dict(passed=True,no_refitting=True,new_model_importance_calculations=20,reused_model_importance_calculations=40,total_saved_models=60,all_gene_donor_pair_moment_checks=20*23*64,train_test_identity_exclusion_checks=pair_checks,byte_identical_splits_and_features=True,reused_models_and_predictions_exactly_match_importance_sources=True,all_selection_counts_and_signs_match_previous_results=True,checks=checks)
    return D,S,G,G2,B,N[0],old['cohort_n'],genes,labels,selection,validation

def summarize(D,S,G,G2,B,n,cohort,genes,labels,selection,validation):
    previous=pd.read_csv(OLD/'gene_importance_summary.tsv',sep='\t');oldboot=np.load(OLD/'bootstrap_draws.npz');draws23=oldboot['lineage_picks23'];draws19=oldboot['lineage_picks19'];eligible=cohort>=10;strict=n>=5
    assert draws23.shape==(5000,23) and draws19.shape==(5000,19) and eligible.sum()==19 and strict.sum()==12
    scopes={'pooled23':n.astype(float),'macro23':np.ones(23),'pooled19':n*eligible,'macro19':eligible.astype(float)};summaries=[];repeat_rows=[];bootstrap_arrays={};global_values={};maxboot=maxaggregate=0.
    for scope,w in scopes.items():
        keep=np.flatnonzero(w>0);pick=draws23 if len(keep)==23 else draws19;weights=w[keep][pick];den=weights.sum(axis=1)
        for m,model in enumerate(MODELS):
            perseed=np.einsum('rlg,l->rg',D[m],w)/w.sum();point=perseed.mean(axis=0);rank=ranks(point);seedrank=ranks(perseed);influence=np.sqrt(np.einsum('rlg,l->rg',S[m],w).mean(axis=0)/w.sum());gradient=np.sqrt(np.einsum('rlg,l->rg',G2[m],w).mean(axis=0)/w.sum());signed=np.einsum('rlg,l->rg',G[m],w).mean(axis=0)/w.sum();base=B[m]@w/w.sum()
            selected=D[m].mean(axis=0)[keep];boot=(selected[pick]*weights[:,:,None]).sum(axis=1)/den[:,None];lo,hi=np.percentile(boot,[2.5,97.5],axis=0);bootstrap_arrays[model+'__'+scope+'__delta_mse']=boot
            for b in range(5):
                ix=keep[pick[b]];expected=np.array([np.average(D[m,r,ix],axis=0,weights=w[ix]) for r in range(20)]).mean(axis=0);maxboot=max(maxboot,float(np.max(np.abs(expected-boot[b]))));assert maxboot<1e-12
            if model!='elastic_net':
                ref=previous[(previous.model==model)&(previous.scope==scope)].set_index('gene').loc[genes];maxaggregate=max(maxaggregate,float(np.max(np.abs(ref.delta_mse.to_numpy()-point))));assert maxaggregate<1e-12;np.testing.assert_array_equal(ref['rank'],rank);np.testing.assert_allclose(boot,oldboot[model+'__'+scope+'__delta_mse'],rtol=0,atol=1e-12)
            for j,gene in enumerate(genes):summaries.append(dict(model=model,scope=scope,gene=str(gene),rank=float(rank[j]),delta_mse=float(point[j]),relative_mse_increase_percent=float(100*point[j]/base.mean()),conditional_ci_low=float(lo[j]),conditional_ci_high=float(hi[j]),positive_repeats=int(np.sum(perseed[:,j]>0)),top10_repeats=int(np.sum((seedrank[:,j]<=10)&(perseed[:,j]>0))),rank_median=float(np.median(seedrank[:,j])),positive_lineages_23=int(np.sum(D[m].mean(axis=0)[:,j]>0)),positive_lineages_19=int(np.sum(D[m].mean(axis=0)[eligible,j]>0)),rms_prediction_change=float(influence[j]),mean_standardized_slope=float(signed[j]),rms_standardized_slope=float(gradient[j]),slope_rms_rank=float(ranks(gradient)[j])))
            if scope=='pooled23':
                global_values[model]=dict(predictive_importance=point,prediction_influence=influence,standardized_sensitivity=gradient,perseed=perseed,signed=signed)
                for r in range(20):
                    for j,gene in enumerate(genes):repeat_rows.append(dict(model=model,repeat=r+1,gene=str(gene),delta_mse=float(perseed[r,j]),rank=float(seedrank[r,j])))
    # Verify lineage indexing explicitly (avoid mixing advanced-index axis order).
    for m,model in enumerate(MODELS):
        expected=np.sum(D[m].mean(axis=0)[eligible]>0,axis=0);found=[v['positive_lineages_19'] for v in summaries if v['model']==model and v['scope']=='pooled23'];np.testing.assert_array_equal(found,expected)
    summary=table('gene_importance_summary.tsv',summaries);table('per_repeat_importance.tsv',repeat_rows);lineage=[]
    for m,model in enumerate(MODELS):
        avg=D[m].mean(axis=0);lr=ranks(avg)
        for l,label in enumerate(labels):
            for j,gene in enumerate(genes):lineage.append(dict(model=model,lineage=str(label),gene=str(gene),test_n=int(n[l]),cohort_n=int(cohort[l]),delta_mse=float(avg[l,j]),rank=float(lr[l,j]),positive_repeats=int(np.sum(D[m,:,l,j]>0))))
    perlineage=table('per_lineage_importance.tsv',lineage);cross=[];stability=[];overlap=[]
    for first,ref in itertools.combinations(MODELS,2):
        for kind in ['predictive_importance','prediction_influence','standardized_sensitivity']:cross.append(dict(first_model=first,reference_model=ref,scope='pooled23',measure=kind,spearman=corr(global_values[first][kind],global_values[ref][kind])))
        for r in range(20):cross.append(dict(first_model=first,reference_model=ref,scope='repeat_'+str(r+1),measure='predictive_importance',spearman=corr(global_values[first]['perseed'][r],global_values[ref]['perseed'][r])))
        for l,label in enumerate(labels):cross.append(dict(first_model=first,reference_model=ref,scope=str(label),measure='predictive_importance',spearman=corr(D[MODELS.index(first),:,l].mean(axis=0),D[MODELS.index(ref),:,l].mean(axis=0))))
        for k in [5,10,20]:
            a=set(genes[ranks(global_values[first]['predictive_importance'])<=k]);b=set(genes[ranks(global_values[ref]['predictive_importance'])<=k]);overlap.append(dict(first_model=first,reference_model=ref,top_k=k,overlap=len(a&b),shared_genes=', '.join(sorted(a&b))))
    for m,model in enumerate(MODELS):
        for unit,values in [('between_splits',global_values[model]['perseed']),('between_lineages19',D[m].mean(axis=0)[eligible]),('between_lineages12',D[m].mean(axis=0)[strict])]:
            vv=[corr(values[i],values[j]) for i,j in itertools.combinations(range(len(values)),2)];stability.append(dict(model=model,unit=unit,measure='predictive_importance',pairs=len(vv),median=float(np.median(vv)),q25=float(np.percentile(vv,25)),q75=float(np.percentile(vv,75))))
    cross=table('cross_model_rank_correlations.tsv',cross);table('top_gene_overlap.tsv',overlap);table('rank_stability.tsv',stability)
    pooled=summary[summary.scope=='pooled23'];combined=selection.set_index('gene')
    for model in MODELS:
        d=pooled[pooled.model==model].set_index('gene');combined=combined.join(d[['rank','delta_mse','relative_mse_increase_percent','conditional_ci_low','conditional_ci_high','positive_repeats','top10_repeats','positive_lineages_19','mean_standardized_slope','slope_rms_rank']].add_prefix(model+'__'))
    combined=table('gene_agreement.tsv',combined.reset_index().sort_values(['selected_fits','elastic_net__rank'],ascending=[False,True]))
    np.savez_compressed(OUT/'bootstrap_draws.npz',lineage_picks23=draws23,lineage_picks19=draws19,**bootstrap_arrays)
    validation.update(bootstrap_expansion_max_error=maxboot,reused_importance_score_max_error=maxaggregate,original_bootstrap_and_ranks_exactly_reproduced=True);dump(OUT/'validation.json',validation)
    result=dict(always_selected=selection[selection.selected_fits==20].gene.tolist(),selected_19_of_20=selection[selection.selected_fits==19].gene.tolist(),selected_18_of_20=selection[selection.selected_fits==18].gene.tolist(),top10_predictive={m:pooled[pooled.model==m].sort_values('rank').head(10).gene.tolist() for m in MODELS},positive_pointwise_intervals={m:pooled[(pooled.model==m)&(pooled.conditional_ci_low>0)].sort_values('rank').gene.tolist() for m in MODELS},global_agreement=cross[cross.scope=='pooled23'].to_dict('records'),stability=stability,top_overlap=overlap)
    dump(OUT/'summary.json',result)
    from report_agreement import render
    render(combined,summary,perlineage,cross,result,validation)
    return result

def main():
    p=prepare();v=calculate();result=summarize(*v)
    for a in json.load(open(OUT/'source_manifest.json'))['files']:assert sha(WS/a['path'])==a['sha256']
    for name,h in p['scripts'].items():assert sha(OUT/name)==h
    dump(OUT/'progress.json',dict(phase='complete',validation_passed=True,models=60,refitting=False));seal();print(json.dumps(clean(result),indent=2),flush=True)

def seal():
    manifest=dict(files=[dict(path=str(p.relative_to(OUT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts]);dump(OUT/'artifact_manifest.json',manifest);archive=OUT.parent/'erastin_elastic_net_gene_agreement_results.zip';prefix=OUT.name+'/'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for row in manifest['files']:z.write(OUT/row['path'],prefix+row['path'])
        z.write(OUT/'artifact_manifest.json',prefix+'artifact_manifest.json')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in manifest['files']:assert hashlib.sha256(z.read(prefix+row['path'])).hexdigest()==row['sha256']
    print(json.dumps(dict(archive_bytes=archive.stat().st_size,files=len(manifest['files'])+1)),flush=True)
if __name__=='__main__':main()
