#!/usr/bin/env python3
"""Frozen CTRP-trained models tested on official PRISM erastin AUC measurements."""
import hashlib,json,itertools,shutil,sys,time,zipfile
from pathlib import Path
import joblib,numpy as np,pandas as pd
from scipy.stats import spearmanr,pearsonr
from metric_helpers import weighted_ranker
sys.dont_write_bytecode=True
OUT=Path(__file__).resolve().parent;WS=OUT.parent.parent;RAW=WS/'work/erastin_prism_validation/raw';SRC=WS/'outputs/erastin_elastic_net_comparison';ORIG=WS/'work/erastin/raw'
FAMILIES=['ridge','elastic_net','neural_network'];PANELS=[5,64];ARMS=[f+'__'+str(p) for f in FAMILIES for p in PANELS];BROAD='BRD-A25004090-001-08-4'
BOOT_KEYS=['spearman','pearson','uncalibrated_auc_rmse','uncalibrated_auc_mae','uncalibrated_auc_r2']

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v
def dump(p,v):p.write_text(json.dumps(clean(v),indent=2,allow_nan=False)+'\n')
def table(name,rows):
    d=rows if isinstance(rows,pd.DataFrame) else pd.DataFrame(rows);d.to_csv(OUT/name,sep='\t',index=False,float_format='%.17g');return d

def source_checks():
    records=[];known={v['path']:v['sha256'] for v in json.load(open(SRC/'artifact_manifest.json'))['files']}
    names=['protocol.json','validation.json','panel_inputs.npz','shared_splits.npz','weighted_mlp.py','metric_helpers.py']+[f'folds/{f}__p{p}__r{r+1:02d}.{ext}' for f in FAMILIES for p in PANELS for r in range(20) for ext in ['json','npz','joblib']]
    for name in names:
        assert sha(SRC/name)==known[name];records.append(dict(path=str((SRC/name).relative_to(WS)),sha256=known[name]))
    download=json.load(open(RAW/'download_manifest.json'));article=json.load(open(RAW/'figshare_20564034.json'));expected={v['name']:v for v in article['files']}
    for rec in download['files']:
        p=RAW/rec['name'];assert sha(p)==rec['sha256'] and p.stat().st_size==rec['bytes']==expected[rec['name']]['size'];assert rec['md5']==expected[rec['name']]['computed_md5'];records.append(dict(path=str(p.relative_to(WS)),sha256=rec['sha256']))
    original=json.load(open(WS/'outputs/erastin_40gene_comparison/source_manifest.json'))
    for name in ['Model.csv','Gene.csv','OmicsExpressionProteinCodingGenesTPMLogp1.csv']:
        rec=next(v for v in original['files'] if v['local_name']==name);assert sha(ORIG/name)==rec['sha256'];records.append(dict(path=str((ORIG/name).relative_to(WS)),sha256=rec['sha256']))
    for root,names in [(WS/'outputs/erastin_40gene_comparison',['panel_mapping.tsv','compound_identity.json']), (WS/'outputs/erastin_panel_expansion',['candidate_gene_evidence.tsv'])]:
        known={v['path']:v['sha256'] for v in json.load(open(root/'artifact_manifest.json'))['files']}
        for name in names:assert sha(root/name)==known[name];records.append(dict(path=str((root/name).relative_to(WS)),sha256=known[name]))
    for name in ['weighted_mlp.py','metric_helpers.py']:assert sha(OUT/name)==sha(SRC/name)
    dump(OUT/'source_manifest.json',dict(files=records,prism_downloads=download['files'],prism_doi=article['doi'],rna_doi=original['depmap_doi']))
    for name in ['figshare_20564034.json','download_manifest.json','prism-repurposing-20q2-secondary-screen-readme']:
        shutil.copyfile(RAW/name,OUT/name)

def identity_groups(m):
    parent={k:k for k in m.index}
    def find(a):
        while parent[a]!=a:parent[a]=parent[parent[a]];a=parent[a]
        return a
    for field in ['PatientID','RRID','CCLEName','StrippedCellLineName']:
        seen={}
        for mid,value in m[field].items():
            if not value:continue
            if value in seen:
                aa,bb=find(mid),find(seen[value]);parent[max(aa,bb)]=min(aa,bb)
            else:seen[value]=mid
    return {mid:'G_'+find(mid) for mid in m.index}

def prepare():
    source_checks();old=np.load(SRC/'panel_inputs.npz');sp=np.load(SRC/'shared_splits.npz');models=pd.read_csv(ORIG/'Model.csv',dtype=str,keep_default_na=False).set_index('ModelID');assert models.index.is_unique
    groups=identity_groups(models);np.testing.assert_array_equal([groups[mid] for mid in old['sample_ids']],old['group_ids'])
    portal=json.load(open(WS/'outputs/erastin_40gene_comparison/compound_identity.json'));assert portal['CompoundID']=='DPC-002562' and 'BRD:'+BROAD in portal['SampleIDs'].split(';')
    treatment=pd.read_csv(RAW/'prism-repurposing-20q2-secondary-screen-replicate-collapsed-treatment-info.csv');treatment=treatment[treatment.broad_id.eq(BROAD)];assert set(treatment.name)=={'erastin'};table('erastin_treatment_metadata.tsv',treatment)
    pieces=[]
    for d in pd.read_csv(RAW/'prism-repurposing-20q2-secondary-screen-dose-response-curve-parameters.csv',chunksize=50000,low_memory=False):pieces.append(d[d.broad_id.eq(BROAD)|d.name.str.lower().eq('erastin')])
    er=pd.concat(pieces,ignore_index=True);assert set(er.broad_id)=={BROAD} and set(er.name)=={'erastin'};table('prism_erastin_all_rows.tsv',er)
    assert set(er.screen_id)<=set(treatment.screen_id)
    # Preserve exact prior feature identities and source precision.
    pm=pd.read_csv(WS/'outputs/erastin_40gene_comparison/panel_mapping.tsv',sep='\t',dtype=str).set_index('bulkformer_symbol');add=pd.read_csv(WS/'outputs/erastin_panel_expansion/candidate_gene_evidence.tsv',sep='\t',dtype=str).set_index('symbol');mapping=[]
    for j,g in enumerate(old['genes_64']):
        row=pm.loc[g] if g in pm.index else add.loc[g];mapping.append(dict(order=j,gene=str(g),entrez_id=row.entrez_id,ensembl_gene_id=row.ensembl_gene_id,depmap_column=row.depmap_column))
    mapping=table('gene_mapping.tsv',mapping);header=pd.read_csv(ORIG/'OmicsExpressionProteinCodingGenesTPMLogp1.csv',nrows=0).columns.tolist();idcol=header[0];cols=mapping.depmap_column.tolist();assert len(set(cols))==64 and set(cols)<=set(header)
    expression=pd.read_csv(ORIG/'OmicsExpressionProteinCodingGenesTPMLogp1.csv',usecols=[idcol]+cols).set_index(idcol)[cols].astype(np.float32);assert expression.index.is_unique
    np.testing.assert_array_equal(expression.loc[old['sample_ids']].to_numpy().astype(float),old['x_64']);gene_expression_checks=len(old['sample_ids'])*64
    audit=[]
    for i,row in er.iterrows():
        mid=row.depmap_id;reasons=[]
        if not bool(row.passed_str_profiling==True):reasons.append('STR_not_explicitly_passed')
        if not np.isfinite(row.auc):reasons.append('nonfinite_auc')
        if pd.isna(mid) or mid not in models.index:reasons.append('missing_release_ModelID')
        else:
            m=models.loc[mid]
            if m.ModelType!='Cell Line':reasons.append('not_cell_line')
            if m.OncotreePrimaryDisease=='Non-Cancerous':reasons.append('non_cancerous')
            if m.OncotreeLineage not in set(old['lineage']):reasons.append('lineage_not_in_training_benchmark')
        if mid not in expression.index:reasons.append('no_release_expression')
        elif not np.isfinite(expression.loc[mid].to_numpy()).all():reasons.append('nonfinite_panel_expression')
        audit.append(dict(source_row=i,ModelID=mid,row_name=row.row_name,screen_id=row.screen_id,included=not reasons,exclusion_reasons=';'.join(reasons)))
    audit=table('cohort_audit.tsv',audit);usable=er.loc[audit.included.to_numpy()].copy();cohort=[]
    for mid,rows in usable.groupby('depmap_id',sort=True):
        preferred='MTS010' if 'MTS010' in set(rows.screen_id) else 'HTS002';chosen=rows[rows.screen_id.eq(preferred)];m=models.loc[mid];group=groups[mid]
        cohort.append(dict(ModelID=mid,CellLineName=m.CellLineName,CCLEName=m.CCLEName,lineage=m.OncotreeLineage,group_id=group,prism_screen=preferred,prism_auc=float(chosen.auc.median()),prism_sensitivity=-float(chosen.auc.median()),prism_r2_median=float(chosen.r2.median()),passing_rows_aggregated=len(chosen),exact_training_cohort_id=mid in set(old['sample_ids']),novel_identity_group=group not in set(old['group_ids']),original_benchmark_auc=float(-old['y'][np.flatnonzero(old['sample_ids']==mid)[0]]) if mid in old['sample_ids'] else np.nan,ccle_name_agrees_with_prism=all(chosen.ccle_name==m.CCLEName)))
    cohort=pd.DataFrame(cohort);ids=cohort.ModelID.to_numpy(dtype=str);x=expression.loc[ids].to_numpy(dtype=float);assert np.isfinite(x).all();eligible=np.empty((20,len(ids)),bool)
    for r in range(20):
        training=set(old['group_ids'][~sp['outer_test'][r]]);eligible[r]=~cohort.group_id.isin(training).to_numpy()
        assert set(cohort.group_id[eligible[r]]).isdisjoint(training)
        for j,mid in enumerate(ids):
            if mid in old['sample_ids']:assert eligible[r,j]==sp['outer_test'][r,np.flatnonzero(old['sample_ids']==mid)[0]]
    novel=cohort.novel_identity_group.to_numpy();assert eligible[:,novel].all();cohort['eligible_outer_fits']=eligible.sum(axis=0);cohort['scored']=eligible.any(axis=0);table('matched_cohort.tsv',cohort)
    lookup={str(g):i for i,g in enumerate(old['genes_64'])};five=np.array([lookup[str(g)] for g in old['genes_5']]);np.testing.assert_array_equal(five,old['five_gene_indices_in_64'])
    np.savez_compressed(OUT/'matched_prism_inputs.npz',sample_ids=ids,group_ids=cohort.group_id.to_numpy(dtype=str),lineage=cohort.lineage.to_numpy(dtype=str),x_64=x,x_5=x[:,five],genes_64=old['genes_64'],genes_5=old['genes_5'],y=cohort.prism_sensitivity.to_numpy(),prism_auc=cohort.prism_auc.to_numpy(),eligible_fits=eligible,novel_identity_group=novel,original_benchmark_sensitivity=-cohort.original_benchmark_auc.to_numpy())
    info=dict(raw_erastin_rows=len(er),raw_distinct_nonmissing_ModelIDs=er.depmap_id.nunique(),STR_passed_rows=int(er.passed_str_profiling.eq(True).sum()),passing_matched_rows=int(audit.included.sum()),matched_lines=len(cohort),matched_identity_groups=cohort.group_id.nunique(),lineages=cohort.lineage.nunique(),novel_lines=int(novel.sum()),novel_groups=cohort.loc[novel,'group_id'].nunique(),novel_lineages=cohort.loc[novel,'lineage'].nunique(),shared_lines=int((~novel).sum()),shared_scored_lines=int((~novel&eligible.any(axis=0)).sum()),never_heldout_shared_lines=int((~novel&~eligible.any(axis=0)).sum()),gene_expression_values_exactly_verified=gene_expression_checks,original_training_identity_groups_verified=True,auc_min=cohort.prism_auc.min(),auc_max=cohort.prism_auc.max(),auc_greater_than_one=int((cohort.prism_auc>1).sum()),duplicate_passing_rows_collapsed=int(cohort.passing_rows_aggregated.sum()-len(cohort)),historical_CTRP_response_source='Archived redistribution attributed to Harmonized CTD2 25Q2; upstream AUC generation not independently established.',PRISM_source='Official Figshare PRISM Repurposing 20Q2 v1, secondary screen HTS002 for erastin; direct checksum verification.')
    dump(OUT/'cohort_summary.json',info);print(json.dumps(clean(info)),flush=True);return cohort,old,sp

def predict(cohort,old,sp):
    inp=np.load(OUT/'matched_prism_inputs.npz');eligible=inp['eligible_fits'];P={a:np.full(eligible.shape,np.nan) for a in ARMS};checks=[]
    for r in range(20):
        ix=np.flatnonzero(eligible[r]);ids=inp['sample_ids'][ix]
        for f in FAMILIES:
            for panel in PANELS:
                arm=f+'__'+str(panel);root=SRC/'folds'/f'{f}__p{panel}__r{r+1:02d}';m=json.load(open(root.with_suffix('.json')));fit=joblib.load(root.with_suffix('.joblib'));a=np.load(root.with_suffix('.npz'));x=inp['x_'+str(panel)][ix]
                np.testing.assert_array_equal(a['train_rows'],np.flatnonzero(~sp['outer_test'][r]));np.testing.assert_array_equal(a['test_rows'],np.flatnonzero(sp['outer_test'][r]));assert len(a['train_rows'])==595 and m['protocol_sha256']==sha(SRC/'protocol.json')
                assert set(inp['group_ids'][ix]).isdisjoint(old['group_ids'][a['train_rows']])
                if f=='ridge':
                    pred=(x-fit['mean'])/fit['scale']@fit['coef']+fit['intercept_after_scaling'];manual=pred.copy()
                else:
                    z=fit['x_scaler'].transform(x);sy=fit['y_scaler']
                    if f=='elastic_net':pred=sy.inverse_transform(fit['model'].predict(z)[:,None]).ravel();manual=(z@fit['model'].coef_+fit['model'].intercept_)*sy.scale_[0]+sy.mean_[0]
                    else:
                        pred=np.mean([sy.inverse_transform(v.predict(z)[:,None]).ravel() for v in fit['members']],axis=0);manual=np.mean([((np.tanh(z@v.coefs_[0]+v.intercepts_[0])@v.coefs_[1]+v.intercepts_[1]).ravel()*sy.scale_[0]+sy.mean_[0]) for v in fit['members']],axis=0)
                error=float(np.max(np.abs(pred-manual)));assert error<1e-10;knownerror=0.;knowncount=0
                for k,mid in enumerate(ids):
                    if mid in old['sample_ids']:
                        position=int(np.flatnonzero(old['sample_ids']==mid)[0]);j=int(np.flatnonzero(a['test_rows']==position)[0]);knownerror=max(knownerror,float(abs(pred[k]-a['test_prediction'][j])));knowncount+=1
                assert knownerror<1e-10 and np.isfinite(pred).all();P[arm][r,ix]=pred;checks.append(dict(model=arm,repeat=r+1,predicted_lines=len(ix),known_original_test_predictions_checked=knowncount,max_saved_prediction_error=knownerror,max_native_manual_error=error))
        if (r+1)%5==0:print('PREDICTED_REPEATS',r+1,flush=True)
    count=eligible.sum(axis=0);ensembles={a:np.divide(np.nansum(p,axis=0),count,out=np.full(len(cohort),np.nan),where=count>0) for a,p in P.items()};rows=[]
    for j,row in cohort.iterrows():
        for arm in ARMS:rows.append(dict(ModelID=row.ModelID,group_id=row.group_id,lineage=row.lineage,novel_identity_group=row.novel_identity_group,eligible_fits=int(count[j]),model=arm,predicted_sensitivity=ensembles[arm][j],observed_prism_sensitivity=row.prism_sensitivity))
    table('ensemble_predictions.tsv',rows);np.savez_compressed(OUT/'predictions.npz',sample_ids=inp['sample_ids'],eligible_fits=eligible,**P,**{a+'__ensemble':v for a,v in ensembles.items()});dump(OUT/'prediction_validation.json',dict(passed=True,no_refitting=True,models_checked=len(checks),checks=checks,all_predictions_identity_group_held_out=True))
    return inp,P,ensembles

def metrics(y,p,lin=None):
    n=len(y);error=y-p;den=np.sum((y-y.mean())**2);result=dict(n=n,spearman=float(spearmanr(y,p).statistic) if n>=3 and np.ptp(y)>0 and np.ptp(p)>0 else np.nan,pearson=float(pearsonr(y,p).statistic) if n>=3 and np.ptp(y)>0 and np.ptp(p)>0 else np.nan,uncalibrated_auc_rmse=float(np.sqrt(np.mean(error**2))),uncalibrated_auc_mae=float(np.mean(np.abs(error))),uncalibrated_auc_r2=float(1-np.sum(error**2)/den) if den>0 and n>=2 else np.nan)
    if lin is not None:
        yy=y.copy();pp=p.copy()
        for l in np.unique(lin):mask=lin==l;yy[mask]-=yy[mask].mean();pp[mask]-=pp[mask].mean()
        result['within_lineage_centered_pearson']=float(pearsonr(yy,pp).statistic) if n>=3 and np.ptp(yy)>0 and np.ptp(pp)>0 else np.nan
    return result

def boot(y,pp,clusters,count,seed):
    labels,inv=np.unique(clusters,return_inverse=True);K=len(labels);rng=np.random.default_rng(seed);yrank=weighted_ranker(y);prank={a:weighted_ranker(p) for a,p in pp.items()};result={a:{key:np.empty(count) for key in BOOT_KEYS} for a in pp};maxcheck=0.;checks=0
    for offset in range(0,count,100):
        b=min(100,count-offset);draws=rng.integers(K,size=(b,K));freq=np.array([np.bincount(d,minlength=K) for d in draws]);w=freq[:,inv];tot=w.sum(axis=1);ym=(w*y).sum(axis=1)/tot;yc=y[None,:]-ym[:,None];yss=(w*yc**2).sum(axis=1);yr=yrank(w);yrc=yr-(tot[:,None]+1)/2;yrss=(w*yrc**2).sum(axis=1)
        for a,p in pp.items():
            pc=p[None,:]-(w*p).sum(axis=1)[:,None]/tot[:,None];pss=(w*pc**2).sum(axis=1);pr=prank[a](w);prc=pr-(tot[:,None]+1)/2;prss=(w*prc**2).sum(axis=1);sse=w@((y-p)**2)
            with np.errstate(divide='ignore',invalid='ignore'):
                values=dict(spearman=(w*yrc*prc).sum(axis=1)/np.sqrt(yrss*prss),pearson=(w*yc*pc).sum(axis=1)/np.sqrt(yss*pss),uncalibrated_auc_rmse=np.sqrt(sse/tot),uncalibrated_auc_mae=w@np.abs(y-p)/tot,uncalibrated_auc_r2=1-sse/yss)
            for key,v in values.items():result[a][key][offset:offset+b]=v
            if offset==0:
                for i in range(min(5,b)):
                    rows=np.concatenate([np.flatnonzero(inv==j) for j in draws[i]]);ref=metrics(y[rows],p[rows])
                    for key in BOOT_KEYS:
                        if np.isfinite(ref[key]) and np.isfinite(values[key][i]):maxcheck=max(maxcheck,float(abs(ref[key]-values[key][i])));checks+=1;assert abs(ref[key]-values[key][i])<1e-10
    return result,dict(expanded_scipy_metric_checks=checks,max_error=maxcheck,clusters=K)

def evaluate(inp,P,ensembles,cohort):
    plan=json.load(open(OUT/'protocol.json'));y=inp['y'];lin=inp['lineage'];group=inp['group_ids'];novel=inp['novel_identity_group'];eligible=inp['eligible_fits'];ever=eligible.any(axis=0);scopes={'novel_identity_groups':novel,'shared_groups_held_out':~novel&ever,'combined_held_out':ever};summary=[];repeat=[];lineage=[];ci=[];paired=[];bootstrap_arrays={};validations=[]
    for name,mask in scopes.items():
        ix=np.flatnonzero(mask)
        if len(ix)<3:continue
        yp=y[ix];pp={a:p[ix] for a,p in ensembles.items()};points={a:metrics(yp,p,lin[ix]) for a,p in pp.items()}
        for a,v in points.items():summary.append(dict(scope=name,model=a,groups=len(np.unique(group[ix])),lineages=len(np.unique(lin[ix])),**v))
        for unit,cl,seed in [('identity_group',group[ix],plan['uncertainty']['seed']),('lineage',lin[ix],plan['uncertainty']['seed']+1)]:
            draws,val=boot(yp,pp,cl,plan['uncertainty']['count'],seed);validations.append(dict(scope=name,unit=unit,**val))
            for a in ARMS:
                for key,vals in draws[a].items():
                    finite=vals[np.isfinite(vals)];lo,hi=np.percentile(finite,[2.5,97.5]) if len(finite) else (np.nan,np.nan);ci.append(dict(scope=name,model=a,metric=key,unit=unit,point=points[a][key],conditional_ci_low=lo,conditional_ci_high=hi,valid_bootstrap_draws=len(finite),bootstrap_draws=len(vals)));bootstrap_arrays[name+'__'+unit+'__'+a+'__'+key]=vals
            for first,ref in plan['comparisons']:
                for key in ['spearman','pearson']:
                    diff=draws[first][key]-draws[ref][key];lo,hi=np.nanpercentile(diff,[2.5,97.5]);paired.append(dict(scope=name,first_model=first,reference_model=ref,metric=key,unit=unit,improvement=points[first][key]-points[ref][key],conditional_ci_low=lo,conditional_ci_high=hi))
        for a,p in pp.items():
            for label in np.unique(lin[ix]):
                mm=lin[ix]==label;v=metrics(yp[mm],p[mm]);
                if mm.sum()<5:v['spearman']=np.nan;v['pearson']=np.nan
                lineage.append(dict(scope=name,model=a,lineage=str(label),**v))
        print('SCORED_SCOPE',name,len(ix),flush=True)
    for r in range(20):
        for name,mask in [('all_identity_held_out',eligible[r]),('novel_identity_groups',novel),('shared_groups_held_out',~novel&eligible[r])]:
            ix=np.flatnonzero(mask)
            if len(ix)<3:continue
            for a in ARMS:repeat.append(dict(scope=name,model=a,repeat=r+1,**metrics(y[ix],P[a][r,ix],lin[ix])))
    summary=table('summary.tsv',summary);ci=table('confidence_intervals.tsv',ci);paired=table('paired_comparisons.tsv',paired);table('per_repeat_metrics.tsv',repeat);table('per_lineage_metrics.tsv',lineage);np.savez_compressed(OUT/'bootstrap_draws.npz',**bootstrap_arrays);diagnostics=[]
    ctrp=inp['original_benchmark_sensitivity'];matched=np.isfinite(ctrp)
    for name,mask in [('all_exact_CTRP_PRISM_overlap',matched),('scored_shared_overlap',matched&ever)]:
        ix=np.flatnonzero(mask);diagnostics.append(dict(scope=name,comparison='measured_CTRP_vs_measured_PRISM',**metrics(y[ix],ctrp[ix],lin[ix])))
    table('cross_assay_measurement_agreement.tsv',diagnostics);table('lineage_cohort_counts.tsv',cohort.groupby('lineage',sort=True).agg(matched_lines=('ModelID','size'),novel_lines=('novel_identity_group','sum'),scored_lines=('scored','sum')).reset_index());val=json.load(open(OUT/'prediction_validation.json'));val.update(source_hashes_verified=True,gene_expression_source_exactly_matched=True,bootstrap_checks=validations,raw_PRISM_AUC_preserved=True,no_PRISM_response_used_for_training_or_calibration=True);dump(OUT/'validation.json',val)
    from report_validation import render
    render(summary,ci,paired,cohort,ensembles,diagnostics,val)
    dump(OUT/'final_summary.json',dict(validation_passed=True,cohort=json.load(open(OUT/'cohort_summary.json')),summary=summary.to_dict('records'),measurement_agreement=diagnostics))

def seal():
    manifest=dict(files=[dict(path=str(p.relative_to(OUT)),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='artifact_manifest.json' and '__pycache__' not in p.parts]);dump(OUT/'artifact_manifest.json',manifest);archive=OUT.parent/'erastin_prism_validation_results.zip';prefix=OUT.name+'/'
    with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for row in manifest['files']:z.write(OUT/row['path'],prefix+row['path'])
        z.write(OUT/'artifact_manifest.json',prefix+'artifact_manifest.json')
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        for row in manifest['files']:assert hashlib.sha256(z.read(prefix+row['path'])).hexdigest()==row['sha256']
    print(json.dumps(dict(archive_bytes=archive.stat().st_size,files=len(manifest['files'])+1)),flush=True)

def main():
    ph=sha(OUT/'protocol.json');scripts={name:sha(OUT/name) for name in ['run_validation.py','report_validation.py','weighted_mlp.py','metric_helpers.py']};dump(OUT/'analysis_source_manifest.json',dict(protocol_sha256=ph,scripts=scripts));cohort,old,sp=prepare();inp,P,ensembles=predict(cohort,old,sp);evaluate(inp,P,ensembles,cohort)
    assert sha(OUT/'protocol.json')==ph
    for name,h in scripts.items():assert sha(OUT/name)==h
    for rec in json.load(open(OUT/'source_manifest.json'))['files']:assert sha(WS/rec['path'])==rec['sha256']
    dump(OUT/'progress.json',dict(phase='complete',validation_passed=True,no_refitting=True));seal()
if __name__=='__main__':main()
