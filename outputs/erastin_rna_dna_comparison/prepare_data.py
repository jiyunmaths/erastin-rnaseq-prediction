#!/usr/bin/env python3
"""Response-independent matching and pinned-release RNA/DNA input preparation."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from make_splits import inner_folds,inner_rows

OUT=Path(__file__).resolve().parent;WS=OUT.parent.parent
RAW=WS/'work/erastin_rna_dna/raw';OLDRAW=WS/'work/erastin/raw'
BASE={'CTRP':WS/'outputs/erastin_elastic_net_comparison','PRISM':WS/'outputs/erastin_prism_direct_comparison'}
MUTATIONS=[('KEAP1_likely_lof','damaging','KEAP1'),('NFE2L2_hotspot','hotspot','NFE2L2'),
    ('BAP1_likely_lof','damaging','BAP1'),('TP53_likely_lof','damaging','TP53'),
    ('TP53_hotspot','hotspot','TP53'),('KRAS_hotspot','hotspot','KRAS')]
FILES={'damaging':'OmicsSomaticMutationsMatrixDamaging.csv','hotspot':'OmicsSomaticMutationsMatrixHotspot.csv','cnv':'OmicsCNGene.csv'}


def sha(p):
    with Path(p).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple,np.ndarray)):return [clean(x) for x in v]
    if isinstance(v,np.generic):return clean(v.item())
    if isinstance(v,float) and not np.isfinite(v):return None
    return v


def dump(path,obj):path.write_text(json.dumps(clean(obj),indent=2,allow_nan=False)+'\n')


def read_selected(name,genes,rna_columns):
    with (RAW/name).open() as f:header=next(csv.reader(f))
    mapping={g:[c for c in header if c.split(' (')[0]==g] for g in genes}
    assert all(len(v)<=1 for v in mapping.values()),mapping
    missing=[g for g,v in mapping.items() if not v]
    mapping={g:v[0] for g,v in mapping.items() if v}
    for g,c in mapping.items():assert c==rna_columns[g],(name,g,c,rna_columns[g])
    selected=[0]+[header.index(c) for c in mapping.values()]
    frame=pd.read_csv(RAW/name,usecols=selected,index_col=0).rename(columns={v:k for k,v in mapping.items()})
    assert frame.index.is_unique and all(str(v).startswith('ACH-') for v in frame.index)
    frame=frame.reindex(columns=list(mapping))
    return frame,mapping,missing


def main():
    registry_path=WS/'outputs/retained_64_gene_models/model_registry.json';reg=json.load(open(registry_path))
    genes=reg['genes_in_input_order'];cn_candidates=genes+['KEAP1','TP53','KRAS']
    download=json.load(open(RAW/'download_manifest.json'));assert download['official_metadata_verified_live']
    records=[dict(path=str(registry_path.relative_to(WS)),sha256=sha(registry_path))]
    for file in download['files']:
        p=RAW/file['name'];assert p.stat().st_size==file['bytes'] and sha(p)==file['sha256']
        records.append(dict(path=str(p.relative_to(WS)),sha256=file['sha256']))
    for name in ['download_manifest.json','release_selected_files.json']:
        records.append(dict(path=str((RAW/name).relative_to(WS)),sha256=sha(RAW/name)))
    with (OLDRAW/'OmicsExpressionProteinCodingGenesTPMLogp1.csv').open() as f:header=next(csv.reader(f))
    rna_columns={g:[c for c in header if c.split(' (')[0]==g] for g in cn_candidates}
    assert all(len(v)==1 for v in rna_columns.values());rna_columns={g:v[0] for g,v in rna_columns.items()}
    damaging,dm,dmiss=read_selected(FILES['damaging'],['KEAP1','BAP1','TP53'],rna_columns)
    hotspot,hm,hmiss=read_selected(FILES['hotspot'],['NFE2L2','TP53','KRAS'],rna_columns)
    assert not dmiss+hmiss
    cnv,cm,cmiss=read_selected(FILES['cnv'],cn_candidates,rna_columns)
    assert np.nanmin(cnv.to_numpy())>=0
    # A header exclusion is a release-level feature-universe fact, not a test-driven filter.
    cn_genes=list(cm)
    mapping=[]
    for name,kind,g in MUTATIONS:mapping.append(dict(block='mutation',feature=name,gene=g,source_file=FILES[kind],source_column=(dm if kind=='damaging' else hm)[g],encoding='1 if finite official genotype > 0; 0 if finite genotype == 0; missing otherwise'))
    for g in cn_candidates:mapping.append(dict(block='cnv',feature='CN_'+g,gene=g,source_file=FILES['cnv'],source_column=cm.get(g,''),encoding='log2(relative copy ratio + 1)' if g in cm else 'not present in release matrix; no substituted gene'))
    pd.DataFrame(mapping).to_csv(OUT/'gene_mapping.tsv',sep='\t',index=False)
    coverage=[];exclusions=[];summary={};repairs=[]
    for ds,root in BASE.items():
        (OUT/ds).mkdir(exist_ok=True)
        co=reg['cohorts'][ds]
        for k in ['panel_inputs.npz','shared_splits.npz','protocol.json','validation.json']:
            r=co['input_and_validation_files'][k];assert sha(WS/r['path'])==r['sha256'];records.append(r)
        for family in co['models'].values():
            for entry in family['outer_models']:
                for r in entry['files'].values():assert sha(WS/r['path'])==r['sha256'];records.append(r)
        a=np.load(root/'panel_inputs.npz');s=np.load(root/'shared_splits.npz')
        ids,groups,lin=a['sample_ids'],a['group_ids'],a['lineage'];np.testing.assert_array_equal(a['genes_64'],genes)
        for key in ['sample_ids','group_ids','lineage']:np.testing.assert_array_equal(a[key],s[key])
        raw_mut=np.column_stack([(damaging if kind=='damaging' else hotspot).reindex(ids)[g].to_numpy() for _,kind,g in MUTATIONS])
        assert set(np.unique(raw_mut[np.isfinite(raw_mut)]))<=set([0.,1.,2.])
        mut=np.where(np.isfinite(raw_mut),(raw_mut>0).astype(float),np.nan)
        cc=cnv.reindex(ids).to_numpy();cn_fraction=np.isfinite(cc).mean(axis=1)
        mut_ok=np.isfinite(mut).all(axis=1);cn_ok=np.isin(ids,cnv.index)&(cn_fraction>=.8)
        keep=mut_ok&cn_ok
        invalid=[]
        for label in np.unique(lin):
            if not all(np.any(keep&(lin==label)&mask) and np.any(keep&(lin==label)&~mask) for mask in s['outer_test']):
                invalid.append(str(label))
        keep&=~np.isin(lin,invalid)
        ix=np.flatnonzero(keep);inner=s['inner_fold'][:,ix].copy();test=s['outer_test'][:,ix]
        for r in range(20):
            tr=np.flatnonzero(~test[r]);te=np.flatnonzero(test[r])
            assert set(groups[ix[tr]]).isdisjoint(groups[ix[te]])
            assert set(lin[ix[tr]])==set(lin[ix[te]])==set(lin[ix])
            affected=[]
            for label in np.unique(lin[ix]):
                if any(not np.any((lin[ix]==label)&np.isin(np.arange(len(ix)),inner_rows(inner[r],f)[0])) for f in range(5)):
                    affected.append(label)
            if affected:
                rebuilt=inner_folds(lin[ix],groups[ix],test[r],int(s['inner_split_seeds'][r]),k=5)
                flag=np.isin(lin[ix],affected)&~test[r];inner[r,flag]=rebuilt[flag]
            repairs.append(dict(assay=ds,repeat=r+1,lineages_repaired=[str(v) for v in affected],changed_assignments=int(np.sum(inner[r]!=s['inner_fold'][r,ix]))))
            for f in range(5):
                tr0,va=inner_rows(inner[r],f)
                assert set(tr0)|set(va)==set(tr) and set(groups[ix[tr0]]).isdisjoint(groups[ix[va]])
                assert set(lin[ix[tr0]])==set(lin[ix])
        np.savez_compressed(OUT/ds/'inputs.npz',rna=a['x_64'][ix],mutation=mut[ix],cnv=np.log2(cc[ix]+1),
            y=a['y'][ix],sample_ids=ids[ix],group_ids=groups[ix],lineage=lin[ix],genes=np.array(genes),
            mutation_features=np.array([v[0] for v in MUTATIONS]),cnv_features=np.array(['CN_'+g for g in cn_genes]),original_rows=ix)
        np.savez_compressed(OUT/ds/'shared_splits.npz',sample_ids=ids[ix],group_ids=groups[ix],lineage=lin[ix],outer_test=test,inner_fold=inner,
            outer_split_seeds=s['outer_split_seeds'],inner_split_seeds=s['inner_split_seeds'],neural_initialization_seeds=s['neural_initialization_seeds'])
        for i,mid in enumerate(ids):
            why=[]
            if not mut_ok[i]:why.append('missing genotyped mutation information')
            if not cn_ok[i]:why.append('copy-number coverage below fixed 80% threshold or no profile')
            if str(lin[i]) in invalid:why.append('lineage cannot occupy both sides of every original outer split after matching')
            row=dict(assay=ds,ModelID=str(mid),lineage=str(lin[i]),mutation_complete=bool(mut_ok[i]),cnv_finite_fraction=float(cn_fraction[i]),included=bool(keep[i]),reason='; '.join(why))
            coverage.append(row)
            if not keep[i]:exclusions.append(row)
        summary[ds]=dict(original_samples=len(ids),matched_samples=len(ix),original_lineages=len(np.unique(lin)),matched_lineages=len(np.unique(lin[ix])),
            excluded_samples=int((~keep).sum()),excluded_lineages=invalid,
            mutation_positive_counts={name:int(mut[ix,j].sum()) for j,(name,_,_) in enumerate(MUTATIONS)},
            outer_train_sizes=(~test).sum(axis=1),outer_test_sizes=test.sum(axis=1),
            identical_original_cohort=bool(keep.all()),inner_changed_assignments=int(np.sum(inner!=s['inner_fold'][:,ix])))
        print(json.dumps(clean(dict(assay=ds,**summary[ds]))),flush=True)
    pd.DataFrame(coverage).to_csv(OUT/'sample_coverage.tsv',sep='\t',index=False)
    pd.DataFrame(exclusions).to_csv(OUT/'excluded_samples.tsv',sep='\t',index=False)
    dump(OUT/'coverage_summary.json',dict(assays=summary,cnv_candidate_genes=cn_candidates,cnv_available_genes=cn_genes,cnv_absent_from_header=cmiss,mutations=MUTATIONS,inner_repairs=repairs))
    records.extend(dict(path=str((OLDRAW/n).relative_to(WS)),sha256=sha(OLDRAW/n)) for n in ['README.txt','OmicsDefaultModelProfiles.csv','OmicsProfiles.csv'])
    dump(OUT/'source_manifest.json',dict(release='DepMap 24Q4 Public v1',files=records))
    dump(OUT/'preparation_validation.json',dict(passed=True,response_independent_cohort_matching=True,all_outer_identity_and_lineage_splits_checked=True,
        all_inner_identity_and_lineage_splits_checked=True,original_outer_masks_projected_exactly=True,genotype_missing_never_recoded_as_wild_type=True,
        exact_gene_column_identity_verified_against_RNA=True,checksum_verification_passed=True,assays=summary,cnv_absent_from_header=cmiss))
    print('RNA_DNA_PREPARATION_COMPLETE',flush=True)


if __name__=='__main__':main()
