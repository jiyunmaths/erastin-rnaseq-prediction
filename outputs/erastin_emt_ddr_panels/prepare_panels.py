#!/usr/bin/env python3
"""Freeze Hallmark EMT/DNA-repair panels without using response associations."""
import hashlib,json,shutil,time,warnings
from pathlib import Path
from collections import defaultdict
import numpy as np,pandas as pd
with warnings.catch_warnings():
    warnings.simplefilter('ignore');from bs4 import BeautifulSoup
OUT=Path(__file__).resolve().parent;WS=OUT.parent.parent;RAW=WS/'work/erastin_emt_ddr/raw';BASE=WS/'outputs/erastin_elastic_net_comparison';ORIGINAL=WS/'outputs/erastin_40gene_comparison'
SETS={'emt':'HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION','ddr':'HALLMARK_DNA_REPAIR'}
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def dump(p,v):p.write_text(json.dumps(v,indent=2,allow_nan=False)+'\n')
def table(name,v):pd.DataFrame(v).to_csv(OUT/name,sep='\t',index=False)
def main():
    records=[];known={r['path']:r['sha256'] for r in json.load(open(BASE/'artifact_manifest.json'))['files']};original_known={r['path']:r['sha256'] for r in json.load(open(ORIGINAL/'artifact_manifest.json'))['files']};downloads=json.load(open(RAW/'downloads.json'));(OUT/'sources').mkdir(exist_ok=True)
    for source,names,manifest in [(BASE,['panel_inputs.npz','shared_splits.npz','protocol.json'],known),(ORIGINAL,['matched_depmap_expression.npz'],original_known)]:
        for name in names:assert sha(source/name)==manifest[name];records.append(dict(path=str((source/name).relative_to(WS)),sha256=manifest[name]))
    release=WS/'work/erastin/raw/Gene.csv';raw_manifest=json.load(open(ORIGINAL/'source_manifest.json'));assert sha(release)==next(r['sha256'] for r in raw_manifest['files'] if r['local_name']=='Gene.csv');records.append(dict(path=str(release.relative_to(WS)),sha256=sha(release)))
    rel=pd.read_csv(release,dtype=str,keep_default_na=False);hgnc=defaultdict(list)
    for _,r in rel.iterrows():
        if r.entrez_id:hgnc[r.entrez_id.removesuffix('.0')].append(r)
    full=np.load(ORIGINAL/'matched_depmap_expression.npz');old=np.load(BASE/'panel_inputs.npz');columns=full['gene_columns'];by_id=defaultdict(list);by_symbol={}
    for j,col in enumerate(columns):
        symbol,ent=col.rsplit(' (',1);ent=ent[:-1];by_id[ent].append(j);by_symbol[symbol]=j
    sample_lookup={v:j for j,v in enumerate(full['sample_ids'])};rows=np.array([sample_lookup[v] for v in old['sample_ids']]);x=full['X_log2tpm1'][rows].astype(float);base_cols=[by_symbol[g] for g in old['genes_64']];np.testing.assert_array_equal(x[:,base_cols],old['x_64']);base_ids=[str(columns[j]).rsplit(' (',1)[1][:-1] for j in base_cols]
    membership=[];set_ids={};set_rows={};counts=[]
    for key,name in SETS.items():
        for suffix in ['json','html']:
            src=RAW/(name+'.'+suffix);d=next(r for r in downloads if r['path']==src.name);assert sha(src)==d['sha256'];shutil.copyfile(src,OUT/'sources'/src.name);records.append(dict(path=str(src.relative_to(WS)),sha256=d['sha256'],url=d['url'],retrieved_utc=d['retrieved_utc']))
        document=json.load(open(RAW/(name+'.json')))[name];symbols=document['geneSymbols'];assert len(symbols)=={'emt':200,'ddr':150}[key];soup=BeautifulSoup((RAW/(name+'.html')).read_bytes().decode('iso-8859-1'),'html.parser');maps=[]
        for tr in soup.find_all('tr'):
            cells=tr.find_all('td',recursive=False)
            if len(cells)!=4:continue
            link=cells[1].find('a',href=lambda h:bool(h and 'ncbi.nlm.nih.gov/gene/' in h))
            if link is None:continue
            source_symbol=cells[0].get_text(strip=True);entrez=link.get_text(strip=True);current=cells[2].get_text(strip=True);assert entrez.isdigit()
            assert current in symbols;indices=list(by_id.get(entrez,[]));candidate_columns=[str(columns[j]) for j in indices];rule='unique_entrez'
            if len(indices)>1:
                release_symbols={r.symbol for r in hgnc.get(entrez,[])};exact=[j for j in indices if str(columns[j]).rsplit(' (',1)[0]==current and current in release_symbols];assert len(exact)==1;indices=exact;by_id[entrez]=indices;rule='duplicate_entrez_resolved_by_current_symbol_and_release_HGNC'
            assert len(indices)<=1
            idx=int(indices[0]) if indices else -1;dep_symbol=str(columns[idx]).rsplit(' (',1)[0] if idx>=0 else '';hg=hgnc.get(entrez,[]);hg_ids=sorted(set(r.hgnc_id for r in hg));hg_ens=sorted(set(r.ensembl_gene_id for r in hg if r.ensembl_gene_id));item=dict(gene_set=name,module=key,mapping_rule=rule,candidate_depmap_columns='|'.join(candidate_columns),source_symbol=source_symbol,current_symbol=current,entrez_id=entrez,release_hgnc_id='|'.join(hg_ids),release_ensembl_ids='|'.join(hg_ens),depmap_symbol=dep_symbol,depmap_column=str(columns[idx]) if idx>=0 else '',expression_column_0based=idx,available=idx>=0,in_ferroptosis64=entrez in base_ids,symbol_changed_since_depmap=idx>=0 and current!=dep_symbol)
            maps.append(item)
        assert len(maps)==len(symbols) and {r['current_symbol'] for r in maps}==set(symbols);assert len(set(r['entrez_id'] for r in maps))==len(maps);membership.extend(maps);set_ids[key]=[r['entrez_id'] for r in maps if r['available']];set_rows[key]=maps;counts.append(dict(module=key,source_genes=len(maps),mapped_expression_genes=sum(r['available'] for r in maps),already_in_64=sum(r['in_ferroptosis64'] and r['available'] for r in maps),missing_genes=[r['current_symbol'] for r in maps if not r['available']],symbol_updates=[dict(current=r['current_symbol'],depmap=r['depmap_symbol'],entrez_id=r['entrez_id']) for r in maps if r['symbol_changed_since_depmap']]))
    definitions={};arrays={k:old[k] for k in ['y','sample_ids','group_ids','lineage']};panel_modules={'ferro5':[],'ferro64':[],'ferro64_emt':['emt'],'ferro64_ddr':['ddr'],'ferro64_emt_ddr':['emt','ddr']}
    for panel,modules in panel_modules.items():
        if panel=='ferro5':panel_columns=[by_symbol[g] for g in old['genes_5']];ids=[str(columns[j]).rsplit(' (',1)[1][:-1] for j in panel_columns]
        else:
            ids=list(base_ids)
            for module in modules:
                for ent in set_ids[module]:
                    if ent not in ids:ids.append(ent)
            panel_columns=[by_id[ent][0] for ent in ids]
        values=x[:,panel_columns];assert np.isfinite(values).all();gene_names=np.array([str(columns[j]).rsplit(' (',1)[0] for j in panel_columns]);arrays['x_'+panel]=values;arrays['genes_'+panel]=gene_names;arrays['entrez_'+panel]=np.array(ids);definitions[panel]=dict(gene_count=len(ids),entrez_ids=ids,depmap_symbols=gene_names.tolist(),added_modules=modules)
    np.testing.assert_array_equal(arrays['x_ferro5'],old['x_5']);np.testing.assert_array_equal(arrays['x_ferro64'],old['x_64']);np.savez_compressed(OUT/'panel_inputs.npz',**arrays);shutil.copyfile(BASE/'shared_splits.npz',OUT/'shared_splits.npz');assert sha(OUT/'shared_splits.npz')==known['shared_splits.npz'];table('gene_mapping.tsv',membership)
    coverage=dict(modules=counts,panels={k:v['gene_count'] for k,v in definitions.items()},samples=len(rows),identity_groups=len(set(old['group_ids'])),lineages=len(set(old['lineage'])),all_values_finite=True,base_5_and_64_arrays_identical=True,response_associations_used_for_gene_selection=False,culture_predictors=False,requested_scope='Full Hallmark EMT and DNA repair sets, as explicitly selected by user. No BRCA/EMT-regulator extras added.',limitations='Hallmark sets are expression programs, not comprehensive catalogues. For example BRCA1/BRCA2 and ZEB1/ZEB2 are absent from the chosen Hallmark DNA-repair/EMT sets. Their absence is documented rather than silently adding extra genes.')
    dump(OUT/'coverage_summary.json',coverage);dump(OUT/'panel_definitions.json',definitions);dump(OUT/'source_manifest.json',dict(files=records));dump(OUT/'preparation_validation.json',dict(passed=True,panel_input_sha256=sha(OUT/'panel_inputs.npz'),shared_split_sha256=sha(OUT/'shared_splits.npz'),raw_array_row_order_verified=True,entrez_mapping_one_to_one=True,source_gene_counts_verified={'emt':200,'ddr':150},source_response_arrays_unchanged=True,created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())))
    print(json.dumps(coverage,indent=2))
if __name__=='__main__':main()
