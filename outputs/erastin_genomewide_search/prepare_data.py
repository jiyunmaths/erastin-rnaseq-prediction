#!/usr/bin/env python3
"""Response-independent, unique-Entrez protein-coding expression universe."""
import hashlib
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parent
WS = OUT.parent.parent
RAW = WS/'work/erastin/raw'
BASE = {'CTRP': WS/'outputs/erastin_elastic_net_comparison', 'PRISM': WS/'outputs/erastin_prism_direct_comparison'}


def sha(p):
    with Path(p).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def dump(p, v):
    p.write_text(json.dumps(v, indent=2, allow_nan=False)+'\n')


def main():
    OUT.mkdir(exist_ok=True)
    records = []
    for name in ['OmicsExpressionProteinCodingGenesTPMLogp1.csv', 'Gene.csv']:
        side = json.load(open(RAW/(name+'.download.json')))
        assert sha(RAW/name) == side['sha256']
        records.append(dict(path=str((RAW/name).relative_to(WS)), sha256=side['sha256']))
    release = pd.read_csv(RAW/'Gene.csv', dtype=str, keep_default_na=False)
    rel = defaultdict(list)
    for _, r in release.iterrows():
        if r.entrez_id and r.status == 'Approved':
            rel[r.entrez_id.removesuffix('.0')].append(r)
    columns = pd.read_csv(RAW/'OmicsExpressionProteinCodingGenesTPMLogp1.csv', nrows=0).columns.tolist()
    idcol, headers = columns[0], columns[1:]
    groups = defaultdict(list)
    for j, col in enumerate(headers):
        symbol, ent = col.rsplit(' (', 1)
        ent = ent[:-1]
        assert ent.isdigit()
        groups[ent].append((j, symbol, col))
    mapping, exclusions = [], []
    for ent, candidates in groups.items():
        approved = {r.symbol for r in rel[ent]}
        matches = [x for x in candidates if x[1] in approved]
        if len(candidates) == 1:
            picked, rule = candidates[0], 'unique_Entrez'
        elif len(matches) == 1:
            picked, rule = matches[0], 'duplicate_Entrez_resolved_by_approved_HGNC_symbol'
        else:
            exclusions.append(dict(entrez_id=ent, columns=[x[2] for x in candidates], reason='ambiguous_duplicate_Entrez'))
            continue
        j, symbol, col = picked
        annotation = [r for r in rel[ent] if r.symbol == symbol] or rel[ent]
        mapping.append(dict(raw_column_0based=j, depmap_column=col, depmap_symbol=symbol, entrez_id=ent,
                            current_symbols='|'.join(sorted({r.symbol for r in annotation})),
                            hgnc_ids='|'.join(sorted({r.hgnc_id for r in annotation})),
                            ensembl_ids='|'.join(sorted({r.ensembl_gene_id for r in annotation if r.ensembl_gene_id})),
                            mapping_rule=rule, alternative_columns='|'.join(x[2] for x in candidates if x != picked)))
    mapping.sort(key=lambda r: r['raw_column_0based'])
    assert len({r['entrez_id'] for r in mapping}) == len(mapping)
    table = pd.DataFrame(mapping)
    table.insert(0, 'feature_index', np.arange(len(table)))
    inputs = {}
    for assay, source in BASE.items():
        known = {r['path']: r['sha256'] for r in json.load(open(source/'artifact_manifest.json'))['files']}
        for name in ['panel_inputs.npz', 'shared_splits.npz', 'protocol.json', 'validation.json']:
            assert sha(source/name) == known[name]
            records.append(dict(path=str((source/name).relative_to(WS)), sha256=known[name]))
        a = np.load(source/'panel_inputs.npz')
        inputs[assay] = {k: a[k] for k in ['x_64', 'genes_64', 'y', 'sample_ids', 'group_ids', 'lineage']}
    ids = np.array(sorted(set(inputs['CTRP']['sample_ids'])|set(inputs['PRISM']['sample_ids'])))
    lookup = {v: j for j, v in enumerate(ids)}
    raw_index = {h: j for j, h in enumerate(headers)}
    chosen = table.raw_column_0based.to_numpy()
    expression = np.empty((len(ids), len(chosen)), dtype=np.float32)
    found = set()
    dtype = {h: np.float32 for h in headers}
    dtype[idcol] = str
    for frame in pd.read_csv(RAW/'OmicsExpressionProteinCodingGenesTPMLogp1.csv', dtype=dtype, chunksize=64):
        for _, row in frame[frame[idcol].isin(lookup)].iterrows():
            sid = row[idcol]
            assert sid not in found
            expression[lookup[sid]] = row.iloc[1:].to_numpy(dtype=np.float32)[chosen]
            found.add(sid)
    assert found == set(ids) and np.isfinite(expression).all()
    by_symbol = defaultdict(list)
    for r in table.to_dict('records'):
        by_symbol[r['depmap_symbol']].append(r['feature_index'])
    core = []
    for symbol in inputs['CTRP']['genes_64']:
        assert len(by_symbol[symbol]) == 1, symbol
        core.append(by_symbol[symbol][0])
    core = np.array(core, dtype=int)
    assert len(set(core)) == 64
    candidates = np.setdiff1d(np.arange(len(table)), core)
    table['in_core64'] = table.feature_index.isin(core)
    table.to_csv(OUT/'gene_mapping.tsv', sep='\t', index=False)
    np.save(OUT/'expression.npy', expression)
    np.savez_compressed(OUT/'expression_index.npz', sample_ids=ids, core_indices=core, candidate_indices=candidates,
                        gene_columns=table.depmap_column.to_numpy(dtype=str), genes=table.depmap_symbol.to_numpy(dtype=str),
                        entrez_ids=table.entrez_id.to_numpy(dtype=str))
    for assay, a in inputs.items():
        dest = OUT/assay
        dest.mkdir(exist_ok=True)
        rows = np.array([lookup[v] for v in a['sample_ids']])
        np.testing.assert_array_equal(expression[rows[:, None], core].astype(float), a['x_64'])
        np.savez_compressed(dest/'inputs.npz', expression_rows=rows, **a)
        shutil.copyfile(BASE[assay]/'shared_splits.npz', dest/'shared_splits.npz')
    # Independently compare every retained CTRP expression entry to the original
    # raw-matched matrix, not only the core columns.
    old = np.load(WS/'outputs/erastin_40gene_comparison/matched_depmap_expression.npz')
    ix = {v: j for j, v in enumerate(old['sample_ids'])}
    rows = np.array([ix[v] for v in inputs['CTRP']['sample_ids']])
    actual = expression[[lookup[v] for v in inputs['CTRP']['sample_ids']]]
    expected = old['X_log2tpm1'][rows][:, chosen]
    np.testing.assert_array_equal(actual, expected)
    coverage = dict(raw_protein_coding_columns=len(headers), unique_Entrez_features=len(table),
                    core_genes=64, candidate_genes=len(candidates), raw_duplicate_Entrez_ids=sum(len(v) > 1 for v in groups.values()),
                    ambiguous_excluded_ids=exclusions, union_lines=len(ids), all_expression_finite=True,
                    scope='All measured protein-coding expression features, deduplicated by stable Entrez ID; no pathway restrictions. This is not a noncoding-transcript or whole-DNA-genome scan.',
                    response_independent_mapping=True, core64_expression_exact=True,
                    original_CTRP_full_expression_entries_checked=int(actual.size))
    dump(OUT/'coverage_summary.json', coverage)
    dump(OUT/'preparation_source_manifest.json', dict(files=records))
    paths = ['expression.npy', 'expression_index.npz', 'gene_mapping.tsv']+[f'{a}/{n}' for a in BASE for n in ['inputs.npz', 'shared_splits.npz']]
    dump(OUT/'preparation_validation.json', dict(passed=True, created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                                                **coverage, files={n: sha(OUT/n) for n in paths}))
    print(json.dumps(coverage, indent=2), flush=True)


if __name__ == '__main__':
    main()
