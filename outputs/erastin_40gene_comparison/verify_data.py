#!/usr/bin/env python3
"""Check artifacts and independently compare selected data to source files."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np


def rows(path):
    return list(csv.DictReader(path.open(), delimiter='\t'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', type=Path, required=True)
    ap.add_argument('--bundle-dir', type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    out, raw = args.bundle_dir, args.raw_dir
    d = np.load(out / 'comparison_inputs.npz', allow_pickle=False)
    full = np.load(out / 'matched_depmap_expression.npz', allow_pickle=False)
    cohort = rows(out / 'cohort.tsv')
    panel = rows(out / 'panel_mapping.tsv')
    split = json.loads((out / 'splits.json').read_text())
    report = json.loads((out / 'preparation_report.json').read_text())
    ids = d['sample_ids'].tolist()
    assert ids == [r['ModelID'] for r in cohort] == full['sample_ids'].tolist()
    assert len(ids) == len(set(ids)) == report['cancer_cell_lines_matched']
    assert d['X_panel_log2tpm1'].shape == (len(ids), 40)
    assert d['X_bulkformer_lntpm1'].shape == (len(ids), 20010)
    assert len(set(d['panel_indices_0based'].tolist())) == 40
    assert d['bulkformer_observed_gene_mask'][d['panel_indices_0based']].all()
    assert np.isfinite(d['X_bulkformer_lntpm1']).all()
    observed = d['bulkformer_observed_gene_mask']
    assert (d['X_bulkformer_lntpm1'][:, ~observed] == -10).all()
    assert (d['X_bulkformer_lntpm1'][:, observed] >= 0).all()
    assert np.array_equal(d['y_sensitivity'], -d['y_auc'])
    assert np.allclose(d['y_auc'], [float(r['erastin_auc_reported']) for r in cohort], rtol=0, atol=1e-15)
    assert all(r['OncotreePrimaryDisease'] != 'Non-Cancerous' for r in cohort)

    # Independently re-read every selected label with csv, not the build-time pandas join.
    source_labels = {r['']: float(r['DPC-002562']) for r in csv.DictReader((raw / 'CTRPAUCMatrix.csv').open())
                     if r['DPC-002562']}
    assert np.allclose(d['y_auc'], [source_labels[i] for i in ids], rtol=0, atol=1e-15)
    # Check 40 genes in first/middle/last rows and all source genes for 3 rows.
    checked = set([ids[0], ids[len(ids)//2], ids[-1]])
    id_to_row = {mid: i for i, mid in enumerate(ids)}
    with (raw / 'OmicsExpressionProteinCodingGenesTPMLogp1.csv').open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            mid = row['']
            if mid not in checked:
                continue
            i = id_to_row[mid]
            source_panel = np.array([float(row[p['depmap_column']]) for p in panel])
            assert np.allclose(d['X_panel_log2tpm1'][i], source_panel, atol=1e-6, rtol=1e-7)
            assert np.allclose(d['X_bulkformer_lntpm1'][i, d['panel_indices_0based']], source_panel * np.log(2), atol=1e-6, rtol=1e-7)
            assert np.array_equal(full['X_log2tpm1'][i], np.array([float(row[c]) for c in full['gene_columns']], dtype=np.float32))
    # A real zero must survive unchanged, unlike an absent vocabulary gene.
    assert (d['X_bulkformer_lntpm1'][:, observed] == 0).any()
    group = dict(zip(ids, d['group_ids']))
    tested = []
    for fold in split['outer_folds']:
        tr, te = set(fold['train_ModelIDs']), set(fold['test_ModelIDs'])
        assert tr.isdisjoint(te) and tr | te == set(ids)
        assert {group[i] for i in tr}.isdisjoint({group[i] for i in te})
        tested.extend(te)
        validated = []
        for inner in fold['inner_folds']:
            a, b = set(inner['train_ModelIDs']), set(inner['validation_ModelIDs'])
            assert a.isdisjoint(b) and a | b == tr
            assert {group[i] for i in a}.isdisjoint({group[i] for i in b})
            validated.extend(b)
        assert len(validated) == len(tr) and set(validated) == tr
    assert len(tested) == len(ids) and set(tested) == set(ids)
    for fold in split['leave_lineage_out']:
        a, b = fold['train_ModelIDs'], fold['test_ModelIDs']
        assert {group[i] for i in a}.isdisjoint({group[i] for i in b})
    result = {'passed': True, 'sample_count': len(ids), 'panel_genes': 40,
        'all_labels_rechecked_against_source': True, 'expression_source_rows_rechecked': sorted(checked),
        'npz_load_without_pickle': True, 'missing_vs_zero_preserved': True,
        'outer_inner_and_leave_lineage_group_separation': True,
        'row_alignment_and_log_base_conversion': True}
    (out / 'validation_results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
