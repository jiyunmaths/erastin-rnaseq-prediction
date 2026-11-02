#!/usr/bin/env python3
"""Rebuild matched, unscaled data. Requires numpy and pandas; see README.txt."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20261031
COMMIT = '6d7f21b513299c7803887a7954a11d2c95039f47'
COMPOUND = 'DPC-002562'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def dump(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def table(path, frame):
    frame.to_csv(path, sep='\t', index=False, float_format='%.17g')


def read_genes(path, sep):
    rows = list(csv.DictReader(open(path), delimiter=sep))
    for r in rows:
        r['entrez_id'] = r['entrez_id'].removesuffix('.0')
    return rows


def make_folds(frame, k, seed):
    """Greedy group assignment balancing lineage counts and total fold sizes.

    Uses only identity and lineage, never expression or response. Every related
    model group is assigned intact. Rare lineages cannot occur in every fold.
    """
    rng = np.random.default_rng(seed)
    lineages = sorted(frame.OncotreeLineage.unique())
    counts = pd.crosstab(frame.group_id, frame.OncotreeLineage).reindex(columns=lineages, fill_value=0)
    groups = counts.index.to_numpy()
    rng.shuffle(groups)
    groups = sorted(groups, key=lambda g: -int(counts.loc[g].sum()))
    target = counts.sum(axis=0).to_numpy() / k
    total_target = len(frame) / k
    per_lineage = np.zeros((k, len(lineages)))
    total = np.zeros(k)
    assignment = {}
    for group in groups:
        c = counts.loc[group].to_numpy()
        n = c.sum()
        costs = (((per_lineage + c) / target) ** 2 - (per_lineage / target) ** 2).sum(axis=1)
        costs += 0.25 * (((total + n) / total_target) ** 2 - (total / total_target) ** 2)
        choices = np.flatnonzero(np.isclose(costs, costs.min(), rtol=0, atol=1e-12))
        fold = int(rng.choice(choices))
        assignment[group] = fold
        total[fold] += n
        per_lineage[fold] += c
    result = frame.group_id.map(assignment).astype(int)
    assert set(result) == set(range(k))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--raw-dir', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, default=Path(__file__).resolve().parent)
    args = ap.parse_args()
    raw, out = args.raw_dir, args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((out / 'source_manifest.json').read_text())
    for item in manifest['files']:
        assert digest(raw / item['local_name']) == item['sha256'], item['local_name']
    panel = json.loads((out / 'panel_definition.json').read_text())
    symbols = panel['primary_panel_symbols']
    assert len(symbols) == len(set(symbols)) == 40
    vocab = pd.read_csv(raw / 'vocabulary.txt', dtype=str, keep_default_na=False)
    assert len(vocab) == 20010 and vocab.ensg_id.is_unique
    release_genes = read_genes(raw / 'Gene.csv', ',')
    current_genes = read_genes(raw / 'hgnc.txt', '\t')
    by_ens, by_symbol = defaultdict(list), defaultdict(list)
    for g in release_genes:
        if g['ensembl_gene_id']:
            by_ens[g['ensembl_gene_id']].append(g)
        by_symbol[g['symbol']].append(g)
    current_by_symbol = {g['symbol']: g for g in current_genes}

    expression_file = raw / 'OmicsExpressionProteinCodingGenesTPMLogp1.csv'
    columns = pd.read_csv(expression_file, nrows=0).columns.tolist()[1:]
    by_entrez = defaultdict(list)
    for i, col in enumerate(columns):
        sym, ent = col.rsplit(' (', 1)
        by_entrez[ent[:-1]].append((col, sym, i))
    # Ensembl -> release-matched HGNC -> Entrez -> expression column.
    # Reject symbols that are approved identifiers for a DIFFERENT HGNC gene.
    # An unannotated AC/AL symbol can still be linked by the two stable IDs.
    mapping = []
    for i, v in vocab.iterrows():
        row = {'bulkformer_index_0based': int(i), 'bulkformer_symbol': v.gene_symbol,
               'ensembl_gene_id': v.ensg_id, 'hgnc_id': '', 'approved_symbol': '',
               'entrez_id': '', 'depmap_column': '', 'depmap_index_0based': -1,
               'status': 'missing_release_ensembl_mapping', 'symbol_note': ''}
        hg = by_ens[v.ensg_id]
        if len(hg) == 1:
            g = hg[0]
            row.update(hgnc_id=g['hgnc_id'], approved_symbol=g['symbol'], entrez_id=g['entrez_id'])
            cs = by_entrez[g['entrez_id']]
            row['status'] = 'absent_from_expression' if not cs else 'ambiguous_expression_entrez'
            if len(cs) == 1:
                col, ds, di = cs[0]
                conflict = any(other['hgnc_id'] != g['hgnc_id']
                               for s in [ds, v.gene_symbol] for other in by_symbol[s])
                row['status'] = 'conflicting_approved_symbols' if conflict else 'mapped'
                if not conflict:
                    row.update(depmap_column=col, depmap_index_0based=di)
                    names = {g['symbol'], *g['alias_symbol'].split('|'), *g['prev_symbol'].split('|')}
                    row['symbol_note'] = ('exact' if ds == v.gene_symbol == g['symbol'] else
                                          'approved_alias_or_previous_symbol' if ds in names and v.gene_symbol in names else
                                          'stable_id_bridge_with_unannotated_symbol')
        elif len(hg) > 1:
            row['status'] = 'ambiguous_release_ensembl_mapping'
        mapping.append(row)
    gm = pd.DataFrame(mapping)
    mapped = gm.status.eq('mapped')
    assert gm.loc[mapped, 'depmap_column'].is_unique, 'Do not duplicate expression columns across model tokens'
    panel_rows = []
    for order, s in enumerate(symbols):
        candidates = gm[gm.bulkformer_symbol.eq(s)]
        assert len(candidates) == 1 and candidates.iloc[0].status == 'mapped', s
        r = candidates.iloc[0].to_dict()
        current = current_by_symbol[s]
        assert all(r[k] == current[k] for k in ['hgnc_id', 'ensembl_gene_id', 'entrez_id']), s
        assert r['approved_symbol'] == s and r['depmap_column'] == f"{s} ({r['entrez_id']})", s
        r.update(panel_index_0based=order, current_hgnc_agrees=True)
        panel_rows.append(r)
    table(out / 'gene_mapping.tsv.gz', gm)
    table(out / 'panel_mapping.tsv', pd.DataFrame(panel_rows))
    dump(out / 'panel_mapping.json', panel_rows)
    pidx = np.array([r['bulkformer_index_0based'] for r in panel_rows], dtype=np.int64)

    models = pd.read_csv(raw / 'Model.csv', dtype=str, keep_default_na=False).set_index('ModelID')
    curves = pd.read_csv(raw / 'CTRPResponseCurves.csv', low_memory=False)
    curves = curves[curves.CompoundID.eq(COMPOUND)].copy().set_index('ModelID')
    assert len(curves) == 795 and curves.index.is_unique
    assert set(curves.CompoundName) == {'ERASTIN'} and set(curves.SampleID) == {'CTRP:63578'}
    assert set(curves.DoseUnit) == {'uM'}
    compound = pd.read_csv(raw / 'PortalCompounds.csv', dtype=str, keep_default_na=False)
    compound = compound[compound.CompoundID.eq(COMPOUND)]
    assert len(compound) == 1 and 'CTRP:63578' in compound.iloc[0].SampleIDs.split(';')
    dump(out / 'compound_identity.json', compound.iloc[0].to_dict())
    auc = pd.read_csv(raw / 'CTRPAUCMatrix.csv', index_col=0)[COMPOUND].dropna()
    assert auc.index.is_unique and set(auc.index) == set(curves.index)
    assert np.isfinite(auc).all()
    expression_ids = pd.read_csv(expression_file, usecols=[0]).iloc[:, 0]
    assert expression_ids.is_unique and models.index.is_unique
    expr_set = set(expression_ids)
    audit = []
    for mid in sorted(auc.index):
        reasons = []
        if mid not in models.index:
            reasons.append('no_24Q4_model_metadata')
        else:
            m = models.loc[mid]
            if m.OncotreePrimaryDisease == 'Non-Cancerous':
                reasons.append('non_cancerous_model')
            if m.ModelType != 'Cell Line':
                reasons.append('not_cell_line')
            if not m.OncotreeLineage:
                reasons.append('missing_lineage')
        if mid not in expr_set:
            reasons.append('no_24Q4_expression')
        cr = curves.loc[mid]
        if not np.isfinite(cr[['EC50', 'LowerAsymptote', 'UpperAsymptote', 'Slope', 'MinimumDose', 'MaximumDose']].astype(float)).all():
            reasons.append('nonfinite_curve_parameters')
        if not 0 < cr.MinimumDose < cr.MaximumDose:
            reasons.append('invalid_dose_range')
        audit.append({'ModelID': mid, 'included': not reasons, 'exclusion_reasons': ';'.join(reasons)})
    audit = pd.DataFrame(audit)
    ids = audit.loc[audit.included, 'ModelID'].tolist()
    # Stream the original matrix; use exactly one default expression row per ModelID.
    pieces = []
    for chunk in pd.read_csv(expression_file, index_col=0, chunksize=100):
        pieces.append(chunk.loc[chunk.index.isin(ids)])
    expr = pd.concat(pieces).loc[ids]
    assert np.isfinite(expr.to_numpy()).all() and (expr.to_numpy() >= 0).all()
    x_depmap = expr.to_numpy(dtype=np.float32)
    panel_x = expr[[r['depmap_column'] for r in panel_rows]].to_numpy(dtype=np.float32)
    x = np.full((len(ids), len(vocab)), -10, dtype=np.float32)
    mapped_positions = gm.loc[mapped, 'bulkformer_index_0based'].to_numpy(dtype=int)
    source_positions = gm.loc[mapped, 'depmap_index_0based'].to_numpy(dtype=int)
    x[:, mapped_positions] = (expr.iloc[:, source_positions].to_numpy(dtype=np.float64) * np.log(2)).astype(np.float32)
    assert np.allclose(x[:, pidx], panel_x * np.log(2), atol=2e-6)

    samples = models.loc[ids].copy()
    samples.index.name = 'ModelID'
    samples = samples.join(curves).join(auc.rename('erastin_auc_reported'))
    samples['erastin_sensitivity'] = -samples.erastin_auc_reported
    samples['modal_dose_range_subset'] = samples.MinimumDose.eq(0.002) & samples.MaximumDose.eq(66.0)
    defaults = pd.read_csv(raw / 'OmicsDefaultModelProfiles.csv', dtype=str)
    defaults = defaults[defaults.ProfileType.eq('rna')].set_index('ModelID')
    assert defaults.index.is_unique
    samples = samples.join(defaults[['ProfileID']].rename(columns={'ProfileID': 'RNAProfileID'}))
    assert samples.RNAProfileID.notna().all()
    profiles = pd.read_csv(raw / 'OmicsProfiles.csv', dtype=str, keep_default_na=False).set_index('ProfileID')
    assert profiles.index.is_unique
    for mid, row in samples.iterrows():
        pr = profiles.loc[row.RNAProfileID]
        assert pr.ModelID == mid and pr.Datatype == 'rna'
    samples['RNAConditionID'] = samples.RNAProfileID.map(profiles.ModelCondition)
    samples['RNASequencingSource'] = samples.RNAProfileID.map(profiles.Source)

    # Connected components over all release models capture shared patients and
    # duplicate cell-line identifiers, even through an intermediate non-cohort model.
    parent = {m: m for m in models.index}
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a
    for field in ['PatientID', 'RRID', 'CCLEName', 'StrippedCellLineName']:
        seen = {}
        for mid, value in models[field].items():
            if not value:
                continue
            if value in seen:
                a, b = find(mid), find(seen[value])
                parent[max(a, b)] = min(a, b)
            else:
                seen[value] = mid
    samples['group_id'] = ['G_' + find(mid) for mid in ids]
    samples['row_index_0based'] = np.arange(len(samples))
    samples['outer_fold'] = make_folds(samples, 5, SEED)
    split_def = {'seed': SEED, 'algorithm': 'deterministic greedy lineage-balanced group assignment',
                 'group_fields': ['PatientID', 'RRID', 'CCLEName', 'StrippedCellLineName'],
                 'outer_folds': [], 'leave_lineage_out': []}
    for fold in range(5):
        tr = samples.outer_fold.ne(fold)
        inner = make_folds(samples[tr], 4, SEED + fold + 1)
        samples[f'inner_fold_when_outer_{fold}'] = -1
        samples.loc[tr, f'inner_fold_when_outer_{fold}'] = inner
        train, test = samples[tr], samples[~tr]
        assert set(train.group_id).isdisjoint(test.group_id)
        split_def['outer_folds'].append({'fold': fold, 'train_ModelIDs': train.index.tolist(),
                                        'test_ModelIDs': test.index.tolist(), 'inner_folds': [
                                            {'fold': j,
                                             'train_ModelIDs': train.index[inner.ne(j)].tolist(),
                                             'validation_ModelIDs': train.index[inner.eq(j)].tolist()}
                                            for j in range(4)]})
    for lineage in sorted(samples.OncotreeLineage.unique()):
        test = samples[samples.OncotreeLineage.eq(lineage)]
        train = samples[~samples.group_id.isin(test.group_id)]
        split_def['leave_lineage_out'].append({'lineage': lineage,
            'train_ModelIDs': train.index.tolist(), 'test_ModelIDs': test.index.tolist(),
            'test_n': len(test), 'interpretation': 'descriptive_only_small_n' if len(test) < 10 else 'eligible'})
    table(out / 'cohort.tsv', samples.reset_index())
    table(out / 'cohort_audit.tsv', audit)
    table(out / 'erastin_all_795_labels.tsv', curves.join(auc.rename('erastin_auc_reported')).reset_index())
    table(out / 'lineage_counts.tsv', samples.groupby('OncotreeLineage').agg(n=('group_id', 'size'), groups=('group_id', 'nunique')).reset_index())
    table(out / 'fold_lineage_counts.tsv', pd.crosstab(samples.OncotreeLineage, samples.outer_fold).reset_index())
    dump(out / 'splits.json', split_def)
    np.savez_compressed(out / 'comparison_inputs.npz', sample_ids=np.array(ids),
        X_panel_log2tpm1=panel_x, panel_symbols=np.array(symbols),
        X_bulkformer_lntpm1=x, bulkformer_ensembl_ids=vocab.ensg_id.to_numpy(dtype=str),
        bulkformer_observed_gene_mask=mapped.to_numpy(), panel_indices_0based=pidx,
        y_auc=samples.erastin_auc_reported.to_numpy(), y_sensitivity=samples.erastin_sensitivity.to_numpy(),
        group_ids=samples.group_id.to_numpy(dtype=str), lineage=samples.OncotreeLineage.to_numpy(dtype=str),
        outer_fold=samples.outer_fold.to_numpy(dtype=np.int64),
        modal_dose_range_subset=samples.modal_dose_range_subset.to_numpy())
    np.savez_compressed(out / 'matched_depmap_expression.npz',
                        sample_ids=np.array(ids), gene_columns=np.array(columns), X_log2tpm1=x_depmap)

    cfg = {'status': 'data_prepared_embeddings_not_computed_models_not_trained',
        'target': 'y_sensitivity = -reported_harmonized_CTRP_AUC',
        'head': 'ridge', 'ridge_alphas': np.logspace(-4, 4, 17).tolist(),
        'tuning': '4 inner grouped folds within 5 outer grouped folds; minimize validation RMSE',
        'preprocessing': 'Fit mean and standard deviation on training rows only, separately for each fold and representation; no cohort-wide scaling or feature selection.',
        'primary_metrics': ['out_of_fold_Spearman', 'out_of_fold_RMSE', 'out_of_fold_R2'],
        'uncertainty': 'paired bootstrap by group_id across the same out-of-fold rows',
        'models': [
            {'name': 'panel_expression', 'input': 'X_panel_log2tpm1', 'feature_count': 40},
            {'name': 'panel_contextual_embedding', 'input': 'X_bulkformer_lntpm1',
             'pooling': 'mean of model output at panel_indices_0based', 'expected_feature_count': 643},
            {'name': 'all_gene_contextual_embedding', 'input': 'X_bulkformer_lntpm1',
             'pooling': 'mean of model output across all 20010 positions, including missing-gene tokens, matching the pinned notebook all-gene protocol', 'expected_feature_count': 643}],
        'bulkformer': {'family': 'original_bulkformer_147M', 'repository_commit': COMMIT,
            'checkpoint_filename': 'bulkformer_147M.pt', 'checkpoint_sha256': None,
            'checkpoint_status': 'not_downloaded_or_loaded', 'hidden_dimension': 640,
            'output_dimension': 643, 'additional_channels': ['mask_fraction', 'observed_expression_mean', 'nonzero_ratio_as_implemented'],
            'freeze': True, 'eval': True, 'additional_random_masking': False,
            'mask_prob_argument': float((~mapped).mean()),
            'mask_prob_note': 'In the pinned forward method this argument appends a scalar feature; it does not randomly mask input. Use the missing-gene fraction as in the notebook.',
            'missing_value': -10, 'observed_zero_value': 0,
            'normalization': 'ln(TPM+1) = source log2(TPM+1) * ln(2); no re-TPM normalization.',
            'context_warning': 'The panel pooling arm sees the full transcriptome and the 3 global summary channels, so it is not a panel-only information model.',
            'pretraining_overlap': 'not_resolved; no claim that evaluation cell lines were absent from foundation-model pretraining'},
        'robustness_checks': ['modal_dose_range_subset without changing the primary labels',
            'group-safe leave-lineage-out evaluation', 'lineage-only baseline',
            '640-channel trunk-only pooling for both embedding arms',
            'observed-gene-only all-gene pooling as a secondary protocol']}
    dump(out / 'comparison_config.json', cfg)
    report = {'status': cfg['status'], 'depmap_release': '24Q4',
        'compound_id': COMPOUND, 'ctrp_compound_sample_id': 'CTRP:63578',
        'label_source': 'Zenodo 22125121 archived CTRPAUCMatrix.csv and CTRPResponseCurves.csv; attributed to Harmonized CTD2 25Q2; direct upstream byte identity not independently verified',
        'auc_definition_status': 'reported values preserved unchanged; observed range is [0,1]; exact upstream integration/clipping recipe not independently verified',
        'genes_in_panel': 40, 'panel_verified_in_depmap_and_bulkformer': 40,
        'depmap_expression_rows': len(expression_ids), 'depmap_expression_genes': len(columns),
        'bulkformer_vocabulary_genes': len(vocab), 'bulkformer_mapped_genes': int(mapped.sum()),
        'bulkformer_missing_or_unresolved_genes': int((~mapped).sum()),
        'full_mapping_status_counts': gm.status.value_counts().to_dict(),
        'full_mapping_symbol_notes': gm.loc[mapped, 'symbol_note'].value_counts().to_dict(),
        'erastin_labels_available': len(auc), 'cancer_cell_lines_matched': len(ids),
        'excluded_label_rows': int((~audit.included).sum()),
        'exclusion_reason_combinations': audit.loc[~audit.included, 'exclusion_reasons'].value_counts().to_dict(),
        'lineages': samples.OncotreeLineage.nunique(), 'patients': samples.PatientID.nunique(),
        'identity_groups': samples.group_id.nunique(),
        'non_singleton_groups': int((samples.groupby('group_id').size() > 1).sum()),
        'modal_dose_range_n': int(samples.modal_dose_range_subset.sum()),
        'dose_range_counts': [{'minimum_uM': float(a), 'maximum_uM': float(b), 'n': int(n)}
            for (a, b), n in samples.groupby(['MinimumDose', 'MaximumDose']).size().items()],
        'auc_summary': {k: float(v) for k, v in samples.erastin_auc_reported.describe().items()},
        'outer_fold_sizes': {int(k): int(v) for k, v in samples.outer_fold.value_counts().sort_index().items()},
        'panel_missing_values': int(np.isnan(panel_x).sum()),
        'panel_constant_genes': [s for s, sd in zip(symbols, panel_x.std(axis=0)) if sd == 0],
        'expression_nonfinite_values': int((~np.isfinite(x_depmap)).sum()),
        'measured_expression_zeros': int((x_depmap == 0).sum()),
        'validation': ['source SHA256 checks', '40-way HGNC release/current/Entrez/Ensembl agreement',
            'unique ModelID and compound-specific joins', 'finite nonnegative source expression',
            'RNA profile IDs agree with model IDs', 'exact shared row order',
            'log-base conversion consistency', 'related-group separation across outer folds'],
        'limitations': ['No raw well-level QC, replicate-level uncertainty, or exposure-time field in the archived curve file.',
            'Tested dose intervals differ; retain the modal-interval sensitivity analysis.',
            'Baseline RNA and drug response are not guaranteed to come from identical passages or culture conditions.',
            'Current 26Q3 portal was unavailable to automated download; 24Q4 is intentionally pinned.',
            'Full-transcriptome mappings with conflicting approved symbols are masked rather than guessed.',
            'BulkFormer checkpoint assets and embeddings still need to be obtained and verified before training.']}
    dump(out / 'preparation_report.json', report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == '__main__':
    main()
