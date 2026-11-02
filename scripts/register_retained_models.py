#!/usr/bin/env python3
"""Verify and register the existing 64-gene CTRP/PRISM models without refitting."""
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'outputs/retained_64_gene_models'
FAMILIES = ['ridge', 'elastic_net', 'neural_network']
SOURCES = {
    'CTRP': ROOT / 'outputs/erastin_elastic_net_comparison',
    'PRISM': ROOT / 'outputs/erastin_prism_direct_comparison',
}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def relative(path):
    return path.relative_to(ROOT).as_posix()


def verified(path, expected):
    actual = sha(path)
    assert actual == expected, f'Checksum mismatch: {relative(path)}'
    return dict(path=relative(path), sha256=actual)


def main():
    OUT.mkdir(exist_ok=True)
    cohorts = {}
    panel = None
    checked = 0
    for assay, source in SOURCES.items():
        manifest = json.loads((source / 'artifact_manifest.json').read_text())
        expected = {r['path']: r['sha256'] for r in manifest['files']}
        refs = {}
        for name in ['protocol.json', 'validation.json', 'panel_inputs.npz',
                     'shared_splits.npz', 'summary.tsv']:
            refs[name] = verified(source / name, expected[name])
        protocol = json.loads((source / 'protocol.json').read_text())
        validation = json.loads((source / 'validation.json').read_text())
        assert validation['passed']
        genes = protocol['panels']['64']
        assert len(genes) == len(set(genes)) == 64
        if panel is None:
            panel = genes
        else:
            assert panel == genes, 'The assays have different panel gene orders.'
        with (source / 'summary.tsv').open() as stream:
            summaries = {r['model']: r for r in csv.DictReader(stream, delimiter='\t')}
        models = {}
        for family in FAMILIES:
            folds = []
            for repeat in range(1, 21):
                stem = f'folds/{family}__p64__r{repeat:02d}'
                files = {ext: verified(source / (stem + '.' + ext),
                                       expected[stem + '.' + ext])
                         for ext in ['joblib', 'npz', 'json']}
                meta = json.loads((source / (stem + '.json')).read_text())
                assert (meta['family'], meta['panel'], meta['repeat']) == (family, 64, repeat)
                assert meta['joblib_sha256'] == files['joblib']['sha256']
                assert meta['npz_sha256'] == files['npz']['sha256']
                assert meta['train_lineages'] == meta['test_lineages'] == protocol['lineages']
                folds.append(dict(repeat=repeat,
                                  outer_split_seed=meta['outer_split_seed'],
                                  inner_split_seed=meta['inner_split_seed'],
                                  neural_initialization_seeds=meta['neural_seeds'],
                                  selected_parameters=meta['selected_parameters'],
                                  files=files))
                checked += 1
            row = summaries[family + '__64']
            macro = 'macro23_rmse_mean' if assay == 'CTRP' else 'macro_all_rmse_mean'
            models[family] = dict(
                mean_heldout_metrics=dict(
                    rmse=float(row['pooled_rmse_mean']),
                    spearman=float(row['pooled_spearman_mean']),
                    r2=float(row['pooled_r2_mean']),
                    equal_lineage_rmse=float(row[macro])),
                outer_models=folds)
        cohorts[assay] = dict(
            source_experiment=relative(source), samples=protocol['samples'],
            identity_groups=protocol['identity_groups'], lineages=protocol['lineages'],
            response='Negative harmonized CTRP erastin AUC' if assay == 'CTRP'
                     else 'Negative reported PRISM erastin dose-response AUC',
            source_artifact_manifest=dict(path=relative(source / 'artifact_manifest.json'),
                                          sha256=sha(source / 'artifact_manifest.json')),
            input_and_validation_files=refs, models=models)
    assert checked == 120
    registry = dict(
        schema_version=1, decision_date='2026-11-02',
        status='Retained for further work at the user request',
        gene_count=64, genes_in_input_order=panel,
        model_families=FAMILIES, training_weights='Unweighted',
        culture_covariates=False, extra_Hallmark_EMT_DNA_repair_genes=False,
        validation_design='Existing 20 shared lineage-stratified identity-group holdouts per cohort; every evaluated lineage in outer train and test.',
        model_scope='Saved outer-training models; no new all-cohort refit or cross-assay pooling.',
        cohorts=cohorts, retained_outer_models=checked,
        integrity='All 360 retained model/prediction/metadata files match their original artifact manifests; input, split, protocol, validation and summary hashes verified.',
        history='Earlier studies remain unchanged and available locally; they are not the retained configuration.',
        generator=dict(path=relative(Path(__file__)), sha256=sha(Path(__file__))))
    (OUT / 'model_registry.json').write_text(json.dumps(registry, indent=2, allow_nan=False) + '\n')
    print(f'Verified and registered {checked} retained 64-gene models across CTRP and PRISM.')


if __name__ == '__main__':
    main()
