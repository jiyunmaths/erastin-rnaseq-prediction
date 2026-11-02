"""Fixed genome-wide discovery design and source verification."""
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from models import GRID, ALPHAS, FULL_ALPHAS, TAB_MAX_ADDED

OUT = Path(__file__).resolve().parent
WS = OUT.parent.parent
BASE = {'CTRP': WS/'outputs/erastin_elastic_net_comparison', 'PRISM': WS/'outputs/erastin_prism_direct_comparison'}
TAB = WS/'outputs/erastin_tabpfn_comparison'
CHECKPOINT = WS/'work/tabpfn/checkpoints/tabpfn-v3.5-fast-20260909.safetensors'
ASSAYS = list(BASE)
WORKERS = 4
STATE = {}
EXPRESSION = None
INDEX = None


def sha(p):
    with Path(p).open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


def clean(v):
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple, np.ndarray)):
        return [clean(x) for x in v]
    if isinstance(v, np.generic):
        return clean(v.item())
    if isinstance(v, float) and not np.isfinite(v):
        return None
    return v


def dump(p, v):
    temp = p.with_suffix(p.suffix+'.tmp')
    temp.write_text(json.dumps(clean(v), indent=2, allow_nan=False)+'\n')
    temp.replace(p)


def initialize():
    global EXPRESSION, INDEX
    EXPRESSION = np.load(OUT/'expression.npy', mmap_mode='r')
    a = np.load(OUT/'expression_index.npz')
    INDEX = {k: a[k] for k in a.files}
    for assay in ASSAYS:
        a = np.load(OUT/assay/'inputs.npz')
        s = np.load(OUT/assay/'shared_splits.npz')
        STATE[assay] = dict(rows=a['expression_rows'], xcore=a['x_64'], y=a['y'], ids=a['sample_ids'],
                            groups=a['group_ids'], lineage=a['lineage'], test=s['outer_test'], inner=s['inner_fold'],
                            seeds=s['outer_split_seeds'], inner_seeds=s['inner_split_seeds'])


def features(assay, rows, genes=None):
    ix = STATE[assay]['rows'][rows]
    if genes is None:
        return np.asarray(EXPRESSION[ix], dtype=float)
    return np.asarray(EXPRESSION[ix[:, None], genes], dtype=float)


def root_for(assay, arm, r):
    return OUT/assay/'folds'/f'{arm}__r{r+1:02d}'


def prepare():
    pv = json.load(open(OUT/'preparation_validation.json'))
    assert pv['passed']
    for n, h in pv['files'].items():
        assert sha(OUT/n) == h
    records = json.load(open(OUT/'preparation_source_manifest.json'))['files']
    for source, names in [(TAB, ['protocol.json', 'validation.json', 'summary.tsv', 'source_manifest.json', 'common.py']),
                          *[(BASE[a], ['summary.tsv', 'predictions.npz']) for a in ASSAYS]]:
        known = {r['path']: r['sha256'] for r in json.load(open(source/'artifact_manifest.json'))['files']}
        for name in names:
            assert sha(source/name) == known[name]
            records.append(dict(path=str((source/name).relative_to(WS)), sha256=known[name]))
    for a in ASSAYS:
        (OUT/a/'folds').mkdir(exist_ok=True)
        known = {r['path']: r['sha256'] for r in json.load(open(TAB/'artifact_manifest.json'))['files']}
        name = f'{a}/predictions.npz'
        assert sha(TAB/name) == known[name]
        records.append(dict(path=str((TAB/name).relative_to(WS)), sha256=known[name]))
        for repeat in range(1, 21):
            for ext in ['json', 'npz', 'tabpfn_fit']:
                n = f'{a}/folds/tabpfn35_fast__p64__r{repeat:02d}.{ext}'
                assert sha(TAB/n) == known[n]
                records.append(dict(path=str((TAB/n).relative_to(WS)), sha256=known[n]))
    for row in json.load(open(TAB/'source_manifest.json'))['files']:
        assert sha(WS/row['path']) == row['sha256']
        records.append(row)
    p = dict(status='frozen_before_genomewide_screening', created_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
             core_panel=64, user_numbering='User said 60; interpreted as the existing unchanged 64-gene panel.',
             universe={k: pv[k] for k in ['raw_protein_coding_columns', 'unique_Entrez_features', 'candidate_genes', 'raw_duplicate_Entrez_ids']},
             scope='Hypothesis-agnostic search over all unique measured protein-coding genes outside the core. No pathway whitelist. Noncoding transcripts and unmeasured DNA loci are outside this RNA file.',
             cohorts=dict(CTRP=dict(samples=744, groups=733, lineages=23), PRISM=dict(samples=386, groups=379, lineages=20)),
             split_design='Same response-blind 20 lineage-stratified identity-group holdouts and five saved inner folds per assay; all evaluated lineages in both outer sets. Cervix excluded and Ampulla handling unchanged.',
             selectors=dict(marginal='Absolute training-only Pearson correlation with erastin response.',
                            conditional='Absolute training-only partial Pearson correlation after projecting gene and response onto the orthogonal complement of the 64 core genes plus training-lineage indicators and intercept. Controls use an exact SVD column-space projection, without response-tuned penalties.'),
             filters='Training-constant candidate genes excluded (centered SS <=1e-12). Conditional screen additionally excludes residual SS <=max(1e-12, raw SS*1e-8) to avoid numerical core aliases. No all-cohort variance/expression selection.',
             augmentation_grid=GRID, full_genome_ridge_alphas=FULL_ALPHAS,
             selection='Screening is repeated separately inside every inner-training fold. Choose selector, added-gene count and ridge penalty by equal-mean five-fold RMSE in original units. Re-screen on outer training only at selected refit. Core64 always retained; zero-addition fallback included.',
             full_genome_control='Ridge on all measured unique genes, with training-only StandardScaler and nested penalty selection. A separate dense-transcriptome control, not a sparse discovered panel.',
             tabpfn=dict(checkpoint=json.load(open(TAB/'protocol.json'))['checkpoint'], package='9.0.0', n_estimators=8,
                         device='cpu', inference_precision='float32', tuning_config=None,
                         ridge_screen='Use the outer-training ridge-selected selector and min(selected added count,32); zero addition reuses exact saved core64 TabPFN prediction.',
                         conditional32='Prespecified 32 highest conditional-ranked genes from outer training; fixed size without TabPFN outcome tuning.',
                         max_added=TAB_MAX_ADDED, inference='Same fixed checkpoint/preset and split seed as prior core64 model. No TabPFN-based selection using test data. Identical panels can reuse one context.'),
             weighting='Unweighted training; no culture or lineage predictors. Lineage is used only in the conditional screening control.',
             primary_comparisons=[['ridge_augmented', 'ridge64'], ['ridge_genome', 'ridge64'],
                                  ['tab_ridge_screen32', 'tab64'], ['tab_conditional32', 'tab64']],
             secondary_comparisons=[['ridge_augmented', 'tab64'], ['tab_ridge_screen32', 'svr64'], ['tab_conditional32', 'svr64']],
             primary_metric='pooled_rmse', secondary_metrics=['pooled_spearman', 'pooled_r2', 'macro_all_rmse'],
             uncertainty=dict(count=5000, seed=2026112201, unit='Whole-lineage paired bootstrap retaining all models and repeats; conditional exploratory intervals.'),
             gene_reporting='Outer-training selection frequency, ranking and standardized ridge coefficient stability. Overlapping repetitions are not independent confirmations. No test-response-based shortlist is fed back into prediction models.',
             limitations=['Selection is protected against direct validation-label leakage, but this previously inspected cohort is not a fresh confirmatory dataset.',
                          'Performance belongs to the training-fold gene-selection procedure; a consolidated final gene list is not thereby independently validated.',
                          'Training-only univariate screens can miss nonlinear interaction-only signals. TabPFN additions are CPU-capped at 32 and selected without TabPFN-specific tuning.',
                          'Lineages are represented in training/testing. Candidate associations are predictive, not causal ferroptosis mechanisms.',
                          'Original CTRP harmonized AUC recipe remains incompletely verified; both assays/endpoints remain separate.'],
             references=['https://pmc.ncbi.nlm.nih.gov/articles/PMC124442/'],
             data_sha256=pv['files'], fitting_scripts={n: sha(OUT/n) for n in ['prepare_data.py', 'models.py', 'common.py', 'run_ridge.py', 'run_tabpfn.py', 'validate_preflight.py', 'make_splits.py']})
    if (OUT/'protocol.json').exists():
        old = json.load(open(OUT/'protocol.json'))
        p['created_utc'] = old['created_utc']
        assert clean(p) == old
    else:
        dump(OUT/'protocol.json', p)
    dump(OUT/'source_manifest.json', dict(files=list({r['path']: r for r in records}.values())))
    return sha(OUT/'protocol.json')


def verify_sources():
    for r in json.load(open(OUT/'source_manifest.json'))['files']:
        assert sha(WS/r['path']) == r['sha256']
    p = json.load(open(OUT/'protocol.json'))
    for n, h in {**p['data_sha256'], **p['fitting_scripts']}.items():
        assert sha(OUT/n) == h
