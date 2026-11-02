"""Literature-based proposal; expression coverage only, no response-based selection."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

WS = Path('/home/sji/Documents/Codex/2026-10-31/t')
OUT = WS/'outputs/erastin_panel_expansion'
SRC = WS/'outputs/erastin_40gene_comparison'

REFERENCES = {
 'xc': ('Dixon et al., 2014', 'Pharmacological inhibition of cystine-glutamate exchange induces endoplasmic reticulum stress and ferroptosis', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC4054777/', 'primary'),
 'transsulfuration': ('Hayano et al., 2016', 'Loss of cysteinyl-tRNA synthetase induces the transsulfuration pathway and inhibits ferroptosis induced by cystine deprivation', 'https://www.nature.com/articles/cdd201593', 'primary'),
 'glutaminolysis': ('Gao et al., 2015', 'Glutaminolysis and Transferrin Regulate Ferroptosis', 'https://www.sciencedirect.com/science/article/pii/S1097276515004505', 'primary'),
 'prdx6': ('Fujita et al., 2024', 'PRDX6 augments selenium utilization to limit iron toxicity and ferroptosis', 'https://www.nature.com/articles/s41594-024-01329-z', 'primary'),
 'bh4': ('Kraft et al., 2020', 'GTP Cyclohydrolase 1/Tetrahydrobiopterin Counteract Ferroptosis through Lipid Remodeling', 'https://pubmed.ncbi.nlm.nih.gov/31989025/', 'primary'),
 'bh4_screen': ('Soula et al., 2020', 'Metabolic determinants of cancer cell sensitivity to canonical ferroptosis inducers', 'https://www.nature.com/articles/s41589-020-0613-y', 'primary'),
 'original': ('Dixon et al., 2012', 'Ferroptosis: An Iron-Dependent Form of Non-Apoptotic Cell Death', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC3367386/', 'primary'),
 'iron': ('Nishizawa et al., 2020', 'Ferroptosis is controlled by the coordinated transcriptional regulation of glutathione and labile iron metabolism by BACH1', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC6952604/', 'primary'),
 'prom2': ('Brown et al., 2019', 'Prominin2 Drives Ferroptosis Resistance by Stimulating Multivesicular Body/Exosome-Mediated Iron Export', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC8316835/', 'primary'),
 'pufa': ('Lee et al., 2020', 'Polyunsaturated fatty acid biosynthesis pathway determines ferroptosis sensitivity in gastric cancer', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC7768719/', 'primary'),
 'fads': ('2024 primary study', 'FADS1/2 control lipid metabolism and ferroptosis susceptibility in triple-negative breast cancer', 'https://pubmed.ncbi.nlm.nih.gov/38926633/', 'primary'),
 'ether': ('2021 primary study', 'Peroxisome-driven ether-linked phospholipids biosynthesis is essential for ferroptosis', 'https://www.nature.com/articles/s41418-021-00769-0', 'primary'),
 'tmem': ('Reed et al., 2023', 'TMEM164 is an acyltransferase that forms ferroptotic C20:4 ether phospholipids', 'https://www.nature.com/articles/s41589-022-01253-7', 'primary'),
 'tmem_function': ('2022 primary study', 'TMEM164 is a new determinant of autophagy-dependent ferroptosis', 'https://pubmed.ncbi.nlm.nih.gov/35947500/', 'primary'),
 'becn': ('Song et al., 2018', 'AMPK-Mediated BECN1 Phosphorylation Promotes Ferroptosis by Directly Blocking System Xc- Activity', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC6081251/', 'primary'),
 'bap': ('Zhang et al., 2018', 'BAP1 links metabolic regulation of ferroptosis to tumour suppression', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC6170713/', 'primary'),
 'bap_update': ('2025 primary study', 'Epigenetic regulation of ACSL4 via H2A monoubiquitylation connects lipid metabolism to BAP1-mediated ferroptosis', 'https://www.nature.com/articles/s41418-025-01624-2.pdf', 'primary'),
 'abcc': ('Cao et al., 2019', 'A Genome-wide Haploid Genetic Screen Identifies Regulators of Glutathione Abundance and Ferroptosis Sensitivity', 'https://pubmed.ncbi.nlm.nih.gov/30726737/', 'primary'),
 'sterol': ('2024 primary study', '7-Dehydrocholesterol dictates ferroptosis sensitivity', 'https://www.nature.com/articles/s41586-023-06983-9', 'primary'),
 'vitk': ('Yang et al., 2023', 'Regulation of VKORC1L1 is critical for p53-mediated tumor suppression through vitamin K metabolism', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC10529626/', 'primary'),
 'dhodh': ('Mao et al., 2021', 'DHODH-mediated ferroptosis defence is a targetable vulnerability in cancer', 'https://www.nature.com/articles/s41586-021-03539-7', 'primary'),
 'dhodh_dispute': ('Mishima et al., 2023', 'DHODH inhibitors sensitize to ferroptosis by FSP1 inhibition', 'https://www.nature.com/articles/s41586-023-06269-0', 'primary Matters Arising'),
 'gpd2': ('Wu et al., 2022', 'A ferroptosis defense mechanism mediated by glycerol-3-phosphate dehydrogenase 2 in mitochondria', 'https://pubmed.ncbi.nlm.nih.gov/35749365/', 'primary'),
 'hmox_pro': ('Kwon et al., 2015', 'Heme oxygenase-1 accelerates erastin-induced ferroptotic cell death', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC4695193/', 'primary'),
 'hmox_anti': ('Adedoyin et al., 2018', 'Heme oxygenase-1 mitigates ferroptosis in renal proximal tubule cells', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC6031916/', 'primary'),
 'nrf2': ('Sun et al., 2016', 'Activation of the p62-Keap1-NRF2 Pathway Protects against Ferroptosis in Hepatocellular Carcinoma Cells', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC4688087/', 'primary'),
 'p53': ('Xie et al., 2017', 'The Tumor Suppressor p53 Limits Ferroptosis by Blocking DPP4 Activity', 'https://pubmed.ncbi.nlm.nih.gov/28813679/', 'primary'),
 'contact': ('Wu et al., 2019', 'Intercellular interaction dictates cancer cell ferroptosis via NF2-YAP signalling', 'https://www.nature.com/articles/s41586-019-1426-6', 'primary'),
 'taz': ('Yang et al., 2019', 'The Hippo Pathway Effector TAZ Regulates Ferroptosis in Renal Cell Carcinoma', 'https://www.sciencedirect.com/science/article/pii/S221112471931023X', 'primary'),
 'zeb': ('2024 primary study', 'Zeb1 mediates EMT/plasticity-associated ferroptosis sensitivity in cancer cells by regulating lipogenic enzyme expression and phospholipid composition', 'https://www.nature.com/articles/s41556-024-01464-1', 'primary'),
 'hypoxia': ('Zou et al., 2019', 'A GPX4-dependent cancer cell state underlies the clear-cell morphology and confers sensitivity to ferroptosis', 'https://pubmed.ncbi.nlm.nih.gov/30962421/', 'primary'),
 'otub': ('Liu et al., 2019', 'The deubiquitylase OTUB1 mediates ferroptosis via stabilization of SLC7A11', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC6467774/', 'primary'),
 'screen2026': ('Murray et al., 2026; PREPRINT', 'Dissecting Complex Interactions Between Ferroptosis and the Proteasome', 'https://pmc.ncbi.nlm.nih.gov/articles/PMC13060147/', 'preprint; not peer reviewed in retrieved version'),
 'ferrdb': ('FerrDb V3 documentation; checked 2026-11-01', 'FerrDb V3 categories, confidence levels and coverage', 'https://www.zhounan.org/ferrdb/v3/pages/help.html', 'resource documentation'),
}

rows = []
def add(symbols, tier, module, mechanism, inducer, caveat, refs, basis='functional perturbation'):
    for symbol in symbols.split():
        rows.append(dict(symbol=symbol, tier=tier, module=module, mechanism=mechanism,
                         inducer_relevance=inducer, interpretation_caveat=caveat, literature_basis=basis,
                         reference_ids=refs.split(), predictive_coefficient_sign='Not prespecified; functional direction does not fix an RNA-expression coefficient.'))

add('SLC3A2', 'add24', 'Cystine import', 'Heavy-chain partner of SLC7A11 in system xc-.', 'Direct erastin mechanism; erastin2 screen corroboration.', 'Also partners with other amino-acid transporters; do not label SLC3A2 itself as the established direct erastin-binding target.', 'xc screen2026', 'transporter mechanism plus perturbation screen')
add('CBS CTH', 'add24', 'Alternative cysteine synthesis', 'Transsulfuration supplies cysteine when extracellular cystine uptake is limited.', 'Cystine deprivation and erastin-resistance mechanism.', 'Depends on methionine supply and pathway flux; enzyme RNA alone does not measure flux.', 'transsulfuration', 'pathway induction with functional inhibition/rescue')
add('GLS2 GOT1', 'add24', 'Glutaminolysis', 'Glutamine-derived carbon and alpha-ketoglutarate support ferroptosis in tested starvation models.', 'Cystine/amino-acid deprivation; relevant to system xc- blockade.', 'Nutrient and cell-type dependent; GLS and GLS2 are distinct genes.', 'glutaminolysis')
add('PRDX6 SEPHS2', 'add24', 'Selenium and GPX4 synthesis', 'PRDX6 transfers selenium to the selenoprotein synthesis machinery including SEPHS2, supporting GPX4 abundance.', 'PRDX6 functional evidence includes IKE, RSL3 and iron; erastin2 screen support for PRDX6.', 'SEPHS2 evidence is partly synthesis-pathway evidence; selenium availability matters. PRDX6 is not interchangeable with GPX4.', 'prdx6 screen2026', 'functional PRDX6 experiments; SEPHS2 biochemical and rescue evidence')
add('GCH1 PTS SPR', 'add24', 'BH4 antioxidant defense', 'The BH4 synthesis pathway provides a GPX4-independent ferroptosis defense; complements DHFR already in the panel.', 'Strong general/GPX4-inhibitor evidence; erastin relevance is context dependent.', 'RSL3 and erastin2 screens differ; do not infer universal protection against every inducer.', 'bh4 bh4_screen screen2026')
add('IREB2', 'add24', 'Iron regulation', 'Iron regulatory protein 2 controls the iron-handling program; knockdown protected cells in the original erastin screen.', 'Direct erastin perturbation evidence in HT-1080 and Calu-1.', 'IRP2 activity is strongly controlled at the protein level.', 'original')
add('FTH1', 'add24', 'Iron storage', 'Ferritin heavy chain limits the labile iron available to sustain lipid oxidation.', 'Erastin/NRF2 studies and coordinated iron regulation.', 'Ferritin protein, iron loading and ferritinophagy can dissociate from baseline RNA.', 'nrf2 iron')
add('SLC40A1', 'add24', 'Iron export', 'Ferroportin exports cellular iron and is part of the BACH1-regulated ferroptosis program.', 'Erastin-associated iron-handling pathway evidence.', 'Activity also depends on post-translational regulation and extracellular iron handling; not a universal RNA biomarker.', 'iron', 'functional iron-handling pathway; gene-specific evidence varies')
add('PROM2', 'add24', 'Iron export', 'Prominin2 promotes ferritin-containing extracellular vesicles and iron export.', 'Perturbation studies include erastin and GPX4-targeting compounds.', 'Observed in epithelial/breast models; vesicle production and culture context matter.', 'prom2')
add('FADS1 ELOVL5', 'add24', 'PUFA synthesis', 'Desaturation and elongation create oxidizable PUFA substrates.', 'Functional ferroptosis evidence in gastric cancer; broader lipid mechanism.', 'Lineage and lipid-supply dependent; much evidence uses GPX4 inhibitors.', 'pufa fads')
add('FADS2', 'add24', 'PUFA synthesis', 'PUFA desaturation contributes to the lipid substrates that permit ferroptosis.', 'Functional ferroptosis evidence in cancer models; not erastin-specific.', 'Dietary/culture lipids and alternative desaturation routes affect the phenotype.', 'fads zeb')
add('FAR1 GNPAT AGPS', 'add24', 'Ether phospholipids', 'Peroxisomal ether-lipid synthesis provides substrates for ferroptotic lipid oxidation.', 'Strong genetic/pathway evidence, mainly GPX4 inhibition.', 'Ether-lipid products can have differing effects; availability is not determined by transcript abundance alone.', 'ether')
add('TMEM164', 'add24', 'Ether phospholipid remodeling', 'Acyltransferase generates arachidonoyl ether phospholipids; loss can suppress ferroptosis.', 'General ferroptosis and inducer-specific perturbation studies.', 'Reported autophagy and lipid-remodeling mechanisms should not be assumed identical in all models.', 'tmem tmem_function')
add('BECN1', 'add24', 'System xc- regulation', 'AMPK-dependent BECN1 phosphorylation promotes binding to SLC7A11 and inhibits system xc-.', 'Direct erastin/system xc- inhibitor evidence.', 'Phosphorylation and protein interactions matter; baseline BECN1 RNA is an indirect proxy.', 'becn')
add('BAP1', 'add24', 'Transcriptional regulation', 'BAP1 affects ferroptosis through SLC7A11 repression and additional lipid-regulatory mechanisms.', 'Direct erastin perturbation/rescue evidence.', 'Mutation/catalytic activity can matter more than transcript abundance; not exclusively an SLC7A11 mechanism.', 'bap bap_update')
add('ABCC1', 'add24', 'Glutathione export', 'MRP1/ABCC1 exports glutathione; its disruption can preserve glutathione and inhibit ferroptosis.', 'Direct functional ferroptosis study; counterintuitive for a generic drug-efflux marker.', 'Do not assume all ABC-transporter expression predicts resistance; substrates and metabolic context matter.', 'abcc')

add('SC5D', 'add16', 'Sterol antioxidant defense', 'SC5D synthesizes the 7-DHC precursor pool that can protect membranes from peroxidation.', '7-DHC protects against multiple inducers including erastin.', 'Sterol supply and flux matter; the cited study also analyzed CTRP, so its expression associations are not independent validation here.', 'sterol')
add('DHCR7', 'add16', 'Sterol antioxidant defense', 'DHCR7 consumes protective 7-DHC; its loss can protect cells from ferroptosis.', 'Multiple-inducer functional studies; erastin protection by 7-DHC.', 'Its functional effect can oppose SC5D despite both being cholesterol-pathway genes. Do not force a coefficient sign.', 'sterol')
add('VKORC1L1', 'add16', 'Vitamin K antioxidant defense', 'Vitamin K reduction provides a radical-trapping defense independent of GSH/GPX4.', 'Strong ferroptosis evidence, especially RSL3; not interchangeable with VKORC1.', 'Vitamin K supply, protein activity and p53 state matter.', 'vitk screen2026')
add('G6PD', 'add16', 'NADPH metabolism', 'Supplies NADPH to both protective redox systems and pro-oxidant pathways.', 'Knockdown suppressed erastin ferroptosis in the original Calu-1 experiments.', 'NADPH is not uniformly protective: NOX dependence can reverse the functional direction.', 'original')
add('DHODH', 'add16', 'Mitochondrial redox defense', 'Proposed mitochondrial CoQ-reducing defense parallel to mitochondrial GPX4.', 'Mostly GPX4-inhibitor evidence.', 'Contribution is context dependent and disputed; high-dose inhibitor effects can reflect FSP1 inhibition.', 'dhodh dhodh_dispute')
add('GPD2', 'add16', 'Mitochondrial redox defense', 'Glycerol-3-phosphate oxidation reduces CoQ to antioxidant ubiquinol.', 'Primary evidence emphasizes GPX4 inhibition.', 'Do not equate GPX4-inhibitor response with erastin response.', 'gpd2')
add('HMOX1', 'add16', 'Heme and iron stress', 'Heme breakdown affects labile iron and the oxidative-stress response.', 'Erastin perturbation evidence in both promoting and protective settings.', 'Opposite effects reported across cell types; neither a fixed driver nor fixed suppressor for pan-cancer RNA prediction.', 'hmox_pro hmox_anti nrf2')
add('KEAP1', 'add16', 'NRF2 regulation', 'KEAP1 regulates NRF2 stability and its ferroptosis-defense transcriptional program.', 'Erastin/NRF2 pathway evidence in liver cancer.', 'KEAP1 mutation and protein function are poorly represented by mRNA alone.', 'nrf2')
add('TP53', 'add16', 'Context-dependent stress regulation', 'p53 can promote or suppress ferroptosis through different downstream mechanisms.', 'Includes direct inhibition of erastin ferroptosis through DPP4 in colorectal models.', 'TP53 expression is not a surrogate for functional p53 status; mutation information is a separate modality.', 'p53 vitk')
add('CDH1 NF2 YAP1', 'add16', 'Cell contact and Hippo state', 'E-cadherin/NF2/Hippo signaling connects cell contact to YAP activity and ferroptosis susceptibility.', 'Functional cancer-cell perturbation and density experiments.', 'May encode lineage or culture density; nuclear localization and activity differ from RNA abundance.', 'contact')
add('WWTR1', 'add16', 'Cell contact and Hippo state', 'TAZ promotes an EMP1-NOX4 ferroptosis program in renal cancer.', 'Direct erastin studies in renal cancer models.', 'TAZ is WWTR1, not TAFAZZIN; effects and target genes are tissue dependent.', 'taz')
add('ZEB1', 'add16', 'Mesenchymal lipid state', 'ZEB1 controls lipogenic enzymes and phospholipid composition linked to ferroptosis susceptibility.', 'Cancer-cell perturbation evidence, including experiments with erastin.', 'May act as a lineage/state proxy; published expression-response associations may reuse public cell-line resources.', 'zeb')
add('EPAS1 HILPDA', 'add16', 'Clear-cell and hypoxic lipid state', 'The HIF-2alpha-HILPDA axis contributes to a PUFA-rich, GPX4-dependent clear-cell state.', 'Strong clear-cell cancer/GPX4-inhibition evidence.', 'Especially context dependent; should not be assumed a pan-cancer erastin mechanism.', 'hypoxia')

# Reserve genes are defensible candidates but are not silently added to the
# proposed 80-gene set; preserving them makes the bounded selection explicit.
add('FTL', 'reserve', 'Iron storage', 'Ferritin light chain participates in iron sequestration.', 'BACH1/iron-handling evidence.', 'Closely related to the FTH1 module; not a claim that this gene lacks ferroptosis relevance.', 'iron', 'pathway evidence')
add('PGD', 'reserve', 'NADPH metabolism', '6-phosphogluconate dehydrogenase supplies NADPH through the pentose phosphate pathway.', 'Knockdown suppressed erastin ferroptosis in Calu-1.', 'PGD is not PHGDH. Kept as a reserve member of the G6PD/NADPH module.', 'original')
add('OTUB1', 'reserve', 'SLC7A11 protein stability', 'OTUB1 stabilizes SLC7A11 and can protect against ferroptosis.', 'Direct cystine starvation/ferroptosis perturbation evidence.', 'Protein stability and activity may be more informative than RNA; another candidate for a separately frozen follow-up.', 'otub')

def sha(path): return hashlib.sha256(path.read_bytes()).hexdigest()
def dump(path, obj): path.write_text(json.dumps(obj, indent=2, allow_nan=False)+'\n')

panel = json.loads((SRC/'panel_definition.json').read_text())['primary_panel_symbols']
assert len(panel) == 40
assert len(rows) == len(set(r['symbol'] for r in rows)) == 43
assert all(r['symbol'] not in panel for r in rows)
mapping = pd.read_csv(SRC/'gene_mapping.tsv.gz', sep='\t')
hgnc = pd.read_csv(SRC/'source_snapshots/hgnc.txt.gz', sep='\t', dtype=str)
data = np.load(SRC/'matched_depmap_expression.npz')
core = np.load(SRC/'comparison_inputs.npz')
np.testing.assert_array_equal(data['sample_ids'], core['sample_ids'])
assert len(data['sample_ids']) == 745
x = data['X_log2tpm1']; cols = data['gene_columns']
for row in rows:
    g = row['symbol']; matches = mapping[mapping.approved_symbol==g]
    assert len(matches) == 1 and matches.iloc[0].status == 'mapped'
    m = matches.iloc[0]; idx = int(m.depmap_index_0based); bf = int(m.bulkformer_index_0based)
    approved = hgnc[(hgnc.symbol==g) & (hgnc.status=='Approved')]
    assert len(approved) == 1
    assert approved.iloc[0].ensembl_gene_id == m.ensembl_gene_id
    assert int(approved.iloc[0].entrez_id) == int(m.entrez_id)
    assert cols[idx] == m.depmap_column and cols[idx].split(' (')[0] == g
    assert core['bulkformer_ensembl_ids'][bf] == m.ensembl_gene_id
    assert bool(core['bulkformer_observed_gene_mask'][bf])
    v = x[:, idx]; assert np.isfinite(v).all() and np.std(v) > 0
    row.update(ensembl_gene_id=str(m.ensembl_gene_id), hgnc_id=str(m.hgnc_id), entrez_id=int(m.entrez_id),
               depmap_column=str(m.depmap_column), depmap_index_0based=idx, bulkformer_index_0based=bf,
               finite_samples=int(np.isfinite(v).sum()), fraction_zero=float(np.mean(v==0)),
               mean_log2tpm1=float(np.mean(v)), sd_log2tpm1=float(np.std(v)),
               reference_urls=[REFERENCES[k][2] for k in row['reference_ids']])

first = [r['symbol'] for r in rows if r['tier']=='add24']
second = [r['symbol'] for r in rows if r['tier']=='add16']
assert len(first)==24 and len(second)==16
proposal = dict(status='PROPOSED_NOT_FITTED', date='2026-11-01',
    purpose='Prospective literature-based panel expansion for the existing erastin benchmark.',
    original_40=panel, first_additions_24=first, broader_additions_16=second,
    proposed_64=panel+first, proposed_80=panel+first+second,
    reserve_genes=[r['symbol'] for r in rows if r['tier']=='reserve'],
    selection='Mechanistic coverage and perturbation evidence, not correlations with the current response labels or held-out model importance.',
    tier_meaning='Order of proposed comparisons; not a FerrDb confidence label or a claim that the second module has uniformly weaker evidence.',
    response_analysis_performed=False, models_fitted=False,
    caveat='Prior results are already known and some source papers reuse CTRP/CCLE. Reusing these holdouts remains exploratory; external validation is needed.')
dump(OUT/'panel_definitions.json', proposal)
dump(OUT/'candidate_gene_evidence.json', rows)
frame = pd.DataFrame(rows)
for column in ['reference_ids', 'reference_urls']: frame[column] = frame[column].map(lambda a: ' ; '.join(a))
frame.to_csv(OUT/'candidate_gene_evidence.tsv', sep='\t', index=False, float_format='%.8g')
refs = {k:dict(citation=v[0], title=v[1], url=v[2], type=v[3]) for k,v in REFERENCES.items()}
dump(OUT/'references.json', refs)
dump(OUT/'coverage_summary.json', dict(samples=745, additions_proposed=40, reserves_checked=3,
    approved_unique_mappings=43, measured_in_depmap=43, in_bulkformer_vocabulary_and_observed=43,
    all_finite_and_variable=True, response_labels_used=False,
    mapping_authority='Saved HGNC snapshot from original preparation; verified against saved DepMap and BulkFormer column identities.',
    note='Vocabulary coverage does not change the old 40-token BulkFormer pooled embedding; a new pooling/extraction is needed if embeddings are revisited.'))
sources = [SRC/'panel_definition.json', SRC/'gene_mapping.tsv.gz', SRC/'source_snapshots/hgnc.txt.gz',
           SRC/'matched_depmap_expression.npz', SRC/'comparison_inputs.npz', Path(__file__)]
dump(OUT/'source_manifest.json', dict(files=[dict(path=str(p), sha256=sha(p)) for p in sources]))

lines = ['# Proposed ferroptosis-panel expansion', '',
 'The literature supports testing a broader panel, but does not establish that adding genes will improve erastin prediction. '
 'I propose preserving the original 40-gene baseline, adding 24 genes for a 64-gene comparison, then testing a broader 80-gene panel with 16 additional genes. '
 'These are analyst-selected mechanistic sets, not published predictive signatures. No new models have been fitted.', '',
 'All 40 proposed additions (and three reserve candidates) have unique approved-symbol/Ensembl/Entrez mappings, finite variable expression in all 745 matched cell lines, '
 'and observed entries in the saved BulkFormer vocabulary. These checks use the saved HGNC and DepMap data, not a new data release. '
 'The selection did not use current erastin labels, feature-response correlations, or lineage-specific model wins.', '',
 '## First expansion: 40 to 64 genes', '',
 '| Module | Added genes | Why include it | Evidence |', '|---|---|---|---|']
groups = [
 ('Cystine import and alternate cysteine', 'SLC3A2 CBS CTH', 'Completes system xc- and represents a bypass around cystine uptake.', 'xc transsulfuration'),
 ('Glutaminolysis', 'GLS2 GOT1', 'Captures nutrient-dependent support for cystine-deprivation ferroptosis.', 'glutaminolysis'),
 ('Selenium handling', 'PRDX6 SEPHS2', 'Represents control of GPX4 synthesis rather than GPX4 transcript alone.', 'prdx6'),
 ('BH4 defense', 'GCH1 PTS SPR', 'Adds a parallel antioxidant system; DHFR is already included.', 'bh4 bh4_screen'),
 ('Iron regulation/storage/export', 'IREB2 FTH1 SLC40A1 PROM2', 'Complements TFRC and NCOA4 with labile-iron control and export.', 'original iron prom2'),
 ('PUFA biosynthesis', 'FADS1 FADS2 ELOVL5', 'Adds substrate synthesis upstream of ACSL4/LPCAT3.', 'pufa fads'),
 ('Ether-lipid synthesis/remodeling', 'FAR1 GNPAT AGPS TMEM164', 'Adds an incompletely represented lipid substrate pathway.', 'ether tmem'),
 ('System xc-/glutathione regulation', 'BECN1 BAP1 ABCC1', 'Adds transporter regulation and glutathione export with functional evidence.', 'becn bap abcc')]
for module, genes, why, keys in groups:
    links = ', '.join(f'[{REFERENCES[k][0]}]({REFERENCES[k][2]})' for k in keys.split())
    lines.append(f'| {module} | {genes.replace(" ", ", ")} | {why} | {links} |')
lines += ['', '## Broader expansion: 64 to 80 genes', '',
 ', '.join(second)+'.', '',
 'This second module includes additional antioxidant chemistry, mitochondrial redox defenses, and cell-state regulators. '
 'Its tier reflects the proposed comparison order, not uniformly weaker biological evidence.', '',
 f'- SC5D and DHCR7 control the protective sterol 7-DHC and can have opposing functional effects. [2024 primary study]({REFERENCES["sterol"][2]}).',
 f'- VKORC1L1 supplies reduced vitamin K as a parallel defense. It is a different gene from VKORC1. [2023 primary study]({REFERENCES["vitk"][2]}).',
 f'- G6PD supplies NADPH, but the original erastin study found its silencing protective in Calu-1, illustrating why a universal expression sign is inappropriate. [Dixon et al.]({REFERENCES["original"][2]}).',
 f'- DHODH and GPD2 mainly have mitochondrial/GPX4-inhibitor evidence. DHODH inhibitor specificity and contribution are disputed. [Original DHODH study]({REFERENCES["dhodh"][2]}), [Matters Arising]({REFERENCES["dhodh_dispute"][2]}), [GPD2 study]({REFERENCES["gpd2"][2]}).',
 f'- HMOX1 can promote or suppress erastin ferroptosis depending on the model. [Promoting setting]({REFERENCES["hmox_pro"][2]}), [protective setting]({REFERENCES["hmox_anti"][2]}).',
 f'- KEAP1 and TP53 are relevant but their RNA levels do not establish functional genotype or pathway activity. [KEAP1/NRF2 study]({REFERENCES["nrf2"][2]}), [p53/DPP4 study]({REFERENCES["p53"][2]}).',
 f'- CDH1, NF2, YAP1, WWTR1 and ZEB1 capture contact, Hippo signaling and mesenchymal lipid state. They may also carry lineage and culture-density information. [NF2/YAP]({REFERENCES["contact"][2]}), [TAZ]({REFERENCES["taz"][2]}), [ZEB1]({REFERENCES["zeb"][2]}).',
 f'- EPAS1 and HILPDA have strong clear-cell/GPX4-dependency evidence; pan-cancer erastin relevance is less certain. [Clear-cell study]({REFERENCES["hypoxia"][2]}).', '',
 'FTL, PGD and OTUB1 remain documented reserve candidates. Exclusion from this bounded proposal does not mean absence of ferroptosis relevance.', '',
 '## What recent literature changes', '',
 f'PRDX6 is especially useful to add: the 2024 study identifies selenium delivery as a route to maintaining GPX4, with IKE and RSL3 perturbation evidence. [Fujita et al.]({REFERENCES["prdx6"][2]}). '
 f'A March 2026 preprint reports different NALM6 genetic sensitizers for erastin2 and RSL3: SLC3A2/PRDX6 feature in the erastin2 results, while BH4-pathway genes feature in RSL3 results. '
 f'This supports annotating inducer-specific evidence, but the preprint is not treated as peer-reviewed confirmation or pan-cancer validation. [Murray et al., preprint]({REFERENCES["screen2026"][2]}).', '',
 f'FerrDb V3 now spans 22 regulated-cell-death modalities and distinguishes drivers, suppressors, markers and unclassified genes, with different evidence-confidence categories. '
 f'A larger FerrDb-derived set should explicitly filter to ferroptosis and inspect validated human perturbation evidence. '
 f'The resource was checked at the documentation level; this proposal is not a full FerrDb export or a gene-by-gene reproduction of its labels. [FerrDb documentation]({REFERENCES["ferrdb"][2]}).', '',
 '## Proposed modeling comparison', '',
 '1. Freeze the 40-, 64- and 80-gene sets before fitting. Keep the existing 745 samples, targets and 20 saved leave-lineage-out repeats.',
 '2. Use the same ridge, neural-network and XGBoost pipelines, with all tuning confined to the existing inner folds. Compare added information before changing model families or search grids.',
 '3. Report paired pooled and equal-lineage metrics, lineage win counts and conditional whole-lineage bootstrap intervals. Include separate ablations of the extra antioxidant and cell-state modules if predeclared.',
 '4. Fit any data-driven variance filtering, feature selection, transformations or pathway aggregation inside training folds. Do not choose genes from their full-cohort erastin correlations.',
 '5. Treat this as exploratory reuse of the current cohort. Several source papers also analyze CTRP/CCLE; mechanistic perturbations, rather than published response correlations, motivate these additions. External screen validation is still needed.', '',
 'Do not equate gene perturbation effects with the direction of a baseline RNA regression coefficient. '
 'Ferroptosis defense depends on metabolites, protein activity, mutations, culture conditions and lipid availability as well as RNA. '
 'Post-treatment markers such as PTGS2 and CHAC1 are not automatically valid baseline predictors; their induction alone is insufficient evidence for inclusion. '
 'If BulkFormer is revisited, the old pooled 40-gene embedding does not become a larger-panel embedding merely because the extra genes are in its vocabulary.', '',
 '## Scope and files', '',
 'This was a targeted literature check through 2026-11-01 using the supplied 2024 review, primary-paper abstracts/results and current resource documentation. '
 'It was not an exhaustive systematic review; complete full-text access varied. Gene-specific evidence, stimulus and caveats are retained below.', '',
 f'- [Gene evidence and coverage table]({OUT}/candidate_gene_evidence.tsv)',
 f'- [Machine-readable panel definitions]({OUT}/panel_definitions.json)',
 f'- [Coverage audit]({OUT}/coverage_summary.json)',
 f'- [Reference list]({OUT}/references.json)', '']
(OUT/'REVIEW.md').write_text('\n'.join(lines))
dump(OUT/'artifact_manifest.json', dict(files=[dict(path=p.name, sha256=sha(p), bytes=p.stat().st_size)
     for p in sorted(OUT.iterdir()) if p.is_file() and p.name!='artifact_manifest.json']))
print(json.dumps(dict(proposed_panel_sizes=[40, 64, 80], new_genes_mapped=40, reserve_genes_mapped=3,
                     all_745_finite_and_variable=True, models_fitted=False)))
