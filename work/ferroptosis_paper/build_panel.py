"""Build a paper-derived candidate panel; no response data are consulted."""
import hashlib
import json
from pathlib import Path

ROOT = Path('/home/sji/Documents/Codex/2026-10-31/t')
PDF = Path('/home/sji/Downloads/The_cell_biology_of_ferroptosis_2024.pdf')
OUT = ROOT / 'outputs'
records = []

def add(symbol, paper_name, module, pages, refs, role, relevance, caveat, evidence='mechanism_or_function_reported'):
    records.append(dict(
        symbol=symbol, paper_name=paper_name, module=module,
        journal_pages=pages, pdf_pages=[p - 423 for p in pages],
        review_reference_numbers=refs,
        mechanism_as_described_in_review=role,
        erastin_relevance=relevance,
        interpretation_caveat=caveat,
        evidence_basis=evidence,
        expression_effect_direction='Not prespecified; estimate within training data',
        ensembl_gene_id=None,
    ))

add('SLC7A11','SLC7A11 / transport subunit of system xc-','cystine_gsh', [426,430,433],[58],
    'Imports cystine through system xc-, supplying cysteine for glutathione and other protective sulfur metabolites.',
    'Directly proximal to erastin mechanism: erastin inhibits system xc-.',
    'Target expression does not by itself determine inhibitor sensitivity; transporter activity and alternative cysteine supplies matter.')
add('GCLC','GCLC / catalytic subunit of GCL','cystine_gsh',[427],[68,73],
    'Catalytic subunit of glutamate cysteine ligase; supports glutathione synthesis and alternative glutamate-sequestering peptides during cysteine shortage.',
    'Cysteine-deprivation response downstream of system xc- inhibition.',
    'Review explicitly states that GCLC disruption and loss of de novo glutathione do not invariably cause ferroptosis.')
add('GCLM','GCLM / modifier subunit of GCL','cystine_gsh',[427],[68,73],
    'Modifier subunit of glutamate cysteine ligase in the glutathione biosynthesis pathway.',
    'Pathway component downstream of cystine supply.',
    'The local discussion establishes pathway membership, not a universal independent GCLM-expression effect.', 'explicit_pathway_component')
add('GSS','GSH synthetase (GSS)','cystine_gsh',[427],[],
    'Catalyzes the second step of glutathione synthesis.',
    'Supports the glutathione pool affected by cysteine depletion.',
    'Glutathione abundance alone is insufficient to explain all cystine-deprivation phenotypes.', 'explicit_pathway_component')
add('GSR','Glutathione-disulfide reductase / GSR / GR','cystine_gsh',[427,428],[74,75],
    'Uses NADPH to regenerate reduced glutathione from oxidized glutathione.',
    'Maintains the glutathione-dependent lipid peroxide defense.',
    'Recycling capacity depends on NADPH and substrate availability.', 'explicit_pathway_component')

add('GGT1','Gamma-glutamyltransferase 1 (GGT1)','alternative_cysteine',[426],[60,61,62],
    'Cleaves extracellular glutathione, enabling cysteine salvage that can bypass dependence on imported free cystine.',
    'Particularly relevant to resistance to system xc- inhibition.',
    'Effect depends on extracellular glutathione and tissue context; review highlights brain and chromophobe renal cancers.')
add('CNDP2','Carnosine dipeptidase 2 (CNDP2)','alternative_cysteine',[426],[60],
    'Catabolizes imported cysteinylglycine to release cysteine for glutathione synthesis.',
    'Alternative cysteine acquisition downstream of extracellular glutathione cleavage.',
    'Requires extracellular substrate and uptake; the review discusses the pathway rather than a universal predictive marker.', 'explicit_pathway_component')
add('CTSD','Cathepsin D','alternative_cysteine',[435],[59],
    'Lysosomal catabolism of cysteine-rich extracellular proteins supplies cysteine equivalents and supports glutathione synthesis.',
    'Potential bypass when extracellular free cystine is limiting.',
    'Requires protein uptake and appropriate medium composition; broad lysosomal expression is not a ferroptosis-specific readout.')
add('CTNS','Cystinosin','alternative_cysteine',[435],[59,158],
    'Exports lysosomal cystine to the cytosol after protein catabolism.',
    'Alternative cystine supply and lysosomal cystine stress response.',
    'Lysosomal cystine depletion can also trigger protective ATF4 signaling; net effects may be nonmonotonic.')

add('GPX4','Glutathione peroxidase 4 (GPX4)','gpx4_selenium',[425,426,430],[34,35,36,63],
    'Reduces lipid hydroperoxides using glutathione and protects against ferroptosis.',
    'Convergence point of the glutathione defense compromised by erastin.',
    'RNA is an incomplete proxy for selenoprotein translation, isoform location, protein turnover, and enzymatic activity.')
add('LRP8','LDL receptor-related protein 8 (LRP8)','gpx4_selenium',[426],[63,64,65],
    'Imports selenium-bearing selenoprotein P, supporting GPX4 synthesis; loss can sensitize cells or induce ferroptosis.',
    'Modulates GPX4 defense capacity in parallel with cysteine availability.',
    'Dependence varies with selenium sources and cell lineage.')

add('AIFM2','FSP1 (also known as AIFM2)','parallel_antioxidants',[425,426,429],[41,42,43],
    'Regenerates reduced CoQ and vitamin K at the plasma membrane to terminate lipid radical propagation.',
    'Parallel defense that may compensate for weakening of the glutathione-GPX4 system.',
    'Membrane targeting and protein modification are required; gene expression alone does not establish activity.')
add('STARD7','StAR-related lipid transfer protein 7 (STARD7)','parallel_antioxidants',[428,431],[86],
    'A processed form transports mitochondrial CoQ to the plasma membrane for FSP1-mediated protection.',
    'Supports the parallel CoQ-FSP1 defense.',
    'Function depends on proteolytic processing, localization, and CoQ supply.')
add('DHFR','Dihydrofolate reductase (DHFR)','parallel_antioxidants',[427],[23,45],
    'Regenerates reduced tetrahydrobiopterin from BH2 using NADPH.',
    'Supports an endogenous radical-trapping antioxidant pathway.',
    'This is general ferroptosis relevance; erastin-specific predictive value is unestablished.', 'explicit_pathway_component')

add('TFRC','Transferrin receptor (TFRC)','iron_handling',[435,438],[52,53,191],
    'Mediates uptake of transferrin-bound iron; silencing reduces ferroptosis.',
    'Supplies redox-active iron needed for lipid peroxidation.',
    'RNA does not report the labile iron pool or stimulus-dependent receptor trafficking.')
add('NCOA4','Nuclear receptor coactivator 4 (NCOA4)','iron_handling',[430,435],[151,152,153,219],
    'Drives ferritinophagy and release of iron from ferritin; disruption reduces ferroptosis sensitivity.',
    'Review notes ferritin catabolism may matter more for system xc- inhibition than direct GPX4 inhibition.',
    'Functional effect depends on ferritin cargo, lysosomal activity, and cellular iron status.')
add('NFS1','NFS1 cysteine desulfurase','iron_handling',[429],[92],
    'Supports iron-sulfur cluster synthesis; depletion triggers iron-starvation feedback, increased iron uptake, and sensitization.',
    'Cysteine-derived sulfur links this pathway to cystine deprivation.',
    'The review presents selective perturbation of iron-sulfur synthesis by cystine deprivation as a mechanistic model, not a settled universal mechanism.')

add('ACSL4','Acyl-CoA synthetase long-chain family member 4 (ACSL4)','lipid_substrates',[430,431,437],[24,27,30,99,105,106,185],
    'Activates polyunsaturated fatty acids for incorporation into oxidizable lipids and generally increases sensitivity.',
    'Determines availability of substrates for ferroptotic lipid peroxidation.',
    'Localization, phosphorylation, substrate availability, and lipid flux can change effects.')
add('LPCAT3','LPCAT3 / MBOAT5','lipid_substrates',[430,433],[25,105,115],
    'Incorporates polyunsaturated acyl chains into phospholipids; inhibition can suppress ferroptosis.',
    'Sets membrane susceptibility to peroxidation.',
    'Effect depends on competing acyltransferases and available fatty acids.')
add('AGPAT3','1-acylglycerol-3-phosphate O-acyltransferase 3 (AGPAT3)','lipid_substrates',[433],[25,105,115],
    'Participates in acylation of phospholipids with polyunsaturated fatty acids.',
    'Supports oxidizable phospholipid synthesis.',
    'Pathway role is established in the review; a fixed RNA-response direction is not.', 'explicit_pathway_component')
add('MBOAT7','MBOAT7','lipid_substrates',[433],[24],
    'Acylates lysophospholipids with polyunsaturated fatty acids in the lipid remodeling pathway.',
    'May influence the pool of peroxidizable membrane lipids.',
    'Contribution depends on lipid class and cellular context.')
add('ACSL3','ACSL3','lipid_substrates',[431],[28,99,107],
    'Monounsaturated fatty-acid metabolism generally supports ferroptosis resistance.',
    'Controls competition between less and more oxidizable lipid species.',
    'Review explicitly describes a pro-ferroptotic exception in KRAS-mutant lung cancer.')
add('SCD','SCD1 / stearoyl-CoA desaturase 1','lipid_substrates',[433,436],[125,179,180,181],
    'Produces monounsaturated fatty acids that can reduce membrane susceptibility to peroxidation.',
    'Potential protective lipid state across cancer cell lines.',
    'Human gene symbol is SCD; effects depend on substrates and signaling context.')
add('MBOAT1','MBOAT1','lipid_substrates',[433,438],[116,117],
    'Acylates lysophospholipids with monounsaturated fatty acids and can promote resistance.',
    'Controls a protective membrane lipid program.',
    'Review highlights estrogen-receptor-positive breast cancer; assess lineage dependence.')
add('MBOAT2','MBOAT2','lipid_substrates',[433,438],[116,117],
    'Acylates lysophospholipids with monounsaturated fatty acids and can promote resistance.',
    'Controls a protective membrane lipid program.',
    'Review highlights androgen-receptor-positive prostate cancer; assess lineage dependence.')
add('PLA2G6','PLA2G6 / PNPLA9 / iPLA2beta','lipid_substrates',[432,433],[112,113,114],
    'Removes oxidized polyunsaturated acyl chains from phospholipids, limiting lipid peroxide propagation.',
    'Repairs substrates damaged during ferroptosis.',
    'Activity and localization are not captured completely by basal RNA levels.')
add('ABHD12','ABhydrolase domain-containing protein 12 (ABHD12)','lipid_substrates',[433],[],
    'Cleaves polyunsaturated lysophosphatidylserines and oxidized phosphatidylserines; inhibition increases sensitivity.',
    'Controls oxidizable and oxidized phosphatidylserine pools.',
    'The local paragraph does not attach a separate numbered reference to this claim; the review is the evidence source.')

add('POR','Cytochrome P450 oxidoreductase (POR)','redox_supply',[427,429,432],[76,77],
    'ER oxidoreductase that promotes reactive oxygen production and lipid peroxidation.',
    'Provides oxidative pressure that can promote ferroptotic death.',
    'NADPH can support both oxidant production and antioxidant regeneration.')
add('CYB5R1','Cytochrome b5 reductase 1 (CYB5R1)','redox_supply',[427,429,432],[76,77],
    'ER oxidoreductase implicated in reactive oxygen production and membrane lipid peroxidation.',
    'Provides oxidative pressure during ferroptosis.',
    'Net effect depends on electron supply and other redox defenses.')
add('NADK','NAD kinase (NADK)','redox_supply',[427],[74,78],
    'Generates the cytosolic NADP pool supporting NADPH-dependent reactions; manipulation links cytosolic NADPH to sensitivity.',
    'Supports redox capacity under cysteine stress.',
    'NADPH is used by both antioxidant and pro-oxidant reactions; do not impose a universal coefficient sign.')
add('HDDC3','HDDC3 / MESH1','redox_supply',[427],[78],
    'Proposed cytosolic NADPH phosphatase that increases sensitivity by reducing NADPH.',
    'Potential regulator of antioxidant reserve.',
    'The review describes the phosphatase mechanism as proposed.')

add('NFE2L2','NRF2 / NFE2L2','stress_and_gpx4_turnover',[435,438],[57,130,154,165,166,167,168],
    'Coordinates glutathione, FSP1, and iron-handling programs that usually protect against ferroptosis.',
    'Integrates several defenses relevant to erastin.',
    'Review explicitly notes opposite effects through ABCC1-mediated glutathione export; RNA abundance does not establish transcription-factor activation.')
add('NFE2L1','NFE2L1','stress_and_gpx4_turnover',[432,433],[126,127,128,129,130,131],
    'ER-processed transcription factor whose NGLY1-dependent pathway supports GPX4 stability and protection.',
    'Controls GPX4 defense capacity.',
    'Protein processing and proteasome regulation matter; NFE2L1 and NRF2/NFE2L2 are distinct.')
add('NGLY1','N-glycanase 1 (NGLY1)','stress_and_gpx4_turnover',[432,433],[127,130,131],
    'Deglycosylates NFE2L1 and supports the NGLY1-NFE2L1 ferroptosis-protection pathway.',
    'Potential determinant of GPX4 protein maintenance.',
    'The review notes that the mechanism connecting this pathway to GPX4 stability is indirect and incompletely resolved.')
add('SREBF1','SREBP1','stress_and_gpx4_turnover',[432,433,438],[125,180,194],
    'Regulates lipid metabolic programs, including SCD-mediated monounsaturated fatty-acid production that can protect cells.',
    'Integrates protective lipid composition with metabolic signaling.',
    'Cleavage and nuclear localization determine activity; the review also discusses potentially competing metabolic effects.')
add('ATF4','Activating transcription factor 4 (ATF4)','stress_and_gpx4_turnover',[427,431,435],[84,157,158],
    'Integrated stress-response factor that can support glutathione metabolism, GPX4 stability, and protection during cysteine stress.',
    'Particularly relevant to cysteine and lysosomal cystine deprivation.',
    'Baseline RNA does not capture stress-induced translation or downstream activation; induction is not always equivalent to successful protection.')
add('ATF3','ATF3','stress_and_gpx4_turnover',[435],[171],
    'Named in the review as a ferroptosis regulator; cited primary study shows repression of SLC7A11 and promotion of erastin-induced ferroptosis.',
    'Direct erastin and system xc- relevance supported by the review-cited primary paper.',
    'A stress-responsive transcription factor; basal expression across cell lines is not equivalent to experimental activation.',
    'review_named_regulator_with_targeted_primary_source_check')
add('HSPA8','Heat shock 70 kDa protein 8 (HSPA8)','stress_and_gpx4_turnover',[435],[155,156],
    'Participates in chaperone-mediated autophagic degradation of GPX4.',
    'System xc- inhibitors can promote this GPX4-degradation pathway.',
    'A broad chaperone with many functions; protein interactions and autophagic flux matter.')
add('LAMP2','Lysosome-associated membrane glycoprotein 2 (LAMP2)','stress_and_gpx4_turnover',[435],[155,156],
    'Participates in chaperone-mediated autophagic degradation of GPX4.',
    'System xc- inhibitors can increase LAMP2 and GPX4 degradation.',
    'Gene-level RNA does not directly measure chaperone-mediated autophagy activity or GPX4 degradation.')
add('CKB','Creatine kinase B (CKB)','stress_and_gpx4_turnover',[435],[156],
    'Phosphorylates GPX4 at Ser104 and limits its interaction with HSPA8/LAMP2, reducing degradation.',
    'Counteracts loss of the GPX4 defense through chaperone-mediated autophagy.',
    'Protein kinase activity and phosphorylation state are not determined by CKB RNA alone.')

contextual = [
    ('TP53','p53',[435,438],'Both pro- and anti-ferroptotic effects are explicitly described; variant and stimulus matter.'),
    ('KEAP1','KEAP1',[437],'Regulates NRF2 stability; discussed in the ACOD1-itaconate pathway in tumor-infiltrating neutrophils. Mutation/protein function is poorly represented by RNA alone.'),
    ('FADS1','FADS1',[430,436],'Desaturation changes PUFA/MUFA composition; context and substrates determine the phenotype.'),
    ('FADS2','FADS2',[430,436],'Desaturation changes PUFA/MUFA composition; context and substrates determine the phenotype.'),
    ('FAR1','FAR1',[434],'Peroxisomal ether lipid synthesis affects ferroptosis in selected models.'),
    ('GNPAT','GNPAT',[434],'Peroxisomal ether lipid synthesis affects ferroptosis in selected models.'),
    ('AGPS','AGPS',[434],'Review explicitly contrasts sensitizing, neutral, and opposite effects across cells/species; human cancer context must be established.'),
    ('EPAS1','HIF2alpha / EPAS1',[433,435],'HIF2alpha-HILPDA signaling is linked to sensitivity in clear-cell renal carcinoma.'),
    ('HILPDA','HILPDA',[433,435],'In renal cancer, increased lipid droplets and PUFA-TAGs accompany increased sensitivity; not a universal lipid-droplet protection rule.'),
    ('PNPLA2','ATGL / PNPLA2',[433,434],'Controls lipid-droplet lipolysis; the effect of PUFA storage and release is context dependent.'),
    ('DGAT1','DGAT1',[433,434],'Lipid-droplet synthesis can protect, promote, or have little effect on ferroptosis depending on the setting.'),
    ('DGAT2','DGAT2',[433,434],'Lipid-droplet synthesis can protect, promote, or have little effect on ferroptosis depending on the setting.'),
    ('DECR1','DECR1',[428,429,438],'PUFA breakdown can protect against ferroptosis; review emphasizes prostate-cancer context.'),
    ('DHODH','DHODH',[429,431],'Mitochondrial CoQ defense is explicitly presented as conditional and controversial; avoid equating GPX4-inhibitor evidence with erastin response.'),
    ('GPD2','GPD2',[429],'Mitochondrial CoQ reduction and ETC dependence; context-specific contribution.'),
    ('SLC25A37','MFRN1 / SLC25A37',[429,431],'Mitochondrial iron import can increase sensitivity, but the review states that the precise mechanism is unclear.'),
    ('MARCHF6','MARCHF6',[427],'NADPH-sensing E3 ligase affects ACSL4 and p53 levels; net effects and developmental evidence are context dependent.'),
    ('MS4A15','MS4A15',[430],'ER calcium regulation affects lipid desaturation and sensitivity.'),
    ('UBIAD1','UBIAD1',[434,435],'Vitamin K metabolism is relevant; direct Golgi CoQ synthesis and its relative contribution remain unresolved.'),
    ('ABCC1','MRP1 / ABCC1',[438],'Glutathione export can make NRF2 activation pro-ferroptotic in some cells.'),
    ('YAP1','YAP1',[435],'Named transcriptional regulator; cell interaction and context require separate interpretation.'),
    ('WWTR1','TAZ',[435],'Named Hippo-pathway transcriptional regulator; TAZ here maps to WWTR1, not tafazzin.'),
    ('ZEB1','ZEB1',[435],'Named regulator associated with a cell-state program; prioritize within-lineage evaluation.'),
    ('VKORC1','VKORC1',[435],'Vitamin K recycling is relevant; keep this gene distinct from VKORC1L1, and establish the specific contribution of each.'),
    ('BAP1','BAP1',[435],'Named epigenetic regulator; loss-of-function status may matter more than transcript abundance.'),
]

aliases = {
    'FSP1':'AIFM2', 'NRF2':'NFE2L2', 'SCD1':'SCD', 'SREBP1':'SREBF1',
    'MESH1':'HDDC3', 'MBOAT5':'LPCAT3', 'PNPLA9':'PLA2G6',
    'iPLA2beta':'PLA2G6', 'cathepsin D':'CTSD', 'cystinosin':'CTNS',
    'TAZ':'WWTR1', 'HIF2alpha':'EPAS1', 'ATGL':'PNPLA2',
    'MFRN1':'SLC25A37', 'MRP1':'ABCC1', 'p53':'TP53',
}

priority = ['SLC7A11','GCLC','GCLM','GSS','GSR','GGT1','CNDP2','CTSD','CTNS','GPX4','NCOA4','NFS1','ATF3','ATF4','HSPA8','LAMP2','CKB']
symbols = [r['symbol'] for r in records]
assert len(symbols) == len(set(symbols)) == 40
assert set(priority) <= set(symbols)
assert not (set(symbols) & {x[0] for x in contextual})

payload = {
    'name':'Dixon-Olzmann 2024 paper-derived erastin candidate panel',
    'version':'0.1-proposed', 'curation_date':'2026-10-31',
    'status':'Literature-derived proposal; not a validated predictive signature or an exhaustive gene catalog.',
    'source':{
        'title':'The cell biology of ferroptosis',
        'authors':['Scott J. Dixon','James A. Olzmann'],
        'doi':'10.1038/s41580-024-00703-5',
        'citation':'Nature Reviews Molecular Cell Biology 25, 424-442 (2024)',
        'path':str(PDF), 'sha256':hashlib.sha256(PDF.read_bytes()).hexdigest(),
        'reference_number_convention':'Numbers refer to the supplied review bibliography. The underlying primary literature has not been systematically re-audited; targeted checks are listed separately.',
    },
    'selection_rule':[
        'Gene or unambiguous protein name appears in the review main text or figure captions.',
        'The review describes a ferroptosis mechanism, a functional perturbation, or an explicit component of a mechanistically linked pathway.',
        'Primary-panel prioritization emphasizes cystine supply, lipid peroxide defense, lipid substrates, iron, and GPX4 maintenance relevant to the erastin question.',
        'This ranking is an analyst proposal, not a panel published or validated by the review authors.',
        'No DepMap expression-response associations or erastin labels were used to select this panel.',
    ],
    'primary_panel_count':40, 'primary_panel_symbols':symbols,
    'erastin_proximal_priority_symbols':priority,
    'alias_map':aliases,
    'alias_authority_sources':[
        'https://www.uniprot.org/uniprotkb/Q9BRQ8/entry',
        'https://www.uniprot.org/entry/O00767',
        'https://www.ncbi.nlm.nih.gov/gene/1509',
        'https://ncbi.nlm.nih.gov/gene?from_uid=219800&linkname=omim_gene',
        'https://www.ncbi.nlm.nih.gov/gene/6720',
        'https://www.ncbi.nlm.nih.gov/gene/25937',
    ],
    'targeted_primary_source_checks':[
        {'symbol':'ATF3','review_reference':171,'url':'https://pmc.ncbi.nlm.nih.gov/articles/PMC7206049/','finding':'ATF3 suppresses SLC7A11/system xc- and promotes erastin-induced ferroptosis in tested cell models.'},
        {'symbol':'NCOA4','review_reference':153,'url':'https://iris.unibs.it/bitstream/11379/542015/1/NCOA4.pdf','finding':'Erastin-versus-RSL3 distinction was demonstrated in HeLa cells; do not assume it is universal.'},
    ],
    'genes':records,
    'secondary_contextual_candidates':[
        dict(symbol=s,paper_name=n,journal_pages=p,pdf_pages=[v-423 for v in p],reason_for_separate_analysis=c)
        for s,n,p,c in contextual
    ],
    'exclusions_and_scope':[
        'Do not expand generic protein-family mentions (for example NOX, COQ, TRP, or AMPK) into arbitrary paralog sets.',
        'Do not automatically add genes mentioned only in bibliography titles.',
        'Common ferroptosis genes not individually named in the review body/captions, such as GCH1 and SLC3A2, are not silently added.',
        'NINJ1 and membrane-execution/repair components remain outside this compact primary panel; the review describes context-dependent or shared cell-death roles.',
        'Other explicitly mentioned genes can be curated in a separately versioned expansion; absence from this compact panel is not a claim of biological irrelevance.',
    ],
    'model_comparisons':[
        {'name':'panel_expression','input':'Baseline expression of the 40 genes only','head':'Ridge or elastic net with nested tuning','interpretation':'Strict gene-restricted prediction.'},
        {'name':'panel_contextual_embedding','input':'Complete available coding transcriptome into frozen BulkFormer; pool contextual embeddings at observed primary-panel positions only','head':'Ridge with nested tuning','interpretation':'Readout focuses on the panel, but representations still depend on non-panel genes.'},
        {'name':'all_gene_contextual_embedding','input':'Complete available coding transcriptome; checkpoint-standard sample pooling','head':'Same regression head and validation splits','interpretation':'Comparator testing whether panel-focused pooling helps.'},
    ],
    'modeling_constraints':[
        'Do not zero or mask all non-panel genes to force a gene-only BulkFormer input.',
        'Use all-panel mean pooling as the first fixed pooling rule; select module pooling or feature changes only within training folds.',
        'Keep the literature-selected set fixed across folds; estimate scalers, filters, PCA, hyperparameters and any data-driven selection on training data only.',
        'Apply no fixed coefficient signs from mechanistic annotations.',
        'Evaluate pooled and within-lineage predictions; group related cancer models together.',
        'Do not equate predictive feature importance with causal regulation.',
    ],
    'mapping_status':'Symbol curation complete. Ensembl mapping, actual DepMap coverage, checkpoint-vocabulary coverage and prediction performance remain unverified.',
}

OUT.mkdir(exist_ok=True)
(OUT/'ferroptosis_paper_gene_panel.json').write_text(json.dumps(payload,indent=2,ensure_ascii=False)+'\n')
(OUT/'ferroptosis_primary_40_genes.txt').write_text('\n'.join(symbols)+'\n')
print(json.dumps({'primary_genes':len(symbols),'secondary_candidates':len(contextual),'modules':sorted({r['module'] for r in records}),'outputs':[str(OUT/'ferroptosis_paper_gene_panel.json'),str(OUT/'ferroptosis_primary_40_genes.txt')]},indent=2))
