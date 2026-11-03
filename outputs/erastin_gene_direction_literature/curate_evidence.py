#!/usr/bin/env python3
"""Explicit, reviewable literature annotations; no model signs used to set roles."""
import json
from pathlib import Path

OUT=Path(__file__).resolve().parent
WS=OUT.parent.parent

# Reference URLs were checked in the current targeted literature review.
REFERENCES={
 'review':('Dixon and Olzmann, The cell biology of ferroptosis (2024)','https://doi.org/10.1038/s41580-024-00703-5'),
 'xc':('Pharmacological inhibition of cystine-glutamate exchange induces endoplasmic reticulum stress and ferroptosis','https://pmc.ncbi.nlm.nih.gov/articles/PMC4054777/'),
 'gclc':('Non-canonical glutamate-cysteine ligase activity protects against ferroptosis','https://pmc.ncbi.nlm.nih.gov/articles/PMC7839835/'),
 'cndp2':('Carnosine dipeptidase II protects cells under cysteine insufficiency by hydrolyzing glutathione-related peptides','https://pubmed.ncbi.nlm.nih.gov/34324979/'),
 'ggt1':('Expression of gamma-glutamyltransferase 1 in glioblastoma cells confers resistance to cystine deprivation-induced ferroptosis','https://pmc.ncbi.nlm.nih.gov/articles/PMC8897698/'),
 'ctsb':('Ferroptosis inhibition by lysosome-dependent catabolism of extracellular protein','https://pmc.ncbi.nlm.nih.gov/articles/PMC9762237/'),
 'ctns':('Lysosomal cystine governs ferroptosis sensitivity in cancer via cysteine stress response','https://pmc.ncbi.nlm.nih.gov/articles/PMC10529971/'),
 'ctsd2025':('N-glycosylation Modification of CTSD Affects Liver Metastases in Colorectal Cancer','https://pmc.ncbi.nlm.nih.gov/articles/PMC11831497/'),
 'gpx4':('Regulation of ferroptotic cancer cell death by GPX4','https://pmc.ncbi.nlm.nih.gov/articles/PMC4076414/'),
 'lrp8':('Ribosome stalling during selenoprotein translation exposes a ferroptosis vulnerability','https://pmc.ncbi.nlm.nih.gov/articles/PMC9469796/'),
 'fsp1':('FSP1 is a glutathione-independent ferroptosis suppressor','https://www.nature.com/articles/s41586-019-1707-0'),
 'stard7':('Mitochondria regulate intracellular coenzyme Q transport and ferroptotic resistance via STARD7','https://pmc.ncbi.nlm.nih.gov/articles/PMC9928583/'),
 'bh4':('Metabolic determinants of cancer cell sensitivity to canonical ferroptosis inducers','https://pmc.ncbi.nlm.nih.gov/articles/PMC8299533/'),
 'bh4_crc':('Blockade of GCH1/BH4 axis activates ferritinophagy to mitigate colorectal cancer resistance to erastin-induced ferroptosis','https://pmc.ncbi.nlm.nih.gov/articles/PMC8866854/'),
 'glutam':('Glutaminolysis and transferrin regulate ferroptosis','https://pmc.ncbi.nlm.nih.gov/articles/PMC4506736/'),
 'got1':('GOT1 inhibition promotes pancreatic cancer cell death by ferroptosis','https://pmc.ncbi.nlm.nih.gov/articles/PMC8357841/'),
 'ncoa4':('NCOA4-mediated ferritinophagy promotes ferroptosis induced by erastin, but not by RSL3 in HeLa cells','https://pubmed.ncbi.nlm.nih.gov/33245979/'),
 'nfs1':('NFS1 undergoes positive selection in lung tumours and protects cells from ferroptosis','https://pmc.ncbi.nlm.nih.gov/articles/PMC5808442/'),
 'acsl4':('ACSL4 dictates ferroptosis sensitivity by shaping cellular lipid composition','https://pmc.ncbi.nlm.nih.gov/articles/PMC5610546/'),
 'mboat7':('MMD collaborates with ACSL4 and MBOAT7 to promote polyunsaturated phosphatidylinositol remodeling and susceptibility to ferroptosis','https://pmc.ncbi.nlm.nih.gov/articles/PMC10591818/'),
 'ether':('Plasticity of ether lipids promotes ferroptosis susceptibility and evasion','https://pmc.ncbi.nlm.nih.gov/articles/PMC8051864/'),
 'ether_opposite':('Dietary lipids induce ferroptosis in Caenorhabditis elegans and human cancer cells','https://pmc.ncbi.nlm.nih.gov/articles/PMC7483868/'),
 'kras':('Targeting de novo lipogenesis and the Lands cycle induces ferroptosis in KRAS-mutant lung cancer','https://www.nature.com/articles/s41467-022-31963-4'),
 'mboat12':('Ferroptosis surveillance independent of GPX4 and differentially regulated by sex hormones','https://pmc.ncbi.nlm.nih.gov/articles/PMC10330611/'),
 'srebf':('Oncogenic activation of PI3K-AKT-mTOR signaling suppresses ferroptosis via SREBP-mediated lipogenesis','https://pmc.ncbi.nlm.nih.gov/articles/PMC7733797/'),
 'pla2g6':('Phospholipase iPLA2beta averts ferroptosis by eliminating a redox lipid death signal','https://pmc.ncbi.nlm.nih.gov/articles/PMC8152680/'),
 'abhd12':('Blockade of the lysophosphatidylserine lipase ABHD12 potentiates ferroptosis in cancer cells','https://pmc.ncbi.nlm.nih.gov/articles/PMC7386317/'),
 'por':('Membrane damage during ferroptosis is caused by oxidation of phospholipids catalyzed by the oxidoreductases POR and CYB5R1','https://doi.org/10.1016/j.molcel.2020.11.024'),
 'nadk':('NADK governs ferroptosis susceptibility by orchestrating NADPH homeostasis','https://pmc.ncbi.nlm.nih.gov/articles/PMC12730079/'),
 'mesh1':('MESH1 is a cytosolic NADPH phosphatase that regulates ferroptosis','https://www.nature.com/articles/s42255-020-0181-1'),
 'ngly1':('Ferroptosis regulation by the NGLY1/NFE2L1 pathway','https://pmc.ncbi.nlm.nih.gov/articles/PMC8931371/'),
 'nrf2':('Activation of the p62-Keap1-NRF2 pathway protects against ferroptosis in hepatocellular carcinoma cells','https://pubmed.ncbi.nlm.nih.gov/26403645/'),
 'atf3':('ATF3 promotes erastin-induced ferroptosis by suppressing system Xc-','https://pubmed.ncbi.nlm.nih.gov/31273299/'),
 'atf4_protect':('The regulation of ferroptosis by MESH1 through the activation of the integrative stress response','https://www.nature.com/articles/s41419-021-04018-7'),
 'atf4_promote':('EIF2alpha-ATF4-CHAC1 signalling links ER stress to ferroptosis in human aortic smooth muscle cells','https://pmc.ncbi.nlm.nih.gov/articles/PMC13135238/'),
 'cma':('Chaperone-mediated autophagy is involved in the execution of ferroptosis','https://pmc.ncbi.nlm.nih.gov/articles/PMC6386716/'),
 'ckb':('Creatine kinase B suppresses ferroptosis by phosphorylating GPX4 through a moonlighting function','https://pubmed.ncbi.nlm.nih.gov/37156912/'),
 'trans':('Loss of cysteinyl-tRNA synthetase induces the transsulfuration pathway and inhibits ferroptosis induced by cystine deprivation','https://pmc.ncbi.nlm.nih.gov/articles/PMC4716307/'),
 'prdx6':('PRDX6 augments selenium utilization to limit iron toxicity and ferroptosis','https://pmc.ncbi.nlm.nih.gov/articles/PMC11327102/'),
 'original':('Ferroptosis: an iron-dependent form of non-apoptotic cell death','https://pmc.ncbi.nlm.nih.gov/articles/PMC3367386/'),
 'iron':('Ferroptosis is controlled by coordinated transcriptional regulation of glutathione and labile iron metabolism by BACH1','https://pmc.ncbi.nlm.nih.gov/articles/PMC6952604/'),
 'prom2':('Prominin2 drives ferroptosis resistance by stimulating multivesicular body/exosome-mediated iron export','https://pmc.ncbi.nlm.nih.gov/articles/PMC8316835/'),
 'pufa':('Polyunsaturated fatty acid biosynthesis pathway determines ferroptosis sensitivity in gastric cancer','https://pubmed.ncbi.nlm.nih.gov/33288688/'),
 'fads':('FADS1/2 control lipid metabolism and ferroptosis susceptibility in triple-negative breast cancer','https://pubmed.ncbi.nlm.nih.gov/38926633/'),
 'far1':('Peroxisome-driven ether-linked phospholipid biosynthesis is essential for ferroptosis','https://pmc.ncbi.nlm.nih.gov/articles/PMC8329287/'),
 'gnpat':('CSE triggers ferroptosis via SIRT4-mediated GNPAT deacetylation in the pathogenesis of COPD','https://pubmed.ncbi.nlm.nih.gov/38041059/'),
 'tmem':('TMEM164 is an acyltransferase that forms ferroptotic C20:4 ether phospholipids','https://pubmed.ncbi.nlm.nih.gov/36782012/'),
 'becn':('AMPK-mediated BECN1 phosphorylation promotes ferroptosis by directly blocking system Xc- activity','https://pmc.ncbi.nlm.nih.gov/articles/PMC6081251/'),
 'bap1':('BAP1 links metabolic regulation of ferroptosis to tumour suppression','https://pubmed.ncbi.nlm.nih.gov/30202049/'),
 'abcc1':('A genome-wide haploid genetic screen identifies regulators of glutathione abundance and ferroptosis sensitivity','https://pmc.ncbi.nlm.nih.gov/articles/PMC6424331/'),
 'gsr':('Glutathione reductase deficiency potentiates the immunogenicity of ferroptosis and cuproptosis via amplified ROS accumulation and cGAS-STING activation','https://pmc.ncbi.nlm.nih.gov/articles/PMC13531920/'),
}

# gene | role for increased functional activity | evidence level | inducer scope |
# references | concise interpretation, including context and uncertainty.
# R/S are functional expectations, never a prespecified universal RNA coefficient.
DATA='''
SLC7A11|R|functional|erastin_cystine|xc|Cystine import supports GSH defense; target abundance is not target inhibition or dependency.
GCLC|R|functional|cystine|gclc|GCLC protects during cystine starvation through GSH and noncanonical peptide metabolism.
GCLM|R|pathway|pathway|gclc;iron|GCL modifier subunit; protective pathway expectation, not a universal independent RNA-expression effect.
GSS|R|pathway|pathway|review;gclc|GSH synthesis supports defense; cystine-starved cells can also use GSH-independent protective pathways.
GSR|R|functional|erastin_and_other|gsr|GSH recycling protects and knockout sensitizes tested cells. The 2026 paper also analyzes CTRP expression-response data; that observational result is not independent of our public-data benchmark.
GGT1|R|functional|cystine|ggt1|Extracellular GSH salvage protects glioblastoma cells; depends on substrate availability and cell density.
CNDP2|R|functional|cystine|cndp2|Cysteinylglycine hydrolysis supplies cysteine; protection requires extracellular GSH-derived substrates.
CTSD|U|source_conflict|unresolved|review;ctsb;ctsd2025|Review names cathepsin D, but reference 59 identifies cathepsin B/CTSB. Newer CTSD glycosylation/CRC work does not establish a pan-cancer erastin mRNA direction.
CTNS|R|functional|cystine|ctsb;ctns|Lysosomal cystine export supports cysteine supply and adaptive ATF4; albumin and lysosomal context matter.
GPX4|R|functional|erastin_and_other|gpx4|Lipid peroxide detoxification protects; mRNA does not measure selenoprotein abundance or enzymatic activity.
LRP8|R|functional|other_inducer|lrp8|Selenium uptake maintains GPX4 translation; strongest direct evidence involves GPX4-inhibitor sensitivity.
AIFM2|R|functional|general_ferroptosis|fsp1|FSP1-dependent antioxidant defense protects; localization and reductase activity matter.
STARD7|R|functional|other_inducer|stard7|Cytosolic STARD7 supplies plasma-membrane CoQ; total RNA does not resolve processing or localization.
DHFR|R|functional|other_inducer|bh4|BH4 recycling protects against GPX4 inhibition. That study did not establish the same requirement under cystine depletion.
TFRC|S|functional|cystine_and_starvation|glutam|Transferrin-mediated iron uptake supports death in tested starvation models; trafficking and iron loading matter.
NCOA4|S|functional|erastin|ncoa4|Ferritinophagy sensitizes to erastin in HeLa; the same study did not find equivalent RSL3 dependence.
NFS1|R|functional|cystine|nfs1|Iron-sulfur maintenance protects from ferroptosis; dependence varies with oxygen tension.
ACSL4|S|functional|general_ferroptosis|acsl4|PUFA activation usually sensitizes; requirement depends on cell context and available lipids.
LPCAT3|M|mixed|mixed|review;kras|Usually sensitizes through PUFA remodeling; protective palmitate remodeling is reported in KRAS-mutant lung cancer.
AGPAT3|S|functional|other_inducer|ether|PUFA phospholipid synthesis promotes GPX4-inhibitor sensitivity in tested cancer models.
MBOAT7|S|functional|other_inducer|mboat7|MMD-dependent PUFA-PI remodeling promotes ferroptosis; overexpression alone need not be sufficient.
ACSL3|M|mixed|mixed|review;kras|Usually supports MUFA-mediated resistance, with a pro-ferroptotic exception in KRAS-mutant lung cancer.
SCD|R|functional|other_inducer|srebf|MUFA production supports resistance through lipid remodeling; substrate and signaling context matter.
MBOAT1|R|functional|other_inducer|mboat12|MUFA remodeling protects, notably in estrogen-receptor-positive breast cancer.
MBOAT2|R|functional|other_inducer|mboat12|MUFA remodeling protects, notably in androgen-receptor-positive prostate cancer.
PLA2G6|R|functional|general_ferroptosis|pla2g6|Oxidized phospholipid hydrolysis removes pro-ferroptotic lipid signals.
ABHD12|R|functional|other_inducer|abhd12|Inhibition potentiates ferroptosis by altering phosphatidylserine lipid remodeling.
POR|S|functional|general_ferroptosis|por|Electron-transfer-dependent lipid oxidation promotes ferroptosis; model RNA associations need not follow this function.
CYB5R1|S|functional|general_ferroptosis|por|Oxidoreductase activity supports lipid oxidation; distinguish CYB5R1 from CYB5R3.
NADK|R|functional|other_inducer|nadk|Recent HT1080 perturbation evidence supports NADPH-dependent protection; erastin was explicitly left for future testing.
HDDC3|S|functional|erastin|mesh1|MESH1 overexpression sensitizes and depletion protects tested cells against erastin; multiple substrates and stress programs may contribute.
NFE2L2|M|mixed|mixed|nrf2;abcc1|NRF2 usually protects, but can also induce sensitizing ABCC1-mediated GSH efflux.
NFE2L1|R|functional|general_ferroptosis|ngly1|NRF1 supports GPX4-dependent protection; requires post-translational maturation.
NGLY1|R|functional|general_ferroptosis|ngly1|Deglycosylation enables NFE2L1-mediated protection; RNA is not enzyme or processed NRF1 activity.
SREBF1|R|functional|other_inducer|srebf|SREBP-driven lipogenesis commonly protects via SCD/MUFAs; transcription-factor activation is not basal RNA.
ATF4|M|mixed|mixed|ctns;atf4_protect;atf4_promote|Adaptive cysteine-stress programs can protect, whereas CHAC1-associated stress programs can promote ferroptosis.
ATF3|S|functional|erastin|atf3|Direct study shows SLC7A11 repression and erastin sensitization; stress-induced activation differs from basal RNA.
HSPA8|S|functional|erastin|cma|HSC70 participates in chaperone-mediated GPX4 degradation; broad chaperone functions complicate expression interpretation.
LAMP2|S|isoform|erastin_isoform|cma|Evidence concerns LAMP2A-mediated GPX4 degradation; gene-level LAMP2 RNA combines isoforms and other lysosomal functions.
CKB|R|functional|general_ferroptosis|ckb|GPX4 phosphorylation prevents chaperone-mediated degradation; kinase activation is not inferred from RNA alone.
SLC3A2|R|pathway|pathway|xc|System xc- heavy-chain partner supports cystine import; also serves other amino-acid transporters.
CBS|R|pathway|erastin_cystine_pathway|trans|Transsulfuration supplies cysteine; reported pathway protection does not establish a universal CBS-expression coefficient.
CTH|R|functional|erastin_cystine|trans|Transsulfuration can bypass cystine-import blockade; methionine availability and pathway flux matter.
GLS2|S|functional|cystine_and_starvation|glutam|Glutaminolysis promotes ferroptosis in tested starvation models; GLS2 and GLS1 are distinct genes.
GOT1|M|mixed|mixed|glutam;got1|Can promote starvation-associated ferroptosis; in pancreatic cancer, GOT1 loss sensitizes cells to antioxidant-pathway inhibition.
PRDX6|R|functional|general_ferroptosis|prdx6|Selenium delivery supports GPX4 synthesis and protection; not simply a lipid peroxide expression marker.
SEPHS2|R|pathway|pathway|prdx6|Selenoprotein synthesis machinery supports GPX4; gene-specific pan-cancer erastin-expression direction not established here.
GCH1|R|functional|erastin_and_other|bh4;bh4_crc|BH4 defense protects against GPX4 inhibition; erastin protection is demonstrated in selected colorectal cancer models.
PTS|R|functional|other_inducer|bh4|BH4 synthesis supports GPX4-inhibitor resistance in tested cells; not a universal erastin dependency.
SPR|R|functional|other_inducer|bh4|BH4 synthesis supports GPX4-inhibitor resistance in tested cells; inducer and salvage routes matter.
IREB2|S|functional|erastin|original|IRP2 knockdown protected cells in the original erastin screen; iron-regulatory activity is central.
FTH1|R|functional|erastin_and_other|nrf2;iron|Ferritin heavy chain limits labile iron; loading and ferritinophagy can differ from RNA abundance.
SLC40A1|R|pathway|erastin_cystine_pathway|iron|Ferroportin-mediated iron export is protective; BACH1 pathway evidence is not a universal independent expression effect.
PROM2|R|functional|erastin_and_other|prom2|Iron export through ferritin-containing vesicles protects tested epithelial/breast cells.
FADS1|S|functional|other_inducer|pufa;fads|PUFA synthesis sensitizes tested gastric and breast cancer cells; extracellular lipids affect flux.
ELOVL5|S|functional|other_inducer|pufa|PUFA elongation supports sensitization in gastric cancer; not a universal erastin-expression biomarker.
FADS2|S|functional|other_inducer|fads|PUFA synthesis promotes ferroptosis in tested breast cancer models; substrate context matters.
FAR1|S|functional|other_inducer|ether;far1|Ether-lipid substrate production promotes ferroptosis in tested cancer settings.
GNPAT|S|functional|other_inducer|ether;gnpat|Ether-lipid synthesis supports ferroptosis; direct GNPAT perturbation evidence also involves cigarette-smoke extract.
AGPS|M|mixed|mixed|ether;ether_opposite|Loss can protect some cancer cells, while AGPS inhibition sensitizes other human cells and ether lipids protect worms.
TMEM164|S|functional|other_inducer|tmem|Pro-ferroptotic arachidonoyl ether phospholipids support GPX4-inhibitor sensitivity.
BECN1|S|functional|erastin|becn|Phosphorylated BECN1 inhibits system xc-; basal RNA does not report phosphorylation or complex formation.
BAP1|S|functional|erastin|bap1|Wild-type BAP1 represses SLC7A11 and promotes ferroptosis; mutation and catalytic function matter.
ABCC1|S|functional|erastin_and_other|abcc1|MRP1 GSH export sensitizes to ferroptosis despite resistance to some other drugs.
'''


def main():
    refs={k:dict(id=k,title=title,url=url,source_type='supplied_review' if k=='review' else 'primary_study',
        checked='2026-11-02',retrieval='supplied PDF and current live title/abstract/indexed article excerpts' if k=='review' else 'current live title/abstract or indexed article excerpts') for k,(title,url) in REFERENCES.items()}
    rows=[]
    for line in DATA.strip().splitlines():
        gene,role,level,scope,refids,note=line.split('|')
        r=refids.split(';');assert set(r)<=set(refs)
        rows.append(dict(gene=gene,literature_role=role,expected_sensitivity_sign={'R':-1,'S':1,'M':0,'U':0}[role],
            evidence_level=level,inducer_scope=scope,reference_ids=r,reference_urls=[refs[i]['url'] for i in r],note=note))
    genes=json.load(open(WS/'outputs/retained_64_gene_models/model_registry.json'))['genes_in_input_order']
    assert [r['gene'] for r in rows]==genes and len(rows)==len({r['gene'] for r in rows})==64
    (OUT/'references.json').write_text(json.dumps(refs,indent=2)+'\n')
    evidence=dict(scope='Targeted functional-direction audit, not a systematic or exhaustive literature review; cutoff 2026-11-02.',
        roles=dict(R='Higher functional activity commonly protects/resistance',S='Higher functional activity commonly sensitizes',M='Documented context-dependent roles; no single assigned sign',U='Direction unresolved or source conflict'),
        comparison_limit='Functional perturbation direction is a mechanistic expectation; not proof that baseline mRNA must have the same conditional coefficient.',genes=rows)
    (OUT/'candidate_gene_evidence.json').write_text(json.dumps(evidence,indent=2)+'\n')
    print(f'Curated {len(rows)} genes and {len(refs)} reference records.')


if __name__=='__main__':main()
