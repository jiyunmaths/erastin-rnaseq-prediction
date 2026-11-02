# Proposed ferroptosis-panel expansion

The literature supports testing a broader panel, but does not establish that adding genes will improve erastin prediction. I propose preserving the original 40-gene baseline, adding 24 genes for a 64-gene comparison, then testing a broader 80-gene panel with 16 additional genes. These are analyst-selected mechanistic sets, not published predictive signatures. No new models have been fitted.

All 40 proposed additions (and three reserve candidates) have unique approved-symbol/Ensembl/Entrez mappings, finite variable expression in all 745 matched cell lines, and observed entries in the saved BulkFormer vocabulary. These checks use the saved HGNC and DepMap data, not a new data release. The selection did not use current erastin labels, feature-response correlations, or lineage-specific model wins.

## First expansion: 40 to 64 genes

| Module | Added genes | Why include it | Evidence |
|---|---|---|---|
| Cystine import and alternate cysteine | SLC3A2, CBS, CTH | Completes system xc- and represents a bypass around cystine uptake. | [Dixon et al., 2014](https://pmc.ncbi.nlm.nih.gov/articles/PMC4054777/), [Hayano et al., 2016](https://www.nature.com/articles/cdd201593) |
| Glutaminolysis | GLS2, GOT1 | Captures nutrient-dependent support for cystine-deprivation ferroptosis. | [Gao et al., 2015](https://www.sciencedirect.com/science/article/pii/S1097276515004505) |
| Selenium handling | PRDX6, SEPHS2 | Represents control of GPX4 synthesis rather than GPX4 transcript alone. | [Fujita et al., 2024](https://www.nature.com/articles/s41594-024-01329-z) |
| BH4 defense | GCH1, PTS, SPR | Adds a parallel antioxidant system; DHFR is already included. | [Kraft et al., 2020](https://pubmed.ncbi.nlm.nih.gov/31989025/), [Soula et al., 2020](https://www.nature.com/articles/s41589-020-0613-y) |
| Iron regulation/storage/export | IREB2, FTH1, SLC40A1, PROM2 | Complements TFRC and NCOA4 with labile-iron control and export. | [Dixon et al., 2012](https://pmc.ncbi.nlm.nih.gov/articles/PMC3367386/), [Nishizawa et al., 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC6952604/), [Brown et al., 2019](https://pmc.ncbi.nlm.nih.gov/articles/PMC8316835/) |
| PUFA biosynthesis | FADS1, FADS2, ELOVL5 | Adds substrate synthesis upstream of ACSL4/LPCAT3. | [Lee et al., 2020](https://pmc.ncbi.nlm.nih.gov/articles/PMC7768719/), [2024 primary study](https://pubmed.ncbi.nlm.nih.gov/38926633/) |
| Ether-lipid synthesis/remodeling | FAR1, GNPAT, AGPS, TMEM164 | Adds an incompletely represented lipid substrate pathway. | [2021 primary study](https://www.nature.com/articles/s41418-021-00769-0), [Reed et al., 2023](https://www.nature.com/articles/s41589-022-01253-7) |
| System xc-/glutathione regulation | BECN1, BAP1, ABCC1 | Adds transporter regulation and glutathione export with functional evidence. | [Song et al., 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6081251/), [Zhang et al., 2018](https://pmc.ncbi.nlm.nih.gov/articles/PMC6170713/), [Cao et al., 2019](https://pubmed.ncbi.nlm.nih.gov/30726737/) |

## Broader expansion: 64 to 80 genes

SC5D, DHCR7, VKORC1L1, G6PD, DHODH, GPD2, HMOX1, KEAP1, TP53, CDH1, NF2, YAP1, WWTR1, ZEB1, EPAS1, HILPDA.

This second module includes additional antioxidant chemistry, mitochondrial redox defenses, and cell-state regulators. Its tier reflects the proposed comparison order, not uniformly weaker biological evidence.

- SC5D and DHCR7 control the protective sterol 7-DHC and can have opposing functional effects. [2024 primary study](https://www.nature.com/articles/s41586-023-06983-9).
- VKORC1L1 supplies reduced vitamin K as a parallel defense. It is a different gene from VKORC1. [2023 primary study](https://pmc.ncbi.nlm.nih.gov/articles/PMC10529626/).
- G6PD supplies NADPH, but the original erastin study found its silencing protective in Calu-1, illustrating why a universal expression sign is inappropriate. [Dixon et al.](https://pmc.ncbi.nlm.nih.gov/articles/PMC3367386/).
- DHODH and GPD2 mainly have mitochondrial/GPX4-inhibitor evidence. DHODH inhibitor specificity and contribution are disputed. [Original DHODH study](https://www.nature.com/articles/s41586-021-03539-7), [Matters Arising](https://www.nature.com/articles/s41586-023-06269-0), [GPD2 study](https://pubmed.ncbi.nlm.nih.gov/35749365/).
- HMOX1 can promote or suppress erastin ferroptosis depending on the model. [Promoting setting](https://pmc.ncbi.nlm.nih.gov/articles/PMC4695193/), [protective setting](https://pmc.ncbi.nlm.nih.gov/articles/PMC6031916/).
- KEAP1 and TP53 are relevant but their RNA levels do not establish functional genotype or pathway activity. [KEAP1/NRF2 study](https://pmc.ncbi.nlm.nih.gov/articles/PMC4688087/), [p53/DPP4 study](https://pubmed.ncbi.nlm.nih.gov/28813679/).
- CDH1, NF2, YAP1, WWTR1 and ZEB1 capture contact, Hippo signaling and mesenchymal lipid state. They may also carry lineage and culture-density information. [NF2/YAP](https://www.nature.com/articles/s41586-019-1426-6), [TAZ](https://www.sciencedirect.com/science/article/pii/S221112471931023X), [ZEB1](https://www.nature.com/articles/s41556-024-01464-1).
- EPAS1 and HILPDA have strong clear-cell/GPX4-dependency evidence; pan-cancer erastin relevance is less certain. [Clear-cell study](https://pubmed.ncbi.nlm.nih.gov/30962421/).

FTL, PGD and OTUB1 remain documented reserve candidates. Exclusion from this bounded proposal does not mean absence of ferroptosis relevance.

## What recent literature changes

PRDX6 is especially useful to add: the 2024 study identifies selenium delivery as a route to maintaining GPX4, with IKE and RSL3 perturbation evidence. [Fujita et al.](https://www.nature.com/articles/s41594-024-01329-z). A March 2026 preprint reports different NALM6 genetic sensitizers for erastin2 and RSL3: SLC3A2/PRDX6 feature in the erastin2 results, while BH4-pathway genes feature in RSL3 results. This supports annotating inducer-specific evidence, but the preprint is not treated as peer-reviewed confirmation or pan-cancer validation. [Murray et al., preprint](https://pmc.ncbi.nlm.nih.gov/articles/PMC13060147/).

FerrDb V3 now spans 22 regulated-cell-death modalities and distinguishes drivers, suppressors, markers and unclassified genes, with different evidence-confidence categories. A larger FerrDb-derived set should explicitly filter to ferroptosis and inspect validated human perturbation evidence. The resource was checked at the documentation level; this proposal is not a full FerrDb export or a gene-by-gene reproduction of its labels. [FerrDb documentation](https://www.zhounan.org/ferrdb/v3/pages/help.html).

## Proposed modeling comparison

1. Freeze the 40-, 64- and 80-gene sets before fitting. Keep the existing 745 samples, targets and 20 saved leave-lineage-out repeats.
2. Use the same ridge, neural-network and XGBoost pipelines, with all tuning confined to the existing inner folds. Compare added information before changing model families or search grids.
3. Report paired pooled and equal-lineage metrics, lineage win counts and conditional whole-lineage bootstrap intervals. Include separate ablations of the extra antioxidant and cell-state modules if predeclared.
4. Fit any data-driven variance filtering, feature selection, transformations or pathway aggregation inside training folds. Do not choose genes from their full-cohort erastin correlations.
5. Treat this as exploratory reuse of the current cohort. Several source papers also analyze CTRP/CCLE; mechanistic perturbations, rather than published response correlations, motivate these additions. External screen validation is still needed.

Do not equate gene perturbation effects with the direction of a baseline RNA regression coefficient. Ferroptosis defense depends on metabolites, protein activity, mutations, culture conditions and lipid availability as well as RNA. Post-treatment markers such as PTGS2 and CHAC1 are not automatically valid baseline predictors; their induction alone is insufficient evidence for inclusion. If BulkFormer is revisited, the old pooled 40-gene embedding does not become a larger-panel embedding merely because the extra genes are in its vocabulary.

## Scope and files

This was a targeted literature check through 2026-11-01 using the supplied 2024 review, primary-paper abstracts/results and current resource documentation. It was not an exhaustive systematic review; complete full-text access varied. Gene-specific evidence, stimulus and caveats are retained below.

- [Gene evidence and coverage table](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_panel_expansion/candidate_gene_evidence.tsv)
- [Machine-readable panel definitions](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_panel_expansion/panel_definitions.json)
- [Coverage audit](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_panel_expansion/coverage_summary.json)
- [Reference list](/home/sji/Documents/Codex/2026-10-31/t/outputs/erastin_panel_expansion/references.json)
