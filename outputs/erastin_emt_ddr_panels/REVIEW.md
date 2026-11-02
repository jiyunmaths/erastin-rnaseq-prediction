# Hallmark EMT and DNA-repair expression panels

The selected full Hallmark EMT (200 genes) and DNA Repair (150 genes) sets have complete coverage in the saved DepMap expression matrix. The culture-free CTRP cohort remains 744 lines, 733 identity groups and 23 lineages, with byte-identical shared validation splits.

| Panel | Genes |
|---|---:|
| Five-gene reference | 5 |
| Ferroptosis baseline | 64 |
| Baseline + EMT | 264 |
| Baseline + DNA repair | 213 |
| Baseline + both | 413 |

GPX4 is already in the 64-gene panel and the DNA-repair set; it is included once. All input values are finite. The original five- and 64-gene arrays are unchanged.

Membership JSON and mapped NCBI identifiers from the official MSigDB cards were retrieved on 2026-11-02 and agree exactly. DepMap has two columns carrying Entrez 552900. BOLA2 is resolved to BOLA2 (552900), supported by release HGNC:29488 / ENSG00000183336; the BOLA2B column is not substituted or averaged. See gene_mapping.tsv for every mapping and candidate column.

Hallmark sets are expression programs, not comprehensive pathway inventories. BRCA1/BRCA2 are absent from this DNA-repair set and ZEB1/ZEB2 are absent from this EMT set. No extra mechanistic genes were added beyond the user-selected Hallmark scope.

- [Official Hallmark EMT](https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_EPITHELIAL_MESENCHYMAL_TRANSITION.html)
- [Official Hallmark DNA Repair](https://www.gsea-msigdb.org/gsea/msigdb/human/geneset/HALLMARK_DNA_REPAIR.html)

The prediction experiment is saved separately in erastin_emt_ddr_comparison. This folder contains source snapshots, mapping, fixed definitions and matched unscaled expression/response arrays. Source files and preparation checksums are recorded in source_manifest.json and preparation_validation.json.
