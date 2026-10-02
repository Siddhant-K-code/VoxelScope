# Glioblastoma Evidence Atlas source qualification v1

## Status and boundary

This package qualifies public data sources and freezes an open-only first-study protocol. It does not download source datasets, analyze patient records, report a biological result, or implement a product or interface.

Research evidence only. Not for diagnosis or treatment decisions.

This record does not support claims about prognosis, druggability, treatment suitability, causal mechanisms, therapeutic targets, or patient-specific validity. A reported association or agreement would apply only to the declared source release, cohort, assay, and comparison rule.

Two merged pull requests define the implementation baseline:

- [PR #12](https://github.com/Siddhant-K-code/VoxelScope/pull/12) merged at commit `88d074b9116d913ce720bd1a5e72a7e24933110f` and owns the synthetic implementation, strict source manifest, identifier normalization, protein-card compiler, conflict and missing states, and canonical receipts.
- [PR #13](https://github.com/Siddhant-K-code/VoxelScope/pull/13) merged at commit `4bd9d60b895981112af267f813e1a31c1aacf4fe` and owns the product contract in `docs/glioblastoma-evidence-atlas-v0.md`.

This branch is rebased on both merged baselines and adds no competing runtime schema or implementation. A future real-data adapter must map qualified release records into the merged implementation under the merged product contract without changing their schemas or meanings.

The machine-readable records are:

- [`source-qualification-registry.json`](../../research/gbm-evidence-atlas-qualification-v1/source-qualification-registry.json)
- [`join-map.json`](../../research/gbm-evidence-atlas-qualification-v1/join-map.json)
- [`first-study-preregistration.json`](../../research/gbm-evidence-atlas-qualification-v1/first-study-preregistration.json)

## Qualification method

Qualification used primary official portal pages, documentation, and first-party APIs retrieved on 2026-10-02. Search summaries were not accepted as evidence. Dynamic portals were checked through their official APIs where possible. Unknown license text, mutable releases, blocked downloads, and unresolved crosswalks remain blockers rather than assumptions.

No controlled MRI, source data file, patient record, credential, token, or large dataset was downloaded or committed.

## Source decisions

| Source | Qualified release or study | Access decision | Atlas role | Current limitation |
|---|---|---|---|---|
| [TCIA CPTAC-GBM](https://www.cancerimagingarchive.net/collection/cptac-gbm/) | Version 16, DOI `10.7937/K9/TCIA.2018.3RJE41Q1` | Mixed. Radiology is controlled. Pathology slides are open under CC BY 4.0. | Optional later imaging and pathology context | MRI is excluded from the first study. A release-pinned crosswalk and separate authorization are required. |
| [PDC000204](https://proteomic.datacommons.cancer.gov/pdc/study/PDC000204) | CPTAC GBM Discovery Proteome, TMT11, 111 catalog cases | Metadata is open. File-byte access and general terms remain unverified. | First-study protein abundance | The portal describes 99 GBM participants plus 10 GTEx controls, while the catalog reports 111 cases. Case-level filtering is required. |
| [PDC000205](https://proteomic.datacommons.cancer.gov/pdc/study/PDC000205) | CPTAC GBM Discovery Phosphoproteome, TMT11, 111 catalog cases | Metadata is open. File-byte access and general terms remain unverified. | First-study phosphorylation | A version-specific study UUID and sequence-bound site identities must be pinned. |
| [PDC000446](https://proteomic.datacommons.cancer.gov/pdc/study/PDC000446) and [PDC000448](https://proteomic.datacommons.cancer.gov/pdc/study/PDC000448) | CPTAC GBM Confirmatory Proteome and Phosphoproteome, 118 cases each | Metadata is open. File-byte access and general terms remain unverified. | Later cohort-level replication | The cohort contains GBM, other gliomas, and other diseases. A GBM-only filter is mandatory. |
| [PDC000514](https://proteomic.datacommons.cancer.gov/pdc/study/PDC000514) and [PDC000515](https://proteomic.datacommons.cancer.gov/pdc/study/PDC000515) | KNCC Glioblastoma Evolution Proteome, 111 cases, and Phosphoproteome, 91 cases | Metadata is open. File-byte access and general terms remain unverified. | Later longitudinal replication | The KNCC namespace is independent from CPTAC-3. Pair counts require reconciliation from source relationships. |
| [GDC CPTAC-3](https://gdc.cancer.gov/about-gdc/contributed-genomic-data-cancer-research/clinical-proteomic-tumor-analysis-consortium-cptac) | Data release 46.0, project `CPTAC-3`, dbGaP `phs001287` | Mixed. Masked MAF, gene expression, clinical, and biospecimen files are open. Raw and aligned sequencing and several derived types are controlled. | First-study mutation and RNA data | CPTAC-3 is pan-cancer. The GDC Gliomas filter has 211 cases. A GBM-specific release-pinned crosswalk is required. |
| [GDC TCGA-GBM](https://portal.gdc.cancer.gov/projects/TCGA-GBM) | Data release 46.0, 617 cases, dbGaP `phs000178` | Mixed open and controlled categories | External cohort context only | It is not assumed to share or exclude patients with CPTAC-3. Its RPPA data are not CPTAC mass spectrometry. |
| [Ivy GAP](https://glioblastoma.alleninstitute.org/static/download.html) | Unversioned live download retrieved 2026-10-02 | Open under Allen Institute terms | Independent spatial RNA context | There is no official patient crosswalk to CPTAC, GDC, PDC, TCIA, or GLASS. No explicit release number was found. |
| [Human Protein Atlas](https://www.proteinatlas.org/about/download) | Version 25.1, Ensembl 109, released 2026-05-25 | Open under CC BY 4.0 for HPA-generated content | Normal brain and general protein context | It is not a matched GBM cohort. Integrated third-party records can retain separate terms. |
| [DepMap](https://depmap.org/portal/) | Public 26Q1 | Portal descriptions are open, but automated download verification was blocked by a bot check. | Later cell-model context | Current per-file terms, bulk access, and GBM model counts require human-browser verification. |
| [Open Targets](https://platform.opentargets.org/) | Platform 26.09, released 2026-09-24 | Open under CC0 1.0 | Outcome-independent candidate frame and later source-reported context | It must not be presented as a VoxelScope protein ranking or therapeutic claim. |
| [AlphaFold DB](https://alphafold.ebi.ac.uk/) | Version 6, UniProt 2025_03 | Open under CC BY 4.0 | Later structure availability context | Predictions have varying confidence and do not establish function, binding, or clinical use. |
| [GLASS](https://glass-consortium.org/datasets/) | Public data resource dated 2022-05-31, Synapse `syn17038081` | Mixed. Catalog metadata is open. Data file content returned HTTP 403 with unmet access requirements. | Gated later longitudinal context | Exact terms and files require Synapse access. GLASS is excluded from the open-only first study. |

### PDC access blocker

The official [PDC GraphQL documentation](https://proteomic.datacommons.cancer.gov/pdc/api-documentation) confirms that studies can have multiple versions and that a stable PDC study ID returns the latest version unless a version-specific UUID is supplied. Metadata queries returned without credentials. This does not prove that every file byte is open or establish a general license. The first study remains blocked until selected file downloads and current terms are verified without credentials or an application.

### GDC open-only boundary

The official [GDC CPTAC page](https://gdc.cancer.gov/about-gdc/contributed-genomic-data-cancer-research/clinical-proteomic-tumor-analysis-consortium-cptac) distinguishes open and controlled data types. The first study permits only:

- Clinical and biospecimen metadata marked open.
- WXS masked somatic mutation MAF marked open.
- Gene expression quantification marked open.

Aligned reads, raw variants, annotated variants, aggregated variants, and any file requiring a GDC token or dbGaP approval are excluded.

## Join map

Joins are level-specific. A valid join at one level does not authorize a join at another.

| Level | Required identity | First-study rule |
|---|---|---|
| Patient | Source-scoped case identity plus a release-pinned relationship or crosswalk | PDC to GDC is blocked until an official crosswalk binds PDC and GDC UUIDs. Similar `C3L` or `C3N` labels are a cross-check, not a join. |
| Sample | Source sample and aliquot identities linked to one patient | The exact tumor specimen used for mutation, RNA, proteome, and phosphoproteome must be one-to-one. A patient-only match is insufficient. |
| Gene | HGNC ID and unversioned Ensembl gene ID under pinned releases | Symbols are lookup inputs only. Ambiguous mappings stop the record. |
| Protein | Exactly one UniProt Swiss-Prot accession and declared isoform policy | More than one accession stops the candidate. |
| Phosphosite | UniProt accession, residue, one-based position, and exact sequence digest | A residue or sequence mismatch stops the site. |
| Cohort | Release-pinned project or study identifier and denominator | Discovery, Confirmatory, KNCC, TCGA-GBM, Ivy GAP, and GLASS remain separate cohorts. |

The TCIA collection states that its de-identified Patient ID matches the same subject in other CPTAC systems. The first imaging join still requires the collection-qualified identifier, frozen release, source relationship, and controlled-access approval.

Ivy GAP and GLASS have independent identifier namespaces. HPA, DepMap, Open Targets, and AlphaFold can join only at gene or protein level under pinned references. They do not add a patient match.

## Frozen first study

### Question

> Within the open-only CPTAC GBM Discovery cohort, how often do exact sample-linked somatic mutation group contrasts have the same direction across RNA abundance, protein abundance, and an outcome-blind selected phosphosite for the frozen candidate set?

### Candidate cohort

The primary cohort is the exact intersection of:

1. GBM tumor cases in frozen versions of PDC000204 and PDC000205.
2. Open GDC release 46.0 CPTAC-3 clinical, biospecimen, masked somatic mutation MAF, and gene-expression records.
3. One exact tumor specimen per patient with one-to-one patient and sample joins across all required layers.

GTEx controls, non-GBM cases, recurrent samples, cell models, organoids, xenografts, controlled files, and unmatched samples are excluded.

PDC000446 and PDC000448 are a later GBM-filtered cohort-level replication. PDC000514 and PDC000515 are a later independent longitudinal replication. Neither can change the first-study hypothesis or thresholds.

## Starting protein set

The candidate frame is frozen before GDC or PDC outcomes are read:

1. Use Open Targets release 26.09 and disease `MONDO_0018177`.
2. Request the first 20 `associatedTargets` rows in the API-defined overall association order.
3. Keep only a row with one Ensembl target ID, one approved symbol, and exactly one UniProt Swiss-Prot accession.
4. Do not backfill an excluded row.
5. Discard score order from the analysis and display the set by Ensembl gene ID.

`CDKN2A` is excluded because the source record returned more than one Swiss-Prot accession. The resulting 19-protein set is:

`FGFR1`, `ATRX`, `STAG2`, `GSR`, `VEGFA`, `PIK3CA`, `PDGFRA`, `IDH1`, `RB1`, `TP53`, `PIK3R1`, `EGFR`, `CDKN2B`, `BRAF`, `H3-3A`, `TERT`, `PTEN`, `PTPN11`, and `NF1`.

The exact Ensembl and UniProt identities are in the preregistration. This is an outcome-independent study frame. It is not a product ranking and does not imply causal, clinical, or therapeutic relevance.

Candidates remain in every result table even when a source, mapping, group-size, or missingness rule makes them unavailable.

## Preregistered analysis

For each candidate, altered means at least one eligible nonsynonymous somatic variant in the open GDC masked MAF for the exact joined tumor sample. Unaltered means no eligible variant in that same frozen MAF and an otherwise eligible joined sample.

RNA, protein, and phosphosite abundance are compared between altered and unaltered groups using:

- Cliff's delta as the effect size.
- A two-sided Wilcoxon rank-sum test.
- One Benjamini-Hochberg family across every confirmatory candidate-layer test.
- `q <= 0.05` for a significance label.

For each candidate, the primary phosphosite is chosen without outcome inspection. Select the sequence-valid site with the highest nonmissing sample coverage. Break ties by lowest one-based position, then residue ASCII order. All other valid sites are exploratory in a separate corrected family and cannot change the primary state.

### Hypotheses

The primary exact-binomial population contains only candidates for which RNA, protein, and the selected phosphosite all have `q <= 0.05`. A success is a candidate for which all three Cliff's delta signs match. A failure is a candidate for which the three signs are not all the same. The primary null is that the success probability is at most 0.5. The alternative is that it is greater than 0.5. The aggregate test is a one-sided exact binomial test with `p0 = 0.5` after candidate-layer correction.

Candidates with exactly two significant layers and opposing signs have the secondary state `secondary_two_layer_discordant`. Candidates with exactly two same-sign significant layers are `mixed_or_indeterminate`. Neither state enters the primary exact-binomial population. If no candidate has three significant layers, the primary aggregate test is unavailable and has no p-value. Candidate-layer and secondary results are still reported.

The candidate-layer null is equal altered and unaltered abundance distributions for each frozen candidate and layer. The alternative is a difference in at least one layer. A significant test does not establish causation.

## Result shapes

These are prospective shapes, not observed findings:

| Shape | Rule |
|---|---|
| `concordant_up` | RNA, protein, and selected phosphosite all have `q <= 0.05` and positive Cliff's delta. This is a primary success. |
| `concordant_down` | RNA, protein, and selected phosphosite all have `q <= 0.05` and negative Cliff's delta. This is a primary success. |
| `discordant` | All three layers have `q <= 0.05`, and their three effect signs are not all the same. This is a primary failure. |
| `secondary_two_layer_discordant` | Exactly two layers have `q <= 0.05` and their effect signs oppose. This state is excluded from the primary population. |
| `null` | No layer has `q <= 0.05`. Null results remain visible. |
| `mixed_or_indeterminate` | The candidate is analyzable but has one significant layer, two same-sign significant layers, or another pattern outside the declared primary, secondary, and null states. |
| `unavailable` | A source, access, join, identity, sequence, group-size, or missingness gate prevents analysis. |
| `feasibility_refusal` | A study-level stop gate fails. Only gate metrics and blockers may be released. |

Missing, restricted, unsupported, stale, and unmeasured records are not negative evidence.

## Measurable outcomes

Every released report must include:

- Missing count and fraction for mutation, RNA, protein, and phosphorylation before and after joining.
- Zero-match, one-match, and multiple-match counts at patient, sample, gene, protein, phosphosite, and cohort levels.
- Patient join coverage and exact sample join coverage with explicit denominators.
- Cliff's delta, raw p-value, BH q-value, direction, and state for mutation to RNA, mutation to protein, and mutation to the selected phosphosite.
- The primary population count, concordant success count, discordant failure count, concordant fraction, exact-binomial p-value, and empty-population availability state.
- Secondary two-layer opposing-sign and same-sign counts and the opposing-sign fraction, without adding them to the primary population.
- Null, mixed, unavailable, and feasibility-refusal counts.
- Identity, one-to-many join, scope, unit, sequence, and directional conflict counts.
- Canonical output and receipt SHA-256 values, repeat equality, and independent integer-metric agreement.
- All 19 candidate rows, including every null, negative, excluded, or unavailable case.

## Release and stop gates

| Gate | GO threshold | Failure action |
|---|---|---|
| Source terms | 100 percent of selected files have reviewed current terms and require no credential, application, or disallowed use | Stop before acquisition |
| Source identity | Every PDC version UUID, GDC release manifest, size, and SHA-256 is pinned | Stop before transformation |
| Patient joins | At least 80 percent of eligible PDC Discovery GBM cases join one-to-one to GDC, with zero one-to-many joins | Release only a feasibility refusal |
| Sample joins | At least 60 percent of patient-joined GBM samples and at least 60 cases have all four layers, with zero one-to-many joins | Release only a feasibility refusal |
| Candidate analyzability | At least 8 of 19 candidates have at least 5 altered and 20 unaltered complete cases plus one valid phosphosite | Stop inferential release |
| Missingness | Mutation status has zero missing values; RNA, protein, and selected phosphosite each have at most 30 percent missing among exact joined samples | Mark affected layers unavailable and apply the complete-case stop |
| Identifier integrity | Zero unresolved identity conflicts, zero guessed aliases, and 100 percent selected-site sequence validation | Stop before analysis |
| Multiple testing | One declared BH family and `q <= 0.05` for significance labels | Do not publish inferential labels |
| Primary binomial population | At least one candidate has significant RNA, protein, and selected phosphosite results | If zero, report the primary aggregate test as unavailable with no p-value and continue reporting candidate-layer and secondary results |
| Reproducibility | Two clean offline builds are byte-identical; an independent verifier agrees exactly on denominators, joins, missingness, and conflicts | Stop release |
| Reporting | All 19 candidates have exactly one state; primary population, success, and failure counts and secondary two-layer counts are present with every preregistered metric | Stop release |
| Privacy and copy | Zero controlled files, patient records, credentials, private paths, diagnostic or treatment claims, target labels, or causal claims | Stop release |

No passing gate overrides a failed gate.

## Current blockers and migration needs

The first study is currently blocked by:

- Unverified PDC file-byte access and current formal data-use terms.
- Missing version-specific UUIDs for PDC000205, PDC000446, PDC000448, PDC000514, and PDC000515.
- No release-pinned PDC to GDC patient and sample crosswalk approved for execution.
- No real-data adapter or source receipt maps qualified releases into the merged PR #12 implementation under the merged PR #13 product contract.
- No pinned HGNC, Ensembl, UniProt, or phosphosite sequence reference bundle.
- DepMap bot-check behavior and per-file terms.
- GLASS Synapse access requirements and exact terms.

The next safe action is a metadata-only source-freeze milestone. It should verify PDC file download behavior and terms, pin every PDC study version UUID, acquire the official PDC-GDC crosswalk, and produce content hashes for open-only manifests. It must not acquire controlled MRI, protected sequencing, GLASS gated files, or patient data for this branch.
