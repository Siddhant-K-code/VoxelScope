# Glioblastoma Evidence Atlas metadata source freeze v1

## Status and boundary

This milestone freezes source metadata only. It does not acquire or inspect molecular
measurements, patient or sample rows, controlled files, MRI, protected sequencing, GLASS
files, credentials, tokens, or private paths.

Research evidence only. Not for diagnosis or treatment decisions.

The real-data adapter gate remains **NO-GO**. No finding, protein ranking, causal claim, or
clinical claim is reported.

The canonical records are:

- [`metadata-source-freeze.json`](../../research/gbm-evidence-atlas-source-freeze-v1/metadata-source-freeze.json)
- [`gdc-open-file-manifest.json`](../../research/gbm-evidence-atlas-source-freeze-v1/gdc-open-file-manifest.json)
- [`access-decisions.json`](../../research/gbm-evidence-atlas-source-freeze-v1/access-decisions.json)
- [`crosswalk-decision.json`](../../research/gbm-evidence-atlas-source-freeze-v1/crosswalk-decision.json)
- [`receipt.json`](../../research/gbm-evidence-atlas-source-freeze-v1/receipt.json)

## Frozen source identities

Official PDC GraphQL metadata retrieved on 2026-10-02 pins the current and only reported
version of each required study.

| Stable study ID | Version UUID | Study name | Fraction | Cases |
|---|---|---|---|---:|
| `PDC000204` | `cfe9f4a2-1797-11ea-9bfa-0a42f3c845fe` | CPTAC GBM Discovery Study - Proteome | Proteome | 111 |
| `PDC000205` | `efc3143a-1797-11ea-9bfa-0a42f3c845fe` | CPTAC GBM Discovery Study - Phosphoproteome | Phosphoproteome | 111 |
| `PDC000446` | `b25c2cea-4a49-4bc3-9f87-3cf5ee026865` | CPTAC GBM Confirmatory Study - Proteome | Proteome | 118 |
| `PDC000448` | `e1f7dcb3-db7f-4f04-bc0f-b150af3888b9` | CPTAC GBM Confirmatory Study - Phosphoproteome | Phosphoproteome | 118 |
| `PDC000514` | `524d5116-b6de-4e36-892a-e35dba7d0170` | KNCC Glioblastoma Evolution - Proteome | Proteome | 111 |
| `PDC000515` | `e5e0dd84-f982-46e3-b78a-5cb19eef31a8` | KNCC Glioblastoma Evolution - Phosphoproteome | Phosphoproteome | 91 |

All six report study version `1`, latest version `yes`, and experiment type `TMT11`.
The live PDC Study schema exposes no created, updated, or release timestamp. Its only
date field is `embargo_date`, which is null for these studies. The freeze records that
timestamp as unavailable instead of substituting a retrieval date.

The official GDC status endpoint reports Data Release 46.0, released 2026-08-10. The
freeze pins project `CPTAC-3`, dbGaP accession `phs001287`, 1,866 project cases, and 211
cases under the `Gliomas` disease filter. The file metadata query returned a candidate
superset of 498 open files for the eligible workflows:

- 229 `STAR - Counts` gene expression files.
- 269 `Aliquot Ensemble Somatic Variant Merging and Masking` files.

This inventory is not a selected analysis input and does not establish membership in the
PDC000204 and PDC000205 Discovery cohort. Exact membership remains unresolved until the
official target-study crosswalk and one-to-one patient and sample join checks pass. The
211 GDC `CPTAC-3 + Gliomas` cases must not be compared with or treated as equivalent to
the 111 PDC Discovery catalog cases.

For each candidate inventory file, the freeze records the GDC file ID, name, type,
format, workflow, size, access tier, and source-supplied MD5. It requests no case,
sample, aliquot, clinical, or biospecimen fields and downloads no file bytes.

## Access decision

The [deployed PDC source template](https://github.com/esacinc/PDC-Public/blob/e1c926c03dd384e86aaf9b7d46d36c6a381f15ec/ui-app/src/app/navbar/data-use-guidelines/data-use-guidelines.component.html)
for the official
[PDC data-use guidelines](https://pdc.cancer.gov/pdc/data-use-guidelines) states that
PDC data submission and use are governed by CC BY 4.0. The live route returned HTTP 200,
but its static response contained only the Angular application shell. The current
rendered terms text was not independently captured from the live response. PDC GraphQL
metadata is available without credentials. A one-row metadata probe for each required
study returned a representative file marked `Open` and `downloadable: Yes`.

That probe does not establish that every selectable file in each study is open. The
live file query returned the same total when given open, controlled, and invalid access
filters, so the access filter could not support an exhaustive negative claim. A full
file inventory was not fetched because it would enumerate hundreds or thousands of
rows per study. No PDC file URL was requested and no PDC file byte was read.

PDC general terms and file-byte access remain formally refused for acquisition. The
current rendered terms must be reviewed directly, and every selected file must have an
exact metadata record showing `Open` and `downloadable: Yes`, with any source-supplied
hash pinned.

Official GDC policy states that open data requires no login and controlled data requires
dbGaP authorization and authentication. Every record in the candidate GDC metadata
superset is marked `open`. This qualifies metadata openness only. It does not qualify
Discovery cohort inclusion, authorize input selection, or authorize acquisition. The
release manifest locators are pinned from the release notes, but their contents were not
downloaded. Clinical and biospecimen rows were not acquired.

## Crosswalk decision

The live PDC schema provides an official patient-level relationship through case
`externalReferences` entries labeled `GDC`, with a GDC case UUID and locator. The
target-study rows and coverage were not acquired in this metadata-only milestone.

The live PDC Sample schema exposes `gdc_sample_id` and `gdc_project_id`. Their population
and uniqueness for the six frozen studies remain unverified. The live PDC Aliquot schema
does not expose `gdc_aliquot_id`.

The execution crosswalk is therefore refused. Matching `C3L` or `C3N` strings does not
authorize a join. No patient, sample, or aliquot join may execute from these records.

## Gate decision

Resolved blockers:

- All six version-specific PDC study UUIDs, stable IDs, names, fractions, counts, and
  version states are pinned.
- GDC Data Release 46.0, project identity, and the candidate open workflow metadata
  superset with sizes, access tiers, and source-supplied MD5 values are pinned.
- The official PDC to GDC relationship capabilities are bounded by identifier level.

Unresolved blockers:

- PDC access is not exhaustively verified for every selected file.
- Current rendered PDC data-use terms are not independently captured and reviewed.
- Exact PDC report files, sizes, and source-supplied hashes are not yet selected and
  pinned.
- Target-study PDC to GDC patient and sample relationship rows are not release-pinned.
- Exact Discovery cohort membership and GDC input selection remain blocked pending the
  official target-study crosswalk and one-to-one join checks.
- The 211 GDC `CPTAC-3 + Gliomas` cases and 111 PDC Discovery catalog cases are not
  equivalent or directly comparable denominators.
- No official GDC aliquot relationship is exposed by the live PDC schema.
- GDC supplies MD5 in the frozen file metadata, not the SHA-256 required by the
  preregistered source-identity gate.
- The pinned HGNC, Ensembl, UniProt, and phosphosite sequence reference bundle and a
  qualified real-data adapter remain absent.

The exact next gate is a reviewed private-custody plan that first selects an exhaustive
open-only PDC file inventory, then freezes target-study patient and sample relationships,
measures zero, one, and multiple matches, resolves exact GDC Discovery membership, and
either obtains an official aliquot-level relationship or refuses the four-layer join.
No GDC input may be selected and no molecular file may be acquired until the crosswalk,
source-terms, and source-identity gates are complete.
