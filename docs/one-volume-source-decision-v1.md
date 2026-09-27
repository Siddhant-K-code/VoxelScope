# One-volume source decision v1

## Decision

The milestone 3 source path is **GO**. Real-data acquisition and inference remain **NO-GO**.

OpenNeuro dataset `ds007045` snapshot `2.0.1` exposes one selected case as:

1. Four separate N4-corrected MRI objects in model order T1c, T1, T2, and FLAIR.
2. One spatially aligned tumor mask.
3. Three version-pinned metadata files covering the dataset DOI, CC0 license, BIDS version, processing description, and snapshot changes.

The five medical objects total 24,219,725 bytes. Each has an official versioned S3 URL, exact byte size, HTTP ETag, and SHA-256 from the git-annex key at the tagged dataset commit. The three metadata files have commit-pinned raw URLs, byte sizes, Git blob SHA-1 values, and SHA-256 values.

The canonical records are:

- [`research/one-volume-source-decision-v1.json`](../research/one-volume-source-decision-v1.json)
- [`research/one-volume-acquisition-plan-v1.json`](../research/one-volume-acquisition-plan-v1.json)

The acquisition plan is network-disabled and explicitly blocked. It has no CLI acquisition command. No medical data were downloaded.

## Outcome-blind case selection

The selected case is the lexicographically first root subject directory in snapshot `ds007045:2.0.1` that has all four N4-only modalities and an aligned mask. Selection uses only OpenNeuro snapshot tree filenames, annex flags, sizes, and content identities. It does not inspect medical bytes, mask values, lesion properties, or model outputs.

The official snapshot API reports 363 root subject directories. All 363 have the required N4-only four-modality and aligned-mask set under the path rules pinned by this record. Lexical order begins with the selected public BIDS locator, followed by the other two locators from the same source site. This establishes the selected case without outcome-based choice.

## Search and verification method

Independent searches were run through both Exa.ai and Parallel.ai. Direct source verification covered Project MONAI, TCIA and NBIA, BraTS and CBICA, FeTS, Synapse, Zenodo, OpenNeuro, and the peer-reviewed dataset publication.

The selected path was verified through four independent official control surfaces:

- OpenNeuro snapshot `ds007045:2.0.1`, created 2026-09-04T14:33:03Z.
- OpenNeuro GraphQL at `https://openneuro.org/crn/graphql`, whose recursive snapshot tree reports the five selected objects as annexed and returns their exact git-annex SHA256E identifiers and sizes.
- OpenNeuroDatasets Git tag `2.0.1`, commit `283007b96977b5b3285bd22cd67b7d97114b1eed`, tree `5d779f59a31e418ef39f06cf9bfa07292e4da41f`.
- Versioned OpenNeuro S3 object HEAD responses, which returned HTTP 200 and matched every expected S3 version ID, size, ETag, and server-side encryption header.

Search results from mirrors, personal repositories, and dataset aggregators were excluded as evidence. Search-provider summaries were used only to locate sources. Candidate facts were accepted only when an official publisher page, repository, API, or archival record exposed them directly.

## Selected source evidence

The tagged `dataset_description.json` declares:

- Dataset DOI `10.18112/openneuro.ds007045.v2.0.1`.
- BIDS version 1.8.0.
- CC0 license.

The tagged README and publication DOI `10.1038/s41597-026-07953-2` state that the data include pre-contrast T1, post-contrast T1, T2, and FLAIR. The N4 derivative is skull-stripped and linearly registered to MNI152. The paper states that masks are spatially aligned with processed images and were reviewed and corrected by neuroradiologists.

The paper cites the older OpenNeuro snapshot v1.0.1. The selected artifacts are pinned to v2.0.1, whose tagged CHANGES file documents cohort corrections. The publication supports preprocessing and mask semantics, while the OpenNeuro snapshot controls exact file identity.

The README describes the N4 derivative without a duplicated directory segment. The actual v2.0.1 Git tree contains `processing_MNI152_skull-stripped_N4` twice in the selected file paths. The canonical record pins the observed Git and S3 paths rather than correcting them.

## Model compatibility boundary

The pinned bundle remains MONAI `brats_mri_segmentation` 0.5.2.

- Model input channel order is T1c, T1, T2, and FLAIR.
- The selected plan lists the four image artifacts in exactly that order.
- The selected derivative is N4-corrected, skull-stripped, and MNI152-registered.
- The pre-z-scored derivative is intentionally not selected because the bundle performs channel-wise intensity normalization over nonzero voxels.
- Model output region order is TC, WT, and ET.

Private NIfTI checks remain required for shape, affine, dtype, finite values, and exact cross-file geometry. No source file has been opened or parsed.

## Label boundary

The OpenNeuro publication defines mask values as:

- 0: background or non-tumor tissue.
- 1: necrotic core.
- 2: non-enhancing or edematous tumor.
- 3: contrast-enhancing tumor.

The pinned MONAI output uses nested TC, WT, and ET channels and writes labels 1, 2, and 4 with ET priority. A prospective numeric mapping from dataset value 3 to evaluation value 4 is necessary, but it is not sufficient to establish full semantic equivalence. In particular, the source wording for value 2 may combine non-enhancing tumor and edema.

The label is therefore approved for matched custody identity only. It is not approved for a reference-label accuracy metric. This does not prevent a future output-preservation comparison between runtime executions on the same four-channel input.

## Candidate matrix

| Candidate | Official identity | Access result | Compatibility and lineage | Source decision |
|---|---|---|---|---|
| OpenNeuro ds007045 v2.0.1 | DOI `10.18112/openneuro.ds007045.v2.0.1`, tagged commit and versioned S3 objects | Direct four-modality image set, aligned mask, and metadata | Source identities pass; NIfTI inspection, label semantics, and overlap remain later gates | GO |
| MSD Task01 official S3 | Versioned 7,608,266,240-byte S3 object | Full tar only | Expected compatibility remains unverified because the descriptor and pair are inaccessible separately | NO-GO |
| MONAI extra test data 0.8.1 | GitHub release ID 59795362 | Individual assets, but only a BraTS folds JSON | No BraTS image or matching label | NO-GO |
| MONAI BraTS tutorial | Git commit `d948baa25946d109556f4304ba999d7c74a97c4d` | Code only | Requires separately obtained challenge data | NO-GO |
| TCIA BraTS 2021 | DOI `10.7937/jc8x-9874` | 142 GB Aspera transfer | Compatible NIfTI and labels, but TCIA reports that exact source DICOM to derived NIfTI linkage was lost | NO-GO |
| TCIA BraTS-TCGA-GBM | DOI `10.7937/K9/TCIA.2017.KLXWJJ1Q` | 767 MB ZIP for 102 subjects | Compatible processed data, no per-subject objects or hashes | NO-GO |
| TCIA BraTS-TCGA-LGG | DOI `10.7937/K9/TCIA.2017.GJQ7R0EF` | 536 MB ZIP for 65 subjects | Compatible processed data, no per-subject objects or hashes | NO-GO |
| TCIA DICOM-Glioma-SEG | DOI `10.7937/TCIA.2018.ow6ce3ml` | 3.8 GB controlled collection | Patient-space DICOM representation does not match the aligned NIfTI input contract | NO-GO |
| TCIA UPENN-GBM | DOI `10.7937/TCIA.709X-DN49` | Per-series raw DICOM, but aligned NIfTI and labels are a 69 GB transfer | TCIA states aligned NIfTI does not align with raw DICOM by design | NO-GO |
| CBICA BraTS 2020 | Official CBICA access workflow | Account approval and private result links | No public immutable per-subject identities, sizes, or hashes | NO-GO |
| FeTS 2021 | Official FeTS and CBICA workflow | Registration and private result links | Compatible representation, but no public direct one-subject identities | NO-GO |
| Synapse BraTS 2023 | Synapse entity `syn51156910` | Challenge access and bulk ZIP entities | No verified direct image-label-metadata set | NO-GO |
| Zenodo record 19541844 | DOI `10.5281/zenodo.19541844`, revision 4 | 436,401,049-byte bulk ZIP | Depositor is not an identified official BraTS publisher | NO-GO |
| Zenodo record 239084 | DOI `10.5281/zenodo.239084`, revision 6 | Per-case HDF5 objects | Five-channel transformed features and no matching segmentation label | NO-GO |

## NBIA boundary

The NBIA Search REST API documents `getSeriesSize` and `getImageWithMD5Hash`. This is meaningful direct access for raw DICOM series, but it does not satisfy the selected input path:

- A raw series is one acquisition, not the aligned four-channel NIfTI image expected by the bundle.
- The matching processed label is not available as a direct paired object.
- For UPENN-GBM, TCIA explicitly says the aligned NIfTI data do not align with the raw DICOM by design.

## Overlap boundary

The pinned model was trained on BraTS 2018. The selected OpenNeuro cohort reports acquisitions from 2018 through 2025 and is described as a separate multi-institutional collection, but the selected case acquisition date is not present in the public artifact contract. The model's exact training roster is also not published as a hash-bound list.

Exact subject overlap is therefore unresolved. The permitted future scope remains output preservation against the same FP32 reference. No reference-label accuracy or diagnostic claim is permitted.

## Acquisition and inference gates

Source identity, license, direct access, and image-label pairing are GO. Acquisition remains blocked until an operator approves the canonical plan and supplies a reviewed private-custody mechanism. After acquisition:

1. Verify every byte size and hash before opening a file.
2. Inspect NIfTI headers and finite-value properties without loading a model.
3. Confirm exact geometry across all four modalities and the mask.
4. Review the source normalization against the pinned bundle transforms.
5. Preregister the numeric ET remap from 3 to 4.
6. Keep broader label semantic equivalence and model-training overlap unresolved unless stronger evidence closes them.

No medical data were downloaded, ranged from an archive, extracted, or inspected. No model was loaded. No inference, GPU use, cloud provisioning, or paid service occurred.

The canonical records contain the selected dataset's public deidentified BIDS locator because it is required to bind the four images and mask. They contain no participant crosswalk, direct identity, clinical metadata, private path, credential, or signed URL.

The active execution boundary is [`one-volume-feasibility-protocol-v2.md`](one-volume-feasibility-protocol-v2.md). Protocol v1 remains an unexecuted historical Task01 plan.
