# Contributing

VoxelScope accepts narrowly scoped changes that strengthen reproducible research evidence.

Before opening a pull request:

1. Keep all tests synthetic and offline. Do not add medical data, model weights, PHI, secrets, or remote fetches.
2. Preserve canonical serialization and fail-closed validation. Schema changes require a version decision, migration notes, and new tamper tests.
3. State whether a scientific contract changes. Update the preregistration before implementing a changed prospective rule.
4. Do not add diagnostic, treatment, clinical-safety, radiologist-equivalence, or patient-specific claims.
5. Run `make validate`.
6. Inspect the generated summary and root hash. Do not commit generated bundles.

Real-data source or custody changes must keep all artifacts and private receipts outside the repository. Public source and acquisition records must remain canonical, versioned, reviewable, and NO-GO until every readiness gate is approved.

Bug reports should include synthetic reproduction steps, the VoxelScope version, the failure code, and non-sensitive evidence hashes. Never attach medical images or patient-derived metadata.
