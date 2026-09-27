# Private custody workflow v1

The custody workflow stores artifacts and receipts outside the repository. It does not extract archives, deserialize model weights, inspect voxel arrays, run inference, or contact a network unless the operator supplies an explicit network flag.

## Offline public validation

```bash
uv run voxelscope source verify \
  --registry research/source-registry-v1.json

uv run voxelscope custody plan \
  --registry research/source-registry-v1.json \
  --plan research/acquisition-plan-v1.json

uv run voxelscope custody scan-public --root .
```

These commands use only repository content. The plan remains NO-GO because Task01 single-file acquisition is blocked.

## Initialize owner-private custody

Choose an absolute directory outside the repository. Its parent must already exist.

```bash
uv run voxelscope custody init --root "$CUSTODY_ROOT"
```

On POSIX systems the root and created subdirectories must be owner-only. Receipt files are created owner-readable and owner-writable only. The CLI refuses symlinks, existing destinations, and non-private roots.

## Acquire the permitted model archive

Network access is disabled by default. The following command is the only planned network acquisition:

```bash
uv run voxelscope custody acquire \
  --registry research/source-registry-v1.json \
  --plan research/acquisition-plan-v1.json \
  --artifact-id monai-brats-bundle-v0.5.2 \
  --root "$CUSTODY_ROOT" \
  --allow-network
```

The downloader:

- Accepts only the exact source URL in the plan.
- Allows redirects only between the exact HTTPS origins in the plan.
- Sends no credentials.
- Performs no retry.
- Requires the expected byte size and every published content hash.
- Refuses overwrite, symlinks, encrypted ZIP members, traversal, duplicate paths, excessive file count, excessive expanded size, excessive per-file size, and excessive compression ratio.
- Requires the exact closed archive member set and license file.
- Hashes archive members without extraction or model loading.
- Writes an owner-only private receipt after verification.

## Verify an already-staged artifact

An operator may place the exact archive at its planned relative location under the private root and run:

```bash
uv run voxelscope custody verify \
  --registry research/source-registry-v1.json \
  --plan research/acquisition-plan-v1.json \
  --artifact-id monai-brats-bundle-v0.5.2 \
  --root "$CUSTODY_ROOT"
```

Verification writes a new no-clobber receipt. A second receipt with the same artifact ID is refused.

## Dataset stop

Do not invoke acquisition for `msd-task01-single-volume`. The plan marks it blocked and the CLI refuses it mechanically. The official bucket exposes only the full 7.6 GB Task01 tar. VoxelScope does not download that archive, range-extract members, or infer candidate identities from unofficial mirrors.

The next acceptable data source must provide direct official URLs for one image, its matching label, and `dataset.json`, with immutable expected sizes and hashes.
