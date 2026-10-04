# VoxelScope launch assets

These repository-owned visuals are generated from committed synthetic study, audit,
contract, and deterministic control evidence. They use no stock art, third-party
creative asset, desktop capture, model execution, network execution, biomedical
data, or private machine path.

| Asset | Size | Intended use | Accessible alt text |
|---|---:|---|---|
| [`architecture-trust-boundary.svg`](architecture-trust-boundary.svg) and [`PNG`](architecture-trust-boundary.png) | 1600 x 900 | Article hero and social card | Evidence identity and typed semantics enter a pre-inference eligibility gate. The model path is ineligible. An exact optimizer, code-owned renderer, closed receipt, and offline replay complete publication. |
| [`v3-v4-v5-evolution.svg`](v3-v4-v5-evolution.svg) and [`PNG`](v3-v4-v5-evolution.png) | 1600 x 900 | Evidence evolution chart and social card | Version 3 exposed a repeatable construction-verifier mismatch. Version 4 used code-owned skeletons and eliminated invalid outputs in the observed study, while lexical exclusions and semantic variance remained. Version 5 used exact deterministic optimization with zero model and network actions. |
| [`v5-offline-workflow.gif`](v5-offline-workflow.gif) | 1200 x 720 | GitHub-compatible animated demo | A scripted terminal demo compiles the deterministic v5 control, highlights zero model and network actions, shows the closed receipt, and verifies offline replay. |

## Regenerate

Asset dependencies are isolated from the product dependency manifest:

```bash
uv run --with Pillow --with imageio --with numpy python \
  docs/assets/launch/generate_launch_assets.py
```

The GIF is a deterministic scripted terminal animation, not a captured shell. It
shows safe relative commands and sanitized outputs only. Type frames use 100 ms
timing, pauses precede annotations, and neutral callouts use `#FF9F1C`. GIF assembly
uses `imageio.v3.imwrite`, not Pillow's GIF writer.

[`provenance.json`](provenance.json) records every committed input digest, output
digest, dimension, file size, GIF frame count, and variable duration.
