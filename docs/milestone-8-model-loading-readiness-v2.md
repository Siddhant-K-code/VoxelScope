# Milestone 8 model-loading readiness v2

## Decision

The corrected prospective CPU model-loading gate is **IMPLEMENTATION-READY**. Private extraction and loading remain **NO-GO** pending a separate approval bound to the final v2 plan, private receipt/archive, interpreter, and runtime fingerprint. Inference remains **NO-GO**.

Runtime preflight found that PyTorch `v2.4.0` points to tag commit `d990dada86a8ad94882b5c23e859b88c0c255bda`, while the official Linux arm64 CPU wheel reports source commit `e4ee3be4063b7c430974252fdf7db42273388d86`. The tag is one release-only packaging commit ahead. Plan v1 conflated those identities and correctly refused the official wheel. It is superseded and must not be executed.

## Corrected runtime contract

Plan v2 binds both identities:

- PyTorch tag: `v2.4.0`;
- tag commit: `d990dada86a8ad94882b5c23e859b88c0c255bda`;
- official CPU wheel source commit: `e4ee3be4063b7c430974252fdf7db42273388d86`;
- Linux arm64 and CPython `3.12.14`;
- MONAI `1.4.0` at commit `46a5272196a6c2590ca2589029eed8e4d56ff008`;
- reviewed base image, requirements, direct wheel filenames, indexes, sizes, and SHA-256 identities in `research/model-loading-runtime-build-v1.json`.

The interpreter SHA-256 and closed PyTorch/MONAI distribution fingerprint remain outside Git. They must be reviewed and supplied through the owner-private inherited authorization descriptor.

## Evidence boundary

The prepared runtime was fingerprinted reproducibly twice in fresh network-disabled containers. That preflight did not read the private receipt or archive, write an attempt marker, extract or deserialize the checkpoint, instantiate or load the model, allocate model input, call forward, use an accelerator, provision cloud resources, or spend money.

The v1 plan and public bundle remain byte-identical historical evidence:

- v1 plan SHA-256 `48902842a3de6db5efc72f5c499550a92a656459125c18cac785da7407a0d91c`;
- v1 bundle SHA-256 `5b4e11c50330c13a8d5110731b390e59f12d46ae2bbb9fb4dfeacef7a281b9fd`.

The final corrected identities are:

- v2 plan SHA-256 `fb149979dd93f9cc0cb719b849e852b68a4cd7a779d346522a09e86bb76cf762`;
- reviewed runtime-build SHA-256 `aafbf1e3a55358614b7c96ee24c9e8114e19f4f008b1ef19784f38d6e148c557`;
- synthetic-only v2 bundle SHA-256 `66663fcede233a4dfbc980861441ba15da6607e141b024d5867af4abdf92f8a2`.

## Next gate

Obtain explicit owner approval bound to the exact v2 plan, reviewed private model receipt/archive identities, exact prepared interpreter, and reviewed runtime-distribution fingerprint. Execute once offline and stop after strict CPU construction, weights-only loading, introspection, and cleanup. No input tensor or forward call is authorized.
