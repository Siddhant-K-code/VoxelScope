# Deterministic v5 discourse planner

## Scope

VoxelScope v5 implements the merged
[prospective discourse-planner contract](../research/evidence-communication-v5-discourse-planner-contract-v1/README.md)
as deterministic code. It is not an inference study and does not present observed
model evidence. The implementation has no model runner, prompt builder, model
endpoint, artifact acquisition, or network path.

The only shipped catalog is a small code-owned control case used as contract and
test evidence. Trusted code owns every sentence, citation, source identifier,
fact/caveat binding, safety boundary, optional-unit utility, ordering rule, anchor,
byte budget, renderer identity, and audience identity. The request contains no
free-form field. Its only audience profile is the caller-frozen
`research_technical` value required by the merged schema.

Clinical use, diagnosis, prognosis, treatment, patient-specific claims,
biological causality, ranking, target designation, druggability, real-data
acquisition, production-readiness claims, and superiority claims remain out of
scope. The control output does not claim improved subjective communication
quality.

## Exact optimizer

`evidence_communication_v5.py` performs bounded complete enumeration:

1. enumerate every optional-unit subset up to the request limit;
2. enumerate every request-allowed caveat anchor and placement decision;
3. enumerate every topological unit order permitted by the ordering and selected
   caveat constraints;
4. reject invalid optional anchors, incomplete mandatory coverage, invalid
   citations, and over-budget output; and
5. select the lexicographic optimum: maximum integer optional utility, minimum
   rendered UTF-8 bytes, then minimum canonical decision-key bytes.

The implementation does not switch to a heuristic. It fails closed when the
explicit configuration or candidate-evaluation bound would be exceeded. The
optimality certificate records all enumeration counts, the bounded search limit,
the full search-space digest, the winner decision-key digest, objective values,
and implementation identities. Offline replay reruns the complete search and
requires the same certificate and winner byte-for-byte.

The test suite also implements a separate permutation oracle and compares its
winner with the production optimizer across varied bounded catalogs, constraints,
budgets, zero-utility units, and ties in utility, bytes, order, and caveat-anchor
decisions.

## Eligibility and measurement semantics

The content-addressed eligibility record is persisted and fsynced before
optimization. It always records:

```text
gate_result=model_inference_forbidden
model_eligibility_status=ineligible_deterministic_dominance
model_candidate_action_count=0
```

There is no injectable model hook. Planner measurements are deterministic
operation counts. Wall-clock timing is deliberately marked
`not_recorded_to_preserve_byte_for_byte_replay`. Model inference, input tokens,
output tokens, and model latency are each explicitly
`not_applicable_no_model`; they are not recorded as observed zeros.

## Publication and replay

A closed bundle contains exactly:

```text
eligibility.json
implementation-manifest.json
index.json
measurement-status.json
optimality-certificate.json
plan.json
publication.json
receipt.json
rendered-artifact.json
request.json
sentence-catalog.json
verification.json
```

Core publication accepts a repository root, not caller-asserted custody. Before
writing eligibility it requires a clean Git index and worktree, including no
untracked entries, and derives the exact `HEAD` commit, root Git tree, and a
canonical implementation-manifest identity. The manifest closes these files:

```text
src/voxelscope/atomic.py
src/voxelscope/canonical.py
src/voxelscope/evidence_communication_v5.py
src/voxelscope/evidence_communication_v5_cli.py
src/voxelscope/evidence_communication_v5_records.py
src/voxelscope/records.py
research/evidence-communication-v5-discourse-planner-contract-v1/design-contract.json
research/evidence-communication-v5-discourse-planner-contract-v1/discourse-plan.schema.json
research/evidence-communication-v5-discourse-planner-contract-v1/discourse-request.schema.json
research/evidence-communication-v5-discourse-planner-contract-v1/publication-record.schema.json
```

`sentence-catalog.json` closes every code-owned sentence and citation used by the
renderer. Replay parses it strictly, verifies the request-pinned digest, and
requires both the catalog and implementation manifest to equal the code-owned
inputs in the exact recorded checkout.

The request, catalog, implementation manifest, and eligibility record are
written before optimization. The remaining records are generated from the exact
winner and all staged records are independently reconstructed. Publication then
fsyncs every staged file and the staging directory, atomically renames without
replacement, fsyncs the destination parent directory, exclusively creates and
fsyncs `receipt.json`, and finally fsyncs the output directory. A crash before
the receipt leaves an unclosed bundle that replay refuses.

Replay rejects missing or extra files, noncanonical JSON, unknown or duplicate
IDs, identity drift, profile drift, custody drift, terminal-state drift, digest
tampering, symlinks, ordering cycles, bad anchors or placements, infeasible
budgets, nonoptimal plans, and publication collisions. Replay reconstructs the
code-owned request, optimizer winner and certificate, rendering, metrics/status,
verification, index, and receipt without network or a model. It first rederives
custody from the supplied repository root and requires exact equality with the
stored commit, root tree, and manifest identity. Historical replay therefore
requires checkout of the exact recorded revision.

## CLI

Recompute the merged contract and schema identities:

```bash
python -m voxelscope.evidence_communication_v5_cli contract-verify \
  --repository-root .
```

Create the deterministic control bundle from a clean Git checkout:

```bash
python -m voxelscope.evidence_communication_v5_cli fixture-compile \
  --repository-root . \
  --run-id deterministic-control-v1 \
  --output build/evidence-communication-v5-control
```

The compile command derives custody inside the trusted core and refuses a dirty
index or worktree. Its publication optimizer limit is fixed to the contract
default; only the standalone optimizer API accepts custom limits for fail-closed
unit tests. The output is contract/test evidence, never observed model evidence.

Replay the closed bundle offline:

```bash
python -m voxelscope.evidence_communication_v5_cli replay \
  --repository-root . \
  --bundle build/evidence-communication-v5-control
```

The implementation and control CLI do not start, contact, download, pull, create,
remove, or execute a model. They do not access biomedical data.
