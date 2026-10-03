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
index.json
measurement-status.json
optimality-certificate.json
plan.json
publication.json
receipt.json
rendered-artifact.json
request.json
verification.json
```

The request and eligibility record are written first. The remaining records are
generated from the exact winner, all staged records are independently
reconstructed, every file is fsynced, and the staging directory is atomically
renamed without replacement. `receipt.json` is then created with exclusive
no-clobber semantics and fsynced last.

Replay rejects missing or extra files, noncanonical JSON, unknown or duplicate
IDs, identity drift, profile drift, custody drift, terminal-state drift, digest
tampering, symlinks, ordering cycles, bad anchors or placements, infeasible
budgets, nonoptimal plans, and publication collisions. Replay reconstructs the
code-owned request, optimizer winner and certificate, rendering, metrics/status,
verification, index, and receipt without network or a model.

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

The compile command obtains the exact Git revision itself and refuses a dirty
index or worktree. The output is contract/test evidence, never observed model
evidence.

Replay the closed bundle offline:

```bash
python -m voxelscope.evidence_communication_v5_cli replay \
  --bundle build/evidence-communication-v5-control
```

The implementation and control CLI do not start, contact, download, pull, create,
remove, or execute a model. They do not access biomedical data.
