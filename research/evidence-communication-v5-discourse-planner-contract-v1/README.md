# Prospective v5 deterministic discourse-planner ablation contract

This directory freezes a **design contract only** for the question:

> Does a local model add measurable communication value beyond deterministic code
> once evidence claims and final prose are code-owned?

It does not implement the v5 compiler, declare or authorize a future study
execution, contain an observed v5 fixture, or present synthetic output as model
evidence. During the proposed ablation, trusted code publishes the deterministic
baseline. Under the current endpoints, exact deterministic code is dominant and
the pre-inference eligibility gate forbids a model run. A future model diagnostic
would require a separately reviewed, prospectively measurable objective that
deterministic optimization cannot settle.

Canonical artifacts:

| Artifact | SHA-256 |
|---|---|
| [`design-contract.json`](design-contract.json) | `7389b004f6a75ca00399573bf3ca45074f2941e4ac9772385d88f69a422a4fae` |
| [`discourse-request.schema.json`](discourse-request.schema.json) | `a347247bfaa818738f748de61633b47ccea640083df9aebde7f262fc54cfa038` |
| [`discourse-plan.schema.json`](discourse-plan.schema.json) | `806d847af7dcf0484d57291a468d3e6141e557547c8bb293ae21be272d1d6c1a` |
| [`publication-record.schema.json`](publication-record.schema.json) | `c4e85b0e5e1c821f29431487782df7f1247a8028d3c619a2018fa57e21666dd3` |

See the
[prospective v5 contract](../../docs/evidence-communication-v5-discourse-planner-contract.md)
for the architecture, caller-owned audience identity, exact deterministic
optimizer, pre-inference eligibility gate, trust boundaries, state machine,
custody, endpoints, decision rule, and explicit NO-GO boundaries.

The merged contract is implemented by the isolated
[deterministic v5 planner](../../docs/evidence-communication-v5-discourse-planner.md).
That implementation remains no-model and labels its control output as contract
and test evidence rather than observed model evidence.
