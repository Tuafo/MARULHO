# Semantics

Use this with [../../../README.md](../../../README.md) and
[../../../CONTEXT.md](../../../CONTEXT.md).

`semantics` owns grounded language/readout contracts, cognitive signal
surfaces, decoder probes, concept evidence, and support diagnostics.

## Owns

- Bounded readout artifacts and grounded support evidence.
- ConceptStore assignment and structural concept maintenance.
- Language-from-spikes evaluation surfaces when they are bounded and
  evidence-gated.

## Must Not Own

- Free-form cognition claims.
- Fact promotion or action authority.
- External checkpoint loading or hidden LLM generation.

## Runtime Rules

- A readable label, readout slot, or draft is evidence for review, not proof of
  thought or autonomous fact formation.
- `ConceptStore` may cache derived normalized centroids, but checkpoint state
  stays canonical and restore rebuilds derived caches.
- Language-from-spikes remains MARULHO-owned, sparse, grounded, and
  checkpoint-backed. Live generation claims require trained readout evidence,
  rollback evidence, and throughput protection.
