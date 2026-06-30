# MARULHO Package Docs

Use this package map with [../../README.md](../../README.md) and
[../../CONTEXT.md](../../CONTEXT.md).

`CONTEXT.md` is the domain vocabulary source of truth. The root `README.md`
explains the project shape and current runtime spine. Each package README
below records local ownership rules and machinery-specific runtime rules.

## Machinery Folders

- [brain](brain/README.md): main `MarulhoBrain` runtime spine and compact trace.
- [core](core/README.md): local SNN tensor mechanisms and mutation algorithms.
- [data](data/README.md): source loaders and sparse encoder boundaries.
- [semantics](semantics/README.md): grounded readout, concepts, and language evidence.
- [training](training/README.md): trainer-owned CUDA, checkpoint, sequence, and replay execution.
- [service](service/README.md): thin HTTP/UI adapter and composition root.
- [consolidation](consolidation/README.md): CPU archival memory and replay records.
- [retrieval](retrieval/README.md): exact tensor routing caches and candidate search.
- [evaluation](evaluation/README.md): benchmarks, gates, and validation evidence.
- [interaction](interaction/README.md): operator-facing response shaping.
- [reporting](reporting/README.md): human-readable summaries over evidence.
- [config](config/README.md): model/runtime presets and configuration boundaries.

## Documentation Rule

Keep documentation close to the owning code. If a term changes, update
`CONTEXT.md`. If ownership or local machinery changes, update the closest
package README.
