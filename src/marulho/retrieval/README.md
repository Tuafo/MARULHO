# Retrieval

Use this with [../../../README.md](../../../README.md) and
[../../../CONTEXT.md](../../../CONTEXT.md).

`retrieval` owns exact tensor candidate search, routing caches, and decoder
support paths.

## Owns

- `search_tensors()` candidate ids and distances that stay on the routing
  device for trainer competition and CUDA kernels.
- Routing tensor cache invalidation and rebuild policy.
- Routing cache generation stamps used by graph eligibility.

## Must Not Own

- CUDA capability claims without observed telemetry.
- Trainer mutation policy.
- Hidden alternate routing backends in the live path.

## Ported Guidance

- The legacy list-returning search API and selectable CPU FAISS/numpy routing
  backends are not the maintained live path.
- Future IVF or quantized routing work must enter as a bounded GPU-owned
  candidate router with capacity, fallback, recall, device, and complete-run
  throughput evidence.
- Same-shape, same-device cache rebuilds should copy into existing tensors to
  preserve graph-safe addresses. Shape or device changes may replace tensors
  and must disable graph replay before mutation.
