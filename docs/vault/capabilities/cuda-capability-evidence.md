---
type: capability
status: draft
related_code:
  - ../../../src/marulho/evaluation/service_benchmark.py
related_docs: []
related_papers: []
related_benchmarks:
  - ../benchmarks/hot-path-latency.md
---

# Cuda Capability Evidence

CUDA claims are valid only when observed device/backend telemetry supports them.

## Evidence Rule

Do not claim this capability as live unless linked Runtime Evidence or benchmark output supports it.

Observed CPU placement is still useful Runtime Evidence, but it is not a CUDA acceleration claim. A CUDA capability claim needs `observed_cuda_execution=true` plus CPU/CUDA parity or benchmark-delta evidence.

## Latest Local Evidence

The 2026-06-09 configured-source service benchmark at ignored `reports/service_benchmark_cycle_configured/service-benchmark.json` reported:

- `summary_role`: `observed_runtime_device_evidence_not_acceleration_claim`
- `requested_device`: `auto`
- `resolved_device`: `cpu`
- `tensor_device`: `cpu`
- `routing_search_device`: `cpu`
- `encoder_device`: `cpu`
- `cuda_available`: `false`
- `observed_cuda_execution`: `false`
- `cuda_fallback_reason`: `cuda_not_available`
- Claim boundary: `observed_device_placement_only_not_cuda_speedup`
- Runtime Truth verdict during the same run: `alive`
- Source configuration: `benchmark_local_source`, local file source, `24` manual tick tokens

This proves the benchmark now surfaces device placement honestly during a locally configured Runtime Truth `alive` run. It does not prove CUDA execution, CUDA parity, or CUDA speedup.

## Links

- [Runtime Truth](../concepts/runtime-truth.md)
- [Capability Claim](../concepts/capability-claim.md)
- [Hot Path Latency](../benchmarks/hot-path-latency.md)
