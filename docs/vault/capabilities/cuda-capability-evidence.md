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

On 2026-06-10, Runtime Truth status surfaces were updated to expose the same claim boundary directly under `runtime_truth.evidence.runtime_device`, with compatibility fields `device`, `cuda_available`, and `observed_cuda_execution`. Local live status after quick-start reported `device=cpu`, `cuda_available=false`, `observed_cuda_execution=false`, and claim boundary `observed_device_placement_only_not_cuda_speedup`. This is observed CPU placement evidence only; it is not a CUDA acceleration claim.

Later on 2026-06-10, the local Python environment was corrected from `torch 2.11.0+cpu` to `torch 2.11.0+cu128` using the official PyTorch CUDA wheel index. Direct probes reported NVIDIA GeForce RTX 3060 driver visibility, `torch.cuda.is_available() == true`, one CUDA device, CUDA runtime `12.8`, and a tensor matmul probe on `cuda:0`. After restarting the MARULHO service and applying `/terminus/quick-start?preset=curriculum`, live Runtime Truth reported `resolved_device=cuda`, `tensor_device=cuda`, `routing_search_device=cuda`, `cuda_available=true`, `observed_cuda_execution=true`, and claim boundary `observed_cuda_execution_only_not_cuda_speedup`. A short follow-up status sample reported `configured=true`, `running=true`, `tick_count=2`, `background_tokens_processed=128`, and about `2.21` tokens/second. This proves local CUDA execution and runtime placement, but it still does not prove CUDA parity, CUDA throughput improvement, or broad GPU coverage across every slow path.

A paired CPU/CUDA service benchmark was then added under the explicit evaluation slow path. The 2026-06-10 local report at ignored `reports/service_benchmark_device_compare/device-comparison.json` forced `MARULHO_DEVICE=cpu` and `MARULHO_DEVICE=cuda` in separate runs. It passed endpoint success-name parity and observed CUDA execution, but the comparison status is `failed` because CUDA hot-path latency exceeded budget: CPU p95 `574.0 ms` and total `1012.343 ms`; CUDA p95 `2216.148 ms` and total `3272.994 ms`. This is evidence of CUDA placement, not CUDA speedup. The next CUDA work should isolate cold-start/warmup cost and small-tensor Python overhead before promoting CUDA as a hot-path throughput improvement.

## Links

- [Runtime Truth](../concepts/runtime-truth.md)
- [Capability Claim](../concepts/capability-claim.md)
- [Hot Path Latency](../benchmarks/hot-path-latency.md)
