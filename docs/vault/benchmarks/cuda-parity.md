---
type: benchmark
status: draft
related_code:
  - ../../../src/marulho/evaluation/service_benchmark.py
  - ../../../tests/test_service_benchmark.py
related_docs:
  - ../capabilities/cuda-capability-evidence.md
related_papers: []
related_benchmarks:
  - hot-path-latency.md
---

# Cuda Parity

CPU/CUDA parity and observed-device checks.

## Commands

- Focused tests: `python -m pytest tests\test_service_benchmark.py -q`
- Paired device benchmark:
  `powershell -Command "$env:PYTHONPATH='src'; python -m marulho.evaluation.service_benchmark --compare-devices --checkpoint reports\service_benchmark_device_compare\tiny.pt --output reports\service_benchmark_device_compare --web-dist-dir MARULHO_UI\dist --configure-local-source --local-source-tick-steps 1"`

## Latest Known Result

Measured on 2026-06-10 after correcting the local PyTorch install to `torch 2.11.0+cu128`. Raw JSON lives under ignored `reports/service_benchmark_device_compare/device-comparison.json`.

- Artifact: `marulho_service_benchmark_device_comparison`
- Status: `failed`
- CPU report success: `true`
- CUDA report success: `true`
- CPU Runtime Truth: `alive`
- CUDA Runtime Truth: `alive`
- CPU observed CUDA execution: `false`
- CUDA observed execution: `true`
- Endpoint success-name parity: `true`
- Parity scope: endpoint success names only, not semantic output equivalence
- CPU hot-path p95: `574.0 ms`
- CPU hot-path total: `1012.343 ms`
- CUDA hot-path p95: `2216.148 ms`
- CUDA hot-path total: `3272.994 ms`
- CPU hot-path budget: passed
- CUDA hot-path budget: failed
- CUDA minus CPU p95 delta: `1642.148 ms`
- CUDA minus CPU total delta: `2260.651 ms`
- Claim boundary: `observed_cpu_cuda_device_and_latency_delta_only_not_cuda_speedup_claim_without_repeated_parity_runs`

## Interpretation

This proves the current local benchmark harness can force CPU and CUDA runs, capture observed device placement, and compare hot-path latency without adding work to the always-on runtime.

It does not prove CUDA speedup. On this small configured-source service benchmark, CUDA was slower than CPU and exceeded the existing hot-path total budget. The likely next work is to separate CUDA warmup/cold-start cost from steady-state tick/feed/query/respond latency, then identify whether Python control overhead or small tensor size dominates this path.
