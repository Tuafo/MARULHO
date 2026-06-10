---
type: benchmark
status: draft
related_code:
  - ../../../src/marulho/evaluation/service_benchmark.py
  - ../../../tests/test_service_benchmark.py
related_docs: []
related_papers: []
related_benchmarks: []
---

# Hot Path Latency

Latency-sensitive runtime surface checks.

## Commands

- Search tests: `rg -n "hot|path|latency" tests src`
- Focused tests: `python -m pytest tests\test_service_benchmark.py`
- Local benchmark:
  `powershell -Command "$env:PYTHONPATH='src'; python -m marulho.evaluation.service_benchmark --checkpoint reports\service_benchmark_cycle_configured\tiny.pt --output reports\service_benchmark_cycle_configured\service-benchmark.json --trace-dir reports\service_benchmark_cycle_configured\traces --env-root reports\service_benchmark_cycle_configured --web-dist-dir MARULHO_UI\dist --create-synthetic-checkpoint --configure-local-source --local-source-tick-steps 1"`
- Regression gate:
  `powershell -Command "$env:PYTHONPATH='src'; python -m marulho.evaluation.service_benchmark --compare-before reports\service_benchmark_cycle_configured\service-benchmark.json --compare-after reports\service_benchmark_cycle_configured\service-benchmark.json --output reports\service_benchmark_regression_gate\comparison.json"`
- Accept configured benchmark as a reviewed baseline:
  `powershell -Command "$env:PYTHONPATH='src'; python -m marulho.evaluation.service_benchmark --accept-baseline-from reports\service_benchmark_cycle_configured\service-benchmark.json --accepted-by codex-local-cycle --baseline-label configured-source-cpu-2026-06-09 --baseline-note \"Accepted local configured-source CPU benchmark for regression-gate smoke comparison.\" --output reports\service_benchmark_baseline\accepted-baseline.json"`
- Compare a run against the accepted baseline:
  `powershell -Command "$env:PYTHONPATH='src'; python -m marulho.evaluation.service_benchmark --compare-baseline reports\service_benchmark_baseline\accepted-baseline.json --compare-after reports\service_benchmark_cycle_configured\service-benchmark.json --output reports\service_benchmark_baseline\comparison.json"`
- Run a fresh configured-source benchmark and compare it against the accepted baseline:
  `powershell -Command "$env:PYTHONPATH='src'; python -m marulho.evaluation.service_benchmark --run-against-baseline reports\service_benchmark_baseline\accepted-baseline.json --checkpoint reports\service_benchmark_cycle_configured\tiny.pt --output reports\service_benchmark_baseline_fresh_cycle --trace-dir reports\service_benchmark_baseline_fresh_cycle\traces --web-dist-dir MARULHO_UI\dist --configure-local-source --local-source-tick-steps 1"`

## Latest Known Result

Measured on 2026-06-09 with a tiny synthetic checkpoint, a generated local file source, and one manual Terminus tick on the local CPU test environment. Raw JSON lives under ignored `reports/service_benchmark_cycle_configured/service-benchmark.json`.

- Benchmark success: `true`
- Total endpoint sweep latency: `1466.864 ms`
- Setup endpoints: `terminus_configure`, `terminus_tick`
- Setup total latency: `452.745 ms`
- Setup evidence: `24` tick tokens processed from `benchmark_local_source`; setup is marked `not_hot_path=true`
- Hot-path endpoints: `feed`, `query`, `respond`
- Hot-path total latency: `818.798 ms`
- Hot-path p95 latency: `439.258 ms`
- Hot-path budget verdict: within `1000.0 ms` p95 and `3000.0 ms` total budgets
- Regression gate status: `passed`
- Regression tolerance: `25%`
- Regression gate allowed after p95: `549.072 ms`
- Regression gate allowed after total: `1023.497 ms`
- Regression gate grouping: setup and slow-path endpoints did not leak into hot path
- Accepted baseline status: `accepted`
- Accepted baseline id: `service-benchmark-baseline:cc257251119a5335`
- Accepted baseline label: `configured-source-cpu-2026-06-09`
- Accepted baseline report hash: `cc257251119a53356ead486f456bfce7cd950337877d12d2b2add7140dc97645`
- Baseline comparison status: `passed`; the compared after-report hash matched the accepted baseline hash in this smoke comparison
- Fresh baseline-run bundle status: `passed`
- Fresh baseline-run bundle path: `reports/service_benchmark_baseline_fresh_cycle/bundle-summary.json`
- Fresh run total endpoint sweep latency: `1368.191 ms`
- Fresh run hot-path p95 latency: `432.406 ms`
- Fresh run hot-path total latency: `739.666 ms`
- Fresh run comparison bounds: allowed p95 `549.072 ms`, allowed total `1023.497 ms`
- Fresh run comparison hashes: baseline `cc257251119a53356ead486f456bfce7cd950337877d12d2b2add7140dc97645`, after `9a15573b07e26727fa256d8e4841de791f73117ad49abc53852f3bb28c6bc45c`
- Validation discovery: `/terminus/validation/reports` recognizes `marulho_service_benchmark_regression_gate`, `marulho_service_benchmark_accepted_baseline`, and `marulho_service_benchmark_baseline_run_bundle`, exposing Runtime Truth, hot-path budget, grouping, configured-source, accepted-baseline identity, source report hash, baseline snapshot hash, operator acceptance hash, fresh-run hashes, integrity statuses, evidence freshness, failed-check summary fields, and inert operator action hints for review
- Dashboard visibility: the Validation UI renders the latest regression gate, fresh benchmark bundle, and accepted benchmark baseline as separate operator cards. The benchmark cards show freshness status and evidence age, alerting on stale or unknown timestamps. The bundle card shows status, Runtime Truth, fresh hot-path p95/total, allowed p95/total, configured-source ticks, baseline hash, fresh hash, bundle paths, and failed checks. The baseline card shows status, snapshot integrity, approval integrity, baseline id, label, reviewer, Runtime Truth, baseline hot-path p95/total, source report hash, snapshot hash, acceptance hash, action hint, command templates, and failed checks.
- UI metabolism evidence: the Neural Space route now opens through a lightweight telemetry shell and loads the WebGL canvas only after an explicit operator action. The 2026-06-10 `npm run build` output reports `NeuralSpace3D` at `4.44 kB` minified / `1.89 kB` gzip, with the visual slow path isolated as `NeuralSpaceCanvas` (`20.45 kB` / `6.58 kB` gzip), `r3f` (`487.78 kB` / `160.03 kB` gzip), and `three` (`724.38 kB` / `187.35 kB` gzip). The remaining Vite chunk warning is therefore retained as an explicit visual-tooling cost, not routine dashboard startup or runtime evidence-path cost.
- Status sidecar total latency: `92.49 ms`
- Slow-path replay/export/dataset total latency: `86.397 ms`
- Runtime Truth verdict: `alive`
- Runtime Truth recommended action: `continue_monitoring`
- Source configuration evidence: `configured=true`, `source_count=1`, `source_names=["benchmark_local_source"]`
- Feed evidence: `44` tokens processed, `lexical_rolling_segments`, `7` sampled concept observations
- Device evidence: `tensor_device=cpu`, `encoder_device=cpu`, `routing_search_device=cpu`, `cuda_available=false`

## Interpretation

The benchmark now emits `endpoint_metabolism_summary`, a non-runtime evidence field that separates setup work, hot-path service endpoints, status sidecars, and explicit replay/export/dataset slow paths. This protects the always-on runtime by measuring setup and slow tooling without making them part of the hot path.

The benchmark also emits `runtime_device_evidence`, a compact observed-device summary from `/status` and `/terminus`. This run proves local configured-source liveness for the benchmark harness, but it does not prove CUDA acceleration: the focused test environment observed CPU placement and CUDA was unavailable.

The regression gate is a report-only evaluator. It compares benchmark JSON artifacts for Runtime Truth regression, configured-source liveness, absolute hot-path budgets, relative hot-path latency regression, and endpoint grouping boundaries. It does not mutate runtime state or claim speedup.

Benchmark evidence freshness is a validation projection only. The service API classifies benchmark reports from their `generated_at` timestamp as `fresh` through 24 hours, `aging` through 72 hours, `stale` after 72 hours, or `unknown_timestamp` when the timestamp is missing or unparsable. This status does not rerun benchmarks, alter Runtime Truth, or mutate baselines; it tells operators whether the saved hot-path evidence is current enough to rely on without a new slow-path run.

Runtime Truth now also exposes Benchmark Evidence Currency as advisory read-only evidence under `runtime_truth.evidence.benchmark_evidence_currency`. It scans the reports directory for the latest accepted baseline, fresh benchmark bundle, and regression gate report, then reports whether that saved evidence is `current`, `missing`, `stale`, or `failed`. This status never changes the Runtime Truth verdict and never runs benchmark work from the status path; it only prevents operators from mistaking absent or stale benchmark files for current hot-path evidence.

The accepted baseline manifest is also report-only. It records operator review metadata, a canonical JSON hash over the accepted benchmark snapshot, and a separate canonical JSON hash over the operator acceptance material. The acceptance hash binds reviewer id, note, acceptance time, baseline id/label, source report hash, Runtime Truth verdict, and hot-path p95/total into the manifest so later comparison and validation surfaces can detect review metadata drift. This is not a cryptographic identity signature; it is deterministic tamper evidence for the local slow-path report. Creating or comparing a baseline does not start the service, run replay, apply plasticity, write checkpoints, or claim CUDA acceleration.

The baseline-run bundle is the one-command slow path for this workflow. It runs a fresh configured-source service benchmark, writes `fresh-benchmark.json`, compares it against the accepted baseline, writes `comparison.json`, and records a compact `bundle-summary.json`. The validation API summarizes that bundle as read-only operator evidence, including fresh-run hot-path metrics, Runtime Truth, accepted-baseline identity, report hashes, configured-source ticks, paths, and failed checks. The bundle remains evaluation tooling; it is not an always-on runtime loop.

The service validation report endpoint now summarizes regression gate artifacts without moving the comparison logic into `service`. The API remains a read-only projection over reports generated by the explicit evaluation slow path.

The service validation report endpoint now also summarizes accepted baseline artifacts directly and recomputes the embedded snapshot hash and operator acceptance hash. A snapshot mismatch is surfaced as `baseline_integrity_status=failed` with `baseline_snapshot_hash_match` in failed checks. An approval-material mismatch is surfaced as `acceptance_integrity_status=failed` with `baseline_acceptance_hash_match` in failed checks. Older baselines without acceptance hashes are reported as `acceptance_integrity_status=legacy_unbound` and should be re-accepted before becoming durable comparison anchors. These are read-only operator evidence surfaces, not repair or mutation paths. The dashboard consumes the same report projection and does not call benchmark or replay work itself, so UI visibility adds no runtime hot-path cost.
