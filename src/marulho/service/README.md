# Service

Use this with [../../../README.md](../../../README.md) and
[../../../CONTEXT.md](../../../CONTEXT.md).

`service` is a thin HTTP/UI adapter over `MarulhoBrain`.

## Owns

- FastAPI route registration.
- UI-facing status stream, start/stop, feed, tick, generate, replay,
  grow/prune, checkpoints, and traces through `/brain/*`.
- Runtime composition for the active `MarulhoBrain` checkpoint.

## Must Not Own

- Neural algorithms, CUDA execution policy, or readout learning.
- Hidden replay, consolidation, or delayed-consequence work inside status reads.
- Old service-owned route/schema/readiness families kept for compatibility.

## Active Contract

The active API surface is `/health`, `/`, and `/brain/*`. `create_app()` builds
`MarulhoBrainServiceManager`, not the legacy `MarulhoServiceManager`. Legacy `/status`,
`/feed`, `/query`, `/respond`, root checkpoint aliases, `/terminus/*`, `/traces`,
`/stream/status`, and `/datasets` are retired from active FastAPI routing.

`/brain/start` and `/brain/stop` call the brain-owned lifecycle loop. They must
not return or depend on legacy Terminus runtime-control payloads.

`api_schemas.py` contains only the active checkpoint request/response models.
The old giant service schema module is deleted. The legacy manager remains
quarantined for old offline tests and runners, but it is not the HTTP/UI spine.
It is also not exported from `marulho.service`; callers must use `create_app`
or `MarulhoBrainServiceManager` for active service integration.
`tests/test_service_manager.py` is module-skipped during the spine refactor;
port only still-useful machinery assertions into package-local brain/core/data
tests instead of reactivating the whole Terminus service suite.

## Ported Guidance

- Status and Runtime Truth reads must remain read-only.
- Checkpoint writes/restores and trace persistence are explicit slow paths.
- Runtime scope/status caches may keep the control room responsive, but they
  are not model memory, mutation state, scheduler state, or CUDA speed evidence.
- Service may offer encoded quanta and lifecycle controls; trainer owns burst
  algorithms and mutation semantics.
