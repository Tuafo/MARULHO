# Config

Use this with [../../../README.md](../../../README.md) and
[../../../CONTEXT.md](../../../CONTEXT.md).

`config` owns model/runtime configuration objects and presets.

## Owns

- Default model and runtime settings.
- Preset construction and normalization.
- Configuration fields that let tests and runners select CPU/CUDA behavior.

## Must Not Own

- Capability proof. Configured CUDA intent is not CUDA evidence.
- Compatibility aliases that obscure the current brain, service, or executor
  owner.
- Runtime mutation decisions.

## Runtime Rules

Promoted defaults should be tied to measured evidence. The current service
source tick width and execution quantum are evidence-backed defaults, not
biology claims or permanent compatibility promises. When a default changes,
record the evidence in `CONTEXT.md` or the relevant package README.
