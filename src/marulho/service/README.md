# Service

The service package is a thin HTTP/UI adapter over one `MarulhoBrain`.

## Files

- `brain_manager.py`: loads, saves, restores, and delegates to the brain.
- `api.py`: the maintained `/brain/*` FastAPI routes.
- `api_schemas.py`: checkpoint request/response shapes.
- `server.py`: local uvicorn entry point.

The service exposes status, traces, saved evidence, feed, tick, generation,
replay/growth hooks, checkpoints, and start/stop. It does not implement neural
algorithms, language decoding, replay policy, memory selection, structural
plasticity, or autonomous action loops.

Status and evidence reads are non-mutating. Model and tokenizer state live in
`MarulhoBrain`; training and evaluation live in their owning packages.

The retired service-owned cognition stack, SNN language ledger, readout
plasticity executor, autonomy loop, and compatibility facade were deleted.

Focused validation:

```powershell
python -m pytest -q tests/test_marulho_brain.py
```
