---
type: map
status: active
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

# Codex Vault Workflow

Use this vault as a context-retrieval and navigation layer. The repository remains the source of truth.

## Start Here

1. Read [index](index.md), [system map](maps/marulho-system-map.md), and [code organization map](maps/code-organization-map.md).
2. Check [CONTEXT.md](../../CONTEXT.md) before naming or widening MARULHO concepts.
3. Check [ADRs](adrs/index.md) before changing service/runtime ownership boundaries.
4. Use [open questions](questions/index.md) to preserve uncertainty instead of inventing unsupported claims.

## Graphify Use

`graphify-out/` is a local generated cache and is git-ignored. Rebuild it when architecture questions need graph evidence:

```powershell
& "C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" update . --no-cluster
& "C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" query "<question>" --graph graphify-out/graph.json --budget 1400
& "C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" explain "<node>" --graph graphify-out/graph.json
& "C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" path "<source>" "<target>" --graph graphify-out/graph.json
python scripts/build_vault.py
```

Keep only compact generated Markdown summaries under [generated](generated/README.md). Do not commit `graphify-out/graph.json`, cache files, or HTML visualizations unless a future ADR says otherwise.

## Update Rule

Curated notes should be stable and short. Generated notes may be overwritten. When a generated result matters for a decision, summarize the implication in a curated note and link to the source code, docs, ADR, or benchmark that proves it.
