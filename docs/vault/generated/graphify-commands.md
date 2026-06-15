---
type: generated
source: graphify
generated_by: "manual command log from vault build"
last_verified: 2026-06-15
---

# Graphify Command Notes

## Commands Run

- `python -m pip show graphify graphifyy`
  - Result: packages were not installed initially.
- `graphify --help`
  - Result: command not found on PATH before installation.
- `python -m pip install graphifyy`
  - Result: installed `graphifyy` and `graphify.exe`; script directory is not on PATH.
- `"C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" --help`
  - Result: command list includes `update`, `query`, `path`, `explain`, `tree`, `extract`, and platform setup commands.
- `"C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" update . --no-cluster`
  - Result: timed out after 3 minutes in this shell, but produced local ignored `graphify-out/graph.json`, manifest, and AST cache.
- `"C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" query "How do runtime truth, service manager, replay, language readout, and CUDA evidence connect in MARULHO?" --graph graphify-out/graph.json --budget 1400`
  - Result: traversed README orientation nodes from `MARULHO`.
- `"C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" explain "RuntimeFacade" --graph graphify-out/graph.json`
  - Result: found RuntimeFacade with 45 connections including manager/service dependencies.
- `"C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" path "MarulhoServiceManager" "RuntimeEvidenceReporter" --graph graphify-out/graph.json`
  - Result: one-hop inferred relation.

## Obsidian CLI

- `obsidian --help`
  - Result: not on PATH in this PowerShell session.
- `"C:\Users\thiag\AppData\Local\Programs\Obsidian\Obsidian.com" --help`
  - Result: Obsidian CLI is available by full path.

## Update Workflow

Prefer:

```powershell
& "C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" update . --no-cluster
python scripts/build_vault.py
python scripts/validate_vault_links.py
```

Use `--force` only after large refactors or stale graph state.

## Git Policy

`graphify-out/` is intentionally ignored. Commit curated vault notes, generated Markdown summaries, and scripts; do not commit Graphify JSON/cache/HTML output by default.
