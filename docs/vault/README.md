---
type: map
status: active
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

# MARULHO Knowledge Vault

This is a plain Markdown vault for MARULHO. It is Obsidian-compatible, but repository documentation work must not depend on Obsidian being open.

Start at [index](index.md).

## Workflow

- Rebuild Graphify output: `"C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\graphify.exe" update . --no-cluster`
- Rebuild generated vault notes: `python scripts/build_vault.py`
- Validate vault links/frontmatter: `python scripts/validate_vault_links.py`
- Optional Obsidian CLI: `"C:\Users\thiag\AppData\Local\Programs\Obsidian\Obsidian.com" files vault=vault`

## Source Of Truth

The repository remains the source of truth. Curated vault notes summarize and link to source docs/code. Generated notes live under [generated](generated/README.md) and may be overwritten.
