# Domain docs

This repo uses a **single-context plus package-local machinery docs** layout.

## Layout

| File | Purpose |
|------|---------|
| `CONTEXT.md` | Project domain language, key concepts, and terminology |
| `src/marulho/README.md` | Package-level machinery map |
| `src/marulho/*/README.md` | Local ownership rules and evidence boundaries |

## Consumer rules

1. **Read `CONTEXT.md` first** — before modifying any domain logic, read this file to learn the project's vocabulary and mental model.
2. **Read the nearest package README** — package-local docs carry the machinery-specific ownership rules.
3. **Update `CONTEXT.md` when terminology evolves** — if a concept is renamed or introduced, update the file so future agents stay aligned.
4. **Update package READMEs when machinery ownership changes** — keep docs close to the current code owners.
