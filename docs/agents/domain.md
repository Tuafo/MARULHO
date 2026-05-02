# Domain docs

This repo uses a **single-context** layout.

## Layout

| File | Purpose |
|------|---------|
| `CONTEXT.md` | Project domain language, key concepts, and terminology |
| `docs/adr/` | Architecture Decision Records (ADRs) |

## Consumer rules

1. **Read `CONTEXT.md` first** — before modifying any domain logic, read this file to learn the project's vocabulary and mental model.
2. **Read ADRs before changing architecture** — check `docs/adr/` for past decisions that may be relevant. Do not contradict a past ADR without writing a new one that supersedes it.
3. **Update `CONTEXT.md` when terminology evolves** — if a concept is renamed or introduced, update the file so future agents stay aligned.
4. **Write ADRs for significant decisions** — any non-trivial architectural choice should be recorded as an ADR in `docs/adr/`.
