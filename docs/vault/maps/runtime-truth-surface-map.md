---
type: map
status: active
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

# Runtime Truth Surface Map

Capability and liveness surfaces that must remain evidence summaries rather than execution commands.

```mermaid
flowchart TD
    RuntimeTruth[Runtime Truth] --> Liveness[runtime liveness]
    RuntimeTruth --> CUDA[CUDA capability evidence]
    RuntimeTruth --> Replay[replay status]
    RuntimeTruth --> Readout[language-readout status]
    RuntimeTruth --> Growth[growth/pruning status]
    RuntimeTruth --> Action[action/tool-loop status]
    RuntimeTruth --> Memory[memory/consolidation status]
    RuntimeTruth --> ColumnScheduler[column scheduler metabolism]
    Replay --> Gates[read-only gates]
    Readout --> Gates
    Growth --> Gates
    ColumnScheduler --> Gates
    Gates --> Operators[operator review]
    Operators --> Executors[checkpoint-backed executors]
```

## Links

- [Runtime Truth](../concepts/runtime-truth.md)
- [Column Runtime](../concepts/column-runtime.md)
- [Subcortex](../concepts/subcortex.md)
- [Code organization](code-organization-map.md)
- [Capability notes](../capabilities/index.md)
- [Generated graph summary](../generated/graph-summary.md)
