---
type: concept
status: active
related_code:
  - ../../../MARULHO_UI/src/components/MachineryWorkspace.jsx
related_docs:
  - ../../../CONTEXT.md
  - ../../retired-paths.md
related_papers: []
related_benchmarks: []
---

# Runtime Machinery Plane

The Runtime Machinery Plane is the UI's read-only spatial projection of current MARULHO telemetry.

The center is a sparse predictive-column field: observed awake votes are explicit points inside a larger registered-column field. Grounded input and spike encoding enter from the left, context and binding evidence sit above, sparse memory and replay remain below as explicit slow-path machinery, and column consensus/readout exits to the right.

Motion is allowed only while Runtime Truth reports active work. Dormant points communicate registered capacity, not executed columns. Replay, topology feedback, and sleeping-column behavior stay visually subdued unless promoted telemetry proves they are active.

The plane does not schedule columns, mutate state, run replay, prove CUDA speed, or establish liveness. Those claims remain owned by Runtime Truth and benchmark evidence.

## Links

- [Column Runtime](column-runtime.md)
- [Runtime Truth](runtime-truth.md)
- [Runtime Evidence](runtime-evidence.md)
- [Service](../modules/service.md)
