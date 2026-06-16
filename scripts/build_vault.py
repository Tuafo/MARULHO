"""Build the MARULHO Markdown/Obsidian vault.

The script is intentionally conservative:
- generated files under docs/vault/generated are overwritten;
- curated starter notes are created only when missing;
- source docs remain the source of truth and are linked, not copied.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path
from textwrap import dedent


ROOT = Path(__file__).resolve().parents[1]
VAULT = ROOT / "docs" / "vault"
GENERATED = VAULT / "generated"
GRAPH = ROOT / "graphify-out" / "graph.json"
TODAY = date.today().isoformat()


GRAPHIFY_EXE = (
    r"C:\Users\thiag\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0"
    r"\LocalCache\local-packages\Python311\Scripts\graphify.exe"
)
OBSIDIAN_EXE = r"C:\Users\thiag\AppData\Local\Programs\Obsidian\Obsidian.com"


def write(path: Path, content: str, *, overwrite: bool = True) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return False
    normalized = dedent(content).strip()
    normalized = "\n".join(line[8:] if line.startswith("        ") else line for line in normalized.splitlines())
    normalized += "\n"
    if path.exists() and path.read_text(encoding="utf-8") == normalized:
        return False
    path.write_text(normalized, encoding="utf-8")
    return True


def rel(path: str) -> str:
    return path.replace("\\", "/")


def frontmatter(note_type: str, status: str = "draft") -> str:
    return f"""---
type: {note_type}
status: {status}
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---"""


def generated_frontmatter(command: str) -> str:
    return f"""---
type: generated
source: graphify
generated_by: "{command}"
last_verified: {TODAY}
---"""


def load_graph() -> tuple[list[dict], list[dict]]:
    if not GRAPH.exists():
        return [], []
    data = json.loads(GRAPH.read_text(encoding="utf-8"))
    return data.get("nodes", []), data.get("links", data.get("edges", []))


def graph_stats(nodes: list[dict], edges: list[dict]) -> dict:
    by_source = Counter(rel(str(n.get("source_file", ""))) for n in nodes if n.get("source_file"))
    by_package: Counter[str] = Counter()
    for source, count in by_source.items():
        parts = source.split("/")
        if len(parts) >= 3 and parts[0] == "src" and parts[1] == "marulho":
            by_package[parts[2]] += count
        elif parts and parts[0] == "tests":
            by_package["tests"] += count
        elif parts and parts[0] == "MARULHO_UI":
            by_package["MARULHO_UI"] += count
    relation_counts = Counter(str(e.get("relation", "")) for e in edges)
    return {
        "files": by_source,
        "packages": by_package,
        "relations": relation_counts,
    }


MODULES = {
    "core": {
        "responsibility": "Local SNN mechanisms: columns, context, binding, abstraction, topography, plasticity, surprise, and sparsity.",
        "owns": "Tensor/state mechanisms and device-reportable substrate behavior.",
        "not_own": "Operator HTTP surfaces, persistence policy, or language-facing claims.",
        "concepts": ["Subcortex", "Metabolism", "Plasticity Gate", "Dynamic Growth", "Pruning", "CUDA Evidence"],
    },
    "data": {
        "responsibility": "Source loaders and sparse encoders for text, semantic, audio, event-camera, and multimodal evidence.",
        "owns": "Input normalization and emitted encoder/device evidence.",
        "not_own": "Runtime Truth verdicts or promotion of observations into facts/actions.",
        "concepts": ["Runtime Evidence", "CUDA Evidence", "Capability Claim"],
    },
    "semantics": {
        "responsibility": "Grounded language/readout contracts, cognitive signal surfaces, decoder probes, and concept evidence.",
        "owns": "Bounded readout artifacts and support/grounding diagnostics.",
        "not_own": "Free-form cognition, fact promotion, action authority, or external checkpoint loading.",
        "concepts": ["Spike Readout", "Language from Spikes", "Thought Trajectory", "Capability Claim"],
    },
    "service": {
        "responsibility": "FastAPI service, runtime composition, status projections, Runtime Truth, replay ledgers, persistence, and gated executors.",
        "owns": "Operator-facing runtime boundaries and evidence/mutation orchestration.",
        "not_own": "Low-level neural math that belongs in core or semantics.",
        "concepts": ["Runtime Truth", "Runtime Evidence", "Replay Window", "Retired Path"],
    },
    "training": {
        "responsibility": "Bootstrap, developmental, autonomy, consolidation, query, and long-run runners.",
        "owns": "Offline or explicitly invoked training/evaluation workflows.",
        "not_own": "Live runtime mutation authority without service gates.",
        "concepts": ["Replay Window", "Plasticity Gate", "Metabolism"],
    },
    "evaluation": {
        "responsibility": "Promotion gates, benchmarks, readiness checks, and validation harnesses.",
        "owns": "Evidence standards for speed, readiness, CUDA placement, liveness, and promotion.",
        "not_own": "Runtime state mutation.",
        "concepts": ["Runtime Truth", "CUDA Evidence", "Capability Claim"],
    },
    "consolidation": {
        "responsibility": "Archival memory store and consolidation records.",
        "owns": "CPU-resident archival evidence and explicit memory records.",
        "not_own": "Device-local replay computation or live plasticity application.",
        "concepts": ["Replay Window", "Runtime Evidence"],
    },
    "retrieval": {
        "responsibility": "Exact torch-cache routing, tensor candidate search, and decoder support.",
        "owns": "Lookup and routing experiments with explicit performance/device evidence and no retired backend language.",
        "not_own": "Claiming CUDA acceleration without observed telemetry.",
        "concepts": ["Hot Path", "Slow Path", "CUDA Evidence"],
    },
    "interaction": {
        "responsibility": "Responder and operator-facing answer formation.",
        "owns": "Grounded response shaping over evidence surfaces.",
        "not_own": "Hidden thought loops or unsupported fluency claims.",
        "concepts": ["Language from Spikes", "Runtime Evidence"],
    },
    "reporting": {
        "responsibility": "Reports, benchmark plots, validation summaries, and autonomy/readme report helpers.",
        "owns": "Human-readable evidence summaries.",
        "not_own": "Primary runtime truth or mutation decisions.",
        "concepts": ["Runtime Truth", "Capability Claim", "CUDA Evidence"],
    },
}


CONCEPTS = {
    "runtime-truth": ("Runtime Truth", "Operator-facing status that reports what the runtime can prove now, including liveness, device truth, and blocked mutation surfaces."),
    "runtime-evidence": ("Runtime Evidence", "Auditable observations, hashes, telemetry, ledgers, and status records used to support or reject capability claims."),
    "hot-path": ("Hot Path", "Latency-sensitive runtime path that should remain sparse, observable, and free of avoidable reporting or archival work."),
    "slow-path": ("Slow Path", "Review, reporting, archival, generation, graph, or research-memory work that should not sit inside runtime-critical execution."),
    "metabolism": ("Metabolism", "Runtime pressure signals such as fatigue, sleep pressure, spike health, memory pressure, and replay readiness."),
    "subcortex": ("Subcortex", "The grounded predictive spiking substrate. It owns sparse routing, multimodal grounding, prediction error, replay, curiosity pressure, and local plasticity."),
    "spike-readout": ("Spike Readout", "Read-only population-code evidence that bridges Subcortex state into bounded language/readout slots without generating free-form text."),
    "thought-trajectory": ("Thought Trajectory", "A bounded trajectory-like readout over sparse state; it is evidence for review, not proof of autonomous thought."),
    "replay-window": ("Replay Window", "A bounded, provenance-hashed window selected for review or isolated replay before any consolidation or plasticity authority."),
    "plasticity-gate": ("Plasticity Gate", "A read-only or operator-confirmed boundary that blocks learning/mutation until evidence, rollback, and revision checks align."),
    "dynamic-growth": ("Dynamic Growth", "Bounded structural addition driven by local evidence such as surprise, mismatch, replay failures, or concept pressure."),
    "pruning": ("Pruning", "Bounded structural removal or repair, evaluated by retained support, topology evidence, and rollback readiness."),
    "cuda-evidence": ("CUDA Evidence", "Observed tensor/device placement and backend telemetry, stronger than configured intent and required before CUDA claims."),
    "language-from-spikes": ("Language from Spikes", "The research direction of MARULHO-owned language surfaces over sparse SNN state, with grounding and promotion gates."),
    "retired-path": ("Retired Path", "Former external LLM/Cortex/ThoughtLoop paths that should remain absent from active runtime truth and action ledgers."),
    "capability-claim": ("Capability Claim", "Any public or internal statement about what MARULHO can do; it must be backed by Runtime Evidence or marked as intended/research-only."),
}


PAPERS = {
    "neuronspark-v1": ("NeuronSpark-V1", "Emerging SNN language reference; inspiration-only until MARULHO owns training, grounding, and checkpoints."),
    "nord-ai": ("Nord-AI", "Pure-SNN language/memory reference pressure; inspiration-only and not a runtime dependency."),
    "spikegpt": ("SpikeGPT", "Supports event-coded sequence framing for language, but not external checkpoint adoption."),
    "spiking-ssms": ("SpikingSSMs", "Supports sparse temporal state rollout as a readout direction."),
    "predictive-coding": ("Predictive coding", "Supports prediction error, confidence, and context-sensitive sequence learning as control evidence."),
    "active-inference": ("Active inference", "Supports advisory policy/control candidates separated from execution authority."),
    "structural-plasticity": ("Structural plasticity", "Supports bounded growth/pruning with explicit topology, device, and rollback evidence."),
    "replay-consolidation": ("Replay/consolidation", "Supports separated replay nomination, review, artifact recording, and mutation gates."),
    "cuda-triton-snn-optimization": ("CUDA/Triton SNN optimization", "Supports sparse GPU execution only when observed runtime tensors prove placement."),
    "quantum-inspired-routing-search-memory": ("Quantum-inspired routing/search/memory", "Research-only unless a MARULHO benchmark shows useful routing or memory behavior."),
}


MAPS = {
    "marulho-system-map": ("MARULHO System Map", "A linked overview of runtime evidence, Subcortex, service gates, and operator-facing surfaces."),
    "brain-region-map": ("Brain Region Map", "Project-language map from Subcortex mechanisms to modules without claiming biological equivalence."),
    "hot-path-map": ("Hot Path Map", "Latency-sensitive surfaces and where slow reporting/research workflows must stay out."),
    "language-from-spikes-map": ("Language From Spikes Map", "Readout, prediction, rollout, replay, and promotion gates for MARULHO-owned language."),
    "replay-plasticity-map": ("Replay Plasticity Map", "Replay windows, tickets, consolidation pressure, permits, and mutation boundaries."),
    "cuda-metabolism-map": ("CUDA Metabolism Map", "Device evidence, spike health, sleep pressure, and Runtime Truth visibility."),
    "code-organization-map": ("Code Organization Map", "How package areas and ADRs divide ownership."),
    "runtime-truth-surface-map": ("Runtime Truth Surface Map", "Capability and liveness surfaces that must remain evidence summaries rather than execution commands."),
}


def build_indexes(changed: list[str]) -> None:
    write(
        VAULT / "README.md",
        f"""
        {frontmatter("map", "active")}

        # MARULHO Knowledge Vault

        This is a plain Markdown vault for MARULHO. It is Obsidian-compatible, but repository documentation work must not depend on Obsidian being open.

        Start at [index](index.md).

        ## Workflow

        - Rebuild Graphify output: `"{GRAPHIFY_EXE}" update . --no-cluster`
        - Rebuild generated vault notes: `python scripts/build_vault.py`
        - Validate vault links/frontmatter: `python scripts/validate_vault_links.py`
        - Optional Obsidian CLI: `"{OBSIDIAN_EXE}" files vault=vault`

        ## Source Of Truth

        The repository remains the source of truth. Curated vault notes summarize and link to source docs/code. Generated notes live under [generated](generated/README.md) and may be overwritten.
        """,
    )
    write(
        VAULT / "index.md",
        f"""
        {frontmatter("map", "active")}

        # MARULHO Vault Index

        - [System map](maps/marulho-system-map.md)
        - [Concepts](concepts/index.md)
        - [Modules](modules/index.md)
        - [Papers](papers/index.md)
        - [ADRs](adrs/index.md)
        - [Capabilities](capabilities/index.md)
        - [Benchmarks](benchmarks/index.md)
        - [Retired paths](retired/index.md)
        - [Open questions](questions/index.md)
        - [Generated graph outputs](generated/README.md)
        - [Graphify commands](generated/graphify-commands.md)
        - [Codex vault workflow](codex-vault-workflow.md)

        ## Orientation

        MARULHO is a grounded Subcortex runtime for auditable autonomous cognition. The vault preserves navigation and research memory; it does not replace [CONTEXT.md](../../CONTEXT.md), [README.md](../../README.md), or ADRs.
        """,
    )
    write(
        VAULT / "codex-vault-workflow.md",
        f"""
        {frontmatter("map", "active")}

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
        & "{GRAPHIFY_EXE}" update . --no-cluster
        & "{GRAPHIFY_EXE}" query "<question>" --graph graphify-out/graph.json --budget 1400
        & "{GRAPHIFY_EXE}" explain "<node>" --graph graphify-out/graph.json
        & "{GRAPHIFY_EXE}" path "<source>" "<target>" --graph graphify-out/graph.json
        python scripts/build_vault.py
        ```

        Keep only compact generated Markdown summaries under [generated](generated/README.md). Do not commit `graphify-out/graph.json`, cache files, or HTML visualizations unless a future ADR says otherwise.

        ## Update Rule

        Curated notes should be stable and short. Generated notes may be overwritten. When a generated result matters for a decision, summarize the implication in a curated note and link to the source code, docs, ADR, or benchmark that proves it.
        """,
    )

    for folder, title in [
        ("concepts", "Concepts"),
        ("modules", "Modules"),
        ("papers", "Papers"),
        ("adrs", "ADRs"),
        ("capabilities", "Capabilities"),
        ("benchmarks", "Benchmarks"),
        ("retired", "Retired Paths"),
        ("questions", "Open Questions"),
        ("maps", "Maps"),
    ]:
        (VAULT / folder).mkdir(parents=True, exist_ok=True)
        if folder != "adrs":
            links = "\n".join(
                f"- [{p.stem.replace('-', ' ').title()}]({p.name})"
                for p in sorted((VAULT / folder).glob("*.md"))
                if p.name != "index.md"
            )
            write(
                VAULT / folder / "index.md",
                f"""
                {frontmatter("map", "active")}

                # {title}

                {links or "- No notes yet."}
                """,
            )


def build_curated_notes() -> list[str]:
    changed: list[str] = []

    for slug, (title, definition) in CONCEPTS.items():
        path = VAULT / "concepts" / f"{slug}.md"
        content = f"""
        {frontmatter("concept", "active")}

        # {title}

        ## Definition

        {definition}

        ## Relationships

        - [Subcortex](subcortex.md)
        - [Runtime Truth](runtime-truth.md)
        - [Runtime Evidence](runtime-evidence.md)

        ## Source Links

        - [CONTEXT.md](../../../CONTEXT.md)
        - [README.md](../../../README.md)
        - [Research notes](../../research-living-brain.md)

        ## Ambiguity

        Keep claims evidence-gated. Do not widen this term into a generic programming or biology concept without updating [CONTEXT.md](../../../CONTEXT.md).
        """
        if write(path, content, overwrite=False):
            changed.append(str(path.relative_to(ROOT)))

    for name, info in MODULES.items():
        path = VAULT / "modules" / f"{name}.md"
        concept_links = ", ".join(f"[{c}](../concepts/{c.lower().replace(' ', '-')}.md)" for c in info["concepts"])
        content = f"""
        {frontmatter("module", "active")}

        # marulho.{name}

        ## Responsibility

        {info["responsibility"]}

        ## Owns

        {info["owns"]}

        ## Should Not Own

        {info["not_own"]}

        ## Hot-Path Relevance

        Treat runtime-critical tensor/state work as hot path only when it is required for live service behavior. Reporting, vault generation, and research-memory work stay slow path.

        ## Key Files

        - [src/marulho/{name}](../../../src/marulho/{name})
        - [tests](../../../tests)

        ## Related Concepts

        {concept_links}

        ## Graphify

        - Query: `"{GRAPHIFY_EXE}" explain "{name}" --graph graphify-out/graph.json`
        - Generated module summary: [generated module index](../generated/module-index.md)
        """
        if write(path, content, overwrite=False):
            changed.append(str(path.relative_to(ROOT)))

    for slug, (title, summary) in PAPERS.items():
        path = VAULT / "papers" / f"{slug}.md"
        content = f"""
        {frontmatter("paper", "draft")}

        # {title}

        ## Claim

        {summary}

        ## MARULHO Relevance

        Use this as research pressure only when it supports a MARULHO-owned mechanism, evidence gate, benchmark, or rejection note.

        ## Implementation Implication

        Do not import external runtime code or checkpoints unless a future ADR explicitly accepts that dependency. Prefer local probes, heldout gates, and rollback-aware experiments.

        ## Status

        inspiration-only

        ## Links

        - [Research notes](../../research-living-brain.md)
        - [Language from Spikes](../concepts/language-from-spikes.md)
        - [CUDA Evidence](../concepts/cuda-evidence.md)
        """
        if write(path, content, overwrite=False):
            changed.append(str(path.relative_to(ROOT)))

    for slug, (title, summary) in MAPS.items():
        path = VAULT / "maps" / f"{slug}.md"
        mermaid = ""
        if slug == "marulho-system-map":
            mermaid = dedent("""
            ```mermaid
            flowchart LR
                Sources[Grounded sources] --> Data[data encoders]
                Data --> Subcortex[Subcortex runtime]
                Subcortex --> Semantics[semantics/readout]
                Subcortex --> Replay[replay and plasticity gates]
                Replay --> Truth[Runtime Truth]
                Semantics --> Truth
                Truth --> Operator[operator surfaces]
                Replay --> Checkpoints[checkpoint-backed executors]
                Checkpoints --> Subcortex
            ```
            """).strip()
        elif slug == "runtime-truth-surface-map":
            mermaid = dedent("""
            ```mermaid
            flowchart TD
                RuntimeTruth[Runtime Truth] --> Liveness[runtime liveness]
                RuntimeTruth --> CUDA[CUDA capability evidence]
                RuntimeTruth --> Replay[replay status]
                RuntimeTruth --> Readout[language-readout status]
                RuntimeTruth --> Growth[growth/pruning status]
                RuntimeTruth --> Action[action/tool-loop status]
                RuntimeTruth --> Memory[memory/consolidation status]
                Replay --> Gates[read-only gates]
                Readout --> Gates
                Growth --> Gates
                Gates --> Operators[operator review]
                Operators --> Executors[checkpoint-backed executors]
            ```
            """).strip()
        content = f"""
        {frontmatter("map", "active")}

        # {title}

        {summary}

        {mermaid}

        ## Links

        - [Runtime Truth](../concepts/runtime-truth.md)
        - [Subcortex](../concepts/subcortex.md)
        - [Code organization](code-organization-map.md)
        - [Capability notes](../capabilities/index.md)
        - [Generated graph summary](../generated/graph-summary.md)
        """
        if write(path, content, overwrite=False):
            changed.append(str(path.relative_to(ROOT)))

    capability_notes = {
        "cuda-capability-evidence": "CUDA claims are valid only when observed device/backend telemetry supports them.",
        "runtime-liveness": "Behavioral liveness is evidence-gated by runtime progress, Runtime Truth, and absence of active retired paths.",
        "replay-status": "Replay status is advisory until review, artifact, permit, and executor gates align.",
        "language-readout-status": "Language readout status covers grounded bounded surfaces, not free-form generation.",
        "growth-pruning-status": "Growth/pruning status is readiness evidence until checkpoint-backed mutation is confirmed.",
        "action-tool-loop-status": "Action/tool-loop status belongs to Subcortex action ledgers and verified execution evidence.",
        "memory-consolidation-status": "Memory/consolidation status separates CPU archival records from device-local replay computation.",
    }
    for slug, body in capability_notes.items():
        if write(
            VAULT / "capabilities" / f"{slug}.md",
            f"""
            {frontmatter("capability", "draft")}

            # {slug.replace('-', ' ').title()}

            {body}

            ## Evidence Rule

            Do not claim this capability as live unless linked Runtime Evidence or benchmark output supports it.

            ## Links

            - [Runtime Truth](../concepts/runtime-truth.md)
            - [Capability Claim](../concepts/capability-claim.md)
            """,
            overwrite=False,
        ):
            changed.append(f"docs/vault/capabilities/{slug}.md")

    benchmark_notes = {
        "hot-path-latency": "Latency-sensitive runtime surface checks.",
        "startup": "Service and runtime initialization checks.",
        "memory-lookup": "Retrieval and concept memory lookup checks.",
        "replay-cost": "Replay selection, rehearsal, and artifact-review cost checks.",
        "cuda-parity": "CPU/CUDA parity and observed-device checks.",
        "language-readout-speed": "Bounded language/readout surface speed checks.",
        "growth-pruning-cost": "Structural evaluation and mutation-preflight cost checks.",
        "tool-loop-throughput": "Action/tool-loop throughput and verification checks.",
    }
    for slug, body in benchmark_notes.items():
        if write(
            VAULT / "benchmarks" / f"{slug}.md",
            f"""
            {frontmatter("benchmark", "draft")}

            # {slug.replace('-', ' ').title()}

            {body}

            ## Commands

            - Search tests: `rg -n "{slug.replace('-', '|')}" tests src`
            - Full tests: `pytest`

            ## Latest Known Result

            Not measured in this vault pass. Link exact command output here when a benchmark is run.
            """,
            overwrite=False,
        ):
            changed.append(f"docs/vault/benchmarks/{slug}.md")

    write(
        ROOT / "docs" / "retired-paths.md",
        """
        # Retired Paths

        This file records paths that should not be revived without new evidence and, when appropriate, an ADR.

        | Path/name | Status | Why retired | Replacement | Revisit condition |
        | --- | --- | --- | --- | --- |
        | External LLM/Cortex/ThoughtLoop runtime path | retired | Added external dependency and ambiguous cognition claims without being the living substrate. | Subcortex-owned runtime evidence, semantics/readout surfaces, replay, and Runtime Truth. | A future ADR proves a bounded dependency does not become the cognition substrate and preserves evidence gates. |
        | Manager mixin compatibility aliases | retired | Preserved shallow ownership after the service split. | Explicit deep modules and RuntimeFacade/StatusReadModel ownership. | Only if a compatibility period is required by a released public API. |
        | Static CUDA intent as capability proof | retired | Configuration can hide CPU fallback. | Observed tensor/device evidence in Runtime Truth and gates. | Never as proof; static intent may remain diagnostic only. |
        """,
        overwrite=False,
    )
    write(
        VAULT / "retired" / "index.md",
        f"""
        {frontmatter("retired-path", "active")}

        # Retired Paths

        - [Repository retired paths](../../retired-paths.md)
        - [Retired Path concept](../concepts/retired-path.md)
        - [ADR 0005](../../adr/0005-cuda-first-subcortex-runtime.md)
        """,
    )
    write(
        VAULT / "questions" / "open-research-questions.md",
        f"""
        {frontmatter("question", "draft")}

        # Open Research Questions

        - What evidence is enough to move bounded SNN readout from review-only labels into a trained local generator?
        - Which replay windows most reliably improve sparse transition memory without weakening grounding support?
        - Which CUDA/Triton sparse SNN kernels preserve MARULHO's evidence requirements instead of hiding dense rewrites?
        - Which structural-plasticity signals best predict useful growth/pruning without language-planner intervention?
        """,
        overwrite=False,
    )
    write(
        VAULT / "questions" / "open-architecture-questions.md",
        f"""
        {frontmatter("question", "draft")}

        # Open Architecture Questions

        - Which selected Graphify summaries should be promoted into curated notes after repeated use?
        - Which Runtime Truth surfaces should get first-class vault capability notes once benchmark evidence is available?
        - Where should long-running benchmark results live so the vault can link to them without duplicating reports?
        - Should an ADR document the vault/Graphify workflow after a few iterations prove it stable?
        """,
        overwrite=False,
    )

    return changed


def build_adrs() -> None:
    lines = []
    for adr in sorted((ROOT / "docs" / "adr").glob("*.md")):
        title = adr.read_text(encoding="utf-8").splitlines()[0].lstrip("# ")
        lines.append(f"- [{title}](../../adr/{adr.name})")
    write(
        VAULT / "adrs" / "index.md",
        f"""
        {frontmatter("adr", "active")}

        # ADR Index

        {chr(10).join(lines)}

        ## Related Maps

        - [Code organization map](../maps/code-organization-map.md)
        - [CUDA metabolism map](../maps/cuda-metabolism-map.md)
        - [Replay plasticity map](../maps/replay-plasticity-map.md)
        """,
    )


def build_generated(nodes: list[dict], edges: list[dict]) -> None:
    stats = graph_stats(nodes, edges)
    package_lines = "\n".join(f"- `{k}`: {v} nodes" for k, v in stats["packages"].most_common())
    relation_lines = "\n".join(f"- `{k}`: {v}" for k, v in stats["relations"].most_common(12))
    source_lines = "\n".join(f"- `{k}`: {v} nodes" for k, v in stats["files"].most_common(20))
    write(
        GENERATED / "README.md",
        f"""
        {generated_frontmatter("python scripts/build_vault.py")}

        # Generated Vault Notes

        These files are generated and may be overwritten.

        - [Graph summary](graph-summary.md)
        - [Module index](module-index.md)
        - [Graphify commands](graphify-commands.md)
        """,
    )
    write(
        GENERATED / "graph-summary.md",
        f"""
        {generated_frontmatter(f'{GRAPHIFY_EXE} update . --no-cluster')}

        # Graphify Summary

        - Graph file: `graphify-out/graph.json` local generated cache, ignored by git.
        - Nodes: {len(nodes)}
        - Edges: {len(edges)}
        - Mode: AST/local graph update; clustering and graph HTML were not required for this pass.

        ## Package Node Counts

        {package_lines or "- Graph unavailable."}

        ## Common Relations

        {relation_lines or "- Graph unavailable."}

        ## Largest Source Files By Node Count

        {source_lines or "- Graph unavailable."}
        """,
    )

    module_sections = []
    for module in MODULES:
        files = [
            (source, count)
            for source, count in stats["files"].items()
            if source.startswith(f"src/marulho/{module}/")
        ]
        top = "\n".join(f"- `{source}`: {count} nodes" for source, count in sorted(files, key=lambda x: x[1], reverse=True)[:10])
        module_sections.append(f"## marulho.{module}\n\n{top or '- No Graphify nodes found.'}")
    write(
        GENERATED / "module-index.md",
        f"""
        {generated_frontmatter("python scripts/build_vault.py")}

        # Generated Module Index

        {chr(10).join(module_sections)}
        """,
    )
    write(
        GENERATED / "graphify-commands.md",
        f"""
        {generated_frontmatter("manual command log from vault build")}

        # Graphify Command Notes

        ## Commands Run

        - `python -m pip show graphify graphifyy`
          - Result: packages were not installed initially.
        - `graphify --help`
          - Result: command not found on PATH before installation.
        - `python -m pip install graphifyy`
          - Result: installed `graphifyy` and `graphify.exe`; script directory is not on PATH.
        - `"{GRAPHIFY_EXE}" --help`
          - Result: command list includes `update`, `query`, `path`, `explain`, `tree`, `extract`, and platform setup commands.
        - `"{GRAPHIFY_EXE}" update . --no-cluster`
          - Result: timed out after 3 minutes in this shell, but produced local ignored `graphify-out/graph.json`, manifest, and AST cache.
        - `"{GRAPHIFY_EXE}" query "How do runtime truth, service manager, replay, language readout, and CUDA evidence connect in MARULHO?" --graph graphify-out/graph.json --budget 1400`
          - Result: traversed README orientation nodes from `MARULHO`.
        - `"{GRAPHIFY_EXE}" explain "RuntimeFacade" --graph graphify-out/graph.json`
          - Result: found RuntimeFacade with 45 connections including manager/service dependencies.
        - `"{GRAPHIFY_EXE}" path "MarulhoServiceManager" "RuntimeEvidenceReporter" --graph graphify-out/graph.json`
          - Result: one-hop inferred relation.

        ## Obsidian CLI

        - `obsidian --help`
          - Result: not on PATH in this PowerShell session.
        - `"{OBSIDIAN_EXE}" --help`
          - Result: Obsidian CLI is available by full path.

        ## Update Workflow

        Prefer:

        ```powershell
        & "{GRAPHIFY_EXE}" update . --no-cluster
        python scripts/build_vault.py
        python scripts/validate_vault_links.py
        ```

        Use `--force` only after large refactors or stale graph state.

        ## Git Policy

        `graphify-out/` is intentionally ignored. Commit curated vault notes, generated Markdown summaries, and scripts; do not commit Graphify JSON/cache/HTML output by default.
        """,
    )


def main() -> None:
    nodes, edges = load_graph()
    changed = build_curated_notes()
    build_adrs()
    build_generated(nodes, edges)
    build_indexes(changed)
    print("Vault build complete.")
    print(f"Graph nodes: {len(nodes)}")
    print(f"Graph edges: {len(edges)}")
    if changed:
        print("Created curated starters:")
        for item in changed:
            print(f"- {item}")


if __name__ == "__main__":
    main()
