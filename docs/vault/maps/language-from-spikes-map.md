---
type: map
status: active
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

        # Language From Spikes Map

        Readout, prediction, rollout, replay, and promotion gates for MARULHO-owned language.

        ## Current Evaluation Path

        `src/marulho/evaluation/snn_language_readout_corpus.py` evaluates bounded next-readout trajectories over explicit corpus windows and writes `snn_language_readout_corpus_evaluation.v1` reports. It can also write `snn_language_readout_corpus_checkpoint_review.v1` reports with an isolated sparse transition checkpoint and rollback manifest. `StatusReadModel` projects the latest saved reports into Runtime Truth without running evaluation, training, or mutating runtime state.


        ## Links

        - [Runtime Truth](../concepts/runtime-truth.md)
        - [Subcortex](../concepts/subcortex.md)
        - [Language from Spikes](../concepts/language-from-spikes.md)
        - [Language readout status](../capabilities/language-readout-status.md)
        - [Language readout speed](../benchmarks/language-readout-speed.md)
        - [Code organization](code-organization-map.md)
        - [Generated graph summary](../generated/graph-summary.md)
