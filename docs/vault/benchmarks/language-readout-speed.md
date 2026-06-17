---
type: benchmark
status: draft
related_code: []
related_docs: []
related_papers: []
related_benchmarks: []
---

            # Language Readout Speed

            Bounded language/readout surface speed checks.

            ## Commands

            - Search tests: `rg -n "language|readout|speed" tests src`
            - Bounded corpus evaluator: `python -m marulho.evaluation.snn_language_readout_corpus --input <corpus.json> --output reports/snn_language_readout_corpus/readout-corpus-evaluation.json --top-k 4`
            - Focused tests: `python -m pytest tests/test_snn_language_readout_corpus_evaluation.py tests/test_status_read_model.py::StatusReadModelSNNLanguageReadoutCorpusEvaluationTests -q`
            - Full tests: `pytest`

            ## Latest Known Result

            Current slice adds slow-path latency and memory/VRAM reporting to the corpus evaluator. It does not enter the hot path. Long-run service throughput was checked with the existing continuous runtime benchmark before treating the report as operator-review evidence.

            `reports/snn_language_readout_corpus/post-readout-corpus-long-run-65536-262144-i32.json` processed `262144` tokens on `NVIDIA GeForce RTX 3060` at `6307.305 tokens/sec`, above the same-day active-pressure `65536`-column baseline at `6297.455 tokens/sec`. The run reported `velocity_environment.contention.verdict=not_observed`, `train_compute=0.129545 ms/token`, `prepare_training=0.006152 ms/token`, `finalize_total=0.006144 ms/token`, `route_input_rows_scored=12/65536`, `state_transition_runs_all_columns=false`, `native_sequence_loop_failure_count=0`, and `native_sequence_loop_fallback_count=0`. This preserves the current throughput band while leaving the corpus evaluator slow-path only.
