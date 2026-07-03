from __future__ import annotations

import shutil
from pathlib import Path
from uuid import uuid4

from marulho.evaluation.service_benchmark import (
    create_tiny_service_benchmark_checkpoint,
    run_service_benchmark,
)


def test_service_benchmark_targets_brain_api_contract() -> None:
    root = Path("reports") / "brain_service_benchmark_tests" / uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    try:
        checkpoint_path = create_tiny_service_benchmark_checkpoint(root / "brain-benchmark.pt")
        output_path = root / "result.json"
        result = run_service_benchmark(
            checkpoint_path=checkpoint_path,
            output_path=output_path,
            configure_local_source=True,
            local_source_tick_steps=1,
            local_source_tick_tokens=8,
            trace_dir=root / "traces",
            web_dist_dir=root / "missing-ui-dist",
            env_root=root,
            feed_text="brain service feeds sparse local state.",
            query_text="brain service",
            top_k_candidates=4,
            top_k_memories=4,
            top_chars=4,
            export_limit=2,
        )

        assert output_path.is_file()
        assert result["success"] is True
        assert set(result["endpoints_by_name"]) == {
            "brain_checkpoints",
            "brain_evidence_reports",
            "brain_feed",
            "brain_feed_configured",
            "brain_generate",
            "brain_grow_prune",
            "brain_replay",
            "brain_status",
            "brain_tick",
            "brain_tick_configured",
            "brain_traces",
            "health",
        }
        assert result["endpoints_by_name"]["brain_status"]["path"] == "/brain/status"
        assert result["endpoints_by_name"]["brain_feed"]["path"] == "/brain/feed"
        assert result["endpoints_by_name"]["brain_tick"]["path"] == "/brain/tick"
        assert result["endpoints_by_name"]["brain_generate"]["path"] == "/brain/generate"
        assert result["endpoint_metabolism_summary"]["hot_path"]["endpoint_names"] == [
            "brain_feed",
            "brain_tick",
            "brain_generate",
        ]
        assert result["brain_status_summary"]["verdict"] == "alive"
        assert result["configured_source_summary"]["accepted_tokens"] > 0
        assert result["configured_source_summary"]["tick_tokens_processed"] > 0
        assert result["feed_summary"]["tokens_processed"] > 0
        assert result["runtime_device_evidence"]["brain"]["summary_role"] == "observed_brain_device_evidence_not_acceleration_claim"
    finally:
        shutil.rmtree(root, ignore_errors=True)
