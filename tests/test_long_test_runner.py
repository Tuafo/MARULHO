from __future__ import annotations

import json
import tempfile
from pathlib import Path

from hecsn.training.long_test_runner import TestReport as LongTestReport, write_report


def test_write_report_handles_unicode_text() -> None:
    report = LongTestReport(
        start_time="2026-04-21T00:00:00+00:00",
        end_time="2026-04-21T00:01:00+00:00",
        duration_minutes=1.0,
        preset="curriculum",
        cortex_model="multi(test-fast,test-deep)",
        total_thoughts=2,
        unique_topics=4,
        topic_diversity_ratio=2.0,
        avg_latency_ms=1234.0,
        final_narrative_summary="Coral reefs balance calcium carbonate growth with ocean chemistry — a fragile equilibrium.",
        sample_thoughts=[
            "Aurora Borealis occurs when charged solar particles strike Earth's atmosphere.",
            "Zonal reef growth depends on carbonate saturation — a delicate équilibre.",
        ],
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        json_path, md_path = write_report(report, output_dir=tmpdir)
        json_data = json.loads(Path(json_path).read_text(encoding="utf-8"))
        md_text = Path(md_path).read_text(encoding="utf-8")

    assert json_data["preset"] == "curriculum"
    assert "équilibre" in md_text
    assert "Aurora Borealis" in md_text
