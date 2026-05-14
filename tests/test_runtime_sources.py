from __future__ import annotations

from collections import deque
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from hecsn.service.runtime_sources import RuntimeSources, RuntimeSourcesDependencies, _BrainSourceRuntime


class _FakeTrainerConfig:
    window_size = 4
    cross_modal_dim_visual = 32
    cross_modal_dim_audio = 16


class _FakeTrainerModel:
    device = torch.device("cpu")


class _FakeTrainer:
    def __init__(self) -> None:
        self.config = _FakeTrainerConfig()
        self.model = _FakeTrainerModel()


class _FakeManager:
    def __init__(self, root: Path) -> None:
        self._trainer = _FakeTrainer()
        self._encoder = object()
        self._checkpoint_path = root / "checkpoints" / "service.pt"
        self._checkpoint_dir = self._checkpoint_path.parent
        self._brain_config = {
            "tick_tokens": 4,
            "ingestion": {"enabled": True, "queue_target_tokens": 4},
        }
        self._brain_source_runtimes = []
        self._sensory_source_runtimes = []

    def _sensory_queue_target_items_locked(self) -> int:
        return 2


def _runtime_sources(fake: _FakeManager) -> RuntimeSources:
    return RuntimeSources(
        RuntimeSourcesDependencies(
            brain_config=lambda: fake._brain_config,
            brain_source_runtimes=lambda: fake._brain_source_runtimes,
            set_brain_source_runtimes=lambda value: setattr(fake, "_brain_source_runtimes", list(value)),
            checkpoint_dir=lambda: fake._checkpoint_dir,
            checkpoint_path=lambda: fake._checkpoint_path,
            encoder=lambda: fake._encoder,
            sensory_queue_target_items=fake._sensory_queue_target_items_locked,
            sensory_source_runtimes=lambda: fake._sensory_source_runtimes,
            set_sensory_source_runtimes=lambda value: setattr(fake, "_sensory_source_runtimes", list(value)),
            trainer=lambda: fake._trainer,
        )
    )


class _FakeLoader:
    def __init__(self, *, source: str, source_type: str, text_field: str, hf_config: str | None) -> None:
        self.source = source
        self.source_type = source_type
        self.text_field = text_field
        self.hf_config = hf_config

    def char_stream(self):
        yield "abc"


class _FakePrefetchIterator:
    def __init__(self, stream, max_buffer: int, name: str) -> None:
        self.stream = list(stream)
        self.max_buffer = max_buffer
        self.name = name

    def __iter__(self):
        return iter(self.stream)


class RuntimeSourcesSeamTests(unittest.TestCase):
    def test_remote_detection_and_window_reconstruction(self) -> None:
        module = _runtime_sources(_FakeManager(Path(".")))

        self.assertTrue(module._source_spec_uses_live_remote({"source": "https://example.com"}))
        self.assertFalse(module._source_spec_uses_live_remote({"source": "notes.txt", "source_type": "file"}))
        self.assertTrue(module._sensory_spec_uses_live_remote({"adapter": "audiocaps", "source": ""}))
        self.assertEqual(
            module._reconstruct_text_from_windows(["cats chase", "chase mice", "mice at night"]),
            "cats chase mice at night",
        )

    def test_build_brain_source_stream_wraps_remote_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            module = _runtime_sources(_FakeManager(root))

            with patch("hecsn.service.runtime_sources.StreamingCorpusLoader", _FakeLoader), patch(
                "hecsn.service.runtime_sources.labeled_pattern_stream",
                return_value=iter([("window-1", "pattern-1")]),
            ), patch("hecsn.service.runtime_sources.BackgroundPrefetchIterator", _FakePrefetchIterator):
                stream = module._build_brain_source_stream_locked(
                    {
                        "name": "remote_source",
                        "source": "https://example.com/corpus",
                        "source_type": "hf",
                        "text_field": "text",
                    }
                )

            self.assertIsInstance(stream, _FakePrefetchIterator)
            self.assertEqual(stream.name, "remote_source")
            self.assertEqual(stream.max_buffer, 4)
            self.assertEqual(list(stream), [("window-1", "pattern-1")])

    def test_brain_runtime_cache_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            module = _runtime_sources(_FakeManager(root))
            runtime = _BrainSourceRuntime(
                spec={
                    "name": "remote_source",
                    "source": "https://example.com/corpus",
                    "source_type": "hf",
                },
                stream=iter([]),
                buffered_patterns=deque([("window-1", "pattern-1"), ("window-2", "pattern-2")]),
            )

            module._update_brain_runtime_cache_locked(
                runtime,
                served_examples=[("window-0", "pattern-0")],
            )
            cache_path = module._brain_runtime_cache_path(runtime.spec)
            self.assertTrue(cache_path.exists())

            restored_runtime = _BrainSourceRuntime(
                spec=runtime.spec,
                stream=iter([]),
                buffered_patterns=deque(),
            )

            with patch(
                "hecsn.service.runtime_sources.labeled_pattern_stream",
                return_value=iter([("restored-window", "restored-pattern")]),
            ):
                restored = module._restore_brain_runtime_cache_locked(restored_runtime)

            self.assertEqual(restored, 1)
            self.assertEqual(list(restored_runtime.buffered_patterns), [("restored-window", "restored-pattern")])
