from __future__ import annotations

from collections import deque
from pathlib import Path
import tempfile
from threading import Event
import time
import unittest
from unittest.mock import patch

import torch

from marulho.service.runtime_sources import RuntimeSources, RuntimeSourcesDependencies, _BrainSourceRuntime


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

            with patch("marulho.service.runtime_sources.StreamingCorpusLoader", _FakeLoader), patch(
                "marulho.service.runtime_sources.labeled_pattern_stream",
                return_value=iter([("window-1", "pattern-1")]),
            ), patch("marulho.service.runtime_sources.BackgroundPrefetchIterator", _FakePrefetchIterator):
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

    def test_live_source_stream_disables_chunk_plasticity(self) -> None:
        captured: dict[str, object] = {}

        def _stream(chars, encoder, window_size, *, learn_chunking=False):
            captured["chars"] = chars
            captured["encoder"] = encoder
            captured["window_size"] = window_size
            captured["learn_chunking"] = learn_chunking
            return iter([("window-1", "pattern-1")])

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source.txt"
            source.write_text("source material", encoding="utf-8")
            manager = _FakeManager(root)
            module = _runtime_sources(manager)

            with patch(
                "marulho.service.runtime_sources.labeled_pattern_stream",
                side_effect=_stream,
            ):
                built = module._build_source_stream_from_spec(
                    {"name": "local", "source": str(source), "source_type": "file"},
                    manager._encoder,
                    manager._trainer.config.window_size,
                )
                self.assertEqual(next(built), ("window-1", "pattern-1"))

        self.assertFalse(captured["learn_chunking"])

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
            module.flush_brain_runtime_cache_writes()
            cache_path = module._brain_runtime_cache_path(runtime.spec)
            self.assertTrue(cache_path.exists())

            restored_runtime = _BrainSourceRuntime(
                spec=runtime.spec,
                stream=iter([]),
                buffered_patterns=deque(),
            )

            with patch(
                "marulho.service.runtime_sources.labeled_pattern_stream",
                return_value=iter([("restored-window", "restored-pattern")]),
            ):
                restored = module._restore_brain_runtime_cache_locked(restored_runtime)

            self.assertEqual(restored, 1)
            self.assertEqual(list(restored_runtime.buffered_patterns), [("restored-window", "restored-pattern")])
            self.assertIsInstance(restored_runtime.cache_material_hash, str)

    def test_brain_runtime_cache_skips_unchanged_material_rewrite(self) -> None:
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

            module._update_brain_runtime_cache_locked(runtime)
            module.flush_brain_runtime_cache_writes()
            cache_hash = runtime.cache_material_hash
            self.assertIsInstance(cache_hash, str)
            self.assertEqual(runtime.cache_schedule_count, 1)
            self.assertEqual(runtime.cache_write_count, 1)
            self.assertEqual(runtime.last_cache_update_mode, "written")

            with patch("marulho.service.runtime_sources.torch.save") as save:
                module._update_brain_runtime_cache_locked(runtime)

            save.assert_not_called()
            self.assertEqual(runtime.cache_material_hash, cache_hash)
            self.assertEqual(runtime.cache_skip_count, 1)
            self.assertEqual(runtime.last_cache_update_mode, "skipped_unchanged_material")

    def test_restored_brain_runtime_cache_skips_same_served_window_rewrite(self) -> None:
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

            module._update_brain_runtime_cache_locked(runtime)
            module.flush_brain_runtime_cache_writes()
            restored_runtime = _BrainSourceRuntime(
                spec=runtime.spec,
                stream=iter([]),
                buffered_patterns=deque(),
            )
            with patch(
                "marulho.service.runtime_sources.labeled_pattern_stream",
                return_value=iter([("window-1", "pattern-1"), ("window-2", "pattern-2")]),
            ):
                restored = module._restore_brain_runtime_cache_locked(restored_runtime)

            self.assertEqual(restored, 2)
            self.assertEqual(restored_runtime.last_cache_update_mode, "restored")
            served_examples = list(restored_runtime.buffered_patterns)
            restored_runtime.buffered_patterns.clear()
            with patch("marulho.service.runtime_sources.torch.save") as save:
                module._update_brain_runtime_cache_locked(
                    restored_runtime,
                    served_examples=served_examples,
                )

            save.assert_not_called()
            self.assertEqual(restored_runtime.cache_skip_count, 1)
            self.assertEqual(restored_runtime.last_cache_update_mode, "skipped_unchanged_material")

    def test_local_file_runtime_cache_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "local-source.txt"
            source_path.write_text("local source text", encoding="utf-8")
            module = _runtime_sources(_FakeManager(root))
            spec = {
                "name": "local_source",
                "source": str(source_path),
                "source_type": "file",
            }
            runtime = _BrainSourceRuntime(
                spec=spec,
                stream=iter([]),
                buffered_patterns=deque([("window-1", "pattern-1")]),
            )

            module._update_brain_runtime_cache_locked(
                runtime,
                served_examples=[("window-0", "pattern-0")],
            )
            module.flush_brain_runtime_cache_writes()
            cache_path = module._brain_runtime_cache_path(spec)
            self.assertTrue(cache_path.exists())

            restored_runtime = _BrainSourceRuntime(
                spec=spec,
                stream=iter([]),
                buffered_patterns=deque(),
            )
            with patch(
                "marulho.service.runtime_sources.labeled_pattern_stream",
                return_value=iter([
                    ("restored-window-0", "restored-pattern-0"),
                    ("restored-window-1", "restored-pattern-1"),
                ]),
            ):
                restored = module._restore_brain_runtime_cache_locked(restored_runtime)

            self.assertEqual(restored, 2)
            self.assertEqual(
                list(restored_runtime.buffered_patterns),
                [
                    ("restored-window-0", "restored-pattern-0"),
                    ("restored-window-1", "restored-pattern-1"),
                ],
            )

    def test_brain_runtime_cache_save_runs_outside_calling_tick(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            module = _runtime_sources(_FakeManager(Path(tmpdir)))
            runtime = _BrainSourceRuntime(
                spec={
                    "name": "remote_source",
                    "source": "https://example.com/corpus",
                    "source_type": "hf",
                },
                stream=iter([]),
                buffered_patterns=deque([("window-1", "pattern-1")]),
            )
            save_started = Event()
            release_save = Event()
            real_save = torch.save

            def blocked_save(payload, path) -> None:
                save_started.set()
                self.assertTrue(release_save.wait(timeout=2.0))
                real_save(payload, path)

            with patch("marulho.service.runtime_sources.torch.save", side_effect=blocked_save):
                started = time.perf_counter()
                module._update_brain_runtime_cache_locked(runtime)
                elapsed_ms = (time.perf_counter() - started) * 1000.0
                self.assertTrue(save_started.wait(timeout=1.0))
                self.assertLess(elapsed_ms, 50.0)
                self.assertTrue(runtime.cache_pending)
                self.assertEqual(runtime.last_cache_update_mode, "scheduled")
                release_save.set()
                module.flush_brain_runtime_cache_writes()

            self.assertFalse(runtime.cache_pending)
            self.assertEqual(runtime.cache_write_count, 1)
            module.close()

    def test_brain_runtime_cache_coalesces_duplicate_while_write_is_pending(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            module = _runtime_sources(_FakeManager(Path(tmpdir)))
            runtime = _BrainSourceRuntime(
                spec={
                    "name": "remote_source",
                    "source": "https://example.com/corpus",
                    "source_type": "hf",
                },
                stream=iter([]),
                buffered_patterns=deque([("window-1", "pattern-1")]),
            )
            save_started = Event()
            release_save = Event()

            def blocked_save(_payload, _path) -> None:
                save_started.set()
                self.assertTrue(release_save.wait(timeout=2.0))

            with patch("marulho.service.runtime_sources.torch.save", side_effect=blocked_save) as save:
                module._update_brain_runtime_cache_locked(runtime)
                self.assertTrue(save_started.wait(timeout=1.0))
                module._update_brain_runtime_cache_locked(runtime)
                self.assertEqual(runtime.cache_schedule_count, 1)
                self.assertEqual(runtime.cache_skip_count, 1)
                release_save.set()
                module.flush_brain_runtime_cache_writes()

            save.assert_called_once()
            module.close()

    def test_brain_runtime_cache_failure_is_visible_and_retryable(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            module = _runtime_sources(_FakeManager(Path(tmpdir)))
            runtime = _BrainSourceRuntime(
                spec={
                    "name": "remote_source",
                    "source": "https://example.com/corpus",
                    "source_type": "hf",
                },
                stream=iter([]),
                buffered_patterns=deque([("window-1", "pattern-1")]),
            )

            with patch(
                "marulho.service.runtime_sources.torch.save",
                side_effect=OSError("disk unavailable"),
            ):
                module._update_brain_runtime_cache_locked(runtime)
                module.flush_brain_runtime_cache_writes()

            self.assertEqual(runtime.cache_failure_count, 1)
            self.assertEqual(runtime.last_cache_update_mode, "write_failed")
            self.assertIsNone(runtime.cache_material_hash)
            self.assertFalse(runtime.cache_pending)

            module._update_brain_runtime_cache_locked(runtime)
            module.close()
            self.assertEqual(runtime.cache_schedule_count, 2)
            self.assertEqual(runtime.cache_write_count, 1)

    def test_local_file_runtime_cache_key_changes_when_source_changes(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source_path = root / "local-source.txt"
            source_path.write_text("first", encoding="utf-8")
            module = _runtime_sources(_FakeManager(root))
            spec = {
                "name": "local_source",
                "source": str(source_path),
                "source_type": "file",
            }

            before = module._brain_runtime_cache_path(spec)
            source_path.write_text("second version with different size", encoding="utf-8")
            after = module._brain_runtime_cache_path(spec)

            self.assertNotEqual(before, after)
