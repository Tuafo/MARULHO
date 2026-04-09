from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

from hecsn.config.model_config import HECSNConfig
from hecsn.data.pattern_loader import load_probe_train_examples
from hecsn.data.rtf_encoder import RTFEncoder


class PatternLoaderTests(unittest.TestCase):
    def test_load_probe_train_examples_repeats_prefix_for_train_stream(self) -> None:
        cfg = HECSNConfig(
            n_columns=4,
            column_latent_dim=8,
            bootstrap_tokens=0,
            memory_capacity=64,
            eta_competitive=0.05,
            eta_decay=0.0,
            input_weight_blend=0.0,
        )
        encoder = RTFEncoder.from_config(cfg)
        prefix_text = (
            "Terms: submarine, buoyancy, ballast. "
            "Ballast tank: submarines use ballast tanks to control buoyancy."
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            source_path = Path(tmpdir) / "source.txt"
            source_path.write_text(
                "neutral background signal " * 16,
                encoding="utf-8",
            )
            probe_patterns, probe_raw_windows, train_patterns, train_raw_windows = load_probe_train_examples(
                source=str(source_path),
                source_type="file",
                hf_config=None,
                text_field="text",
                encoder=encoder,
                window_size=cfg.window_size,
                probe_tokens=24,
                train_tokens=24,
                prefix_text=prefix_text,
            )

        self.assertTrue(probe_patterns)
        self.assertTrue(train_patterns)
        self.assertTrue(probe_raw_windows)
        self.assertTrue(train_raw_windows)
        probe_prefix = probe_raw_windows[0] + "".join(window[-1] for window in probe_raw_windows[1:5])
        train_prefix = train_raw_windows[0] + "".join(window[-1] for window in train_raw_windows[1:5])
        self.assertEqual(probe_prefix, "Terms")
        self.assertEqual(train_prefix, "Terms")


if __name__ == "__main__":
    unittest.main()
