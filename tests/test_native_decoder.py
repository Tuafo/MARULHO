from __future__ import annotations

import unittest

from marulho.retrieval.decoder import NativeAssemblyDecoder


class NativeAssemblyDecoderTests(unittest.TestCase):
    def test_decoder_stitches_overlapping_windows(self) -> None:
        decoder = NativeAssemblyDecoder(max_steps=8, max_output_chars=64)
        memory_matches = [
            {"memory_index": 10, "raw_window": "predictive c", "similarity": 0.96, "bucket_id": 7, "importance": 1.0, "tag_strength": 0.0},
            {"memory_index": 11, "raw_window": "redictive co", "similarity": 0.95, "bucket_id": 7, "importance": 1.0, "tag_strength": 0.0},
            {"memory_index": 12, "raw_window": "edictive cod", "similarity": 0.94, "bucket_id": 7, "importance": 1.0, "tag_strength": 0.0},
            {"memory_index": 13, "raw_window": "dictive codi", "similarity": 0.93, "bucket_id": 7, "importance": 1.0, "tag_strength": 0.0},
            {"memory_index": 14, "raw_window": "ictive codin", "similarity": 0.92, "bucket_id": 7, "importance": 1.0, "tag_strength": 0.0},
            {"memory_index": 15, "raw_window": "ctive coding", "similarity": 0.91, "bucket_id": 7, "importance": 1.0, "tag_strength": 0.0},
        ]

        result = decoder.decode(query_window="predictive ", winner_column=7, memory_matches=memory_matches)

        self.assertTrue(result["available"])
        self.assertEqual(result["decoded_text"], "predictive coding")
        self.assertEqual(result["continuation_text"], "coding")
        self.assertGreaterEqual(result["confidence"], 0.75)


if __name__ == "__main__":
    unittest.main()