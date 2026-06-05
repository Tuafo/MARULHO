from __future__ import annotations

import json
import os
import unittest
from unittest.mock import patch

from marulho.data.corpus_loader import StreamingCorpusLoader, extract_web_text, load_hf_first_rows


class CorpusLoaderTests(unittest.TestCase):
    def test_auto_detects_web_sources(self) -> None:
        loader = StreamingCorpusLoader("https://example.com/page")
        self.assertEqual(loader.source_type, "web")

    def test_extract_web_text_prefers_main_content(self) -> None:
        html = """
        <html>
          <body>
            <nav>Jump to content Donate</nav>
            <main>
              <article>
                <h1>Predictive coding</h1>
                <p>The brain continuously updates internal models from sensory input.</p>
              </article>
            </main>
            <footer>Privacy policy</footer>
          </body>
        </html>
        """

        text = extract_web_text(html, content_type="text/html")

        self.assertIn("Predictive coding", text)
        self.assertIn("internal models", text)
        self.assertNotIn("Donate", text)
        self.assertNotIn("Privacy policy", text)

    def test_extract_web_text_normalizes_mediawiki_raw_revisions(self) -> None:
        mediawiki = """
        {{short description|Neuroscientific theory}}
        '''Hebbian theory''' is a [[neuropsychological]] theory of [[learning]].
        [[File:Hebb.jpg|thumb|Donald Hebb]]
        == History ==
        The rule was discussed in [https://example.com detail] and refined later.<ref>citation</ref>
        [[Category:Neuroscience]]
        """

        text = extract_web_text(mediawiki, content_type="text/x-wiki")

        self.assertIn("Hebbian theory is a neuropsychological theory of learning.", text)
        self.assertIn("History", text)
        self.assertIn("detail", text)
        self.assertNotIn("short description", text)
        self.assertNotIn("File:Hebb", text)
        self.assertNotIn("Category:Neuroscience", text)
        self.assertNotIn("https://example.com", text)

    def test_hf_loader_passes_env_token_to_load_dataset(self) -> None:
        calls: list[dict[str, object]] = []

        def _fake_load_dataset(*args, **kwargs):
            calls.append({"args": args, "kwargs": kwargs})
            return [{"text": "abc"}]

        with patch.dict(os.environ, {"HF_TOKEN": "test-token"}, clear=False):
            with patch("datasets.load_dataset", side_effect=_fake_load_dataset):
                loader = StreamingCorpusLoader(
                    "HuggingFaceFW/fineweb-edu",
                    source_type="hf",
                    hf_config="sample-10BT",
                    text_field="text",
                )
                chars = loader.char_stream()
                observed = "".join(next(chars) for _ in range(3))

        self.assertEqual(observed, "abc")
        self.assertTrue(calls)
        self.assertEqual(calls[0]["args"][:2], ("HuggingFaceFW/fineweb-edu", "sample-10BT"))
        self.assertEqual(calls[0]["kwargs"]["token"], "test-token")
        self.assertTrue(calls[0]["kwargs"]["streaming"])
        self.assertEqual(calls[0]["kwargs"]["split"], "train")

    def test_load_hf_first_rows_filters_requested_columns(self) -> None:
        payload = {
            "rows": [
                {"row": {"id": "1", "title": "One", "text": "hello world"}},
                {"row": {"id": "2", "title": "Two", "text": "goodbye"}},
            ]
        }

        class _Response:
            def __enter__(self):
                return self

            def __exit__(self, exc_type, exc, tb):
                return False

            def read(self):
                return json.dumps(payload)

        with patch("marulho.data.corpus_loader.urlopen", return_value=_Response()):
            rows = load_hf_first_rows(
                "wikimedia/wikipedia",
                hf_config="20231101.en",
                columns=["text"],
                max_rows=1,
            )

        self.assertEqual(rows, [{"text": "hello world"}])

    def test_hf_loader_projects_requested_text_column_when_supported(self) -> None:
        calls: list[list[str]] = []

        class _FakeDataset:
            def select_columns(self, columns):
                calls.append(list(columns))
                return [{"text": "abc"}]

        with patch("datasets.load_dataset", return_value=_FakeDataset()):
            loader = StreamingCorpusLoader(
                "wikimedia/wikipedia",
                source_type="hf",
                hf_config="20231101.en",
                text_field="text",
            )
            chars = loader.char_stream()
            observed = "".join(next(chars) for _ in range(3))

        self.assertEqual(observed, "abc")
        self.assertEqual(calls, [["text"]])

    def test_extract_web_text_normalizes_openalex_work_json(self) -> None:
        payload = json.dumps(
            {
                "id": "https://openalex.org/W999",
                "display_name": "Closed-access submarine paper",
                "abstract_inverted_index": {
                    "Submarine": [0],
                    "ballast": [1],
                    "control": [2],
                    "improves": [3],
                    "buoyancy": [4],
                },
                "primary_location": {
                    "source": {
                        "display_name": "Journal of Marine Systems",
                    }
                },
                "primary_topic": {
                    "display_name": "Marine engineering systems",
                    "subfield": {"display_name": "Marine engineering"},
                    "field": {"display_name": "Engineering"},
                    "domain": {"display_name": "Physical sciences"},
                },
                "topics": [
                    {
                        "display_name": "Ballast control",
                        "subfield": {"display_name": "Naval architecture"},
                    }
                ],
                "keywords": [
                    {"display_name": "Submarine"},
                    {"display_name": "Ballast tank"},
                ],
                "concepts": [
                    {"display_name": "Buoyancy"},
                ],
                "authorships": [
                    {"author": {"display_name": "Ada Lovelace"}},
                    {"author": {"display_name": "Grace Hopper"}},
                ],
            }
        )

        text = extract_web_text(payload, content_type="application/json")

        self.assertIn("Closed-access submarine paper", text)
        self.assertIn("Submarine ballast control improves buoyancy", text)
        self.assertIn("Journal of Marine Systems", text)
        self.assertIn("Marine engineering systems", text)
        self.assertIn("Ballast control", text)
        self.assertIn("Ada Lovelace", text)
        self.assertNotIn('"display_name"', text)
        self.assertNotIn("{", text)


if __name__ == "__main__":
    unittest.main()
