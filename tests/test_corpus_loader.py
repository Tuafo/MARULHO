from __future__ import annotations

import unittest

from hecsn.data.corpus_loader import StreamingCorpusLoader, extract_web_text


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


if __name__ == "__main__":
    unittest.main()