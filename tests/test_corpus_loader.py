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


if __name__ == "__main__":
    unittest.main()
