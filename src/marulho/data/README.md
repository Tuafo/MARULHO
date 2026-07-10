# Data

Use this with [../../../README.md](../../../README.md) and
[../../../CONTEXT.md](../../../CONTEXT.md).

`data` owns source loaders and sparse encoders for text, semantic features,
audio, event-camera style input, and multimodal streams.

## Owns

- Input normalization.
- Emitted encoder tensors and device evidence.
- Dataset/source adapters and bounded source-stream preparation.
- MARULHO-owned tokenizer adapters and deterministic source-to-token
  preparation for the language-model path.

## Must Not Own

- Runtime Truth verdicts.
- Promotion of observations into facts or actions.
- Hot-path mutation of learned chunk codebooks.

## Runtime Rules

- Live RTF ingestion is inference-only. When the learned chunk codebook is
  empty, build bounded CPU control-plane windows and emit one device batch.
- Source concept observation is bounded per service tick. Training may consume
  a larger sequential source window, but concept observation and structural
  maintenance must report caps and skipped attempts.
- The maintained source tick width is evidence-tuned at `128` source tokens per
  tick until a same-checkpoint benchmark promotes another value.
- Slow-memory archival is cadenced and evidence-backed; do not force replay
  memory writes every few tokens.
- Cross-modal text-only background ticks should use specialist sleep semantics
  rather than decaying dormant sensory traces every token.
- Semantic encoder construction is offline/deterministic by default. GloVe
  bucket initialization is an explicit `semantic_initialize_from_glove=True`
  setup step because it may touch cache, downloader, PCA, and ridge-solve work.
- Language tokenizers share one checkpoint-owned contract. The deterministic
  `ByteLevelLanguageTokenizer` remains the no-training baseline. The current
  quality branch can instead train `BytePairLanguageTokenizer` directly on the
  experiment corpus, using a byte-level alphabet so arbitrary UTF-8 remains
  lossless. Its complete learned BPE JSON, special-token layout, vocabulary
  hash, and normalization policy are embedded in the model checkpoint. The
  Rust `tokenizers` package is an execution dependency only: MARULHO loads no
  external tokenizer or language-model checkpoint.
- BPE training and source encoding consume bounded text chunks, preferring
  blank-line document boundaries and preserving the exact concatenated source.
  Corpus size must not become one monolithic Rust tokenizer allocation.
- Hugging Face text sources may expose structured rows instead of a single
  `text` column. `StreamingCorpusLoader` flattens role/content `messages` rows
  and accepts comma-separated `text_field` values such as
  `problem,generated_solution,expected_answer`, so NVIDIA/Nemotron-style SFT,
  math, and preference rows can feed the MARULHO-owned tokenizer without a
  dataset-specific hidden generation path.
