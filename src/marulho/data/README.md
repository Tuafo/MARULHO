# Data

Use this with [../../../README.md](../../../README.md) and
[../../../CONTEXT.md](../../../CONTEXT.md).

`data` owns source loaders and sparse encoders for text, semantic features,
audio, event-camera style input, and multimodal streams.

## Owns

- Input normalization.
- Emitted encoder tensors and device evidence.
- Dataset/source adapters and bounded source-stream preparation.

## Must Not Own

- Runtime Truth verdicts.
- Promotion of observations into facts or actions.
- Hot-path mutation of learned chunk codebooks.

## Ported Guidance

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
