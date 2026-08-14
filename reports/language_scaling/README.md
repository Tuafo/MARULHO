# Marulho Language Generation Coherence Report

## Summary

| Field | Value |
|-------|-------|
| Artifact Kind | marulho_language_generation_coherence_report |
| Surface | marulho_language_generation_coherence_report.v1 |
| Owned By Marulho | true |
| External Llm Used | false |
| Loads External Checkpoint | false |
| Active Language Path | marulho_transformer |
| Checkpoint Path | reports\language_scaling\v77-static-long-document-qualified-100m-225m-20260813.pt |
| Output Path | reports\language_scaling\v77-unseen-cosmopedia-controlled-20260813.json |

## Checkpoint

| Field | Value |
|-------|-------|
| Path | reports\language_scaling\v77-static-long-document-qualified-100m-225m-20260813.pt |
| Sha256 | 3755bfb683b77bbf74811d58b9d3db404cdca4143b82e1f6f427077ea4487074 |
| Kind | transformer |
| Model Surface | marulho_transformer_language_model.v2 |
| Tokenizer Hash | faca1e26aa29e897bef4e4335a0300f90e3996723d556a681b4495240f660715 |

## Source

| Field | Value |
|-------|-------|
| Path | reports\language_curriculum\cosmopedia-v2-eval-10k-shard2-20260710.txt |
| Sha256 | e0a86c6014f701b5fa91578cf2e9079e9351c61778ac3917acacc3f166c97491 |
| Size Bytes | 37553596 |
| Role | generation_prompt_and_continuation_holdout |
| Raw Source Text Retained | false |

## Prompt Suite

| Field | Value |
|-------|-------|
| Surface | marulho_language_generation_coherence_prompt_suite.v1 |
| Case Count | 4 |
| Min Case Pass Rate | 1 |
| Review Kind | automated_grounded_prompt_suite_not_human_review |

## Summary

| Field | Value |
|-------|-------|
| Surface | marulho_language_generation_coherence_summary.v1 |
| Case Count | 4 |
| Passed Case Count | 0 |
| Case Pass Rate | 0 |
| Mean Prefix Match Chars | 3 |
| Mean Prefix Match Fraction | 0.00979792 |
| Mean Printable Fraction | 1 |
| Mean Distinct Bigram Fraction | 0.97619 |
| Max Token Run Length | 1 |
| Next Character Match Rate | 1 |
| Source Continuation Loss Available | true |
| Source Continuation Loss Case Count | 4 |
| Mean Source Continuation Loss | 2.6683 |
| Mean Source Continuation Perplexity | 16.7325 |
| Source Continuation Loss Token Count | 256 |

## Promotion Gate

| Field | Value |
|-------|-------|
| Status | blocked_generation_coherence |
| Generation Coherence Available | false |
| Grounded Prompt Suite Available | false |
| Human Review Available | false |
| Promotes Prompt Suite Coherence Claim | false |
| Promotes Generation Quality Claim | false |
| Promotes Runtime Claim | false |
| Requires Long Run Pairing | true |

## JSON Preview

```json
{
  "active_language_path": "marulho_transformer",
  "artifact_kind": "marulho_language_generation_coherence_report",
  "cases": [
    {
      "active_language_path": "marulho_transformer",
      "active_language_path_matches_model": true,
      "batched_decode_group_size": 1,
      "continuation_sequence_hash": "311aa052f661e6d647b65e327a5ba3dc917325ecd9b7020bc5c34213537aff98",
      "continuation_text": " age, businesses are constantly seeking ways to reach their customers and expand their reach. One such approach that has gained significant traction is the use of social media platforms like Instagram, Twitter, and TikTok. This course unit will delve into the concept of social networks, exploring how they can be",
      "continuation_token_count": 64,
      "distinct_bigram_fraction": 0.9841269841269841,
      "expected_active_language_path": "marulho_transformer",
      "expected_source_continuation": " age, computers have become an essential part of our daily lives, enabling us to perform various tasks efficiently. One critical aspect of using a computer is ensuring that its hardware components are functioning optimally. This requires keeping the device drivers updated, especially when transitioning to a newe",
      "external_llm_used": false,
      "failure_reasons": [
        "source_prefix_match_below_threshold"
      ],
      "generated_text": "In today's digital age, businesses are constantly seeking ways to reach their customers and expand their reach. One such approach that has gained significant traction is the use of social media platforms like Instagram, Twitter, and TikTok. This course unit will delve into the concept of social networks, exploring how they can be",
      "generated_token_count": 69,
      "generation_decode": {
        "decode_control_scope": "generated_continuation_only",
        "decode_control_window": 320,
        "decode_strategy": "greedy_argmax",
        "external_llm_used": false,
        "full_model_vocab_logits_materialized": true,
        "generation_vocab_size": 8192,
        "kv_cache": "bounded_per_layer",
        "model_vocab_size": 8192,
        "no_repeat_ngram_size": 3,
        "prompt_tokens_eligible_for_penalty": false,
        "repetition_penalty": 1.1,
        "sampling_seed": null,
        "surface": "marulho_transformer_decode_policy.v4",
        "temperature": 0.0,
        "top_p": 1.0,
        "top_p_applied": false
      },
      "generation_surface": "marulho_transformer_generation.v3",
      "max_token_run_length": 1,
      "new_token_count": 64,
      "next_character_matches_source": true,
      "owned_by_marulho": true,
      "passed": false,
      "prefix_match_chars": 6,
      "prefix_match_fraction": 0.019169329073482427,
      "printable_fraction": 1.0,
      "prompt_text": "In today's digital",
      "prompt_token_count": 5,
      "sequence_hash": "c869db60de625a6c1c49819e3781c6c1dd14f04dd34f637c683dbbcd0ac889ee",
      "source_continuation_loss": {
        "continuation_add_bos": false,
        "continuation_clipped_to_context": false,
        "continuation_source_characters_scanned": 2048,
        "continuation_target_start_index": 4,
        "decode_vocab_only": true,
        "enabled": true,
        "evaluated_token_count": 64,
        "full_model_vocab_logits_materialized": true,
        "generation_vocab_size": 8192,
        "loss": 2.0215673446655273,
        "model_context_length": 320,
        "model_vocab_size": 8192,
        "perplexity": 7.550149917602539,
        "perplexity_capped": false,
        "prompt_token_count": 5,
        "reason": null,
        "source_continuation_token_count": 64,
        "surface": "marulho_language_generation_source_continuation_loss.v2"
      },
      "source_prompt_found": true,
      "source_text_hash": "c52c8c2b50cef14bdc707c7d1a9908b92555a1d96e0bd76235bfb360fb99e1c6",
      "surface": "marulho_language_generation_coherence_case.v1",
      "thresholds": {
        "max_token_run_length": 8,
        "min_distinct_
```
