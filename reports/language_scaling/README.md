# MARULHO V57 Native Long Context

## Summary

| Field | Value |
|-------|-------|
| Surface | marulho_native_context_falsification.v1 |
| Decision | retire_v57_context_exonerated_base_or_objective_failure |
| Owned By Marulho | true |
| External Llm Used | false |
| External Text Data | true |
| Experiment Contract Sha256 | 06c2fe44bdb75e29b12c2eafb2b13af62a720d8d97758b1831e68634160ef129 |
| Total Wall Seconds | 2818.59 |
| Boundary | V57 tests whether V39 can learn source-visible long-context QA when evidence participates in every causal layer. It does not prove open-domain retrieval, arbitrary-length memory, or a post-Transformer architecture. |

## Configuration

| Field | Value |
|-------|-------|
| Context Length | 320 |
| Batch Size | 32 |
| Grounding Epochs | 4 |
| Optimizer Steps | 2048 |
| Padded Position Budget Per Arm | 20971520 |
| Grounding Fraction | 0.5 |
| General Fraction | 0.25 |
| Relation Fraction | 0.25 |
| Answer Weight | 4 |
| Learning Rate | 0.0003 |
| Minimum Learning Rate Fraction | 0.1 |
| Warmup Fraction | 0.05 |
| Weight Decay | 0.1 |
| Gradient Clip | 1 |
| Precision | bfloat16 |
| Execution Backend | pytorch_eager |
| Data Seed | 57121 |
| Model Seed | 57131 |
| Sample Bytes Per Replay Source | 16777216 |
| Sample Bytes Per Eval Source | 1048576 |
| Sample Range Count | 16 |
| General Eval Batches | 16 |
| Relation Case Count | 64 |
| Relation Eval Batch Size | 8 |
| Relation Generation Tokens | 16 |
| Grounding Generation Tokens | 16 |
| Maximum Training Seconds Per Arm | 1800 |
| Minimum Oracle Answer Count | 128 |
| Minimum Native Answer Count | 128 |
| Minimum Native Source Gain | 0.45 |
| Maximum Native Oracle Gap | 16 |
| Maximum Mismatched Answer Count | 16 |
| Maximum General Loss Regression | 0.1 |
| Maximum Relation Generation Regression | 0.05 |

## Checkpoint

| Field | Value |
|-------|-------|
| Path | reports\language_scaling\v39-answer-objective-qualified-100m-218m-20260810.pt |
| Sha256 | 6caf97be17d49cd3fc70501b50cadd39897fd85000b121e107f13a5417a1068d |

## Data

| Field | Value |
|-------|-------|
| Training Manifest Path | reports\language_curriculum\squad-v57-native-train-8192-20260812.json |
| Training Manifest Sha256 | aae376dcf95ab887aeb67abc135b9f9f8dd1f19699935053efa8b66e5ffc9133 |
| Training Manifest Contract Sha256 | fef030f0c5a66381d9088cc72d38a284fd711a0a663f0e5f0d9b5376509760f7 |
| Validation Manifest Path | reports\language_curriculum\squad-v57-native-validation-256-20260812.json |
| Validation Manifest Sha256 | b85f1da5d7d5c3b8bd1e9f1339ab1235028c8c8f1fb8db3b3042e3c99b3c0f80 |
| Validation Manifest Contract Sha256 | 9a6922f4ca6bd3fac5d099ba53ef33f63b66fd59b41e639785d936ca78ece15c |
| Training Validation Case Overlap | 0 |

## Setup Timings

| Field | Value |
|-------|-------|
| Parent And Data Preparation Seconds | 42.8091 |
| Baseline General Seconds | 1.57366 |
| Baseline Relation Seconds | 9.11414 |

## Parent

| Field | Value |
|-------|-------|
| Checkpoint Sha256 Before | 6caf97be17d49cd3fc70501b50cadd39897fd85000b121e107f13a5417a1068d |
| Checkpoint Sha256 After | 6caf97be17d49cd3fc70501b50cadd39897fd85000b121e107f13a5417a1068d |
| Checkpoint File Exact | true |
| Tokenizer Hash Before | faca1e26aa29e897bef4e4335a0300f90e3996723d556a681b4495240f660715 |
| Tokenizer Hash After | faca1e26aa29e897bef4e4335a0300f90e3996723d556a681b4495240f660715 |
| Tokenizer Exact | true |
| Initial Short Prefix Exact | true |
| Initial Short Prefix Max Absolute Delta | 0 |
| Parameter Count | 100679424 |

## Gate

| Field | Value |
|-------|-------|
| Passed | false |

## JSON Preview

```json
{
  "arms": {
    "native_full": {
      "checkpoint_fidelity": {
        "context_exact": true,
        "expected_state_sha256": "f0679d7f1311ab71e0e54161738b29a9b1e7e76280ada9a6b73052444ec766e1",
        "logits_exact": true,
        "metadata": {
          "arm": "native_full",
          "cumulative_tokens": 231986680,
          "parent_checkpoint_sha256": "6caf97be17d49cd3fc70501b50cadd39897fd85000b121e107f13a5417a1068d",
          "processed_nonpad_tokens": 13878520,
          "processed_padded_positions": 20971520,
          "source_experiment": "marulho_native_context_falsification.v1"
        },
        "passed": true,
        "path": "reports\\language_scaling\\.v57-native-context-arms\\v57-native_full.pt",
        "restored_state_sha256": "f0679d7f1311ab71e0e54161738b29a9b1e7e76280ada9a6b73052444ec766e1",
        "sha256": "dd4255998b5c3519584ddbf663827b593f52dd52560f04c96133c408304a1c85",
        "size_bytes": 428144230,
        "state_exact": true,
        "tokenizer_exact": true
      },
      "general": {
        "batch_count": 16,
        "batch_transfer_policy": "cpu_split_per_batch_to_model_device",
        "external_llm_used": false,
        "heldout_loss": 3.3552746772766113,
        "heldout_perplexity": 28.653473567972686,
        "owned_by_marulho": true,
        "surface": "marulho_transformer_heldout_evaluation.v3",
        "token_count": 9216,
        "tokens_per_second": 15977.589766097884
      },
      "parameter_count": 100679424,
      "relation": {
        "accuracy": 1.0,
        "case_count": 64,
        "correct_index_metrics_only": true,
        "evaluation_batch_size": 8,
        "evaluation_mode": "length_grouped_batched",
        "generation_exact_accuracy": 0.75,
        "generation_kind_accuracy": {
          "container": 0.0,
          "event_order": 1.0,
          "ownership": 1.0,
          "property": 1.0
        },
        "generation_max_new_tokens": 16,
        "kind_accuracy": {
          "container": 1.0,
          "event_order": 1.0,
          "ownership": 1.0,
          "property": 1.0
        },
        "prediction_uses_correct_index": false,
        "rows": [
          {
            "candidate_losses": [
              2.9063968658447266,
              2.929298162460327,
              2.7794487476348877,
              0.3555024564266205
            ],
            "case_id": "container-0000",
            "correct": true,
            "generation_continuation": " Cora carried the cup to the shelf.",
            "generation_exact_answer_match": false,
            "kind": "container",
            "label_used_for_generation": false,
            "label_used_for_prediction": false,
            "predicted_index": 3
          },
          {
            "candidate_losses": [
              3.4058094024658203,
              0.018950097262859344,
              0.25312748551368713,
              4.400670528411865
            ],
            "case_id": "event_order-0000",
            "correct": true,
            "generation_continuation": " Some water reaches the studio.",
            "generation_exact_answer_match": true,
            "kind": "event_order",
            "label_used_for_generation": false,
            "label_used_for_prediction": false,
            "predicted_index": 1
          },
          {
            "candidate_losses": [
              0.7191920280456543,
              1.0910178422927856,
              0.0009006732143461704,
              1.0131338834762573
            ],
            "case_id": "ownership-0000",
            "correct": true,
            "generation_continuation": " Ben has the red key.",
            "generation_exact_answer_match": true,
            "kind": "ownership",
            "label_used_for_generation": false,
            "label_used_for_prediction": false,
            "predicted_index": 2
          },
          {
            "candidate_losses": [
              1.241619348526001,
              1.4364808797836304,
              0.008584472350776196,
```
