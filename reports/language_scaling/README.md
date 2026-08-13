# Marulho Exact Token Kv Falsification

## Summary

| Field | Value |
|-------|-------|
| Artifact Kind | marulho_exact_token_kv_falsification |
| Surface | marulho_exact_token_kv_falsification.v1 |
| Owned By Marulho | true |
| External Llm Used | false |
| Decision | retire_v39_protected_memory_adaptation |
| Experiment Contract Sha256 | 23d8cce93478509c546a7f6d6002a879d11ec2a008de107b237daf6217cd8398 |

## Configuration

| Field | Value |
|-------|-------|
| Context Length | 320 |
| Batch Size | 32 |
| Epochs | 8 |
| Optimizer Steps | 2048 |
| Padded Position Budget | 20971520 |
| Correction Scale | 0.25 |
| Learning Rate | 0.0003 |
| Minimum Learning Rate Fraction | 0.1 |
| Warmup Fraction | 0.05 |
| Weight Decay | 0.1 |
| Gradient Clip | 1 |
| Generation Tokens | 16 |
| Repetition Penalty | 1.1 |
| No Repeat Ngram Size | 3 |
| Minimum True Exact Answers | 64 |
| Minimum True Source Gain | 0.2 |
| Maximum Shuffled Exact Answers | 16 |
| Minimum Oracle Exact Answers | 128 |
| Maximum True Oracle Gap | 64 |
| Maximum Parameter Fraction | 0.0125 |
| Maximum Training Seconds | 1800 |
| Maximum Total Setup Training Seconds | 2400 |
| Data Seed | 63121 |
| Model Seed | 63131 |
| Precision | bfloat16_parent_fp32_controller |
| Execution Backend | pytorch_eager_sdpa |

## Data

| Field | Value |
|-------|-------|
| Train Manifest Path | reports\language_curriculum\squad-v57-native-train-8192-20260812.json |
| Train Manifest Sha256 | aae376dcf95ab887aeb67abc135b9f9f8dd1f19699935053efa8b66e5ffc9133 |
| Train Manifest Contract Sha256 | fef030f0c5a66381d9088cc72d38a284fd711a0a663f0e5f0d9b5376509760f7 |
| Validation Manifest Path | reports\language_curriculum\squad-v57-native-validation-256-20260812.json |
| Validation Manifest Sha256 | b85f1da5d7d5c3b8bd1e9f1339ab1235028c8c8f1fb8db3b3042e3c99b3c0f80 |
| Validation Manifest Contract Sha256 | 9a6922f4ca6bd3fac5d099ba53ef33f63b66fd59b41e639785d936ca78ece15c |
| Train Case Count | 8192 |
| Validation Case Count | 256 |
| Train Title Count | 171 |
| Validation Title Count | 22 |
| Title Intersection Count | 0 |
| Prepared Tensor Sha256 | 7a77a0a7fd88ea5326c12e2d28cceb89cfdc0290c7f163908f2ca1f4a8638062 |
| Schedule Sha256 | 286027e5783e93f276e918f2f1b71bc4392cfd26a8bc764c7b667debc5548518 |
| Cache Policy | online_exact_token_kv_no_persistent_hidden_cache |
| Persistent Cache Bytes | 0 |
| Source Mask Excludes Answer | true |
| Source Mask Excludes Question | true |
| Source Mask Excludes Labels | true |
| Source Mask Uses Answer Span | false |

## Architecture

| Field | Value |
|-------|-------|
| Controller Parameter Count | 983040 |
| Parent Parameter Count | 100679424 |
| Controller Parameter Fraction | 0.00976406 |
| Correction Matrix Count | 240 |
| Head Dim | 64 |
| Initial Controller State Sha256 | 3e87cecb41a9707df70b846e178ebdafebea69a8a967f76dc69b627bd9f10e6b |
| Initial Controller Zero | true |
| Bounded Scale | 0.03125 |
| Adapted State | exact_source_token_keys_and_values |
| Parent Update | none_frozen |

## Setup

| Field | Value |
|-------|-------|
| Seconds | 29.8655 |
| Persistent Cache Bytes | 0 |

## Training

| Field | Value |
|-------|-------|
| Optimizer Steps | 2048 |
| Padded Positions | 20971520 |
| Answer Target Positions | 470952 |
| Final Training Loss | 3.23131 |
| Training Seconds | 751.066 |
| Positions Per Second | 27922.3 |
| Peak Cuda Bytes | 4315288576 |
| Controller Finite | true |
| Maximum Bounded Coefficient | 0.0097761 |

## Parent

| Field | Value |
|-------|-------|
| Path | reports\language_scaling\v39-answer-objective-qualified-100m-218m-20260810.pt |
| Checkpoint Sha256 Before | 6caf97be17d49cd3fc70501b50cadd39897fd85000b121e107f13a5417a1068d |
| Checkpoint Sha256 After | 6caf97be17d49cd3fc70501b50cadd39897fd85000b121e107f13a5417a1068d |
| State Sha256 Before | 76b195a6c0706928927c0d2517e119ca30574c9917f5cbba8be048a5b1672082 |
| State Sha256 After | 76b195a6c0706928927c0d2517e119ca30574c9917f5cbba8be048a5b1672082 |
| Tokenizer Hash Before | faca1e26aa29e897bef4e4335a0300f90e3996723d556a681b4495240f660715 |
| Tokenizer Hash After | faca1e26aa29e897bef4e4335a0300f90e3996723d556a681b4495240f660715 |

## Checkpoint

| Field | Value |
|-------|-------|
| Saved | false |
| Path | - |
| Sha256 | - |
| Strict Tensor Reload | false |
| Strict Logit Reload | false |

## Runtime

| Field | Value |
|-------|-------|
| Total Setup Training Seconds | 780.932 |
| Total Wall Seconds | 1073.49 |
| Peak Cuda Bytes | 4315288576 |

## JSON Preview

```json
{
  "architecture": {
    "adapted_state": "exact_source_token_keys_and_values",
    "bounded_scale": 0.03125,
    "controller_dtypes": [
      "torch.float32"
    ],
    "controller_parameter_count": 983040,
    "controller_parameter_fraction": 0.009764060628713966,
    "correction_matrix_count": 240,
    "head_dim": 64,
    "initial_controller_state_sha256": "3e87cecb41a9707df70b846e178ebdafebea69a8a967f76dc69b627bd9f10e6b",
    "initial_controller_zero": true,
    "parent_parameter_count": 100679424,
    "parent_update": "none_frozen"
  },
  "artifact_kind": "marulho_exact_token_kv_falsification",
  "checkpoint": {
    "path": null,
    "saved": false,
    "sha256": null,
    "strict_logit_reload": false,
    "strict_tensor_reload": false
  },
  "configuration": {
    "batch_size": 32,
    "context_length": 320,
    "correction_scale": 0.25,
    "data_seed": 63121,
    "epochs": 8,
    "execution_backend": "pytorch_eager_sdpa",
    "generation_tokens": 16,
    "gradient_clip": 1.0,
    "learning_rate": 0.0003,
    "maximum_parameter_fraction": 0.0125,
    "maximum_shuffled_exact_answers": 16,
    "maximum_total_setup_training_seconds": 2400.0,
    "maximum_training_seconds": 1800.0,
    "maximum_true_oracle_gap": 64,
    "minimum_learning_rate_fraction": 0.1,
    "minimum_oracle_exact_answers": 128,
    "minimum_true_exact_answers": 64,
    "minimum_true_source_gain": 0.2,
    "model_seed": 63131,
    "no_repeat_ngram_size": 3,
    "optimizer_steps": 2048,
    "padded_position_budget": 20971520,
    "precision": "bfloat16_parent_fp32_controller",
    "repetition_penalty": 1.1,
    "warmup_fraction": 0.05,
    "weight_decay": 0.1
  },
  "data": {
    "boundary_audit": {
      "all_delimiter_normalized_suffixes_exact": true,
      "all_token_boundaries_exact": true,
      "crossing_token_count": 0,
      "expected_record_view_count": 25344,
      "maximum_source_token_count": 280,
      "minimum_source_token_count": 5,
      "record_view_count": 25344,
      "sha256": "cf1caa523a37d343937de364cf432497c54a8ced61de584d37f6b1b81a4e6919"
    },
    "cache_policy": "online_exact_token_kv_no_persistent_hidden_cache",
    "persistent_cache_bytes": 0,
    "prepared_tensor_sha256": "7a77a0a7fd88ea5326c12e2d28cceb89cfdc0290c7f163908f2ca1f4a8638062",
    "schedule_sha256": "286027e5783e93f276e918f2f1b71bc4392cfd26a8bc764c7b667debc5548518",
    "source_mask_excludes_answer": true,
    "source_mask_excludes_labels": true,
    "source_mask_excludes_question": true,
    "source_mask_uses_answer_span": false,
    "title_intersection_count": 0,
    "train_case_count": 8192,
    "train_manifest_contract_sha256": "fef030f0c5a66381d9088cc72d38a284fd711a0a663f0e5f0d9b5376509760f7",
    "train_manifest_path": "reports\\language_curriculum\\squad-v57-native-train-8192-20260812.json",
    "train_manifest_sha256": "aae376dcf95ab887aeb67abc135b9f9f8dd1f19699935053efa8b66e5ffc9133",
    "train_title_count": 171,
    "validation_case_count": 256,
    "validation_manifest_contract_sha256": "9a6922f4ca6bd3fac5d099ba53ef33f63b66fd59b41e639785d936ca78ece15c",
    "validation_manifest_path": "reports\\language_curriculum\\squad-v57-native-validation-256-20260812.json",
    "validation_manifest_sha256": "b85f1da5d7d5c3b8bd1e9f1339ab1235028c8c8f1fb8db3b3042e3c99b3c0f80",
    "validation_title_count": 22
  },
  "decision": "retire_v39_protected_memory_adaptation",
  "experiment_contract_sha256": "23d8cce93478509c546a7f6d6002a879d11ec2a008de107b237daf6217cd8398",
  "external_llm_used": false,
  "gate": {
    "behavioral_checks": {
      "maximum_shuffled_exact_answers": true,
      "maximum_true_oracle_gap": true,
      "minimum_oracle_exact_answers": false,
      "minimum_true_exact_answers": false,
      "minimum_true_source_gain": false
    },
    "behavioral_passed": false,
    "checkpoint_passed": false,
    "mechanical_checks": {
      "active_zero_hidden_exact": true,
      "active_zero_logits_exact": true,
      "active_zero_state_exact": true,
      "al
```
