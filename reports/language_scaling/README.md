# Marulho Protected Bidirectional Evidence Organ Falsification

## Summary

| Field | Value |
|-------|-------|
| Artifact Kind | marulho_protected_bidirectional_evidence_organ_falsification |
| Surface | marulho_protected_bidirectional_evidence_organ_falsification.v1 |
| Owned By Marulho | true |
| External Llm Used | false |
| Decision | retire_v58_extractive_evidence_organ_capacity_failure |
| Experiment Contract Sha256 | 8fe237511560f12f1f225c20b4f9e24e9e01c73f99189af76090f78a39f12d96 |
| Random Control | - |

## Configuration

| Field | Value |
|-------|-------|
| Context Length | 320 |
| Maximum Source Characters | 1408 |
| Maximum Answer Characters | 96 |
| Maximum Token Character Offset | 64 |
| Character Feature Dim | 16 |
| Batch Size | 32 |
| Epochs | 8 |
| Optimizer Steps | 2048 |
| Padded Position Budget | 20971520 |
| Learning Rate | 0.0001 |
| Minimum Learning Rate Fraction | 0.1 |
| Warmup Fraction | 0.05 |
| Weight Decay | 0.1 |
| Gradient Clip | 1 |
| Minimum Exact Answers | 192 |
| Minimum Source Gain | 0.7 |
| Maximum Mismatched Answers | 8 |
| Minimum Initialized Advantage | 16 |
| Maximum Training Seconds | 1800 |
| Data Seed | 58121 |
| Model Seed | 58131 |
| Precision | bfloat16 |
| Execution Backend | pytorch_eager |

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
| Mechanical Oracle Exact Answer Count | 256 |
| Train Answer Offsets Corrected | 34 |
| Validation Answer Offsets Corrected | 2 |
| Schedule Sha256 | 4f63a4cd547696017908e53bc26543d32d8b3b96ff1fb094b796de56514262fb |
| Setup Seconds | 20.0963 |

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
| Parameter Count | 100679424 |

## Primary

| Field | Value |
|-------|-------|
| Arm Name | v39_initialized |
| Initialized From Parent | true |
| Parameter Count | 100686146 |
| Optimizer Steps | 2048 |
| Padded Positions | 20971520 |
| Final Training Loss | 1.96379 |
| Training Seconds | 871.752 |
| Training Positions Per Second | 24056.8 |
| Peak Cuda Bytes | 5386713600 |

## Transfer

| Field | Value |
|-------|-------|
| Random Control Required | false |
| Random Control Completed | false |
| Initialized Advantage Cases | - |
| Minimum Initialized Advantage | 16 |
| Language Pretraining Transfer Supported | false |
| Interpretation | not_run_after_primary_capability_failure |

## Checkpoint

| Field | Value |
|-------|-------|
| Saved | false |
| Path | - |
| Sha256 | - |
| Strict Tensor Reload | false |
| Strict Logit Reload | false |

## Gate

| Field | Value |
|-------|-------|
| Passed | false |
| Checkpoint Passed | false |

## JSON Preview

```json
{
  "artifact_kind": "marulho_protected_bidirectional_evidence_organ_falsification",
  "checkpoint": {
    "path": null,
    "saved": false,
    "sha256": null,
    "strict_logit_reload": false,
    "strict_tensor_reload": false
  },
  "configuration": {
    "batch_size": 32,
    "character_feature_dim": 16,
    "context_length": 320,
    "data_seed": 58121,
    "epochs": 8,
    "execution_backend": "pytorch_eager",
    "gradient_clip": 1.0,
    "learning_rate": 0.0001,
    "maximum_answer_characters": 96,
    "maximum_mismatched_answers": 8,
    "maximum_source_characters": 1408,
    "maximum_token_character_offset": 64,
    "maximum_training_seconds": 1800.0,
    "minimum_exact_answers": 192,
    "minimum_initialized_advantage": 16,
    "minimum_learning_rate_fraction": 0.1,
    "minimum_source_gain": 0.7,
    "model_seed": 58131,
    "optimizer_steps": 2048,
    "padded_position_budget": 20971520,
    "precision": "bfloat16",
    "warmup_fraction": 0.05,
    "weight_decay": 0.1
  },
  "data": {
    "mechanical_oracle_exact_answer_count": 256,
    "schedule_sha256": "4f63a4cd547696017908e53bc26543d32d8b3b96ff1fb094b796de56514262fb",
    "setup_seconds": 20.09630769999785,
    "train_answer_offsets_corrected": 34,
    "train_case_count": 8192,
    "train_manifest_contract_sha256": "fef030f0c5a66381d9088cc72d38a284fd711a0a663f0e5f0d9b5376509760f7",
    "train_manifest_path": "reports\\language_curriculum\\squad-v57-native-train-8192-20260812.json",
    "train_manifest_sha256": "aae376dcf95ab887aeb67abc135b9f9f8dd1f19699935053efa8b66e5ffc9133",
    "validation_answer_offsets_corrected": 2,
    "validation_case_count": 256,
    "validation_manifest_contract_sha256": "9a6922f4ca6bd3fac5d099ba53ef33f63b66fd59b41e639785d936ca78ece15c",
    "validation_manifest_path": "reports\\language_curriculum\\squad-v57-native-validation-256-20260812.json",
    "validation_manifest_sha256": "b85f1da5d7d5c3b8bd1e9f1339ab1235028c8c8f1fb8db3b3042e3c99b3c0f80"
  },
  "decision": "retire_v58_extractive_evidence_organ_capacity_failure",
  "experiment_contract_sha256": "8fe237511560f12f1f225c20b4f9e24e9e01c73f99189af76090f78a39f12d96",
  "external_llm_used": false,
  "gate": {
    "checkpoint_passed": false,
    "observed": {
      "primary_exact_answers": 20,
      "primary_mismatched_answers": 0,
      "primary_source_gain": 0.078125
    },
    "parent_checks": {
      "checkpoint_file_exact": true,
      "sample_logits_exact": true,
      "state_exact": true,
      "tokenizer_exact": true
    },
    "passed": false,
    "primary_checks": {
      "bounded_training_time": true,
      "capacity_ceiling_bounded": true,
      "complete_final_gradients": true,
      "exact_optimizer_steps": true,
      "exact_position_budget": true,
      "maximum_mismatched_answers": true,
      "mechanical_oracle_256": true,
      "minimum_exact_answers": false,
      "minimum_source_gain": false
    },
    "thresholds": {
      "batch_size": 32,
      "character_feature_dim": 16,
      "context_length": 320,
      "data_seed": 58121,
      "epochs": 8,
      "execution_backend": "pytorch_eager",
      "gradient_clip": 1.0,
      "learning_rate": 0.0001,
      "maximum_answer_characters": 96,
      "maximum_mismatched_answers": 8,
      "maximum_source_characters": 1408,
      "maximum_token_character_offset": 64,
      "maximum_training_seconds": 1800.0,
      "minimum_exact_answers": 192,
      "minimum_initialized_advantage": 16,
      "minimum_learning_rate_fraction": 0.1,
      "minimum_source_gain": 0.7,
      "model_seed": 58131,
      "optimizer_steps": 2048,
      "padded_position_budget": 20971520,
      "precision": "bfloat16",
      "warmup_fraction": 0.05,
      "weight_decay": 0.1
    }
  },
  "owned_by_marulho": true,
  "parent": {
    "checkpoint_sha256_after": "6caf97be17d49cd3fc70501b50cadd39897fd85000b121e107f13a5417a1068d",
    "checkpoint_sha256_before": "6caf97be17d49cd3fc70501b50cadd39897fd85000b121e107f13a5417a1068d",
    "checks": {
      "chec
```
