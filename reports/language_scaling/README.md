# Marulho Meta Gradient Episodic Matrix Falsification

## Summary

| Field | Value |
|-------|-------|
| Artifact Kind | marulho_meta_gradient_episodic_matrix_falsification |
| Surface | marulho_meta_gradient_episodic_matrix_falsification.v1 |
| Owned By Marulho | true |
| External Llm Used | false |
| Decision | retire_v60_one_step_linear_meta_gradient_memory |
| Experiment Contract Sha256 | 0cd402c55122dd0739de905e6befbeaca5f45525b5045dfb4628a7b6661df0a8 |

## Configuration

| Field | Value |
|-------|-------|
| Context Length | 96 |
| Source Chunk Length | 64 |
| Source Chunk Count | 5 |
| Source Memory Positions | 320 |
| Batch Size | 32 |
| Epochs | 8 |
| Optimizer Steps | 2048 |
| Padded Source Position Budget | 20971520 |
| Memory Heads | 8 |
| Key Width Per Head | 16 |
| Value Width Per Head | 96 |
| Learning Rate | 0.0003 |
| Warmup Fraction | 0.05 |
| Minimum Learning Rate Fraction | 0.1 |
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
| Maximum Parameter Fraction | 0.01 |
| Maximum Training Seconds | 1800 |
| Maximum Total Setup Training Seconds | 2400 |
| Data Seed | 60121 |
| Model Seed | 60131 |
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
| Train Title Count | 171 |
| Validation Title Count | 22 |
| Title Intersection Count | 0 |
| Prepared Tensor Sha256 | c011802453da81f4db953fceef3856f567d513b3a9055c69bf1e24163e0d7c02 |
| Schedule Sha256 | dae37df427c0d0de57137980a88aa8811ef23f6b808720565c7eab646b52343c |
| Cache Policy | online_frozen_v39_no_persistent_hidden_cache |
| Persistent Cache Bytes | 0 |
| Write Inputs Exclude Question | true |
| Write Inputs Exclude Answer | true |
| Write Inputs Exclude Span | true |
| Write Inputs Exclude Labels | true |

## Architecture

| Field | Value |
|-------|-------|
| Controller Parameter Count | 786449 |
| Parent Parameter Count | 100679424 |
| Controller Parameter Fraction | 0.00781142 |
| Fast State Values Per Document | 12288 |
| Initial Controller State Sha256 | b61ed0525380332eaca97a68cd99e973b2b4654801859c6351d394f5b34dec14 |
| Fast Write | one_exact_linear_reconstruction_gradient_step_from_zero |
| Source Value Owner | frozen_v39_next_token_embeddings |

## Setup

| Field | Value |
|-------|-------|
| Seconds | 11.2678 |
| Persistent Cache Bytes | 0 |

## Training

| Field | Value |
|-------|-------|
| Optimizer Steps | 2048 |
| Padded Source Positions | 20971520 |
| Answer Target Positions | 470952 |
| Final Training Loss | 5.03118 |
| Training Seconds | 372.549 |
| Source Positions Per Second | 56292 |
| Peak Cuda Bytes | 1272701952 |

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
| Total Setup Training Seconds | 383.817 |
| Total Wall Seconds | 1272.85 |
| Peak Cuda Bytes | 1272701952 |

## JSON Preview

```json
{
  "architecture": {
    "controller_parameter_count": 786449,
    "controller_parameter_fraction": 0.007811417355744903,
    "fast_state_values_per_document": 12288,
    "fast_write": "one_exact_linear_reconstruction_gradient_step_from_zero",
    "initial_controller_state_sha256": "b61ed0525380332eaca97a68cd99e973b2b4654801859c6351d394f5b34dec14",
    "parent_parameter_count": 100679424,
    "source_value_owner": "frozen_v39_next_token_embeddings"
  },
  "artifact_kind": "marulho_meta_gradient_episodic_matrix_falsification",
  "checkpoint": {
    "path": null,
    "saved": false,
    "sha256": null,
    "strict_logit_reload": false,
    "strict_tensor_reload": false
  },
  "configuration": {
    "batch_size": 32,
    "context_length": 96,
    "data_seed": 60121,
    "epochs": 8,
    "execution_backend": "pytorch_eager",
    "generation_tokens": 16,
    "gradient_clip": 1.0,
    "key_width_per_head": 16,
    "learning_rate": 0.0003,
    "maximum_parameter_fraction": 0.01,
    "maximum_shuffled_exact_answers": 16,
    "maximum_total_setup_training_seconds": 2400.0,
    "maximum_training_seconds": 1800.0,
    "maximum_true_oracle_gap": 64,
    "memory_heads": 8,
    "minimum_learning_rate_fraction": 0.1,
    "minimum_oracle_exact_answers": 128,
    "minimum_true_exact_answers": 64,
    "minimum_true_source_gain": 0.2,
    "model_seed": 60131,
    "no_repeat_ngram_size": 3,
    "optimizer_steps": 2048,
    "padded_source_position_budget": 20971520,
    "precision": "bfloat16",
    "repetition_penalty": 1.1,
    "source_chunk_count": 5,
    "source_chunk_length": 64,
    "source_memory_positions": 320,
    "value_width_per_head": 96,
    "warmup_fraction": 0.05,
    "weight_decay": 0.1
  },
  "data": {
    "cache_policy": "online_frozen_v39_no_persistent_hidden_cache",
    "persistent_cache_bytes": 0,
    "prepared_tensor_sha256": "c011802453da81f4db953fceef3856f567d513b3a9055c69bf1e24163e0d7c02",
    "schedule_sha256": "dae37df427c0d0de57137980a88aa8811ef23f6b808720565c7eab646b52343c",
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
    "validation_title_count": 22,
    "write_inputs_exclude_answer": true,
    "write_inputs_exclude_labels": true,
    "write_inputs_exclude_question": true,
    "write_inputs_exclude_span": true
  },
  "decision": "retire_v60_one_step_linear_meta_gradient_memory",
  "experiment_contract_sha256": "0cd402c55122dd0739de905e6befbeaca5f45525b5045dfb4628a7b6661df0a8",
  "external_llm_used": false,
  "gate": {
    "behavioral_checks": {
      "complete_final_gradients": true,
      "exact_optimizer_steps": true,
      "exact_position_budget": true,
      "maximum_shuffled_exact_answers": true,
      "maximum_total_setup_training_seconds": true,
      "maximum_training_seconds": true,
      "maximum_true_oracle_gap": true,
      "minimum_oracle_exact_answers": false,
      "minimum_true_exact_answers": false,
      "minimum_true_source_gain": false,
      "parameter_fraction": true,
      "parent_fidelity": true
    },
    "checkpoint_passed": false,
    "observed": {
      "oracle_exact_answers": 0,
      "shuffled_exact_answers": 0,
      "true_exact_answers": 0,
      "true_oracle_gap": 0,
      "true_source_gain": 0.0,
      "untrained_true_exact_answers": 0,
      "zero_exact_answers": 0
    },
    "passe
```
