# Marulho Source Native Write Time Learning Falsification

## Summary

| Field | Value |
|-------|-------|
| Artifact Kind | marulho_source_native_write_time_learning_falsification |
| Surface | marulho_source_native_write_time_learning_falsification.v1 |
| Owned By Marulho | true |
| External Llm Used | false |
| Decision | retire_v59_naive_source_only_gradient_memory |
| Experiment Contract Sha256 | 2c7b5676241e5df5e154a0bb214e3696425a83ceedd3882fb3160fd613aee5c3 |

## Configuration

| Field | Value |
|-------|-------|
| Panel Case Count | 64 |
| Expected Panel Title Count | 22 |
| Context Length | 72 |
| Write Epochs | 4 |
| Learning Rate | 0.0001 |
| Weight Decay | 0 |
| Gradient Clip | 1 |
| Generation Tokens | 16 |
| Repetition Penalty | 1.1 |
| No Repeat Ngram Size | 3 |
| Minimum True Exact Answers | 16 |
| Minimum True Control Margin | 12 |
| Maximum Mismatched Exact Answers | 8 |
| Minimum Oracle Exact Answers | 24 |
| Minimum True Loss Improvement Fraction | 0.9 |
| Maximum Total Wall Seconds | 2400 |
| Precision | bfloat16 |
| Execution Backend | pytorch_eager |

## Data

| Field | Value |
|-------|-------|
| Manifest Path | reports\language_curriculum\squad-v57-native-validation-256-20260812.json |
| Manifest Sha256 | b85f1da5d7d5c3b8bd1e9f1339ab1235028c8c8f1fb8db3b3042e3c99b3c0f80 |
| Manifest Contract Sha256 | 9a6922f4ca6bd3fac5d099ba53ef33f63b66fd59b41e639785d936ca78ece15c |
| Panel Case Count | 64 |
| Panel Title Count | 22 |
| Panel Case Ids Sha256 | 185a9963bd28d53f04d075cc54937e0d6ca75ffc7719ac5979359ca1ee84e94f |
| Write Inputs Exclude Question | true |
| Write Inputs Exclude Answer | true |
| Write Inputs Exclude Span | true |
| Write Inputs Exclude Labels | true |

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

## Runtime

| Field | Value |
|-------|-------|
| Total Wall Seconds | 388.299 |
| Peak Cuda Bytes | 1125761536 |

## Gate

| Field | Value |
|-------|-------|
| Passed | false |

## Checkpoint

| Field | Value |
|-------|-------|
| Transient Case States Saved | false |
| Durable Candidate Saved | false |
| Policy | reset_per_case_then_discard |

## JSON Preview

```json
{
  "arms": {
    "mismatched_write": {
      "all_resets_exact": true,
      "arm_name": "mismatched_write",
      "case_count": 64,
      "exact_answer_accuracy": 0.0,
      "exact_answer_count": 0,
      "final_gradients": {
        "all_trainable_tensors_nonzero": true,
        "by_parameter": {
          "state_block.layers.0.attention.output.weight": true,
          "state_block.layers.0.attention.qkv.weight": true,
          "state_block.layers.0.attention_norm.weight": true,
          "state_block.layers.0.down.weight": true,
          "state_block.layers.0.gate_up.weight": true,
          "state_block.layers.0.mlp_norm.weight": true,
          "state_block.layers.1.attention.output.weight": true,
          "state_block.layers.1.attention.qkv.weight": true,
          "state_block.layers.1.attention_norm.weight": true,
          "state_block.layers.1.down.weight": true,
          "state_block.layers.1.gate_up.weight": true,
          "state_block.layers.1.mlp_norm.weight": true,
          "state_block.layers.2.attention.output.weight": true,
          "state_block.layers.2.attention.qkv.weight": true,
          "state_block.layers.2.attention_norm.weight": true,
          "state_block.layers.2.down.weight": true,
          "state_block.layers.2.gate_up.weight": true,
          "state_block.layers.2.mlp_norm.weight": true,
          "state_block.layers.3.attention.output.weight": true,
          "state_block.layers.3.attention.qkv.weight": true,
          "state_block.layers.3.attention_norm.weight": true,
          "state_block.layers.3.down.weight": true,
          "state_block.layers.3.gate_up.weight": true,
          "state_block.layers.3.mlp_norm.weight": true,
          "state_block.layers.4.attention.output.weight": true,
          "state_block.layers.4.attention.qkv.weight": true,
          "state_block.layers.4.attention_norm.weight": true,
          "state_block.layers.4.down.weight": true,
          "state_block.layers.4.gate_up.weight": true,
          "state_block.layers.4.mlp_norm.weight": true,
          "state_block.layers.5.attention.output.weight": true,
          "state_block.layers.5.attention.qkv.weight": true,
          "state_block.layers.5.attention_norm.weight": true,
          "state_block.layers.5.down.weight": true,
          "state_block.layers.5.gate_up.weight": true,
          "state_block.layers.5.mlp_norm.weight": true,
          "state_block.layers.6.attention.output.weight": true,
          "state_block.layers.6.attention.qkv.weight": true,
          "state_block.layers.6.attention_norm.weight": true,
          "state_block.layers.6.down.weight": true,
          "state_block.layers.6.gate_up.weight": true,
          "state_block.layers.6.mlp_norm.weight": true,
          "state_block.layers.7.attention.output.weight": true,
          "state_block.layers.7.attention.qkv.weight": true,
          "state_block.layers.7.attention_norm.weight": true,
          "state_block.layers.7.down.weight": true,
          "state_block.layers.7.gate_up.weight": true,
          "state_block.layers.7.mlp_norm.weight": true,
          "state_block.layers.8.attention.output.weight": true,
          "state_block.layers.8.attention.qkv.weight": true,
          "state_block.layers.8.attention_norm.weight": true,
          "state_block.layers.8.down.weight": true,
          "state_block.layers.8.gate_up.weight": true,
          "state_block.layers.8.mlp_norm.weight": true,
          "state_block.layers.9.attention.output.weight": true,
          "state_block.layers.9.attention.qkv.weight": true,
          "state_block.layers.9.attention_norm.weight": true,
          "state_block.layers.9.down.weight": true,
          "state_block.layers.9.gate_up.weight": true,
          "state_block.layers.9.mlp_norm.weight": true,
          "state_block.output_norm.weight": true,
          "token_embedding.weight": true
        },
        "nonzero_tensor_count": 62,
        "tensor_count": 62
      },
      "optimizer_steps":
```
