# MARULHO V56 Landmark Evidence Retrofit

## Summary

| Field | Value |
|-------|-------|
| Surface | marulho_landmark_retrofit_falsification.v1 |
| Decision | retire_v56_landmark_retrofit_capability_or_retrieval_failure |
| Owned By Marulho | true |
| External Llm Used | false |
| External Text Data | true |
| Experiment Contract Sha256 | b74eee5f98d08de98a32e82da9f090c9138ded1b0ddbadc4453a2a59e4935449 |
| Total Wall Seconds | 621.777 |
| Boundary | V56 tests frozen-parent block retrieval and causal evidence injection on heldout extractive long-context questions. It does not prove open-domain retrieval, abstractive synthesis, continual writes, or a replacement base architecture. |

## Configuration

| Field | Value |
|-------|-------|
| Epoch Count | 15 |
| Adapter Position Budget | 20643840 |
| Query Length | 72 |
| Batch Size | 32 |
| Block Tokens | 48 |
| Maximum Blocks | 5 |
| Selected Blocks | 2 |
| Retrieval Width | 128 |
| Adapter Width | 256 |
| Adapter Layers | 2 |
| Adapter Heads | 8 |
| Maximum Answer Tokens | 12 |
| Learning Rate | 0.0003 |
| Minimum Learning Rate Fraction | 0.1 |
| Warmup Fraction | 0.05 |
| Weight Decay | 0.1 |
| Gradient Clip | 1 |
| Precision | bfloat16 |
| Data Seed | 56121 |
| Model Seed | 56131 |
| Relation Case Count | 64 |
| Relation Eval Batch Size | 8 |
| Relation Generation Tokens | 16 |
| General Eval Batches | 16 |
| Maximum Cache Plus Training Seconds | 1200 |
| Maximum Parameter Fraction | 0.03 |
| Minimum Predicted Coverage | 0.8 |
| Minimum Predicted Answer Count | 64 |
| Minimum Source Gain | 0.45 |
| Minimum Oracle Answer Count | 72 |
| Maximum Predicted Oracle Gap | 10 |
| Maximum Shuffled Answer Count | 8 |

## Checkpoint

| Field | Value |
|-------|-------|
| Path | reports\language_scaling\v39-answer-objective-qualified-100m-218m-20260810.pt |
| Sha256 | 6caf97be17d49cd3fc70501b50cadd39897fd85000b121e107f13a5417a1068d |

## Data

| Field | Value |
|-------|-------|
| Training Manifest Path | reports\language_curriculum\squad-v56c-long-train-8192-20260812.json |
| Training Manifest Sha256 | ebc512f0a1d680ce3c9b0f11b52ed9a86395f035be125c5470d4b326e902a5e3 |
| Training Manifest Contract Sha256 | efd56051f98ea32fad2474e3f9504d33bec6aac4d7e69978378f3c9547d5552d |
| Validation Manifest Path | reports\language_curriculum\squad-v56-long-validation-128-20260812.json |
| Validation Manifest Sha256 | fa609b4c6c381d1d0c347fc3286dc2ed5e35daea4c57da8400b20056f0facbc6 |
| Validation Manifest Contract Sha256 | feca4f4088d3452265f2fc35240f7aa45de68dfc856e0be80af7f45a9e470a84 |
| Training Validation Case Overlap | 0 |
| Combined Cache Elapsed Seconds | 64.9132 |
| Combined Cache Host Storage Bytes | 4907335680 |

## Arm

| Field | Value |
|-------|-------|
| Architecture | frozen_v39_landmark_retrieval_causal_cross_attention |
| Processed Adapter Positions | 20643840 |
| Epoch Count | 15 |
| Optimizer Steps | 3840 |
| Positions Per Step | 5376 |
| Training Seconds | 268.021 |
| Cache Plus Training Seconds | 332.934 |
| Training Positions Per Second | 77023.2 |
| Cache Amortized Positions Per Second | 62005.7 |
| Initial Loss | 6.48435 |
| Final Loss | 2.81317 |
| Final Generator Loss | 2.45805 |
| Final Retrieval Loss | 0.355119 |
| Peak Cuda Allocated Bytes | 1236118528 |
| Optimizer State Bytes | 19067108 |
| Schedule Sha256 | ff7f18cfd914c9fee9485eb771c321c2f68e687e9d0ccccbc421bc5d4ee0d6ff |
| Warmup Steps | 192 |
| All Parameters Received Gradient | true |
| All Parameters Received Nonzero Gradient | true |
| All Parameters Received Final Gradient | true |
| All Parameters Received Final Nonzero Gradient | true |
| Parent Frozen | true |
| Replay Used | false |
| Retrofit Parameters | 2383361 |
| Parent Parameters | 100679424 |
| Experiment Contract Sha256 | b74eee5f98d08de98a32e82da9f090c9138ded1b0ddbadc4453a2a59e4935449 |

## Parent

| Field | Value |
|-------|-------|
| Checkpoint Sha256 Before | 6caf97be17d49cd3fc70501b50cadd39897fd85000b121e107f13a5417a1068d |
| Checkpoint Sha256 After | 6caf97be17d49cd3fc70501b50cadd39897fd85000b121e107f13a5417a1068d |
| Checkpoint File Exact | true |
| State Sha256 Before | 76b195a6c0706928927c0d2517e119ca30574c9917f5cbba8be048a5b1672082 |
| State Sha256 After | 76b195a6c0706928927c0d2517e119ca30574c9917f5cbba8be048a5b1672082 |
| State Exact | true |
| Tokenizer Hash Before | faca1e26aa29e897bef4e4335a0300f90e3996723d556a681b4495240f660715 |
| Tokenizer Hash After | faca1e26aa29e897bef4e4335a0300f90e3996723d556a681b4495240f660715 |
| Tokenizer Exact | true |
| Logits Exact | true |
| General Loss Before | 3.14903 |
| General Loss After | 3.14903 |
| General Loss Exact | true |
| Relation Exact | true |

## Checkpoint Fidelity

| Field | Value |
|-------|-------|
| Performed | true |
| Arm Checkpoint Path | reports\language_scaling\landmark-retrofit-v56-20m-20260812-arm.pt |
| Arm Checkpoint Sha256 | f53293ae896d51b3d93ea898a564ced5c013d1e0343b6dbd3eb39a415aeab6b2 |
| Arm Checkpoint Size Bytes | 9559858 |
| Expected State Sha256 | 306bdbf056e2ffd9a2c6e7cad0870509245d961328259edadda1d81ecb67cea3 |
| Restored State Sha256 | 306bdbf056e2ffd9a2c6e7cad0870509245d961328259edadda1d81ecb67cea3 |
| Tokenizer Hash Before | faca1e26aa29e897bef4e4335a0300f90e3996723d556a681b4495240f660715 |
| Tokenizer Hash After | faca1e26aa29e897bef4e4335a0300f90e3996723d556a681b4495240f660715 |
| Passed | true |

## Gate

| Field | Value |
|-------|-------|
| Passed | false |

## JSON Preview

```json
{
  "arm": {
    "all_parameters_received_final_gradient": true,
    "all_parameters_received_final_nonzero_gradient": true,
    "all_parameters_received_gradient": true,
    "all_parameters_received_nonzero_gradient": true,
    "architecture": "frozen_v39_landmark_retrieval_causal_cross_attention",
    "cache_amortized_positions_per_second": 62005.74562715139,
    "cache_plus_training_seconds": 332.9343078000238,
    "epoch_count": 15,
    "experiment_contract_sha256": "b74eee5f98d08de98a32e82da9f090c9138ded1b0ddbadc4453a2a59e4935449",
    "final_generator_loss": 2.458051919937134,
    "final_loss": 2.813171148300171,
    "final_missing_gradient_parameters": [],
    "final_retrieval_loss": 0.3551193177700043,
    "final_zero_gradient_parameters": [],
    "initial_loss": 6.484348773956299,
    "missing_gradient_parameters": [],
    "optimizer_state_bytes": 19067108,
    "optimizer_steps": 3840,
    "parent_frozen": true,
    "parent_parameters": 100679424,
    "peak_cuda_allocated_bytes": 1236118528,
    "positions_per_step": 5376,
    "processed_adapter_positions": 20643840,
    "replay_used": false,
    "retrofit_parameters": 2383361,
    "schedule_sha256": "ff7f18cfd914c9fee9485eb771c321c2f68e687e9d0ccccbc421bc5d4ee0d6ff",
    "source_grounding": {
      "conditions": {
        "mismatched_source": {
          "case_count": 128,
          "exact_answer_accuracy": 0.0,
          "exact_answer_count": 0,
          "rows": [
            {
              "answers": [
                "February 7, 2016",
                "February 7"
              ],
              "case_id": "56be8e613aeaaa14008c90d2",
              "continuation": "247",
              "exact_answer_match": false
            },
            {
              "answers": [
                "\"golden anniversary\"",
                "gold-themed",
                "gold"
              ],
              "case_id": "56bea9923aeaaa14008c91b9",
              "continuation": "substantial",
              "exact_answer_match": false
            },
            {
              "answers": [
                "American Football Conference"
              ],
              "case_id": "56bea9923aeaaa14008c91ba",
              "continuation": "the AFC C stands for the Air Flex",
              "exact_answer_match": false
            },
            {
              "answers": [
                "February 7, 2016",
                "February 7"
              ],
              "case_id": "56bea9923aeaaa14008c91bb",
              "continuation": "2554",
              "exact_answer_match": false
            },
            {
              "answers": [
                "Denver Broncos"
              ],
              "case_id": "56beace93aeaaa14008c91df",
              "continuation": "May 21",
              "exact_answer_match": false
            },
            {
              "answers": [
                "Levi's Stadium",
                "Levi's Stadium in the San Francisco Bay Area at Santa Clara"
              ],
              "case_id": "56beace93aeaaa14008c91e0",
              "continuation": "second party",
              "exact_answer_match": false
            },
            {
              "answers": [
                "Santa Clara"
              ],
              "case_id": "56beace93aeaaa14008c91e1",
              "continuation": "the city of",
              "exact_answer_match": false
            },
            {
              "answers": [
                "2015",
                "the 2015 season"
              ],
              "case_id": "56beace93aeaaa14008c91e3",
              "continuation": "~140",
              "exact_answer_match": false
            },
            {
              "answers": [
                "2015",
                "2016"
              ],
              "case_id": "56bf10f43aeaaa14008c94fd",
              "continuation": "1978",
              "exact_answer_match": false
            },
            {
              "answers": [
                "Santa Clara"
              ],
    
```
