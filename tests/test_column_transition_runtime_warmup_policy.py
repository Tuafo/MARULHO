import pytest
import torch

from marulho.config.model_config import MarulhoConfig
from marulho.training.model import MarulhoModel
from marulho.training.trainer import MarulhoTrainer


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA device required")
def test_inplace_runtime_skips_dense_warmup_when_candidate_gate_is_due() -> None:
    config = MarulhoConfig(
        n_columns=16,
        column_latent_dim=8,
        bootstrap_tokens=0,
        k_routing=4,
        memory_capacity=16,
        routing_index_mode="torch_topk",
        predictive_dense_transition_mode="inplace_triton",
        predictive_route_vote_mode="tensor",
        candidate_homeostasis_start_tokens=0,
        candidate_predictive_update_start_tokens=0,
        plasticity_mode="lite",
        input_weight_blend=0.0,
        enable_context_layer=False,
        enable_binding_layer=False,
        enable_abstraction_layer=False,
        device="cuda",
    )

    trainer = MarulhoTrainer(MarulhoModel(config), config)
    torch.cuda.synchronize()

    report = trainer.column_transition_runtime_report()

    assert report["active"] is True
    assert report["resolved_mode"] == "inplace_triton"
    assert report["precompiled_candidate_counts"] == [4]
