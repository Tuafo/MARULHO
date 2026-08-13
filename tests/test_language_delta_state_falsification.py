import torch

from marulho.evaluation.language_delta_state_falsification import (
    V64Config,
    preflight_decision,
    weighted_causal_loss,
)
from marulho.training.language_delta_state import (
    DeltaStateLanguageModelConfig,
    MarulhoDeltaStateLanguageModel,
)


def _arm(*, throughput: float = 80.0, maximum_delta: float = 0.001):
    return {
        "positions_per_second": throughput,
        "parameter_count": 100,
        "peak_cuda_bytes": 1000,
        "compiled_gradients": {"all_present_finite_nonzero": True},
        "compiled_eager_parity": {
            "loss_absolute_delta": 0.0001,
            "gradients": {
                "names_equal": True,
                "global_cosine": 0.9999,
                "maximum_absolute_element_delta": maximum_delta,
            },
        },
    }


def test_preflight_decision_requires_every_execution_gate() -> None:
    config = V64Config(maximum_peak_cuda_bytes=2000)
    decision, gates = preflight_decision(
        _arm(throughput=60.0), _arm(throughput=100.0), config
    )
    assert decision == "advance_v64_to_terminal_training"
    assert all(gates.values())
    failed = _arm(throughput=49.0)
    decision, gates = preflight_decision(failed, _arm(throughput=100.0), config)
    assert decision == "stop_v64_for_kernel_redesign_no_quality_verdict"
    assert gates["throughput_floor"] is False


def test_weighted_loss_trains_every_position_and_emphasizes_suffix() -> None:
    config = DeltaStateLanguageModelConfig(
        vocab_size=41,
        embedding_dim=16,
        state_dim=16,
        state_layers=4,
        attention_heads=2,
        transformer_context_length=12,
        transformer_mlp_ratio=2.0,
        local_attention_window=4,
        delta_chunk_size=3,
    )
    model = MarulhoDeltaStateLanguageModel(config)
    inputs = torch.randint(0, config.vocab_size, (2, 9))
    targets = torch.randint(0, config.vocab_size, (2, 9))
    ordinary = weighted_causal_loss(model, inputs, targets, torch.ones_like(inputs))
    weights = torch.ones_like(inputs)
    weights[:, -3:] = 4
    emphasized = weighted_causal_loss(model, inputs, targets, weights)
    assert torch.isfinite(ordinary)
    assert torch.isfinite(emphasized)
    assert not torch.equal(ordinary, emphasized)
