import torch

from marulho.core.language_memory_slots_triton import (
    can_use_language_memory_slots_triton,
    language_memory_slots,
    language_memory_slots_torch_reference,
    language_memory_slots_triton_stats,
    language_memory_slots_triton_stats_delta,
)


def _inputs(device: torch.device | str = "cpu") -> dict[str, torch.Tensor]:
    torch.manual_seed(20260705)
    hidden = torch.randn(64, 16, device=device)
    memory_slots = torch.randn(32, 16, device=device)
    row_offsets = torch.arange(64, device=device, dtype=torch.long).unsqueeze(1)
    candidate_offsets = torch.arange(4, device=device, dtype=torch.long).unsqueeze(0)
    candidate_ids = (row_offsets + candidate_offsets).remainder(32)
    gate = torch.tensor(0.35, device=device)
    return {
        "hidden": hidden,
        "candidate_ids": candidate_ids,
        "memory_slots": memory_slots,
        "memory_slot_gate": gate,
    }


def test_language_memory_slots_cpu_matches_reference() -> None:
    inputs = _inputs()

    output = language_memory_slots(**inputs, active_count=2)
    reference = language_memory_slots_torch_reference(**inputs, active_count=2)

    torch.testing.assert_close(output, reference)


def test_language_memory_slots_requires_no_grad_for_triton() -> None:
    if not torch.cuda.is_available() or not language_memory_slots_triton_stats()[
        "triton_available"
    ]:
        return
    inputs = _inputs("cuda")

    assert can_use_language_memory_slots_triton(**inputs, active_count=2) is False
    with torch.no_grad():
        assert can_use_language_memory_slots_triton(**inputs, active_count=2) is True


def test_language_memory_slots_triton_matches_reference_when_available() -> None:
    if not torch.cuda.is_available() or not language_memory_slots_triton_stats()[
        "triton_available"
    ]:
        return
    inputs = _inputs("cuda")
    reference_inputs = {key: value.detach().clone() for key, value in inputs.items()}

    with torch.no_grad():
        before = language_memory_slots_triton_stats()
        output = language_memory_slots(**inputs, active_count=2, force_triton=True)
        reference = language_memory_slots_torch_reference(
            **reference_inputs,
            active_count=2,
        )
        delta = language_memory_slots_triton_stats_delta(
            before,
            language_memory_slots_triton_stats(),
        )

    torch.testing.assert_close(output, reference, atol=1e-5, rtol=1e-5)
    assert delta["triton_kernel_used"] is True
    assert delta["triton_forward_calls"] == 1
