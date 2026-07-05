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


def test_language_memory_slots_triton_autograd_matches_reference_when_available() -> None:
    if not torch.cuda.is_available() or not language_memory_slots_triton_stats()[
        "triton_available"
    ]:
        return
    inputs = _inputs("cuda")
    triton_hidden = inputs["hidden"].detach().clone().requires_grad_()
    triton_slots = inputs["memory_slots"].detach().clone().requires_grad_()
    triton_gate = inputs["memory_slot_gate"].detach().clone().requires_grad_()
    reference_hidden = inputs["hidden"].detach().clone().requires_grad_()
    reference_slots = inputs["memory_slots"].detach().clone().requires_grad_()
    reference_gate = inputs["memory_slot_gate"].detach().clone().requires_grad_()
    probe = torch.randn_like(triton_hidden)

    before = language_memory_slots_triton_stats()
    triton_output = language_memory_slots(
        triton_hidden,
        inputs["candidate_ids"],
        triton_slots,
        triton_gate,
        active_count=2,
        force_triton=True,
    )
    reference_output = language_memory_slots_torch_reference(
        reference_hidden,
        inputs["candidate_ids"],
        reference_slots,
        reference_gate,
        active_count=2,
    )
    (triton_output * probe).sum().backward()
    (reference_output * probe).sum().backward()
    delta = language_memory_slots_triton_stats_delta(
        before,
        language_memory_slots_triton_stats(),
    )

    torch.testing.assert_close(triton_output, reference_output, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(triton_hidden.grad, reference_hidden.grad)
    torch.testing.assert_close(triton_slots.grad, reference_slots.grad)
    torch.testing.assert_close(triton_gate.grad, reference_gate.grad)
    assert delta["triton_autograd_used"] is True
    assert delta["triton_autograd_forward_calls"] == 1
    assert delta["torch_autograd_backward_calls"] == 1


def test_language_memory_slots_triton_autograd_is_default_when_available(
    monkeypatch,
) -> None:
    if not torch.cuda.is_available() or not language_memory_slots_triton_stats()[
        "triton_available"
    ]:
        return
    monkeypatch.delenv("MARULHO_LANGUAGE_MEMORY_SLOTS_TRITON_TRAINING", raising=False)
    monkeypatch.setenv("MARULHO_LANGUAGE_MEMORY_SLOTS_TRITON_MIN_ROWS", "1")
    inputs = _inputs("cuda")
    hidden = inputs["hidden"].detach().clone().requires_grad_()
    memory_slots = inputs["memory_slots"].detach().clone().requires_grad_()
    gate = inputs["memory_slot_gate"].detach().clone().requires_grad_()

    before = language_memory_slots_triton_stats()
    output = language_memory_slots(
        hidden,
        inputs["candidate_ids"],
        memory_slots,
        gate,
        active_count=2,
    )
    output.square().mean().backward()
    delta = language_memory_slots_triton_stats_delta(
        before,
        language_memory_slots_triton_stats(),
    )

    assert delta["triton_autograd_used"] is True
    assert hidden.grad is not None
    assert memory_slots.grad is not None
    assert gate.grad is not None
