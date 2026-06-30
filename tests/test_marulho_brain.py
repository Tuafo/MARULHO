from __future__ import annotations

from pathlib import Path
import time

from fastapi.testclient import TestClient

from marulho.brain import MarulhoBrain
from marulho.config.model_config import MarulhoConfig
from marulho.service.api import create_app
from marulho.training.checkpointing import save_trainer_checkpoint


def _tiny_config() -> MarulhoConfig:
    return MarulhoConfig(
        n_columns=16,
        column_latent_dim=16,
        window_size=4,
        bootstrap_tokens=0,
        k_routing=4,
        memory_capacity=128,
        enable_learned_chunking=False,
        micro_sleep_interval_tokens=10_000,
        deep_sleep_interval_tokens=10_000,
        trainer_telemetry_interval_tokens=10_000,
        device="cpu",
    )


def _trained_brain() -> MarulhoBrain:
    brain = MarulhoBrain.fresh(_tiny_config())
    feed = brain.feed("marulho learns local sparse loops.", source="test")
    assert feed["accepted_tokens"] > 0
    tick = brain.tick(tokens=feed["accepted_tokens"], source="test")
    assert tick["trained_tokens"] == feed["accepted_tokens"]
    return brain


def test_marulho_brain_feed_tick_generate_and_trace() -> None:
    brain = _trained_brain()

    generation = brain.generate(prompt="mar", max_tokens=12)
    status = brain.status()

    assert generation["owned_by_marulho"] is True
    assert generation["external_llm_used"] is False
    assert generation["thought_loop_used"] is False
    assert generation["cortex_used"] is False
    assert generation["emitted_tokens"] > 0
    assert status["surface"] == "marulho_brain_runtime.v1"
    assert status["readout"]["observed_transition_count"] > 0
    assert status["last_trace"]["executor"] == brain.trainer.config.cuda_graph_sequence_executor
    assert status["retired_brain_surfaces"]["external_llm_used"] is False


def test_marulho_brain_checkpoint_roundtrip_preserves_readout(tmp_path: Path) -> None:
    brain = _trained_brain()
    before = brain.generate(max_tokens=12)
    checkpoint_path = tmp_path / "brain.pt"

    saved = brain.save(checkpoint_path)
    restored = MarulhoBrain.load(saved["path"])
    after = restored.generate(max_tokens=12)

    assert saved["path"] == str(checkpoint_path)
    assert restored.status()["token_count"] == brain.status()["token_count"]
    assert restored.status()["readout"]["observed_transition_count"] > 0
    assert after["text"] == before["text"]


def test_marulho_brain_replay_and_growth_reports_are_local() -> None:
    brain = _trained_brain()

    replay = brain.replay(window="micro", cycles=1)
    growth = brain.grow_prune(budget="small")

    assert replay["surface"] == "marulho_brain_replay.v1"
    assert replay["replay_updates"] >= 0
    assert growth["surface"] == "marulho_brain_growth_prune.v1"
    assert growth["growth_events"] >= 0
    assert growth["prune_events"] == 0
    assert growth["trace"]["retired_brain_surfaces"]["thought_loop_used"] is False


def test_marulho_brain_start_stop_are_brain_owned() -> None:
    brain = MarulhoBrain.fresh(_tiny_config())
    feed = brain.feed("loop owned by marulho brain.", source="loop-test")
    assert feed["accepted_tokens"] > 0

    start = brain.start(tick_tokens=4, interval_seconds=0.01, source="loop-test")
    assert start["surface"] == "marulho_brain_loop_start.v1"
    assert start["started"] is True
    assert start["loop"]["owner"] == "MarulhoBrain"
    assert start["loop"]["legacy_terminus_runtime_control"] is False

    deadline = time.time() + 1.0
    while time.time() < deadline and brain.status()["loop"]["tick_count"] == 0:
        time.sleep(0.01)

    stop = brain.stop(timeout_seconds=1.0)
    assert stop["surface"] == "marulho_brain_loop_stop.v1"
    assert stop["stopped"] is True
    assert stop["loop"]["running"] is False
    assert stop["loop"]["legacy_terminus_runtime_control"] is False
    assert brain.status()["loop"]["owner"] == "MarulhoBrain"


def test_brain_service_contract(tmp_path: Path) -> None:
    brain = MarulhoBrain.fresh(_tiny_config())
    checkpoint_path = tmp_path / "service-brain.pt"
    save_trainer_checkpoint(
        checkpoint_path,
        brain.trainer,
        metadata={"brain_state": brain.export_state()},
    )
    app = create_app(
        checkpoint_path=checkpoint_path,
        trace_dir=tmp_path / "traces",
        env_root=tmp_path,
    )
    assert app.state.marulho_manager.__class__.__name__ == "MarulhoBrainServiceManager"
    assert not hasattr(app.state.marulho_manager, "_status_read_model")
    assert not hasattr(app.state.marulho_manager, "_runtime_control")

    with TestClient(app) as client:
        feed = client.post(
            "/brain/feed",
            json={"text": "service brain learns local output.", "source": "test"},
        )
        assert feed.status_code == 200
        accepted = feed.json()["accepted_tokens"]

        tick = client.post("/brain/tick", json={"tokens": accepted, "source": "test"})
        assert tick.status_code == 200
        assert tick.json()["trained_tokens"] == accepted

        generate = client.post("/brain/generate", json={"prompt": "ser", "max_tokens": 12})
        assert generate.status_code == 200
        assert generate.json()["external_llm_used"] is False
        assert generate.json()["emitted_tokens"] > 0

        status = client.get("/brain/status")
        assert status.status_code == 200
        assert status.json()["surface"] == "marulho_brain_runtime.v1"
        assert status.json()["readout"]["observed_transition_count"] > 0

        traces = client.get("/brain/traces?limit=4")
        assert traces.status_code == 200
        assert traces.json()["surface"] == "marulho_brain_trace_history.v1"

        checkpoints = client.get("/brain/checkpoints")
        assert checkpoints.status_code == 200
        assert any(item["path"].endswith("service-brain.pt") for item in checkpoints.json()["checkpoints"])

        start = client.post(
            "/brain/start",
            json={"tick_tokens": 4, "interval_seconds": 0.01, "source": "test"},
        )
        assert start.status_code == 200
        assert start.json()["surface"] == "marulho_brain_loop_start.v1"
        assert "control" not in start.json()
        assert start.json()["loop"]["owner"] == "MarulhoBrain"

        stop = client.post("/brain/stop", json={"timeout_seconds": 1.0})
        assert stop.status_code == 200
        assert stop.json()["surface"] == "marulho_brain_loop_stop.v1"
        assert "control" not in stop.json()
        assert stop.json()["loop"]["legacy_terminus_runtime_control"] is False

        saved_path = tmp_path / "service-brain-saved.pt"
        saved = client.post("/brain/checkpoint/save", json={"path": str(saved_path)})
        assert saved.status_code == 200
        assert saved.json()["surface"] == "marulho_brain_checkpoint_save.v1"
        assert Path(saved.json()["path"]).is_file()

        restored = client.post("/brain/checkpoint/restore", json={"path": str(saved_path)})
        assert restored.status_code == 200
        assert restored.json()["restore"]["surface"] == "marulho_brain_checkpoint_restore.v1"
        assert restored.json()["brain"]["surface"] == "marulho_brain_runtime.v1"

        openapi_paths = set(client.get("/openapi.json").json()["paths"])
        assert openapi_paths == {
            "/health",
            "/",
            "/brain/status",
            "/brain/checkpoints",
            "/brain/traces",
            "/brain/feed",
            "/brain/tick",
            "/brain/generate",
            "/brain/replay",
            "/brain/grow-prune",
            "/brain/checkpoint/save",
            "/brain/checkpoint/restore",
            "/brain/start",
            "/brain/stop",
            "/brain/stream/status",
        }
        for legacy_path in (
            "/status",
            "/feed",
            "/query",
            "/respond",
            "/checkpoint/save",
            "/terminus",
            "/terminus/snn-language-readiness",
            "/stream/status",
        ):
            assert client.get(legacy_path).status_code == 404
