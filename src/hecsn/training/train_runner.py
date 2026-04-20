"""Standalone training runner for HECSN with full developmental protocol.

Trains a model through all developmental stages using a HuggingFace
dataset and saves checkpoints. Stages progressively enable layers:

  Stage 1 — Bootstrap: competitive routing + context + cross-modal
  Stage 2 — Binding: enables binding layer (hypercube/spatial/dense)
  Stage 3 — Abstraction: enables slow-feature abstraction layer

Usage:
    PYTHONPATH=src python -m hecsn.training.train_runner \
        --output-dir checkpoints/my_run \
        --dataset-name wikitext \
        --dataset-config wikitext-103-raw-v1 \
        --text-field text \
        --max-tokens 500000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from hecsn.config.model_config import HECSNConfig
from hecsn.training.trainer import HECSNModel, HECSNTrainer
from hecsn.training.checkpointing import save_trainer_checkpoint
from hecsn.data.corpus_loader import StreamingCorpusLoader


STAGE_THRESHOLDS = {
    1: 0,       # start immediately
    2: 0.20,    # enable binding after 20% of tokens
    3: 0.50,    # enable abstraction after 50% of tokens
}


def _enable_binding(trainer: HECSNTrainer, cfg: HECSNConfig) -> None:
    """Hot-enable the binding layer on an existing model."""
    if trainer.model.binding_layer is not None:
        return
    cfg.enable_binding_layer = True
    if cfg.binding_mode == "hypercube":
        from hecsn.core.hypercube import HypercubeBindingLayer
        trainer.model.binding_layer = HypercubeBindingLayer(
            n_columns=cfg.n_columns,
            device=trainer.model.device,
            threshold=cfg.binding_threshold,
            gain_strength=cfg.binding_gain_strength,
            tau_binding=cfg.binding_tau,
            stp_u_inc=cfg.binding_stp_u_inc,
            stp_tau_f=cfg.binding_stp_tau_f,
            stp_tau_d=cfg.binding_stp_tau_d,
            pv_threshold=cfg.binding_pv_threshold,
            pv_gain=cfg.binding_pv_gain,
            association_lr=cfg.binding_association_lr,
            association_decay=cfg.binding_association_decay,
        )
    elif cfg.binding_mode == "spatial":
        from hecsn.core.topographic import SpatialBindingLayer
        trainer.model.binding_layer = SpatialBindingLayer(
            n_columns=cfg.n_columns,
            device=trainer.model.device,
            threshold=cfg.binding_threshold,
            gain_strength=cfg.binding_gain_strength,
            tau_binding=cfg.binding_tau,
            stp_u_inc=cfg.binding_stp_u_inc,
            stp_tau_f=cfg.binding_stp_tau_f,
            stp_tau_d=cfg.binding_stp_tau_d,
            pv_threshold=cfg.binding_pv_threshold,
            pv_gain=cfg.binding_pv_gain,
            association_lr=cfg.binding_association_lr,
            association_decay=cfg.binding_association_decay,
        )
    else:
        from hecsn.core.context import BindingLayer
        trainer.model.binding_layer = BindingLayer(
            n_columns=cfg.n_columns,
            device=trainer.model.device,
            threshold=cfg.binding_threshold,
            gain_strength=cfg.binding_gain_strength,
            tau_binding=cfg.binding_tau,
            stp_u_inc=cfg.binding_stp_u_inc,
            stp_tau_f=cfg.binding_stp_tau_f,
            stp_tau_d=cfg.binding_stp_tau_d,
            pv_threshold=cfg.binding_pv_threshold,
            pv_gain=cfg.binding_pv_gain,
            association_lr=cfg.binding_association_lr,
            association_decay=cfg.binding_association_decay,
        )


def _enable_abstraction(trainer: HECSNTrainer, cfg: HECSNConfig) -> None:
    """Hot-enable the abstraction layer on an existing model."""
    if trainer.model.abstraction_layer is not None:
        return
    cfg.enable_abstraction_layer = True
    from hecsn.core.abstraction import AbstractionLayer
    trainer.model.abstraction_layer = AbstractionLayer(
        n_columns=cfg.n_columns,
        n_concepts=cfg.abstraction_n_concepts,
        device=trainer.model.device,
    )


def train(
    output_dir: Path,
    dataset_name: str,
    dataset_config: str | None,
    text_field: str,
    max_tokens: int,
    n_columns: int,
    binding_mode: str,
    checkpoint_interval: int,
    resume_from: str | None,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    if resume_from:
        from hecsn.training.checkpointing import load_trainer_checkpoint
        trainer, meta = load_trainer_checkpoint(resume_from)
        config = trainer.config
        print(f"[train] Resumed from {resume_from} at token {trainer.token_count}")
    else:
        config = HECSNConfig(
            n_columns=n_columns,
            binding_mode=binding_mode,
            enable_context_layer=True,
            enable_cross_modal=True,
            context_mode="adaptive",
            plasticity_rule="triplet",
            plasticity_mode="local_stdp",
            plasticity_spike_backend="adex",
        )
        model = HECSNModel(config)
        trainer = HECSNTrainer(model, config)
        trainer.developmental_stage = 1
        print(f"[train] Fresh model: {n_columns} columns, {binding_mode} binding")

    loader = StreamingCorpusLoader(
        source=dataset_name,
        source_type="hf",
        text_field=text_field,
        hf_config=dataset_config,
    )

    print(f"[train] Dataset: {dataset_name} ({dataset_config or 'default'})")
    print(f"[train] Target: {max_tokens:,} tokens → {output_dir}")
    print(f"[train] Stage plan: 1→bootstrap, 2→binding @ {STAGE_THRESHOLDS[2]:.0%}, 3→abstraction @ {STAGE_THRESHOLDS[3]:.0%}")

    encoder = trainer.encoder
    start_tokens = trainer.token_count
    tokens_done = start_tokens
    current_stage = trainer.developmental_stage
    start_time = time.time()
    last_report = start_time
    last_checkpoint = tokens_done

    for window_text, pattern_vec in encoder.iter_char_patterns(
        loader.char_stream(), window_size=config.window_size, learn=True
    ):
        # Check developmental stage transitions
        progress = (tokens_done - start_tokens) / max(max_tokens, 1)
        if current_stage < 2 and progress >= STAGE_THRESHOLDS[2]:
            print(f"\n[train] ═══ Stage 2: Enabling binding layer ({binding_mode}) at {tokens_done:,} tokens ═══")
            _enable_binding(trainer, config)
            trainer.developmental_stage = 2
            current_stage = 2
        elif current_stage < 3 and progress >= STAGE_THRESHOLDS[3]:
            print(f"\n[train] ═══ Stage 3: Enabling abstraction layer at {tokens_done:,} tokens ═══")
            _enable_abstraction(trainer, config)
            trainer.developmental_stage = 3
            current_stage = 3

        trainer.train_step(pattern_vec, raw_window=window_text)
        tokens_done += 1

        now = time.time()
        if now - last_report >= 10.0:
            elapsed = now - start_time
            rate = (tokens_done - start_tokens) / max(elapsed, 1e-6)
            print(
                f"[train] {tokens_done:,} tokens | "
                f"{rate:.1f} tok/s | "
                f"stage={current_stage} | "
                f"sleep={trainer.sleep_events} micro={trainer.micro_sleep_events} deep={trainer.deep_sleep_events} | "
                f"bootstrap={'yes' if trainer.is_bootstrap else 'no'}"
            )
            last_report = now

        if checkpoint_interval > 0 and tokens_done - last_checkpoint >= checkpoint_interval:
            ckpt_path = output_dir / f"checkpoint_{tokens_done}.pt"
            save_trainer_checkpoint(ckpt_path, trainer, {
                "tokens": tokens_done,
                "dataset": dataset_name,
                "stage": current_stage,
            })
            print(f"[train] Checkpoint saved → {ckpt_path}")
            last_checkpoint = tokens_done

        if tokens_done >= max_tokens:
            break

    final_path = output_dir / "model.pt"
    save_trainer_checkpoint(final_path, trainer, {
        "tokens": tokens_done,
        "dataset": dataset_name,
        "dataset_config": dataset_config,
        "elapsed_seconds": time.time() - start_time,
        "final_stage": current_stage,
    })

    elapsed = time.time() - start_time
    trained = tokens_done - start_tokens
    print(f"\n[train] Done! {tokens_done:,} tokens in {elapsed:.1f}s ({trained / max(elapsed, 1e-6):.1f} tok/s)")
    print(f"[train] Final stage: {current_stage}")
    print(f"[train] Checkpoint → {final_path}")

    summary = {
        "tokens": tokens_done,
        "elapsed_seconds": round(elapsed, 2),
        "tokens_per_second": round(trained / max(elapsed, 1e-6), 2),
        "sleep_events": trainer.sleep_events,
        "micro_sleep_events": trainer.micro_sleep_events,
        "deep_sleep_events": trainer.deep_sleep_events,
        "n_columns": config.n_columns,
        "binding_mode": config.binding_mode,
        "final_stage": current_stage,
        "dataset": dataset_name,
        "dataset_config": dataset_config,
    }
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[train] Summary → {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an HECSN model from a HuggingFace dataset"
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for checkpoints and summary")
    parser.add_argument("--dataset-name", type=str, required=True, help="HuggingFace dataset name (e.g. wikitext)")
    parser.add_argument("--dataset-config", type=str, default=None, help="HF dataset config (e.g. wikitext-2-raw-v1)")
    parser.add_argument("--text-field", type=str, default="text", help="Field name containing text")
    parser.add_argument("--max-tokens", type=int, default=500_000, help="Maximum tokens to train on")
    parser.add_argument("--n-columns", type=int, default=256, help="Number of competitive columns")
    parser.add_argument("--binding-mode", type=str, default="hypercube", choices=["dense", "spatial", "hypercube"], help="Binding layer topology")
    parser.add_argument("--checkpoint-interval", type=int, default=100_000, help="Save checkpoint every N tokens (0 to disable)")
    parser.add_argument("--resume-from", type=str, default=None, help="Resume from a .pt checkpoint file")
    args = parser.parse_args()

    train(
        output_dir=args.output_dir,
        dataset_name=args.dataset_name,
        dataset_config=args.dataset_config,
        text_field=args.text_field,
        max_tokens=args.max_tokens,
        n_columns=args.n_columns,
        binding_mode=args.binding_mode,
        checkpoint_interval=args.checkpoint_interval,
        resume_from=args.resume_from,
    )


if __name__ == "__main__":
    main()
