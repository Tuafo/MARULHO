"""Standalone training runner for HECSN.

Trains a model from a HuggingFace dataset and saves a checkpoint.

Usage:
    PYTHONPATH=src python -m hecsn.training.train_runner \
        --output-dir checkpoints/my_run \
        --dataset-name wikitext \
        --dataset-config wikitext-2-raw-v1 \
        --text-field text \
        --max-tokens 500000
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from hecsn.config.model_config import HECSNConfig
from hecsn.training.trainer import HECSNModel, HECSNTrainer
from hecsn.training.checkpointing import save_trainer_checkpoint
from hecsn.data.corpus_loader import StreamingCorpusLoader


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
        print(f"[train_runner] Resumed from {resume_from} at token {trainer.token_count}")
    else:
        config = HECSNConfig(
            n_columns=n_columns,
            binding_mode=binding_mode,
            enable_binding_layer=True,
            enable_context_layer=True,
        )
        model = HECSNModel(config)
        trainer = HECSNTrainer(model, config)
        print(f"[train_runner] Fresh model: {n_columns} columns, {binding_mode} binding")

    loader = StreamingCorpusLoader(
        source=dataset_name,
        source_type="hf",
        text_field=text_field,
        hf_config=dataset_config,
    )

    print(f"[train_runner] Dataset: {dataset_name} ({dataset_config or 'default'})")
    print(f"[train_runner] Target: {max_tokens} tokens → {output_dir}")

    encoder = trainer.encoder
    start_tokens = trainer.token_count
    tokens_done = start_tokens
    start_time = time.time()
    last_report = start_time
    last_checkpoint = tokens_done

    for window_text, pattern_vec in encoder.iter_char_patterns(
        loader.char_stream(), window_size=config.window_size, learn=True
    ):
        trainer.train_step(pattern_vec, raw_window=window_text)
        tokens_done += 1

        now = time.time()
        if now - last_report >= 10.0:
            elapsed = now - start_time
            rate = (tokens_done - start_tokens) / max(elapsed, 1e-6)
            print(
                f"[train_runner] {tokens_done:,} tokens | "
                f"{rate:.1f} tok/s | "
                f"sleep={trainer.sleep_events} micro={trainer.micro_sleep_events} deep={trainer.deep_sleep_events} | "
                f"bootstrap={'yes' if trainer.is_bootstrap else 'no'}"
            )
            last_report = now

        if checkpoint_interval > 0 and tokens_done - last_checkpoint >= checkpoint_interval:
            ckpt_path = output_dir / f"checkpoint_{tokens_done}.pt"
            save_trainer_checkpoint(ckpt_path, trainer, {"tokens": tokens_done, "dataset": dataset_name})
            print(f"[train_runner] Checkpoint saved → {ckpt_path}")
            last_checkpoint = tokens_done

        if tokens_done >= max_tokens:
            break

    final_path = output_dir / "model.pt"
    save_trainer_checkpoint(final_path, trainer, {
        "tokens": tokens_done,
        "dataset": dataset_name,
        "dataset_config": dataset_config,
        "elapsed_seconds": time.time() - start_time,
    })

    elapsed = time.time() - start_time
    print(f"\n[train_runner] Done! {tokens_done:,} tokens in {elapsed:.1f}s ({(tokens_done - start_tokens) / max(elapsed, 1e-6):.1f} tok/s)")
    print(f"[train_runner] Checkpoint → {final_path}")

    summary = {
        "tokens": tokens_done,
        "elapsed_seconds": round(elapsed, 2),
        "tokens_per_second": round((tokens_done - start_tokens) / max(elapsed, 1e-6), 2),
        "sleep_events": trainer.sleep_events,
        "micro_sleep_events": trainer.micro_sleep_events,
        "deep_sleep_events": trainer.deep_sleep_events,
        "n_columns": config.n_columns,
        "binding_mode": config.binding_mode,
        "dataset": dataset_name,
        "dataset_config": dataset_config,
    }
    summary_path = output_dir / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[train_runner] Summary → {summary_path}")


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
