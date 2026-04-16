"""10M-token scale test for HECSN full architecture.

Streams wikitext-103 via HuggingFace, trains with all layers
(context, binding, cross-modal, STDP), and logs throughput,
memory, and grounding metrics at regular intervals.

Usage:
    python scripts/scale_test_10m.py [--tokens 10000000] [--cols 256] [--seed 42]
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import torch

# Ensure src/ is on the path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from hecsn.config.model_config import HECSNConfig
from hecsn.data.corpus_loader import StreamingCorpusLoader
from hecsn.data.rtf_encoder import RTFEncoder
from hecsn.training.trainer import HECSNModelLite, HECSNTrainer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HECSN 10M-token scale test")
    p.add_argument("--tokens", type=int, default=10_000_000)
    p.add_argument("--cols", type=int, default=256)
    p.add_argument("--latent-dim", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-interval", type=int, default=10_000,
                   help="Tokens between progress logs")
    p.add_argument("--checkpoint-interval", type=int, default=500_000,
                   help="Tokens between checkpoint saves")
    p.add_argument("--output-dir", type=str, default="reports/scale_10m")
    p.add_argument("--source", type=str, default="wikitext",
                   help="HuggingFace dataset or file path")
    p.add_argument("--hf-config", type=str, default="wikitext-103-raw-v1")
    return p.parse_args()


def get_memory_mb() -> float:
    """Approximate process memory in MB."""
    try:
        import psutil
        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        return 0.0


def run_scale_test(args: argparse.Namespace) -> None:
    torch.manual_seed(args.seed)

    config = HECSNConfig(
        n_columns=args.cols,
        column_latent_dim=args.latent_dim,
        plasticity_mode="local_stdp",
        enable_context_layer=True,
        enable_learned_chunking=True,
        enable_binding_layer=True,
        binding_mode="spatial",
        enable_abstraction_layer=True,
        enable_cross_modal=True,
        cross_modal_dim_visual=32,
        cross_modal_dim_audio=32,
    )

    model = HECSNModelLite(config)
    trainer = HECSNTrainer(model, config)
    encoder = trainer.encoder

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loader = StreamingCorpusLoader(
        source=args.source,
        source_type="hf",
        hf_config=args.hf_config,
    )

    print(f"=== HECSN 10M Scale Test ===")
    print(f"Target: {args.tokens:,} tokens, {args.cols} columns, dim={args.latent_dim}")
    print(f"Source: {args.source} / {args.hf_config}")
    print(f"Output: {output_dir}")
    print(f"Started: {datetime.now().isoformat()}")
    print()

    interval_logs: list[dict] = []
    t_start = time.perf_counter()
    t_interval = t_start
    interval_recon_sum = 0.0
    interval_sleep_count = 0
    token_count = 0

    for ch in loader.char_stream():
        code = ord(ch) if ord(ch) < config.n_ascii else 0
        pattern = encoder.spike_trace(
            chars=torch.tensor([code], dtype=torch.long),
            context_confidence=0.0,
        )
        if pattern is None or float(pattern.sum().item()) <= 0.0:
            continue

        row = trainer.train_step(pattern, raw_window=ch)
        token_count += 1
        interval_recon_sum += float(row.get("recon_error", 0.0))
        if row.get("sleep_type", "none") != "none":
            interval_sleep_count += 1

        if token_count % args.log_interval == 0:
            now = time.perf_counter()
            elapsed = now - t_start
            interval_dt = now - t_interval
            interval_tps = args.log_interval / max(interval_dt, 0.001)
            overall_tps = token_count / max(elapsed, 0.001)
            mem_mb = get_memory_mb()
            avg_recon = interval_recon_sum / args.log_interval
            eta_s = (args.tokens - token_count) / max(overall_tps, 0.1)

            entry = {
                "token": token_count,
                "elapsed_s": round(elapsed, 1),
                "interval_tok_s": round(interval_tps, 1),
                "overall_tok_s": round(overall_tps, 1),
                "memory_mb": round(mem_mb, 1),
                "avg_recon_error": round(avg_recon, 5),
                "sleep_events": interval_sleep_count,
                "eta_hours": round(eta_s / 3600, 2),
            }
            interval_logs.append(entry)

            print(
                f"[{token_count:>10,}/{args.tokens:,}] "
                f"{interval_tps:6.1f} tok/s (overall {overall_tps:.1f}) "
                f"mem={mem_mb:.0f}MB recon={avg_recon:.4f} "
                f"sleep={interval_sleep_count} "
                f"ETA={timedelta(seconds=int(eta_s))}"
            )

            t_interval = now
            interval_recon_sum = 0.0
            interval_sleep_count = 0

        if token_count % args.checkpoint_interval == 0:
            ckpt_path = output_dir / f"checkpoint_{token_count}.pt"
            torch.save({
                "token_count": token_count,
                "model_state": model.state_dict(),
                "elapsed_s": time.perf_counter() - t_start,
            }, ckpt_path)
            print(f"  >> Checkpoint saved: {ckpt_path}")

        if token_count >= args.tokens:
            break

    t_end = time.perf_counter()
    total_s = t_end - t_start
    final_tps = token_count / max(total_s, 0.001)

    summary = {
        "total_tokens": token_count,
        "total_seconds": round(total_s, 1),
        "total_hours": round(total_s / 3600, 2),
        "final_tok_s": round(final_tps, 1),
        "n_columns": args.cols,
        "latent_dim": args.latent_dim,
        "seed": args.seed,
        "source": f"{args.source}/{args.hf_config}",
        "finished": datetime.now().isoformat(),
        "interval_logs": interval_logs,
    }

    summary_path = output_dir / "scale_test_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print()
    print(f"=== COMPLETE ===")
    print(f"Tokens: {token_count:,} in {timedelta(seconds=int(total_s))}")
    print(f"Throughput: {final_tps:.1f} tok/s")
    print(f"Summary: {summary_path}")


if __name__ == "__main__":
    run_scale_test(parse_args())
