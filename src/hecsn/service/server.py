from __future__ import annotations

import argparse
from pathlib import Path

import uvicorn

from hecsn.service.api import create_app


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch the local HECSN FastAPI service")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--trace-history-limit", type=int, default=200)
    parser.add_argument("--trace-dir", type=Path, default=Path("reports") / "service" / "traces")
    parser.add_argument("--web-dist-dir", type=Path, default=Path("web") / "dist")
    parser.add_argument("--log-level", type=str, default="info")
    parser.add_argument("--reload", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    app = create_app(
        checkpoint_path=args.checkpoint,
        trace_history_limit=args.trace_history_limit,
        trace_dir=args.trace_dir,
        web_dist_dir=args.web_dist_dir,
        env_root=Path.cwd(),
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level, reload=args.reload)


if __name__ == "__main__":
    main()