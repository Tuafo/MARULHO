from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import load


_MODULE: Any | None = None
_LOAD_ERROR: str | None = None


def _prepend_path(path: str | None) -> None:
    if not path:
        return
    if path in os.environ.get("PATH", "").split(os.pathsep):
        return
    os.environ["PATH"] = path + os.pathsep + os.environ.get("PATH", "")


def _load_msvc_environment() -> None:
    if os.name != "nt" or shutil.which("cl") is not None:
        return
    candidates = [
        Path(r"C:\PROGRA~1\MICROS~1\2022\COMMUN~1\VC\AUXILI~1\Build\VCVARS~1.BAT"),
        Path(
            r"C:\Program Files\Microsoft Visual Studio\2022\Community"
            r"\VC\Auxiliary\Build\vcvarsall.bat"
        ),
    ]
    vcvars = next((path for path in candidates if path.exists()), None)
    if vcvars is None:
        return
    output = subprocess.check_output(
        ["cmd", "/d", "/s", "/c", f"call {vcvars} x64 >nul && set"],
        text=True,
        encoding="mbcs",
        errors="replace",
    )
    for line in output.splitlines():
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key] = value


def _load_ninja_path() -> None:
    try:
        import ninja
    except ImportError:
        return
    _prepend_path(getattr(ninja, "BIN_DIR", None))


def _torch_lib_dir() -> Path:
    return Path(torch.__file__).resolve().parent / "lib"


def native_cuda_graph_replay_available() -> bool:
    return native_cuda_graph_replay_error() is None


def native_cuda_graph_replay_error() -> str | None:
    global _LOAD_ERROR
    if _MODULE is not None:
        return None
    if _LOAD_ERROR is not None:
        return _LOAD_ERROR
    try:
        _load_extension()
    except Exception as exc:  # pragma: no cover - environment dependent
        _LOAD_ERROR = f"{type(exc).__name__}: {exc}"
    return _LOAD_ERROR


def _load_extension() -> Any:
    global _MODULE
    if _MODULE is not None:
        return _MODULE
    _load_ninja_path()
    _load_msvc_environment()
    source = Path(__file__).with_suffix(".cpp")
    extra_include_paths: list[str] = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        extra_include_paths.append(str(Path(cuda_path) / "include"))
    extra_ldflags: list[str]
    if os.name == "nt":
        torch_lib = _torch_lib_dir()
        extra_ldflags = [
            f"/LIBPATH:{torch_lib}",
            "c10_cuda.lib",
            "torch_cuda.lib",
        ]
    else:
        extra_ldflags = ["-lc10_cuda", "-ltorch_cuda"]
    _MODULE = load(
        name="marulho_native_cuda_graph_replay",
        sources=[str(source)],
        extra_include_paths=extra_include_paths,
        extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
        extra_ldflags=extra_ldflags,
        with_cuda=False,
        verbose=False,
    )
    return _MODULE


def make_repeated_cuda_graph_exec(
    graph: torch.cuda.CUDAGraph,
    count: int,
) -> Any:
    if count <= 0:
        raise ValueError("count must be positive")
    module = _load_extension()
    return module.make_repeated_graph_exec(int(graph.raw_cuda_graph()), int(count))


def replay_repeated_cuda_graph_exec(graph_exec: Any) -> None:
    module = _load_extension()
    module.replay_repeated_graph_exec(graph_exec)
