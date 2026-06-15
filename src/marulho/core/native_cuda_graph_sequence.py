from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.cpp_extension import get_default_build_root, load


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


def _windows_sequence_build_dir() -> Path:
    cuda_tag = str(torch.version.cuda or "unknown").replace(".", "")
    path = (
        Path(get_default_build_root())
        / f"py{sys.version_info.major}{sys.version_info.minor}_cu{cuda_tag}"
        / "marulho_native_cuda_graph_sequence_win"
    )
    path.mkdir(parents=True, exist_ok=True)
    return path


def _compile_windows_cuda_object(source: Path, build_dir: Path) -> Path:
    cuda_path = os.environ.get("CUDA_PATH")
    if not cuda_path:
        raise RuntimeError("CUDA_PATH is required for native CUDA sequence build")
    nvcc = Path(cuda_path) / "bin" / "nvcc.exe"
    if not nvcc.exists():
        raise RuntimeError(f"nvcc not found at {nvcc}")
    output = build_dir / "native_cuda_graph_sequence_cuda.obj"
    stamp = output.with_suffix(".stamp")
    build_signature = "native_cuda_graph_sequence_cuda_obj_v2_md_runtime"
    rebuilt = (
        not output.exists()
        or output.stat().st_mtime < source.stat().st_mtime
        or not stamp.exists()
        or stamp.read_text(encoding="utf-8") != build_signature
    )
    if not rebuilt:
        return output
    arch_flags: list[str] = []
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        arch = f"{major}{minor}"
        arch_flags = [
            f"-gencode=arch=compute_{arch},code=compute_{arch}",
            f"-gencode=arch=compute_{arch},code=sm_{arch}",
        ]
    command = [
        str(nvcc),
        "-c",
        str(source),
        "-o",
        str(output),
        "-std=c++17",
        "-O3",
        "-Xcompiler",
        "/MD",
        *arch_flags,
    ]
    subprocess.check_output(
        command,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    stale_pyd = build_dir / "marulho_native_cuda_graph_sequence.pyd"
    if stale_pyd.exists():
        stale_pyd.unlink()
    stamp.write_text(build_signature, encoding="utf-8")
    return output


def native_cuda_graph_sequence_available() -> bool:
    return native_cuda_graph_sequence_error() is None


def native_cuda_graph_sequence_error() -> str | None:
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
    source_dir = Path(__file__).resolve().parent
    cpp_source = source_dir / "native_cuda_graph_sequence.cpp"
    cu_source = source_dir / "native_cuda_graph_sequence.cu"
    sources = [str(cpp_source), str(cu_source)]
    extra_include_paths: list[str] = []
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        extra_include_paths.append(str(Path(cuda_path) / "include"))
    build_directory: str | None = None
    extra_ldflags: list[str]
    if os.name == "nt":
        build_dir = _windows_sequence_build_dir()
        cuda_object = _compile_windows_cuda_object(cu_source, build_dir)
        sources = [str(cpp_source)]
        build_directory = str(build_dir)
        torch_lib = _torch_lib_dir()
        extra_ldflags = [
            str(cuda_object),
            f"/LIBPATH:{torch_lib}",
            "c10_cuda.lib",
            "torch_cuda.lib",
        ]
    else:
        extra_ldflags = ["-lc10_cuda", "-ltorch_cuda"]
    _MODULE = load(
        name="marulho_native_cuda_graph_sequence",
        sources=sources,
        extra_include_paths=extra_include_paths,
        extra_cflags=["/O2"] if os.name == "nt" else ["-O3"],
        extra_cuda_cflags=["-O3"],
        extra_ldflags=extra_ldflags,
        build_directory=build_directory,
        with_cuda=True,
        verbose=False,
    )
    return _MODULE


def cuda_graph_sequence_capabilities() -> dict[str, Any]:
    module = _load_extension()
    return dict(module.cuda_graph_sequence_capabilities())


def make_conditional_loop_cuda_graph_exec(
    graph: torch.cuda.CUDAGraph,
    count: int,
) -> Any:
    if count <= 0:
        raise ValueError("count must be positive")
    module = _load_extension()
    return module.make_conditional_loop_graph_exec(
        int(graph.raw_cuda_graph()),
        int(count),
    )


def replay_conditional_loop_cuda_graph_exec(graph_exec: Any) -> None:
    module = _load_extension()
    module.replay_conditional_loop_graph_exec(graph_exec)
