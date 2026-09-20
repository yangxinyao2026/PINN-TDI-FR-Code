# -*- coding: utf-8 -*-
"""Configure and validate the Ipopt runtime used by the experiment runners."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


def _prepend_path(directories: list[Path]) -> None:
    parts = os.environ.get("PATH", "").split(os.pathsep)
    known = {part.lower() for part in parts if part}
    prepend: list[str] = []
    for directory in directories:
        if directory.is_dir() and str(directory).lower() not in known:
            prepend.append(str(directory))
            known.add(str(directory).lower())
    if prepend:
        os.environ["PATH"] = os.pathsep.join(prepend + parts)


def _runtime_directories(root: Path) -> list[Path]:
    return [
        root / "Library" / "mingw-w64" / "bin",
        root / "Library" / "usr" / "bin",
        root / "Library" / "bin",
        root / "Scripts",
        root / "bin",
    ]


def configure_ipopt() -> Path | None:
    """PositioningIpoptand configureDLLSearch path; returned if not foundNone, Don’t actively throw errors."""
    executable_name = "ipopt.exe" if os.name == "nt" else "ipopt"
    env_root = Path(sys.prefix).resolve()
    python_root = Path(sys.executable).resolve().parent
    candidates: list[Path] = []

    configured = os.environ.get("IPOPT_EXECUTABLE")
    if configured:
        candidates.append(Path(configured))

    candidates.extend([
        env_root / "Library" / "bin" / executable_name,
        env_root / "bin" / executable_name,
        env_root / "Scripts" / executable_name,
        python_root / "Library" / "bin" / executable_name,
        python_root / "Scripts" / executable_name,
    ])

    conda_base = None
    if env_root.parent.name.lower() == "envs":
        conda_base = env_root.parent.parent
        candidates.extend([
            conda_base / "Library" / "bin" / executable_name,
            conda_base / "bin" / executable_name,
        ])

    on_path = shutil.which("ipopt")
    if on_path:
        candidates.append(Path(on_path))

    checked: list[str] = []
    for candidate in candidates:
        candidate = candidate.expanduser()
        checked.append(str(candidate))
        if not candidate.is_file():
            continue
        executable = candidate.resolve()
        roots = [env_root]
        if conda_base is not None:
            roots.append(conda_base)
        if (executable.parent.name.lower() == "bin" and
                executable.parent.parent.name.lower() == "library"):
            roots.append(executable.parent.parent.parent)
        runtime_dirs = [executable.parent]
        for root in roots:
            runtime_dirs.extend(_runtime_directories(root))
        _prepend_path(runtime_dirs)
        os.environ["IPOPT_EXECUTABLE"] = str(executable)
        configure_ipopt.checked_paths = checked
        return executable

    configure_ipopt.checked_paths = checked
    return None


def validate_ipopt(ipopt_executable: Path) -> str:
    """Actual startIpoptand returns the version string."""
    proc = subprocess.run(
        [str(ipopt_executable), "--version"], capture_output=True,
        text=True, timeout=20, check=False, env=os.environ.copy())
    output = (proc.stdout or proc.stderr or "").strip()
    if proc.returncode == 0:
        return output

    unsigned_code = proc.returncode & 0xFFFFFFFF
    dll_hint = (
        "Windowsexit code0xC0000135Express dependenceDLLMissing or unable to load.\n"
        if unsigned_code == 0xC0000135 else "")
    raise RuntimeError(
        "Foundipopt.exe, but it doesn’t start properly.\n"
        f"Ipopt: {ipopt_executable}\n"
        f"exit code: {proc.returncode} (0x{unsigned_code:08X})\n"
        f"{dll_hint}"
        f"output: {output or '<No output>'}\n"
        "Please enter the server’sAnaconda Promptexecuted in: \n"
        "  conda activate omo_bi\n"
        "  conda install -c conda-forge --force-reinstall ipopt\n"
        "and confirm `ipopt --version` Restart after normal outputPyCharm.")


def prepare_ipopt() -> dict[str, str]:
    """Configure and rigorously verifyIpopt, For formal reproduction entrance call."""
    executable = configure_ipopt()
    if executable is None:
        checked = getattr(configure_ipopt, "checked_paths", [])
        checked_text = "\n".join(f"  - {path}" for path in checked)
        raise RuntimeError(
            "currentPyCharmThe interpreter was not found in its environment.Ipopt.\n"
            f"Python: {sys.executable}\n"
            f"checked: \n{checked_text}\n"
            "Please enter the server’sAnaconda Promptexecuted in: \n"
            "  conda activate omo_bi\n"
            "  conda install -c conda-forge ipopt\n"
            "Installation is complete and restartPyCharmrun again after.")
    version = validate_ipopt(executable)
    return {"executable": str(executable), "version": version}
