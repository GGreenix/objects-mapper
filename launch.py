#!/usr/bin/env python3
"""One-click setup and start, run by start.bat / start.sh with the project's .venv Python.

First run: installs PyTorch (the CUDA build that matches your NVIDIA driver, or a
CPU build if there is no NVIDIA GPU) and everything in requirements.txt.
Later runs: starts straight away. Setup is redone only if requirements.txt
changes, or when started with --reinstall.

Only the standard library is used here, because nothing else is installed yet.
"""

from __future__ import annotations

import hashlib
import json
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REQUIREMENTS = ROOT / "requirements.txt"
STAMP = Path(sys.prefix) / "fieldmapper-setup.json"
TORCH_INDEX = "https://download.pytorch.org/whl/"
# Newest first. A build is only tried if the driver supports at least that CUDA version.
CUDA_BUILDS = [((13, 0), "cu130"), ((12, 8), "cu128"), ((12, 6), "cu126"), ((12, 4), "cu124"), ((12, 1), "cu121"), ((11, 8), "cu118")]


def say(msg: str) -> None:
    print(f"[setup] {msg}", flush=True)


def pip(*args: str) -> bool:
    return subprocess.call([sys.executable, "-m", "pip", *args]) == 0


def driver_cuda_version() -> tuple[int, int] | None:
    """Highest CUDA version the installed NVIDIA driver supports, from nvidia-smi."""
    exe = shutil.which("nvidia-smi")
    if not exe:
        return None
    try:
        out = subprocess.run([exe], capture_output=True, text=True, timeout=30).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", out)
    return (int(m.group(1)), int(m.group(2))) if m else None


def torch_status() -> tuple[bool, bool]:
    """(installed, cuda_works), checked in a subprocess so this process never imports torch."""
    code = "import torch; print(int(torch.cuda.is_available()))"
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode != 0:
        return False, False
    return True, r.stdout.strip().endswith("1")


def install_torch() -> None:
    cuda = driver_cuda_version()
    installed, cuda_ok = torch_status()
    if cuda is None:
        if installed:
            return
        say("No NVIDIA GPU found - installing CPU PyTorch (detection will be slower).")
        if platform.system() == "Linux":  # PyPI's Linux wheel is a multi-GB CUDA build; use the CPU one
            pip("install", "torch", "torchvision", "--index-url", TORCH_INDEX + "cpu")
        else:
            pip("install", "torch", "torchvision")
        return
    if installed and cuda_ok:
        say("GPU PyTorch already installed.")
        return
    say(f"NVIDIA driver supports CUDA {cuda[0]}.{cuda[1]} - installing GPU PyTorch (large one-time download).")
    for version, tag in CUDA_BUILDS:
        if version > cuda:
            continue
        args = ["install", "torch", "torchvision", "--index-url", TORCH_INDEX + tag]
        if installed:
            args[1:1] = ["--force-reinstall", "--no-deps"]
        if pip(*args) and torch_status()[1]:
            say(f"GPU PyTorch ({tag}) installed and working.")
            return
        installed = torch_status()[0]
        say(f"{tag} build didn't work here, trying an older one...")
    say("Could not install a working GPU PyTorch. Continuing on CPU; updating the NVIDIA driver usually fixes this.")
    if not torch_status()[0]:
        pip("install", "torch", "torchvision")


def requirements_hash() -> str:
    return hashlib.sha256(REQUIREMENTS.read_bytes()).hexdigest()[:16]


def setup_needed(force: bool) -> bool:
    if force:
        return True
    try:
        return json.loads(STAMP.read_text()).get("requirements") != requirements_hash()
    except (OSError, ValueError):
        return True


def setup() -> bool:
    say("Installing Field Mapper's dependencies. The first time takes a few minutes and needs internet.")
    pip("install", "--upgrade", "pip")
    install_torch()  # before the requirements, so nothing pulls in a CPU-only torch first
    if not pip("install", "-r", str(REQUIREMENTS)):
        say("Installing the requirements failed. Check the internet connection and run start again.")
        return False
    STAMP.write_text(json.dumps({"requirements": requirements_hash()}))
    say("Setup complete.")
    return True


def main() -> int:
    if sys.version_info[:2] < (3, 11):
        say(f"Python {platform.python_version()} is too old; 3.11 or newer is needed.")
        return 1
    args = sys.argv[1:]
    force = "--reinstall" in args
    args = [a for a in args if a != "--reinstall"]
    if setup_needed(force) and not setup():
        return 1
    sys.path.insert(0, str(ROOT))
    from field_mapper.app import run

    try:
        return run(["--config", str(ROOT / "config.yaml"), *args])
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    sys.exit(main())
