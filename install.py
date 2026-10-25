"""Set up the camera PC: venv, torch, this package and the baked model.

Usage: python install.py [--cpu] [--cuda cu128] [--reinstall-torch] [--skip-prepare]
Safe to rerun. Uses only the standard library.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import venv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
DEFAULT_CUDA = os.environ.get("UTRA_TORCH_CUDA", "cu128")
TORCH_INDEX = "https://download.pytorch.org/whl/{tag}"
CLIP_GIT = "clip @ git+https://github.com/ultralytics/CLIP.git"
CLIP_ZIP = "clip @ https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip"


def venv_python() -> Path:
    if os.name == "nt":
        return VENV / "Scripts" / "python.exe"
    return VENV / "bin" / "python"


def step(message: str) -> None:
    print(f"\n==> {message}", flush=True)


def run(*args: str | Path, check: bool = True) -> int:
    print("    $ " + " ".join(str(a) for a in args), flush=True)
    code = subprocess.call([str(a) for a in args], cwd=ROOT)
    if check and code != 0:
        sys.exit(f"Command failed with exit code {code}.")
    return code


def pip(*args: str, check: bool = True) -> int:
    return run(venv_python(), "-m", "pip", "install", "--disable-pip-version-check", *args, check=check)


def installed_torch_variant() -> str | None:
    """Return cpu, cuda or None when torch is not installed in the venv."""
    code = "import torch; print('cuda' if torch.version.cuda else 'cpu')"
    try:
        out = subprocess.run(
            [str(venv_python()), "-c", code], capture_output=True, text=True, check=True, cwd=ROOT
        )
    except subprocess.CalledProcessError:
        return None
    return out.stdout.strip() or None


def install_torch(tag: str, reinstall: bool) -> None:
    want = "cpu" if tag == "cpu" else "cuda"
    have = installed_torch_variant()
    if have == want and not reinstall:
        print(f"    torch with {want} support already installed, skipping")
        return
    extra = ["--force-reinstall"] if (have and have != want) or reinstall else []
    index = TORCH_INDEX.format(tag=tag)
    if pip("--upgrade", *extra, "torch", "torchvision", "--index-url", index, check=False) == 0:
        return
    if have:
        print(f"    Could not reach {index}, keeping the torch already installed.")
        return
    print(f"    Could not reach {index}, falling back to the default PyPI torch build.")
    pip("torch", "torchvision")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cpu", action="store_true", help="force CPU torch even if a GPU is found")
    parser.add_argument("--cuda", default=DEFAULT_CUDA, help="CUDA wheel tag, also set by UTRA_TORCH_CUDA")
    parser.add_argument("--reinstall-torch", action="store_true", help="reinstall torch even if present")
    parser.add_argument("--skip-prepare", action="store_true", help="do not build the model file")
    args = parser.parse_args()

    if sys.version_info < (3, 10):
        sys.exit("Python 3.10 or newer is required.")

    step(f"Virtual environment at {VENV}")
    if venv_python().exists():
        print("    already exists")
    else:
        venv.EnvBuilder(with_pip=True).create(VENV)
    pip("--upgrade", "pip")

    gpu = shutil.which("nvidia-smi") is not None and not args.cpu
    tag = args.cuda if gpu else "cpu"
    step(f"PyTorch, {'NVIDIA GPU found, using ' + tag if gpu else 'CPU build'}")
    install_torch(tag, args.reinstall_torch)

    step("utra-tracker package")
    if shutil.which("git"):
        pip("-e", ".[prepare]")
    else:
        pip("-e", ".", CLIP_ZIP)

    utra = (venv_python(), "-m", "utra_tracker")
    if not args.skip_prepare:
        step("Prepare model")
        run(*utra, "prepare")

    step("Environment check")
    code = run(*utra, "check", check=False)
    if code == 0:
        print("\nInstall complete. Start tracking with: python start.py")
    sys.exit(code)


if __name__ == "__main__":
    main()
