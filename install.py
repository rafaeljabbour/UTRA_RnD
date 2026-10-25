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
from typing import NamedTuple

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
TORCH_INDEX = "https://download.pytorch.org/whl/{tag}"
CLIP_GIT = "clip @ git+https://github.com/ultralytics/CLIP.git"
CLIP_ZIP = "clip @ https://github.com/ultralytics/CLIP/archive/refs/heads/main.zip"

# CUDA 12.8 wheels dropped Maxwell, Pascal and Volta. CUDA 12.6 wheels still cover
# compute capability 5.0 to 7.0 but are not published after torch 2.14, so they get a pin.
CUDA_DEFAULT = "cu128"
CUDA_LEGACY = "cu126"
LEGACY_TORCH = "torch<2.15"
MIN_CAP = (5, 0)
DEFAULT_MIN_CAP = (7, 5)
GPU_QUERY = ["nvidia-smi", "--query-gpu=name,compute_cap,driver_version", "--format=csv,noheader"]
CUDA_TEST = "import torch; x = torch.zeros(1, device='cuda') + 1; torch.cuda.synchronize(); print(x.item())"


class Gpu(NamedTuple):
    name: str
    cap: tuple[int, int]
    driver: str


def parse_gpu(text: str) -> Gpu | None:
    """Parse the first line of the nvidia-smi query, or return None when it looks wrong."""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return None
    parts = [p.strip() for p in lines[0].rsplit(",", 2)]
    if len(parts) != 3 or not parts[0]:
        return None
    name, cap, driver = parts
    major, _, minor = cap.partition(".")
    if not (major.isdigit() and minor.isdigit()):
        return None
    return Gpu(name, (int(major), int(minor)), driver)


def choose_tag(gpu: Gpu | None) -> str:
    """Pick the torch wheel tag for a GPU. Unknown GPUs get the default and rely on the verify step."""
    if gpu is None:
        return CUDA_DEFAULT
    if gpu.cap < MIN_CAP:
        return "cpu"
    if gpu.cap < DEFAULT_MIN_CAP:
        return CUDA_LEGACY
    return CUDA_DEFAULT


def torch_requirements(tag: str) -> list[str]:
    """Return the pip requirements for a tag. Pip resolves torchvision to match the pinned torch."""
    return [LEGACY_TORCH if tag == CUDA_LEGACY else "torch", "torchvision"]


def query_gpu() -> str:
    try:
        out = subprocess.run(GPU_QUERY, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout if out.returncode == 0 else ""


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


def venv_run(code: str) -> subprocess.CompletedProcess[str] | None:
    try:
        return subprocess.run([str(venv_python()), "-c", code], capture_output=True, text=True, cwd=ROOT)
    except OSError:
        return None


def installed_torch_tag() -> str | None:
    """Return cpu, a tag such as cu126, or None when torch is not installed in the venv."""
    code = "import torch; v = torch.version.cuda; print('cu' + v.replace('.', '') if v else 'cpu')"
    out = venv_run(code)
    if out is None or out.returncode != 0:
        return None
    return out.stdout.strip() or None


def cuda_works() -> bool:
    """Run one tiny op on the GPU with the venv torch."""
    out = venv_run(CUDA_TEST)
    if out is not None and out.returncode == 0:
        return True
    lines = (out.stderr if out else "").strip().splitlines()
    print(f"    CUDA test failed: {lines[-1] if lines else 'no output'}")
    return False


def install_torch(tag: str, reinstall: bool) -> None:
    have = installed_torch_tag()
    if not reinstall and have == tag:
        print(f"    torch for {tag} already installed, skipping")
        return
    if not reinstall and tag != "cpu" and have not in (None, "cpu") and cuda_works():
        print(f"    torch for {have} already installed and CUDA works, skipping")
        return
    extra = ["--force-reinstall"] if have or reinstall else []
    reqs = torch_requirements(tag)
    index = TORCH_INDEX.format(tag=tag)
    if pip("--upgrade", *extra, *reqs, "--index-url", index, check=False) == 0:
        return
    if have:
        print(f"    Could not reach {index}, keeping the torch already installed.")
        return
    print(f"    Could not reach {index}, falling back to the default PyPI torch build.")
    pip(*reqs)


def setup_torch(tag: str, reinstall: bool) -> bool:
    """Install torch, then check CUDA and fall back to the legacy wheels once if it fails."""
    install_torch(tag, reinstall)
    if tag == "cpu":
        return True
    if not cuda_works():
        if installed_torch_tag() != CUDA_LEGACY:
            print(f"    Retrying with {CUDA_LEGACY} wheels and {LEGACY_TORCH} for older GPUs")
            install_torch(CUDA_LEGACY, reinstall=True)
        if not cuda_works():
            print("    Warning: CUDA does not work with this torch build, tracking will run on CPU.")
            return False
    print("    CUDA works")
    return True


def select_tag(force_cpu: bool, explicit: str | None) -> str:
    """Choose the wheel tag from the flags and the GPU, printing what was found."""
    if force_cpu or shutil.which("nvidia-smi") is None:
        print("    No NVIDIA GPU in use, installing the CPU build")
        return "cpu"
    gpu = parse_gpu(query_gpu())
    if gpu:
        print(f"    GPU {gpu.name}, compute capability {gpu.cap[0]}.{gpu.cap[1]}, driver {gpu.driver}")
    else:
        print("    Could not read GPU details from nvidia-smi")
    if explicit:
        print(f"    Using {explicit} as requested")
        return explicit
    tag = choose_tag(gpu)
    if tag == "cpu":
        print("    This GPU is older than Maxwell and current PyTorch has no CUDA wheels for it.")
        print("    Installing the CPU build, tracking will run on CPU.")
    else:
        print(f"    Using {tag}{' with ' + LEGACY_TORCH if tag == CUDA_LEGACY else ''}")
    return tag


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cpu", action="store_true", help="force CPU torch even if a GPU is found")
    parser.add_argument("--cuda", help="CUDA wheel tag such as cu126, also set by UTRA_TORCH_CUDA")
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

    step("PyTorch")
    tag = select_tag(args.cpu, args.cuda or os.environ.get("UTRA_TORCH_CUDA") or None)
    setup_torch(tag, args.reinstall_torch)

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
