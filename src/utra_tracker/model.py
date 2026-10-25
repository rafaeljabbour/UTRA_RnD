"""Model preparation, loading and device selection."""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from utra_tracker.config import Config

CLIP_INSTALL_HINT = 'pip install "clip @ git+https://github.com/ultralytics/CLIP.git"'


class ModelError(RuntimeError):
    """Raised when the detector cannot be prepared or loaded."""


def resolve_device(device: str) -> str:
    """Map auto to cuda:0 when a GPU is available, else cpu."""
    if device != "auto":
        return device
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def prepare_model(cfg: Config, force: bool = False) -> Path:
    """Download the open vocabulary base model, bake the prompt classes and save it.

    The saved file holds the text embeddings, so runtime needs no text encoder and no network.
    """
    out = Path(cfg.model.weights).resolve()
    if out.is_file() and not force:
        print(f"Model already prepared: {out}")
        return out
    try:
        import clip  # noqa: F401
    except ImportError as exc:
        raise ModelError(f"The CLIP tokenizer is needed to bake prompts. Run: {CLIP_INSTALL_HINT}") from exc
    from ultralytics import YOLOE

    out.parent.mkdir(parents=True, exist_ok=True)
    base = Path(cfg.model.base)
    base_ref = str(base.resolve()) if base.is_file() else base.name
    classes = list(cfg.model.classes)
    print(f"Baking classes {classes} into {base_ref}")
    # Downloads land next to the output file instead of the current directory.
    with _working_dir(out.parent):
        model = YOLOE(base_ref)
        model.set_classes(classes)
        model.save(str(out))
    verify_model(out)
    print(f"Saved {out}")
    return out


def load_model(cfg: Config) -> Any:
    """Load the detector with plain YOLO so baked and fine tuned files both work."""
    path = Path(cfg.model.weights)
    if not path.is_file():
        raise ModelError(f"Model file not found: {path}. Run 'utra-track prepare' first.")
    from ultralytics import YOLO

    model = YOLO(str(path))
    names = set(model.names.values())
    if names != set(cfg.model.classes):
        print(
            f"Warning: model classes {sorted(names)} differ from config classes. "
            "Run 'utra-track prepare --force' if you changed the prompts."
        )
    return model


def verify_model(path: Path) -> list[str]:
    """Load a weights file and run one blank frame through it."""
    import numpy as np
    from ultralytics import YOLO

    model = YOLO(str(path))
    model.predict(np.zeros((320, 320, 3), dtype=np.uint8), verbose=False, device="cpu")
    return list(model.names.values())


@contextmanager
def _working_dir(path: Path) -> Iterator[None]:
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)
