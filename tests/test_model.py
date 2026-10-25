import sys
from types import SimpleNamespace

import pytest

from utra_tracker.model import resolve_device


class FakeTensor:
    def __add__(self, other):
        return self

    def cpu(self):
        return self


def fake_torch(available: bool, works: bool) -> SimpleNamespace:
    """Stand in for torch so the device logic runs without torch or a GPU."""

    def zeros(*args, **kwargs):
        if not works:
            raise RuntimeError("CUDA error: no kernel image is available for execution on the device")
        return FakeTensor()

    cuda = SimpleNamespace(
        is_available=lambda: available, synchronize=lambda: None, get_device_name=lambda i: "Quadro M4000"
    )
    return SimpleNamespace(cuda=cuda, zeros=zeros)


@pytest.mark.parametrize(
    ("available", "works", "expected"),
    [(True, True, "cuda:0"), (True, False, "cpu"), (False, False, "cpu")],
)
def test_auto_device(monkeypatch, capsys, available, works, expected):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(available, works))
    assert resolve_device("auto") == expected
    out = capsys.readouterr().out
    if available and not works:
        assert "Quadro M4000" in out and "install.py --reinstall-torch" in out
    else:
        assert out == ""


def test_explicit_cuda_is_kept(monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", fake_torch(True, False))
    assert resolve_device("cuda:0") == "cuda:0"
