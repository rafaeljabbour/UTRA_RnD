import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("install", ROOT / "install.py")
install = importlib.util.module_from_spec(spec)
spec.loader.exec_module(install)


@pytest.mark.parametrize(
    ("text", "name", "cap", "tag"),
    [
        ("Quadro M4000, 5.2, 580.159.03\n", "Quadro M4000", (5, 2), "cu126"),
        ("Quadro P2000, 6.1, 550.54", "Quadro P2000", (6, 1), "cu126"),
        ("Tesla V100-SXM2-16GB, 7.0, 570.86", "Tesla V100-SXM2-16GB", (7, 0), "cu126"),
        ("Tesla K80, 3.7, 470", "Tesla K80", (3, 7), "cpu"),
        ("NVIDIA GeForce RTX 3060, 8.6, 570.1", "NVIDIA GeForce RTX 3060", (8, 6), "cu128"),
        ("Tesla T4, 7.5, 550.0", "Tesla T4", (7, 5), "cu128"),
        ("NVIDIA B200, 10.0, 580.1", "NVIDIA B200", (10, 0), "cu128"),
    ],
)
def test_parse_and_choose(text, name, cap, tag):
    gpu = install.parse_gpu(text)
    assert gpu.name == name
    assert gpu.cap == cap
    assert install.choose_tag(gpu) == tag


def test_first_gpu_wins():
    gpu = install.parse_gpu("Quadro M4000, 5.2, 580\nNVIDIA GeForce RTX 3060, 8.6, 580\n")
    assert gpu.name == "Quadro M4000"


@pytest.mark.parametrize(
    "text",
    ["", "\n\n", "garbage", 'Field "compute_cap" is not a valid field to query.', "GPU, [N/A], 470"],
)
def test_bad_output_falls_back_to_default(text):
    assert install.parse_gpu(text) is None
    assert install.choose_tag(None) == install.CUDA_DEFAULT


def test_legacy_tag_pins_torch():
    assert install.torch_requirements("cu126") == ["torch<2.15", "torchvision"]
    assert install.torch_requirements("cu128") == ["torch", "torchvision"]
    assert install.torch_requirements("cpu") == ["torch", "torchvision"]


class FakeVenv:
    """Records torch installs and answers the CUDA test from a set of working tags."""

    def __init__(self, have, working):
        self.have = have
        self.working = working
        self.installs = []

    def install(self, tag, reinstall):
        self.installs.append(tag)
        self.have = tag

    def patch(self, monkeypatch):
        monkeypatch.setattr(install, "install_torch", self.install)
        monkeypatch.setattr(install, "installed_torch_tag", lambda: self.have)
        monkeypatch.setattr(install, "cuda_works", lambda: self.have in self.working)


@pytest.mark.parametrize(
    ("have", "tag", "working", "installs", "ok"),
    [
        ("cu128", "cu128", {"cu128"}, ["cu128"], True),
        ("cu128", "cu128", {"cu126"}, ["cu128", "cu126"], True),
        (None, "cu126", {"cu126"}, ["cu126"], True),
        (None, "cu126", set(), ["cu126"], False),
        (None, "cu128", set(), ["cu128", "cu126"], False),
        (None, "cpu", set(), ["cpu"], True),
    ],
)
def test_setup_torch_falls_back_once(monkeypatch, have, tag, working, installs, ok):
    fake = FakeVenv(have, working)
    fake.patch(monkeypatch)
    assert install.setup_torch(tag, reinstall=False) is ok
    assert fake.installs == installs
