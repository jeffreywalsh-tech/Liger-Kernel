"""Exercise the diagnostic script without installing accelerator dependencies."""

import runpy
import sys

from pathlib import Path
from types import SimpleNamespace

import pytest

REPORT_PATH = Path(__file__).resolve().parents[1] / "src" / "liger_kernel" / "env_report.py"


def torch_module(**attributes):
    return SimpleNamespace(
        __version__="test-version",
        cuda=SimpleNamespace(is_available=lambda: False),
        version=SimpleNamespace(cuda=None, hip=None),
        **attributes,
    )


@pytest.mark.parametrize(
    ("torch", "expected_xpu"),
    [
        (None, "Unable to query"),
        (torch_module(), "Unable to query"),
        (torch_module(xpu=SimpleNamespace(is_available=lambda: True)), "Unable to query"),
        (torch_module(xpu=SimpleNamespace(is_available=lambda: False)), "XPU Not Available"),
    ],
    ids=["torch-missing", "xpu-api-missing", "xpu-version-missing", "xpu-unavailable"],
)
def test_report_handles_missing_xpu_information(monkeypatch, capsys, torch, expected_xpu):
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "triton", None)
    monkeypatch.setitem(sys.modules, "transformers", None)

    runpy.run_path(str(REPORT_PATH), run_name="__main__")

    output = capsys.readouterr().out
    assert "Environment Report:" in output
    assert "Triton: Not installed" in output
    assert "Transformers: Not installed" in output
    assert f"XPU version: {expected_xpu}" in output
    if torch is None:
        assert "PyTorch: Not installed" in output


def test_report_preserves_available_xpu_version(monkeypatch, capsys):
    torch = torch_module(xpu=SimpleNamespace(is_available=lambda: True))
    torch.version.xpu = "test-xpu-version"
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "triton", None)
    monkeypatch.setitem(sys.modules, "transformers", None)

    runpy.run_path(str(REPORT_PATH), run_name="__main__")

    output = capsys.readouterr().out
    assert "PyTorch version: test-version" in output
    assert "XPU version: test-xpu-version" in output
