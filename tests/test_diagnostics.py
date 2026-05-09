from __future__ import annotations

import os

import pytest

from setiastro.saspro import diagnostics


def test_collect_diagnostics_renders_expected_sections(monkeypatch):
    monkeypatch.setattr(diagnostics, "_collect_app_info", lambda: {
        "version": "1.2.3",
        "mode": "dev",
        "executable": "/tmp/saspro",
        "cwd": "/work",
    })
    monkeypatch.setattr(diagnostics, "_collect_system_info", lambda: {
        "system": "Windows",
        "release": "11",
        "version": "build",
        "machine": "AMD64",
        "architecture": "64bit",
        "platform": "Windows-11",
    })
    monkeypatch.setattr(diagnostics, "_collect_python_info", lambda: {
        "executable": "/tmp/python",
        "version": "3.14.0",
        "full_version": "3.14.0",
        "supported_runtime_versions": ["3.12", "3.13", "3.14"],
        "runtime_python_supported": True,
        "supported_system_pythons": {"3.14": "py -3.14"},
    })
    monkeypatch.setattr(diagnostics, "_collect_environment_info", lambda: {
        "SASPRO_RUNTIME_DIR": None,
        "PYTHONPATH_set": False,
        "QT_QPA_PLATFORM": None,
        "PATH_summary": {
            "entry_count": 3,
            "runtime_python_dir_in_path": True,
            "sample_entries": ["System32", "Python", "Scripts"],
        },
    })
    monkeypatch.setattr(diagnostics, "_collect_runtime_info", lambda: {
        "base_dir": "/runtime",
        "active_dir": "/runtime/py314",
        "active_tag": "py314",
        "expected_tag": "py314",
        "venv_exists": False,
        "runtime_python": "/runtime/py314/venv/Scripts/python.exe",
        "runtime_python_exists": False,
        "runtime_python_version": None,
        "available_runtimes": [],
    })
    monkeypatch.setattr(diagnostics, "_collect_backend_info", lambda: {
        "detected_hardware": {
            "nvidia": False,
            "intel_arc": False,
            "amd_rocm": False,
            "amd_rocm_arch": None,
        },
        "torch": {
            "import_ok": False,
            "version": None,
            "file": None,
            "backend": "Not installed",
            "cuda_available": False,
            "cuda_version": None,
            "cuda_device": None,
            "xpu_available": False,
            "xpu_device": None,
            "mps_available": False,
            "rocm_version": None,
            "directml_available": False,
            "import_error": "ImportError: no torch",
        },
    })
    monkeypatch.setattr(diagnostics, "_collect_dependencies_info", lambda: {
        "numpy": {"present": True, "version": "2.0.0"},
        "torch": {"present": False, "version": None, "error": "missing"},
    })
    monkeypatch.setattr(diagnostics, "_collect_log_info", lambda: {
        "primary_log_dir": "/logs",
        "primary_log_dir_exists": True,
        "app_log": {"path": "/logs/setiastrosuite.log", "exists": True},
        "fallback_app_log": {"path": "/home/user/.setiastrosuite/logs/setiastrosuite.log", "exists": False},
        "crash_log": {"path": "/logs/saspro_crash.log", "exists": False},
        "report_dir": "/reports",
    })

    report = diagnostics.collect_diagnostics()

    assert "# SASpro Diagnostics Report" in report.markdown
    assert "## Runtime" in report.markdown
    assert "PyTorch import failed: ImportError: no torch" in report.markdown
    assert report.hints


def test_build_hints_detects_runtime_mismatch_and_cpu_fallback():
    data = {
        "python": {
            "version": "3.11.9",
            "runtime_python_supported": False,
            "supported_system_pythons": {"3.14": "py -3.14"},
        },
        "runtime": {
            "expected_tag": "py314",
            "active_tag": "py312",
            "runtime_python_version": (3, 12),
            "venv_exists": True,
            "runtime_python_exists": True,
        },
        "backend": {
            "detected_hardware": {"nvidia": True, "intel_arc": False, "amd_rocm": False},
            "torch": {
                "import_ok": True,
                "import_error": None,
                "backend": "CPU",
            },
        },
        "environment": {
            "PATH_summary": {"runtime_python_dir_in_path": False},
        },
    }

    hints = diagnostics._build_hints(data)

    assert any("outside the supported runtime range" in hint for hint in hints)
    assert any("Runtime folder mismatch" in hint for hint in hints)
    assert any("CPU-only" in hint for hint in hints)


@pytest.fixture
def qapp():
    pytest.importorskip("PyQt6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    return app


def test_diagnostics_dialog_smoke(monkeypatch, qapp):
    from setiastro.saspro.gui import diagnostics_dialog
    report = diagnostics.DiagnosticsReport(
        generated_at_utc="2026-05-08T00:00:00+00:00",
        data={},
        hints=[],
        markdown="# SASpro Diagnostics Report\n\n- ok\n",
    )
    monkeypatch.setattr(diagnostics_dialog, "collect_diagnostics", lambda: report)

    dlg = diagnostics_dialog.DiagnosticsReportDialog()

    assert "# SASpro Diagnostics Report" in dlg.editor.toPlainText()
