from __future__ import annotations

from pathlib import Path

from setiastro.saspro import cli, diagnostics


def test_cli_report_prints_markdown(monkeypatch, capsys):
    report = diagnostics.DiagnosticsReport(
        generated_at_utc="2026-05-08T00:00:00+00:00",
        data={},
        hints=[],
        markdown="# SASpro Diagnostics Report\n\n- sample\n",
    )
    monkeypatch.setattr(cli, "collect_diagnostics", lambda: report)

    rc = cli.main(["report"])

    captured = capsys.readouterr()
    assert rc == 0
    assert "# SASpro Diagnostics Report" in captured.out
    assert captured.err == ""


def test_cli_report_writes_output_file(monkeypatch, tmp_path, capsys):
    report = diagnostics.DiagnosticsReport(
        generated_at_utc="2026-05-08T00:00:00+00:00",
        data={},
        hints=[],
        markdown="# SASpro Diagnostics Report\n\n- written\n",
    )
    monkeypatch.setattr(cli, "collect_diagnostics", lambda: report)

    output = tmp_path / "report.md"
    rc = cli.main(["report", "--output", str(output)])

    captured = capsys.readouterr()
    assert rc == 0
    assert output.read_text(encoding="utf-8") == report.markdown
    assert "# SASpro Diagnostics Report" in captured.out
    assert "wrote diagnostics report" in captured.err
