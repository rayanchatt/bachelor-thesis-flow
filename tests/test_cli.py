"""Smoke tests that every CLI subcommand parses and exposes --help."""

from __future__ import annotations

import pytest

from btflow.cli import build_parser, main

SUBCOMMANDS = [
    "rgb-stack",
    "defmap",
    "match-labels",
    "lagcorr",
    "confidence-kde",
    "heatmap",
    "iou-boxplot",
]


def test_parser_builds() -> None:
    parser = build_parser()
    assert parser.prog == "btflow"


@pytest.mark.parametrize("cmd", SUBCOMMANDS)
def test_subcommand_help_exits_zero(cmd: str, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main([cmd, "--help"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "usage:" in out.lower()


def test_version(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc_info:
        main(["--version"])
    assert exc_info.value.code == 0
    out = capsys.readouterr().out
    assert "btflow" in out
