"""Tests for data_engineering.cli."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

from typer.testing import CliRunner

from data_engineering.cli import app

runner = CliRunner()


class TestCli:
    def test_help(self) -> None:
        """Running with --help should succeed."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "SWE-Qwen data pipeline" in result.stdout

    def test_config(self) -> None:
        """Running config subcommand should show config."""
        result = runner.invoke(app, ["config"])
        assert result.exit_code == 0
        data = json.loads(result.stdout)
        assert "batch_size" in data
        assert data["max_issues_per_repo"] == 2000

    def test_validate_manifest_valid(self) -> None:
        """Valid manifest should pass validation."""
        manifest = {"version": "1", "repositories": [{"id": "r1", "owner": "o", "name": "n"}]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(manifest, f)
            tmp = Path(f.name)

        result = runner.invoke(app, ["validate-manifest", str(tmp)])
        tmp.unlink()
        assert result.exit_code == 0
        assert "valid" in result.stdout.lower()

    def test_validate_manifest_invalid(self) -> None:
        """Manifest missing required keys should fail."""
        manifest = {"version": "1", "repositories": [{"id": "r1"}]}
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(manifest, f)
            tmp = Path(f.name)

        result = runner.invoke(app, ["validate-manifest", str(tmp)])
        tmp.unlink()
        assert result.exit_code == 1
        assert "FAILED" in result.stderr

    def test_validate_manifest_bad_json(self) -> None:
        """Invalid JSON should fail."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("not json at all")
            tmp = Path(f.name)

        result = runner.invoke(app, ["validate-manifest", str(tmp)])
        tmp.unlink()
        assert result.exit_code == 1
        assert "Invalid JSON" in result.stderr
