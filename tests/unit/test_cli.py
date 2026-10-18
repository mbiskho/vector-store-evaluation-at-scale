import pytest
from click.testing import CliRunner

from vsebench.cli import main


class TestCLI:
    @pytest.fixture
    def runner(self):
        return CliRunner()

    def test_help(self, runner):
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "Vector Store Evaluation" in result.output

    def test_generate_help(self, runner):
        result = runner.invoke(main, ["generate", "--help"])
        assert result.exit_code == 0
        assert "--target" in result.output

    def test_run_help(self, runner):
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "--stores" in result.output

    def test_report_help(self, runner):
        result = runner.invoke(main, ["report", "--help"])
        assert result.exit_code == 0
        assert "--results" in result.output

    def test_ground_truth_help(self, runner):
        result = runner.invoke(main, ["ground-truth", "--help"])
        assert result.exit_code == 0

    def test_generate_missing_corpus(self, runner):
        result = runner.invoke(main, ["run", "--corpus", "/nonexistent"])
        assert result.exit_code != 0
