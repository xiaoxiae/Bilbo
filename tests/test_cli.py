from click.testing import CliRunner

from bilbo.cli import cli


def test_cli_help():
    runner = CliRunner()
    result = runner.invoke(cli, ["--help"])
    assert result.exit_code == 0
    assert "Bilingual audiobook interleaver" in result.output


def test_cli_list_empty():
    runner = CliRunner()
    result = runner.invoke(cli, ["list"])
    assert result.exit_code == 0


def test_cli_info_not_found():
    runner = CliRunner()
    result = runner.invoke(cli, ["info", "nonexistent"])
    assert result.exit_code != 0


def test_cli_version():
    runner = CliRunner()
    result = runner.invoke(cli, ["--version"])
    assert result.exit_code == 0
    assert "0.1.0" in result.output
