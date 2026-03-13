from unittest.mock import patch

from click.testing import CliRunner

from bilbo.cli import cli
from bilbo.library import Library
from bilbo.models import BookMeta


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
    assert "version" in result.output


def _make_lib(tmp_path):
    lib = Library(root=tmp_path / "bilbo_lib")
    lib.init()
    meta = BookMeta(
        slug="test-book",
        title="Test Book",
        l1_lang="en", l2_lang="de",
        l1_audio=str(tmp_path / "en.mp3"),
        l2_audio=str(tmp_path / "de.mp3"),
    )
    lib.add_or_update(meta)
    return lib


def test_cli_rename(tmp_path):
    lib = _make_lib(tmp_path)
    runner = CliRunner()
    with patch("bilbo.cli.Library", return_value=lib):
        result = runner.invoke(cli, ["rename", "Test Book", "New Title"])
    assert result.exit_code == 0
    assert "New Title" in result.output
    assert lib.find_by_title("New Title") is not None
    assert lib.find_by_title("Test Book") is None


def test_cli_rename_not_found(tmp_path):
    lib = _make_lib(tmp_path)
    runner = CliRunner()
    with patch("bilbo.cli.Library", return_value=lib):
        result = runner.invoke(cli, ["rename", "Nope", "Whatever"])
    assert result.exit_code != 0


def test_cli_info(tmp_path):
    lib = _make_lib(tmp_path)
    runner = CliRunner()
    with patch("bilbo.cli.Library", return_value=lib):
        result = runner.invoke(cli, ["info", "Test Book"])
    assert result.exit_code == 0
    assert "Test Book" in result.output
    assert "en" in result.output


def test_cli_delete(tmp_path):
    lib = _make_lib(tmp_path)
    runner = CliRunner()
    with patch("bilbo.cli.Library", return_value=lib):
        result = runner.invoke(cli, ["delete", "Test Book", "--yes"])
    assert result.exit_code == 0
    assert lib.find_by_title("Test Book") is None
