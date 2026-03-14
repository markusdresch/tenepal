"""Tests for setup-omnilingual command."""

from pathlib import Path
from unittest.mock import call

import pytest

from tenepal import cli
from tenepal.phoneme.omnilingual_backend import OmnilingualBackend


class TestOmnilingualSetup:
    def test_setup_omnilingual_no_python312(self, monkeypatch, capsys):
        monkeypatch.setattr(cli.shutil, "which", lambda _cmd: None)
        with pytest.raises(SystemExit):
            cli.setup_omnilingual()
        captured = capsys.readouterr()
        assert "Python 3.12 is required" in captured.err

    def test_setup_omnilingual_creates_venv(self, monkeypatch, capsys):
        venv_path = OmnilingualBackend.VENV_PATH
        monkeypatch.setattr(cli.shutil, "which", lambda _cmd: "/usr/bin/python3.12")
        monkeypatch.setattr(Path, "exists", lambda self: False)

        calls = []

        def fake_run(cmd, check):
            calls.append(cmd)

        monkeypatch.setattr(cli.subprocess, "run", fake_run)

        cli.setup_omnilingual()
        assert calls[0][:3] == ["/usr/bin/python3.12", "-m", "venv"]
        assert str(venv_path) in calls[0]

        captured = capsys.readouterr()
        assert "Creating Python 3.12 virtual environment" in captured.out

    def test_setup_omnilingual_installs_deps(self, monkeypatch):
        venv_path = OmnilingualBackend.VENV_PATH
        monkeypatch.setattr(cli.shutil, "which", lambda _cmd: "/usr/bin/python3.12")
        monkeypatch.setattr(Path, "exists", lambda self: True)

        calls = []

        def fake_run(cmd, check):
            calls.append(cmd)

        monkeypatch.setattr(cli.subprocess, "run", fake_run)

        cli.setup_omnilingual()
        assert calls == [[str(venv_path / "bin" / "pip"), "install", "omnilingual-asr"]]

    def test_setup_omnilingual_existing_venv(self, monkeypatch, capsys):
        monkeypatch.setattr(cli.shutil, "which", lambda _cmd: "/usr/bin/python3.12")
        monkeypatch.setattr(Path, "exists", lambda self: True)
        monkeypatch.setattr(cli.subprocess, "run", lambda cmd, check: None)

        cli.setup_omnilingual()
        captured = capsys.readouterr()
        assert "Omnilingual environment already exists" in captured.out

    def test_setup_omnilingual_pip_failure(self, monkeypatch, capsys):
        monkeypatch.setattr(cli.shutil, "which", lambda _cmd: "/usr/bin/python3.12")
        monkeypatch.setattr(Path, "exists", lambda self: True)

        def fake_run(cmd, check):
            raise cli.subprocess.CalledProcessError(1, cmd)

        monkeypatch.setattr(cli.subprocess, "run", fake_run)

        with pytest.raises(SystemExit):
            cli.setup_omnilingual()
        captured = capsys.readouterr()
        assert "Error installing omnilingual-asr" in captured.err
