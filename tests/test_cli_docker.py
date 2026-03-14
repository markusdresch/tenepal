"""Tests for docker CLI commands."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from tenepal import cli


def test_setup_docker_no_docker(monkeypatch, capsys):
    monkeypatch.setattr("tenepal.docker.is_docker_available", lambda: False)
    with pytest.raises(SystemExit):
        cli.setup_docker()
    captured = capsys.readouterr()
    assert "Docker is not available" in captured.err


def test_setup_docker_builds_image(monkeypatch, capsys):
    monkeypatch.setattr("tenepal.docker.is_docker_available", lambda: True)
    monkeypatch.setattr("tenepal.docker.TENEPAL_IMAGE", "tenepal-gpu:latest")

    calls = {}

    def fake_run(cmd, capture_output=False, text=True):
        calls["cmd"] = cmd
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    cli.setup_docker()
    assert "docker" in calls["cmd"][0]
    assert "build" in calls["cmd"]
    assert "-t" in calls["cmd"]
    captured = capsys.readouterr()
    assert "Docker image built successfully" in captured.out


def test_setup_docker_build_failure(monkeypatch, capsys):
    monkeypatch.setattr("tenepal.docker.is_docker_available", lambda: True)

    def fake_run(cmd, capture_output=False, text=True):
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    with pytest.raises(SystemExit):
        cli.setup_docker()
    captured = capsys.readouterr()
    assert "Docker build failed" in captured.err


def test_doctor_docker_unavailable(monkeypatch, capsys):
    monkeypatch.setattr("tenepal.docker.is_docker_available", lambda: False)
    monkeypatch.setattr("tenepal.docker.is_vulkan_available", lambda: False)
    monkeypatch.setattr("tenepal.docker.is_image_built", lambda *_: False)

    cli.doctor()
    captured = capsys.readouterr()
    assert "Docker" in captured.out
    assert "FAIL" in captured.out


def test_doctor_runs_healthcheck(monkeypatch, capsys):
    monkeypatch.setattr("tenepal.docker.is_docker_available", lambda: True)
    monkeypatch.setattr("tenepal.docker.is_vulkan_available", lambda: True)
    monkeypatch.setattr("tenepal.docker.is_image_built", lambda *_: True)
    monkeypatch.setattr("tenepal.docker.get_device_mounts", lambda: ["--device=/dev/dri"])
    monkeypatch.setattr("tenepal.docker.TENEPAL_IMAGE", "tenepal-gpu:latest")

    payload = '{"checks": {"python": {"ok": true}}}'

    def fake_run(cmd, capture_output=True, text=True, timeout=30):
        return SimpleNamespace(stdout=payload, stderr="")

    monkeypatch.setattr(cli.subprocess, "run", fake_run)

    cli.doctor()
    captured = capsys.readouterr()
    assert "Container diagnostics" in captured.out
    assert "python" in captured.out


def test_cli_routing_setup_docker(monkeypatch):
    called = {}

    def fake_setup():
        called["setup"] = True

    monkeypatch.setattr(cli, "setup_docker", fake_setup)
    monkeypatch.setattr(cli, "doctor", lambda: None)

    monkeypatch.setattr("sys.argv", ["tenepal", "setup-docker"])
    with pytest.raises(SystemExit):
        cli.main()
    assert called.get("setup") is True


def test_cli_routing_doctor(monkeypatch):
    called = {}

    def fake_doctor():
        called["doctor"] = True

    monkeypatch.setattr(cli, "setup_docker", lambda: None)
    monkeypatch.setattr(cli, "doctor", fake_doctor)

    monkeypatch.setattr("sys.argv", ["tenepal", "doctor"])
    with pytest.raises(SystemExit):
        cli.main()
    assert called.get("doctor") is True


def test_commands_set_includes_docker(monkeypatch):
    parser = cli.build_parser()
    args = parser.parse_args(["setup-docker"])
    assert args.command == "setup-docker"
    args = parser.parse_args(["doctor"])
    assert args.command == "doctor"
