"""Tests for DockerDiarizer and Docker-backed diarization."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from tenepal.speaker.diarizer import SpeakerSegment, diarize


def _mock_result(stdout: str = "", returncode: int = 0, stderr: str = ""):
    return SimpleNamespace(stdout=stdout, returncode=returncode, stderr=stderr)


def test_docker_diarizer_builds_correct_command(monkeypatch, tmp_path):
    from tenepal.speaker import docker_diarizer

    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"data")

    monkeypatch.setenv("HUGGINGFACE_TOKEN", "token")
    monkeypatch.setattr(docker_diarizer, "TENEPAL_IMAGE", "tenepal-gpu:latest")
    monkeypatch.setattr(docker_diarizer, "get_device_mounts", lambda: ["--device=/dev/dri"])
    monkeypatch.setattr(
        docker_diarizer,
        "get_volume_mounts",
        lambda path: ["-v", f"{path.parent}:/audio:ro", "-v", "/models:/models:ro"],
    )

    calls = {}

    def fake_run(cmd, capture_output=True, text=True, timeout=300):
        calls["cmd"] = cmd
        return _mock_result(stdout=json.dumps({"segments": []}))

    monkeypatch.setattr(docker_diarizer.subprocess, "run", fake_run)

    docker_diarizer.DockerDiarizer().diarize(audio_path)

    cmd = calls["cmd"]
    assert cmd[:3] == ["docker", "run", "--rm"]
    assert "--device=/dev/dri" in cmd
    assert "-v" in cmd
    assert f"{audio_path.parent}:/audio:ro" in cmd
    assert "/models:/models:ro" in cmd
    assert "-e" in cmd
    assert "HUGGINGFACE_TOKEN=token" in cmd
    assert "tenepal-gpu:latest" in cmd
    assert "/app/diarize_worker.py" in cmd
    assert "/audio/test.wav" in cmd


def test_docker_diarizer_parses_json_segments(monkeypatch, tmp_path):
    from tenepal.speaker import docker_diarizer

    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"data")

    payload = {
        "segments": [
            {"speaker": "Speaker A", "start_time": 0.0, "end_time": 5.2},
            {"speaker": "Speaker B", "start_time": 5.2, "end_time": 10.1},
        ]
    }

    monkeypatch.setattr(docker_diarizer, "get_device_mounts", lambda: [])
    monkeypatch.setattr(docker_diarizer, "get_volume_mounts", lambda path: [])
    monkeypatch.setattr(
        docker_diarizer.subprocess,
        "run",
        lambda *args, **kwargs: _mock_result(stdout=json.dumps(payload)),
    )

    segments = docker_diarizer.DockerDiarizer().diarize(audio_path)
    assert segments == [
        SpeakerSegment("Speaker A", 0.0, 5.2),
        SpeakerSegment("Speaker B", 5.2, 10.1),
    ]


def test_docker_diarizer_raises_on_error_json(monkeypatch, tmp_path):
    from tenepal.speaker import docker_diarizer

    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"data")

    monkeypatch.setattr(docker_diarizer, "get_device_mounts", lambda: [])
    monkeypatch.setattr(docker_diarizer, "get_volume_mounts", lambda path: [])
    monkeypatch.setattr(
        docker_diarizer.subprocess,
        "run",
        lambda *args, **kwargs: _mock_result(stdout=json.dumps({"error": "HUGGINGFACE_TOKEN not set"})),
    )

    with pytest.raises(RuntimeError, match="HUGGINGFACE_TOKEN not set"):
        docker_diarizer.DockerDiarizer().diarize(audio_path)


def test_docker_diarizer_raises_on_nonzero_exit(monkeypatch, tmp_path):
    from tenepal.speaker import docker_diarizer

    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"data")

    monkeypatch.setattr(docker_diarizer, "get_device_mounts", lambda: [])
    monkeypatch.setattr(docker_diarizer, "get_volume_mounts", lambda path: [])
    monkeypatch.setattr(
        docker_diarizer.subprocess,
        "run",
        lambda *args, **kwargs: _mock_result(stdout="", returncode=1, stderr="oops"),
    )

    with pytest.raises(RuntimeError):
        docker_diarizer.DockerDiarizer().diarize(audio_path)


def test_docker_diarizer_missing_worker_message(monkeypatch, tmp_path):
    from tenepal.speaker import docker_diarizer

    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"data")

    monkeypatch.setattr(docker_diarizer, "get_device_mounts", lambda: [])
    monkeypatch.setattr(docker_diarizer, "get_volume_mounts", lambda path: [])
    monkeypatch.setattr(
        docker_diarizer.subprocess,
        "run",
        lambda *args, **kwargs: _mock_result(
            stdout="",
            returncode=2,
            stderr="python: can't open file '/app/diarize_worker.py': [Errno 2] No such file or directory",
        ),
    )

    with pytest.raises(RuntimeError, match="Rebuild the image"):
        docker_diarizer.DockerDiarizer().diarize(audio_path)


def test_docker_diarizer_raises_on_invalid_json(monkeypatch, tmp_path):
    from tenepal.speaker import docker_diarizer

    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"data")

    monkeypatch.setattr(docker_diarizer, "get_device_mounts", lambda: [])
    monkeypatch.setattr(docker_diarizer, "get_volume_mounts", lambda path: [])
    monkeypatch.setattr(
        docker_diarizer.subprocess,
        "run",
        lambda *args, **kwargs: _mock_result(stdout="not json"),
    )

    with pytest.raises(RuntimeError):
        docker_diarizer.DockerDiarizer().diarize(audio_path)


def test_docker_diarizer_timeout(monkeypatch, tmp_path):
    from tenepal.speaker import docker_diarizer

    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"data")

    def fake_run(*_args, **_kwargs):
        raise docker_diarizer.subprocess.TimeoutExpired(cmd="docker", timeout=300)

    monkeypatch.setattr(docker_diarizer, "get_device_mounts", lambda: [])
    monkeypatch.setattr(docker_diarizer, "get_volume_mounts", lambda path: [])
    monkeypatch.setattr(docker_diarizer.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="timed out"):
        docker_diarizer.DockerDiarizer().diarize(audio_path)


def test_docker_diarizer_is_available_checks_docker_and_image(monkeypatch):
    from tenepal.speaker import docker_diarizer

    monkeypatch.setattr(docker_diarizer, "is_docker_available", lambda: False)
    monkeypatch.setattr(docker_diarizer, "is_image_built", lambda: True)
    assert docker_diarizer.DockerDiarizer.is_available() is False

    monkeypatch.setattr(docker_diarizer, "is_docker_available", lambda: True)
    monkeypatch.setattr(docker_diarizer, "is_image_built", lambda: False)
    assert docker_diarizer.DockerDiarizer.is_available() is False

    monkeypatch.setattr(docker_diarizer, "is_docker_available", lambda: True)
    monkeypatch.setattr(docker_diarizer, "is_image_built", lambda: True)
    assert docker_diarizer.DockerDiarizer.is_available() is True


def test_diarize_with_docker_flag_uses_docker_diarizer(monkeypatch, tmp_path):
    from tenepal.speaker import docker_diarizer

    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"data")

    monkeypatch.setattr(docker_diarizer.DockerDiarizer, "is_available", classmethod(lambda cls: True))

    called = {}

    def fake_diarize(self, path):  # noqa: ANN001 - test stub
        called["path"] = Path(path)
        return [SpeakerSegment("Speaker A", 0.0, 1.0)]

    monkeypatch.setattr(docker_diarizer.DockerDiarizer, "diarize", fake_diarize)

    result = diarize(audio_path, use_docker=True)
    assert result[0].speaker == "Speaker A"
    assert called["path"] == audio_path


def test_diarize_with_docker_flag_falls_back_when_unavailable(monkeypatch, tmp_path):
    from tenepal.speaker import docker_diarizer
    from tenepal.speaker import diarizer

    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"data")

    monkeypatch.setattr(docker_diarizer.DockerDiarizer, "is_available", classmethod(lambda cls: False))
    monkeypatch.setattr(diarizer, "_load_pipeline", lambda: None)
    monkeypatch.setattr(diarizer.sf, "info", lambda _path: SimpleNamespace(duration=3.2))

    result = diarize(audio_path, use_docker=True)
    assert result == [SpeakerSegment("Speaker ?", 0.0, 3.2)]


def test_diarize_without_docker_flag_uses_host(monkeypatch, tmp_path):
    from tenepal.speaker import diarizer

    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"data")

    monkeypatch.setattr(diarizer, "_load_pipeline", lambda: None)
    monkeypatch.setattr(diarizer.sf, "info", lambda _path: SimpleNamespace(duration=1.5))

    result = diarize(audio_path, use_docker=False)
    assert result == [SpeakerSegment("Speaker ?", 0.0, 1.5)]


def test_pipeline_passes_use_docker_to_diarize(monkeypatch, tmp_path):
    from tenepal import pipeline

    audio_path = tmp_path / "test.wav"
    audio_path.write_bytes(b"data")

    called = {}

    def fake_diarize(path, use_docker=False):  # noqa: ANN001 - test stub
        called["use_docker"] = use_docker
        return [SpeakerSegment("Speaker ?", 0.0, 1.0)]

    monkeypatch.setattr(pipeline, "diarize", fake_diarize)
    monkeypatch.setattr(pipeline, "_process_single_stream", lambda *args, **kwargs: ["ok"])  # noqa: ARG005

    result = pipeline.process_audio(audio_path, use_docker=True)
    assert result == ["ok"]
    assert called.get("use_docker") is True
