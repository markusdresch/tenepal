"""Tests for docker diarization worker."""

from __future__ import annotations

import json
import os
import subprocess
import sys


def _run_worker(audio_path: str, env: dict[str, str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "docker/diarize_worker.py", "--audio-path", audio_path],
        capture_output=True,
        text=True,
        env=env,
    )


def test_worker_missing_token():
    env = os.environ.copy()
    env.pop("HUGGINGFACE_TOKEN", None)
    result = _run_worker("/audio/missing.wav", env)
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload == {"error": "HUGGINGFACE_TOKEN not set"}


def test_worker_missing_audio_file():
    env = os.environ.copy()
    env["HUGGINGFACE_TOKEN"] = "fake"
    result = _run_worker("/audio/missing.wav", env)
    assert result.returncode == 1
    payload = json.loads(result.stdout)
    assert payload == {"error": "Audio file not found: /audio/missing.wav"}


def test_worker_json_contract():
    from docker import diarize_worker

    assert diarize_worker._letter_label(0) == "Speaker A"
    assert diarize_worker._letter_label(25) == "Speaker Z"
    assert diarize_worker._letter_label(26) == "Speaker AA"


def test_worker_argparse():
    result = subprocess.run(
        [sys.executable, "docker/diarize_worker.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert "--audio-path" in result.stdout
