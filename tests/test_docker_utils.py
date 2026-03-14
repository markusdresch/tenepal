"""Tests for Docker utility helpers."""

from pathlib import Path
from unittest.mock import patch

import pytest


def test_is_docker_available_true():
    from tenepal.docker import utils as docker_utils

    with patch("subprocess.run") as run:
        run.return_value.returncode = 0
        assert docker_utils.is_docker_available() is True


def test_is_docker_available_false():
    from tenepal.docker import utils as docker_utils

    with patch("subprocess.run") as run:
        run.return_value.returncode = 1
        assert docker_utils.is_docker_available() is False


def test_is_docker_available_exception():
    from tenepal.docker import utils as docker_utils

    with patch("subprocess.run", side_effect=OSError):
        assert docker_utils.is_docker_available() is False


def test_is_vulkan_available_true():
    from tenepal.docker import utils as docker_utils

    with patch("os.path.exists", return_value=True), patch("glob.glob", return_value=["/dev/dri/renderD128"]):
        assert docker_utils.is_vulkan_available() is True


def test_is_vulkan_available_false_missing_dir():
    from tenepal.docker import utils as docker_utils

    with patch("os.path.exists", return_value=False):
        assert docker_utils.is_vulkan_available() is False


def test_get_device_mounts():
    from tenepal.docker import utils as docker_utils

    with patch.object(docker_utils, "is_vulkan_available", return_value=True):
        assert docker_utils.get_device_mounts() == ["--device=/dev/dri"]

    with patch.object(docker_utils, "is_vulkan_available", return_value=False):
        assert docker_utils.get_device_mounts() == []


def test_get_model_volume(tmp_path):
    from tenepal.docker import utils as docker_utils

    with patch.object(docker_utils, "MODEL_DIR", tmp_path):
        path = docker_utils.get_model_volume()
        assert path == tmp_path
        assert path.exists()


def test_get_volume_mounts(tmp_path):
    from tenepal.docker import utils as docker_utils

    audio_path = tmp_path / "audio.wav"
    audio_path.write_text("x")

    with patch.object(docker_utils, "MODEL_DIR", tmp_path / "models"):
        mounts = docker_utils.get_volume_mounts(audio_path)

    assert mounts[0:2] == ["-v", f"{audio_path.parent}:/audio:ro"]
    assert mounts[2:4] == ["-v", f"{(tmp_path / 'models')}:/models:ro"]


def test_is_image_built_true():
    from tenepal.docker import utils as docker_utils

    with patch("subprocess.run") as run:
        run.return_value.returncode = 0
        assert docker_utils.is_image_built("tenepal:latest") is True


def test_is_image_built_false():
    from tenepal.docker import utils as docker_utils

    with patch("subprocess.run") as run:
        run.return_value.returncode = 1
        assert docker_utils.is_image_built("tenepal:latest") is False


def test_is_image_built_exception():
    from tenepal.docker import utils as docker_utils

    with patch("subprocess.run", side_effect=OSError):
        assert docker_utils.is_image_built("tenepal:latest") is False


def test_docker_error_exists():
    from tenepal.docker import utils as docker_utils

    assert issubclass(docker_utils.DockerError, Exception)
