"""Docker and Vulkan GPU detection utilities."""

from __future__ import annotations

import glob
import os
import subprocess
from pathlib import Path


class DockerError(Exception):
    """Raised when Docker operations fail."""


TENEPAL_IMAGE = "tenepal-gpu:latest"
MODEL_DIR = Path.home() / ".tenepal" / "models"


def is_docker_available() -> bool:
    """Check if Docker daemon is running and accessible."""
    try:
        result = subprocess.run(["docker", "info"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except Exception:
        return False


def is_vulkan_available() -> bool:
    """Check if Vulkan GPU devices are accessible via /dev/dri."""
    if not os.path.exists("/dev/dri"):
        return False
    return bool(glob.glob("/dev/dri/renderD*"))


def get_device_mounts() -> list[str]:
    """Return Docker --device flags for Vulkan GPU passthrough."""
    if is_vulkan_available():
        return ["--device=/dev/dri"]
    return []


def get_model_volume() -> Path:
    """Return path to model weights directory, creating if needed."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    return MODEL_DIR


def get_volume_mounts(audio_path: Path) -> list[str]:
    """Return Docker -v flags for audio and model directories."""
    audio_dir = audio_path.parent.resolve()
    model_dir = get_model_volume().resolve()
    return ["-v", f"{audio_dir}:/audio:ro", "-v", f"{model_dir}:/models:ro"]


def is_image_built(image_name: str = TENEPAL_IMAGE) -> bool:
    """Check if Docker image exists locally."""
    try:
        result = subprocess.run(
            ["docker", "image", "inspect", image_name],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False
