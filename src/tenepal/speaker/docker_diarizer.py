"""Docker-based speaker diarization using GPU container."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

from ..docker.utils import (
    TENEPAL_IMAGE,
    get_device_mounts,
    get_volume_mounts,
    is_docker_available,
    is_image_built,
)
from .diarizer import SpeakerSegment

logger = logging.getLogger(__name__)


class DockerDiarizer:
    """Speaker diarizer that runs pyannote inside Docker GPU container."""

    TIMEOUT = 300

    def diarize(self, audio_path: Path) -> list[SpeakerSegment]:
        audio_path = Path(audio_path).resolve()
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")

        cmd: list[str] = ["docker", "run", "--rm"]
        cmd.extend(get_device_mounts())
        cmd.extend(get_volume_mounts(audio_path))

        token = os.environ.get("HUGGINGFACE_TOKEN", "")
        cmd.extend(["-e", f"HUGGINGFACE_TOKEN={token}"])

        container_audio_path = f"/audio/{audio_path.name}"
        cmd.extend([
            TENEPAL_IMAGE,
            "python",
            "/app/diarize_worker.py",
            "--audio-path",
            container_audio_path,
        ])

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"Docker diarization timed out after {self.TIMEOUT}s"
            ) from exc

        stdout = result.stdout.strip()
        stderr = result.stderr.strip() if result.stderr else ""
        missing_worker = (
            "/app/diarize_worker.py" in stdout
            or "/app/diarize_worker.py" in stderr
        )
        missing_worker = missing_worker and (
            "no such file or directory" in stdout.lower()
            or "no such file or directory" in stderr.lower()
        )
        if missing_worker:
            raise RuntimeError(
                "Docker image missing /app/diarize_worker.py. "
                "Rebuild the image with: tenepal setup-docker"
            )
        if not stdout:
            stderr_msg = stderr if stderr else "no output"
            raise RuntimeError(
                f"Docker diarization produced no output (exit {result.returncode}): {stderr_msg[:500]}"
            )

        try:
            payload = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"Docker diarization returned invalid JSON: {stdout[:200]}"
            ) from exc

        if "error" in payload:
            raise RuntimeError(str(payload["error"]))

        segments: list[SpeakerSegment] = []
        for seg_data in payload.get("segments", []):
            segments.append(
                SpeakerSegment(
                    speaker=seg_data["speaker"],
                    start_time=seg_data["start_time"],
                    end_time=seg_data["end_time"],
                )
            )

        segments.sort(key=lambda s: (s.start_time, s.speaker))
        return segments

    @classmethod
    def is_available(cls) -> bool:
        return is_docker_available() and is_image_built()
