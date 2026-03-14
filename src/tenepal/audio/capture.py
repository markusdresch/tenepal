"""Live system audio capture from PulseAudio/PipeWire."""

import shutil
import subprocess
from typing import Generator, Optional

import numpy as np

from .loader import AudioData
from .preprocessor import preprocess_audio


class AudioCapture:
    """Capture live system audio from PulseAudio or PipeWire monitor.

    Captures what's playing on the system speakers (e.g., a film in VLC)
    and yields preprocessed audio chunks suitable for phoneme recognition.

    Args:
        chunk_duration: Duration of each audio chunk in seconds (default: 3.0)
        sample_rate: Target sample rate in Hz (default: 16000 for Allosaurus)

    Raises:
        RuntimeError: If neither PulseAudio nor PipeWire is available

    Example:
        >>> with AudioCapture(chunk_duration=3.0) as capture:
        ...     for audio_chunk in capture.chunks():
        ...         # Process audio_chunk (AudioData object)
        ...         pass
    """

    def __init__(self, chunk_duration: float = 3.0, sample_rate: int = 16000):
        self.chunk_duration = chunk_duration
        self.sample_rate = sample_rate
        self._process: Optional[subprocess.Popen] = None
        self._audio_system = self._detect_audio_system()
        self._monitor_source = self._find_monitor_source()

        # Calculate bytes per chunk: sample_rate * duration * 2 (16-bit = 2 bytes)
        self.bytes_per_chunk = int(sample_rate * chunk_duration * 2)

    def _detect_audio_system(self) -> str:
        """Detect which audio system is available.

        Returns:
            "pulseaudio" or "pipewire"

        Raises:
            RuntimeError: If neither system is available
        """
        # Check for PulseAudio
        if shutil.which("pactl") and shutil.which("parec"):
            return "pulseaudio"

        # Check for PipeWire
        if shutil.which("pw-cli") and shutil.which("pw-record"):
            return "pipewire"

        raise RuntimeError(
            "Neither PulseAudio nor PipeWire is available. "
            "Install pulseaudio-utils or pipewire-pulse."
        )

    def _find_monitor_source(self) -> str:
        """Find the monitor source for system audio output.

        Returns:
            Name or ID of the monitor source

        Raises:
            RuntimeError: If no monitor source is found
        """
        if self._audio_system == "pulseaudio":
            return self._find_pulseaudio_monitor()
        else:
            return self._find_pipewire_monitor()

    def _find_pulseaudio_monitor(self) -> str:
        """Find PulseAudio monitor source."""
        try:
            result = subprocess.run(
                ["pactl", "list", "short", "sources"],
                capture_output=True,
                text=True,
                check=True
            )

            # Parse output: each line is "ID NAME ..."
            # Look for lines containing "monitor"
            for line in result.stdout.splitlines():
                if "monitor" in line.lower():
                    parts = line.split()
                    if len(parts) >= 2:
                        return parts[1]  # Source name

            raise RuntimeError("No PulseAudio monitor source found")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to list PulseAudio sources: {e}")

    def _find_pipewire_monitor(self) -> str:
        """Find PipeWire monitor node."""
        try:
            result = subprocess.run(
                ["pw-cli", "list-objects", "Node"],
                capture_output=True,
                text=True,
                check=True
            )

            # Parse pw-cli output - look for monitor nodes
            # Output format is multi-line per node
            lines = result.stdout.splitlines()
            for i, line in enumerate(lines):
                if "node.name" in line and "monitor" in line.lower():
                    # Extract node ID from previous lines
                    # Look backwards for "id X" pattern (up to 10 lines back)
                    for j in range(i-1, max(-1, i-10), -1):
                        if "id" in lines[j]:
                            parts = lines[j].split()
                            for k, part in enumerate(parts):
                                if part == "id" and k+1 < len(parts):
                                    # Remove trailing comma if present
                                    node_id = parts[k+1].rstrip(',')
                                    return node_id

            raise RuntimeError("No PipeWire monitor node found")

        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to list PipeWire nodes: {e}")

    def start(self) -> None:
        """Start audio capture subprocess."""
        if self._process is not None:
            raise RuntimeError("Capture already started")

        if self._audio_system == "pulseaudio":
            cmd = [
                "parec",
                "--format=s16le",
                f"--rate={self.sample_rate}",
                "--channels=1",
                f"--device={self._monitor_source}"
            ]
        else:  # pipewire
            cmd = [
                "pw-record",
                "--format=s16",
                f"--rate={self.sample_rate}",
                "--channels=1",
                f"--target={self._monitor_source}",
                "-"  # Output to stdout
            ]

        self._process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

    def stop(self) -> None:
        """Stop audio capture subprocess."""
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait()
            self._process = None

    def chunks(self) -> Generator[AudioData, None, None]:
        """Yield preprocessed audio chunks.

        Yields:
            AudioData objects with preprocessed audio (mono, 16kHz, normalized)
        """
        if self._process is None:
            raise RuntimeError("Capture not started - call start() first")

        while True:
            # Read raw PCM bytes
            raw_bytes = self._process.stdout.read(self.bytes_per_chunk)

            if len(raw_bytes) == 0:
                # End of stream
                break

            # Convert bytes to numpy array
            # Format: s16le (signed 16-bit little-endian)
            samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32)

            # Normalize from int16 range to float32 [-1.0, 1.0]
            samples = samples / 32768.0

            # Calculate actual duration (may be shorter for last chunk)
            duration = len(samples) / self.sample_rate

            # Create AudioData and preprocess
            audio = AudioData(
                samples=samples,
                sample_rate=self.sample_rate,
                duration=duration,
                source_format="capture"
            )

            # Preprocess: already mono and correct sample rate,
            # but this will normalize to peak 0.9
            preprocessed = preprocess_audio(audio, target_sr=self.sample_rate)

            yield preprocessed

    def __enter__(self):
        """Context manager entry: start capture."""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit: stop capture."""
        self.stop()
        return False
