"""Tests for OmnilingualBackend subprocess bridge."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from tenepal.phoneme.backend import get_backend, list_backends
from tenepal.phoneme.omnilingual_backend import OmnilingualBackend


class TestOmnilingualBackend:
    def test_omnilingual_registered(self):
        assert "omnilingual" in list_backends()

    def test_is_available_false_no_venv(self, monkeypatch):
        def fake_exists(_self):
            return False

        monkeypatch.setattr("pathlib.Path.exists", fake_exists)
        assert OmnilingualBackend.is_available() is False

    def test_is_available_true_with_venv(self, monkeypatch):
        def fake_exists(_self):
            return True

        monkeypatch.setattr("pathlib.Path.exists", fake_exists)
        assert OmnilingualBackend.is_available() is True

    def test_recognize_subprocess_call(self, monkeypatch):
        backend = OmnilingualBackend()

        def fake_load_audio(_path):
            return SimpleNamespace(duration=1.0)

        captured = {}

        def fake_run(cmd, capture_output, text, timeout):
            captured["cmd"] = cmd
            return SimpleNamespace(stdout='{"text": "hello world"}')

        def fake_text_to_phonemes(text, language, start_time, duration):
            return [SimpleNamespace(phoneme="h", start_time=0.0, duration=duration)]

        monkeypatch.setattr("pathlib.Path.exists", lambda _self: True)
        monkeypatch.setattr(OmnilingualBackend, "_detect_device", lambda _self: "cpu")
        monkeypatch.setattr("tenepal.phoneme.omnilingual_backend.load_audio", fake_load_audio)
        monkeypatch.setattr("tenepal.phoneme.omnilingual_backend.subprocess.run", fake_run)
        monkeypatch.setattr(
            "tenepal.phoneme.omnilingual_backend.text_to_phonemes",
            fake_text_to_phonemes,
        )

        segments = backend.recognize("/tmp/audio.wav", lang="nah")
        assert segments
        assert captured["cmd"]

    def test_recognize_error_handling(self, monkeypatch):
        backend = OmnilingualBackend()

        def fake_load_audio(_path):
            return SimpleNamespace(duration=1.0)

        def fake_run(*_args, **_kwargs):
            return SimpleNamespace(stdout='{"error": "model not found"}')

        monkeypatch.setattr("pathlib.Path.exists", lambda _self: True)
        monkeypatch.setattr(OmnilingualBackend, "_detect_device", lambda _self: "cpu")
        monkeypatch.setattr("tenepal.phoneme.omnilingual_backend.load_audio", fake_load_audio)
        monkeypatch.setattr("tenepal.phoneme.omnilingual_backend.subprocess.run", fake_run)

        with pytest.raises(RuntimeError):
            backend.recognize("/tmp/audio.wav", lang="nah")

    def test_recognize_subprocess_timeout(self, monkeypatch):
        backend = OmnilingualBackend()

        def fake_load_audio(_path):
            return SimpleNamespace(duration=1.0)

        def fake_run(*_args, **_kwargs):
            raise __import__("subprocess").TimeoutExpired(cmd="x", timeout=1)

        monkeypatch.setattr("pathlib.Path.exists", lambda _self: True)
        monkeypatch.setattr(OmnilingualBackend, "_detect_device", lambda _self: "cpu")
        monkeypatch.setattr("tenepal.phoneme.omnilingual_backend.load_audio", fake_load_audio)
        monkeypatch.setattr("tenepal.phoneme.omnilingual_backend.subprocess.run", fake_run)

        with pytest.raises(RuntimeError):
            backend.recognize("/tmp/audio.wav", lang="nah")

    def test_recognize_language_mapping(self, monkeypatch):
        backend = OmnilingualBackend()

        def fake_load_audio(_path):
            return SimpleNamespace(duration=1.0)

        captured = {}

        def fake_run(cmd, *args, **kwargs):
            captured["cmd"] = cmd
            return SimpleNamespace(stdout='{"text": "hello"}')

        monkeypatch.setattr("pathlib.Path.exists", lambda _self: True)
        monkeypatch.setattr(OmnilingualBackend, "_detect_device", lambda _self: "cpu")
        monkeypatch.setattr("tenepal.phoneme.omnilingual_backend.load_audio", fake_load_audio)
        monkeypatch.setattr("tenepal.phoneme.omnilingual_backend.subprocess.run", fake_run)

        backend.recognize("/tmp/audio.wav", lang="nah")
        assert "ncj_Latn" in captured["cmd"]

    def test_model_card_selection(self, monkeypatch):
        backend = OmnilingualBackend(model_size="7B")

        def fake_load_audio(_path):
            return SimpleNamespace(duration=1.0)

        captured = {}

        def fake_run(cmd, *args, **kwargs):
            captured["cmd"] = cmd
            return SimpleNamespace(stdout='{"text": "hello"}')

        monkeypatch.setattr("pathlib.Path.exists", lambda _self: True)
        monkeypatch.setattr(OmnilingualBackend, "_detect_device", lambda _self: "cpu")
        monkeypatch.setattr("tenepal.phoneme.omnilingual_backend.load_audio", fake_load_audio)
        monkeypatch.setattr("tenepal.phoneme.omnilingual_backend.subprocess.run", fake_run)

        backend.recognize("/tmp/audio.wav", lang="nah")
        assert "omniASR_LLM_7B_v2" in captured["cmd"]

    def test_invalid_model_size(self):
        with pytest.raises(ValueError):
            OmnilingualBackend(model_size="XL")

    def test_detect_device_with_cuda(self, monkeypatch):
        class FakeCuda:
            @staticmethod
            def is_available():
                return True

        fake_torch = SimpleNamespace(cuda=FakeCuda())
        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

        backend = OmnilingualBackend()
        assert backend._detect_device() == "cuda"

    def test_detect_device_without_cuda(self, monkeypatch):
        class FakeCuda:
            @staticmethod
            def is_available():
                return False

        fake_torch = SimpleNamespace(cuda=FakeCuda())
        monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)

        backend = OmnilingualBackend()
        assert backend._detect_device() == "cpu"

    def test_text_to_ipa_integration(self, monkeypatch):
        backend = OmnilingualBackend()

        def fake_load_audio(_path):
            return SimpleNamespace(duration=1.0)

        def fake_run(*_args, **_kwargs):
            return SimpleNamespace(stdout='{"text": "Niltze"}')

        def fake_text_to_phonemes(text, language, start_time, duration):
            return [SimpleNamespace(phoneme="n", start_time=0.0, duration=duration)]

        monkeypatch.setattr("pathlib.Path.exists", lambda _self: True)
        monkeypatch.setattr(OmnilingualBackend, "_detect_device", lambda _self: "cpu")
        monkeypatch.setattr("tenepal.phoneme.omnilingual_backend.load_audio", fake_load_audio)
        monkeypatch.setattr("tenepal.phoneme.omnilingual_backend.subprocess.run", fake_run)
        monkeypatch.setattr(
            "tenepal.phoneme.omnilingual_backend.text_to_phonemes",
            fake_text_to_phonemes,
        )

        segments = backend.recognize("/tmp/audio.wav", lang="nah")
        assert segments
        assert segments[0].phoneme == "n"

    def test_get_backend_with_kwargs(self):
        with patch.object(OmnilingualBackend, "is_available", return_value=True):
            backend = get_backend("omnilingual", model_size="7B")
        assert isinstance(backend, OmnilingualBackend)
        assert backend.model_size == "7B"
