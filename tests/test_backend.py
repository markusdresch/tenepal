"""Tests for backend abstraction layer."""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from tenepal.phoneme.backend import (
    ASRBackend,
    AllosaurusBackend,
    PhonemeSegment,
    get_backend,
    list_backends,
    register_backend,
)
from tenepal.phoneme.language_codes import resolve_language_code
from tenepal.phoneme import recognizer as recognizer_module
from tenepal.phoneme import backend as backend_module


class TestPhonemeSegment:
    def test_create_segment(self):
        seg = PhonemeSegment(phoneme="ae", start_time=0.5, duration=0.1)
        assert seg.phoneme == "ae"
        assert seg.start_time == 0.5
        assert seg.duration == 0.1


class TestAllosaurusBackend:
    @pytest.fixture
    def temp_wav_file(self):
        sample_rate = 16000
        duration = 0.5
        t = np.linspace(0, duration, int(sample_rate * duration))
        samples = np.sin(2 * np.pi * 440 * t).astype(np.float32)

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_file:
            temp_path = Path(temp_file.name)

        sf.write(str(temp_path), samples, sample_rate)

        yield temp_path

        if temp_path.exists():
            temp_path.unlink()

    def test_is_available(self):
        assert AllosaurusBackend.is_available() is True

    def test_name(self):
        backend = AllosaurusBackend()
        assert backend.name == "allosaurus"

    def test_recognize_returns_segments(self, temp_wav_file):
        backend = AllosaurusBackend()
        segments = backend.recognize(temp_wav_file)
        assert isinstance(segments, list)
        for seg in segments:
            assert isinstance(seg, PhonemeSegment)


class TestBackendFactory:
    def test_get_default_backend(self):
        backend = get_backend()
        assert isinstance(backend, AllosaurusBackend)

    def test_get_allosaurus_by_name(self):
        backend = get_backend("allosaurus")
        assert isinstance(backend, AllosaurusBackend)

    def test_unknown_backend_raises(self):
        with pytest.raises(ValueError) as excinfo:
            get_backend("nonexistent")
        assert "Unknown backend" in str(excinfo.value)

    def test_list_backends(self):
        assert "allosaurus" in list_backends()

    def test_backend_caching(self):
        backend_a = get_backend("allosaurus")
        backend_b = get_backend("allosaurus")
        assert backend_a is backend_b


class TestLanguageCodes:
    def test_nah_resolves_to_ncj(self):
        assert resolve_language_code("nah", "allosaurus") == "ncj"

    def test_ipa_passthrough(self):
        assert resolve_language_code("ipa", "allosaurus") == "ipa"

    def test_unknown_passthrough(self):
        assert resolve_language_code("xyz", "allosaurus") == "xyz"


class TestRecognizePhonemesDelegation:
    def test_default_backend_allosaurus(self, monkeypatch):
        calls = []

        def fake_get_backend(name):
            calls.append(name)

            class FakeBackend:
                def recognize(self, audio_path, lang="ipa"):
                    return []

            return FakeBackend()

        monkeypatch.setattr(recognizer_module, "get_backend", fake_get_backend)
        recognizer_module.recognize_phonemes("/tmp/fake.wav")
        assert calls == ["allosaurus"]

    def test_explicit_backend_param(self, monkeypatch):
        calls = []

        def fake_get_backend(name):
            calls.append(name)

            class FakeBackend:
                def recognize(self, audio_path, lang="ipa"):
                    return []

            return FakeBackend()

        monkeypatch.setattr(recognizer_module, "get_backend", fake_get_backend)
        recognizer_module.recognize_phonemes("/tmp/fake.wav", backend="allosaurus")
        assert calls == ["allosaurus"]


class TestBackendFallback:
    def test_unavailable_backend_fallback(self):
        class UnavailableBackend(ASRBackend):
            name = "unavailable"

            def recognize(self, audio_path, lang="ipa"):
                return []

            @classmethod
            def is_available(cls):
                return False

        original_backends = backend_module._BACKENDS.copy()
        original_instances = backend_module._BACKEND_INSTANCES.copy()
        try:
            register_backend("unavailable", UnavailableBackend)
            with pytest.raises(RuntimeError) as excinfo:
                get_backend("unavailable")
            assert "not available" in str(excinfo.value)
        finally:
            backend_module._BACKENDS = original_backends
            backend_module._BACKEND_INSTANCES = original_instances
