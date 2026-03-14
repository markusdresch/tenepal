"""Tests for DualBackend confidence merging and fallback."""

from types import SimpleNamespace

import pytest

from tenepal.phoneme.backend import PhonemeSegment, list_backends
from tenepal.phoneme.dual_backend import DualBackend
from tenepal.phoneme import dual_backend as dual_module


class FakeBackend:
    def __init__(self, segments=None, error=None):
        self.segments = segments or []
        self.error = error
        self.calls = 0

    def recognize(self, audio_path, lang="ipa"):
        self.calls += 1
        if self.error:
            raise self.error
        return self.segments


class TestPhonemeSegmentConfidence:
    def test_phoneme_segment_confidence_default_none(self):
        seg = PhonemeSegment(phoneme="a", start_time=0.0, duration=0.1)
        assert seg.confidence is None

    def test_phoneme_segment_confidence_set(self):
        seg = PhonemeSegment(phoneme="a", start_time=0.0, duration=0.1, confidence=0.8)
        assert seg.confidence == 0.8


class TestDualBackend:
    def test_dual_registered(self):
        assert "dual" in list_backends()

    def test_dual_runs_both_backends(self, monkeypatch):
        allosaurus = FakeBackend([PhonemeSegment("a", 0.0, 0.5)])
        omnilingual = FakeBackend([PhonemeSegment("b", 0.5, 0.5)])

        def fake_get_backend(name, **kwargs):
            return allosaurus if name == "allosaurus" else omnilingual

        monkeypatch.setattr(dual_module, "get_backend", fake_get_backend)

        backend = DualBackend()
        backend.recognize("/tmp/audio.wav", lang="nah")

        assert allosaurus.calls == 1
        assert omnilingual.calls == 1

    def test_dual_merges_by_confidence(self, monkeypatch):
        allosaurus = FakeBackend([PhonemeSegment("a", 0.0, 1.0, confidence=0.5)])
        omnilingual = FakeBackend([PhonemeSegment("b", 0.5, 0.4, confidence=0.9)])

        def fake_get_backend(name, **kwargs):
            return allosaurus if name == "allosaurus" else omnilingual

        monkeypatch.setattr(dual_module, "get_backend", fake_get_backend)

        backend = DualBackend()
        merged = backend.recognize("/tmp/audio.wav", lang="nah")
        assert len(merged) == 1
        assert merged[0].phoneme == "b"

    def test_dual_keeps_non_overlapping(self, monkeypatch):
        allosaurus = FakeBackend([PhonemeSegment("a", 0.0, 0.5, confidence=0.6)])
        omnilingual = FakeBackend([PhonemeSegment("b", 1.0, 0.5, confidence=0.7)])

        def fake_get_backend(name, **kwargs):
            return allosaurus if name == "allosaurus" else omnilingual

        monkeypatch.setattr(dual_module, "get_backend", fake_get_backend)

        backend = DualBackend()
        merged = backend.recognize("/tmp/audio.wav", lang="nah")
        assert len(merged) == 2
        assert merged[0].phoneme == "a"
        assert merged[1].phoneme == "b"

    def test_dual_fallback_single_backend(self, monkeypatch):
        allosaurus = FakeBackend(error=RuntimeError("fail"))
        omnilingual = FakeBackend([PhonemeSegment("b", 0.0, 0.5, confidence=0.7)])

        def fake_get_backend(name, **kwargs):
            return allosaurus if name == "allosaurus" else omnilingual

        monkeypatch.setattr(dual_module, "get_backend", fake_get_backend)

        backend = DualBackend()
        merged = backend.recognize("/tmp/audio.wav", lang="nah")
        assert len(merged) == 1
        assert merged[0].phoneme == "b"

    def test_dual_both_fail_raises(self, monkeypatch):
        allosaurus = FakeBackend(error=RuntimeError("fail"))
        omnilingual = FakeBackend(error=RuntimeError("fail"))

        def fake_get_backend(name, **kwargs):
            return allosaurus if name == "allosaurus" else omnilingual

        monkeypatch.setattr(dual_module, "get_backend", fake_get_backend)

        backend = DualBackend()
        with pytest.raises(RuntimeError):
            backend.recognize("/tmp/audio.wav", lang="nah")

    def test_dual_is_available_one_backend(self, monkeypatch):
        monkeypatch.setattr(dual_module.AllosaurusBackend, "is_available", classmethod(lambda cls: True))
        monkeypatch.setattr(dual_module.OmnilingualBackend, "is_available", classmethod(lambda cls: False))
        assert DualBackend.is_available() is True

    def test_dual_is_available_none(self, monkeypatch):
        monkeypatch.setattr(dual_module.AllosaurusBackend, "is_available", classmethod(lambda cls: False))
        monkeypatch.setattr(dual_module.OmnilingualBackend, "is_available", classmethod(lambda cls: False))
        assert DualBackend.is_available() is False
