import importlib
import sys
import types

import pytest


def _import_tenepal_modal():
    """Import tenepal_modal even when the optional 'modal' package is absent."""
    try:
        return importlib.import_module("tenepal_modal")
    except ModuleNotFoundError as exc:
        if exc.name != "modal":
            raise

    # Minimal modal stub for test environments without modal installed.
    modal_stub = types.ModuleType("modal")

    class _FakeImage:
        @staticmethod
        def debian_slim(*args, **kwargs):
            return _FakeImage()

        def apt_install(self, *args, **kwargs):
            return self

        def pip_install(self, *args, **kwargs):
            return self

        def add_local_dir(self, *args, **kwargs):
            return self

        def __getattr__(self, _name):
            # Support additional fluent image-builder methods used in tenepal_modal
            def _chain(*args, **kwargs):
                return self

            return _chain

    class _FakeVolume:
        @staticmethod
        def from_name(*args, **kwargs):
            return object()

    class _FakeSecret:
        @staticmethod
        def from_name(*args, **kwargs):
            return object()

    class _FakeApp:
        def __init__(self, *args, **kwargs):
            pass

        def function(self, *args, **kwargs):
            def _decorator(fn):
                return fn

            return _decorator

        def local_entrypoint(self, *args, **kwargs):
            def _decorator(fn):
                return fn

            return _decorator

    def _concurrent(*args, **kwargs):
        def _decorator(fn):
            return fn

        return _decorator

    modal_stub.Image = _FakeImage
    modal_stub.Volume = _FakeVolume
    modal_stub.Secret = _FakeSecret
    modal_stub.App = _FakeApp
    modal_stub.concurrent = _concurrent

    sys.modules.setdefault("modal", modal_stub)
    return importlib.import_module("tenepal_modal")


def test_isolated_tl_marker_is_downweighted():
    """Isolated tɬ (e.g. Spanish 'atlántico' artifact) must not get full NAH weight."""
    modal = _import_tenepal_modal()

    lang, score, scores = modal.identify_language_with_scores(["tɬ"])

    assert lang == "nah"
    assert score == pytest.approx(0.3, abs=1e-6)
    assert scores["nah"] == pytest.approx(0.3, abs=1e-6)


def test_tl_with_nah_core_marker_keeps_full_weight():
    """When NAH core evidence exists, tɬ should keep full marker strength."""
    modal = _import_tenepal_modal()

    _, _, isolated_scores = modal.identify_language_with_scores(["tɬ"])
    lang, score, scores = modal.identify_language_with_scores(["tɬ", "kʷ"])

    assert lang == "nah"
    assert score == pytest.approx(2.0, abs=1e-6)
    assert scores["nah"] > isolated_scores["nah"]


def test_detect_spanish_text_leak_flags_atlantico_without_indigenous_ipa():
    """Spanish text with no indigenous IPA markers should trigger SPA leak signal."""
    modal = _import_tenepal_modal()

    is_leak, reason = modal.detect_spanish_text_leak(
        text="El atlantico es grande",
        ipa="a t l a n t i k o",
        lang="nah",
    )

    assert is_leak is True
    assert reason


def test_detect_spanish_text_leak_respects_indigenous_ipa_markers():
    """Strong indigenous IPA markers must suppress Spanish leak override."""
    modal = _import_tenepal_modal()

    is_leak, reason = modal.detect_spanish_text_leak(
        text="El atlantico es grande",
        ipa="tɬ a kʷ i",
        lang="nah",
    )

    assert is_leak is False
    assert reason == ""
