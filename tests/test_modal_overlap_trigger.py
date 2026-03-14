import importlib
import sys
import types


def _import_tenepal_modal():
    """Import tenepal_modal even when optional runtime deps are missing."""
    try:
        return importlib.import_module("tenepal_modal")
    except ModuleNotFoundError as exc:
        if exc.name != "modal":
            raise

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


class _FakePitch:
    def __init__(self, values):
        self._values = list(values)

    def xs(self):
        return [i * 0.01 for i in range(len(self._values))]

    def get_value_at_time(self, t):
        idx = int(round(t / 0.01))
        idx = max(0, min(idx, len(self._values) - 1))
        return self._values[idx]


class _FakeIntensity:
    def __init__(self, values):
        self._values = list(values)

    def xs(self):
        return [i * 0.01 for i in range(len(self._values))]

    def get_value(self, t):
        idx = int(round(t / 0.01))
        idx = max(0, min(idx, len(self._values) - 1))
        return self._values[idx]


class _FakeHarmonicity:
    def __init__(self, hnr):
        self.hnr = hnr


class _FakeWindow:
    def __init__(self, f0_values, intensity_values, hnr):
        self._f0_values = f0_values
        self._intensity_values = intensity_values
        self._hnr = hnr

    def to_pitch(self, *args, **kwargs):
        return _FakePitch(self._f0_values)

    def to_intensity(self, *args, **kwargs):
        return _FakeIntensity(self._intensity_values)


class _FakeSound:
    def __init__(self, window):
        self._window = window

    def extract_part(self, *args, **kwargs):
        return self._window


def _install_fake_parselmouth(hnr_value):
    """Install a minimal parselmouth stub used by detect_overlap()."""
    parselmouth_mod = types.ModuleType("parselmouth")
    parselmouth_mod.WindowShape = types.SimpleNamespace(HAMMING="HAMMING")

    praat_mod = types.ModuleType("parselmouth.praat")

    def _call(obj, command, *args):
        if command == "To Harmonicity (cc)":
            return _FakeHarmonicity(hnr_value)
        if command == "Get mean":
            return getattr(obj, "hnr", 0.0)
        raise ValueError(f"Unexpected praat call: {command}")

    praat_mod.call = _call
    sys.modules["parselmouth"] = parselmouth_mod
    sys.modules["parselmouth.praat"] = praat_mod


def test_detect_overlap_short_segment_returns_too_short():
    modal = _import_tenepal_modal()
    dummy = object()

    is_overlap, details = modal.detect_overlap(dummy, 1.0, 1.3)

    assert is_overlap is False
    assert details["reason"] == "too_short"


def test_detect_overlap_bimodal_f0_triggers_overlap():
    modal = _import_tenepal_modal()
    _install_fake_parselmouth(hnr_value=6.0)  # low HNR supports overlap

    # Balanced low/high voiced bands -> bimodal=True
    f0_values = [120.0] * 30 + [240.0] * 30
    intensity_values = [0.8, 1.2] * 30  # high variance
    window = _FakeWindow(f0_values=f0_values, intensity_values=intensity_values, hnr=6.0)
    snd = _FakeSound(window)

    is_overlap, details = modal.detect_overlap(snd, 0.0, 2.0)

    assert is_overlap is True
    assert details["bimodal"] is True
    assert details["low_ratio"] > 0.2
    assert details["high_ratio"] > 0.2


def test_detect_overlap_requires_bimodal_signal_even_with_low_hnr():
    modal = _import_tenepal_modal()
    _install_fake_parselmouth(hnr_value=4.0)  # very low HNR alone should not be enough

    # Unimodal (only low band) -> must not trigger overlap
    f0_values = [130.0] * 60
    intensity_values = [0.8, 1.2] * 30
    window = _FakeWindow(f0_values=f0_values, intensity_values=intensity_values, hnr=4.0)
    snd = _FakeSound(window)

    is_overlap, details = modal.detect_overlap(snd, 0.0, 2.0)

    assert is_overlap is False
    assert details["bimodal"] is False
    assert details["low_hnr"] is True


def test_detect_overlap_handles_extract_failure():
    modal = _import_tenepal_modal()

    class _BrokenSound:
        def extract_part(self, *args, **kwargs):
            raise RuntimeError("boom")

    is_overlap, details = modal.detect_overlap(_BrokenSound(), 0.0, 2.0)

    assert is_overlap is False
    assert details["reason"] == "extract_failed"
