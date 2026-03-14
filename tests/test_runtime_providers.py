import base64
import types

import pytest

from tenepal.runtime import InferenceRequest, create_provider
from tenepal.runtime.modal_provider import ModalProvider
from tenepal.runtime.runpod_provider import RunPodProvider


def test_factory_creates_modal_provider():
    provider = create_provider("modal")
    assert isinstance(provider, ModalProvider)
    assert provider.name == "modal"


def test_factory_rejects_unknown_provider():
    with pytest.raises(ValueError, match="Unknown provider"):
        create_provider("nope")


def test_modal_provider_uses_remote_when_available(monkeypatch):
    calls = {}

    class _Fn:
        @staticmethod
        def remote(**kwargs):
            calls["payload"] = kwargs
            return {"ok": True}

    fake_module = types.SimpleNamespace(separate_voices_sepformer=_Fn)
    monkeypatch.setattr("importlib.import_module", lambda _name: fake_module)

    provider = ModalProvider()
    req = InferenceRequest(operation="separate_voices_sepformer", payload={"x": 1})
    result = provider.run(req)

    assert calls["payload"] == {"x": 1}
    assert result.output == {"ok": True}


def test_runpod_provider_sync_completed_decodes_bytes():
    captured = {}

    class _FakeRunPod(RunPodProvider):
        def _post_json(self, _url, payload):
            captured["payload"] = payload
            return {
                "status": "COMPLETED",
                "output": {
                    "ok": True,
                    "blob": {"__type__": "bytes", "base64": base64.b64encode(b"A").decode("ascii")},
                },
            }

        def _get_json(self, _url):
            raise AssertionError("Should not poll for sync completion")

    provider = _FakeRunPod(endpoint_id="ep", api_key="k")
    req = InferenceRequest(operation="op", payload={"audio_bytes": b"\x00\x01"})
    result = provider.run(req)

    encoded_audio = captured["payload"]["input"]["audio_bytes"]
    assert encoded_audio["__type__"] == "bytes"
    assert result.output["ok"] is True
    assert result.output["blob"] == b"A"


def test_runpod_provider_polls_until_completed(monkeypatch):
    class _FakeRunPod(RunPodProvider):
        def __init__(self):
            super().__init__(endpoint_id="ep", api_key="k")
            self.poll_calls = 0

        def _post_json(self, _url, _payload):
            return {"status": "IN_QUEUE", "id": "job-1"}

        def _get_json(self, _url):
            self.poll_calls += 1
            if self.poll_calls == 1:
                return {"status": "IN_PROGRESS"}
            return {"status": "COMPLETED", "output": {"value": 7}}

    monkeypatch.setattr("time.sleep", lambda _s: None)
    provider = _FakeRunPod()
    req = InferenceRequest(operation="op", payload={}, timeout_s=5, poll_interval_s=0.01)
    result = provider.run(req)

    assert provider.poll_calls == 2
    assert result.job_id == "job-1"
    assert result.output == {"value": 7}

