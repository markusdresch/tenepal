"""RunPod provider adapter for Tenepal remote operations.

This adapter is intentionally generic: operation routing is encoded in payload
(`operation` field), so a single RunPod endpoint can dispatch multiple tasks.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .base import CloudProvider, InferenceRequest, InferenceResult


def _encode_bytes(value: Any) -> Any:
    """Recursively encode bytes payloads for JSON transport."""
    if isinstance(value, bytes):
        return {
            "__type__": "bytes",
            "base64": base64.b64encode(value).decode("ascii"),
        }
    if isinstance(value, dict):
        return {k: _encode_bytes(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_encode_bytes(v) for v in value]
    return value


def _decode_bytes(value: Any) -> Any:
    """Recursively decode provider output containing encoded bytes."""
    if isinstance(value, dict):
        if value.get("__type__") == "bytes" and "base64" in value:
            return base64.b64decode(value["base64"])
        return {k: _decode_bytes(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_decode_bytes(v) for v in value]
    return value


@dataclass
class RunPodProvider(CloudProvider):
    """Execute operations through RunPod Serverless REST API."""

    endpoint_id: str
    api_key: str
    name: str = "runpod"

    @classmethod
    def from_env(cls) -> "RunPodProvider":
        endpoint_id = os.getenv("RUNPOD_ENDPOINT_ID", "").strip()
        api_key = os.getenv("RUNPOD_API_KEY", "").strip()
        if not endpoint_id:
            raise ValueError("RUNPOD_ENDPOINT_ID is required")
        if not api_key:
            raise ValueError("RUNPOD_API_KEY is required")
        return cls(endpoint_id=endpoint_id, api_key=api_key)

    def run(self, request: InferenceRequest) -> InferenceResult:
        run_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/run"
        status_url = f"https://api.runpod.ai/v2/{self.endpoint_id}/status"

        body = {
            "input": {
                "operation": request.operation,
                **_encode_bytes(request.payload),
            }
        }
        run_resp = self._post_json(run_url, body)

        # Synchronous endpoint response.
        if run_resp.get("status") == "COMPLETED":
            output = _decode_bytes(run_resp.get("output"))
            return InferenceResult(
                output=output,
                provider=self.name,
                operation=request.operation,
                metadata={"status": "COMPLETED"},
            )

        job_id = run_resp.get("id")
        if not job_id:
            raise RuntimeError(f"RunPod response missing job id: {run_resp}")

        deadline = time.time() + request.timeout_s
        while time.time() < deadline:
            poll_resp = self._get_json(f"{status_url}/{job_id}")
            status = str(poll_resp.get("status", "")).upper()
            if status == "COMPLETED":
                output = _decode_bytes(poll_resp.get("output"))
                return InferenceResult(
                    output=output,
                    provider=self.name,
                    operation=request.operation,
                    job_id=job_id,
                    metadata={"status": status},
                )
            if status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
                raise RuntimeError(f"RunPod job {job_id} failed with status={status}: {poll_resp}")
            time.sleep(request.poll_interval_s)

        raise TimeoutError(f"RunPod job {job_id} exceeded timeout ({request.timeout_s}s)")

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            method="POST",
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"RunPod POST {url} failed: {exc.code} {body}") from exc

    def _get_json(self, url: str) -> dict[str, Any]:
        req = urllib.request.Request(
            url,
            method="GET",
            headers={"Authorization": f"Bearer {self.api_key}"},
        )
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"RunPod GET {url} failed: {exc.code} {body}") from exc

