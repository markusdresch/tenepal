"""Provider abstractions for remote inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class InferenceRequest:
    """Provider-agnostic request for a remote inference operation."""

    operation: str
    payload: dict[str, Any]
    timeout_s: int = 900
    poll_interval_s: float = 2.0


@dataclass
class InferenceResult:
    """Provider-agnostic response container."""

    output: Any
    provider: str
    operation: str
    job_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class CloudProvider(ABC):
    """Interface for cloud execution providers (Modal, RunPod, ...)."""

    name: str

    @abstractmethod
    def run(self, request: InferenceRequest) -> InferenceResult:
        """Execute a remote inference request and return normalized result."""

