"""Modal provider adapter for Tenepal remote operations."""

from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any

from .base import CloudProvider, InferenceRequest, InferenceResult


@dataclass
class ModalProvider(CloudProvider):
    """Execute operations via functions exposed in `tenepal_modal`."""

    name: str = "modal"
    module_name: str = "tenepal_modal"
    operation_map: dict[str, str] = field(
        default_factory=lambda: {
            "separate_voices_sepformer": "separate_voices_sepformer",
            "run_vibevoice": "run_vibevoice",
        }
    )

    def run(self, request: InferenceRequest) -> InferenceResult:
        module = importlib.import_module(self.module_name)
        attr = self.operation_map.get(request.operation, request.operation)
        fn = getattr(module, attr)

        # Modal functions expose `.remote(...)`; local test doubles often don't.
        remote_fn = getattr(fn, "remote", None)
        if callable(remote_fn):
            output = remote_fn(**request.payload)
        else:
            output = fn(**request.payload)

        return InferenceResult(
            output=output,
            provider=self.name,
            operation=request.operation,
        )

