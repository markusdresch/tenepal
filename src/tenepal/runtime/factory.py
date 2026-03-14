"""Factory helpers for cloud runtime providers."""

from __future__ import annotations

from .base import CloudProvider
from .modal_provider import ModalProvider
from .runpod_provider import RunPodProvider


def create_provider(name: str, **kwargs) -> CloudProvider:
    provider = name.strip().lower()
    if provider == "modal":
        return ModalProvider(**kwargs)
    if provider == "runpod":
        if kwargs:
            return RunPodProvider(**kwargs)
        return RunPodProvider.from_env()
    raise ValueError(f"Unknown provider '{name}'. Expected one of: modal, runpod")

