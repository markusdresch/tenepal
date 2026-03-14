"""Container health check for GPU and ML dependencies."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import re
from typing import Any, Dict, Optional


def _check_python() -> Dict[str, Any]:
    return {
        "ok": sys.version_info.major == 3 and sys.version_info.minor == 12,
        "version": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
    }


def _check_vulkan() -> Dict[str, Any]:
    device_name: Optional[str] = None
    device_type_raw: Optional[str] = None
    compute_queues = 0

    try:
        result = subprocess.run(
            ["vulkaninfo"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            return {
                "ok": False,
                "device_name": None,
                "device_type": "unknown",
                "compute_queues": 0,
                "software_renderer": False,
            }

        current_flags: Optional[str] = None
        current_count: Optional[int] = None

        def finalize_queue_family() -> None:
            nonlocal compute_queues, current_flags, current_count
            if current_flags and current_count is not None and "COMPUTE" in current_flags:
                compute_queues += current_count
            current_flags = None
            current_count = None

        for line in result.stdout.splitlines():
            if device_name is None:
                match = re.search(r"deviceName\s*=\s*(.+)", line)
                if match:
                    device_name = match.group(1).strip()
            if device_type_raw is None:
                match = re.search(r"deviceType\s*=\s*(.+)", line)
                if match:
                    device_type_raw = match.group(1).strip()

            if "VkQueueFamilyProperties" in line:
                finalize_queue_family()
                continue

            if "queueFlags" in line:
                current_flags = line
                continue

            if "queueCount" in line:
                match = re.search(r"queueCount\s*=\s*(\d+)", line)
                if match:
                    current_count = int(match.group(1))

        finalize_queue_family()

        device_type_value = device_type_raw or ""
        if "PHYSICAL_DEVICE_TYPE_DISCRETE_GPU" in device_type_value:
            device_type = "discrete"
        elif "PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU" in device_type_value:
            device_type = "integrated"
        elif "PHYSICAL_DEVICE_TYPE_CPU" in device_type_value:
            device_type = "cpu"
        else:
            device_type = "unknown"

        software_renderer = device_type == "cpu"
        if device_name and "llvmpipe" in device_name.lower():
            software_renderer = True

        ok = device_type in {"discrete", "integrated"} and compute_queues >= 1

        return {
            "ok": ok,
            "device_name": device_name,
            "device_type": device_type,
            "compute_queues": compute_queues,
            "software_renderer": software_renderer,
        }
    except Exception:
        return {
            "ok": False,
            "device_name": None,
            "device_type": "unknown",
            "compute_queues": 0,
            "software_renderer": False,
        }


def _check_torch() -> Dict[str, Any]:
    try:
        import torch

        return {
            "ok": True,
            "version": torch.__version__,
            "gpu": bool(torch.cuda.is_available()),
        }
    except Exception:
        return {"ok": False, "version": None, "gpu": False}


def _check_vram() -> Dict[str, Any]:
    try:
        import torch

        if not torch.cuda.is_available():
            return {"ok": False, "total_mb": None, "free_mb": None}
        total = torch.cuda.get_device_properties(0).total_memory / (1024 * 1024)
        free = torch.cuda.mem_get_info()[0] / (1024 * 1024)
        return {"ok": True, "total_mb": int(total), "free_mb": int(free)}
    except Exception:
        return {"ok": False, "total_mb": None, "free_mb": None}


def _check_models() -> Dict[str, Any]:
    path = "/models"
    ok = os.path.isdir(path) and os.access(path, os.R_OK)
    return {"ok": ok, "path": path}


def _check_import(module: str) -> Dict[str, Any]:
    try:
        __import__(module)
        return {"ok": True}
    except Exception:
        return {"ok": False}


def main() -> int:
    checks = {
        "python": _check_python(),
        "vulkan": _check_vulkan(),
        "pytorch": _check_torch(),
        "vram": _check_vram(),
        "models": _check_models(),
        "pyannote": _check_import("pyannote.audio"),
        "fairseq2": _check_import("fairseq2"),
    }

    critical_ok = checks["python"]["ok"] and checks["models"]["ok"]
    if not critical_ok:
        status = "unhealthy"
        exit_code = 2
    else:
        noncritical_ok = all(check["ok"] for key, check in checks.items() if key not in {"python", "models"})
        status = "healthy" if noncritical_ok else "degraded"
        exit_code = 0 if noncritical_ok else 1
        vulkan_result = checks.get("vulkan", {})
        if vulkan_result.get("software_renderer") or vulkan_result.get("compute_queues", 0) < 1:
            status = "degraded"
            exit_code = 1

    payload = {"status": status, "checks": checks}
    print(json.dumps(payload))
    print(f"Status: {status}")
    for name, info in checks.items():
        print(f"- {name}: {info}")

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
