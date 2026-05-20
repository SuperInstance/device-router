"""CUDA device detection and capability query."""

import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)


def detect_cuda() -> dict[str, Any]:
    """Detect CUDA-capable GPUs and return capability info.

    Returns dict with keys:
        available: bool
        devices: list of dicts with name, memory_total_mb, compute_capability, etc.
    """
    result: dict[str, Any] = {"available": False, "devices": []}

    try:
        import torch
    except ImportError:
        logger.debug("torch not installed — CUDA detection skipped")
        return result

    if not torch.cuda.is_available():
        return result

    result["available"] = True
    try:
        device_count = torch.cuda.device_count()
        for i in range(device_count):
            props = torch.cuda.get_device_properties(i)
            result["devices"].append({
                "index": i,
                "name": props.name,
                "memory_total_mb": round(props.total_memory / 1e6),
                "major": props.major,
                "minor": props.minor,
                "multi_processor_count": getattr(props, "multi_processor_count", None),
                "compute_capability": f"{props.major}.{props.minor}",
            })
        result["cuda_version"] = torch.version.cuda
        result["cudnn_version"] = getattr(torch.backends.cudnn, "version", lambda: None)()
    except Exception as e:
        logger.warning("CUDA detection error: %s", e)

    return result
