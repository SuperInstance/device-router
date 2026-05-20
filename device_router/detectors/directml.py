"""DirectML / iGPU device detection."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def detect_directml() -> dict[str, Any]:
    """Detect DirectML-capable devices (typically integrated GPUs).

    Returns dict with keys:
        available: bool
        device_name: str or None
    """
    result: dict[str, Any] = {"available": False, "device_name": None}

    try:
        import torch_directml  # type: ignore
        dml = torch_directml.device()
        result["available"] = True
        # torch_directml doesn't expose rich device info easily
        result["device_name"] = "DirectML device"
        result["device"] = dml
    except ImportError:
        logger.debug("torch_directml not installed — DirectML detection skipped")
    except Exception as e:
        logger.warning("DirectML detection error: %s", e)

    return result
