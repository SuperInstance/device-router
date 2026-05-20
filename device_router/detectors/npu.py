"""NPU (Neural Processing Unit) detection — stub for future expansion."""

import logging
from typing import Any

logger = logging.getLogger(__name__)


def detect_npu() -> dict[str, Any]:
    """Detect NPU devices.

    Currently a stub. NPUs are emerging (Intel NPU, Qualcomm Hexagon NPU, Apple Neural Engine).
    Detection will be expanded as support matures.

    Returns dict with keys:
        available: bool
        device_name: str or None
        vendor: str or None
    """
    result: dict[str, Any] = {
        "available": False,
        "device_name": None,
        "vendor": None,
    }

    # Future: detect Intel NPU via openvino
    # Future: detect Qualcomm NPU via QNN
    # Future: detect Apple ANE via coreml

    logger.debug("NPU detection not yet implemented — returning unavailable")
    return result
