"""CPU capability detection — AVX, VNNI, cores, frequency."""

import logging
import os
import platform
import struct
from typing import Any

logger = logging.getLogger(__name__)


def detect_cpu() -> dict[str, Any]:
    """Detect CPU features relevant to ML workloads.

    Returns dict with keys:
        available: always True
        arch, cores, threads, freq_mhz, features (avx, avx2, avx512, vnni, amx, etc.)
    """
    info: dict[str, Any] = {
        "available": True,
        "arch": platform.machine(),
        "processor": platform.processor(),
        "cores": None,
        "threads": None,
        "freq_mhz": None,
        "features": {},
    }

    # Core counts
    try:
        info["cores"] = os.cpu_count() or None
    except Exception:
        pass

    try:
        import psutil
        info["cores"] = psutil.cpu_count(logical=False)
        info["threads"] = psutil.cpu_count(logical=True)
        freq = psutil.cpu_freq()
        if freq:
            info["freq_mhz"] = round(freq.current, 0)
    except ImportError:
        logger.debug("psutil not installed — limited CPU info")

    # Feature detection
    info["features"] = _detect_features()

    return info


def _detect_features() -> dict[str, bool]:
    """Detect CPU instruction set features."""
    features: dict[str, bool] = {
        "avx": False,
        "avx2": False,
        "avx512f": False,
        "avx512_vnni": False,
        "avx512_bf16": False,
        "vnni": False,
        "amx": False,
        "neon": False,
        "sse4": False,
    }

    system = platform.system()

    if system == "Linux":
        _detect_linux(features)
    elif system == "Windows":
        _detect_windows(features)
    elif system == "Darwin":
        _detect_macos(features)

    return features


def _detect_linux(features: dict[str, bool]) -> None:
    """Parse /proc/cpuinfo for CPU features."""
    try:
        with open("/proc/cpuinfo", "r") as f:
            flags_line = ""
            for line in f:
                if line.startswith("flags"):
                    flags_line = line.lower()
                    break

        if not flags_line:
            return

        flag_set = set(flags_line.split())

        features["avx"] = "avx" in flag_set
        features["avx2"] = "avx2" in flag_set
        features["avx512f"] = "av512f" in flag_set or "avx512f" in flag_set
        features["vnni"] = "avx512_vnni" in flag_set or "avx_vnni" in flag_set
        features["sse4"] = "sse4_1" in flag_set or "sse4_2" in flag_set

        # Check for AVX512 VNNI and BF16
        features["avx512_vnni"] = "avx512_vnni" in flag_set
        features["avx512_bf16"] = "avx512_bf16" in flag_set
        features["amx"] = "amx_bf16" in flag_set or "amx_tile" in flag_set

        # ARM NEON
        features["neon"] = "neon" in flag_set or "asimd" in flag_set

    except Exception as e:
        logger.debug("Could not parse /proc/cpuinfo: %s", e)


def _detect_windows(features: dict[str, bool]) -> None:
    """Use ctypes to call CPUID on Windows."""
    try:
        import ctypes

        # CPUID with EAX=7, ECX=0 for extended features
        # This is a simplified check
        buf = (ctypes.c_uint32 * 4)()
        # Can't easily call CPUID from Python on Windows without a C extension.
        # Fall back to checking environment
        processor = platform.processor().lower()
        if "avx" in processor or "11th" in processor or "12th" in processor or "13th" in processor:
            features["avx"] = True
            features["avx2"] = True
    except Exception:
        pass


def _detect_macos(features: dict[str, bool]) -> None:
    """Detect Apple Silicon features."""
    if platform.machine() == "arm64":
        features["neon"] = True
        # Apple M-series has AMX (undocumented) and NEON
        features["amx"] = True
    else:
        features["sse4"] = True
        features["avx"] = True
        features["avx2"] = True
