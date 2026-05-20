"""Device detectors for CUDA, DirectML, CPU, and NPU."""

from device_router.detectors.cuda import detect_cuda
from device_router.detectors.directml import detect_directml
from device_router.detectors.cpu import detect_cpu
from device_router.detectors.npu import detect_npu

__all__ = ["detect_cuda", "detect_directml", "detect_cpu", "detect_npu"]
