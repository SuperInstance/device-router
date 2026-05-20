"""DeviceRouter — routes ML workloads to the optimal compute device."""

import logging
from dataclasses import asdict
from typing import Any, Optional

from device_router.detectors import detect_cuda, detect_directml, detect_cpu, detect_npu
from device_router.strategies import RoutingStrategy, RoutingDecision

logger = logging.getLogger(__name__)

# Thresholds for routing decisions
_SMALL_MODEL_PARAMS = 100_000       # Below this → small model
_LARGE_MODEL_PARAMS = 10_000_000    # Above this → large model
_SMALL_BATCH = 8                    # Below this → small batch


class DeviceRouter:
    """Auto-detect compute devices and route ML workloads optimally.

    Works without any optional dependencies (pure CPU detection).
    With torch installed: adds CUDA detection and benchmarking.
    With torch-directml: adds iGPU detection.
    """

    def __init__(self):
        self._cuda: dict = {}
        self._directml: dict = {}
        self._cpu: dict = {}
        self._npu: dict = {}
        self._detected = False

    def detect(self) -> dict[str, Any]:
        """Detect all available compute devices.

        Returns overview dict with keys: cuda, cpu, directml, npu.
        """
        self._cuda = detect_cuda()
        self._directml = detect_directml()
        self._cpu = detect_cpu()
        self._npu = detect_npu()
        self._detected = True
        return self.overview()

    def overview(self) -> dict[str, Any]:
        """Return a system overview of all detected devices."""
        if not self._detected:
            self.detect()

        return {
            "cuda": {k: v for k, v in self._cuda.items() if k != "devices"} if self._cuda else {},
            "cpu": self._cpu,
            "igpu": {
                "available": self._directml.get("available", False),
                "device_name": self._directml.get("device_name"),
            },
            "npu": self._npu,
        }

    @property
    def cuda_available(self) -> bool:
        if not self._detected:
            self.detect()
        return self._cuda.get("available", False)

    @property
    def directml_available(self) -> bool:
        if not self._detected:
            self.detect()
        return self._directml.get("available", False)

    @property
    def cpu_info(self) -> dict:
        if not self._detected:
            self.detect()
        return self._cpu

    def route(
        self,
        model_size: int = 0,
        batch_size: int = 1,
        precision: str = "fp32",
        strategy: RoutingStrategy = RoutingStrategy.AUTO,
        is_training: bool = False,
        is_onnx: bool = False,
    ) -> RoutingDecision:
        """Route a workload to the optimal device.

        Args:
            model_size: Number of parameters in the model.
            batch_size: Batch size for inference/training.
            precision: "fp32", "fp16", "bf16", or "int8".
            strategy: Routing strategy to use.
            is_training: Whether this is a training workload.
            is_onnx: Whether the model is ONNX format.

        Returns:
            RoutingDecision with device choice and reasoning.
        """
        if not self._detected:
            self.detect()

        # ONNX models → CPU (most optimized path)
        if is_onnx:
            return RoutingDecision(
                device="cpu",
                reason="ONNX models are optimized for CPU execution",
                precision=precision,
                strategy=strategy,
            )

        # Training → always CUDA if available
        if is_training:
            if self.cuda_available:
                return RoutingDecision(
                    device="cuda",
                    reason="Training workloads benefit from GPU parallelism",
                    precision="bf16" if precision == "fp32" else precision,
                    strategy=strategy,
                    use_amp=True,
                )
            return RoutingDecision(
                device="cpu",
                reason="No CUDA GPU available — training on CPU",
                precision=precision,
                strategy=strategy,
                confidence=0.5,
            )

        # Strategy-based routing
        if strategy == RoutingStrategy.POWER:
            return self._route_power(model_size, batch_size, precision)
        elif strategy == RoutingStrategy.LATENCY:
            return self._route_latency(model_size, batch_size, precision)
        elif strategy == RoutingStrategy.THROUGHPUT:
            return self._route_throughput(model_size, batch_size, precision)
        else:
            return self._route_auto(model_size, batch_size, precision)

    def _route_auto(self, model_size: int, batch_size: int, precision: str) -> RoutingDecision:
        """Auto-routing: balance between latency and throughput."""
        # Very small models on CPU
        if model_size < _SMALL_MODEL_PARAMS:
            # Check for VNNI/AVX optimizations
            cpu_feats = self._cpu.get("features", {})
            if cpu_feats.get("vnni") and precision == "int8":
                return RoutingDecision(
                    device="cpu",
                    reason="Small model with int8 — VNNI-optimized CPU is fastest",
                    precision=precision,
                )
            return RoutingDecision(
                device="cpu",
                reason=f"Small model ({model_size:,} params) — CPU is sufficient",
                precision=precision,
            )

        # Large models with batching → GPU
        if model_size >= _LARGE_MODEL_PARAMS and batch_size > _SMALL_BATCH:
            if self.cuda_available:
                return RoutingDecision(
                    device="cuda",
                    reason=f"Large model ({model_size:,} params) with batch {batch_size} — GPU required",
                    precision="fp16" if precision == "fp32" else precision,
                    use_amp=True,
                )

        # Medium models or small batches
        if model_size >= _SMALL_MODEL_PARAMS:
            if self.cuda_available:
                return RoutingDecision(
                    device="cuda",
                    reason=f"Medium/large model ({model_size:,} params) — GPU recommended",
                    precision="fp16" if precision == "fp32" else precision,
                    use_amp=True,
                )
            if self.directml_available:
                return RoutingDecision(
                    device="directml",
                    reason=f"Medium model ({model_size:,} params) — iGPU offload",
                    precision=precision,
                    confidence=0.8,
                )

        # Fallback: CPU
        return RoutingDecision(
            device="cpu",
            reason="No accelerator available — using CPU",
            precision=precision,
            confidence=0.6 if model_size > _SMALL_MODEL_PARAMS else 1.0,
        )

    def _route_latency(self, model_size: int, batch_size: int, precision: str) -> RoutingDecision:
        """Optimize for lowest latency (single-sample)."""
        # For latency, CPU often wins for small models (no transfer overhead)
        if model_size < _SMALL_MODEL_PARAMS and batch_size <= 2:
            return RoutingDecision(
                device="cpu",
                reason="Latency-optimized: small model on CPU avoids GPU transfer overhead",
                precision=precision,
                strategy=RoutingStrategy.LATENCY,
            )
        # Larger models still benefit from GPU
        if self.cuda_available:
            return RoutingDecision(
                device="cuda",
                reason="Latency-optimized: GPU for larger model",
                precision="fp16" if precision == "fp32" else precision,
                strategy=RoutingStrategy.LATENCY,
                use_amp=True,
            )
        return RoutingDecision(
            device="cpu",
            reason="Latency-optimized: no GPU, CPU is the path",
            precision=precision,
            strategy=RoutingStrategy.LATENCY,
        )

    def _route_throughput(self, model_size: int, batch_size: int, precision: str) -> RoutingDecision:
        """Optimize for throughput (batched inference)."""
        if self.cuda_available:
            return RoutingDecision(
                device="cuda",
                reason="Throughput-optimized: GPU for batched inference",
                precision="fp16" if precision == "fp32" else precision,
                strategy=RoutingStrategy.THROUGHPUT,
                use_amp=True,
            )
        if self.directml_available:
            return RoutingDecision(
                device="directml",
                reason="Throughput-optimized: iGPU for batched inference",
                precision=precision,
                strategy=RoutingStrategy.THROUGHPUT,
                confidence=0.7,
            )
        return RoutingDecision(
            device="cpu",
            reason="Throughput-optimized: no accelerator, CPU with large batches",
            precision=precision,
            strategy=RoutingStrategy.THROUGHPUT,
        )

    def _route_power(self, model_size: int, batch_size: int, precision: str) -> RoutingDecision:
        """Optimize for power efficiency."""
        # Prefer CPU/iGPU for power efficiency
        if model_size < _LARGE_MODEL_PARAMS:
            if self.directml_available:
                return RoutingDecision(
                    device="directml",
                    reason="Power-optimized: iGPU for medium/small model",
                    precision=precision,
                    strategy=RoutingStrategy.POWER,
                )
            return RoutingDecision(
                device="cpu",
                reason="Power-optimized: CPU for small/medium model",
                precision=precision,
                strategy=RoutingStrategy.POWER,
            )
        # Large models still need GPU
        if self.cuda_available:
            return RoutingDecision(
                device="cuda",
                reason="Power-optimized: GPU necessary for large model",
                precision="fp16" if precision == "fp32" else precision,
                strategy=RoutingStrategy.POWER,
                use_amp=True,
                confidence=0.7,
            )
        return RoutingDecision(
            device="cpu",
            reason="Power-optimized: CPU (no GPU available)",
            precision=precision,
            strategy=RoutingStrategy.POWER,
        )

    def assign(self, device: str) -> Optional[Any]:
        """Get a torch device object for the given device string.

        Requires torch to be installed for CUDA/DirectML devices.
        Returns None if the device is not available.
        """
        if device == "cuda" and self.cuda_available:
            try:
                import torch
                return torch.device("cuda")
            except ImportError:
                return None
        elif device == "directml" and self.directml_available:
            return self._directml.get("device")
        elif device == "cpu":
            try:
                import torch
                return torch.device("cpu")
            except ImportError:
                return None
        return None
