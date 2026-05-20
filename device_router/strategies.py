"""Routing strategies for device selection."""

from enum import Enum
from dataclasses import dataclass
from typing import Optional


class RoutingStrategy(Enum):
    """Strategy for selecting which device to use for a workload."""
    AUTO = "auto"           # Best guess based on workload characteristics
    LATENCY = "latency"     # Optimize for lowest latency (single-sample)
    THROUGHPUT = "throughput"  # Optimize for throughput (batched)
    POWER = "power"         # Optimize for power efficiency (prefer CPU/iGPU)


@dataclass
class RoutingDecision:
    """Result of a routing decision."""
    device: str               # "cuda", "cpu", "directml", "npu"
    device_index: int = 0     # Device index (for multi-GPU)
    reason: str = ""          # Human-readable explanation
    precision: str = "fp32"   # Recommended precision
    strategy: RoutingStrategy = RoutingStrategy.AUTO
    use_amp: bool = False     # Whether to use automatic mixed precision
    confidence: float = 1.0   # Confidence in this decision (0-1)
