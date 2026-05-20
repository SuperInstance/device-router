"""Tests for routing strategies."""

import pytest
from device_router.strategies import RoutingStrategy, RoutingDecision


class TestRoutingStrategy:
    def test_strategy_values(self):
        assert RoutingStrategy.AUTO.value == "auto"
        assert RoutingStrategy.LATENCY.value == "latency"
        assert RoutingStrategy.THROUGHPUT.value == "throughput"
        assert RoutingStrategy.POWER.value == "power"

    def test_routing_decision_defaults(self):
        d = RoutingDecision(device="cpu")
        assert d.device == "cpu"
        assert d.device_index == 0
        assert d.precision == "fp32"
        assert d.confidence == 1.0
        assert d.use_amp is False

    def test_routing_decision_custom(self):
        d = RoutingDecision(
            device="cuda",
            device_index=1,
            reason="test",
            precision="fp16",
            strategy=RoutingStrategy.THROUGHPUT,
            use_amp=True,
            confidence=0.9,
        )
        assert d.device == "cuda"
        assert d.device_index == 1
        assert d.precision == "fp16"
        assert d.use_amp is True
        assert d.confidence == 0.9
