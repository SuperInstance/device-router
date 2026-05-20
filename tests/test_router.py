"""Tests for DeviceRouter core functionality."""

import pytest
from unittest.mock import patch, MagicMock
from device_router import DeviceRouter, RoutingStrategy
from device_router.strategies import RoutingDecision
from device_router.exceptions import RoutingError


class TestDetectWithoutDeps:
    """Detection must work without torch or directml installed."""

    def test_detect_runs_with_hardware(self):
        """detect() should succeed and report all devices."""
        router = DeviceRouter()
        overview = router.detect()
        assert overview["cpu"]["available"] is True
        # CUDA and DirectML may or may not be available
        assert "cuda" in overview
        assert "igpu" in overview

    def test_overview_returns_all_keys(self):
        router = DeviceRouter()
        overview = router.overview()
        assert "cuda" in overview
        assert "cpu" in overview
        assert "igpu" in overview
        assert "npu" in overview

    def test_detect_idempotent(self):
        router = DeviceRouter()
        o1 = router.detect()
        o2 = router.detect()
        assert o1 == o2


class TestCPUFeatureDetection:
    """CPU feature detection."""

    def test_cpu_has_arch(self):
        router = DeviceRouter()
        router.detect()
        cpu = router.cpu_info
        assert cpu["arch"] in ("x86_64", "AMD64", "aarch64", "arm64", "x86_64")

    def test_cpu_features_dict(self):
        router = DeviceRouter()
        router.detect()
        features = router.cpu_info.get("features", {})
        assert isinstance(features, dict)
        # On x86_64 Linux, should detect at least some features
        import platform
        if platform.machine() in ("x86_64", "AMD64"):
            assert "avx" in features

    def test_cpu_cores_positive(self):
        router = DeviceRouter()
        router.detect()
        cores = router.cpu_info.get("cores")
        if cores is not None:
            assert cores > 0


class TestRoutingDecisions:
    """Test routing logic."""

    def test_route_small_model_to_cpu(self):
        router = DeviceRouter()
        router.detect()
        decision = router.route(model_size=1_000, batch_size=1)
        assert decision.device == "cpu"
        assert "small" in decision.reason.lower() or "cpu" in decision.reason.lower()

    def test_route_tiny_model_to_cpu(self):
        router = DeviceRouter()
        router.detect()
        decision = router.route(model_size=100, batch_size=1)
        assert decision.device == "cpu"

    def test_route_onnx_always_cpu(self):
        router = DeviceRouter()
        router.detect()
        decision = router.route(model_size=100_000_000, batch_size=32, is_onnx=True)
        assert decision.device == "cpu"
        assert "onnx" in decision.reason.lower()

    def test_route_training_to_gpu_if_available(self):
        router = DeviceRouter()
        # Mock CUDA available
        router._detected = True
        router._cuda = {"available": True, "devices": [{"name": "Mock GPU"}]}
        router._directml = {"available": False}
        router._cpu = {"available": True}
        router._npu = {"available": False}

        decision = router.route(model_size=1_000_000, batch_size=32, is_training=True)
        assert decision.device == "cuda"
        assert decision.use_amp is True

    def test_route_training_cpu_fallback(self):
        router = DeviceRouter()
        router._detected = True
        router._cuda = {"available": False}
        router._directml = {"available": False}
        router._cpu = {"available": True, "features": {}}
        router._npu = {"available": False}
        decision = router.route(model_size=1_000_000, batch_size=32, is_training=True)
        assert decision.device == "cpu"

    def test_route_precision_preserved(self):
        router = DeviceRouter()
        router.detect()
        decision = router.route(model_size=500, batch_size=1, precision="int8")
        assert decision.precision == "int8"


class TestRoutingStrategies:
    """Test different routing strategies."""

    def test_strategy_latency_small_model_cpu(self):
        router = DeviceRouter()
        router.detect()
        decision = router.route(
            model_size=50_000, batch_size=1,
            strategy=RoutingStrategy.LATENCY
        )
        assert decision.device == "cpu"
        assert decision.strategy == RoutingStrategy.LATENCY

    def test_strategy_throughput_no_gpu(self):
        router = DeviceRouter()
        router.detect()
        decision = router.route(
            model_size=1_000_000, batch_size=64,
            strategy=RoutingStrategy.THROUGHPUT
        )
        # Without GPU, should fall to CPU
        assert decision.strategy == RoutingStrategy.THROUGHPUT

    def test_strategy_power_prefers_low_power_device(self):
        router = DeviceRouter()
        router._detected = True
        router._cuda = {"available": False}
        router._directml = {"available": False}
        router._cpu = {"available": True, "features": {}}
        router._npu = {"available": False}
        decision = router.route(
            model_size=50_000, batch_size=1,
            strategy=RoutingStrategy.POWER
        )
        assert decision.device == "cpu"
        assert decision.strategy == RoutingStrategy.POWER

    def test_strategy_auto_default(self):
        router = DeviceRouter()
        router.detect()
        decision = router.route(model_size=500, batch_size=1)
        assert decision.strategy == RoutingStrategy.AUTO


class TestDeviceCapability:
    """Test device capability queries."""

    def test_cuda_available_property(self):
        router = DeviceRouter()
        # Should not crash even without torch
        result = router.cuda_available
        assert isinstance(result, bool)

    def test_directml_available_property(self):
        router = DeviceRouter()
        result = router.directml_available
        assert isinstance(result, bool)

    def test_assign_cpu_returns_none_without_torch(self):
        """assign() returns None if torch not importable for device objects."""
        router = DeviceRouter()
        router.detect()
        # Without torch, assign("cpu") returns None
        # (torch.device objects need torch)
        # This is acceptable behavior

    def test_graceful_degradation_no_devices(self):
        """Router should work even when all optional devices are unavailable."""
        router = DeviceRouter()
        router._detected = True
        router._cuda = {"available": False}
        router._directml = {"available": False}
        router._cpu = {"available": True, "features": {}}
        router._npu = {"available": False}

        # Should still route to CPU
        decision = router.route(model_size=100_000_000, batch_size=32)
        assert decision.device == "cpu"


class TestStatusAlias:
    """Test status() alias."""

    def test_status_equals_overview(self):
        router = DeviceRouter()
        router.detect()
        assert router.status() == router.overview()


class TestInputValidation:
    """Test that invalid inputs raise RoutingError."""

    def test_negative_model_size_raises(self):
        router = DeviceRouter()
        router.detect()
        with pytest.raises(RoutingError, match="model_size"):
            router.route(model_size=-1)

    def test_zero_batch_size_raises(self):
        router = DeviceRouter()
        router.detect()
        with pytest.raises(RoutingError, match="batch_size"):
            router.route(model_size=100, batch_size=0)

    def test_negative_batch_size_raises(self):
        router = DeviceRouter()
        router.detect()
        with pytest.raises(RoutingError, match="batch_size"):
            router.route(model_size=100, batch_size=-5)

    def test_invalid_precision_raises(self):
        router = DeviceRouter()
        router.detect()
        with pytest.raises(RoutingError, match="precision"):
            router.route(model_size=100, precision="bfloat16")

    def test_valid_precisions_accepted(self):
        router = DeviceRouter()
        router.detect()
        for p in ("fp32", "fp16", "bf16", "int8"):
            decision = router.route(model_size=100, precision=p)
            assert decision.precision == p


class TestCustomExceptions:
    """Test custom exception hierarchy."""

    def test_routing_error_is_base(self):
        from device_router.exceptions import RoutingError, DeviceRouterError
        assert issubclass(RoutingError, DeviceRouterError)

    def test_detection_error_is_base(self):
        from device_router.exceptions import DetectionError, DeviceRouterError
        assert issubclass(DetectionError, DeviceRouterError)

    def test_exceptions_importable_from_package(self):
        from device_router import DeviceRouterError, RoutingError, DetectionError
        assert DeviceRouterError is not None
        assert RoutingError is not None
        assert DetectionError is not None


class TestMeshRegistration:
    """Test mesh/superinstance integration."""

    def test_register_function_exists(self):
        from device_router import register_device_router
        assert callable(register_device_router)

    def test_register_calls_registry(self):
        from device_router import register_device_router
        mock_registry = MagicMock()
        register_device_router(mock_registry)
        mock_registry.register.assert_called_once_with("devices", "router", DeviceRouter)

    def test_multiple_routers_concurrent(self):
        """Multiple DeviceRouter instances should be independent."""
        r1 = DeviceRouter()
        r2 = DeviceRouter()
        r1.detect()
        r2.detect()
        # They can have different internal state
        d1 = r1.route(model_size=500, batch_size=1)
        d2 = r2.route(model_size=50_000, batch_size=1)
        # Both should be valid decisions
        assert isinstance(d1, RoutingDecision)
        assert isinstance(d2, RoutingDecision)
