"""DeviceRouter — heterogeneous compute routing for ML workloads."""

from device_router.router import DeviceRouter
from device_router.strategies import RoutingStrategy
from device_router.exceptions import DeviceRouterError, RoutingError, DetectionError

__all__ = ["DeviceRouter", "RoutingStrategy", "DeviceRouterError", "RoutingError", "DetectionError"]
__version__ = "0.1.0"


def register_device_router(registry):
    """Mesh integration entry point for superinstance.plugins."""
    registry.register("devices", "router", DeviceRouter)
