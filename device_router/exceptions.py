"""Custom exceptions for device-router."""


class DeviceRouterError(Exception):
    """Base exception for device-router."""


class RoutingError(DeviceRouterError):
    """Raised when a routing decision cannot be made due to invalid inputs."""


class DetectionError(DeviceRouterError):
    """Raised when device detection fails."""
