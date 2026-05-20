"""Tests for CPU detector."""

import pytest
from device_router.detectors.cpu import detect_cpu


class TestCPUDetection:
    def test_always_available(self):
        result = detect_cpu()
        assert result["available"] is True

    def test_has_arch(self):
        result = detect_cpu()
        assert result["arch"] is not None

    def test_features_dict_keys(self):
        result = detect_cpu()
        features = result["features"]
        expected_keys = {"avx", "avx2", "vnni", "neon", "sse4", "amx"}
        assert expected_keys.issubset(set(features.keys()))

    def test_features_are_bools(self):
        result = detect_cpu()
        for key, val in result["features"].items():
            assert isinstance(val, bool), f"{key} should be bool, got {type(val)}"

    def test_cores_count(self):
        result = detect_cpu()
        cores = result.get("cores")
        if cores is not None:
            assert cores > 0
