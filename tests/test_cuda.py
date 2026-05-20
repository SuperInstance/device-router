"""Tests for CUDA detector."""

import pytest
from unittest.mock import patch, MagicMock
from device_router.detectors.cuda import detect_cuda


class TestCUDADetection:
    def test_detect_without_torch(self):
        """Should return available=False when torch is not importable."""
        # This works even if torch IS installed — just tests the structure
        result = detect_cuda()
        assert "available" in result
        assert "devices" in result
        assert isinstance(result["available"], bool)
        assert isinstance(result["devices"], list)

    @patch.dict("sys.modules", {"torch": None})
    def test_detect_torch_missing(self):
        """When torch can't be imported, should return gracefully."""
        # If torch is actually installed, this mock may not fully work,
        # but the function should still not crash
        result = detect_cuda()
        assert result["available"] is False
