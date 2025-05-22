# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from vllm.platforms import PlatformEnum, current_platform


def test_metal_platform_detection():
    if not torch.backends.mps.is_available():
        pytest.skip("Metal is not available")

    assert current_platform._enum == PlatformEnum.METAL
    assert current_platform.device_name == "metal"
    assert current_platform.device_type == "metal"


def test_metal_supported_dtypes():
    if not torch.backends.mps.is_available():
        pytest.skip("Metal is not available")

    supported_dtypes = current_platform.supported_dtypes
    assert torch.float16 in supported_dtypes
    assert torch.float32 in supported_dtypes


def test_metal_device_info():
    if not torch.backends.mps.is_available():
        pytest.skip("Metal is not available")

    device_name = current_platform.get_device_name()
    assert device_name == "Apple Silicon GPU"

    total_memory = current_platform.get_device_total_memory()
    assert total_memory > 0
