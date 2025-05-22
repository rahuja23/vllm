# SPDX-License-Identifier: Apache-2.0

from typing import TYPE_CHECKING, Optional

import torch

from vllm.logger import init_logger

from .interface import Platform, PlatformEnum

if TYPE_CHECKING:
    from vllm.config import VllmConfig
else:
    VllmConfig = None

logger = init_logger(__name__)


class MetalPlatform(Platform):
    _enum = PlatformEnum.METAL  # We'll need to add this to PlatformEnum
    device_name: str = "metal"
    device_type: str = "metal"
    dispatch_key: str = "MPS"  # PyTorch uses MPS for Metal

    @property
    def supported_dtypes(self) -> list:
        # Metal supports float16 and float32
        return [torch.float16, torch.float32]

    @classmethod
    def get_device_name(cls, device_id: int = 0) -> str:
        return "Apple Silicon GPU"

    @classmethod
    def get_device_total_memory(cls, device_id: int = 0) -> int:
        # Get total memory from MPS device
        return torch.mps.get_device_properties(device_id).total_memory

    @classmethod
    def is_async_output_supported(cls, enforce_eager: Optional[bool]) -> bool:
        return True
