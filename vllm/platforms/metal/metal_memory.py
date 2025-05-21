import torch
import torch.mps
from typing import Optional, Tuple, List
import numpy as np
from vllm.logger import init_logger

logger = init_logger(__name__)

class MetalMemoryManager:
    """Manages memory allocation and deallocation for Metal buffers."""
    
    def __init__(self):
        self._allocated_buffers: List[torch.Tensor] = []
        self._total_allocated = 0
        self._max_memory = self._get_max_memory()
        
    def _get_max_memory(self) -> int:
        """Get the maximum available memory on the Metal device."""
        try:
            # Get device memory info from PyTorch MPS
            device = torch.device("mps")
            return torch.mps.current_allocated_memory() + torch.mps.driver_allocated_memory()
        except Exception as e:
            logger.warning(f"Failed to get Metal device memory info: {e}")
            return 0

    def allocate(self, size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Allocate a new Metal buffer of specified size and dtype.
        
        Args:
            size: Size of the buffer in elements
            dtype: Data type of the buffer
            
        Returns:
            torch.Tensor: The allocated buffer on Metal device
            
        Raises:
            RuntimeError: If allocation would exceed device memory capacity
            MemoryError: If allocation fails due to insufficient memory
        """
        try:
            # Calculate required memory
            element_size = torch.tensor([], dtype=dtype).element_size()
            required_memory = size * element_size
            
            # Check if allocation would exceed device capacity
            if self._total_allocated + required_memory > self._max_memory:
                raise RuntimeError(f"Allocation would exceed device memory capacity. Required: {required_memory}, Available: {self._max_memory - self._total_allocated}")
            
            # Create tensor on MPS device
            tensor = torch.empty(size, dtype=dtype, device="mps")
            self._allocated_buffers.append(tensor)
            self._total_allocated += tensor.element_size() * tensor.nelement()
            
            if self._total_allocated > self._max_memory:
                logger.warning("Memory allocation exceeds device capacity")
                # Clean up and raise error
                self.free(tensor)
                raise RuntimeError("Memory allocation exceeds device capacity")
                
            return tensor
        except RuntimeError as e:
            raise
        except Exception as e:
            logger.error(f"Failed to allocate Metal buffer: {e}")
            raise MemoryError(f"Failed to allocate Metal buffer: {e}")

    def free(self, tensor: torch.Tensor) -> None:
        """
        Free a previously allocated Metal buffer.
        
        Args:
            tensor: The tensor to free
        """
        if tensor in self._allocated_buffers:
            self._allocated_buffers.remove(tensor)
            self._total_allocated -= tensor.element_size() * tensor.nelement()
            del tensor
            torch.mps.empty_cache()

    def get_memory_stats(self) -> Tuple[int, int]:
        """
        Get current memory usage statistics.
        
        Returns:
            Tuple[int, int]: (total allocated memory, max available memory)
        """
        return self._total_allocated, self._max_memory

    def clear(self) -> None:
        """Clear all allocated buffers."""
        for tensor in self._allocated_buffers:
            del tensor
        self._allocated_buffers.clear()
        self._total_allocated = 0
        torch.mps.empty_cache()

    def __del__(self):
        """Cleanup when the manager is destroyed."""
        self.clear() 