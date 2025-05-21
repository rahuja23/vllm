import torch
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from vllm.platforms.metal.metal_memory import MetalMemoryManager
from vllm.logger import init_logger

logger = init_logger(__name__)

class MetalMemoryPool:
    """A memory pool for efficient Metal buffer reuse."""
    
    def __init__(self, memory_manager: MetalMemoryManager):
        self.memory_manager = memory_manager
        # Dictionary mapping (size, dtype) to list of available buffers
        self._available_buffers: Dict[Tuple[int, torch.dtype], List[torch.Tensor]] = defaultdict(list)
        # Dictionary mapping tensor to its (size, dtype)
        self._buffer_info: Dict[torch.Tensor, Tuple[int, torch.dtype]] = {}
        
    def allocate(self, size: int, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        """
        Allocate a buffer from the pool or create a new one if none available.
        
        Args:
            size: Size of the buffer in elements
            dtype: Data type of the buffer
            
        Returns:
            torch.Tensor: The allocated buffer
        """
        key = (size, dtype)
        
        # Try to get a buffer from the pool
        if self._available_buffers[key]:
            tensor = self._available_buffers[key].pop()
            self._buffer_info[tensor] = key
            return tensor
            
        # Create new buffer if none available
        tensor = self.memory_manager.allocate(size, dtype)
        self._buffer_info[tensor] = key
        return tensor
        
    def free(self, tensor: torch.Tensor) -> None:
        """
        Return a buffer to the pool instead of freeing it.
        
        Args:
            tensor: The tensor to return to the pool
        """
        if tensor in self._buffer_info:
            key = self._buffer_info[tensor]
            self._available_buffers[key].append(tensor)
            del self._buffer_info[tensor]
            
    def clear(self) -> None:
        """Clear all buffers in the pool."""
        for buffers in self._available_buffers.values():
            for tensor in buffers:
                self.memory_manager.free(tensor)
        self._available_buffers.clear()
        self._buffer_info.clear()
        
    def get_pool_stats(self) -> Dict[str, int]:
        """
        Get statistics about the memory pool.
        
        Returns:
            Dict[str, int]: Statistics about the pool
        """
        total_buffers = sum(len(buffers) for buffers in self._available_buffers.values())
        total_memory = sum(
            size * len(buffers)
            for (size, _), buffers in self._available_buffers.items()
        )
        
        return {
            "total_buffers": total_buffers,
            "total_memory": total_memory,
            "unique_sizes": len(self._available_buffers)
        }
        
    def __del__(self):
        """Cleanup when the pool is destroyed."""
        self.clear() 