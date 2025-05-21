import unittest
import torch
import platform
from vllm.platforms.metal.metal_memory import MetalMemoryManager
from vllm.platforms.metal.metal_memory_pool import MetalMemoryPool

class TestMetalMemoryPool(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Skip tests if not on Apple Silicon
        if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
            raise unittest.SkipTest("Test only runs on Apple Silicon")
        
        # Skip if MPS is not available
        if not torch.backends.mps.is_available():
            raise unittest.SkipTest("MPS backend is not available")
            
        cls.memory_manager = MetalMemoryManager()
        cls.memory_pool = MetalMemoryPool(cls.memory_manager)

    def setUp(self):
        # Clear memory before each test
        self.memory_pool.clear()

    def test_pool_allocation(self):
        """Test basic pool allocation."""
        size = 1000
        tensor = self.memory_pool.allocate(size)
        self.assertEqual(tensor.device.type, "mps")
        self.assertEqual(tensor.nelement(), size)
        self.assertEqual(tensor.dtype, torch.float32)

    def test_buffer_reuse(self):
        """Test buffer reuse from the pool."""
        size = 1000
        dtype = torch.float32
        
        # Allocate and free a buffer
        tensor1 = self.memory_pool.allocate(size, dtype)
        self.memory_pool.free(tensor1)
        
        # Allocate another buffer of the same size and dtype
        tensor2 = self.memory_pool.allocate(size, dtype)
        
        # Should reuse the same buffer
        self.assertEqual(tensor1.data_ptr(), tensor2.data_ptr())

    def test_different_sizes(self):
        """Test handling of different buffer sizes."""
        sizes = [1000, 2000, 3000]
        tensors = []
        
        # Allocate buffers of different sizes
        for size in sizes:
            tensor = self.memory_pool.allocate(size)
            self.assertEqual(tensor.nelement(), size)
            tensors.append(tensor)
            
        # Free all buffers
        for tensor in tensors:
            self.memory_pool.free(tensor)
            
        # Verify pool stats
        stats = self.memory_pool.get_pool_stats()
        self.assertEqual(stats["unique_sizes"], len(sizes))

    def test_pool_stats(self):
        """Test memory pool statistics."""
        size = 1000
        num_buffers = 5
        
        # Allocate and free multiple buffers
        tensors = [self.memory_pool.allocate(size) for _ in range(num_buffers)]
        for tensor in tensors:
            self.memory_pool.free(tensor)
            
        stats = self.memory_pool.get_pool_stats()
        self.assertEqual(stats["total_buffers"], num_buffers)
        self.assertEqual(stats["unique_sizes"], 1)
        self.assertEqual(stats["total_memory"], size * num_buffers)

    def test_clear_pool(self):
        """Test clearing the memory pool."""
        size = 1000
        num_buffers = 5
        
        # Allocate and free multiple buffers
        tensors = [self.memory_pool.allocate(size) for _ in range(num_buffers)]
        for tensor in tensors:
            self.memory_pool.free(tensor)
            
        # Clear the pool
        self.memory_pool.clear()
        
        # Verify pool is empty
        stats = self.memory_pool.get_pool_stats()
        self.assertEqual(stats["total_buffers"], 0)
        self.assertEqual(stats["total_memory"], 0)

if __name__ == '__main__':
    unittest.main() 