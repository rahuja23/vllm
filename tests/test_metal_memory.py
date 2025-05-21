import unittest
import torch
import platform
from vllm.platforms.metal.metal_memory import MetalMemoryManager

class TestMetalMemory(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Skip tests if not on Apple Silicon
        if not (platform.system() == "Darwin" and platform.machine() == "arm64"):
            raise unittest.SkipTest("Test only runs on Apple Silicon")
        
        # Skip if MPS is not available
        if not torch.backends.mps.is_available():
            raise unittest.SkipTest("MPS backend is not available")
            
        cls.memory_manager = MetalMemoryManager()

    def setUp(self):
        # Clear memory before each test
        self.memory_manager.clear()

    def test_allocation(self):
        """Test basic memory allocation."""
        size = 1000
        tensor = self.memory_manager.allocate(size)
        self.assertEqual(tensor.device.type, "mps")
        self.assertEqual(tensor.nelement(), size)
        self.assertEqual(tensor.dtype, torch.float32)

    def test_dtype_allocation(self):
        """Test allocation with different dtypes."""
        size = 1000
        dtypes = [torch.float32, torch.float16, torch.bfloat16]
        
        for dtype in dtypes:
            tensor = self.memory_manager.allocate(size, dtype=dtype)
            self.assertEqual(tensor.dtype, dtype)
            self.memory_manager.free(tensor)

    def test_memory_tracking(self):
        """Test memory usage tracking."""
        size = 1000
        tensor = self.memory_manager.allocate(size)
        total_allocated, max_memory = self.memory_manager.get_memory_stats()
        
        self.assertGreater(total_allocated, 0)
        self.assertGreater(max_memory, 0)
        self.assertLessEqual(total_allocated, max_memory)

    def test_free_memory(self):
        """Test memory deallocation."""
        size = 1000
        tensor = self.memory_manager.allocate(size)
        initial_total, _ = self.memory_manager.get_memory_stats()
        
        self.memory_manager.free(tensor)
        final_total, _ = self.memory_manager.get_memory_stats()
        
        self.assertLess(final_total, initial_total)

    def test_clear_memory(self):
        """Test clearing all allocated memory."""
        sizes = [1000, 2000, 3000]
        tensors = [self.memory_manager.allocate(size) for size in sizes]
        
        self.memory_manager.clear()
        total_allocated, _ = self.memory_manager.get_memory_stats()
        
        self.assertEqual(total_allocated, 0)

    def test_memory_limits(self):
        """Test memory allocation limits."""
        # Get current memory stats
        _, max_memory = self.memory_manager.get_memory_stats()
        
        # Calculate a size that would definitely exceed memory
        element_size = torch.tensor([], dtype=torch.float32).element_size()
        large_size = (max_memory // element_size) + 1
        
        # Try to allocate a buffer that's too large
        with self.assertRaises(RuntimeError) as context:
            self.memory_manager.allocate(large_size)
        
        self.assertIn("exceed device memory capacity", str(context.exception))

if __name__ == '__main__':
    unittest.main() 