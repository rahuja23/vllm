import torch
import platform

print("Platform:", platform.system(), platform.machine())
print("PyTorch version:", torch.__version__)

# Check for Metal (MPS) support (Apple Silicon GPU)
if hasattr(torch.backends, "mps"):
    print("MPS built:", torch.backends.mps.is_built())
    print("MPS available:", torch.backends.mps.is_available())
    if torch.backends.mps.is_available():
        x = torch.ones(1, device="mps")
        print("MPS device tensor created:", x)
else:
    print("MPS backend not present in this PyTorch build.")

# Check for CUDA (NVIDIA GPU) support
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("CUDA device count:", torch.cuda.device_count())
    print("CUDA device name:", torch.cuda.get_device_name(0))