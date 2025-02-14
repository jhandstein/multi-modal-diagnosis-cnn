import os
import torch
from lightning.pytorch.accelerators import find_usable_cuda_devices


def check_cuda(test_tensor_creation: bool = False, custom_device_call: bool = False):
    """Check if CUDA is available and print some information."""

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)} is available.")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        print(f"Current device index: {torch.cuda.current_device()}")

        # Try to create a tensor on GPU
        if test_tensor_creation:
            try:
                test_tensor = torch.tensor([1.0]).cuda()
                print(f"Successfully created tensor on {test_tensor.device}")
            except RuntimeError as e:
                print(f"Failed to create tensor on GPU: {e}")

        # List all visible devices
        if custom_device_call:
            for i in range(torch.cuda.device_count()):
                print(f"Device {i}: {torch.cuda.get_device_name(i)}")
                print(f"Device {i} capability: {torch.cuda.get_device_capability(i)}")

    else:
        print("No GPU available. Training will run on CPU.")

    # Lightning helper function
    find_usable_cuda_devices()  # does not work / only shows "cpu"


def print_cuda_version():
    """Check some CUDA parameters."""

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDNN version: {torch.backends.cudnn.version()}")
    print(f"CUDA available: {torch.cuda.is_available()}")


def print_nccl_vars():
    """Check some NCCL environment variables."""

    print(f"NCCL_DEBUG: {os.environ.get('NCCL_DEBUG', 'Not set')}")
    print(f"NCCL_IB_DISABLE: {os.environ.get('NCCL_IB_DISABLE', 'Not set')}")
    print(f"NCCL_P2P_DISABLE: {os.environ.get('NCCL_P2P_DISABLE', 'Not set')}")

# Add after model creation
def print_model_size(model):
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    print(f'Model Size: {size_all_mb:.2f} MB')

