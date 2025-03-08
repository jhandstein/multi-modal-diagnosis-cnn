import os
import subprocess
import torch
from lightning.pytorch.accelerators import find_usable_cuda_devices


def allocated_free_gpus(num_gpus, max_usage_ratio: float = 0.8) -> list[int]:
    """Return a list of GPUs with least memory usage.
    
    Args:
        num_gpus (int): Number of GPUs requested
        max_usage_ratio (float): Maximum acceptable memory usage ratio (0.0 to 1.0)
        
    Returns:
        list[int]: Indices of least used GPUs
    """
    if not torch.cuda.is_available():
        return []

    # Run nvidia-smi to get memory usage across all processes
    try:
        cmd = "nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,nounits,noheader"
        output = subprocess.check_output(cmd.split(), universal_newlines=True)
    except subprocess.CalledProcessError:
        print("Failed to run nvidia-smi")
        return []

    # Parse nvidia-smi output
    memory_usage = []
    for line in output.strip().split('\n'):
        index, used, total = map(float, line.split(','))
        usage_ratio = used / total
        memory_usage.append((int(index), usage_ratio))

    # print(f"Memory usage ratios: {[(i, f'{ratio:.2%}') for i, ratio in memory_usage]}")
    
    # Filter GPUs below usage threshold
    available_gpus = [(i, ratio) for i, ratio in memory_usage if ratio < max_usage_ratio]
    if len(available_gpus) < num_gpus:
        print(f"Not enough GPUs available below {max_usage_ratio:.0%} usage threshold")
        return []

    # Sort by memory usage and return the indices of least used GPUs
    sorted_gpus = sorted(available_gpus, key=lambda x: x[1])
    print(f"Available GPUs sorted by usage: {[(i, f'{ratio:.2%}') for i, ratio in sorted_gpus]}")
    
    print(f"Selected GPUs: {[gpu[0] for gpu in sorted_gpus]}")
    free_gpus = [gpu[0] for gpu in sorted_gpus[:num_gpus]]
    return free_gpus

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
def calculate_model_size(model):
    """Calculate the size of a model in MB"""
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return f"{size_all_mb:.2f} MB"


def calculate_tensor_size(tensor: torch.Tensor, batch_size: int = 1):
    """Calculate the size of a tensor in MB"""
    return batch_size * tensor.nelement() * tensor.element_size() / 1024**2
