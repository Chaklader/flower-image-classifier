import torch

def get_device():
    """
    Determines and returns the most appropriate PyTorch device available.

    It checks for hardware acceleration in the following order of preference:
    1. Apple Metal Performance Shaders (MPS) for Apple Silicon GPUs.
    2. NVIDIA CUDA for NVIDIA GPUs.
    3. CPU as a fallback.

    Returns:
        torch.device: The selected device object ('mps', 'cuda', or 'cpu').
    """
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

# How to use the function:
# device = get_device()
# print(f"Using device: {device}")