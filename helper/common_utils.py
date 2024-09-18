import torch


def set_device_for_gpu():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def print_device_info(device, args):
    gpu_type = {
        'mps': 'Apple GPU',
        'cuda': 'NVIDIA GPU',
        'cpu': 'CPU'
    }.get(device.type, 'Unknown')

    if device.type == 'cpu' and args.gpu:
        print(f"*** GPU is unavailable, using {gpu_type} ...\n")
    else:
        action = "using" if args.gpu else "training model using"
        print(f"*** {action} {gpu_type} ...\n")

