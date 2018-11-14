import torch
from config import DeviceConfig

def tensor(x):
    if isinstance(x, torch.Tensor):
        return x
    
    return torch.tensor(x, device=DeviceConfig.device, dtype=torch.float32)