from typing import NamedTuple
import torch

class Config(NamedTuple):
    state_normalizer = lambda self, x: x
    reward_normalizer = lambda self, x: x
    optimization_epochs: int = 10
    ppo_ratio_clip: int = 0.2
    rollout_length: int = 2048
    entropy_weight: float = 0.01
    gradient_clip: int = 5
    num_workers: int = 1
    mini_batch_size: int = 128
    use_gae: bool = True
    gae_tau: float = 0.95
        
    discount: float = 0.99
    log_interval: int = 2048
    max_steps: float = 1e5
    episode_count: int = 250
    hidden_size: int = 512
    adam_learning_rate: float = 3e-4
    adam_epsilon: float = 1e-5
        
class DeviceConfig:
    DEVICE = torch.device('cpu')