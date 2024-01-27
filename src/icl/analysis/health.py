import torch
from torch import nn


class ChainHealthException(Exception):
    def __init__(self, value, message="Value is too high"):
        self.value = value
        self.message = message
        super().__init__(self.message)


class HealthCheck:
    def check_param(self, n: str, p: torch.Tensor):
        # Check if nans or very large
        if torch.isnan(p).any() or torch.isinf(p).any():
            raise ChainHealthException(p, f"NaNs/Infs in weights {n}")
        
    def __call__(self, model: nn.Module):
        for n, p in model.named_parameters():
            self.check_param(n, p)

