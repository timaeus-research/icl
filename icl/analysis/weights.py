import torch
from devinfra.utils.tensors import ReturnTensor, convert_tensor
from torch import nn


class WeightsTrace:
    def __init__(self, num_chains: int, num_draws: int, model: nn.Module, device: str = 'cpu'):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.num_params = sum(p.numel() for _, p in model.named_parameters())
        self.init_weights = torch.cat([p.detach().view(-1) for _, p in model.named_parameters()])
        self.weights = torch.zeros((num_chains, num_draws, self.num_params), device=device)

    def update(self, chain: int, draw: int, model):
        i = 0
        for n, p in model.named_parameters():
            p_numel = p.numel()
            self.weights[chain, draw, i:i+p_numel] = p.detach().view(-1)
            i += p_numel

            # Check if nans or very large
            if torch.isnan(p).any() or torch.isinf(p).any():
                raise ValueError(f"NaNs/Infs in weights: {n}\n{p}")

    def __call__(self, chain: int, draw: int, model):
        self.update(chain=chain, draw=draw, model=model)

    def deltas(self, return_type: ReturnTensor = "np"):
        _deltas = self.weights - self.init_weights.view(1, 1, -1)
        return convert_tensor(_deltas, return_type)
