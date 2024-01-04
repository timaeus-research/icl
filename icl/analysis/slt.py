import warnings
from typing import Callable, Dict, Generator, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
import typer
from torch import nn

from icl.analysis.estimators import get_estimator
from icl.experiments.utils import flatten_and_process

app = typer.Typer()

def prepend_keys(d: Dict, prefix: str):
    return {f"{prefix}/{k}": v for k, v in d.items()}


class ExpectedBatchLossEstimator:
    def __init__(self, num_chains: int, num_draws: int, loss_dim: int = 1, device="cpu", online=False, include_trace=False):
        self.online = online
        self.estimator = get_estimator(num_chains, num_draws, loss_dim, device, online=online, include_trace=include_trace)

    def estimate(self):
        return prepend_keys(self.estimator.estimate(), "batch-loss")
    
    def estimates(self):
        if not self.online:
            raise NotImplementedError("Cannot get estimates for offline estimator.")

        return self.estimator.estimates()
    
    def __call__(self, chain: int, draw: int, loss: torch.Tensor):
        self.estimator.update(chain, draw, loss)

    def reset(self):
        self.estimator.reset()

        
class ExpectedLossObservableEstimator:
    def __init__(self, num_chains: int, num_draws: int, loss_fn: Callable[[nn.Module], torch.Tensor], loss_dim: int = 1, device="cpu", online=False, include_trace=False):
        self.online = online
        self.estimator = get_estimator(num_chains, num_draws, loss_dim, device, online=online, include_trace=include_trace)
        self.loss_fn = loss_fn

    def estimate(self):
        return prepend_keys(self.estimator.estimate(), "loss")

    def estimates(self):
        if not self.online:
            raise NotImplementedError("Cannot get estimates for offline estimator.")

        return self.estimator.estimates()
    
    def __call__(self, chain: int, draw: int, model: nn.Module):
        self.estimator.update(chain, draw, self.loss_fn(model))

    def reset(self):
        self.estimator.reset()

Temperature = Union[Literal['adaptive'], float]


class LikelihoodMetricsEstimator:
    """
    Estimate the WBIC and local learning coefficient (LLC).
    """
    def __init__(self, num_chains: int, num_draws: int, dataset_size: int, init_loss: torch.Tensor, temperature: Temperature = 'adaptive', loss_fn: Optional[Callable[[nn.Module], torch.Tensor]]=None, device="cpu", online=False, include_trace=False, log_fn = False):
        self.loss_fn = loss_fn
        self.loss_dim = init_loss.shape[0] if len(init_loss.shape) > 0 else 1
        self.expected_loss_estimator = get_estimator(num_chains, num_draws, self.loss_dim, device=device, online=online, include_trace=include_trace)
        self.num_chains = num_chains
        self.dataset_size = dataset_size
        self.temperature = temperature if temperature != 'adaptive' else 1. / np.log(dataset_size)
        self.init_loss = torch.Tensor(init_loss).to(device)

        self.online = online
        self.log_fn = log_fn
        self.least_num_samples_seen = 0
       
    @staticmethod
    def _estimate(first_moment, second_moment, init_loss, dataset_size, temperature):
        loss_avg = first_moment
        loss_std = torch.sqrt(second_moment - loss_avg ** 2)

        wbic = loss_avg * dataset_size
        wbic_std = loss_std * dataset_size

        llc = (wbic - init_loss * dataset_size) / temperature
        llc_std = wbic_std / temperature 

        return {
            "loss/mean": loss_avg.detach(),
            "loss/std": loss_std.detach(),
            "wbic/mean": wbic.detach(),
            "wbic/std": wbic_std.detach(),
            "llc/mean": llc.detach(),
            "llc/std": llc_std.detach()
        }

    def estimate(self):
        return self._estimate(self.expected_loss_estimator.first_moment, self.expected_loss_estimator.second_moment, self.init_loss, self.dataset_size, self.temperature)

    def estimates(self):
        if not self.online:
            raise NotImplementedError("Cannot get estimates for offline estimator.")
    
        _estimates = []

        for chain in range(self.num_chains):
            for draw in range(self.expected_loss_estimator.num_draws):  
                first_moment = self.expected_loss_estimator.first_moments[chain, draw]
                second_moment = self.expected_loss_estimator.second_moments[chain, draw]                
                _estimate = self._estimate(first_moment, second_moment, self.init_loss, self.dataset_size, self.temperature)
                _estimate = flatten_and_process(_estimate)
                _estimate['chain'] = chain
                _estimate['draw'] = draw
                _estimates.append(_estimate)
        
        return pd.DataFrame(_estimates)

    def update(self, chain: int, draw: int, loss: torch.Tensor):
        self.expected_loss_estimator.update(chain, draw, loss)

    def __call__(self, chain: int, draw: int, loss: torch.Tensor, model: nn.Module):
        if self.loss_fn is None:
            self.update(chain, draw, loss)
        else:
            _loss = self.loss_fn(model)
            total_loss = torch.zeros(1, dtype=torch.float32).to(model.device)

            if isinstance(_loss, Generator):
                n = 0

                for i, _l in enumerate(_loss):
                    total_loss += _l.sum() 
                    n += i

                self.update(chain, draw, total_loss / n)
            else:
                self.update(chain, draw, self.loss_fn(model))

        if self.online:
            new_least_num_samples_seen = self.expected_loss_estimator.least_num_samples_seen 

            if new_least_num_samples_seen > self.least_num_samples_seen:
                self.least_num_samples_seen = new_least_num_samples_seen

                if self.log_fn is not None and new_least_num_samples_seen % 10 == 0:
                    self.log_fn(self.estimate(), step=self.least_num_samples_seen)
            


    def reset(self):
        self.expected_loss_estimator.reset()

class SingularFluctuationEstimator:
    """
    Estimate singular fluctuation based on individual sample losses.
    """
    def __init__(self, num_chains: int, num_draws: int, dataset_size: int, losses_generator: Callable[[nn.Module], Generator[torch.Tensor, None, None]], temperature: Temperature = 'adaptive', device="cpu", online=False, include_trace=False):
        if online:
            warnings.warn("Online singular fluctuation estimation requires O(2n*T*C) memory, where n is the number of samples, T is the number of draws per chain, and C is the number of chains. This is not recommended for large datasets (n > 1000)")

        self.expected_losses_estimator = get_estimator(num_chains, num_draws, dataset_size, device, online=online, include_trace=include_trace)
        self.losses_generator = losses_generator
        self.temperature = temperature if temperature != 'adaptive' else 1. / np.log(dataset_size)
        self.dataset_size = dataset_size

    def estimate(self):
        losses_first_moment = self.expected_losses_estimator.first_moment
        losses_second_moment = self.expected_losses_estimator.second_moment
        losses_variances = losses_second_moment - losses_first_moment ** 2

        functional_variance = losses_variances.mean()
        functional_variance_std = losses_variances.std()

        singular_fluctuation = self.temperature * functional_variance / 2
        singular_fluctuation_std = self.temperature * functional_variance_std / 2

        return {
            "singular_fluctuation/mean": singular_fluctuation.detach(),
            "singular_fluctuation/std": singular_fluctuation_std.detach()
        }

    def iter_update(self, chain: int, draw: int, model: nn.Module):
        yield from self.expected_losses_estimator.iter_update(chain, draw, self.losses_generator(model))

    def __call__(self, chain: int, draw: int, model: nn.Module):
        for _ in self.iter_update(chain, draw, model):
            pass

    def reset(self):
        self.expected_losses_estimator.reset()


class SLTObservablesEstimator:
    """
    Estimate the WBIC, LLC, and singular fluctuation. 
    """
    def __init__(self, num_chains: int, num_draws: int, dataset_size: int, losses_generator: Callable[[nn.Module], Generator[torch.Tensor, None, None]], init_loss: torch.Tensor, temperature: Temperature = 'adaptive', device="cpu", online=False, include_trace=False, log_fn=None):
        self.likelihood_metrics_estimator = LikelihoodMetricsEstimator(num_chains, num_draws, dataset_size, temperature=temperature, init_loss=init_loss, device=device, online=online, include_trace=include_trace)
        self.singular_fluctuation_estimator = SingularFluctuationEstimator(num_chains, num_draws, dataset_size, losses_generator, temperature=temperature, device=device, online=online, include_trace=include_trace)
        
        self.online = online
        self.log_fn = log_fn
        self.least_num_samples_seen = 0

    def estimate(self):
        return {
            **self.likelihood_metrics_estimator.estimate(),
            **self.singular_fluctuation_estimator.estimate()
        }

    @property
    def dataset_size(self):
        return self.likelihood_metrics_estimator.dataset_size
    
    def update(self, chain: int, draw: int, model: nn.Module):
        total_loss = torch.zeros(1, dtype=torch.float32).to(model.device)

        for batch_losses in self.singular_fluctuation_estimator.iter_update(chain, draw, model):
            total_loss += batch_losses.sum()

        self.likelihood_metrics_estimator.update(chain, draw, total_loss / self.dataset_size)

        if self.online:
            new_least_num_samples_seen = self.likelihood_metrics_estimator.expected_loss_estimator.least_num_samples_seen 

            if new_least_num_samples_seen > self.least_num_samples_seen:
                self.least_num_samples_seen = new_least_num_samples_seen

                if self.log_fn is not None and new_least_num_samples_seen % 10 == 0:
                    self.log_fn(self.estimate(), step=self.least_num_samples_seen)

    def __call__(self, chain: int, draw: int, model: nn.Module):
        self.update(chain, draw, model)

    def reset(self):
        self.likelihood_metrics_estimator.expected_loss_estimator.reset()
        self.singular_fluctuation_estimator.expected_losses_estimator.reset()

    @property
    def init_loss(self):
        return self.likelihood_metrics_estimator.init_loss
    
    @init_loss.setter
    def init_loss(self, value):
        self.likelihood_metrics_estimator.init_loss = value