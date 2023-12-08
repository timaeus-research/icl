import itertools
import os
import warnings
from pathlib import Path
from pprint import pp
from typing import (Any, Callable, Dict, Generator, Iterable, Literal,
                    Optional, Tuple, Type, TypeVar, Union)

import devinfra
import devinterp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sentry_sdk
import torch
import typer
import yaml
from devinfra.evals import RepeatEvaluator
from devinfra.utils.device import get_default_device
from dotenv import load_dotenv
from pydantic import BaseModel
from torch import nn
from tqdm import tqdm

import wandb
from icl.analysis.estimators import get_estimator
from icl.analysis.utils import get_sweep_configs
from icl.config import ICLConfig, get_config
from icl.train import Run

app = typer.Typer()

def prepend_keys(d: Dict, prefix: str):
    return {f"{prefix}/{k}": v for k, v in d.items()}


class ExpectedBatchLossEstimator:
    def __init__(self, num_chains: int, num_draws: int, device="cpu", online=False, include_trace=False):
        self.estimator = get_estimator(num_chains, num_draws, 1, device, online=online, include_trace=include_trace)

    def estimate(self):
        return prepend_keys(self.estimator.estimate(), "batch-loss")
    
    def __call__(self, chain: int, draw: int, loss: torch.Tensor):
        self.estimator.update(chain, draw, loss)

    def reset(self):
        self.estimator.reset()
        
class ExpectedLossObservableEstimator:
    def __init__(self, num_chains: int, num_draws: int, loss_fn: Callable[[nn.Module], torch.Tensor], device="cpu", online=False, include_trace=False):
        self.estimator = get_estimator(num_chains, num_draws, 1, device, online=online, include_trace=include_trace)
        self.loss_fn = loss_fn

    def estimate(self):
        return prepend_keys(self.estimator.estimate(), "loss")

    def __call__(self, chain: int, draw: int, model: nn.Module):
        self.estimator.update(chain, draw, self.loss_fn(model))

    def reset(self):
        self.estimator.reset()

Temperature = Union[Literal['adaptive'], float]


class LikelihoodMetricsEstimator:
    """
    Estimate the WBIC and local learning coefficient (LLC).
    """
    def __init__(self, num_chains: int, num_draws: int, dataset_size: int, temperature: Temperature = 'adaptive', loss_fn: Optional[Callable[[nn.Module], torch.Tensor]]=None, device="cpu", online=False, include_trace=False):
        self.loss_fn = loss_fn
        self.expected_loss_estimator = get_estimator(num_chains, num_draws, 1, device=device, online=online, include_trace=include_trace)
        self.num_chains = num_chains
        self.dataset_size = dataset_size
        self.temperature = temperature if temperature != 'adaptive' else 1. / np.log(dataset_size)
        self.init_loss = torch.zeros(1, dtype=torch.float32).to(device)

    def estimate(self):
        loss_avg = self.expected_loss_estimator.first_moment
        loss_std = torch.sqrt(self.expected_loss_estimator.second_moment - loss_avg ** 2)

        wbic = loss_avg * self.dataset_size
        wbic_std = loss_std * self.dataset_size

        llc = (wbic - self.init_loss * self.dataset_size) / self.temperature
        llc_std = wbic_std / self.temperature 

        return {
            "loss/mean": loss_avg,
            "loss/std": loss_std,
            "wbic/mean": wbic,
            "wbic/std": wbic_std,
            "llc/mean": llc,
            "llc/std": llc_std
        }

    def update(self, chain: int, draw: int, loss: torch.Tensor):
        self.expected_loss_estimator.update(chain, draw, loss)

        if draw == 0:
            self.init_loss += loss / self.num_chains   # Will yield wrong answers for llc if you call estimate before all chains have been updated


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
            "singular_fluctuation/mean": singular_fluctuation,
            "singular_fluctuation/std": singular_fluctuation_std
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
    def __init__(self, num_chains: int, num_draws: int, dataset_size: int, losses_generator: Callable[[nn.Module], Generator[torch.Tensor, None, None]], temperature: Temperature = 'adaptive', device="cpu", online=False, include_trace=False):
        self.likelihood_metrics_estimator = LikelihoodMetricsEstimator(num_chains, num_draws, dataset_size, temperature, device=device, online=online, include_trace=include_trace)
        self.singular_fluctuation_estimator = SingularFluctuationEstimator(num_chains, num_draws, dataset_size, losses_generator, temperature, device=device, online=online, include_trace=include_trace)

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

    def __call__(self, chain: int, draw: int, model: nn.Module):
        self.update(chain, draw, model)

    def reset(self):
        self.likelihood_metrics_estimator.expected_loss_estimator.reset()
        self.singular_fluctuation_estimator.expected_losses_estimator.reset()