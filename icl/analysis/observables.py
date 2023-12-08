import warnings
from abc import ABC, abstractmethod
from pathlib import Path
from pprint import pp
from typing import (Any, Callable, Dict, Iterable, Literal, Tuple, Type,
                    TypeVar, Union)

import matplotlib.pyplot as plt
import numpy as torch
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
from tqdm import tqdm

import wandb
from icl.analysis.utils import get_sweep_configs
from icl.config import ICLConfig, get_config
from icl.train import Run

app = typer.Typer()

class ExpectationEstimator:
    def __init__(self, num_samples: int, observable_dim: int = 1, device="cpu"):
        self.num_samples = num_samples
        self.num_samples_seen = 0
        self.observable_dim = observable_dim

        # Unnormalized first and second moments
        self._first_moment = torch.zeros(observable_dim, dtype=torch.float32).to(device)
        self._second_moment = torch.zeros(observable_dim, dtype=torch.float32).to(device)

    def _update(self, chain: int, draw: int, indices: Union[slice, Any], observation: float):
        self._first_moment[indices] += observation
        self._second_moment[indices] += observation ** 2

    def iter_update(self, chain: int, draw: int, iterable: Iterable[torch.Tensor]):
        I = 0        

        for i, observation in enumerate(iterable):
            b = observation.shape[0]
            self._update(chain, draw, slice(I, I + b), observation)
            I += b
            yield observation

        self.increment(chain)

    def update(self, chain: int, draw: int, observation: float):
        self._update(chain, draw, ..., observation)
        self.increment()

    def increment(self):
        self.num_samples_seen += 1
    
    @property
    def first_moment(self):
        return self._first_moment / self.num_samples_seen
    
    @property
    def second_moment(self):
        return self._second_moment / self.num_samples_seen
    
    def estimate(self):
        return {
            "mean": self.first_moment,
            "std": torch.sqrt(self.second_moment - self.first_moment ** 2),
        }


class OnlineExpectationEstimatorWithTrace:
    """Evaluates an observable_fn on each draw and accumulates the results. Meant for calibration/diagnostics."""
    def __init__(self, num_chains: int, num_draws: int, observable_dim: int = 1, device="cpu"):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.observable_dim = observable_dim

        self.num_samples_seen = torch.zeros((num_chains,), dtype=torch.int32).to(device)
        self.first_moments = torch.zeros((num_chains, num_draws, observable_dim), dtype=torch.float32).to(device)
        self.second_moments = torch.zeros((num_chains, num_draws, observable_dim), dtype=torch.float32).to(device)

    def _update(self, chain: int, draw: int, indices: Union[slice, Any], observation: torch.Tensor):
        if draw == 0:
            self.first_moments[chain, draw, indices] = observation
            self.second_moments[chain, draw, indices] = observation ** 2
        else:
            self.first_moments[chain, draw, indices] = (
                draw / (draw + 1) * self.first_moments[chain, draw - 1, indices] + observation / (draw + 1)
            )
            self.second_moments[chain, draw, indices] = (
                draw / (draw + 1) * self.second_moments[chain, draw - 1, indices] + observation ** 2 / (draw + 1)
            )

    def iter_update(self, chain: int, draw: int, iterable: Iterable[torch.Tensor]):
        I = 0        

        for i, observation in enumerate(iterable):
            b = observation.shape[0]
            self._update(chain, draw, slice(I, I + b), observation)
            I += b
            yield observation

        self.increment(chain)
    
    def update(self, chain: int, draw: int, observation: float):
        self._update(chain, draw, ..., observation)
        self.increment(chain)

    def increment(self, chain: int):
        self.num_samples_seen[chain] += 1

    @property
    def least_num_samples_seen(self):
        _least_num_samples_seen = min(self.num_samples_seen)

        if _least_num_samples_seen == 0:
            warnings.warn("No samples seen.")

        return _least_num_samples_seen
    
    def first_moment_at(self, draw: int):
        if any(self.num_samples_seen <= draw):
            raise ValueError("Not enough samples seen.")

        return self.first_moments[:, draw, :].mean(dim=0)
    
    def second_moment_at(self, draw: int):
        if any(self.num_samples_seen <= draw):
            raise ValueError("Not enough samples seen.")

        return self.second_moments[:, draw, :].mean(dim=0)

    @property
    def first_moment(self):
        return self.first_moment_at(self.least_num_samples_seen-1)

    @property 
    def second_moment(self):
        return self.second_moment_at(self.least_num_samples_seen-1)
    
    def estimate(self):
        return {
            "trace": self.first_moments,
            **{f"chain-{i}/mean": self.first_moments[i, self.least_num_samples_seen-1] for i in range(self.num_chains)},
            **{f"chain-{i}/std": torch.sqrt(self.second_moments[i, self.least_num_samples_seen-1] - self.first_moments[i, self.num_samples_seen[i]-1] ** 2) for i in range(self.num_chains)},
            "mean": self.first_moment,
            "std": torch.sqrt(self.second_moment - self.first_moment ** 2),
        }

# class OnlineExpectationEstimator:
#     def __init__(self, num_samples: int, observable_dim: int = 1, device="cpu"):
#         self.num_samples = num_samples
#         self.num_samples_seen = 0
#         self.observable_dim = observable_dim
#         self.first_moment = torch.zeros(observable_dim, dtype=torch.float32).to(device)
#         self.second_moment = torch.zeros(observable_dim, dtype=torch.float32).to(device)

#     def update(self, chain: int, draw: int, observation: float):
#         self.first_moment = self.first_moment  + (observation - self.first_moment) / (self.num_samples_seen + 1)
#         self.second_moment = self.second_moment + (observation ** 2 - self.second_moment) / (self.num_samples_seen + 1)
#         self.increment()

#     def _update(self, chain: int, draw: int, indices: Tuple[int, int], observation: float):
#         self.first_moment[indices] = self.first_moment[indices]  + (observation - self.first_moment[indices]) / (self.num_samples_seen + 1)
#         self.second_moment[indices] = self.second_moment[indices] + (observation ** 2 - self.second_moment[indices]) / (self.num_samples_seen + 1)

#     def increment(self):
#         self.num_samples_seen += 1

#     def estimate(self):
#         return {
#             "mean": self.first_moment,
#             "std": torch.sqrt(self.second_moment - self.first_moment ** 2),
#         }
    

# class ExpectationEstimatorWithTrace:
#     def __init__(self, num_chains: int, num_draws: int, observable_dim: int = 1, device="cpu"):
#         self.num_chains = num_chains
#         self.num_draws = num_draws
#         self.observable_dim = observable_dim
#         self.draws = torch.zeros((num_chains, num_draws, observable_dim), dtype=torch.float32).to(device)
#         self.num_samples_seen = 0

#     def update(self, chain: int, draw: int, observation: float):
#         self.draws[chain, draw, :] = observation
#         self.increment()

#     def _update(self, chain: int, draw: int, indices: Tuple[int, int], observation: float):
#         self.draws[chain, draw, indices] = observation

#     def increment(self):
#         self.num_samples_seen += 1
    
#     def estimate(self):
#         return {
#             "trace": self.draws,
#             **{f"chain-{i}/mean": self.draws.mean(dim=1) for i in range(self.num_chains)},
#             **{f"chain-{i}/std": self.draws.std(dim=1) for i in range(self.num_chains)},
#             "mean": self.draws.mean().item(),
#             "std": self.draws.std().item(),
#             "max": self.draws.max().item(),
#             "min": self.draws.min().item(),
#         }
    
#     @property
#     def first_moment(self):
#         return self.draws[:self.num_samples_seen].mean()
    
#     @property
#     def second_moment(self):
#         return (self.draws[:self.num_samples_seen] ** 2).mean()
    
    

    

def get_estimator(num_chains: int, num_draws: int, observable_dim: int = 1, device="cpu", online: bool = False, include_trace: bool = False):
    if include_trace:
        if online:
            return OnlineExpectationEstimatorWithTrace(num_chains, num_draws, observable_dim, device)

        raise ValueError("Estimators wtih trace must be online.")
        # return ExpectationEstimatorWithTrace(num_chains, num_draws, observable_dim, device)
    elif online:
        raise ValueError("Online estimators must include trace.")
        # return OnlineExpectationEstimator(num_chains * num_draws, observable_dim, device) 
    
    return ExpectationEstimator(num_chains * num_draws, observable_dim, device)
