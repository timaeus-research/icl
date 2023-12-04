import itertools
import os
import warnings
from pathlib import Path
from pprint import pp
from typing import Callable, Dict, Iterable, Literal, Tuple, TypeVar, Union

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
from icl.analysis.utils import get_sweep_configs
from icl.config import ICLConfig, get_config
from icl.train import Run

app = typer.Typer()


class FunctionalVarianceEstimator:
    def __init__(self, num_chains: int, num_draws: int, n: int, loss_fn: Callable[nn.Module], beta: float = 1., device="cpu"):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.n = torch.tensor(n, dtype=torch.float32).to(device)
        self.losses = torch.zeros(n, dtype=torch.float32)
        self.losses_sq = torch.zeros(n, dtype=torch.float32)
        self.loss_fn = loss_fn
        self.beta = beta

    @property
    def total_num_draws(self):
        return self.num_chains * self.num_draws 

    def update(self, chain: int, draw: int, model: nn.Module):
        self.losses[chain, draw, :] = self.loss_fn(model)

    def finalize(self):
        self.losses = self.losses / self.total_num_draws
        self.losses_sq = self.losses_sq / self.total_num_draws

    def sample(self):
        functional_variance = self.losses ** 2 - self.losses_sq
        return {
            "functional_variance": functional_variance,
            "singular_fluctuation": functional_variance * self.beta / 2
        }
    
    def __call__(self, chain: int, draw: int, loss: float):
        self.update(chain, draw, loss)



class SingularFluctuationEstimator:
    pass