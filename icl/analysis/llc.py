import itertools
import os
import warnings
from pathlib import Path
from pprint import pp
from typing import (Callable, Dict, Iterable, List, Literal, Optional, Tuple,
                    TypeVar, Union)

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
from devinterp.optim.sgld import SGLD
from dotenv import load_dotenv
from pydantic import BaseModel
from torch import nn
from torch.nn import functional as F
from tqdm import tqdm

import wandb
from icl.analysis.sample import estimate_slt_observables
from icl.analysis.utils import get_sweep_configs
from icl.config import ICLConfig, get_config
from icl.train import Run

app = typer.Typer()


def make_slt_evals(
    dataset: torch.utils.data.Dataset,
    loader: torch.utils.data.DataLoader,
    lr: float = 1e-4,
    noise_level: float = 1.0,
    weight_decay: float = 0.0,
    elasticity: float = 1.0,
    num_draws: int = 10,
    num_chains: int = 25,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    cores: int = 1,
    device: str = "cpu",
    callbacks: List[Callable] = [],
    num_samples: Optional[int] = None
):
    def eval_rlct(model: nn.Module):
        optimizer_kwargs = dict(
            lr=lr,
            noise_level=noise_level,
            weight_decay=weight_decay,
            elasticity=elasticity,
            temperature="adaptive",
            num_samples=num_samples or len(dataset),
        )
        return estimate_slt_observables(
            model,
            loader,
            F.mse_loss,
            SGLD,
            optimizer_kwargs,
            num_draws=num_draws,
            num_chains=num_chains,
            num_burnin_steps=num_burnin_steps,
            num_steps_bw_draws=num_steps_bw_draws,
            cores=cores,
            device=device,
            callbacks=callbacks
        )

    return eval_rlct
