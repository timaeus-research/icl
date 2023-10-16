import inspect
import itertools
import os
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from logging import Logger
from pathlib import Path
from pprint import pp
from typing import (Any, Callable, Dict, Iterable, List, Literal, Optional,
                    Tuple, Type, Union)

import devinfra
import devinterp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sentry_sdk
import torch
import yaml
from devinfra.evals import Criterion
from devinterp.optim.sgld import SGLD
from torch import nn
from torch.multiprocessing import (Pool, cpu_count, get_context,
                                   set_start_method)
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typer import Typer

from icl.analysis.utils import get_unique_run
from icl.config import ICLConfig, get_config
from icl.train import Run


def get_weights(model, paths):
    for path in paths:
        full_path = path.split(".")
        layer = model

        for p in full_path:
            layer = getattr(layer, p)

        yield layer.weight.view((-1,))

        if layer.bias is not None:
            yield layer.bias.view((-1,))
 
 
class LLCEstimator:
    def __init__(self, num_chains: int, num_draws: int, n: int, device="cpu"):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.n = torch.tensor(n, dtype=torch.float32).to(device)
        self.losses = np.zeros((num_chains, num_draws), dtype=np.float32)
        self.llc_per_chain = torch.zeros(num_chains, dtype=torch.float32).to(device)
        self.llc_mean = torch.tensor(0., dtype=torch.float32).to(device)
        self.llc_std = torch.tensor(0., dtype=torch.float32).to(device)

    def update(self, chain: int, draw: int, loss: float):
        self.losses[chain, draw] = loss 

    @property
    def init_loss(self):
        return self.losses[0, 0]

    def finalize(self):
        avg_losses = self.losses.mean(axis=1)
        self.llc_per_chain = (self.n / self.n.log()).detach().cpu().numpy() * (avg_losses - self.init_loss)
        self.llc_mean = self.llc_per_chain.mean()
        self.llc_std = self.llc_per_chain.std()
        
    def sample(self):
        return {
            "llc/mean": self.llc_mean.item(),
            "llc/std": self.llc_std.item(),
            **{f"llc-chain/{i}": self.llc_per_chain[i].item() for i in range(self.num_chains)},
            "loss/trace": self.losses,
        }
    
    def __call__(self, chain: int, draw: int, loss: float):
        self.update(chain, draw, loss)


class OnlineLLCEstimator:
    def __init__(self, num_chains: int, num_draws: int, n: int, device="cpu"):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.n = torch.tensor(n, dtype=torch.float32).to(device)
        self.losses = np.zeros((num_chains, num_draws), dtype=torch.float32)
        self.llcs = np.zeros((num_chains, num_draws), dtype=torch.float32)
        self.llc_per_chain = torch.zeros(num_chains, dtype=torch.float32).to(device)
        self.llc_means = torch.tensor(num_chains, dtype=torch.float32).to(device)
        self.llc_stds = torch.tensor(num_chains, dtype=torch.float32).to(device)

    def share_memory_(self):
        self.n.share_memory_()
        self.llc_per_chain.share_memory_()
        self.llc_means.share_memory_()
        self.llc_stds.share_memory_()

        return self

    def update(self, chain: int, draw: int, loss: float):
        self.losses[chain, draw] = loss 

        if draw == 0:  # TODO: We can probably drop this and it still works (but harder to read)
            self.llcs[0, draw] = 0.
        else:
            t = draw + 1
            prev_llc = self.llcs[chain, draw - 1]

            with torch.no_grad():
                self.llcs[chain, draw] = (1 / t) * (
                    (t - 1) * prev_llc + (self.n / self.n.log()) * (loss - self.init_loss)
                )

    @property
    def init_loss(self):
        return self.losses[0, 0]

    def finalize(self):
        self.llc_means = self.llcs.mean(axis=0)
        self.llc_stds = self.llcs.std(axis=0)

    def sample(self):
        return {
            "llc/means": self.llc_means.cpu().numpy(),
            "llc/stds": self.llc_stds.cpu().numpy(),
            "llc/trace": self.llcs.cpu().numpy(),
            "loss/trace": self.losses.cpu().numpy(),
        }
    
    def __call__(self, chain: int, draw: int, loss: float):
        self.update(chain, draw, loss)


def call_with(func: Callable, **kwargs):
    """Check the func annotation and call with only the necessary kwargs."""
    sig = inspect.signature(func)
    
    # Filter out the kwargs that are not in the function's signature
    filtered_kwargs = {k: v for k, v in kwargs.items() if k in sig.parameters}
    
    # Call the function with the filtered kwargs
    return func(**filtered_kwargs)


def sample_single_chain(
    ref_model: nn.Module,
    loader: DataLoader,
    criterion: Callable,
    num_draws=100,
    num_burnin_steps=0,
    num_steps_bw_draws=1,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    optimizer_kwargs: Optional[Dict] = None,
    chain: int = 0,
    seed: Optional[int] = None,
    verbose=True,
    device: Union[str, torch.device] = torch.device("cpu"),
    callbacks: List[Callable] = [],
):
    # Initialize new model and optimizer for this chain
    model = deepcopy(ref_model).to(device)

    optimizer_kwargs = optimizer_kwargs or {}
    optimizer = sampling_method(model.parameters(), **optimizer_kwargs)

    if seed is not None:
        torch.manual_seed(seed)

    num_steps = num_draws * num_steps_bw_draws + num_burnin_steps
    model.train()

    for i, (xs, ys) in  tqdm(zip(range(num_steps), itertools.cycle(loader)), desc=f"Chain {chain}", total=num_steps, disable=not verbose):
        optimizer.zero_grad()
        xs, ys = xs.to(device), ys.to(device)
        y_preds = model(xs, ys)
        loss = criterion(y_preds, ys)

        loss.backward()
        optimizer.step()

        if i >= num_burnin_steps and (i - num_burnin_steps) % num_steps_bw_draws == 0:
            draw = (i - num_burnin_steps) // num_steps_bw_draws
            loss = loss.item()

            with torch.no_grad():
                for callback in callbacks:
                    call_with(callback, **locals())  # Cursed but we'll fix it later


def _sample_single_chain(kwargs):
    return sample_single_chain(**kwargs)


def sample(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: Callable,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    optimizer_kwargs: Optional[Dict[str, Union[float, Literal["adaptive"]]]] = None,
    num_draws: int = 100,
    num_chains: int = 10,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    cores: int = 1,
    seed: Optional[Union[int, List[int]]] = None,
    device: torch.device = torch.device("cpu"),
    verbose: bool = True,
    callbacks: List[Callable] = [],
):
    """
    Sample model weights using a given optimizer, supporting multiple chains.

    Parameters:
        model (torch.nn.Module): The neural network model.
        step (Literal['sgld']): The name of the optimizer to use to step.
        loader (DataLoader): DataLoader for input data.
        criterion (torch.nn.Module): Loss function.
        num_draws (int): Number of samples to draw.
        num_chains (int): Number of chains to run.
        num_burnin_steps (int): Number of burn-in steps before sampling.
        num_steps_bw_draws (int): Number of steps between each draw.
        cores (Optional[int]): Number of cores for parallel execution.
        seed (Optional[Union[int, List[int]]]): Random seed(s) for sampling.
        optimizer_kwargs (Optional[Dict[str, Union[float, Literal['adaptive']]]]): Keyword arguments for the optimizer.
    """
    if cores is None:
        cores = min(4, cpu_count())

    if seed is not None:
        if isinstance(seed, int):
            seeds = [seed + i for i in range(num_chains)]
        elif len(seed) != num_chains:
            raise ValueError("Length of seed list must match number of chains")
        else:
            seeds = seed
    else:
        seeds = [None] * num_chains

    def get_args(i):
        return dict(
            chain=i,
            seed=seeds[i],
            ref_model=model,
            loader=loader,
            criterion=criterion,
            num_draws=num_draws,
            num_burnin_steps=num_burnin_steps,
            num_steps_bw_draws=num_steps_bw_draws,
            sampling_method=sampling_method,
            optimizer_kwargs=optimizer_kwargs,
            device=device,
            verbose=verbose,
            callbacks=callbacks,
        )

    results = []

    if cores > 1:
        ctx = get_context("spawn")
        with ctx.Pool(cores) as pool:
            results = pool.map(_sample_single_chain, [get_args(i) for i in range(num_chains)])
    else:
        for i in range(num_chains):
            results.append(_sample_single_chain(get_args(i)))

    for callback in callbacks:
        if hasattr(callback, "finalize"):
            callback.finalize()



def estimate_slt_observables(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: Criterion,
    sampling_method: Type[torch.optim.Optimizer] = SGLD,
    optimizer_kwargs: Optional[Dict[str, Union[float, Literal["adaptive"]]]] = None,
    num_draws: int = 100,
    num_chains: int = 10,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    cores: int = 1,
    seed: Optional[Union[int, List[int]]] = None,
    verbose: bool = True,
    device: str = "cpu",
    callbacks: List[Callable] = [],
    online: bool = False,
):

    if online:
        callbacks.append(OnlineLLCEstimator(num_chains, num_draws, len(loader.dataset), device=device))
    else:
        callbacks.append(LLCEstimator(num_chains, num_draws, len(loader.dataset), device=device))

    sample(
        model=model,
        loader=loader,
        criterion=criterion,
        sampling_method=sampling_method,
        optimizer_kwargs=optimizer_kwargs,
        num_draws=num_draws,
        num_chains=num_chains,
        num_burnin_steps=num_burnin_steps,
        num_steps_bw_draws=num_steps_bw_draws,
        cores=cores,
        seed=seed,
        verbose=verbose,
        device=device,
        callbacks=callbacks,
    )

    results = {}

    for callback in callbacks:
        if hasattr(callback, "sample"):
            results.update(callback.sample())

    return results

