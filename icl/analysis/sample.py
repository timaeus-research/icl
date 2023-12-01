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
from devinterp.optim.sgnht import SGNHT
from torch import nn
from torch.multiprocessing import (Pool, cpu_count, get_context,
                                   set_start_method)
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typer import Typer

from icl.analysis.llc import (LLCEstimator, ObservedOnlineLLCEstimator,
                              OnlineLLCEstimator)
from icl.analysis.utils import get_unique_run
from icl.config import ICLConfig, get_config
from icl.train import Run


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
    online: Union[bool, Literal['observed']] = False,
):
    num_samples: int = optimizer_kwargs["num_samples"]

    if online == "observed":
        llc_estimator = ObservedOnlineLLCEstimator(num_chains, num_draws, num_samples, device=device)
    elif online:
        llc_estimator = OnlineLLCEstimator(num_chains, num_draws, num_samples, device=device)
    else:
        llc_estimator = LLCEstimator(num_chains, num_draws, num_samples, device=device)

    callbacks = [llc_estimator, *callbacks]

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
        callbacks=callbacks
    )

    results = {}

    for callback in callbacks:
        if hasattr(callback, "sample"):
            results.update(callback.sample())

    return results


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
    num_samples: Optional[int] = None,
    sampling_method: Literal["sgld", "sgnht"] = "sgld",
    **optimizer_kwargs
):
    
    if sampling_method == "sgld":
        optimizer_class = SGLD
    elif sampling_method == "sgnht":
        optimizer_class = SGNHT
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")

    optimizer_kwargs.update(dict(
        lr=lr,
        noise_level=noise_level,
        weight_decay=weight_decay,
        elasticity=elasticity,
        temperature="adaptive",
        num_samples=num_samples or len(dataset),
    ))

    def eval_rlct(model: nn.Module):
        return estimate_slt_observables(
            model,
            loader,
            F.mse_loss,
            optimizer_class,
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

