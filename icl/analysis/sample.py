import itertools
import os
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from logging import Logger
from pathlib import Path
from pprint import pp
from typing import (Callable, Dict, Iterable, List, Literal, Optional, Tuple,
                    Type, Union)

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
    online: bool = False,
):
    # Initialize new model and optimizer for this chain
    model = deepcopy(ref_model).to(device)

    num_samples = len(loader.dataset)
    optimizer_kwargs = optimizer_kwargs or {}
    optimizer = sampling_method(model.parameters(), **optimizer_kwargs)

    if seed is not None:
        torch.manual_seed(seed)

    num_steps = num_draws * num_steps_bw_draws + num_burnin_steps

    local_draws = pd.DataFrame(
        index=range(num_draws),
        columns=["chain", "step", "loss"] + ["llc"] if online else [],
    )

    model.train()
    n = torch.tensor(num_samples, device=device)
    t = torch.tensor(0, device=device)
    prev_llc = torch.tensor(0, device=device) if online else None
    init_loss = torch.tensor(0, device=device) if online else None

    for i, (xs, ys) in  tqdm(zip(range(num_steps), itertools.cycle(loader)), desc=f"Chain {chain}", total=num_steps, disable=not verbose):
        optimizer.zero_grad()
        xs, ys = xs.to(device), ys.to(device)
        y_preds = model(xs, ys)
        loss = criterion(y_preds, ys)

        loss.backward()
        optimizer.step()

        if i >= num_burnin_steps and (i - num_burnin_steps) % num_steps_bw_draws == 0:
            t += 1
            draw_idx = (i - num_burnin_steps) // num_steps_bw_draws
            local_draws.loc[draw_idx, "step"] = i
            local_draws.loc[draw_idx, "chain"] = chain
            local_draws.loc[draw_idx, "loss"] = loss.detach().item()
            
            with torch.no_grad():
                for callback in callbacks:
                    callback(model)

            if online:
                if draw_idx == 0:
                    init_loss = loss.detach()
                    local_draws.loc[draw_idx, "llc"] = 0.
                else:
                    with torch.no_grad():
                        llc =  (1 / t) * (
                            (t - 1) * prev_llc + (n / n.log()) * (loss - init_loss)
                        )
                        prev_llc = llc
                        local_draws.loc[draw_idx, "llc"] = llc.detach().item()

    return local_draws


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
    online: bool = False,
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
            online=online,
        )

    results = []

    if cores > 1:
        ctx = get_context("spawn")
        with ctx.Pool(cores) as pool:
            results = pool.map(_sample_single_chain, [get_args(i) for i in range(num_chains)])
    else:
        for i in range(num_chains):
            results.append(_sample_single_chain(get_args(i)))

    results_df = pd.concat([r for r in results], ignore_index=True)

    for callback in callbacks:
        if hasattr(callback, "finalize"):
            callback.finalize()

    return results_df


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

    trace = sample(
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
        online=online,
    )

    baseline_loss = trace.loc[trace["chain"] == 0, "loss"].iloc[0]
    num_samples = len(loader.dataset)
    avg_losses = trace.groupby("chain")["loss"].mean()
    chain_averages = np.zeros(num_chains)

    results = {}

    if online:
        for i in range(num_chains):
            chain_averages[i] = trace.loc[(trace["chain"] == i) & (trace["step"] == num_draws-1), "llc"].values[0]

        llcs_over_time = trace.groupby("step")["llc"]
        results["lc/online/mean"] = llcs_over_time.mean().values
        results["lc/online/std"] = llcs_over_time.std().values

    else:
        for i in range(num_chains):
            chain_avg_loss = avg_losses.iloc[i]
            chain_averages[i] = (chain_avg_loss - baseline_loss) * num_samples / np.log(num_samples)

    avg_loss = chain_averages.mean()
    std_loss = chain_averages.std()

    results.update({
        "lc/mean": avg_loss.item(),
        "lc/std": std_loss.item(),
        "lc/trace": trace,
        **{f"lc/chain_{i}/mean": chain_averages[i].item() for i in range(num_chains)},
        # **{f"lc/chain/{i}/trace": trace.loc[trace["chain"] == i] for i in range(num_chains)},
    })

    for callback in callbacks:
        results.update(callback.sample())
    
    return results




def estimate_rlct(
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
) -> float:
    trace = sample(
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
    )[0]

    baseline_loss = trace.loc[trace["chain"] == 0, "loss"].iloc[0]
    num_samples = len(loader.dataset)
    avg_losses = trace.groupby("chain")["loss"].mean()
    results = torch.zeros(num_chains, device=device)

    for i in range(num_chains):
        chain_avg_loss = avg_losses.iloc[i]
        results[i] = (chain_avg_loss - baseline_loss) * num_samples / np.log(num_samples)

    avg_loss = results.mean()

    return avg_loss.item()
