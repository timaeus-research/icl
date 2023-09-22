import itertools
import multiprocessing
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from logging import Logger
from typing import Callable, Dict, List, Literal, Optional, Union

import numpy as np
import pandas as pd
import torch
import yaml
from devinterp.optim.sgld import SGLD
from devinterp.slt.observables import MicroscopicObservable
from devinterp.utils import Criterion
from torch import nn
from torch.multiprocessing import Pool, cpu_count
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm


def sample_single_chain(
    ref_model: nn.Module,
    loader: DataLoader,
    criterion: Callable,
    num_draws=100,
    num_burnin_steps=0,
    num_steps_bw_draws=1,
    step: Literal["sgld"] = "sgld",
    optimizer_kwargs: Optional[Dict] = None,
    chain: int = 0,
    seed: Optional[int] = None,
    pbar: bool = False,
    observor: Optional[MicroscopicObservable] = None,
    device: Union[str, torch.device] = "cpu",
):
    # Initialize new model and optimizer for this chain
    is_on_xla = str(device).startswith("xla")

    if is_on_xla:
        device = xm.xla_device()
        loader = pl.MpDeviceLoader(loader, device)

    model = deepcopy(ref_model).to(device)

    if step == "sgld":
        optimizer_kwargs = optimizer_kwargs or {}
        optimizer = SGLD(
            model.parameters(), **optimizer_kwargs
        )  # Replace with your actual optimizer kwargs

    if seed is not None:
        torch.manual_seed(seed)

    num_steps = num_draws * num_steps_bw_draws + num_burnin_steps
    local_draws = pd.DataFrame(index=range(num_draws), columns=["chain", "loss"])

    iterator = zip(range(num_steps), itertools.cycle(loader))

    if pbar:
        iterator = tqdm(iterator, desc=f"Chain {chain}", total=num_steps)

    model.train()

    for i, (xs, ys) in iterator:
        optimizer.zero_grad()
        xs, ys = xs.to(device), ys.to(device)
        y_preds = model(xs, ys)  # The difference with the original
        loss = criterion(y_preds, ys)
        loss.backward()
        optimizer.step()

        if is_on_xla:
            xm.mark_step()

        if i >= num_burnin_steps and (i - num_burnin_steps) % num_steps_bw_draws == 0:
            draw_idx = (i - num_burnin_steps) // num_steps_bw_draws
            local_draws.loc[draw_idx, "chain"] = chain
            local_draws.loc[draw_idx, "loss"] = loss.detach().item()

    return local_draws


def _sample_single_chain(index, kwargs):
    seeds = kwargs["seeds"]
    seed = seeds[index]
    chain = index

    new_kwargs = {**kwargs}
    del new_kwargs["seeds"]

    return sample_single_chain(chain=chain, seed=seed, **new_kwargs)


def sample(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    step: Literal["sgld"],
    optimizer_kwargs: Optional[Dict[str, Union[float, Literal["adaptive"]]]] = None,
    num_draws: int = 100,
    num_chains: int = 10,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    cores: int = 1,
    seed: Optional[Union[int, List[int]]] = None,
    pbar: bool = True,
    device: str = "cpu",
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
        progressbar (bool): Whether to display a progress bar.
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

    sample_single_chain_kwargs = dict(
        ref_model=model,
        loader=loader,
        criterion=criterion,
        num_draws=num_draws,
        num_burnin_steps=num_burnin_steps,
        num_steps_bw_draws=num_steps_bw_draws,
        step=step,
        optimizer_kwargs=optimizer_kwargs,
        pbar=pbar,
        device=device,
        seeds=seeds,
    )

    results = []

    if cores > 1:
        if str(device) == "cpu":
            with Pool(cores) as pool:
                results = pool.starmap(
                    _sample_single_chain,
                    [(i, sample_single_chain_kwargs) for i in range(num_chains)],
                )
        elif str(device) == "xla":
            xmp.spawn(
                _sample_single_chain,
                args=(sample_single_chain_kwargs,),
                nprocs=cores,
                start_method="fork",
            )
        else:
            raise NotImplementedError("Cannot currently use multiprocessing with GPU")
    else:
        for i in range(num_chains):
            results.append(_sample_single_chain(i, sample_single_chain_kwargs))

    results_df = pd.concat(results, ignore_index=True)
    return results_df


def estimate_rlct(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: Criterion,
    step: Literal["sgld"],
    optimizer_kwargs: Optional[Dict[str, Union[float, Literal["adaptive"]]]] = None,
    num_draws: int = 100,
    num_chains: int = 10,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    cores: int = 1,
    seed: Optional[Union[int, List[int]]] = None,
    pbar: bool = True,
    baseline: Literal["init", "min"] = "init",
    device: str = "cpu",
) -> float:
    trace = sample(
        model=model,
        loader=loader,
        criterion=criterion,
        step=step,
        optimizer_kwargs=optimizer_kwargs,
        num_draws=num_draws,
        num_chains=num_chains,
        num_burnin_steps=num_burnin_steps,
        num_steps_bw_draws=num_steps_bw_draws,
        cores=cores,
        seed=seed,
        pbar=pbar,
        device=device,
    )

    if baseline == "init":
        baseline_loss = trace.loc[trace["chain"] == 0, "loss"].iloc[0]
    elif baseline == "min":
        baseline_loss = trace["loss"].min()

    avg_loss = trace.groupby("chain")["loss"].mean().mean()
    num_samples = len(loader.dataset)

    return (avg_loss - baseline_loss) * num_samples / np.log(num_samples)
