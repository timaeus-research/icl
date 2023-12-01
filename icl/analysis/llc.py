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
from tqdm import tqdm

import wandb
from icl.analysis.utils import get_sweep_configs
from icl.config import ICLConfig, get_config
from icl.train import Run

app = typer.Typer()

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


class ObservedOnlineLLCEstimator:
    def __init__(self, num_chains: int, num_draws: int, n: int, device="cpu", threshold=0.05):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.n = torch.tensor(n, dtype=torch.float32).to(device)
        self.losses = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(device)
        self.llcs = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(device)
        self.llc_per_chain = torch.zeros(num_chains, dtype=torch.float32).to(device)
        
        # If a chain has a loss < init_loss for more than `threshold` draws, do not include it in the final estimate.
        self.threshold = threshold
        self.num_draws_in_chain_below_init = torch.zeros(num_chains, dtype=torch.float32).to(device)
        self.chain_below_threshold = torch.zeros(num_chains, dtype=torch.bool).to(device)
        self.thresholded_llcs = torch.zeros(num_draws, dtype=torch.float32).to(device)

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

        if self.threshold:
            if loss < self.init_loss:
                self.num_draws_in_chain_below_init[chain] += 1

            if (self.num_draws_in_chain_below_init[chain] / draw) > self.threshold:
                self.chain_below_threshold[chain] = 0.
            else:
                self.chain_below_threshold[chain] = 1.


        # Assumes this is run serially
        if chain == self.num_chains - 1:
            thresholded_llcs = self.llcs[self.chain_below_threshold, draw]
            self.thresholded_llcs[draw] = thresholded_llcs.mean()

            wandb.log({
                "llc/mean": self.llcs[:, draw].mean().item(), 
                "llc/std": self.llcs[:, draw].std().item(),
                "llc/max": self.llcs[:, draw].max().item(),
                "llc/min": self.llcs[:, draw].min().item(),
                "thresholded-llc/mean": thresholded_llcs.mean().item(),
                "thresholded-llc/std": thresholded_llcs.std().item(),
                "thresholded-llc/max": thresholded_llcs.max().item(),
                "thresholded-llc/min": thresholded_llcs.min().item(),
                **{
                    f"chain-llcs/{i}": self.llcs[i, draw].item() for i in range(self.num_chains)
                },    
            }, step=draw)

    @property
    def init_loss(self):
        return self.losses[0, 0]

    def sample(self):
        return {
            # "llc/mean": self.llcs[:, -1].mean().cpu().numpy(),
            # "llc/std": self.llcs[:, -1].std().cpu().numpy(),
            # "llc/thresholded-mean": self.llcs[self.chain_below_threshold, :].mean(axis=0).cpu().numpy(),
            "llc/means": self.llcs.mean(axis=0).cpu().numpy(),
            "llc/stds": self.llcs.std(axis=0).cpu().numpy(),
            "llc/trace": self.llcs.cpu().numpy(),
            "loss/trace": self.losses.cpu().numpy(),

        }
    
    def __call__(self, chain: int, draw: int, loss: float):
        self.update(chain, draw, loss)



