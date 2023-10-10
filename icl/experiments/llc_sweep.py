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
from icl.analysis.sample import sample
from icl.analysis.utils import get_sweep_configs
from icl.config import ICLConfig, get_config
from icl.train import Run

app = typer.Typer()


class ObservedOnlineLLCEstimator:
    def __init__(self, num_chains: int, num_draws: int, n: int, device="cpu"):
        self.num_chains = num_chains
        self.num_draws = num_draws
        self.n = torch.tensor(n, dtype=torch.float32).to(device)
        self.losses = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(device)
        self.llcs = torch.zeros((num_chains, num_draws), dtype=torch.float32).to(device)
        self.llc_per_chain = torch.zeros(num_chains, dtype=torch.float32).to(device)
        self.llc_means = torch.tensor(num_chains, dtype=torch.float32).to(device)
        self.llc_stds = torch.tensor(num_chains, dtype=torch.float32).to(device)

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

        # Assumes this is run serially
        if chain == self.num_chains - 1:
            wandb.log({"llc/mean": self.llcs[:, draw].mean().item(), "llc/std": self.llcs[:, draw].std().item()}, step=draw)

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


def estimate_llc_at_end(
    config: dict,
    gamma: float, 
    lr: float, 
    num_draws: int=1000, 
    num_chains: int=25,
    use_wandb: bool=False,
):      
    config = get_config(**config)

    print("Loaded configs")

    device = get_default_device()
    num_layers = config.task_config.num_layers
    num_heads = config.task_config.num_heads
    num_tasks = config.task_config.num_tasks

    print("\n")
    print("-" * 30 + f" M={num_tasks} " + "-" * 30)
    run = Run.create_and_restore(config)

    xs, ys = run.evaluator.pretrain_xs, run.evaluator.pretrain_ys
    dataset = torch.utils.data.TensorDataset(xs, ys)
    loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset)) 
    
    llc_estimator = ObservedOnlineLLCEstimator(num_chains, num_draws, len(dataset), device=device)
    callbacks=[llc_estimator]

    sample(
        run.model,
        loader,
        F.mse_loss,
        SGLD,
        optimizer_kwargs=dict(
            lr=lr,
            noise_level=1.,
            weight_decay=0.,
            elasticity=gamma,
            temperature="adaptive",
            num_samples=len(dataset),
        ),
        num_draws=num_draws,
        num_chains=num_chains,
        cores=1,
        device=device,
        callbacks=callbacks,
    )

    llcs = llc_estimator.sample()

    trace = llcs["loss/trace"]
    llcs_over_time_mean = llcs["llc/means"]
    llcs_over_time_std = llcs["llc/stds"]

    fig = plt.figure()
    cmap = plt.cm.viridis

    for chain in range(num_chains):
        data = trace[chain, :]
        color = cmap(chain / num_chains)
        sns.lineplot(x=np.arange(num_draws), y=data, color=color, alpha=0.5, label=f"_Chain {chain}")
    
    # Horizontal line at the initial loss
    init_loss = trace[0, 0]
    plt.axhline(y=init_loss, color="k", linestyle="--")

    plt.xlabel("num_steps")
    plt.ylabel("$L_n(w_t)$")
    plt.title(f"LLC trace (L={num_layers}, H={num_heads}, M={num_tasks}, lr={lr}, gamma={gamma}, num_draws={num_draws})")
    
    # Add extra axis to plot for the llcs_over_time
    ax2 = plt.twinx()
    ax2.plot(np.arange(num_draws), llcs_over_time_mean, color="r", alpha=0.5, label=r"$\hat\lambda$")
    ax2.fill_between(np.arange(num_draws), llcs_over_time_mean - llcs_over_time_std, 
                        llcs_over_time_mean + llcs_over_time_std, color="r", alpha=0.15)

    ax2.set_ylabel(r"$\hat\lambda$")
    ax2.legend()
    plt.savefig(f"figures/llc-trace-L{num_layers}-H{num_heads}-M{num_tasks}-lr={lr}-gamma={gamma}-num_draws={num_draws}.png")
    plt.close()


@app.command("wandb")
def llc_sweep_with_wandb():      
    wandb.init(project="icl-llc", entity="devinterp")
    print("Initialized wandb")
    config = dict(wandb.config)
    analysis_config = config.pop("analysis_config")
    estimate_llc_at_end(config, **analysis_config, use_wandb=True)
    wandb.finish()


@app.command("plot")
def plot_grid_search_results(csv_path: str, num_chains: int=25):
    # Read the DataFrame from the CSV file
    df = pd.read_csv(csv_path)

    # Get unique values for lrs, gammas, and num_tasks
    unique_lrs = df['lr'].unique()
    unique_gammas = df['gamma'].unique()
    unique_num_tasks = df['num_tasks'].unique()

    # Sort for visual consistency
    unique_lrs.sort()
    unique_gammas.sort()
    unique_num_tasks.sort()

    # Initialize colormap
    cmap = plt.cm.viridis

    # Create subplots
    fig, axes = plt.subplots(len(unique_lrs), len(unique_gammas), figsize=(15, 15))

    fig.suptitle(f"$\hat\lambda$ grid search ($n_\mathrm{{chains}}={num_chains}$)")

    # Loop through the grid
    for i, lr in enumerate(unique_lrs):
        for j, gamma in enumerate(unique_gammas):
            ax = axes[i, j]

            # Filter DataFrame for specific lr and gamma
            filtered_df = df[(df['lr'] == lr) & (df['gamma'] == gamma)]

            for num_tasks in unique_num_tasks:
                task_specific_df = filtered_df[filtered_df['num_tasks'] == num_tasks]

                # Sort by 'num_draws' for plotting
                task_specific_df = task_specific_df.sort_values('num_draws')

                # Calculate color based on log2(num_tasks)
                color = cmap(np.log2(num_tasks) / np.log2(max(unique_num_tasks)))

                # Plot using Seaborn for better aesthetics
                sns.lineplot(x='num_draws', y='lc/mean', data=task_specific_df, ax=ax, label=f'M={num_tasks}', color=color)
                ax.fill_between(task_specific_df['num_draws'], task_specific_df['lc/mean'] - task_specific_df['lc/std'], 
                                task_specific_df['lc/mean'] + task_specific_df['lc/std'], color=color, alpha=0.15)

            ax.set_title(f"$\epsilon={lr}, \gamma={gamma}$")
            ax.set_xlabel(r"$t_\mathrm{SGLD}$")
            ax.set_ylabel(r"$\hat\lambda$")

    plt.legend()
    plt.savefig("figures/llc-grid-search.png")
    plt.close()



if __name__ == "__main__":
    app()
    


