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
from icl.config import ICLConfig
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
    callbacks: List[Callable] = []
):
    def eval_rlct(model: nn.Module):
        optimizer_kwargs = dict(
            lr=lr,
            noise_level=noise_level,
            weight_decay=weight_decay,
            elasticity=elasticity,
            temperature="adaptive",
            num_samples=len(dataset),
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


@app.command("grid-search")
def llc_hyperparam_grid_search_sgld(
    path: Path,
    gammas: List[float]=[1., 10., 100.], 
    lrs: List[float]=[1e-6, 1e-5, 1e-4], 
    num_draws: int=1000, 
    num_chains: int=25,
    log_num_tasks: Optional[List[int]] = None
):      
    configs = list(get_sweep_configs(path))
    results = []

    print("Loaded configs")

    device = get_default_device()

    for config in configs:
        if log_num_tasks and int(np.log2(config.task_config.num_tasks)) not in log_num_tasks:
            continue

        num_layers = config.task_config.num_layers
        num_heads = config.task_config.num_heads
        num_tasks = config.task_config.num_tasks

        print("\n")
        print("-" * 30 + f" M={num_tasks} " + "-" * 30)
        run = Run.create_and_restore(config)

        xs, ys = run.evaluator.pretrain_xs, run.evaluator.pretrain_ys
        dataset = torch.utils.data.TensorDataset(xs, ys)
        loader = torch.utils.data.DataLoader(dataset, batch_size=len(dataset)) 
        callbacks=[]

        for gamma, lr in itertools.product(gammas, lrs):
            llcs = estimate_slt_observables(
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
                online=True
            )

            trace = llcs["lc/trace"]
            llcs_over_time_mean = llcs["lc/online/mean"]
            llcs_over_time_std = llcs["lc/online/std"]

            plt.figure()
            cmap = plt.cm.viridis

            for chain in range(num_chains):
                del llcs[f"lc/chain_{chain}/mean"]
                data = trace.loc[trace["chain"] == chain]
                color = cmap(chain / num_chains)
                sns.lineplot(x=np.arange(num_draws), y=data["loss"], color=color, alpha=0.5, label=f"_Chain {chain}")
            
            # Horizontal line at the initial loss
            init_loss = trace.loc[trace["step"] == 0, "loss"].iloc[0]
            plt.axhline(y=init_loss, color="k", linestyle="--")

            plt.xlabel("num_steps")
            plt.ylabel("$L_n(w_t)$")
            plt.title(f"LLC trace (L={num_layers}, H={num_heads}, M={num_tasks}, lr={lr}, gamma={gamma}, num_draws={num_draws})")
            
            # Add extra axis to plot for the llcs_over_time
            ax2 = plt.twinx()
            ax2.plot(np.arange(num_draws), llcs_over_time_mean, color="r", alpha=0.5)
            ax2.fill_between(np.arange(num_draws), llcs_over_time_mean - llcs_over_time_std, 
                             llcs_over_time_mean + llcs_over_time_std, color="r", alpha=0.15)

            ax2.set_ylabel(r"$\hat\lambda$")
            
            ax2.legend()
            plt.savefig(f"figures/llc-trace-L{num_layers}-H{num_heads}-M{num_tasks}-lr={lr}-gamma={gamma}-num_draws={num_draws}.png")
            plt.close()

            for t in range(num_draws):
                results.append({
                    "lc/mean": llcs_over_time_mean[t],
                    "lc/std": llcs_over_time_std[t],
                    "gamma": gamma,
                    "lr": lr,
                    "num_draws": t,
                    **(config.task_config.model_dump()),
                })

            print(yaml.dump(results[-1]))
        
    df = pd.DataFrame(results)
    df.to_csv("analysis/llc-grid-search.csv")


@app.command("plot-grid")
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
    load_dotenv()
    app()