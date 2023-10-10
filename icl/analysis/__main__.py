import os

from dotenv import load_dotenv

from icl.analysis.cov import WithinHeadCovarianceCallback

load_dotenv()


import itertools
import logging
import warnings
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from logging import Logger
from pathlib import Path
from pprint import pp
from typing import (Callable, Dict, Iterable, List, Literal, Optional, Tuple,
                    Type, TypeVar, Union)

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
from devinfra.evals import Criterion
from devinfra.utils.device import get_default_device
from devinterp.optim.sgld import SGLD
from devinterp.slt.learning_coeff import plot_learning_coeff_trace
from pydantic import BaseModel
from scipy.sparse.linalg import eigsh
from torch import nn
from torch.multiprocessing import (Pool, cpu_count, get_context,
                                   set_start_method)
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typer import Typer

import wandb
from icl.analysis.rlct import make_slt_evals, map_slt_evals_over_run
from icl.analysis.utils import (generate_config_dicts_from_path,
                                get_unique_run, split_attn_weights)
from icl.config import ICLConfig, get_config
from icl.train import Run

sns.set_theme(style="whitegrid")


FIGURES=Path("figures")
ANALYSIS = Path("analysis")

assert os.path.exists(FIGURES)
assert os.path.exists(ANALYSIS)

DEVICE = devinfra.utils.device.get_default_device()
K=3  # Num cov components

app = typer.Typer()




@app.command("run")
def rlcts_over_run(
    sweep: str = typer.Argument(..., help="Path to sweep config file"),
    run_name: str = typer.Argument(..., help="Name of run to evaluate"),
):
    """Find the RLCT configuration for a given sweep."""
    config_dicts = list(generate_config_dicts_from_path(sweep))
    config_dict = find_obj(config_dicts, run_name=run_name) 
    config = get_config(**config_dict)
    analysis_config = config_dict.get("analysis_config", {})  # Replace this line as appropriate
    run = Run.create_and_restore(config)
    pp(run.evaluator(run.model))

    xs, ys = run.evaluator.pretrain_xs, run.evaluator.pretrain_ys
    trainset = torch.utils.data.TensorDataset(xs, ys)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(xs))

    eval_rlcts = make_slt_evals(
        dataset=trainset,
        loader=trainloader,
        **analysis_config
    )

    evals = eval_rlcts(run.model)
    pp(evals)


@app.command("sweep")
def rlcts_over_sweep(sweep: str = typer.Option(None, help="Path to wandb sweep YAML file")):
    """
    Estimate RLCTs for each checkpoint for each run in a wandb sweep.
    """
    if sweep:
        for config_dict in generate_config_dicts_from_path(sweep, extra="rlct"):
            analysis_config = config_dict.pop("analysis_config")
            config = get_config(**config_dict)
            map_slt_evals_over_run(config, analysis_config)
    else:
        config = get_config(project="icl", entity="devinterp", extra="rlct")  # Replace as needed
        analysis_config = wandb.config["analysis_config"]
        map_slt_evals_over_run(config, analysis_config)



def iter_models(model, checkpointer, verbose=False):
    for file_id in tqdm(checkpointer.file_ids, desc="Iterating over checkpoints", disable=not verbose):
        model.load_state_dict(checkpointer.load_file(file_id)["model"])
        yield model



def plot_attn_weights(W: torch.Tensor, num_heads: int, embed_dim: int, head_size: int, cols=("$W_Q^{(h)}$", "$W_K^{(h)}$", "$W_V^{(h)}$"), title="", save: Optional[str] = None, rows:Optional[List[str]] =None):
    if len(W.shape) == 1:  # Num heads * Embedding dimension * Head size * 3
        heads = list(split_attn_weights(W, num_heads, embed_dim, head_size))
    elif len(W.shape) == 3:  # Num heads, Embedding dimension, Head size * 3
        heads = [tuple(W[:, h, i*head_size:(i+1)*head_size].T for i in range(3)) for h in range(num_heads)]
    else:
        raise ValueError(f"Expected W to have shape (num_heads, embed_dim, head_size * 3) or (num_heads * embed_dim * head_size * 3), got {W.shape}")

    fig, axs = plt.subplots(num_heads, 3, figsize=(25, 10))
    plt.suptitle(title)

    rows = rows or [f"Head {h}" for h in range(num_heads)]

    min_, max_ = W.min(), W.max()

    for h, head in enumerate(heads):
        axs[h, 0].set_ylabel(f"{rows[h]}\nHead Size")

        for i, mat in enumerate(head):
            axs[h, i].matshow(mat.detach().cpu().numpy().T, cmap='viridis', vmin=min_, vmax=max_) 

    for i, col in enumerate(cols):
        axs[0, i].set_title(col)
        axs[-1, i].set_xlabel("Embedding Dimension")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # Plot colorbar somewhere on right
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(axs[0, 0].images[0], cax=cbar_ax)
    

    if save:
        parent_dir = os.path.dirname(save)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        plt.savefig(save)

    # plt.show()


def plot_attn_head_weights(head: torch.Tensor, embed_dim, head_size: int, title="", subtitles=("$W_Q$", "$W_K$", "$W_V$"), save: Optional[str] = None):
    head_Ex3c = head.view((embed_dim, head_size * 3))
    q, k, v = tuple(head_Ex3c[:, i*head_size:(i+1)*head_size].detach().cpu().numpy() for i in range(3))

    fig, ax = plt.subplots(1, 3, figsize=(30, 3.5))
    plt.suptitle(title)

    for i, (mat, subtitle) in enumerate(zip((q, k, v), subtitles)):
        ax[i].set_title(subtitle)
        ax[i].matshow(mat.T, cmap='viridis')
        ax[i].set_xlabel("Embedding Dimension")
        ax[i].set_ylabel("Head Size")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save:
        parent_dir = os.path.dirname(save)
        if not os.path.exists(parent_dir):
            os.makedirs(parent_dir)
        plt.savefig(save)

    # plt.show()

def plot_attn_eigencomponents(evecs, evals, num_layers: int, slug: Optional[str] = None, title=""):
    for i, eval in enumerate(evals):

        if len(evecs.shape) == 2:  # Vectorized attention, Eigenvector index
            # attns = evecs[:evecs.shape[0]//num_layers, -i], evecs[evecs.shape[0]//2:, -i]
            attn_size = evecs.shape[0] // num_layers
            attn_layers = [evecs[attn_size * l:attn_size * (l+1), i] for l in range(num_layers)]
            
        else:  # [Eigenvector index, Layer index, head index, embed dim, head size * 3]
            attn_layers = [evecs[i, l] for l in range(evecs.shape[1])]

        for l, attn_layer in enumerate(attn_layers):
            layer_evals = eval[l]

            rows = None
            evals_label = ""

            if len(layer_evals) > 1: 
                rows = [f"Head {h} (eval={layer_evals[h]:.2f})" for h in range(len(layer_evals))]
            else:
                evals_label = f" with value {layer_evals})"

            plot_attn_weights(
                torch.Tensor(attn_layer), 
                num_heads=4,
                embed_dim=64, 
                head_size=16, 
                title=f"Layer {l} Eigenvector {i}{evals_label}\n{title}",
                cols=(f"$u_{{Q,{i}}}^{{({l})}}$", f"$u_{{K,{i}}}^{{({l})}}$", f"$u_{{V,{i}}}^{{({l})}}$"),
                save=(FIGURES / (f"cov-attn{l}-evec{i}-" + slug + ".png") if slug else None),
                rows=rows
            )


@app.command("cov")
def rlct_and_cov():
    """
    Initialise and train an InContextRegressionTransformer model, tracking
    various metrics.
    """
    config = get_config(project="icl", entity="devinterp")

    run = Run(config)
    print(run.config.to_slug(delimiter="-"))
    model = run.model
    model.train()
    checkpointer = run.checkpointer
    xs, ys = run.evaluator.pretrain_xs, run.evaluator.pretrain_ys
    trainset = torch.utils.data.TensorDataset(xs, ys)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(xs))

    device = str(get_default_device())

    callbacks = [
        WithinHeadCovarianceCallback(
            head_size=run.config.task_config.embed_size // run.config.task_config.num_heads,
            num_heads=run.config.task_config.num_heads,
            embed_size=run.config.task_config.embed_size,
            num_evals=K,
            device=device,
            paths=[
                f"token_sequence_transformer.blocks.{i}.attention.attention"
                for i in range(run.config.task_config.num_layers)
            ],
        )
    ]

    cores = int(os.environ.get("CORES", cpu_count() // 2))

    slt_evals = make_slt_evals(
        dataset=trainset,
        loader=trainloader,
        cores=cores,
        lr=1e-5,
        num_draws=100,
        elasticity=1.,
        num_chains=10,
        device=device,
        callbacks=callbacks
    )

    slug = run.config.to_slug(delimiter="-")

    logger = devinfra.io.logging.CompositeLogger([
        run.logger,
        devinfra.io.logging.CsvLogger(ANALYSIS / f"lc-and-cov-{slug}.csv")
    ])

    steps = list(checkpointer.file_ids)

    for step, model in zip(steps, iter_models(model, checkpointer)):
        print(step)
        observables = slt_evals(run.model)

        trace = observables.pop("lc/trace")
        trace.to_csv(ANALYSIS / f"lc-trace-{slug}@t={step}.csv")

        evecs = observables.pop("cov/evecs")
        evals = observables.pop("cov/evals")

        for i in range(K):
            for l in range(run.config.task_config.num_layers):
                for h in range(run.config.task_config.num_heads):
                    observables[f"cov/eval-{i}/layer_{l}/head_{h}"] = evals[i, l, h]

        logger.log(observables, step=step)

        plot_attn_eigencomponents(
            evecs, 
            evals, 
            num_layers=run.config.task_config.num_layers, slug=slug + f"@t={step}",
            title="Within head covariance matrix\n" + run.config.to_latex() + f"\nt={step}"
        )
        plt.close()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    set_start_method('spawn')  # Required for sharing CUDA tensors
    load_dotenv()
    sentry_sdk.init(
        dsn="https://92ea29f1e366cda4681fb10273e6c2a7@o4505805155074048.ingest.sentry.io/4505805162479616",
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        # We recommend adjusting this value in production.
        traces_sample_rate=1.0,
        # Set profiles_sample_rate to 1.0 to profile 100%
        # of sampled transactions.
        # We recommend adjusting this value in production.
        profiles_sample_rate=1.0,
    )
    app()
