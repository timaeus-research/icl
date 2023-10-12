import os
from typing import Dict, List, Optional, Tuple

import devinfra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import typer
from devinfra.utils.device import get_default_device
from devinfra.utils.iterables import rm_none_vals
from torch.multiprocessing import cpu_count
from tqdm import tqdm

import wandb
from icl.analysis.cov import (WithinHeadCovarianceCallback,
                              make_transformer_cov_accumulator)
from icl.analysis.llc import make_slt_evals
from icl.analysis.utils import (get_sweep_configs, get_unique_config,
                                split_attn_weights)
from icl.config import ICLConfig, get_config
from icl.experiments.utils import *
from icl.train import Run

app = typer.Typer()


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



def plot_evecs(evals: Dict[str, np.ndarray], evecs: Dict[str, np.ndarray], shapes: Dict[str, Tuple[int, ...]], title: str = "", save: Optional[str] = None):
    # Correct shapes to make sure there 2 dimensions and that they are displayed horizontally
    for name in shapes.keys():
        shape = shapes[name]

        if len(shape) == 1:
            shape = (shape[0], 1)
        elif len(shape) > 2:
            shape = (shape[0], np.prod(shape[1:]))
        
        if shape[1] > shape[0]:
            shape = shape[::-1]
        
        shapes[name] = shape
    
    # Scale each subfigure by the height of the matrix
    relative_heights = [shapes[key.split("-")[0]][-1] for key in evecs.keys()]
    total_height = sum(relative_heights)
    relative_heights = [h / total_height for h in relative_heights]
    

    fig, axes = plt.subplots(len(evecs), 1, figsize=(5, 100), gridspec_kw={'height_ratios': relative_heights})
    plt.suptitle(title)
    
    max_, min_ = max((evec.max() for evec in evecs.values())), min((evec.min() for evec in evecs.values()))

    for ax, cov_name in zip(axes, evecs.keys()):
        layer_name = cov_name.split("-")[0]
        evec, shape = evecs[cov_name], shapes[layer_name]

        parts = tuple((n.split(":")[1] for n in cov_name.split("-")))

        if len(parts) == 1 or parts[0] == parts[1]:
            subtitle = f"Within {parts[0]} "
        else:
            subtitle = f"Between {parts[0]} & {parts[1]}"

        subtitle += f"(eval: {evals[cov_name]:.2f})"

        ax.set_title(subtitle)
        ax.matshow(evec.reshape(shape).T, cmap='viridis', vmin=min_, vmax=max_)
        ax.grid(None)
        # ax.set_xlabel("Axis 1")
        # ax.set_ylabel("Axis 0")

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    if save:
        plt.savefig(save)

    plt.close()



def llcs_and_cov(config: ICLConfig, gamma: float=1., lr: float=1e-4, num_draws: int=1000, num_chains: int=10, device: Optional[str]=None, cores: Optional[int]=None, num_evals=3, steps: Optional[list] = None):
    cores = cores or int(os.environ.get("CORES", cpu_count() // 2))
    device = device or get_default_device()

    run = Run(config)
    print(run.config.to_slug(delimiter="-"))
    model = run.model
    model.train()
    checkpointer = run.checkpointer
    xs, ys = run.evaluator.pretrain_xs, run.evaluator.pretrain_ys
    trainset = torch.utils.data.TensorDataset(xs, ys)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(xs))

    device = str(get_default_device())
    cov_accumulator = make_transformer_cov_accumulator(model, device=device, num_evals=num_evals)

    slt_evals = make_slt_evals(
        dataset=trainset,
        loader=trainloader,
        cores=cores,
        lr=lr,
        elasticity=gamma,
        # num_draws=100,
        # num_chains=10, 
        num_draws=num_draws,
        num_chains=num_chains,
        device=device,
        callbacks=[cov_accumulator]
    )

    # slug = run.config.to_slug(delimiter="-")
    # logger = devinfra.io.logging.CompositeLogger([
    #     run.logger,
    #     devinfra.io.logging.CsvLogger(ANALYSIS / f"lc-and-cov-{slug}.csv")
    # ])
    # logger = run.logger
    # print([l for l in logger.loggers])

    steps = list(checkpointer.file_ids) if steps is None else steps
    checkpointer.file_ids = steps  # Gross, sorry. 

    for step, model in zip(steps, iter_models(model, checkpointer)):
        print(step)
        observables = slt_evals(run.model)

        trace = observables.pop("loss/trace")
        covariances = cov_accumulator.to_eigens()

        # original_shapes = {name: tuple(accessor(model).shape) for name, accessor in cov_accumulator.accessors.items()}
        # principal_evals = {}
        # principal_evecs= {}

        for name, results in covariances.items():
            evecs, evals = results["evecs"], results["evals"]

            parts = [p.split(":")[-1].replace("/", ".") for p in name.split("-")]
            obs_name = "x".join(parts)

            if len(parts) == 1:
                obs_name = "within/" + obs_name
            else:
                obs_name = "between/" + obs_name

            for i in range(num_evals):
                observables[f"cov_eval_{i}/{obs_name}"] = evals[i]

            # principal_evals[name] = evals[0]
            # principal_evecs[name] = evecs[:, 0]

        # slug = FIGURES / (f"cov-{config.to_slug()}@t={step}".replace(".", "_"))
        # title = f"Principal covariance eigenvalues\n{config.to_latex()}"

        # plot_evecs(evals=principal_evals, evecs=principal_evecs, shapes=original_shapes, title=title, save=slug)
        # logger.log(observables, step=step)
        wandb.log(observables, step=step)
        cov_accumulator.reset()

@app.command("cov")
def llcs_and_cov_from_cmd_line(
    sweep: str = typer.Option(None, help="Path to wandb sweep YAML file"), 
    num_tasks: int = typer.Option(None, help="Number of tasks to train on"), 
    num_layers: int = typer.Option(None, help="Number of transformer layers"), 
    num_heads: int = typer.Option(None, help="Number of transformer heads"), 
    embed_size: int = typer.Option(None, help="Embedding size"), 
    gamma: float = typer.Option(None, help="Elasticity"), 
    epsilon: float = typer.Option(None, help="SGLD step size"),
    lr: float = typer.Option(None, help="Learning rate"), 
    num_draws: int = typer.Option(None, help="Number of draws"), 
    num_chains: int = typer.Option(None, help="Number of chains"), 
    steps: Optional[List[int]] = typer.Option(None, help="Step"), 
):
    """
    Initialise and train an InContextRegressionTransformer model, tracking
    various metrics.
    """
        
    filters = rm_none_vals(dict(task_config={"num_tasks": num_tasks, "num_layers": num_layers, "num_heads": num_heads, "embed_size": embed_size}, optimizer_config={"lr": lr}))
    analysis_config = rm_none_vals(dict(gamma=gamma, lr=epsilon, num_draws=num_draws, num_chains=num_chains,  steps=steps))
    config = get_unique_config(sweep, **filters)
    llcs_and_cov(config, **analysis_config)


@app.command("from-wandb")
def llcs_and_cov_from_wandb():
    """
    Initialise and train an InContextRegressionTransformer model, tracking
    various metrics.
    """
    config = get_config(project="icl", entity="devinterp")
    wandb.init(project="icl", entity="devinterp")
    print("Initialized wandb")
    config_dict = dict(wandb.config)
    analysis_config = config_dict.pop("analysis_config")
    config = get_config(**config_dict)
    wandb.run.name = f"L{config['task_config']['num_layers']}H{config['task_config']['num_heads']}M{config['task_config']['num_tasks']}"
    llcs_and_cov(config, **analysis_config)
    wandb.finish()



if __name__ == "__main__":
    prepare_experiments()
    app()
