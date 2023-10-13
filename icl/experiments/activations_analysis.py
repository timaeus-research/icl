import os
import warnings
from typing import Callable, Dict, List, Literal, Optional, Tuple

import devinfra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import typer
from devinfra.utils.device import get_default_device
from devinfra.utils.iterables import rm_none_vals
from devinterp.mechinterp.hooks import hook
from torch.multiprocessing import cpu_count
from tqdm import tqdm

import wandb
from icl.analysis.cov import (WithinHeadCovarianceCallback,
                              make_transformer_cov_accumulator)
from icl.analysis.llc import make_slt_evals
from icl.analysis.utils import (get_sweep_configs, get_unique_config,
                                split_attn_weights)
from icl.config import ICLConfig, ICLTaskConfig, get_config
from icl.experiments.utils import *
from icl.tasks import (DiscreteTaskDistribution, GaussianTaskDistribution,
                       RegressionSequenceDistribution, apply_transformations)
from icl.train import Run
from icl.utils import (get_locations, get_model_locations_to_display,
                       get_model_locations_to_slug)

app = typer.Typer()


def iter_models(model, checkpointer, verbose=False):
    for file_id in tqdm(checkpointer.file_ids, desc="Iterating over checkpoints", disable=not verbose):
        model.load_state_dict(checkpointer.load_file(file_id)["model"])
        yield model

def get_ws(
        num_ws: int, 
        pretrain_dist: RegressionSequenceDistribution[DiscreteTaskDistribution], 
        true_dist: RegressionSequenceDistribution[GaussianTaskDistribution],
        ws_source: Literal["pretrain", "true"] = "pretrain", 
):
    if ws_source == "pretrain":
        task_dist = pretrain_dist.task_distribution
        if num_ws > task_dist.num_tasks:
            raise ValueError(f"num_ws ({num_ws}) > num_tasks ({pretrain_dist.task_distribution.num_tasks})")

        task_selection = torch.randperm(task_dist.num_tasks, device=task_dist.device)[:num_ws]
        return task_dist.tasks[task_selection]
    elif ws_source == "true":
        task_dist = true_dist.task_distribution
        return task_dist.sample_tasks(num_ws)
    else:
        raise ValueError(f"Unknown ws_source: {ws_source}")

def get_xs(
        num_xs: int,
        task_config: ICLTaskConfig,
        xs_source: Literal["gaussian"] = "gaussian",
        device: Optional[str]="cpu",
):
    if xs_source == "gaussian":
        return torch.normal(
            mean=0.,
            std=1.,
            size=(num_xs, task_config.max_examples, task_config.task_size,),
            device=device,
        )
    else: 
        raise ValueError(f"Unknown xs_source: {xs_source}")


def get_ys(
        xs: torch.Tensor,
        ws: torch.Tensor,
        task_config: ICLTaskConfig,
        device: Optional[str]="cpu",
):
    return [apply_transformations(w.repeat(len(xs)), xs, task_config.noise_variance, device) for w in ws]

def separate_attention(qkv, num_heads, batch_size, head_size, num_tokens):
    return (qkv.view(batch_size, num_tokens, num_heads, 3*head_size)
            .transpose(-2, -3)
            .split(head_size, dim=-1))

def plot_matrix(ax, data, title: Optional[str] = None):
    ax.matshow(data.detach().to("cpu").numpy())
    ax.grid(None)

    if title is not None:
        ax.set_title(title)


def plot_attention(axs, data, titles, num_heads):
    q, k, softmax, v  = data
    qk = q @ k.transpose(-2, -1)

    # Rows for each head
    # Columns for Q, K, QK, V
    for j, (name, x) in enumerate(zip(titles, [q, k, qk, softmax, v])):
        axs[0, j].set_title(name)

        for h in range(num_heads):
            plot_matrix(axs[h, j], x[h])

    for h in range(num_heads):
        axs[h, 0].set_ylabel(f"Head {h}")


def plot_multi_head(axs, data, title):
    for j, x in enumerate(data):
        axs[j].set_title(title[j])
        plot_matrix(axs[j], x)
        axs[j].grid(None)



def plot_activations_one_sample(model, x, y, yhat, activations, config: ICLConfig, make_slug: Optional[Callable[[str], Path]]=None, make_title: Optional[Callable[[str], str]]=None):
    details = config.to_latex()
    task_config = config.task_config

    batch_size = 1
    num_layers = task_config.num_layers
    num_heads = task_config.num_heads
    head_size = task_config.embed_size // task_config.num_heads
    num_tokens = task_config.max_examples * 2

    locations = get_locations(L=num_layers)
    locations_to_display = get_model_locations_to_display(L=num_layers)
    locations_to_slug = get_model_locations_to_slug(L=num_layers)

    fig, ax = plt.subplots(1, 1)
    xypred = torch.cat([x[0], y[0], yhat[0]], dim=-1).T
    title = make_title("Input, Target, and Output") if make_title is not None else "Input, Target, and Output"
    slug = str(make_slug("4.0-x_y_and_pred")) if make_slug is not None else "4.0-x_y_and_pred"
    plot_matrix(ax, xypred)
    ax.set_yticklabels([None, "$x$", None, None, None, "$y$", r"$\hat y$"])
    plt.savefig(slug + ".png")
    plt.close()

    for location in locations:
        display_location = locations_to_display.get(location, location)
        title =  make_title(display_location) if make_title is not None else display_location
        slug = locations_to_slug.get(location, location)
        slug = str(make_slug(slug)) if make_slug is not None else slug

        if location.endswith("compute"):
            fig, axs = plt.subplots(3, 1, figsize=(15, 15))
            plt.suptitle(title)
            mlp = activations[location + ".0"]
            relu = activations[location + ".1"]
            output = activations[location + ".2"]

            plot_multi_head(axs, [mlp[0], relu[0], output[0]], ["MLP", "ReLU", "Output"])
        elif any(location.endswith(f"blocks.{l}") for l in range(num_layers)):
            fig, axs = plt.subplots(4, 1, figsize=(15, 15))
            plt.suptitle(title)

            ln0 = activations[location + ".layer_norms.0"]
            after_attn = activations[location + ".resid_after_attn"]
            ln1 = activations[location + ".layer_norms.1"]
            after_mlp = activations[location]

            plot_multi_head(axs, [ln0[0], after_attn[0], ln1[0], after_mlp[0]], ["Layer Norm 0", "Attention", "Layer Norm 1", "MLP"])
        
        elif location.endswith("unembedding"):
            fig, axs = plt.subplots(2, 1, figsize=(15, 15))
            plt.suptitle(title)
            ln = activations[location + ".0"]
            unembed = activations[location + ".1"]

            plot_multi_head(axs, [ln[0], unembed[0].T], ["Layer Norm", "Unembedding"])

        elif location not in activations:
            continue
        else:
            curr_activations = activations[location]
            activation_slice = curr_activations[0]

            if location.endswith("attention.attention"):
                q, k, v = separate_attention(curr_activations, num_heads=num_heads, batch_size=batch_size, head_size=head_size, num_tokens=num_tokens)
                softmax = activations[location.replace("attention.attention", "attention.attention_softmax")]
                fig, axs = plt.subplots(num_heads, 5, figsize=(15, 15))
                plt.suptitle(title)
                plot_attention(axs, (q[0], k[0], softmax[0], v[0]), ["Q", "K", "QK", "Softmax", "V"], num_heads=num_heads)

        
            elif len(activation_slice.shape) == 2:
                fig, ax = plt.subplots()
                plot_matrix(ax, activation_slice, title)
                
            elif len(activation_slice.shape) == 3:
                fig, axs = plt.subplots(1, activation_slice.shape[0], figsize=(15, 15))
                plt.suptitle(title)
                plot_multi_head(axs, activation_slice, [str(i) for i in range(activation_slice.shape[0])])
                
            else:
                raise ValueError("Unsupported number of dimensions.")

        plt.savefig(slug + ".png")
        plt.close()




def plot_activations_grids(model, xs, ys, save_path, config: ICLConfig, num_ws: int, num_xs: int, figsize=None):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    figsize = figsize or [10 * num_xs, 10 * num_ws]

    details = config.to_latex()
    task_config = config.task_config

    batch_size = xs.shape[0]
    num_layers = task_config.num_layers
    num_heads = task_config.num_heads
    head_size = task_config.embed_size // task_config.num_heads
    num_tokens = task_config.max_examples * 2

    locations = get_locations(L=num_layers)
    locations_to_display = get_model_locations_to_display(L=num_layers)
    locations_to_slug = get_model_locations_to_slug(L=num_layers)

    yhat, activations = model.run_with_cache(xs, ys)

    fig, axes = plt.subplots(num_ws, num_xs, figsize=figsize)

    for w in range(num_ws):
        axes[w, 0].set_ylabel(f"Task {w}")

        for i in range(num_xs):
            I = w * num_xs + i
            ax = axes[w, i]
            xypred = torch.cat([xs[I], ys[I], yhat[I]], dim=-1).T
            plot_matrix(ax, xypred, "Input and output")
            ax.set_yticklabels([None, "$x$", None, None, None, "$y$", r"$\hat y$"])
        
    for i in range(num_xs):
        axes[0, i].set_title(f"Input {i}")
    
    plt.savefig(save_path + "-4.0-x_y_and_pred.png")
    plt.close()

    # for i in range(batch_size):  # Assuming batch size of 4  
    for location in locations:
        display_location = locations_to_display.get(location, location)
        title = f"{display_location}\n{details}"
        slug = locations_to_slug.get(location, location)

        if location.endswith("compute"):
            mlp = activations[location + ".0"]
            relu = activations[location + ".1"]
            output = activations[location + ".2"]

            fig, axes = plt.subplots(3 * num_ws, num_xs, figsize=figsize)

            for w in range(num_ws):
                axes[3 * w, 0].set_ylabel(f"Task {w}")

                for i in range(num_xs):
                    I = w * num_xs + i
                    ax = axes[w, i]                
                    plot_multi_head(
                        [axes[3 * w + r, i] for r in range(3)],
                        [mlp[I], relu[I], output[I]], 
                        ["MLP", "ReLU", "Output"]
                    )
                
            for i in range(num_xs):
                axes[0, i].set_title(f"Input {i}")
            
            plt.suptitle(title)

        elif any(location.endswith(f"blocks.{l}") for l in range(num_layers)):
            ln0 = activations[location + ".layer_norms.0"]
            after_attn = activations[location + ".resid_after_attn"]
            ln1 = activations[location + ".layer_norms.1"]
            after_mlp = activations[location]

            fig, axes = plt.subplots(4 * num_ws, num_xs, figsize=figsize)

            for w in range(num_ws):
                axes[4 * w, 0].set_ylabel(f"Task {w}")

                for i in range(num_xs):
                    I = w * num_xs + i
                    plot_multi_head(
                        [axes[4 * w + r, i] for r in range(4)], 
                        [ln0[I], after_attn[I], ln1[I], after_mlp[I]], 
                        ["Layer Norm 0", "Attention", "Layer Norm 1", "MLP"]
                    )
                
            for i in range(num_xs):
                axes[0, i].set_title(f"Input {i}")
            
            plt.suptitle(title)
       
        elif location.endswith("unembedding"):
            ln = activations[location + ".0"]
            unembed = activations[location + ".1"]

            fig, axes = plt.subplots(2 * num_ws, num_xs, figsize=figsize)

            for w in range(num_ws):
                axes[2 * w, 0].set_ylabel(f"Task {w}")

                for i in range(num_xs):
                    I = w * num_xs + i
                    plot_multi_head(
                        [axes[2 * w, i], axes[2 * w + 1, i]], [ln[I], unembed[I].T], 
                        ["Layer Norm", "Unembedding"]
                    )

            plt.suptitle(title)

        elif location not in activations:
            continue
        
        else:
            act = activations[location]
    
            if location.endswith("attention.attention"):
                    q, k, v = separate_attention(act, num_heads=num_heads, batch_size=batch_size, head_size=head_size, num_tokens=num_tokens)
                    softmax = activations[location.replace("attention.attention", "attention.attention_softmax")]
                    fig, axes = plt.subplots(num_ws * num_heads, 5 * num_xs, figsize=figsize)
                    
                    for w in range(num_ws):
                        axes[2 * w, 0].set_ylabel(f"Task {w}")

                        for i in range(num_xs):
                            I = w * num_xs + i

                            local_axes = np.array([
                                [axes[w * num_heads + r, i * 5 + c] for c in range(5)]
                                for r in range (num_heads) 
                            ])
                            plot_attention(local_axes, (q[I], k[I], softmax[I], v[I]), ["Q", "K", "QK", "Softmax", "V"], num_heads=num_heads)

                    plt.suptitle(title)

            elif len(act[0].shape) == 2:
                fig, axes = plt.subplots(num_ws, num_xs, figsize=figsize)
                for w in range(num_ws):
                    axes[w, 0].set_ylabel(f"Task {w}")

                    for i in range(num_xs):
                        I = w * num_xs + i
                        plot_matrix(axes[w, i], act[I], title)
                
            else:
                raise ValueError("Unsupported number of dimensions.")

        plt.tight_layout()
        plt.savefig(save_path + f"-{slug}.png")
        plt.close()


def plot_activations(model, xs, ys, config: ICLConfig, num_ws: int, num_xs: int, figsize=None, make_slug: Optional[Callable[[str], Path]]=None, make_title: Optional[Callable[[str], str]]=None):
    figsize = figsize or [10 * num_xs, 10 * num_ws]

    yhats, activations = model.run_with_cache(xs, ys)

    for w in range(num_ws):
        for i in range(num_xs):
            I = w * num_xs + i
            x = xs[I].unsqueeze(0)
            y = ys[I].unsqueeze(0)
            yhat = yhats[I].unsqueeze(0)
            act = {k: v[I].unsqueeze(0) for k, v in activations.items() if v is not None}
        
            def _make_slug(slug):
                slug = f"w{w}x{i}/{slug}"
                return str(make_slug(slug) if make_slug is not None else slug)
            
            def _make_title(title):
                title = f"Task {w}, Input {i}: {title}"
                return make_title(title) if make_title is not None else title 

            plot_activations_one_sample(model, x, y, yhat, act, config, make_slug=_make_slug, make_title=_make_title)
    


def activations_over_time(
        config: ICLConfig, 
        num_ws: int = 2,
        num_xs: int = 2,
        ws_source: Literal["pretrain", "true"] = "pretrain",
        xs_source: Literal["gaussian"] = "gaussian",
        seed: Optional[int] = None,
        device: Optional[str]=None, 
        cores: Optional[int]=None, 
        steps: Optional[list] = None,
    ):
    cores = cores or int(os.environ.get("CORES", cpu_count() // 2))
    device = device or get_default_device()

    run = Run(config)
    print(run.config.to_slug(delimiter="-"))
    model = run.model
    model.train()
    checkpointer = run.checkpointer

    if seed is not None:
        torch.manual_seed(seed)

    steps = list(checkpointer.file_ids) if not steps else steps
    checkpointer.file_ids = steps  # Gross, sorry. 

    if ws_source == "pretrain" and num_ws > run.config.task_config.num_tasks:
        num_ws = run.config.task_config.num_tasks
        warnings.warn(f"num_ws ({num_ws}) > num_tasks ({run.config.task_config.num_tasks}). Setting num_ws to {num_ws}.")

    ws = get_ws(num_ws, run.pretrain_dist, run.true_dist,  ws_source=ws_source)
    xs_unique = get_xs(num_xs, run.config.task_config, xs_source=xs_source, device=device)
    ys_per_ws = get_ys(xs_unique, ws, run.config.task_config, device=device)

    xs = torch.cat([xs_unique for _ in range(num_ws)], dim=0)
    ys = torch.cat(ys_per_ws, dim=0)

    for step, model in zip(steps, iter_models(model, checkpointer)):
        print(step)
        hooked_model = hook(model)

        def make_slug(slug):
            path = FIGURES / (f"activations-{config.to_slug().replace('.', '_')}/t={step}") / slug

            if not os.path.exists(os.path.dirname(path)):
                os.makedirs(os.path.dirname(path), exist_ok=True)

            return path
        
        def make_title(title):
            return title + f" at $t={step}$\n{config.to_latex()}"


        plot_activations(hooked_model, xs, ys, run.config, num_ws=num_ws, num_xs=num_xs, make_slug=make_slug, make_title=make_title)
        

@app.command("activations")
def activations_over_time_from_cmd_line(
    sweep: str = typer.Option(None, help="Path to wandb sweep YAML file"), 
    num_tasks: int = typer.Option(None, help="Number of tasks to train on"), 
    num_layers: int = typer.Option(None, help="Number of transformer layers"), 
    num_heads: int = typer.Option(None, help="Number of transformer heads"), 
    embed_size: int = typer.Option(None, help="Embedding size"), 
    num_ws: int = typer.Option(None, help="How many tasks to examine activations on."), 
    num_xs: int = typer.Option(None, help="How many inputs to consider per task"),
    ws_source: str = typer.Option("pretrain", help="Where to sample the tasks from"), 
    xs_source: str = typer.Option("gaussian", help="Where to sample inputs from."), 
    steps: Optional[List[int]] = typer.Option(None, help="Step"),
    lr: float = typer.Option(None, help="Learning rate"),  
    seed: Optional[int] = typer.Option(None, help="Random seed"),
):
    """
    Initialise and train an InContextRegressionTransformer model, tracking
    various metrics.
    """
        
    filters = rm_none_vals(dict(task_config={"num_tasks": num_tasks, "num_layers": num_layers, "num_heads": num_heads, "embed_size": embed_size}, optimizer_config={"lr": lr}))
    analysis_config = rm_none_vals(dict(num_ws=num_ws, num_xs=num_xs, ws_source=ws_source, xs_source=xs_source,  steps=steps, seed=seed))
    config = get_unique_config(sweep, **filters)
    activations_over_time(config, **analysis_config)


@app.command("from-wandb")
def activations_over_time_from_wandb():
    """
    Initialise and train an InContextRegressionTransformer model, tracking
    various metrics.
    """
    config = get_config(project="icl", entity="devinterp")
    wandb.init(project="icl", entity="devinterp")
    config_dict = dict(wandb.config)
    analysis_config = config_dict.pop("analysis_config")
    config = get_config(**config_dict)
    wandb.run.name = f"L{config['task_config']['num_layers']}H{config['task_config']['num_heads']}M{config['task_config']['num_tasks']}"
    activations_over_time(config, **analysis_config)
    wandb.finish()

if __name__ == "__main__":
    prepare_experiments()
    app()
