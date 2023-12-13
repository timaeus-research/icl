import os
import warnings
from typing import Optional

import torch
import typer
import yaml
from devinfra.utils.device import get_default_device
from devinfra.utils.iterables import flatten_dict, rm_none_vals
from pydantic import BaseModel

import wandb
from icl.analysis.health import ChainHealthException
from icl.analysis.sample import SamplerConfig
from icl.analysis.utils import get_unique_config
from icl.config import ICLConfig, get_config
from icl.experiments.utils import *
from icl.figures.plotting import plot_loss_trace, plot_weights_trace
from icl.train import Run
from icl.utils import pyvar_dict_to_slug

app = typer.Typer()

class PlottingConfig(BaseModel):
    include_loss_trace: bool = True
    include_weights_pca: bool = True

    # Weights pca
    num_components: int = 3
    num_points: int = 10


def estimate_at_checkpoint(
    config: dict,
    sampler_config: dict,
    plotting_config: dict,
    checkpoint_idx: Optional[int] = None,
    step: Optional[int] = None,
):      
    assert step is None or checkpoint_idx is None, "Can only specify one of step or checkpoint_idx"

    cores = int(os.environ.get("CORES", 1))
    device = str(get_default_device())

    config["device"] = device
    config: ICLConfig = get_config(**config)
    run = Run.create_and_restore(config)

    if step is not None:
        checkpoint_idx = run.checkpointer.file_ids.index(step)
    
    step = run.checkpointer.file_ids[checkpoint_idx]

    if step != -1:
        checkpoint = run.checkpointer.load_file(step)
        run.model.load_state_dict(checkpoint["model"])
        run.optimizer.load_state_dict(checkpoint["optimizer"])
        run.scheduler.load_state_dict(checkpoint["scheduler"])

    sampler_config: SamplerConfig = SamplerConfig(**sampler_config, device=device, cores=cores)

    print(yaml.dump(sampler_config.model_dump()))

    def log_fn(data, step=None):
        serialized = {
            k: v.item() if isinstance(v, torch.Tensor) else v
            for k, v in data.items()
        }

        wandb.log(serialized, step=step)
        print(yaml.dump(serialized))

    sampler = sampler_config.to_sampler(run, log_fn=log_fn)

    try:
        results = sampler.eval(run.model)
    except ChainHealthException as e:
        warnings.warn(f"Chain failed to converge: {e}")
        wandb.log({"error": e.message})
        wandb.finish(0)  # Mark it as a success so the sweep continues
        return

    plotting_config: PlottingConfig = PlottingConfig(**plotting_config)

    if plotting_config.include_loss_trace:
        batch_losses = sampler.batch_loss.estimates()
        likelihoods = sampler.likelihood.estimates()

        fig = plot_loss_trace(batch_losses, likelihoods)
        wandb.log({"loss_trace": wandb.Image(fig)})

    if plotting_config.include_weights_pca:
        xs, ys = run.evaluator.pretrain_xs, run.evaluator.pretrain_ys
        fig = plot_weights_trace(run.model, sampler.weights.deltas(), xs, ys, device=device, **plotting_config.dict())
        wandb.log({"weights_trace": wandb.Image(fig)}) # TODO: Alternative plotly figure

    # Save locally
    results["config"] = {
        "run": config.model_dump(),
        "sampler": sampler_config.model_dump(),
    }

    slug = "llc-" + pyvar_dict_to_slug(flatten_dict(results["config"]['run'], delimiter='_')) + pyvar_dict_to_slug(flatten_dict(results["config"]['sampler'], delimiter='_')) + f"@t={step}" + ".pt"

    torch.save(results, ANALYSIS / slug)


@app.command("wandb")
def wandb_sweep_over_final_weights():      
    wandb.init(project="icl-llc", entity="devinterp")
    print("Initialized wandb")
    config = dict(wandb.config)
    sampler_config = config.pop("sampler_config")
    checkpoint_idx = config.pop("checkpoint_idx", None)
    step = config.pop("step", None)
    plotting_config = config.pop("plotting_config", {})
    title_config = {k: v for k, v in sampler_config.items() if k in ['epsilon', 'gamma', 'eval_method', 'eval_loss_fn']}
    wandb.run.name = f"M={config['task_config']['num_tasks']}:{pyvar_dict_to_slug(title_config)}"
    wandb.run.save()
    estimate_at_checkpoint(config, sampler_config, plotting_config, checkpoint_idx=checkpoint_idx, step=step)
    wandb.finish()


@app.command("estimate")
def cmd_line_sweep_over_final_weights(
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
    batch_size: Optional[int] = typer.Option(None, help="Batch size"),
    checkpoint_idx: int = typer.Option(-1, help="Checkpoint index"),
    step: Optional[int] = typer.Option(None, help="Step"),
):
    """
    Initialise and train an InContextRegressionTransformer model, tracking
    various metrics.
    """
        
    filters = rm_none_vals(dict(task_config={"num_tasks": num_tasks, "num_layers": num_layers, "num_heads": num_heads, "embed_size": embed_size}, optimizer_config={"lr": lr}))
    sampler_config = rm_none_vals(dict(gamma=gamma, lr=epsilon, num_draws=num_draws, num_chains=num_chains, batch_size=batch_size))
    config = get_unique_config(sweep, **filters)
    estimate_at_checkpoint(config.model_dump(), sampler_config, checkpoint_idx=checkpoint_idx, step=step)


if __name__ == "__main__":
    prepare_experiments()
    app()
    


