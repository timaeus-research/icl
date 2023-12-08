import os
from typing import Optional

import torch
import typer
from devinfra.utils.device import get_default_device
from devinfra.utils.iterables import rm_none_vals

import wandb
from icl.analysis.sample import SamplerConfig
from icl.analysis.utils import get_unique_config
from icl.config import ICLConfig, get_config
from icl.experiments.utils import *
from icl.train import Run
from icl.utils import pyvar_dict_to_slug

app = typer.Typer()


def estimate_at_checkpoint(
    config: dict,
    sampler_config: dict,
    checkpoint_idx: int,
):      
    cores = int(os.environ.get("CORES", 1))
    device = str(get_default_device())

    config["device"] = device
    config: ICLConfig = get_config(**config)
    run = Run.create_and_restore(config)

    checkpoint_step = run.checkpointer.file_ids[checkpoint_idx]

    if checkpoint_step != -1:
        checkpoint = run.checkpointer.load_file(checkpoint_step)
        run.model.load_state_dict(checkpoint["model"])
        run.optimizer.load_state_dict(checkpoint["optimizer"])
        run.scheduler.load_state_dict(checkpoint["scheduler"])

    sampler_config: SamplerConfig = SamplerConfig(**sampler_config, device=device, cores=cores)
    sampler = sampler_config.to_sampler(run)
    results = sampler.eval(run.model)

    # Save to wandb
    wandb.log(results)

    # Save locally
    results["config"] = {
        "run": config.model_dump(),
        "sampler": sampler_config.model_dump(),
    }

    slug = "llc-" + pyvar_dict_to_slug(results["config"]) + f"@t={checkpoint_step}" + ".pt"
    torch.save(results, ANALYSIS / slug)


@app.command("wandb")
def wandb_sweep_over_final_weights():      
    wandb.init(project="icl-llc", entity="devinterp")
    print("Initialized wandb")
    config = dict(wandb.config)
    sampler_config = config.pop("sampler_config")
    checkpoint_idx = config.pop("checkpoint_idx", -1)
    title_config = sampler_config.copy()
    del title_config["num_draws"]
    del title_config["num_chains"]
    del title_config["batch_size"]
    wandb.run.name = f"L{config['task_config']['num_layers']}H{config['task_config']['num_heads']}M{config['task_config']['num_tasks']}:{pyvar_dict_to_slug(title_config)}"
    wandb.run.save()
    estimate_at_checkpoint(config, sampler_config, checkpoint_idx)
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
    checkpoint_idx: int = typer.Option(-1, help="Checkpoint index")
):
    """
    Initialise and train an InContextRegressionTransformer model, tracking
    various metrics.
    """
        
    filters = rm_none_vals(dict(task_config={"num_tasks": num_tasks, "num_layers": num_layers, "num_heads": num_heads, "embed_size": embed_size}, optimizer_config={"lr": lr}))
    sampler_config = rm_none_vals(dict(gamma=gamma, lr=epsilon, num_draws=num_draws, num_chains=num_chains, batch_size=batch_size))
    config = get_unique_config(sweep, **filters)
    estimate_at_checkpoint(config.model_dump(), sampler_config, checkpoint_idx=checkpoint_idx)


if __name__ == "__main__":
    prepare_experiments()
    app()
    


