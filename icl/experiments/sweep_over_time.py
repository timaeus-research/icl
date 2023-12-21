import os
import warnings
from typing import List, Optional

import torch
import typer
from devinfra.utils.iterables import flatten_dict, rm_none_vals

import wandb
from icl.analysis.health import ChainHealthException
from icl.analysis.sample import SamplerConfig
from icl.analysis.utils import get_unique_config
from icl.config import ICLConfig, get_config
from icl.experiments.utils import *
from icl.setup import DEVICE
from icl.train import Run

app = typer.Typer()


def sweep_over_time(
    config: ICLConfig,
    sampler_config: dict,
    steps: Optional[List[int]] = None,
    use_wandb: bool = False,
):      
    cores = int(os.environ.get("CORES", 1))
    device = str(DEVICE)

    config["device"] = device
    config: ICLConfig = get_config(**config)
    run = Run.create_and_restore(config)

    sampler_config: SamplerConfig = SamplerConfig(**sampler_config, device=device, cores=cores)
    sampler = sampler_config.to_sampler(run)

    # Iterate over checkpoints
    steps = steps or list(run.checkpointer.file_ids)

    def log_fn(data, step=None):
        def process_tensor(a):

            if len(a.shape) == 0 or a.shape == (1,):
                return a.item()
            return a.tolist()

        serialized = flatten_dict({
            k: process_tensor(v) if isinstance(v, torch.Tensor)  else v
            for k, v in data.items()
        }, flatten_lists=True)
        
        wandb.log(serialized, step=step)


    for step, model in zip(steps, iter_models(run.model, run.checkpointer, verbose=True)):
        sampler.update_init_loss(sampler.eval_model(model))
        print(step)
        sampler.reset()

        try:
            results = sampler.eval(run.model)

            if use_wandb:
                log_fn(results, step=step)
        
        except ChainHealthException as e:
            warnings.warn(f"Chain failed to converge: {e}")
       

@app.command("wandb")
def wandb_sweep_over_time():      
    wandb.init(project="icl", entity="devinterp")
    print("Initialized wandb")
    config = dict(wandb.config)
    sampler_config = config.pop("sampler_config")
    wandb.run.name = f"L{config['task_config']['num_layers']}H{config['task_config']['num_heads']}M{config['task_config']['num_tasks']}"
    wandb.run.save()
    sweep_over_time(get_config(**config), sampler_config, use_wandb=True)
    wandb.finish()


@app.command("sweep")
def cmd_line_sweep_over_time(
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
    batch_size: Optional[int] = typer.Option(None, help="Batch size")
):
    """
    Initialise and train an InContextRegressionTransformer model, tracking
    various metrics.
    """
        
    filters = rm_none_vals(dict(task_config={"num_tasks": num_tasks, "num_layers": num_layers, "num_heads": num_heads, "embed_size": embed_size}, optimizer_config={"lr": lr}))
    analysis_config = rm_none_vals(dict(gamma=gamma, lr=epsilon, num_draws=num_draws, num_chains=num_chains, batch_size=batch_size))
    config = get_unique_config(sweep, **filters)
    sweep_over_time(config, analysis_config, steps=steps)


if __name__ == "__main__":
    prepare_experiments()
    app()

