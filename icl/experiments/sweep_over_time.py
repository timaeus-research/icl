import os
import time
import warnings
from contextlib import contextmanager
from typing import List, Optional

import torch
import typer
import yaml
from devinfra.utils.iterables import flatten_dict, rm_none_vals

import wandb
from icl.analysis.health import ChainHealthException
from icl.analysis.sample import SamplerConfig
from icl.analysis.utils import get_unique_config
from icl.config import ICLConfig, get_config
from icl.constants import DEVICE, XLA
from icl.experiments.utils import *
from icl.experiments.utils import flatten_and_process
from icl.monitoring import stdlogger
from icl.train import Run
from icl.utils import prepare_experiments

app = typer.Typer()

if XLA:
    import torch_xla.core.xla_model as xm

def sweep_over_time(
    config: ICLConfig,
    sampler_config: dict,
    steps: Optional[List[int]] = None,
    use_wandb: bool = False,
):      
    cores = int(os.environ.get("CORES", 1))
    device = str(DEVICE)

    if XLA:
        xm.mark_step()

    stdlogger.info("Retrieving & restoring training run...")
    start = time.perf_counter()
    config["device"] = DEVICE
    config: ICLConfig = get_config(**config)
    run = Run.create_and_restore(config)
    end = time.perf_counter()
    stdlogger.info("... %s seconds", end - start)

    if XLA:
        xm.mark_step()

    start = end
    stdlogger.info("Configuring sampler...")
    sampler_config: SamplerConfig = SamplerConfig(**sampler_config, device=device, cores=cores)
    sampler = sampler_config.to_sampler(run)
    end = time.perf_counter()
    stdlogger.info("... %s seconds", end - start)

    # Iterate over checkpoints
    steps = steps or list(run.checkpointer.file_ids)

    if not steps:
        raise ValueError("No checkpoints found")

    def log_fn(data, step=None):
        data = {k: v for k, v in data.items() if 'trace' not in k}
        serialized = flatten_and_process(data)
        
        if use_wandb:
            wandb.log(serialized, step=step)
    
        print(yaml.dump(serialized))

    for step, model in tqdm(zip(steps, iter_models(run.model, run.checkpointer, verbose=True)), total=len(steps), desc="Iterating over checkpoints..."):
        sampler.update_init_loss(sampler.eval_model(model, sampler.config.num_init_loss_batches, verbose=True))

        try:
            results = sampler.eval(run.model)
            log_fn(results, step=step)

            if XLA:
                xm.mark_step()
        
        except ChainHealthException as e:
            warnings.warn(f"Chain failed to converge: {e}")
       
        sampler.reset()


@contextmanager
def wandb_context(config=None):
    wandb.init(project="icl", entity="devinterp")
    config = config or dict(wandb.config)
    wandb.run.name = f"L{config['task_config']['num_layers']}H{config['task_config']['num_heads']}M{config['task_config']['num_tasks']}"
    try:
        yield config
        wandb.finish()
    except Exception as e:
        wandb.finish(1)
        raise e


@app.command("wandb")
def wandb_sweep_over_time():         
    with wandb_context() as config:
        sampler_config = config.pop("sampler_config")
        steps = config.pop("steps", None)
        sweep_over_time(config, sampler_config, steps=steps, use_wandb=True)


@app.command("sweep")
def cmd_line_sweep_over_time(
    sweep: str = typer.Option(None, help="Path to wandb sweep YAML file"), 
    num_tasks: int = typer.Option(None, help="Number of tasks to train on"), 
    num_layers: int = typer.Option(None, help="Number of transformer layers"), 
    num_heads: int = typer.Option(None, help="Number of transformer heads"), 
    embed_size: int = typer.Option(None, help="Embedding size"), 
    gamma: float = typer.Option(None, help="Localization strength"), 
    epsilon: float = typer.Option(None, help="SGLD step size"),
    temperature: float = typer.Option(None, help="Sampling temperature"),
    gradient_scale: float = typer.Option(0.05, help="Gradient scale"), 
    noise_scale: float = typer.Option(0.0003, help="Noise scale"),
    localization_scale: float = typer.Option(0.00015, help="Localization scale"),
    lr: float = typer.Option(None, help="Learning rate"), 
    num_draws: int = typer.Option(None, help="Number of draws"), 
    num_chains: int = typer.Option(None, help="Number of chains"), 
    steps: Optional[List[int]] = typer.Option(None, help="Step"), 
    batch_size: Optional[int] = typer.Option(None, help="Batch size"),
    use_wandb: bool = typer.Option(True, help="Use wandb"),
):
    """
    Initialise and train an InContextRegressionTransformer model, tracking
    various metrics.
    """
        
    filters = rm_none_vals(dict(task_config={
        "num_tasks": num_tasks, 
        "num_layers": num_layers, 
        "num_heads": num_heads,
        "embed_size": embed_size, 
    }, optimizer_config={"lr": lr}))
    sampler_config = rm_none_vals(dict(
        gamma=gamma, 
        lr=epsilon,
        num_draws=num_draws,
        num_chains=num_chains, 
        batch_size=batch_size, 
        temperature=temperature,
        gradient_scale=gradient_scale,
        noise_scale=noise_scale,
        localization_scale=localization_scale,
        eval_metrics=['likelihood-derived', 'hessian']
    ))
    config = get_unique_config(sweep, **filters)
    config_dict = config.model_dump()

    if use_wandb:
        with wandb_context(config=config_dict):
            sweep_over_time(config_dict, sampler_config, steps=steps, use_wandb=True)
    else:
        sweep_over_time(config_dict, sampler_config, steps=steps)


if __name__ == "__main__":
    prepare_experiments()
    app()

