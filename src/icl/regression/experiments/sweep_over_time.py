import os
import time
import warnings
from contextlib import contextmanager
from typing import List, Optional, Union

import torch
import typer
import yaml

import wandb
from icl.analysis.health import ChainHealthException
from icl.analysis.sample import SamplerConfig
from icl.analysis.utils import get_unique_config
from icl.constants import DEVICE, XLA
from icl.monitoring import stdlogger
from icl.regression.config import RegressionConfig, get_config
from icl.regression.experiments.utils import *
from icl.regression.experiments.utils import flatten_and_process
from icl.regression.train import RegressionRun
from icl.utils import prepare_experiments
from infra.monitoring import StepsConfig, expand_steps_config_, process_steps
from infra.utils.iterables import flatten_dict, rm_none_vals

WANDB_ENTITY = os.environ.get("WANDB_ENTITY")

app = typer.Typer()

if XLA:
    import torch_xla.core.xla_model as xm


StepsType = Union[List[int], StepsConfig]

from typing import List


def get_restriction_name(numbers: List[int]) -> str:
    """
    Generates a string representation of a list of numbers, indicating any consecutive sequences of numbers.

    Args:
        numbers (List[int]): A list of integers.

    Returns:
        str: A string representation of the numbers, indicating any consecutive sequences.

    Example:
        >>> get_restriction_name([1, 2, 3, 5, 6, 8, 9])
        '1-3_5-6_8-9'
    """
    
    if not numbers:
        return ""

    result = []
    start = prev = numbers[0]

    for num in numbers[1:]:
        if num != prev + 1:
            if start == prev:
                result.append(str(start))
            elif start + 1 == prev:
                result.append(str(start))
                result.append(str(prev))
            else:
                result.append(f"{start}-{prev}")
            start = num
        prev = num

    if start == prev:
        result.append(str(start))
    elif start + 1 == prev:
        result.append(str(start))
        result.append(str(prev))
    else:
        result.append(f"{start}-{prev}")

    return "_".join(result)


def sweep_over_time(
    config: RegressionConfig,
    sampler_config: dict,
    steps: Optional[StepsType] = None,
    use_wandb: bool = False,
    testing: bool = False
):     
    if testing:
        warnings.warn("Testing mode enabled")

    cores = int(os.environ.get("CORES", 1))
    device = str(DEVICE)

    if XLA:
        xm.mark_step()

    stdlogger.info("Retrieving & restoring training run...")
    start = time.perf_counter()
    config["device"] = 'cpu'
    config: RegressionConfig = get_config(**config)
    run = RegressionRun(config)
    end = time.perf_counter()
    stdlogger.info("... %s seconds", end - start)

    if XLA:
        xm.mark_step()

    start = end
    stdlogger.info("Configuring sampler...")
    
    end = time.perf_counter()
    stdlogger.info("... %s seconds", end - start)

    # Iterate over checkpoints
    num_steps = run.config.num_steps

    if steps is None:
        steps = sorted(list(run.config.checkpointer_config.checkpoint_steps))

    elif isinstance(steps, dict):
        expand_steps_config_(steps, num_steps)
        steps = sorted(list(process_steps(steps)))

    if not steps:
        raise ValueError("No checkpoints found")

    def log_fn(data, step=None):
        data = {k: v for k, v in data.items() if 'trace' not in k}
        serialized = flatten_and_process(data)
        
        if use_wandb:
            wandb.log(serialized, step=step)
    
        print(yaml.dump(serialized))

    sampler_config: SamplerConfig = SamplerConfig(**sampler_config, device=device, cores=cores)
    run.model.train()

    for i, step in enumerate(tqdm(steps, desc="Iterating over checkpoints...")):
        if not testing:
            checkpoint = run.checkpointer.load_file(step)
            run.model.load_state_dict(checkpoint['model'])
        else:
            warnings.warn("Testing mode: Skipping checkpoint loading")

        run.model.to(device)
        sampler = sampler_config.to_sampler(run)

        if i == 0 and ("*" not in sampler.config.include or sampler.config.exclude):
            sampler.restrict_(run.model)

            layer_idxs = []
            layer_names = []

            for i, (n, p) in enumerate(run.model.named_parameters()):
                if p.requires_grad:
                    layer_idxs.append(i)
                    layer_names.append(n)
            
            print("")
            print("Restricting sampler to:")
            for n in layer_names:
                print("\t", n)
            print("")

            wandb.run.name = wandb.run.name + f"-r{get_restriction_name(layer_idxs)}"
            wandb.config['restriction'] = layer_names

        try:
            results = sampler.eval(run.model)
            results['loss/init'] = sampler.init_loss.item()
            log_fn(results, step=step)
        
        except ChainHealthException as e:
            warnings.warn(f"Chain failed to converge: {e}")


@contextmanager
def wandb_context(config=None):
    wandb.init(project="icl", entity=WANDB_ENTITY)
    config = config or dict(wandb.config)
    wandb.run.name = f"L{config['task_config']['num_layers']}H{config['task_config']['num_heads']}M{config['task_config']['num_tasks']}-s{config['task_config']['model_seed']}"
    wandb.config['device'] = str(DEVICE)

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
        testing = config.pop("testing", False)
        if testing:
            wandb.run.name = "TEST-" + wandb.run.name
        sweep_over_time(config, sampler_config, steps=steps, use_wandb=True, testing=testing)


@app.command("sweep")
def cmd_line_sweep_over_time(
    sweep: str = typer.Option(None, help="Path to wandb sweep YAML file"), 
    num_tasks: int = typer.Option(None, help="Number of tasks to train on"), 
    num_layers: int = typer.Option(None, help="Number of transformer layers"), 
    num_heads: int = typer.Option(None, help="Number of transformer heads"), 
    model_seed: int = typer.Option(None, help="Model seed"),
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
    testing: bool = typer.Option(False, help="Testing mode")
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
        "model_seed": model_seed
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
    print(yaml.dump(config_dict))

    if use_wandb:
        with wandb_context(config=config_dict):
            sweep_over_time(config_dict, sampler_config, steps=steps, use_wandb=True, testing=testing)
    else:
        sweep_over_time(config_dict, sampler_config, steps=steps, use_wandb=False, testing=testing)


if __name__ == "__main__":
    prepare_experiments()
    app()

