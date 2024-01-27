import os
import time
import warnings
from typing import Optional

import torch
import typer
import yaml
from devinfra.utils.iterables import flatten_dict, rm_none_vals
from pydantic import BaseModel

import wandb
from icl.analysis.health import ChainHealthException
from icl.analysis.sample import SamplerConfig
from icl.analysis.utils import get_unique_config
from icl.config import ICLConfig, get_config
from icl.constants import ANALYSIS, DEVICE, FIGURES, XLA
from icl.experiments.utils import *
from icl.figures.notation import pyvar_dict_to_slug
from icl.figures.plotting import plot_loss_trace, plot_weights_trace
from icl.monitoring import stdlogger
from icl.train import Run
from icl.utils import prepare_experiments

if XLA:
    import torch_xla.core.xla_model as xm

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
    use_wandb: bool = False
):      
    assert step is None or checkpoint_idx is None, "Can only specify one of step or checkpoint_idx"

    cores = int(os.environ.get("CORES", 1))
    device = str(DEVICE)

    if XLA:
        xm.mark_step()

    stdlogger.info("Retrieving & restoring training run...")
    start = time.perf_counter()
    config["device"] = DEVICE
    config: ICLConfig = get_config(**config)
    run = Run.create_and_restore(config)

    # Iterate over checkpoints
    steps = list(run.checkpointer.file_ids)

    if not steps:
        raise ValueError("No checkpoints found")

    if step is not None:
        checkpoint_idx = steps.index(step)
    
    step = steps[checkpoint_idx]

    if step != steps[-1]:
        checkpoint = run.checkpointer.load_file(step)
        run.model.load_state_dict(checkpoint["model"])
        run.optimizer.load_state_dict(checkpoint["optimizer"])
        run.scheduler.load_state_dict(checkpoint["scheduler"])

    end = time.perf_counter()
    stdlogger.info("... %s seconds", end - start)

    if XLA:
        xm.mark_step()

    def log_fn(data, step=None, figure=None):
        if figure:
            if use_wandb:
                wandb.log({data: wandb.Image(figure)}, step=step)
            else:
                path = FIGURES / f"{data}-{run.config.to_slug()}@t={step}.png"
                figure.savefig(path, dpi=300)
                
        else:
            serialized = flatten_and_process(data)
            
            if use_wandb:
                wandb.log(serialized, step=step)
        
            print(yaml.dump(serialized))

    start = end
    stdlogger.info("Configuring sampler...")
    sampler_config: SamplerConfig = SamplerConfig(**sampler_config, device=device, cores=cores)
    sampler = sampler_config.to_sampler(run, log_fn=log_fn)
    end = time.perf_counter()
    stdlogger.info("... %s seconds", end - start)

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

        fig = plot_loss_trace(batch_losses, likelihoods, title=f"Loss Trace\n{sampler_config.to_latex()}\n{run.config.to_latex()[:-1]}, t={step+1}$")
        log_fn("loss_trace", figure=fig)

    if plotting_config.include_weights_pca:
        xs, ys = run.evaluator.pretrain_xs, run.evaluator.pretrain_ys
        fig = plot_weights_trace(run.model, sampler.weights.deltas(), xs, ys, device=device, num_components=plotting_config.num_components, num_points=plotting_config.num_points)
        log_fn("weights_trace", figure=fig)

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
    stdlogger.info("Initialized wandb")
    config = dict(wandb.config)
    sampler_config = config.pop("sampler_config")
    checkpoint_idx = config.pop("checkpoint_idx", None)
    step = config.pop("step", None)
    plotting_config = config.pop("plotting_config", {})
    title_config = {k: v for k, v in sampler_config.items() if k in ['epsilon', 'gamma', 'eval_method', 'eval_loss_fn']}
    wandb.run.name = f"M={config['task_config']['num_tasks']}:{pyvar_dict_to_slug(title_config)}"
    wandb.run.save()
    estimate_at_checkpoint(config, sampler_config, plotting_config, checkpoint_idx=checkpoint_idx, step=step, use_wandb=True)
    wandb.finish()


@app.command("estimate")
def cmd_line_sweep_over_final_weights(
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
    batch_size: Optional[int] = typer.Option(None, help="Batch size"),
    checkpoint_idx: int = typer.Option(-1, help="Checkpoint index"),
    step: Optional[int] = typer.Option(None, help="Step"),
    use_wandb: bool = typer.Option(True, help="Use wandb"),
):
    """
    Initialise and train an InContextRegressionTransformer model, tracking
    various metrics.
    """
        
    filters = rm_none_vals(dict(task_config={"num_tasks": num_tasks, "num_layers": num_layers, "num_heads": num_heads, "embed_size": embed_size}, optimizer_config={"lr": lr}))
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
        eval_metrics=['likelihood-derived', 'batch-loss'], # 'weights'
        per_token=False,
        eval_online=True
    ))   
    config = get_unique_config(sweep, **filters)
    plotting_config = {
        "include_loss_trace": True,
        "include_weights_pca": False,
        "num_components": 3,
        "num_points": 10,
    }
    estimate_at_checkpoint(config.model_dump(), sampler_config, plotting_config=plotting_config, checkpoint_idx=checkpoint_idx, step=step, use_wandb=use_wandb)


if __name__ == "__main__":
    prepare_experiments()
    app()
    


