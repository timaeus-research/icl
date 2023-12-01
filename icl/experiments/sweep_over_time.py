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
from icl.analysis.sample import make_slt_evals
from icl.analysis.utils import (get_sweep_configs, get_unique_config,
                                split_attn_weights)
from icl.config import ICLConfig, get_config
from icl.evals import ICLEvaluator
from icl.experiments.utils import *
from icl.train import Run

app = typer.Typer()


def sweep_over_time(
    config: ICLConfig,
    sampler_config: dict,
    steps: Optional[List[int]] = None,
):      
    cores = int(os.environ.get("CORES", 1))
    device = get_default_device()
    run = Run.create_and_restore(config)
    
    # Dataset for SGLD
    num_samples = config.eval_batch_size
    batch_size = sampler_config.pop("batch_size", 1024)
    eff_num_samples = sampler_config.pop("num_samples", batch_size)  # For configuring SGLD and estimating Lambdahat

    run.evaluator = ICLEvaluator(
        pretrain_dist=run.pretrain_dist,
        true_dist=run.true_dist,
        max_examples=config.task_config.max_examples,
        eval_batch_size=num_samples, 
        seed=config.task_config.true_seed,
    )

    xs, ys = run.evaluator.pretrain_xs, run.evaluator.pretrain_ys
    dataset = torch.utils.data.TensorDataset(xs, ys)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)  # Shuffle might meant repeats

    # Hyperparameters for posterior sampling
    num_chains = sampler_config.pop("num_chains", 25)
    num_draws = sampler_config.pop("num_draws", 1000)

    # Covariance estimation
    callbacks = []
    num_evals = sampler_config.pop("num_evals", 0)
    if num_evals > 0:
        cov_accumulator = make_transformer_cov_accumulator(run.model, device=device, num_evals=num_evals)
        callbacks.append(cov_accumulator)

    # Sample observables
    slt_evals = make_slt_evals(
        cores=cores,
        device=device,
        dataset=dataset,
        loader=loader,
        num_draws=num_draws,
        num_chains=num_chains,
        callbacks=callbacks,
        # Sampling
        lr=sampler_config.pop("lr", 1e-4),
        elasticity=sampler_config.pop("gamma", 1.),
        num_samples=eff_num_samples,
        **sampler_config
    )

    # Iterate over checkpoints
    steps = steps or list(run.checkpointer.file_ids)

    for step, model in zip(steps, iter_models(run.model, run.checkpointer)):
        print(step)
        results = slt_evals(model)

        if num_evals > 0:
            trace = results.pop("loss/trace")
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
                    results[f"cov_eval_{i}/{obs_name}"] = evals[i]

                # principal_evals[name] = evals[0]
                # principal_evecs[name] = evecs[:, 0]

            # slug = FIGURES / (f"cov-{config.to_slug()}@t={step}".replace(".", "_"))
            # title = f"Principal covariance eigenvalues\n{config.to_latex()}"

            # plot_evecs(evals=principal_evals, evecs=principal_evecs, shapes=original_shapes, title=title, save=slug)
            # logger.log(observables, step=step)
            cov_accumulator.reset()

        # Save to wandb
        wandb.log(results, step=step)
       

@app.command("wandb")
def wandb_sweep_over_time():      
    wandb.init(project="icl", entity="devinterp")
    print("Initialized wandb")
    config = dict(wandb.config)
    sampler_config = config.pop("analysis_config")
    wandb.run.name = f"L{config['task_config']['num_layers']}H{config['task_config']['num_heads']}M{config['task_config']['num_tasks']}"
    wandb.run.save()
    sweep_over_time(get_config(**config), sampler_config)
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

