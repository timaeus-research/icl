
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import typer
from devinfra.utils.device import get_default_device
from devinterp.optim.sgld import SGLD
from devinterp.optim.sgnht import SGNHT
from torch.nn import functional as F

import wandb
from icl.analysis.cov import make_transformer_cov_accumulator
from icl.analysis.llc import ObservedOnlineLLCEstimator
from icl.analysis.sample import estimate_slt_observables
from icl.config import ICLConfig, get_config
from icl.evals import ICLEvaluator
from icl.experiments.utils import *
from icl.train import Run
from icl.utils import pyvar_dict_to_latex, pyvar_dict_to_slug

app = typer.Typer()


def sweep_over_final_weights(
    config: dict,
    sampler_config: dict,
):      
    cores = int(os.environ.get("CORES", 1))
    device = get_default_device()

    config: ICLConfig = get_config(**config)
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
    sampling_method = sampler_config.pop("sampling_method", "sgld")

    if sampling_method == "sgld":
        optimizer_class = SGLD
    elif sampling_method == "sgnht":
        optimizer_class = SGNHT
    else:
        raise ValueError(f"Unknown sampling method: {sampling_method}")

    # Covariance estimation
    callbacks = []
    num_evals = sampler_config.pop("num_evals", 0)
    if num_evals > 0:
        cov_accumulator = make_transformer_cov_accumulator(run.model, device=device, num_evals=num_evals)
        callbacks.append(cov_accumulator)

    # Sample observables
    results = estimate_slt_observables(
        run.model,
        loader,
        F.mse_loss,
        optimizer_class,
        optimizer_kwargs=dict(
            **sampler_config,
            batch_size=batch_size,
            num_samples=eff_num_samples,
        ),
        num_draws=num_draws,
        num_chains=num_chains,
        cores=cores,
        device=device,
        callbacks=callbacks,
        online="observed"
    )

    # Save to wandb
    wandb.log(results)

    # Save locally
    results["config"] = config
    results["sampler_config"] = sampler_config
    slug = "llc-" + pyvar_dict_to_slug({
        "num_layers": config.task_config.num_layers,
        "num_heads": config.task_config.num_heads,
        "num_tasks": config.task_config.num_tasks,
        "num_draws": num_draws,
        **sampler_config,
    }) + ".pt"

    torch.save(results, ANALYSIS / slug)

       

@app.command("wandb")
def wandb_sweep_over_final_weights():      
    wandb.init(project="icl-llc", entity="devinterp")
    print("Initialized wandb")
    config = dict(wandb.config)
    sampler_config = config.pop("analysis_config")
    title_config = sampler_config.copy()
    del title_config["num_draws"]
    del title_config["num_chains"]
    del title_config["batch_size"]
    wandb.run.name = f"L{config['task_config']['num_layers']}H{config['task_config']['num_heads']}M{config['task_config']['num_tasks']}:{pyvar_dict_to_slug(title_config)}"
    wandb.run.save()
    sweep_over_final_weights(config, sampler_config)
    wandb.finish()


if __name__ == "__main__":
    prepare_experiments()
    app()
    


