from typing import Any, List, Literal, Optional, Union

from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import typer
from devinfra.utils.device import get_default_device
from devinterp.optim.sgld import SGLD
from devinterp.optim.sgnht import SGNHT
from pydantic import BaseModel, Field, field_validator, model_validator
from torch.nn import functional as F

import wandb
from icl.analysis.cov import make_transformer_cov_accumulator
from icl.analysis.sample import estimate_slt_observables, sample
from icl.analysis.slt import LikelihoodMetricsEstimator, SLTObservablesEstimator
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
    device = str(get_default_device())

    config: ICLConfig = get_config(**config)
    run = Run.create_and_restore(config)

    sampler_config: SamplerConfig = SamplerConfig(**sampler_config, device=device, cores=cores)
    sampler = sampler_config.to_sampler(run)
    results = sampler.eval(run.model)

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
    


