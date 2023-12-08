from typing import Any, List, Literal, Optional, Union

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
from torch import nn
from torch.nn import functional as F

import wandb
from icl.analysis.cov import make_transformer_cov_accumulator
from icl.analysis.sample import SamplerConfig, estimate_slt_observables, sample
from icl.analysis.slt import (LikelihoodMetricsEstimator,
                              SLTObservablesEstimator)
from icl.config import ICLConfig, get_config
from icl.evals import ICLEvaluator
from icl.experiments.utils import *
from icl.train import Run
from icl.utils import pyvar_dict_to_latex, pyvar_dict_to_slug

app = typer.Typer()


def estimate_at_checkpoint(
    config: dict,
    sampler_config: dict,
    checkpoint_idx: int,
):      
    cores = int(os.environ.get("CORES", 1))
    device = str(get_default_device())

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


if __name__ == "__main__":
    prepare_experiments()
    app()
    


