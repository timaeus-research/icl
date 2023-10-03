import logging
import os
import warnings
from pprint import pp
from typing import List, Optional, TypeVar

import sentry_sdk
import torch
import typer
from devinterp.evals import RepeatEvaluator
from devinterp.optim.schedulers import LRScheduler
from dotenv import load_dotenv
from pydantic import BaseModel
from torch import nn
from torch.nn import functional as F

import wandb
from icl.analysis.rlct import make_slt_evals, map_slt_evals_over_run
from icl.analysis.sample import estimate_rlct
from icl.analysis.utils import generate_config_dicts_from_path
from icl.config import ICLConfig, get_config
from icl.train import Run
from icl.utils import find_obj, find_unique_obj, rm_none_vals

app = typer.Typer()




@app.command("run")
def rlcts_over_run(
    sweep: str = typer.Argument(..., help="Path to sweep config file"),
    run_name: str = typer.Argument(..., help="Name of run to evaluate"),
):
    """Find the RLCT configuration for a given sweep."""
    config_dicts = list(generate_config_dicts_from_path(sweep))
    config_dict = find_obj(config_dicts, run_name=run_name) 
    config = get_config(**config_dict)
    analysis_config = config_dict.get("analysis_config", {})  # Replace this line as appropriate
    run = Run.create_and_restore(config)
    pp(run.evaluator(run.model))

    xs, ys = run.evaluator.pretrain_xs, run.evaluator.pretrain_ys
    trainset = torch.utils.data.TensorDataset(xs, ys)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(xs))

    eval_rlcts = make_slt_evals(
        dataset=trainset,
        loader=trainloader,
        **analysis_config
    )

    evals = eval_rlcts(run.model)
    pp(evals)


@app.command("sweep")
def rlcts_over_sweep(sweep: str = typer.Option(None, help="Path to wandb sweep YAML file")):
    """
    Estimate RLCTs for each checkpoint for each run in a wandb sweep.
    """
    if sweep:
        for config_dict in generate_config_dicts_from_path(sweep, extra="rlct"):
            analysis_config = config_dict.pop("analysis_config")
            config = get_config(**config_dict)
            map_slt_evals_over_run(config, analysis_config)
    else:
        config = get_config(project="icl", entity="devinterp", extra="rlct")  # Replace as needed
        analysis_config = wandb.config["analysis_config"]
        map_slt_evals_over_run(config, analysis_config)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    load_dotenv()
    sentry_sdk.init(
        dsn="https://92ea29f1e366cda4681fb10273e6c2a7@o4505805155074048.ingest.sentry.io/4505805162479616",
        # Set traces_sample_rate to 1.0 to capture 100%
        # of transactions for performance monitoring.
        # We recommend adjusting this value in production.
        traces_sample_rate=1.0,
        # Set profiles_sample_rate to 1.0 to profile 100%
        # of sampled transactions.
        # We recommend adjusting this value in production.
        profiles_sample_rate=1.0,
    )
    app()
