import itertools
import logging
import os
import warnings
from pprint import pp
from typing import List, Optional, TypeVar

import sentry_sdk
import torch
import typer
import yaml
from devinterp.evals import RepeatEvaluator
from devinterp.optim.schedulers import LRScheduler
from devinterp.utils import flatten_dict
from dotenv import load_dotenv
from pydantic import BaseModel
from torch import nn
from torch.nn import functional as F

import wandb
from icl.analysis.rlct import eval_rlcts_over_run, make_rlct_evaluator
from icl.config import ICLConfig, get_config
from icl.sample import estimate_rlct
from icl.train import Run
from icl.utils import filter_objs, find_obj, rm_none_vals, unflatten_dict

app = typer.Typer()


def _wandb_config_expansion(parameters, prefix="", sep="/"):
    """Recursive function to expand nested parameters."""
    keys = list(parameters.keys())
    for key in keys:
        if "parameters" in parameters[key]:
            yield from _wandb_config_expansion(
                parameters[key]["parameters"], prefix=f"{prefix}{key}{sep}"
            )
        else:
            if "values" in parameters[key]:
                yield (f"{prefix}{key}", parameters[key]["values"])
            else:
                yield (f"{prefix}{key}", [parameters[key]["value"]])


def expand_grid(params):
    """Generates a list of dicts for each run config based on the grid of parameter values."""
    keys, value_sets = zip(*params)
    for values in itertools.product(*value_sets):
        yield dict(zip(keys, values))


def generate_config_dicts(sweep_config: dict, **kwargs):
    """Turns a wandb sweep config into a list of configs for each run defined in that sweep. (Assumes strategy is grid)"""
    params = list(_wandb_config_expansion(sweep_config["parameters"]))
    kwargs = flatten_dict(kwargs, delimiter="/")
    
    for config_dict in expand_grid(params):
        _kwargs = kwargs.copy()
        _kwargs.update(config_dict)
        
        yield unflatten_dict(_kwargs, delimiter="/")


def generate_config_dicts_from_path(file_path: str, **kwargs):
    """Load the ICLConfigs for each of the runs defined in a wandb sweep config at the specified file path."""
    with open(file_path, "r") as f:
        config = yaml.safe_load(f)

    yield from generate_config_dicts(config, **kwargs)


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

    eval_rlcts = make_rlct_evaluator(
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
            eval_rlcts_over_run(config, analysis_config)
    else:
        config = get_config(project="icl", entity="devinterp", extra="rlct")  # Replace as needed
        analysis_config = wandb.config["analysis_config"]
        eval_rlcts_over_run(config, analysis_config)


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
