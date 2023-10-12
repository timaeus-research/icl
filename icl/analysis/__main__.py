import os

from dotenv import load_dotenv

from icl.analysis.cov import WithinHeadCovarianceCallback

load_dotenv()


import itertools
import logging
import warnings
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
from logging import Logger
from pathlib import Path
from pprint import pp
from typing import (Callable, Dict, Iterable, List, Literal, Optional, Tuple,
                    Type, TypeVar, Union)

import devinfra
import devinterp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sentry_sdk
import torch
import typer
import yaml
from devinfra.evals import Criterion
from devinfra.utils.device import get_default_device
from devinterp.optim.sgld import SGLD
from devinterp.slt.learning_coeff import plot_learning_coeff_trace
from pydantic import BaseModel
from scipy.sparse.linalg import eigsh
from torch import nn
from torch.multiprocessing import (Pool, cpu_count, get_context,
                                   set_start_method)
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from typer import Typer

import wandb
from icl.analysis.llc import make_slt_evals, map_slt_evals_over_run
from icl.analysis.utils import (generate_config_dicts_from_path,
                                get_unique_run, split_attn_weights)
from icl.config import ICLConfig, get_config
from icl.train import Run

sns.set_theme(style="whitegrid")


FIGURES=Path("figures")
ANALYSIS = Path("analysis")

assert os.path.exists(FIGURES)
assert os.path.exists(ANALYSIS)

DEVICE = devinfra.utils.device.get_default_device()
K=3  # Num cov components

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
    set_start_method('spawn')  # Required for sharing CUDA tensors
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
