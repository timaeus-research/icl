import itertools
import os
import warnings
from pprint import pp
from typing import List, Literal, Optional, TypeVar

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
from icl.config import ICLConfig, get_config
from icl.sample import estimate_rlct
from icl.train import Run


def make_rlct_evaluator(
    dataset: torch.utils.data.Dataset,
    loader: torch.utils.data.DataLoader,
    lr: float = 1e-4,
    noise_level: float = 1.0,
    weight_decay: float = 0.0,
    elasticity: float = 1.0,
    num_draws: int = 10,
    num_chains: int = 5,
    num_burnin_steps: int = 0,
    num_steps_bw_draws: int = 1,
    cores: int = 1,
    num_repeats: int = 5,      
):
    def eval_rlct(model: nn.Module):
        optimizer_kwargs = dict(
            lr=lr,
            noise_level=noise_level,
            weight_decay=weight_decay,
            elasticity=elasticity,
            temperature="adaptive",
            num_samples=len(dataset),
        )
        return {
            "rlct": estimate_rlct(
                model,
                loader,
                F.mse_loss,
                "sgld",
                optimizer_kwargs,
                num_draws=num_draws,
                num_chains=num_chains,
                num_burnin_steps=num_burnin_steps,
                num_steps_bw_draws=num_steps_bw_draws,
                cores=cores,
                pbar=True,
                device="mps",
            )
        }

    return RepeatEvaluator(eval_rlct, num_repeats)


def eval_rlcts_over_run(config: ICLConfig, analysis_config: dict={}):
    run = Run(config)
    print(run.checkpointer)
    print(run.checkpointer.providers[0].file_ids)

    # Print the config for debugging
    total_length = 80 

    print("\n")
    print(f"Run {config.run_name}".center(total_length, "="))
    print(f"Checkpoints available: {run.checkpointer.file_ids}")
    
    file_ids = run.checkpointer.file_ids
    if "checkpoints" in analysis_config:
        file_ids = [file_ids[i] for i in analysis_config.pop("checkpoints")]

    print("\n")
    print("Config".center(total_length, "-"))
    print(yaml.dump(config.model_dump()))

    print("RLCTs".center(total_length, "-"))
    print(yaml.dump(analysis_config))

    xs, ys = run.evaluator.pretrain_xs, run.evaluator.pretrain_ys
    trainset = torch.utils.data.TensorDataset(xs, ys)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=len(xs))
    eval_rlcts = make_rlct_evaluator(
        dataset=trainset,
        loader=trainloader,
        **analysis_config
    )

    for step in file_ids:
        checkpoint = run.checkpointer.load_file(step)
        print(f"Step {step}".center(total_length, "-"))
        run.model.load_state_dict(checkpoint["model"])
        evals = {
            **run.evaluator(run.model), # For testing purposes
            **eval_rlcts(run.model)
        }
        print(yaml.dump(evals))
        
        if run.logger:
            run.logger.log(evals, step=step)