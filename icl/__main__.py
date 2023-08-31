"""
training the transformer on synthetic in-context regression task
"""
# manage environment
from dotenv import load_dotenv
from pydantic import BaseModel, model_validator

load_dotenv()
# in case using mps:
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1" # (! before import torch)

import functools
import logging
import random
import warnings
from typing import Any, Dict, List, Optional, TypedDict

import numpy as np
import torch
import tqdm
import wandb
#
from devinterp.evals import Evaluator
from devinterp.learner import LearnerConfig
from torch import nn

#
from icl.baselines import dmmse_predictor, ridge_predictor
from icl.model import InContextRegressionTransformer
from icl.tasks import (DiscreteTaskDistribution, GaussianTaskDistribution,
                       RegressionSequenceDistribution)


class ICLTaskConfig(BaseModel):
    task_size: int
    max_examples: int
    num_tasks: int
    noise_variance: float
    embed_size: int
    mlp_size: int
    num_heads: int
    num_layers: int

    def model_factory(self):
        return InContextRegressionTransformer(
            task_size=self.task_size,
            max_examples=self.max_examples,
            embed_size=self.embed_size,
            mlp_size=self.mlp_size,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
        )
    
    def pretrain_dist_factory(self):
        return RegressionSequenceDistribution(
            task_distribution=DiscreteTaskDistribution(
                num_tasks=self.num_tasks,
                task_size=self.task_size,
            ),
            noise_variance=self.noise_variance,
        )
    
    def true_dist_factory(self):
        return RegressionSequenceDistribution(
            task_distribution=GaussianTaskDistribution(
                task_size=self.task_size,
            ),
            noise_variance=self.noise_variance,
        )

class ICLConfig(LearnerConfig):
    eval_batch_size: int
    task_config: ICLTaskConfig

    @model_validator(mode='before')
    @classmethod
    def validate_extra(cls, data: Any):
        num_tasks = data["task_config"]["num_tasks"]
        num_steps = data["num_steps"]
        
        # Automatically fill in the project_dir field of the checkpointer
        checkpoint_config = data.get("checkpointer_config", None)
        if num_tasks is not None and checkpoint_config is not None:
            checkpoint_config["project_dir"] = checkpoint_config.get("project_dir", f"icl-ntasks-{num_tasks}-model")

        # Num samples
        data["num_training_samples"] = num_steps * data["batch_size"]

        # LR Scheduler
        optimizer_config = data.get("optimizer_config", None)
        scheduler_config = data.get("scheduler_config", None)
        if scheduler_config is not None:
            scheduler_config["max_lr"] = scheduler_config.get("max_lr", optimizer_config.get("lr", 1e-3))
            scheduler_config["max_steps"] = scheduler_config.get("max_steps", num_steps)
            scheduler_config["div_factor"] = scheduler_config.get("div_factor", (num_steps/2 - 1))
            scheduler_config["final_div_factor"] = scheduler_config.get("final_div_factor", (num_steps/2 - 1))

        return data


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        torch.cuda.manual_seed_all(seed) 
    except AttributeError:
        warnings.info("CUDA not available; failed to seed")


def mse(y1, y2, axis=None):
    """
    Loss function: Mean squared error between the elements of two tensors of
    the same shape (summed along all axes or only `axis`).

    * Used as a loss function for least-squares regression
      (e.g., `mse(ys_true, ys_pred)`).
    * Used to compare the difference between two algorithms' regression
      predictions.
      (e.g., `mse(ys_algo1, ys_algo2)`).
    * If `ys1` and `ys2` are (batch, time, dimension) tensors, then we can
      get a vector of per-token losses by averaging over only the first and
      last dimensions (e.g., `mse(ys1, ys2, axis=(0, 2))`).
    """
    return (y1 - y2).square().mean(axis=axis) 


class StateDict(TypedDict):
    model: Dict
    optimizer: Dict
    scheduler: Dict

def state_dict(model, optimizer, scheduler) -> StateDict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
    }


def train(config: ICLConfig, seed: int = 0, is_debug: bool = False) -> InContextRegressionTransformer:
    """
    Initialise and train an InContextRegressionTransformer model, tracking
    various metrics.
    """
    logging.basicConfig(level=logging.INFO if not is_debug else logging.DEBUG)
    set_seed(seed)

    # initialise model
    model = config.task_config.model_factory()
    model.to(config.device)
    model.train()

    # initialise 'pretraining' data source (for training on fixed task set)
    pretrain_dist = config.task_config.pretrain_dist_factory()
    pretrain_dist.to(config.device)

    # initialise 'true' data source (for evaluation, including unseen tasks)
    true_dist = config.task_config.true_dist_factory()
    true_dist.to(config.device)

    # initialise evaluations
    evaluator = ICLEvaluator(
        pretrain_dist=pretrain_dist,
        true_dist=true_dist,
        max_examples=config.task_config.max_examples,
        eval_batch_size=config.eval_batch_size,
    )

    # initialise monitoring code
    checkpointer = config.checkpointer_config.factory() if config.checkpointer_config is not None else None
    logger = config.logger_config.factory() if config.logger_config is not None else None

    # initialise torch optimiser
    optimizer = config.optimizer_config.factory(model.parameters())
    scheduler = config.scheduler_config.factory(optimizer)  # type: ignore

    # training loop
    for step in tqdm.trange(config.num_steps, desc=f"Epoch 0 Batch 0/{config.num_steps} Loss: ?.??????"):
        set_seed(seed + step)  # For reproducibility if we resume training

        # data generation and forward pass
        xs, ys = pretrain_dist.get_batch(
            num_examples=config.task_config.max_examples,
            batch_size=config.batch_size,
        )
        ys_pred = model(xs, ys)
        loss = mse(ys, ys_pred)
        # backward pass and gradient step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if config.is_wandb_enabled:
            # TODO: Figure out how to make this work with Logger
            wandb.log({"batch/loss": loss.item()}, step=step)

        # Log to wandb & save checkpoints according to log_steps
        if checkpointer and step in config.checkpointer_config.checkpoint_steps:  # type: ignore
            print("saving checkpoint")
            logger.info(f"Saving checkpoint at step {step}")
            checkpointer.save_file((0, step), state_dict(model, optimizer, scheduler))

        if logger and step in config.logger_config.logging_steps:  # type: ignore
            model.eval()
            metrics = evaluator(model)
            model.train()
            logger.log(metrics, step=step)

    if config.is_wandb_enabled:
        wandb.finish()

    return model


class ICLEvaluator(Evaluator):
    """
    Stores fixed evaluation data batches, computed at the start of the
    training run, as well as baseline predictions for these batches.
    """
    def __init__(
        self,
        pretrain_dist: RegressionSequenceDistribution,
        true_dist: RegressionSequenceDistribution,
        max_examples: int,
        eval_batch_size: int,
    ):
        # fixed evaluation batches (computed once at start of training run)
        self.pretrain_xs, self.pretrain_ys = pretrain_dist.get_batch(
            num_examples=max_examples,
            batch_size=eval_batch_size,
        )
        self.true_xs, self.true_ys = true_dist.get_batch(
            num_examples=max_examples,
            batch_size=eval_batch_size,
        )

        # configure baseline predictors
        # dmmse is the bayes-optimal predictor for the pretraining data
        dmmse = functools.partial(
            dmmse_predictor,
            prior=pretrain_dist.task_distribution,
            noise_variance=pretrain_dist.noise_variance,
        )

        # ridge is the bayes-optimal predictor for the true data
        ridge = functools.partial(
            ridge_predictor,
            noise_variance=true_dist.noise_variance,
        )

        # cache baseline predictions (to compare against model predictions)
        self.pretrain_dmmse_preds = dmmse(self.pretrain_xs, self.pretrain_ys)
        self.pretrain_ridge_preds = ridge(self.pretrain_xs, self.pretrain_ys)
        self.true_dmmse_preds = dmmse(self.true_xs, self.true_ys)
        self.true_ridge_preds = ridge(self.true_xs, self.true_ys)


    @torch.no_grad()
    def __call__(self, model: nn.Module, *args, **kwargs):
        """
        Evaluate a model against stored batches, returning a dictionary of
        various metrics.
        """
        # compute model predictions and loss on fixed batch from T_pretrain
        pretrain_model_preds = model(self.pretrain_xs, self.pretrain_ys)
        pretrain_model_losses = mse(self.pretrain_ys, pretrain_model_preds, axis=(0,2))
        # compute model predictions and loss on fixed batch from T_true
        true_model_preds = model(self.true_xs, self.true_ys)
        true_model_losses = mse(self.true_ys, true_model_preds, axis=(0,2))
        # compute and return various metrics based on above
        return {
            "pretrain/mse": pretrain_model_losses.mean().item(),
            "pretrain/per_token": pretrain_model_losses.tolist(),
            "pretrain/last": pretrain_model_losses[-1].item(),
            "pretrain/delta_dmmse": mse(pretrain_model_preds, self.pretrain_dmmse_preds),
            "pretrain/delta_ridge": mse(pretrain_model_preds, self.pretrain_ridge_preds),
            "true/mse": true_model_losses.mean().item(),
            "true/per_token": true_model_losses.tolist(),
            "true/last": true_model_losses[-1].item(),
            "true/delta_dmmse": mse(true_model_preds, self.true_dmmse_preds),
            "true/delta_ridge": mse(true_model_preds, self.true_ridge_preds),
        }


def get_config(project: Optional[str] = None, entity: Optional[str] = None) -> ICLConfig:
    # (shared parameters)
    num_steps = 500_000
    batch_size = 256
    max_learning_rate = 1e-3
    num_tasks = 64

    config_dict = {
        # model & data config
        "task_config": {
            "task_size": 8,
            "max_examples": 16,
            "num_tasks": num_tasks,
            "noise_variance": 0.25,
            "embed_size": 128,
            "mlp_size": 128,
            "num_heads": 2,
            "num_layers": 8,
        },
        # training config
        "num_steps": num_steps, 
        "batch_size": batch_size,
        "optimizer_config": {
            "optimizer_type": "Adam",
            "betas": (0.9, 0.999),
            "weight_decay": 0.0,
            "lr": max_learning_rate,   # unused (overwritten by scheduler)
        },
        "scheduler_config": {
            "scheduler_type": "OneCycleLR",
            "anneal_strategy": 'linear',
            "pct_start": 0.5,
            "cycle_momentum": False,    # N/A but required to avoid error
        },
        # evaluation config
        "eval_batch_size": 2048,
        "checkpointer_config": {
            "checkpoint_steps": {
                "log_space": 50,
                "linear_space": 50
            },
            "bucket": "devinterp",
            # "local_root": "../checkpoints",
        },
        # for wandb?
        "logger_config": {
            "logging_steps": {
                "log_space": 500,
                "linear_space": 500,
            },
            "project": project,
            "entity": entity,
            # "stdout": True
        }
    }

    return ICLConfig(**config_dict)


if __name__ == "__main__":
    # config = get_config(project="icl", entity="devinterp")
    logging.basicConfig(level=logging.INFO)
    config = get_config()
    # print(config)
    # train(config, seed=0, is_debug=False)

