"""
training the transformer on synthetic in-context regression task
"""
# manage environment
from dotenv import load_dotenv

load_dotenv()
# in case using mps:
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1" # (! before import torch)

import functools
import logging
import random
import warnings
from typing import Dict, List, Optional, TypedDict

import numpy as np
import torch
import tqdm
import wandb
#
from devinterp.config import Config
from devinterp.logging import Logger
from devinterp.storage import CheckpointManager

#
from icl.baselines import dmmse_predictor, ridge_predictor
from icl.model import InContextRegressionTransformer
from icl.tasks import (DiscreteTaskDistribution, GaussianTaskDistribution,
                       RegressionSequenceDistribution)


class ICLConfig(Config):
    # dataset & loader
    task_size: int
    max_examples: int
    num_tasks: int
    noise_variance: float

    # model
    embed_size: int
    mlp_size: int
    num_heads: int
    num_layers: int
    
    # evaluation
    eval_batch_size: int


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
    model = InContextRegressionTransformer(
        task_size=config.task_size,
        max_examples=config.max_examples,
        embed_size=config.embed_size,
        mlp_size=config.mlp_size,
        num_heads=config.num_heads,
        num_layers=config.num_layers,
        device=config.device,
    )
    model.to(config.device)
    model.train()

    # initialise 'pretraining' data source (for training on fixed task set)
    pretrain_dist = RegressionSequenceDistribution(
        task_distribution=DiscreteTaskDistribution(
            num_tasks=config.num_tasks,
            task_size=config.task_size,
            device=config.device,
        ),
        noise_variance=config.noise_variance,
    )

    # initialise 'true' data source (for evaluation, including unseen tasks)
    true_dist = RegressionSequenceDistribution(
        task_distribution=GaussianTaskDistribution(
            task_size=config.task_size,
            device=config.device,
        ),
        noise_variance=config.noise_variance,
    )

    # initialise evaluations
    evaluator = Evaluator(
        pretrain_dist=pretrain_dist,
        true_dist=true_dist,
        max_examples=config.max_examples,
        eval_batch_size=config.eval_batch_size,
    )

    # initialise monitoring code
    # TODO: hash the model details to create a unique identifier for the model and use this as project name below
    checkpointer = CheckpointManager(f"icl-ntasks-{config.num_tasks}", "devinterp") # , "experiments/icl")
    logger = Logger(config.project, config.entity, config.logging_steps)

    # initialise torch optimiser
    optimizer = config.optimizer_config.factory(model.parameters())
    scheduler = config.scheduler_config.factory(optimizer)

    # training loop
    for step in tqdm.trange(config.num_steps, desc=f"Epoch 0 Batch 0/{config.num_steps} Loss: ?.??????"):
        set_seed(seed + step)  # For reproducibility if we resume training

        # data generation and forward pass
        xs, ys = pretrain_dist.get_batch(
            num_examples=config.max_examples,
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
        if step in config.checkpoint_steps:
            print("saving checkpoint")
            logger.info(f"Saving checkpoint at step {step}")
            checkpointer.save_file((0, step), state_dict(model, optimizer, scheduler))

        if step in config.logging_steps:
            model.eval()
            metrics = evaluator.eval(model)
            model.train()
            logger.log(metrics, step=step)

    if config.is_wandb_enabled:
        wandb.finish()

    return model


class Evaluator:
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
    def eval(self, model):
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

    config_dict = {
        # data config
        "task_size": 8,
        "max_examples": 16,
        "num_tasks": 64,
        "noise_variance": 0.25,
        # model config
        "embed_size": 128,
        "mlp_size": 128,
        "num_heads": 2,
        "num_layers": 8,
        # training config
        "num_steps": num_steps, 
        "batch_size": batch_size,
        "num_training_samples": num_steps * batch_size,
        "optimizer_config": {
            "optimizer_type": "Adam",
            "betas": (0.9, 0.999),
            "weight_decay": 0.0,
            "lr": max_learning_rate,   # unused (overwritten by scheduler)
        },
        "scheduler_config": {
            # one-cycle triangle learning rate schedule with 50% warmup
            "scheduler_type": "OneCycleLR",
            "max_lr": max_learning_rate,
            "total_steps": num_steps,
            "anneal_strategy": 'linear',
            "div_factor": (num_steps/2 - 1),        # start 1 step past 0
            "final_div_factor": (num_steps/2 - 1),  # end 1 step before 0
            "pct_start": 0.5,
            "cycle_momentum": False,    # N/A but required to avoid error
        },
        # evaluation config
        "eval_batch_size": 2048,
        # "logging_steps": (500, 500), 
        "logging_steps": (500, 500), 
        # "checkpoint_steps": (100, 100),
        "checkpoint_steps": None,
        # for wandb?
        "project": project,
        "entity": entity,
    }

    if project is not None and entity is not None:
        # use wandb
        wandb.init(project=project, entity=entity)
        config_dict.update(wandb.config)

    return ICLConfig(**config_dict)


if __name__ == "__main__":
    config = get_config(project="icl", entity="devinterp")
    # config = get_config()
    train(config, seed=0, is_debug=False)

