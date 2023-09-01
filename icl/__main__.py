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
from typing import Dict, List, TypedDict

import numpy as np
import sentry_sdk
import torch
import tqdm
#
from devinterp.evals import Evaluator
from torch import nn

import wandb
from icl.baselines import dmmse_predictor, ridge_predictor
#
from icl.config import ICLConfig, get_config
from icl.model import InContextRegressionTransformer
from icl.tasks import RegressionSequenceDistribution

stdlogger = logging.getLogger("ICL")


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

    print(checkpointer, config.checkpointer_config.checkpoint_steps)

    # training loop
    for step in tqdm.trange(config.num_steps, desc=f"Training..."):
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
        if step in config.checkpointer_config.checkpoint_steps:
            stdlogger.info(f"Saving checkpoint at step {step}")
            checkpointer.save_file(step, state_dict(model, optimizer, scheduler))

        if step in config.logger_config.logging_steps:
            stdlogger.info(f"Logging at step {step}")
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

        def get_token_losses_dict(losses: torch.Tensor, label: str):
            return {f"{label}/token/{i}": losses[i].item() for i in range(losses.shape[0])}
        
        return {
            "pretrain/mse": pretrain_model_losses.mean().item(),
            "pretrain/delta_dmmse": mse(pretrain_model_preds, self.pretrain_dmmse_preds),
            "pretrain/delta_ridge": mse(pretrain_model_preds, self.pretrain_ridge_preds),
            **get_token_losses_dict(pretrain_model_losses, "pretrain"),
            "true/mse": true_model_losses.mean().item(),
            "true/delta_dmmse": mse(true_model_preds, self.true_dmmse_preds),
            "true/delta_ridge": mse(true_model_preds, self.true_ridge_preds),
            **get_token_losses_dict(true_model_losses, "true"),

        }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = get_config(project="icl", entity="devinterp")
    # config = get_config()
    train(config, seed=0, is_debug=False)

