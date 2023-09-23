"""
training the transformer on synthetic in-context regression task
"""
import torch
import typer
# manage environment
from dotenv import load_dotenv
from pydantic import BaseModel
from torch import nn

from icl.evals import ICLEvaluator
from icl.utils import set_seed

load_dotenv()
# in case using mps:
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # (! before import torch)

import logging
from typing import Annotated, Dict, List, Literal, Optional, Tuple, TypedDict

import numpy as np
import sentry_sdk
import torch.nn.functional as F
import tqdm
from devinfra.io.logging import MetricLogger
from devinfra.io.storage import BaseStorageProvider
from devinfra.optim.schedulers import LRScheduler

import wandb
from icl.config import ICLConfig, get_config
from icl.model import InContextRegressionTransformer
from icl.tasks import (DiscreteTaskDistribution, GaussianTaskDistribution,
                       RegressionSequenceDistribution)

stdlogger = logging.getLogger("ICL")


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


class Run:
    config: ICLConfig
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Optional[LRScheduler]
    pretrain_dist: RegressionSequenceDistribution[DiscreteTaskDistribution]
    true_dist: RegressionSequenceDistribution[GaussianTaskDistribution]
    evaluator: ICLEvaluator
    checkpointer: Optional[BaseStorageProvider]
    logger: Optional[MetricLogger]

    def __init__(
            self, 
            config: ICLConfig, 
            model: Optional[nn.Module] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            scheduler: Optional[LRScheduler] = None,
            checkpointer: Optional[BaseStorageProvider] = None, 
            logger: Optional[MetricLogger]=None, 
    ):
        self.config = config

        # initialise model
        if model is None:
            model = config.task_config.model_factory().to(config.device)

        self.model = model

        # initialise 'pretraining' data source (for training on fixed task set)
        self.pretrain_dist = config.task_config.pretrain_dist_factory().to(
            config.device
        )

        # initialise 'true' data source (for evaluation, including unseen tasks)
        self.true_dist = config.task_config.true_dist_factory().to(config.device)

        # initialise evaluations
        self.evaluator = ICLEvaluator(
            pretrain_dist=self.pretrain_dist,
            true_dist=self.true_dist,
            max_examples=config.task_config.max_examples,
            eval_batch_size=config.eval_batch_size,
            seed=config.task_config.true_seed,
        )

        # initialise monitoring code
        if checkpointer is None and config.checkpointer_config is not None: 
            checkpointer = config.checkpointer_config.factory()

        self.checkpointer = checkpointer

        if logger is None and config.logger_config is not None:
            logger = config.logger_config.factory()

        self.logger = logger

        # initialise torch optimiser
        if optimizer is None:
            optimizer = config.optimizer_config.factory(self.model.parameters())

        self.optimizer = optimizer

        if scheduler is None and config.scheduler_config is not None:
            scheduler = config.scheduler_config.factory(self.optimizer)

        self.scheduler = scheduler

    def restore(self):
        """Restores the last checkpoint for this run."""
        if self.checkpointer:
            self.checkpointer.sync()
            
            if not self.checkpointer.file_ids:
                raise ValueError("No checkpoints found.")
        
            last_checkpoint = self.checkpointer[-1]
            self.model.load_state_dict(last_checkpoint["model"])
            self.optimizer.load_state_dict(last_checkpoint["optimizer"])

            if self.scheduler is not None:
                self.scheduler.load_state_dict(last_checkpoint["scheduler"])

    @classmethod
    def create_and_restore(cls, config: ICLConfig):
        """Load a run from a checkpoint and restore the last checkpoint."""
        self = cls(config)
        self.restore()
        return self


def train(config: ICLConfig, is_debug: bool = False) -> InContextRegressionTransformer:
    """
    Initialise and train an InContextRegressionTransformer model, tracking
    various metrics.
    """
    logging.basicConfig(level=logging.INFO if not is_debug else logging.DEBUG)

    run = Run(config)
    model = run.model
    model.train()
    optimizer = run.optimizer
    scheduler = run.scheduler
    pretrain_dist = run.pretrain_dist
    evaluator = run.evaluator
    checkpointer = run.checkpointer
    logger = run.logger

    num_steps = config.num_steps

    # Let's only train until 50% checkpoint.
    checkpoint_steps = sorted(list(config.checkpointer_config.checkpoint_steps))
    num_steps = checkpoint_steps[len(checkpoint_steps) // 2] + 1

    recent_losses = torch.zeros(100, device=config.device)

    # training loop
    for step in tqdm.trange(num_steps, desc="Training..."):
        set_seed(
            config.task_config.sampling_seed + step
        )  # For reproducibility if we resume training

        # data generation and forward pass
        xs, ys = pretrain_dist.get_batch(
            num_examples=config.task_config.max_examples,
            batch_size=config.batch_size,
        )
        ys_pred = model(xs, ys)
        loss = F.mse_loss(ys, ys_pred)
        # backward pass and gradient step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        recent_losses[step % 100] = loss

        if step % 100 == 0 and step > 0 and config.is_wandb_enabled:
            # TODO: Figure out how to make this work with Logger
            wandb.log({"batch/loss": loss.mean().item()}, step=step)

        # Log to wandb & save checkpoints according to log_steps
        if step in config.checkpointer_config.checkpoint_steps:
            stdlogger.info("Saving checkpoint at step %s", step)
            checkpointer.save_file(step, state_dict(model, optimizer, scheduler))

        if step in config.logger_config.logging_steps:
            stdlogger.info("Logging at step %s", step)
            model.eval()
            metrics = evaluator(model)
            model.train()
            logger.log(metrics, step=step)

    if config.is_wandb_enabled:
        wandb.finish()

    return model


def get_run_config(run: "Run"):
    """Side-effect: resumes the wandb run"""
    return get_config(**run.config, resume="must", id=run.id)


def get_last_checkpoint(config: ICLConfig):
    """Get the last checkpoint for a given run"""
    checkpointer = config.checkpointer_config.factory()
    last_checkpoint_step = sorted([int(x) for x in checkpointer.get_file_ids()])[-1]
    last_checkpoint = checkpointer.load_file(last_checkpoint_step)

    model = config.task_config.model_factory().to(config.device)
    optimizer = config.optimizer_config.factory(model.parameters())
    scheduler = config.scheduler_config.factory(optimizer)

    model.load_state_dict(last_checkpoint["model"])
    optimizer.load_state_dict(last_checkpoint["optimizer"])
    scheduler.load_state_dict(last_checkpoint["scheduler"])

    return {
        "model": model,
        "optimizer": optimizer,
        "scheduler": scheduler,
    }


def resume_run(run, is_debug: bool = False) -> InContextRegressionTransformer:
    """
    Initialise and train an InContextRegressionTransformer model, tracking
    various metrics.
    """
    logging.basicConfig(level=logging.INFO if not is_debug else logging.DEBUG)

    config = get_run_config(run)
    last_log_step = run.summary.get("_step")

    # initialise model, optimizer, and scheduler
    last_checkpoint = get_last_checkpoint(config)
    model = last_checkpoint["model"]
    optimizer = last_checkpoint["optimizer"]
    scheduler = last_checkpoint["scheduler"]
    last_checkpoint_step = scheduler.last_epoch

    logging.info(
        "Resuming run %s at step %s. Last logged at %s",
        run.id,
        last_checkpoint_step,
        last_log_step,
    )

    model.to(config.device).train()

    # initialise 'pretraining' data source (for training on fixed task set)
    pretrain_dist = config.task_config.pretrain_dist_factory().to(config.device)

    # initialise 'true' data source (for evaluation, including unseen tasks)
    true_dist = config.task_config.true_dist_factory().to(config.device)

    # initialise evaluations
    evaluator = ICLEvaluator(
        pretrain_dist=pretrain_dist,
        true_dist=true_dist,
        max_examples=config.task_config.max_examples,
        eval_batch_size=config.eval_batch_size,
        seed=config.task_config.true_seed,
    )

    # initialise monitoring code
    checkpointer = (
        config.checkpointer_config.factory()
        if config.checkpointer_config is not None
        else None
    )
    logger = (
        config.logger_config.factory() if config.logger_config is not None else None
    )

    num_steps = config.num_steps
    recent_losses = torch.zeros(100, device=config.device)

    # training loop
    for step in tqdm.trange(last_checkpoint_step + 1, num_steps, desc="Training..."):
        set_seed(
            config.task_config.sampling_seed + step
        )  # For reproducibility if we resume training

        # data generation and forward pass
        xs, ys = pretrain_dist.get_batch(
            num_examples=config.task_config.max_examples,
            batch_size=config.batch_size,
        )
        ys_pred = model(xs, ys)
        loss = F.mse_loss(ys, ys_pred)
        # backward pass and gradient step
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        recent_losses[step % 100] = loss

        if step % 100 == 0 and step > last_log_step and config.is_wandb_enabled:
            wandb.log({"batch/loss": loss.mean().item()}, step=step)

        # Log to wandb & save checkpoints according to log_steps
        if step in config.checkpointer_config.checkpoint_steps:
            stdlogger.info("Saving checkpoint at step %s", step)
            checkpointer.save_file(step, state_dict(model, optimizer, scheduler))

        if step in config.logger_config.logging_steps and step > last_log_step:
            stdlogger.info("Logging at step %s", step)
            model.eval()
            metrics = evaluator(model)
            model.train()
            logger.log(metrics, step=step)

    if config.is_wandb_enabled:
        wandb.finish()

    return model


def clean_sweep(sweep: "Sweep"):
    """Get rid of any runs that never even got started."""

    # Delete the runs (on wandb) that are finished/crashed and have _step == None
    def _clean_sweep():
        for r in sweep.runs:
            if r.summary.get("_step", None) is None:
                r.delete()
                yield r

    return list(r for r in _clean_sweep())


def get_runs_to_continue(sweep: "Sweep", num_steps: int):
    """Return all runs that have not yet reached the specified number of steps."""
    runs = sorted([r for r in sweep.runs], key=lambda r: r.summary.get("_step", 0))

    return [r for r in runs if r.summary.get("_step", 0) < num_steps]


def resume_sweep(sweep_id: str, is_debug: bool = False):
    api = wandb.Api()
    sweep = api.sweep(sweep_id)

    clean_sweep(sweep)
    num_steps = sweep.config.get("num_steps", 500_000)
    runs = get_runs_to_continue(sweep, num_steps)

    for run in runs:
        print(run)
        resume_run(run, is_debug=is_debug)


def main(
    resume: Annotated[str, typer.Option(help="The id of a sweep or run to resume.")]
):
    is_debug = False

    if resume is None:
        config = get_config(project="icl", entity="devinterp")
        train(config, is_debug=is_debug)
    else:
        if "run" in resume:
            resume_run(resume, is_debug=is_debug)
        elif "sweep" in resume:
            resume_sweep(resume, is_debug=is_debug)
        else:
            typer.echo("Invalid resume command.")


if __name__ == "__main__":
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
    typer.run(main)
