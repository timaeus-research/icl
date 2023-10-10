"""
training the transformer on synthetic in-context regression task
"""
import torch
import typer
from devinfra.utils.seed import set_seed
# manage environment
from dotenv import load_dotenv
from pydantic import BaseModel
from torch import nn

from icl.evals import ICLEvaluator

load_dotenv()
# in case using mps:
import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # (! before import torch)

import logging
# from typing import Annotated, Dict, List, Literal, Optional, Tuple, TypedDict
from typing import Dict, List, Literal, Optional, Tuple, TypedDict

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

    # special code if device is 'xla'
    XLA = (config.device == 'xla')
    if XLA:
        stdlogger.info("device is 'xla'! some special code will run...")
        stdlogger.info("importing torch_xla...")
        import torch_xla.core.xla_model as xm
        # import torch_xla.debug.metrics as met
        stdlogger.info("configuring default XLA device...")
        device = xm.xla_device()
        stdlogger.info("xla ready!")
    else:
        device = config.device

    # model initialisation
    stdlogger.info("initialising model")
    model = config.task_config.model_factory().to(device)
    model.train()

    # initialise 'pretraining' data source (for training on fixed task set)
    stdlogger.info("initialising data (pretrain)")
    pretrain_dist = config.task_config.pretrain_dist_factory().to(device)

    # initialise 'true' data source (for evaluation, including unseen tasks)
    stdlogger.info("initialising data (true)")
    true_dist = config.task_config.true_dist_factory().to(device)

    # initialise evaluations
    stdlogger.info("initialising evaluator")
    if XLA: xm.mark_step()
    evaluator = ICLEvaluator(
        pretrain_dist=pretrain_dist,
        true_dist=true_dist,
        max_examples=config.task_config.max_examples,
        eval_batch_size=config.eval_batch_size,
        seed=config.task_config.true_seed
    )
    if XLA: xm.mark_step()

    # initialise monitoring code
    stdlogger.info("initialising checkpointer and logger")
    checkpointer = config.checkpointer_config.factory() if config.checkpointer_config is not None else None
    logger = config.logger_config.factory() if config.logger_config is not None else None

    # initialise torch optimiser
    stdlogger.info("initialising optimiser and scheduler")
    optimizer = config.optimizer_config.factory(model.parameters())
    scheduler = config.scheduler_config.factory(optimizer)  # type: ignore

    # TODO: this is unused and may be slowing down XLA... use it or lose it
    # recent_losses = torch.zeros(100, device=device)

    # training loop
    stdlogger.info("starting training loop")
    stdlogger.info("note: first two iterations slow while XLA compiles")
    stdlogger.info("note: early iterations slow due to logspace checkpoints")
    for step in tqdm.trange(config.num_steps, desc="training..."):
        # per-step seeds for reproducibility if we resume training
        set_seed(config.task_config.sampling_seed + step)
        
        # training step
        if XLA: xm.mark_step()
        xs, ys = pretrain_dist.get_batch(
            num_examples=config.task_config.max_examples,
            batch_size=config.batch_size,
        )
        ys_pred = model(xs, ys)
        loss = F.mse_loss(ys, ys_pred)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if XLA: xm.mark_step()

        # see above
        # recent_losses[step % 100] = loss

        # wand logging: log batch loss every 100 steps
        if step % 100 == 0 and step > 0 and config.is_wandb_enabled:
            stdlogger.info("logging batch loss at step %s", step)
            # TODO: Figure out how to make this work with `logger`
            wandb.log({"batch/loss": loss.mean().item()}, step=step)
        
        # evaluate and log metrics to wandb according to log_steps
        if step in config.logger_config.logging_steps:
            stdlogger.info("evaluating metrics at step %s", step)
            if XLA: xm.mark_step()
            model.eval()
            metrics = evaluator(model)
            model.train()
            if XLA: xm.mark_step()
            stdlogger.info("logging metrics at step %s", step)
            logger.log(metrics, step=step)


        # save checkpoints according to checkpoint_steps
        if step in config.checkpointer_config.checkpoint_steps:
            # TODO: if xla: move model to CPU before saving
            stdlogger.info("saving checkpoint at step %s", step)
            if XLA: xm.mark_step()
            checkpointer.save_file(step, state_dict(model, optimizer, scheduler))
            if XLA: xm.mark_step()

    if config.is_wandb_enabled:
        wandb.finish()

    # TODO: if XLA, move model off TPU?
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
