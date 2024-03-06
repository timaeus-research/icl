"""
training the transformer on synthetic in-context regression task
"""
import torch
import typer
import yaml
# manage environment
from dotenv import load_dotenv
from pydantic import BaseModel
from torch import nn

from icl.checkpoints import state_dict
from icl.constants import DEVICE, XLA
from icl.regression.evals import RegressionEvaluator
from icl.utils import prepare_experiments
from infra.utils.seed import set_seed

load_dotenv()
import logging
# in case using mps:
import os
from typing import Optional, Tuple

import numpy as np
import sentry_sdk
import torch.nn.functional as F
import tqdm

import wandb
from icl.monitoring import stdlogger
from icl.regression.config import RegressionConfig, get_config
from icl.regression.model import InContextRegressionTransformer
from icl.regression.tasks import (DiscreteTaskDistribution,
                                  GaussianTaskDistribution,
                                  RegressionSequenceDistribution)
from icl.utils import get_device, move_to_device
from infra.io.logging import MetricLogger
from infra.io.storage import BaseStorageProvider
from infra.optim.schedulers import LRScheduler

WANDB_ENTITY = os.getenv("WANDB_ENTITY")

if XLA:
    import torch_xla.core.xla_model as xm
 

class RegressionRun:
    config: RegressionConfig
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Optional[LRScheduler]
    pretrain_dist: RegressionSequenceDistribution[DiscreteTaskDistribution]
    true_dist: RegressionSequenceDistribution[GaussianTaskDistribution]
    evaluator: RegressionEvaluator
    checkpointer: Optional[BaseStorageProvider]
    logger: Optional[MetricLogger]

    def __init__(
            self, 
            config: RegressionConfig, 
            model: Optional[nn.Module] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            scheduler: Optional[LRScheduler] = None,
            checkpointer: Optional[BaseStorageProvider] = None, 
            logger: Optional[MetricLogger]=None, 
    ):
        self.config = config

        # initialise model
        if model is None:
            model = config.task_config.model_factory().to(DEVICE)

        self.model = model

        # initialise 'pretraining' data source (for training on fixed task set)
        self.pretrain_dist = config.task_config.pretrain_dist_factory().to(
            DEVICE
        )

        # initialise 'true' data source (for evaluation, including unseen tasks)
        self.true_dist = config.task_config.true_dist_factory().to(DEVICE)

        # initialise evaluations
        if XLA: xm.mark_step()  
        self.evaluator = RegressionEvaluator(
            pretrain_dist=self.pretrain_dist,
            true_dist=self.true_dist,
            max_examples=config.task_config.max_examples,
            eval_batch_size=config.eval_batch_size,
            seed=config.task_config.true_seed,
        )
        if XLA: xm.mark_step()

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
            if not self.checkpointer.file_ids:
                raise ValueError("No checkpoints found.")
        
            last_checkpoint = self.checkpointer[-1]
            self.model.load_state_dict(last_checkpoint["model"])
            self.optimizer.load_state_dict(last_checkpoint["optimizer"])

            if "rng_state" in last_checkpoint:
                torch.set_rng_state(torch.tensor(last_checkpoint["rng_state"], dtype=torch.uint8))


            if self.scheduler is not None:
                self.scheduler.load_state_dict(last_checkpoint["scheduler"])

    @classmethod
    def create_and_restore(cls, config: RegressionConfig):
        """Load a run from a checkpoint and restore the last checkpoint."""
        self = cls(config)
        self.restore()
        return self


def train(config: RegressionConfig) -> InContextRegressionTransformer:
    """
    Initialise and train an InContextRegressionTransformer model, tracking
    various metrics.
    """    
    stdlogger.info("\n" + "=" * 36 + f" {config.run_name} " + "=" * 36)
    stdlogger.info(yaml.dump(config.model_dump()))
    stdlogger.info("-" * 80 + "\n")

    run = RegressionRun(config)
    model = run.model
    model.train()
    optimizer = run.optimizer
    scheduler = run.scheduler
    pretrain_dist = run.pretrain_dist
    evaluator = run.evaluator
    checkpointer = run.checkpointer
    logger = run.logger

    num_steps = config.num_steps
    sampling_seed = config.task_config.sampling_seed

    if sampling_seed is None and config.task_config.pretrain_seed is not None:
        sampling_seed = config.task_config.pretrain_seed * num_steps

    for step in tqdm.trange(num_steps, desc="Training..."):
        if sampling_seed is not None:
            set_seed(
                sampling_seed + step
            )  # For reproducibility if we resume training

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

        if step % 100 == 0 and step > 0 and config.is_wandb_enabled:
            # TODO: Figure out how to make this work with Logger
            wandb.log({"batch/loss": loss.mean().item()}, step=step)

        # Log to wandb & save checkpoints according to log_steps
        if step in config.checkpointer_config.checkpoint_steps:
            stdlogger.info("Saving checkpoint at step %s", step)
            if XLA: xm.mark_step()

            checkpoint = move_to_device(state_dict(model, optimizer, scheduler, torch.get_rng_state()), 'cpu')
            assert str(get_device(checkpoint)) == 'cpu', "Checkpoint should be on CPU"
            checkpointer.save_file(step, checkpoint)

            if XLA: xm.mark_step()

        if step in config.logger_config.logging_steps:
            stdlogger.info("Logging at step %s", step)
            if XLA: xm.mark_step()
            model.eval()
            metrics = evaluator(model)
            model.train()
            if XLA: xm.mark_step()
            logger.log(metrics, step=step)

    if config.is_wandb_enabled:
        wandb.finish()

    stdlogger.info("\n" + "=" * 36 + f" Finished " + "=" * 36)
    stdlogger.info("\n")

    return model



def main(
    resume: str = typer.Option(None, help="The id of a sweep or run to resume."),
):
    if resume is None:
        config = get_config(project="icl", entity=WANDB_ENTITY)
        train(config)
    else:
        typer.echo("Invalid resume command.")


if __name__ == "__main__":
    prepare_experiments()
    typer.run(main)
