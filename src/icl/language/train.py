"""
Training simple toy transformers on the Pile
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
from icl.language.data import get_loader
from icl.regression.evals import RegressionEvaluator
from icl.utils import get_default_device, prepare_experiments
from infra.utils.seed import set_seed

load_dotenv()
import logging
# in case using mps:
import os
from typing import Dict, List, Literal, Optional, Tuple, TypedDict

import datasets
import numpy as np
import sentry_sdk
import torch.nn.functional as F
import tqdm
from transformer_lens.utils import lm_cross_entropy_loss

import wandb
from icl.language.config import LanguageConfig, get_config
from icl.language.evals import LanguageEvaluator
from icl.language.model import HookedTransformer
from icl.monitoring import stdlogger
from icl.utils import get_device, move_to_device
from infra.io.logging import MetricLogger
from infra.io.storage import BaseStorageProvider
from infra.optim.schedulers import LRScheduler

WANDB_ENTITY = os.getenv("WANDB_ENTITY")

if XLA:
    import torch_xla.core.xla_model as xm
 

class LanguageRun:
    config: LanguageConfig
    model: nn.Module
    optimizer: torch.optim.Optimizer
    scheduler: Optional[LRScheduler]
    checkpointer: Optional[BaseStorageProvider]
    logger: Optional[MetricLogger]
    evaluator: LanguageEvaluator
    trainloader: datasets.Dataset

    def __init__(
            self, 
            config: LanguageConfig, 
            model: Optional[nn.Module] = None,
            optimizer: Optional[torch.optim.Optimizer] = None,
            scheduler: Optional[LRScheduler] = None,
            checkpointer: Optional[BaseStorageProvider] = None, 
            logger: Optional[MetricLogger]=None, 
    ):
        self.config = config

        # initialise model
        if model is None:
            model = config.transformer_factory().to(DEVICE)

        self.model = model

        # initialise datasets
        # if XLA: xm.mark_step()  

        trainset = self.config.trainset_factory()
        self.trainloader = get_loader(model, trainset)

        testset = self.config.testset_factory()
        self.evaluator = LanguageEvaluator(get_loader(model, testset, shuffle=False))

        # if XLA: xm.mark_step()

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
                torch.set_rng_state(torch.tensor(last_checkpoint["rng_state"]))

            if self.scheduler is not None:
                self.scheduler.load_state_dict(last_checkpoint["scheduler"])

    @classmethod
    def create_and_restore(cls, config: LanguageConfig):
        """Load a run from a checkpoint and restore the last checkpoint."""
        self = cls(config)
        self.restore()
        return self


def train(config: LanguageConfig) -> HookedTransformer:
    """
    Initialise and train an HookedTransformer model, tracking
    various metrics.
    """    
    stdlogger.info("\n" + "=" * 36 + f" {config.run_name} " + "=" * 36)
    stdlogger.info(yaml.dump(config.model_dump()))
    stdlogger.info("-" * 80 + "\n")

    run = LanguageRun(config)
    model = run.model
    optimizer = run.optimizer
    scheduler = run.scheduler
    checkpointer = run.checkpointer
    logger = run.logger
    device = get_default_device()

    num_steps = config.num_steps
    num_steps_per_epoch = len(run.trainloader)
    num_epochs = num_steps // num_steps_per_epoch

    step = 0

    model.train()
    model.to(device)

    if config.is_wandb_enabled:
        wandb.watch(model)

    stdlogger.info("Finished initialising model and data loaders.")
    stdlogger.info("Training for", num_epochs, "epochs.")

    for epoch in tqdm.trange(num_epochs, desc="Epochs"):
        rng_state = torch.get_rng_state()
        for b, batch in enumerate(tqdm.tqdm(run.trainloader, desc="Training...")):
            # if XLA: xm.mark_step()

            optimizer.zero_grad()
            tokens = batch['tokens'].to(device)
            logits = model(tokens)
            loss = lm_cross_entropy_loss(logits, tokens)
            loss.backward()
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            if XLA: xm.mark_step()

            if (step % 100 == 0 or step < 100 or (step % 10 == 0 and step < 1000)) and config.is_wandb_enabled:
                wandb.log({"batch/loss": loss.mean().item()}, step=step)

            if step in config.checkpointer_config.checkpoint_steps:
                stdlogger.info("Saving checkpoint at step %s", step)
                # if XLA: xm.mark_step()

                checkpoint = move_to_device(state_dict(model, optimizer, scheduler, rng_state, epoch=epoch, batch=b), 'cpu')
                assert str(get_device(checkpoint)) == 'cpu', "Checkpoint should be on CPU"
                checkpointer.save_file(step, checkpoint)

                # if XLA: xm.mark_step()

            if step in config.logger_config.logging_steps:
                stdlogger.info("Logging at step %s", step)
                # if XLA: xm.mark_step()
                model.eval()
                metrics = run.evaluator(model) 
                model.train()
                # if XLA: xm.mark_step()
                logger.log(metrics, step=step)

            step += 1

    if config.is_wandb_enabled:
        wandb.finish()

    stdlogger.info("\n" + "=" * 36 + f" Finished " + "=" * 36)
    stdlogger.info("\n")

    return model


def main(
    resume: str = typer.Option(None, help="The id of a sweep or run to resume."),
):
    if resume is None:
        config = get_config(project="tetrahedron-3m", entity=WANDB_ENTITY)
        train(config)
    else:
        typer.echo("Invalid resume command.")


if __name__ == "__main__":
    prepare_experiments()
    typer.run(main)
