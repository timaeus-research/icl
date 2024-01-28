import logging
import math
import random
import time
import warnings
from contextlib import contextmanager
from typing import Any, Callable, Dict, List, Optional, TypedDict

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from pydantic import BaseModel, Field, model_validator, validator
from tqdm.notebook import tqdm

import wandb
from infra.data import SeedableDataLoader
from infra.evals import CriterionLiteral, Evaluator
from infra.logging import MetricLogger, MetricLoggingConfig
from infra.monitoring import expand_steps_config_
from infra.optim.optimizers import OptimizerConfig
from infra.optim.schedulers import LRScheduler, SchedulerConfig
from infra.storage import BaseStorageProvider, CheckpointerConfig
from infra.utils.device import get_default_device, temporarily_move_to_
from infra.utils.iterables import nested_update
from infra.utils.seed import set_seed


class LearnerStateDict(TypedDict):
    model: Dict
    optimizer: Dict
    scheduler: Optional[Dict]   


class Learner:
    def __init__(
        self,
        model: torch.nn.Module,
        dataset: torch.utils.data.Dataset,
        optimizer: torch.optim.Optimizer,
        config: "LearnerConfig",
        scheduler: Optional[LRScheduler] = None,
        logger: Optional[MetricLogger] = None,
        checkpointer: Optional[BaseStorageProvider] = None,
        evaluator: Optional[Evaluator] = None,
        criterion: Callable = F.cross_entropy,
    ):
        """
        Initializes the Learner object.

        Args:
            model (torch.nn.Module): The PyTorch model to be trained.
            dataset (torch.utils.data.Dataset): The training dataset.
            config (Config): Configuration object containing hyperparameters.
        evals (Optional[List[Callable]]): List of metric functions to evaluate the model.

        """

        self.config = config
        self.model = model
        self.dataset = dataset
        self.dataloader = SeedableDataLoader(dataset, batch_size=config.batch_size)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.evaluator = evaluator
        self.logger = logger
        self.checkpointer = checkpointer
        self.criterion = criterion

    def resume(self, step: Optional[int] = None):
        """
        Resumes training from a saved checkpoint.

        Args:
            step (Optional[int]): Batch index to resume training from.

        """
        if self.checkpointer is None:
            raise ValueError("Cannot resume training without a checkpointer.")

        if step is None:
            epoch, step = self.checkpointer[-1]
        else:
            epoch, batch = min(self.checkpointer, key=lambda x: abs(x[1] - step))

            if batch != step:
                warnings.warn(
                    f"Could not find checkpoint with step {step}. Resuming from closest batch ({batch}) instead."
                )

            step = batch

            self.dataloader.set_seed(epoch)
            # TODO: loop until this specific batch

        self.load_checkpoint(epoch, step)

    def train(self, resume=0, verbose: bool = True):
        """
        Trains the model.

        Args:
            resume (int): Batch index to resume training from. If 0, will not resume.
            run_id (Optional[str]): Wandb run ID to resume from. If None, will create a new run.

        """

        if resume:
            self.resume(resume)

        self.model.to(self.config.device)
        self.model.train()

        pbar = verbose and tqdm(
            total=self.config.num_steps,
            desc=f"Training...",
        )

        last_logged_time = 0

        def maybe_log_batch_loss(throttle_sec=1):
            """Throttled logging of batch loss."""
            nonlocal last_logged_time
            current_time = time.time()

            if current_time - last_logged_time >= throttle_sec:  # 1 second
                if self.config.is_wandb_enabled:
                    wandb.log({"Batch/Loss": loss.item()}, step=step)

                last_logged_time = current_time

        def log(step):
            self.model.eval()
            evals = (
                self.evaluator(self.model, self.optimizer, self.scheduler)
                if self.evaluator
                else {"Batch/Loss": loss.item()}
            )
            self.logger.log(evals, step=step)
            self.model.train()

        step = 0

        for epoch in range(0, self.config.num_epochs):
            set_seed(epoch, self.dataloader, self.config.device)

            for _batch_idx, (x, y) in enumerate(self.dataloader):
                step = self.config.num_steps_per_epoch * epoch + _batch_idx
                x, y = x.to(self.config.device), y.to(self.config.device)
                self.optimizer.zero_grad()
                y_hat = self.model(x)
                loss = self.criterion(y_hat, y)
                loss.backward()
                self.optimizer.step()

                if self.scheduler:
                    self.scheduler.step()

                maybe_log_batch_loss()

                # Log to wandb & save checkpoints according to log_steps
                if (
                    self.checkpointer
                    and step in self.config.checkpointer_config.checkpoint_steps  # type: ignore
                ):
                    self.save_checkpoint(step)

                if self.logger and step in self.config.logger_config.logging_steps:  # type: ignore
                    log(step=step)

        if pbar:
            pbar.close()

        log(step=step)

        if self.config.is_wandb_enabled:
            wandb.finish()

    def state_dict(self) -> LearnerStateDict:
        """
        Returns the current state of the Learner in a LearnerStateDict format.

        Returns:
            LearnerStateDict: Dictionary containing model, optimizer, and scheduler states.
        """
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict()
            if self.scheduler is not None
            else None,
        }

    def load_state_dict(self, checkpoint: LearnerStateDict):
        """
        Loads a LearnerStateDict into the Learner.

        Args:
            checkpoint (LearnerStateDict): Dictionary containing model, optimizer, and scheduler states.
        """
        with temporarily_move_to_(checkpoint, torch.device("cpu")):
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])

            if self.scheduler is not None and checkpoint["scheduler"] is not None:
                self.scheduler.load_state_dict(checkpoint["scheduler"])

    def save_checkpoint(self, step: int):
        """
        Saves a checkpoint of the current Learner state.

        Args:
            epoch (int): Epoch to save checkpoint at.
            step (int): Batch index to save checkpoint at.
        """
        if self.checkpointer is None:
            raise ValueError("Cannot save checkpoint without a checkpointer.")

        checkpoint = self.state_dict()

        with temporarily_move_to_(checkpoint, torch.device("cpu")):
            self.checkpointer.save_file(step, checkpoint)

    def load_checkpoint(self, step: int):
        """
        Loads a checkpoint of the Learner state.

        Args:
            epoch (int): Epoch to load checkpoint from.
            step (int): Batch index to load checkpoint from.
        """
        if self.checkpointer is None:
            raise ValueError("Cannot load checkpoint without a checkpointer.")

        checkpoint = self.checkpointer.load_file(step)
        self.load_state_dict(checkpoint)


class LearnerConfig(BaseModel):
    # Dataset & loader
    num_training_samples: int
    batch_size: int = 128
    run_name: Optional[str] = None

    # Training loop
    # num_epochs: int = None
    num_steps: int = 100_000
    logger_config: Optional[MetricLoggingConfig] = None
    checkpointer_config: Optional[CheckpointerConfig] = None

    # Optimizer
    optimizer_config: OptimizerConfig = Field(default_factory=OptimizerConfig)
    scheduler_config: Optional[SchedulerConfig] = None

    # Misc
    device: str = Field(default_factory=get_default_device)
    criterion: CriterionLiteral = "cross_entropy"

    # TODO: Make this frozen
    class Config:
        frozen = True

    # Properties

    @property
    def num_steps_per_epoch(self):
        """Number of steps per epoch."""
        return self.num_training_samples // self.batch_size

    @property
    def num_epochs(self):
        """Number of epochs."""
        return math.ceil(self.num_steps / self.num_steps_per_epoch)

    @property
    def is_wandb_enabled(self):
        """Whether wandb is enabled."""
        return self.logger_config and self.logger_config.is_wandb_enabled

    # Validators

    @model_validator(mode="before")
    @classmethod
    def validate_config(cls, data: Any):
        num_steps = data["num_steps"]

        checkpoint_config = data.get("checkpointer_config", None)
        logger_config = data.get("logger_config", None)

        # Automatically expand `checkpoint_steps` for checkpointer and `logging_steps` for logger
        # "log_space": 10 -> "log_space": [1, num_steps, 10]
        checkpoint_steps = checkpoint_config.get("checkpoint_steps", None)
        if isinstance(checkpoint_steps, dict):
            expand_steps_config_(checkpoint_steps, num_steps)

        # Logger
        logger_steps = logger_config.get("logging_steps", None)
        if isinstance(logger_steps, dict):
            expand_steps_config_(logger_steps, num_steps)

        return data

