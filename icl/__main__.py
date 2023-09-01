"""
training the transformer on synthetic in-context regression task
"""
# manage environment
from dotenv import load_dotenv

from icl.evals import ICLEvaluator
from icl.utils import set_seed

load_dotenv()
# in case using mps:
import os

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = "1" # (! before import torch)

import logging
from typing import Dict, List, TypedDict

import numpy as np
import sentry_sdk
import torch.nn.functional as F
import tqdm

import wandb
#
from icl.config import ICLConfig, get_config
from icl.model import InContextRegressionTransformer

#


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
    model = config.task_config.model_factory().to(config.device)
    model.train()

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
        seed=config.task_config.true_seed
    )

    # initialise monitoring code
    checkpointer = config.checkpointer_config.factory() if config.checkpointer_config is not None else None
    logger = config.logger_config.factory() if config.logger_config is not None else None

    # initialise torch optimiser
    optimizer = config.optimizer_config.factory(model.parameters())
    scheduler = config.scheduler_config.factory(optimizer)  # type: ignore

    # training loop
    for step in tqdm.trange(config.num_steps, desc=f"Training..."):
        set_seed(config.task_config.sampling_seed + step)  # For reproducibility if we resume training

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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    config = get_config(project="icl", entity="devinterp")
    # config = get_config()
    train(config, seed=0, is_debug=False)

